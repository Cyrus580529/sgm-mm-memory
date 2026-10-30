"""Replay MemLens (one user per question, its own multi-session haystack) through the service in-process and
measure evidence-session recall at 10 / 30 / 100 results, per question type.

Each user turn plus the assistant turns after it is one Add, stamped with its session date; turn images are sent
inline. --with-date prefixes the question with "Today is <question_date>." (whether the platform passes the question
date is unknown).

--answer N also answers up to N questions per type end to end: the Search result goes to the answer model (ANSWER_*
env, default gpt-4o-mini) with the question date stated, as in MemLens' memory-agent prompts, and is graded with the
official MemLens judge prompts (JUDGE_* env, default gpt-4o-mini; the benchmark itself uses a Qwen3-235B judge).

usage: python -m eval.replay_memlens DATA_DIR [--split 32k] [--limit N] [--with-date] [--answer N] [--out results.json]
DATA_DIR holds dataset_<split>.json and img/ (the extracted release_images.tar.gz)
"""
import argparse
import ast
import base64
import collections
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

KEY = "eval-key"
H = {"Authorization": f"Token {KEY}"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


MEMLENS_ANSWER_SYSTEM = ("You are answering a question using memories retrieved from a long chat history between a "
                         "user and an AI assistant, including images the user shared. Use ONLY the memories provided; "
                         "if they are insufficient, say so briefly. Respond concisely with the final answer.")


def stratified(items: list, n_per_type: int, seed: int = 0) -> set:
    """Question ids to answer: n per question type, spread over its subtypes in proportion (at least one each),
    drawn with a fixed seed so every arm of an ablation answers the same questions."""
    import random
    rng = random.Random(seed)
    by_type = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in items:
        by_type[it["question_type"]][it.get("question_subtype") or ""].append(it["question_id"])
    chosen = set()
    for subs in by_type.values():
        total = sum(len(v) for v in subs.values())
        quota = {k: max(1, round(n_per_type * len(v) / total)) for k, v in subs.items()}
        while sum(quota.values()) > n_per_type and max(quota.values()) > 1:
            quota[max(quota, key=quota.get)] -= 1
        for k, v in sorted(subs.items()):
            chosen.update(rng.sample(sorted(v), min(quota[k], len(v))))
    return chosen


def lst(v):
    return v if isinstance(v, list) else ast.literal_eval(v) if isinstance(v, str) and v.strip() else []


def ms(stamp: str) -> int:
    d = datetime.strptime(stamp.split(" (")[0] + " " + stamp.rsplit(" ", 1)[-1], "%Y/%m/%d %H:%M")
    return int(d.replace(tzinfo=timezone.utc).timestamp() * 1000)


def find_image(img_root: str, rel: str):
    for base in (img_root, os.path.join(img_root, "release_images")):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--split", default="32k")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--with-date", action="store_true")
    ap.add_argument("--no-captions", action="store_true", help="omit the dataset captions the platform sends with images")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="", help="jsonl with, per question, the rank of each gold session")
    ap.add_argument("--types", default="", help="comma-separated question types to run (default all)")
    ap.add_argument("--answer", type=int, default=0, help="answer and judge up to N questions per type")
    ap.add_argument("--only-answered", action="store_true",
                    help="with --answer, write and evaluate only the questions that will be answered (saves cost)")
    args = ap.parse_args()
    os.environ["MEMORY_API_KEY"] = KEY
    os.environ["MEMORY_DATA_DIR"] = os.environ.get("EVAL_STATE_DIR") or tempfile.mkdtemp(prefix="memlens_eval_")
    from fastapi.testclient import TestClient
    import app.main as svc
    from eval.client import add_all
    logging.getLogger("httpx").setLevel(logging.WARNING)

    items = json.load(open(os.path.join(args.data_dir, f"dataset_{args.split}.json")))[: args.limit or None]
    chosen = stratified(items, args.answer) if args.answer else None
    img_root = os.path.join(args.data_dir, "img")
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    missing_img = 0
    t0 = time.time()
    with TestClient(svc.app) as c:
        dump = open(args.dump, "w") if args.dump else None
        answered, pending, scores = collections.Counter(), [], collections.defaultdict(list)
        ans_log = open(args.out.replace(".json", "") + ".answers.jsonl" if args.out else os.devnull, "w")
        judge = None
        if args.answer:
            from eval.memlens_judge import Judge
            judge = Judge(os.path.join(args.data_dir, "official"))

        def flush():
            import asyncio
            from eval.answering import chat_many, items_to_parts
            reqs = []
            for it, data in pending:
                head = (f"Today's date for this question: {it['question_date']}\n\n"
                        "Memories retrieved from the long chat history (text and images):")
                tail = f"\n\nQuestion: {it['question']}\n\nAnswer:"
                reqs.append(([{"role": "system", "content": MEMLENS_ANSWER_SYSTEM},
                              {"role": "user", "content": [{"type": "text", "text": head}] + items_to_parts(data)
                               + [{"type": "text", "text": tail}]}], 300))
            answers = asyncio.run(chat_many(reqs))
            jreqs = []
            for (it, _), a in zip(pending, answers):
                key = judge.task_key(it["question_type"], it.get("question_subtype", ""), it["answer"])
                jreqs.append(([{"role": "user", "content": judge.prompt(key, it["question"], it["answer"], a)}], 400))
            verdicts = asyncio.run(chat_many(jreqs, "JUDGE"))
            for (it, _), a, v in zip(pending, answers, verdicts):
                sc = judge.parse(v)
                scores["ALL"].append(sc)
                scores[it["question_type"]].append(sc)
                ans_log.write(json.dumps({"qid": it["question_id"], "type": it["question_type"], "gold": it["answer"],
                                          "pred": a, "score": sc}, ensure_ascii=False) + "\n")
            pending.clear()
        for n, it in enumerate(items):
            if args.types and it["question_type"] not in args.types.split(","):
                continue
            if args.only_answered and chosen is not None and it["question_id"] not in chosen:
                continue
            user = f"ml:{args.split}:{it['question_id']}"
            adds = []
            sids, dates = lst(it["haystack_session_ids"]), lst(it["haystack_dates"])
            for sid, date, sess in zip(sids, dates, it["haystack_sessions"]):
                base_ts = ms(date)
                turns = sess if isinstance(sess, list) else json.loads(sess)
                groups, cur = [], []
                for t in turns:  # a user turn opens a new Add; assistant turns join it
                    if t["role"] == "user" and cur:
                        groups.append(cur)
                        cur = []
                    cur.append(t)
                if cur:
                    groups.append(cur)
                for g, grp in enumerate(groups):
                    msgs = []
                    for j, t in enumerate(grp):
                        parts = [{"type": "text", "text": t["content"]}] if t.get("content") else []
                        for im in lst(t.get("images")):
                            p = find_image(img_root, im["file"])
                            if p is None:
                                missing_img += 1
                                continue
                            raw = open(p, "rb").read()
                            parts.append({"type": "image_url", "image_url": {
                                "url": f"data:{MIME.get(os.path.splitext(p)[1].lower(), 'image/jpeg')};base64,"
                                       + base64.b64encode(raw).decode()}})
                            if not args.no_captions and im.get("blip_caption"):  # the platform sends captions too
                                parts.append({"type": "text", "text": f"caption: {im['blip_caption']}"})
                        if parts:
                            msgs.append({"role": t["role"], "content": parts, "timestamp": base_ts + g * 60_000 + j})
                    if not msgs:
                        continue
                    adds.append({"request_id": f"{it['question_id']}:{sid}:{g}", "user_id": user, "session_id": sid,
                                 "messages": msgs})
            add_all(c, H, adds)
            gold = set(lst(it["answer_session_ids"]))
            q = it["question"]
            if args.with_date:
                q = f"Today is {it['question_date']}. {q}"
            res = c.post("/v1/memories/search", headers=H, json={"query": q, "user_id": user, "top_k": 100})
            assert res.status_code == 200, res.text
            if chosen is not None and it["question_id"] in chosen:
                answered[it["question_type"]] += 1
                pending.append((it, res.json()["data"]))
                if len(pending) >= 20:
                    flush()
            if not gold or it["question_type"] == "answer_refusal":
                continue
            idx = svc.state["indexes"][user]
            sess_of = {str(u.records[0].id): u.session_id for u in idx.units}
            got = [sess_of[x["id"]] for x in res.json()["data"]]
            for k in (10, 30, 100):
                v = len(gold & set(got[:k])) / len(gold)
                stats["ALL"][k].append(v)
                stats[it["question_type"]][k].append(v)
            stats["_units"]["n"].append(len(idx.units))
            from app.retrieval import item_tokens
            stats["_returned"]["n"].append(len(got))
            stats["_returned"]["tok"].append(sum(item_tokens(x, svc.settings.image_detail_low) for x in res.json()["data"]))
            if dump:
                first = {}
                for r, sid in enumerate(got):
                    first.setdefault(sid, r)
                order = sorted(zip(sids, dates), key=lambda x: ms(x[1]))
                dump.write(json.dumps({"qid": it["question_id"], "type": it["question_type"], "q": it["question"],
                                       "gold": [{"sid": g, "rank": first.get(g), "date": dict(zip(sids, dates))[g],
                                                 "chrono": [x[0] for x in order].index(g)} for g in gold],
                                       "n_sessions": len(sids)}) + "\n")
            if (n + 1) % 100 == 0:
                print(f"{n + 1} questions ({time.time() - t0:.0f}s)", file=sys.stderr)

        if pending:
            flush()
    units = stats.pop("_units")["n"]
    ret = stats.pop("_returned", {"n": [0], "tok": [0]})
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"returned items median={med(ret['n'])} min={min(ret['n'])}; estimated tokens median={med(ret['tok'])} "
          f"max={max(ret['tok'])}")
    summary = {g: {f"R@{k}": round(sum(v) / len(v), 3) for k, v in ks.items()} | {"n": len(ks[10])}
               for g, ks in sorted(stats.items())}
    print(f"split={args.split} with_date={args.with_date} exchanges/user median="
          f"{sorted(units)[len(units) // 2] if units else 0} missing images={missing_img}")
    for g, s in summary.items():
        print(f"{g:24s} n={s['n']:4d}  R@10={s['R@10']:.3f}  R@30={s['R@30']:.3f}  R@100={s['R@100']:.3f}")
    for g, v in sorted(scores.items()):
        summary.setdefault("answer:" + g, {})["acc"] = round(sum(v) / len(v), 3)
        print(f"answer {g:24s} n={len(v):4d}  judge accuracy={sum(v) / len(v):.3f}")
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
