"""Replay MobileMem-Omni (one user per trajectory: ~200 sessions of chat with screenshots and photos) through the
service in-process and measure evidence-session recall at 10 / 30 / 100, per question type and language.

Within a session, consecutive user turns (text and images) plus the assistant replies after them form one Add,
stamped from the session's event start time. Images over 9 MiB are re-encoded, as the platform must do.

--answer N answers up to N questions per type with the answer model (ANSWER_*, default gpt-4o-mini) and grades
them with MobileMem's LLM judge (JUDGE_*): CORRECT only with full coverage of the gold answer and no extra content.

usage: python -m eval.replay_mobilemem DATA_DIR [--users N] [--questions filtered|all] [--answer N] [--out results.json]
DATA_DIR holds data.jsonl, filtered_questions.jsonl / questions.jsonl and img/ (the extracted image.zip)
"""
import argparse
import ast
import base64
import collections
import io
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


def P(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def ms(stamp: str) -> int:
    return int(datetime.strptime(stamp[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)


def data_uri(path: str) -> str:
    raw = open(path, "rb").read()
    mime = MIME.get(os.path.splitext(path)[1].lower(), "image/png")
    if len(raw) > 9 * 1024 * 1024:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((4096, 4096))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        raw, mime = buf.getvalue(), "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def grade(pending: list, scores: dict, log, judge_dir: str) -> None:
    import asyncio
    from eval.answering import chat_many, items_to_parts
    from eval.official_judges import MobileMemJudge
    judge = MobileMemJudge(judge_dir)
    reqs = [([{"role": "user", "content": [{"type": "text", "text": f"Question:\n{qa['question']}\n"
                                            "Please answer the question based on the following memories:"}]
               + items_to_parts(data)
               + [{"type": "text", "text": "\nAnswer Rules:\n1. If options are provided in the question, use the "
                                           "original option text as your answer.\n2. For multiple-choice questions "
                                           "(multiple-select), select all correct options.\n3. For open-ended "
                                           "questions, respond freely."}]}], 300) for qa, data in pending]
    answers = asyncio.run(chat_many(reqs))
    def explain(e):  # explanations are strings in most records, dicts in a few
        x = e.get("explanation", "") if isinstance(e, dict) else e
        return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)

    ev = ["; ".join(explain(e) for e in qa.get("evidence") or []) for qa, _ in pending]
    verdicts = asyncio.run(chat_many([([{"role": "user", "content": judge.prompt(qa["question"], str(qa["answer"]),
                                                                                  e, a)}], 300)
                                      for (qa, _), e, a in zip(pending, ev, answers)], "JUDGE"))
    for (qa, _), a, v in zip(pending, answers, verdicts):
        sc = judge.parse(v)
        scores["ALL"].append(sc)
        scores[qa["question_type"]].append(sc)
        log.write(json.dumps({"qid": qa.get("question_id"), "type": qa["question_type"], "gold": qa["answer"],
                              "pred": a, "score": sc}, ensure_ascii=False) + "\n")
    pending.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--users", type=int, default=0)
    ap.add_argument("--questions", default="filtered", choices=["filtered", "all"])
    ap.add_argument("--out", default="")
    ap.add_argument("--answer", type=int, default=0, help="answer and judge up to N questions per type")
    args = ap.parse_args()
    os.environ["MEMORY_API_KEY"] = KEY
    os.environ["MEMORY_DATA_DIR"] = tempfile.mkdtemp(prefix="mobilemem_eval_")
    from fastapi.testclient import TestClient
    import app.main as svc
    logging.getLogger("httpx").setLevel(logging.WARNING)

    qfile = "filtered_questions.jsonl" if args.questions == "filtered" else "questions.jsonl"
    questions = {r["uuid"]: P(r["questions"]) for r in map(json.loads, open(os.path.join(args.data_dir, qfile)))}
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "data.jsonl"))]
    rows = [r for r in rows if r["uuid"] in questions][: args.users or None]
    img_root = os.path.join(args.data_dir, "img")
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    answered, pending, scores = collections.Counter(), [], collections.defaultdict(list)
    ans_log = open(args.out.replace(".json", "") + ".answers.jsonl" if args.out else os.devnull, "w")
    missing = 0
    t0 = time.time()
    with TestClient(svc.app) as c:
        for row in rows:
            user = f"mm:{row['uuid']}"
            for sess in P(row["sessions"]):
                base = ms(sess["event_start_time"])
                groups, cur = [], []
                for t in P(sess["dialogue"]):  # user turns open an exchange; assistant turns close it
                    if t["role"] == "user" and cur and cur[-1]["role"] != "user":
                        groups.append(cur)
                        cur = []
                    cur.append(t)
                if cur:
                    groups.append(cur)
                for g, grp in enumerate(groups):
                    msgs = []
                    for t in grp:
                        k = int(t.get("turn", 0))
                        if t.get("content_type") == "image":
                            rel = t["image_inline"].split("image/", 1)[-1]
                            # the dialogues write some folders with spaces ("chat records"), the archive with "_"
                            p = next((x for x in (os.path.join(img_root, rel),
                                                  os.path.join(img_root, rel.replace(" ", "_"))) if os.path.exists(x)),
                                     None)
                            if p is None:
                                missing += 1
                                continue
                            content = [{"type": "image_url", "image_url": {"url": data_uri(p)}}]
                        elif t.get("content"):
                            content = t["content"]
                        else:
                            continue
                        msgs.append({"role": t["role"], "content": content, "timestamp": base + k * 30_000})
                    if not msgs:
                        continue
                    r = c.post("/v1/memories/add", headers=H, json={
                        "request_id": f"{sess['session_id']}:{g}", "user_id": user,
                        "session_id": sess["session_id"], "messages": msgs})
                    assert r.status_code == 200, r.text
            print(f"user {row['uuid']} written ({time.time() - t0:.0f}s)", file=sys.stderr)

            for qa in questions[row["uuid"]]:
                gold = set(qa.get("source_session_ids") or []) | {e["session_id"] for e in qa.get("evidence") or []
                                                                   if e.get("session_id")}
                res = c.post("/v1/memories/search", headers=H,
                             json={"query": qa["question"], "user_id": user, "top_k": 100})
                assert res.status_code == 200, res.text
                if args.answer and answered[qa["question_type"]] < args.answer:
                    answered[qa["question_type"]] += 1
                    pending.append((qa, res.json()["data"]))
                    if len(pending) >= 20:
                        grade(pending, scores, ans_log, os.path.join(args.data_dir, "official"))
                if not gold or qa["question_type"] == "abstention":
                    continue
                idx = svc.state["indexes"][user]
                sess_of = {str(u.records[0].id): u.session_id for u in idx.units}
                got = [sess_of[x["id"]] for x in res.json()["data"]]
                for k in (10, 30, 100):
                    v = len(gold & set(got[:k])) / len(gold)
                    for g in ("ALL", qa["question_type"], "lang:" + row.get("language", "?")):
                        stats[g][k].append(v)

        if pending:
            grade(pending, scores, ans_log, os.path.join(args.data_dir, "official"))
    summary = {g: {f"R@{k}": round(sum(v) / len(v), 3) for k, v in ks.items()} | {"n": len(ks[10])}
               for g, ks in sorted(stats.items())}
    for g, v in sorted(scores.items()):
        summary.setdefault("answer:" + g, {})["acc"] = round(sum(v) / len(v), 3)
        print(f"answer {g:22s} n={len(v):4d}  judge accuracy={sum(v) / len(v):.3f}")
    print(f"users={len(rows)} questions={args.questions} missing images={missing} ({time.time() - t0:.0f}s)")
    for g, s in ((g, s) for g, s in summary.items() if not g.startswith("answer:")):
        print(f"{g:22s} n={s['n']:4d}  R@10={s['R@10']:.3f}  R@30={s['R@30']:.3f}  R@100={s['R@100']:.3f}")
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
