"""Replay Mem-Gallery through the Add/Search service in-process and measure evidence recall.

Each dialog file is one user; each round is one Add (user message with images, then the assistant reply).
For every QA with annotated clue rounds we report the share of clue rounds found in the top 10 / 30 / 100
search results, overall and per question type.

With --answer N, up to N questions per type are also answered end to end: the Search result (text and
images) goes to the answer model (ANSWER_* env, OpenAI-compatible; default the LLM_* settings) with the
Mem-Gallery task instructions, and scored by Mem-Gallery token F1.

usage: python -m eval.replay_memgallery DATA_DIR [--files N] [--answer N] [--out results.json]
"""
import argparse
import base64
import collections
import glob
import io
import logging
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

KEY = "eval-key"
H = {"Authorization": f"Token {KEY}"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def data_uri(path):
    """Images over the 10 MiB contract limit are re-encoded, as the platform must do."""
    with open(path, "rb") as f:
        raw = f.read()
    mime = MIME[os.path.splitext(path)[1].lower()]
    if len(raw) > 9 * 1024 * 1024:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((4096, 4096))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        raw, mime = buf.getvalue(), "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def img_path(data_dir, rel):
    """Mem-Gallery paths are relative to dialog/ ("../image/..."); MemEye paths to image/."""
    for p in (os.path.join(data_dir, "dialog", rel), os.path.join(data_dir, "image", rel)):
        if os.path.exists(p):
            return p
    return None


def as_list(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(str(v).replace("'", '"'))
    except Exception:
        return []


def point_label(qa: dict) -> str:
    """Question category: Mem-Gallery's point code, or MemEye's capability cell "[['X4'], ['Y1']]" -> "X4/Y1"."""
    p = qa.get("point")
    try:
        cells = as_list(p) if isinstance(p, str) and p.startswith("[") else None
        if cells:
            return "/".join(x[0] if isinstance(x, list) and x else str(x) for x in cells)
    except Exception:
        pass
    return str(p)


def grade_memeye(pending: list, ans_path: str, judge_dir: str) -> dict:
    """MemEye: MCQ by extracted letter, open questions by the official LLM judge."""
    import asyncio
    from eval.answering import chat_many, items_to_parts
    from eval.official_judges import MemEyeJudge
    judge = MemEyeJudge(judge_dir)
    reqs = []
    for qa, query, data, options, _ in pending:
        ask = qa["question"] + ("\nOptions:\n" + "\n".join(options) + "\nReply with only the option letter."
                                if options else "")
        content = [{"type": "text", "text": "Memories retrieved from earlier conversations (text and images):"}]
        content += items_to_parts(data)
        if isinstance(query, list):
            content += [{"type": "text", "text": "\nImage attached to the question:"}] + \
                       [p for p in query if p["type"] == "image_url"]
        content.append({"type": "text", "text": f"\nQuestion: {ask}\nAnswer:"})
        reqs.append(([{"role": "user", "content": content}], 200))
    answers = asyncio.run(chat_many(reqs))
    scores, jreqs, jidx = collections.defaultdict(list), [], []
    rows = []
    for i, ((qa, _, _, options, letter), a) in enumerate(zip(pending, answers)):
        if options:
            valid = {o.split(".", 1)[0] for o in options}
            sc = float(judge.extract_choice(a, valid) == letter)
            rows.append([qa, a, sc])
        else:
            jreqs.append(([{"role": "user", "content": judge.prompt(qa["question"], str(qa["answer"]), a)}], 400))
            jidx.append(len(rows))
            rows.append([qa, a, None])
    for k, v in zip(jidx, asyncio.run(chat_many(jreqs, "JUDGE")) if jreqs else []):
        rows[k][2] = judge.parse(v)
    with open(ans_path, "a") as f:
        for qa, a, sc in rows:
            pt = point_label(qa)
            scores[pt].append(sc)
            scores["ALL"].append(sc)
            kind = "mcq" if isinstance(qa.get("options"), (list, str)) and qa.get("options") else "open"
            f.write(json.dumps({"q": qa["question"], "type": f"{kind}:{point_label(qa)}", "gold": qa["answer"],
                                "pred": a, "score": sc}, ensure_ascii=False) + "\n")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--files", type=int, default=0)
    ap.add_argument("--glob", default="*.json", help="dialog file pattern, e.g. '*[!n].json' to skip *_Open files")
    ap.add_argument("--out", default="")
    ap.add_argument("--answer", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--no-captions", action="store_true", help="omit the dataset captions the platform sends with images")
    ap.add_argument("--grader", default="f1", choices=["f1", "memeye"],
                    help="f1: Mem-Gallery token F1; memeye: MemEye LLM judge (open) / letter match (MCQ files)")
    args = ap.parse_args()

    os.environ.setdefault("MEMORY_API_KEY", KEY)
    os.environ["MEMORY_API_KEY"] = KEY
    os.environ["MEMORY_DATA_DIR"] = os.environ.get("EVAL_STATE_DIR") or tempfile.mkdtemp(prefix="mg_eval_")
    from fastapi.testclient import TestClient
    logging.getLogger("httpx").setLevel(logging.WARNING)
    import app.main as svc

    files = sorted(glob.glob(os.path.join(args.data_dir, "dialog", args.glob)))
    if args.files:
        files = files[: args.files]
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    answered, pending, f1s = collections.Counter(), [], collections.defaultdict(list)
    ans_path = args.out.replace(".json", "") + ".answers.jsonl" if args.out else "answers.jsonl"
    t0 = time.time()
    with TestClient(svc.app) as c:
        store = svc.state["store"]
        for fp in files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            d = json.load(open(fp))
            user = f"mg:{stem}"
            for sess in d["multi_session_dialogues"]:
                day = datetime.strptime(sess["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                for k, rd in enumerate(sess["dialogues"]):
                    ts = int(day.timestamp() * 1000) + k * 60_000
                    parts = [{"type": "text", "text": rd["user"]}]
                    caps = as_list(rd.get("image_caption"))
                    for j, (iid, ip) in enumerate(zip(as_list(rd.get("image_id")), as_list(rd.get("input_image")))):
                        parts.append({"type": "text", "text": f"image_id: {iid}"})
                        path = img_path(args.data_dir, ip)
                        if path:  # a few images are missing from the release
                            parts.append({"type": "image_url", "image_url": {"url": data_uri(path)}})
                        if not args.no_captions and j < len(caps) and caps[j]:  # the platform sends captions too
                            parts.append({"type": "text", "text": f"caption: {caps[j]}"})
                    msgs = [{"role": "user", "content": parts if len(parts) > 1 else rd["user"], "timestamp": ts},
                            {"role": "assistant", "content": rd["assistant"] or "(no reply)", "timestamp": ts + 1}]
                    r = c.post("/v1/memories/add", headers=H, json={
                        "request_id": f"{stem}:{rd['round']}", "user_id": user,
                        "session_id": f"{stem}:{sess['session_id']}", "messages": msgs})
                    assert r.status_code == 200, r.text
            id2round = {str(rec.id): rec.request_id.split(":", 1)[1] for rec in store.user_records(user)}  # unit id = first record

            for qa in d["human-annotated QAs"]:
                clues = set(as_list(qa.get("clue")))
                if not clues:
                    continue
                q = qa["question"]
                variants = as_list(qa.get("options")) if qa.get("options") else []
                opts = variants[0] if variants and isinstance(variants[0], dict) else None
                options = [f"{k}. {v}" for k, v in opts.items() if k != "answer"] if opts else None
                qimg = qa.get("question_image")
                qimg = as_list(qimg)[0] if isinstance(qimg, str) and qimg.startswith("[") else qimg
                if qimg and img_path(args.data_dir, qimg):
                    q = [{"type": "text", "text": q},
                         {"type": "image_url", "image_url": {"url": data_uri(img_path(args.data_dir, qimg))}}]
                body = {"query": q, "user_id": user, "top_k": args.top_k}
                if options:
                    body["options"] = options
                res = c.post("/v1/memories/search", headers=H, json=body)
                assert res.status_code == 200, res.text
                rounds = [id2round[it["id"]] for it in res.json()["data"]]
                for k in (10, 30, 100):
                    got = len(clues & set(rounds[:k])) / len(clues)
                    stats[point_label(qa)][k].append(got)
                    stats["ALL"][k].append(got)
                if args.answer and answered[str(qa["point"])] < args.answer:
                    answered[str(qa["point"])] += 1
                    pending.append((qa, q, res.json()["data"], options, opts["answer"] if opts else None))
            if pending:  # answer per file so image payloads do not pile up in memory
                if args.grader == "f1":
                    from eval.answering import answer_all
                    for pt, v in answer_all([p[:3] for p in pending], ans_path).items():
                        f1s[pt].extend(v)
                else:
                    for pt, v in grade_memeye(pending, ans_path, judge_dir=os.path.join(args.data_dir, "official")).items():
                        f1s[pt].extend(v)
                pending.clear()
            print(f"{stem}: done ({time.time() - t0:.0f}s)", file=sys.stderr)

    summary = {p: {f"R@{k}": round(sum(v) / len(v), 3) for k, v in ks.items()} | {"n": len(ks[10])}
               for p, ks in sorted(stats.items())}
    for p, v in f1s.items():
        summary.setdefault(p, {})["F1"] = round(sum(v) / len(v), 3)
        summary[p]["n_ans"] = len(v)
    if f1s:
        print(f"answers: {ans_path}", file=sys.stderr)
    for p, s in summary.items():
        f1 = f"  {'F1' if args.grader == 'f1' else 'score'}={s['F1']:.3f} (n={s['n_ans']})" if "F1" in s else ""
        print(f"{p:5s} n={s['n']:4d}  R@10={s['R@10']:.3f}  R@30={s['R@30']:.3f}  R@100={s['R@100']:.3f}{f1}")
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
