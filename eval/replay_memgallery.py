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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--files", type=int, default=0)
    ap.add_argument("--glob", default="*.json", help="dialog file pattern, e.g. '*[!n].json' to skip *_Open files")
    ap.add_argument("--out", default="")
    ap.add_argument("--answer", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    os.environ.setdefault("MEMORY_API_KEY", KEY)
    os.environ["MEMORY_API_KEY"] = KEY
    os.environ["MEMORY_DATA_DIR"] = tempfile.mkdtemp(prefix="mg_eval_")
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
                    for iid, ip in zip(as_list(rd.get("image_id")), as_list(rd.get("input_image"))):
                        parts.append({"type": "text", "text": f"image_id: {iid}"})
                        path = img_path(args.data_dir, ip)
                        if path:  # a few images are missing from the release
                            parts.append({"type": "image_url", "image_url": {"url": data_uri(path)}})
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
                qimg = qa.get("question_image")
                qimg = as_list(qimg)[0] if isinstance(qimg, str) and qimg.startswith("[") else qimg
                if qimg and img_path(args.data_dir, qimg):
                    q = [{"type": "text", "text": q},
                         {"type": "image_url", "image_url": {"url": data_uri(img_path(args.data_dir, qimg))}}]
                res = c.post("/v1/memories/search", headers=H, json={"query": q, "user_id": user, "top_k": args.top_k})
                assert res.status_code == 200, res.text
                rounds = [id2round[it["id"]] for it in res.json()["data"]]
                for k in (10, 30, 100):
                    got = len(clues & set(rounds[:k])) / len(clues)
                    stats[str(qa["point"])[:12]][k].append(got)
                    stats["ALL"][k].append(got)
                if args.answer and answered[str(qa["point"])] < args.answer:
                    answered[str(qa["point"])] += 1
                    pending.append((qa, q, res.json()["data"]))
            if pending:  # answer per file so image payloads do not pile up in memory
                from eval.answering import answer_all
                for pt, v in answer_all(pending, ans_path).items():
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
        f1 = f"  F1={s['F1']:.3f} (n={s['n_ans']})" if "F1" in s else ""
        print(f"{p:5s} n={s['n']:4d}  R@10={s['R@10']:.3f}  R@30={s['R@30']:.3f}  R@100={s['R@100']:.3f}{f1}")
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
