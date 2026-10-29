"""Replay ATM-Bench (one user: emails + photos) through the service in-process and measure evidence recall.

Emails are text messages; photos are image messages carrying the original JPEG bytes (EXIF included).
Videos are not replayed (the contract accepts only JPEG/PNG/WebP); questions whose evidence is only video
are reported separately.

usage: python -m eval.replay_atm ATM_DIR [--limit-q N] [--out results.json]
ATM_DIR holds atm-bench.json, emails.json, image_batch_results.json, video_batch_results.json and raw/*.jpg
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
USER = "atm:user"


def ms(stamp: str) -> int:
    d = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("atm_dir")
    ap.add_argument("--limit-q", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.environ["MEMORY_API_KEY"] = KEY
    os.environ["MEMORY_DATA_DIR"] = tempfile.mkdtemp(prefix="atm_eval_")
    from fastapi.testclient import TestClient
    import app.main as svc
    logging.getLogger("httpx").setLevel(logging.WARNING)

    j = lambda f: json.load(open(os.path.join(args.atm_dir, f)))
    emails, images = j("emails.json"), j("image_batch_results.json")
    videos = {os.path.splitext(os.path.basename(v["video_path"]))[0] for v in j("video_batch_results.json")}
    qas = j("atm-bench.json")[: args.limit_q or None]

    t0 = time.time()
    with TestClient(svc.app) as c:
        store = svc.state["store"]
        for e in emails:
            r = c.post("/v1/memories/add", headers=H, json={
                "request_id": e["id"], "user_id": USER, "session_id": "email:" + e["timestamp"][:10],
                "messages": [{"role": "user", "content": e["detail"], "timestamp": ms(e["timestamp"])}]})
            assert r.status_code == 200, r.text
        missing = 0
        for im in images:
            stem = os.path.splitext(os.path.basename(im["image_path"]))[0]
            path = os.path.join(args.atm_dir, "raw", stem + ".jpg")
            if not os.path.exists(path):
                missing += 1
                continue
            uri = "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()
            r = c.post("/v1/memories/add", headers=H, json={
                "request_id": stem, "user_id": USER, "session_id": "photo:" + stem[:8],
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}],
                              "timestamp": ms(im["timestamp"])}]})
            assert r.status_code == 200, r.text
        print(f"added {len(emails)} emails, {len(images) - missing} photos ({missing} missing) "
              f"in {time.time() - t0:.0f}s", file=sys.stderr)
        c.post("/v1/memories/search", headers=H, json={"query": "warm up", "user_id": USER, "top_k": 1})
        id2src = {str(u.records[0].id): {r.request_id for r in u.records} for u in svc.state["indexes"][USER].units}

        stats = collections.defaultdict(lambda: collections.defaultdict(list))
        t1 = time.time()
        for qa in qas:
            ev = set(ast.literal_eval(qa["evidence_ids"]) if isinstance(qa["evidence_ids"], str) else qa["evidence_ids"])
            if not ev:
                continue
            kinds = {"email" if x.startswith("email") else ("video" if x in videos else "photo") for x in ev}
            group = "+".join(sorted(kinds))
            res = c.post("/v1/memories/search", headers=H, json={"query": qa["question"], "user_id": USER, "top_k": 100})
            got = [id2src[it["id"]] for it in res.json()["data"]]
            for k in (10, 30, 100):
                v = len(ev & set().union(*got[:k])) / len(ev)
                for g in ("ALL", group, qa["qtype"]):
                    stats[g][k].append(v)
        print(f"{len(qas)} searches in {time.time() - t1:.0f}s", file=sys.stderr)

    summary = {g: {f"R@{k}": round(sum(v) / len(v), 3) for k, v in ks.items()} | {"n": len(ks[10])}
               for g, ks in sorted(stats.items())}
    for g, s in summary.items():
        print(f"{g:18s} n={s['n']:4d}  R@10={s['R@10']:.3f}  R@30={s['R@30']:.3f}  R@100={s['R@100']:.3f}")
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
