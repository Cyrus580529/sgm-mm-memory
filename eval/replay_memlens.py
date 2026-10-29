"""Replay MemLens (one user per question, its own multi-session haystack) through the service in-process and
measure evidence-session recall at 10 / 30 / 100 results, per question type.

Each user turn plus the assistant turns after it is one Add, stamped with its session date; turn images are sent
inline. --with-date prefixes the question with "Today is <question_date>." (whether the platform passes the question
date is unknown).

usage: python -m eval.replay_memlens DATA_DIR [--split 32k] [--limit N] [--with-date] [--out results.json]
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
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.environ["MEMORY_API_KEY"] = KEY
    os.environ["MEMORY_DATA_DIR"] = tempfile.mkdtemp(prefix="memlens_eval_")
    from fastapi.testclient import TestClient
    import app.main as svc
    logging.getLogger("httpx").setLevel(logging.WARNING)

    items = json.load(open(os.path.join(args.data_dir, f"dataset_{args.split}.json")))[: args.limit or None]
    img_root = os.path.join(args.data_dir, "img")
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    missing_img = 0
    t0 = time.time()
    with TestClient(svc.app) as c:
        for n, it in enumerate(items):
            user = f"ml:{args.split}:{it['question_id']}"
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
                        if parts:
                            msgs.append({"role": t["role"], "content": parts, "timestamp": base_ts + g * 60_000 + j})
                    if not msgs:
                        continue
                    r = c.post("/v1/memories/add", headers=H, json={
                        "request_id": f"{it['question_id']}:{sid}:{g}", "user_id": user, "session_id": sid,
                        "messages": msgs})
                    assert r.status_code == 200, r.text
            gold = set(lst(it["answer_session_ids"]))
            if not gold or it["question_type"] == "answer_refusal":
                continue
            q = it["question"]
            if args.with_date:
                q = f"Today is {it['question_date']}. {q}"
            res = c.post("/v1/memories/search", headers=H, json={"query": q, "user_id": user, "top_k": 100})
            assert res.status_code == 200, res.text
            idx = svc.state["indexes"][user]
            sess_of = {str(u.records[0].id): u.session_id for u in idx.units}
            got = [sess_of[x["id"]] for x in res.json()["data"]]
            for k in (10, 30, 100):
                v = len(gold & set(got[:k])) / len(gold)
                stats["ALL"][k].append(v)
                stats[it["question_type"]][k].append(v)
            stats["_units"]["n"].append(len(idx.units))
            if (n + 1) % 100 == 0:
                print(f"{n + 1} questions ({time.time() - t0:.0f}s)", file=sys.stderr)

    units = stats.pop("_units")["n"]
    summary = {g: {f"R@{k}": round(sum(v) / len(v), 3) for k, v in ks.items()} | {"n": len(ks[10])}
               for g, ks in sorted(stats.items())}
    print(f"split={args.split} with_date={args.with_date} exchanges/user median="
          f"{sorted(units)[len(units) // 2] if units else 0} missing images={missing_img}")
    for g, s in summary.items():
        print(f"{g:24s} n={s['n']:4d}  R@10={s['R@10']:.3f}  R@30={s['R@30']:.3f}  R@100={s['R@100']:.3f}")
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
