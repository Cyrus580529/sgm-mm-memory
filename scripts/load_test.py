"""Concurrent Add/Search load test against a running service, with contract checks on every response.

Replays Mem-Gallery dialogs (one user per file, one Add per round) with --add-conc concurrent Adds
(rounds of one user stay in order, users run in parallel), then --search-conc concurrent Searches.

usage: python scripts/load_test.py BASE_URL MEMGALLERY_DATA_DIR [--files 4] [--add-conc 8] [--search-conc 16]
the service key is read from MEMORY_API_KEY
"""
import argparse
import asyncio
import base64
import glob
import json
import os
import statistics
import time
import uuid

import httpx


def uri(path):
    ext = os.path.splitext(path)[1].lower()
    mime = {".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def check_search(resp_json, top_k):
    data = resp_json["data"]
    assert isinstance(data, list) and len(data) <= top_k
    for it in data:
        assert isinstance(it["id"], str) and it["id"]
        c = it["content"]
        assert (isinstance(c, str) and c) or (isinstance(c, list) and c)
        if isinstance(c, list):
            for p in c:
                assert p["type"] in ("text", "image_url")
                if p["type"] == "image_url":
                    assert p["image_url"]["url"].startswith("data:image/")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("data_dir")
    ap.add_argument("--files", type=int, default=4)
    ap.add_argument("--add-conc", type=int, default=8)
    ap.add_argument("--search-conc", type=int, default=16)
    args = ap.parse_args()
    H = {"Authorization": f"Token {os.environ['MEMORY_API_KEY']}"}
    run = uuid.uuid4().hex[:8]
    files = sorted(glob.glob(os.path.join(args.data_dir, "dialog", "*.json")))[: args.files]
    lat = {"add": [], "search": []}
    errors, sizes = [], []
    add_sem, search_sem = asyncio.Semaphore(args.add_conc), asyncio.Semaphore(args.search_conc)

    async with httpx.AsyncClient(base_url=args.base, timeout=1800) as c:
        assert (await c.get("/health")).status_code == 200

        async def post(kind, path, body, sem):
            async with sem:
                t = time.time()
                r = await c.post(path, headers=H, json=body)
                lat[kind].append(time.time() - t)
            if r.status_code != 200:
                errors.append((kind, r.status_code, r.text[:200]))
                return None
            return r

        async def replay(fp):
            stem = os.path.splitext(os.path.basename(fp))[0]
            d = json.load(open(fp))
            user = f"load:{run}:{stem}"
            for sess in d["multi_session_dialogues"]:
                for k, rd in enumerate(sess["dialogues"]):
                    parts = [{"type": "text", "text": rd["user"]}]
                    for iid, ip in zip(rd.get("image_id") or [], rd.get("input_image") or []):
                        p = os.path.join(args.data_dir, "dialog", ip)
                        if os.path.exists(p) and os.path.getsize(p) < 9 * 2 ** 20:
                            parts += [{"type": "text", "text": f"image_id: {iid}"},
                                      {"type": "image_url", "image_url": {"url": uri(p)}}]
                    body = {"request_id": f"{run}:{stem}:{rd['round']}", "user_id": user,
                            "session_id": f"{stem}:{sess['session_id']}",
                            "messages": [{"role": "user", "content": parts, "timestamp": 1719200000000 + k},
                                         {"role": "assistant", "content": rd["assistant"] or "-"}]}
                    r = await post("add", "/v1/memories/add", body, add_sem)
                    if r is not None:
                        j = r.json()
                        assert j["success"] is True and j["request_id"] == body["request_id"]
            return user, [qa["question"] for qa in d["human-annotated QAs"]]

        t0 = time.time()
        users = await asyncio.gather(*(replay(f) for f in files))
        t_add = time.time() - t0

        async def ask(user, q):
            r = await post("search", "/v1/memories/search", {"query": q, "user_id": user, "top_k": 100}, search_sem)
            if r is not None:
                check_search(r.json(), 100)
                sizes.append(len(r.content))

        t1 = time.time()
        await asyncio.gather(*(ask(u, q) for u, qs in users for q in qs))
        t_search = time.time() - t1

    def pct(v, p):
        return sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")

    for kind, v in lat.items():
        print(f"{kind:6s} n={len(v):5d} p50={statistics.median(v):.2f}s p95={pct(v, .95):.2f}s max={max(v):.2f}s")
    print(f"add phase {t_add:.0f}s, search phase {t_search:.0f}s; response size p50={statistics.median(sizes)/2**20:.1f} MiB "
          f"max={max(sizes)/2**20:.1f} MiB; errors={len(errors)}")
    for e in errors[:5]:
        print("  ", e)


if __name__ == "__main__":
    asyncio.run(main())
