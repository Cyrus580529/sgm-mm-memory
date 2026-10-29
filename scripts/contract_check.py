"""Check a running service against the AML Add/Search contract (multimodal track) before the official Smoke test.

Each check follows a rule of the public API guide (agentmemoryleaderboard.ai/api-guide). Uses its own random user
ids, so it can run against a deployed instance.

usage: MEMORY_API_KEY=... python scripts/contract_check.py BASE_URL [--add-path /v1/memories/add]
       [--search-path /v1/memories/search]
"""
import argparse
import base64
import io
import json
import os
import sys
import uuid

import httpx

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))


def image(fmt="PNG", size=(64, 64), color=(200, 30, 30), noise_mb=0.0):
    from PIL import Image
    if noise_mb:  # incompressible image of roughly the given size
        side = int((noise_mb * 1024 * 1024 / 3) ** 0.5)
        im = Image.frombytes("RGB", (side, side), os.urandom(side * side * 3))
    else:
        im = Image.new("RGB", size, color)
    buf = io.BytesIO()
    im.save(buf, fmt)
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[fmt]
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--add-path", default="/v1/memories/add")
    ap.add_argument("--search-path", default="/v1/memories/search")
    args = ap.parse_args()
    key = os.environ["MEMORY_API_KEY"]
    run = uuid.uuid4().hex[:10]
    u1, u2 = f"check:{run}:u1", f"check:{run}:u2"
    c = httpx.Client(base_url=args.base, timeout=600)
    tok = {"Authorization": f"Token {key}"}

    def add(body, headers=tok):
        return c.post(args.add_path, headers=headers, json=body)

    def search(body, headers=tok):
        return c.post(args.search_path, headers=headers, json=body)

    def item_text(it):
        ct = it["content"]
        return ct if isinstance(ct, str) else " ".join(p.get("text", "") for p in ct if p.get("type") == "text")

    # health: unauthenticated GET, any 2xx
    r = c.get("/health")
    check("health is an unauthenticated 2xx GET", 200 <= r.status_code < 300, r.status_code)

    # authentication
    probe = {"query": "x", "user_id": u1, "top_k": 5}
    check("missing credentials rejected", search(probe, {}).status_code in (401, 403))
    check("wrong key rejected", search(probe, {"Authorization": "Token wrong"}).status_code in (401, 403))
    for h in ({"Authorization": f"Token {key}"}, {"Authorization": f"Bearer {key}"}, {"X-Api-Key": key}):
        check(f"auth scheme {list(h)[0]}: {list(h.values())[0].split()[0] if ' ' in list(h.values())[0] else 'key'}",
              search(probe, h).status_code == 200)

    # empty user
    r = search({"query": "anything", "user_id": f"check:{run}:nobody", "top_k": 10})
    check("unknown user returns 200 {data: []}", r.status_code == 200 and r.json() == {"data": []}, r.text[:100])

    # Add: echo, persistence, idempotency, conflict
    body = {"request_id": f"{run}:r1", "user_id": u1, "session_id": f"{run}:s1",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "My bicycle is the blue Brompton parked in garage 7."},
                {"type": "image_url", "image_url": {"url": image("PNG", color=(20, 40, 220))}}],
                "timestamp": 1717000000000},
                {"role": "assistant", "content": "Noted, the blue Brompton in garage 7."}]}
    r = add(body)
    ok = r.status_code == 200 and r.json().get("success") is True and all(
        r.json().get(k) == body[k] for k in ("request_id", "user_id", "session_id"))
    check("Add returns 200 and echoes request_id/user_id/session_id exactly", ok, r.text[:200])
    r = search({"query": "Where is my bicycle parked?", "user_id": u1, "top_k": 10})
    check("written memory is searchable right after Add returns", r.status_code == 200 and any(
        "garage 7" in item_text(it) for it in r.json()["data"]), r.text[:200])
    check("repeating the same Add is idempotent (200)", add(body).status_code == 200)
    n_before = len(search({"query": "bicycle garage", "user_id": u1, "top_k": 100}).json()["data"])
    check("repeated Add does not duplicate memories", n_before == 1, n_before)
    changed = json.loads(json.dumps(body))
    changed["messages"][0]["content"][0]["text"] = "something else"
    check("same request_id with different content -> 409", add(changed).status_code == 409)

    # content forms
    r = add({"request_id": f"{run}:r2", "user_id": u1, "session_id": f"{run}:s2",
             "messages": [{"role": "user", "content": "Plain string content: my cat is called Pixel."}]})
    check("string content accepted", r.status_code == 200, r.text[:200])
    for fmt in ("JPEG", "WEBP"):
        r = add({"request_id": f"{run}:{fmt}", "user_id": u1, "session_id": f"{run}:s3",
                 "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image(fmt)}}]}]})
        check(f"{fmt} image accepted", r.status_code == 200, r.text[:200])
    for name, url in (("remote image URL", "https://example.com/a.png"), ("GIF image", "data:image/gif;base64,R0lGODlhAQABAAAAACw=")):
        r = add({"request_id": f"{run}:{name}", "user_id": u1, "session_id": f"{run}:s4",
                 "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]})
        check(f"{name} rejected with 4xx", 400 <= r.status_code < 500, r.status_code)
    big = image("PNG", noise_mb=10.5)
    r = add({"request_id": f"{run}:big", "user_id": u1, "session_id": f"{run}:s5",
             "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": big}}]}]})
    check("image over 10 MiB decoded rejected with 4xx", 400 <= r.status_code < 500, r.status_code)
    two = [{"type": "image_url", "image_url": {"url": image("PNG", noise_mb=9.0)}} for _ in range(2)]
    r = add({"request_id": f"{run}:near", "user_id": u2, "session_id": f"{run}:s6",
             "messages": [{"role": "user", "content": [{"type": "text", "text": "two large photos"}] + two}]})
    check("Add with ~18 MiB of images (under the 30 MiB limit) accepted", r.status_code == 200, r.text[:200])
    r = add({"request_id": f"{run}:empty", "user_id": u1, "session_id": f"{run}:s7",
             "messages": [{"role": "user", "content": ""}]})
    check("empty content rejected with 4xx", 400 <= r.status_code < 500, r.status_code)

    # Search response contract
    r = search({"query": [{"type": "text", "text": "which bicycle?"},
                          {"type": "image_url", "image_url": {"url": image("PNG", color=(20, 40, 220))}}],
                "user_id": u1, "top_k": 3, "options": ["A. blue Brompton", "B. red Trek"]})
    data = r.json().get("data", []) if r.status_code == 200 else []
    check("multimodal query with options returns 200", r.status_code == 200, r.text[:200])
    check("result count <= top_k", len(data) <= 3, len(data))
    check("every item has a non-empty string id and non-empty content",
          all(isinstance(it.get("id"), str) and it["id"] and it.get("content") for it in data))
    parts_ok = all(isinstance(it["content"], str) or all(
        p.get("type") in ("text", "image_url") and (p["type"] != "image_url" or p["image_url"]["url"].startswith("data:image/"))
        and (p["type"] != "text" or p.get("text")) for p in it["content"]) for it in data)
    check("content parts are text or inline data-URI images", parts_ok)
    scores = [it["score"] for it in data if "score" in it]
    check("scores (if present) are numeric and non-increasing",
          all(isinstance(s, (int, float)) for s in scores) and scores == sorted(scores, reverse=True))
    check("Search response under 30 MiB", len(r.content) < 30 * 1024 * 1024, len(r.content))
    check("top_k=1 honoured", len(search({"query": "cat", "user_id": u1, "top_k": 1}).json()["data"]) <= 1)
    check("top_k=0 rejected with 4xx", 400 <= search({"query": "cat", "user_id": u1, "top_k": 0}).status_code < 500)
    r = search({"query": "bicycle garage cat", "user_id": u2, "top_k": 100})
    check("user isolation: other user's memories never returned",
          r.status_code == 200 and not any("garage 7" in item_text(it) or "Pixel" in item_text(it) for it in r.json()["data"]))

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
