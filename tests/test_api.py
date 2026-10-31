"""Contract tests for Add/Search (no external model calls: captions and dense retrieval disabled)."""
import base64
import json
import hashlib
import importlib
import io
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

KEY = "test-key"


def png(r, g, b, size=8) -> str:
    """A tiny solid-colour PNG as a data URI."""
    raw = b"".join(b"\x00" + bytes([r, g, b]) * size for _ in range(size))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_API_KEY", KEY)
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("EMB_API_KEY", raising=False)
    import app.config, app.main
    importlib.reload(app.config)
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c


H = {"Authorization": f"Token {KEY}"}


def add(c, rid, user, session, messages):
    return c.post("/v1/memories/add", headers=H,
                  json={"request_id": rid, "user_id": user, "session_id": session, "messages": messages})


def search(c, user, query, top_k=100, **kw):
    return c.post("/v1/memories/search", headers=H, json={"query": query, "user_id": user, "top_k": top_k, **kw})


def test_health_and_auth(client):
    assert client.get("/health").status_code == 200
    assert client.post("/v1/memories/search", json={"query": "x", "user_id": "u", "top_k": 5}).status_code == 401
    for h in ({"Authorization": f"Bearer {KEY}"}, {"X-Api-Key": KEY}, H):
        assert client.post("/v1/memories/search", headers=h,
                           json={"query": "x", "user_id": "u", "top_k": 5}).json() == {"data": []}


def test_add_echo_and_idempotency(client):
    msgs = [{"role": "user", "content": "I baked a lemon tart today", "timestamp": 1719200000000}]
    r = add(client, "r1", "u1", "s1", msgs)
    assert r.status_code == 200
    assert r.json() == {"success": True, "request_id": "r1", "user_id": "u1", "session_id": "s1"}
    assert add(client, "r1", "u1", "s1", msgs).status_code == 200          # same payload: idempotent
    assert add(client, "r1", "u1", "s1", [{"role": "user", "content": "other"}]).status_code == 409
    assert len(search(client, "u1", "lemon tart").json()["data"]) == 1      # not stored twice


def test_user_isolation_and_ranking(client):
    add(client, "a", "u1", "s1", [{"role": "user", "content": "My dog Biscuit loves the beach"},
                                  {"role": "assistant", "content": "Beaches are great for dogs."}])
    add(client, "b", "u1", "s2", [{"role": "user", "content": "I started learning the violin"}])
    add(client, "c", "u2", "s1", [{"role": "user", "content": "Biscuit is my cat"}])
    data = search(client, "u1", "What is the name of my dog?").json()["data"]
    assert "Biscuit" in data[0]["content"] and "dog" in data[0]["content"]
    assert all("cat" not in d["content"] for d in data)
    assert len(search(client, "u1", "dog", top_k=1).json()["data"]) == 1


def test_image_roundtrip_and_exact_match(client):
    red, blue = png(255, 0, 0), png(0, 0, 255)
    add(client, "m1", "u1", "s1", [{"role": "user", "content": [
        {"type": "text", "text": "Here is my new mug"}, {"type": "image_url", "image_url": {"url": red}}]}])
    add(client, "m2", "u1", "s2", [{"role": "user", "content": [
        {"type": "text", "text": "And this is a postcard"}, {"type": "image_url", "image_url": {"url": blue}}]}])
    data = search(client, "u1", [{"type": "text", "text": "Which one is this?"},
                                 {"type": "image_url", "image_url": {"url": blue}}],
                  options=["A. mug", "B. postcard"]).json()["data"]
    top = data[0]["content"]
    assert isinstance(top, list)
    assert any(p["type"] == "image_url" and p["image_url"]["url"] == blue for p in top)


def test_bad_inputs(client):
    assert add(client, "x", "u", "s", [{"role": "user", "content": ""}]).status_code == 400
    bad_img = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://e.com/a.png"}}]}]
    assert add(client, "y", "u", "s", bad_img).status_code == 400
    assert client.post("/v1/memories/add", headers=H, json={"request_id": "z"}).status_code == 422


def test_neighbour_context(client):
    add(client, "n", "u1", "s1", [{"role": "user", "content": "Can you recommend a book?"},
                                  {"role": "assistant", "content": "Try Dune by Frank Herbert."},
                                  {"role": "user", "content": "Thanks, I will order it."}])
    data = search(client, "u1", "Frank Herbert").json()["data"]
    # one exchange = the question and its reply; the follow-up turn comes next as neighbouring context
    assert "recommend a book" in data[0]["content"] and "Herbert" in data[0]["content"]
    assert len(data) == 2 and "order it" in data[1]["content"]


def jpeg_with_exif() -> str:
    from PIL import Image
    im = Image.new("RGB", (32, 32), (10, 200, 10))
    ex = Image.Exif()
    ex[0x0132] = "2022:12:24 18:03:04"
    ex[0x8825] = {1: "N", 2: (51.0, 16.0, 48.0), 3: "E", 4: (1.0, 4.0, 48.0)}  # Canterbury
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=ex)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def test_photo_exif_time_and_place(client):
    add(client, "p1", "u1", "s1", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": jpeg_with_exif()}}],
                                    "timestamp": 1730000000000}])
    add(client, "p2", "u1", "s2", [{"role": "user", "content": "Lunch with Ana in London", "timestamp": 1730000000000}])
    data = search(client, "u1", "Today is January 3, 2023. Where was I on Christmas Eve last month, in Kent?").json()["data"]
    top = data[0]["content"]
    text = " ".join(p.get("text", "") for p in top) if isinstance(top, list) else top
    assert "Canterbury" in text and "2022-12-24" in text


@pytest.fixture()
def model_client(tmp_path, monkeypatch):
    """All model features switched on, with fake model calls (no network)."""
    import numpy as np
    monkeypatch.setenv("MEMORY_API_KEY", KEY)
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_API_KEY", "fake")
    monkeypatch.setenv("EMB_API_KEY", "fake")
    monkeypatch.setenv("EMB_DIM", "16")
    monkeypatch.setenv("EMB_SPAN_CHARS", "50")
    monkeypatch.setenv("EXPAND_N", "3")  # off by default; switched on here so its code path stays tested
    import app.config, app.main, app.models
    importlib.reload(app.config)
    importlib.reload(app.models)
    importlib.reload(app.main)
    calls = {"describe": 0, "embed": 0, "rerank": []}

    def vec(t):
        v = np.zeros(16, dtype=np.float32)
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 16] += 1
        return v / max(np.linalg.norm(v), 1e-8)

    async def describe(self, uri, ctx):
        calls["describe"] += 1
        return "Objects: a green ceramic teapot\nText: none"

    async def embed(self, texts, text_type="document"):
        calls["embed"] += len(texts)
        return np.stack([vec(t) for t in texts])

    async def rerank(self, query, docs, model):
        calls["rerank"].append((model, len(docs)))
        # text reranker: keeps order; image reranker: judges every image irrelevant
        return [(i, 0.9 if model == "qwen3-rerank" else 0.1) for i in range(len(docs))]

    monkeypatch.setattr(app.models.Models, "describe_image", describe)
    monkeypatch.setattr(app.models.Models, "embed", embed)
    async def expand(self, q, n):
        calls.setdefault("expand", []).append(q)
        return ["green teapot photo", "ceramic teapot"]  # deliberately off-topic for other questions

    monkeypatch.setattr(app.models.Models, "rerank", rerank)
    monkeypatch.setattr(app.models.Models, "expand_query", expand)
    with TestClient(app.main.app) as c:
        yield c, calls


def test_model_paths(model_client):
    c, calls = model_client
    img = png(0, 200, 0)
    long_text = "We talked about trains. " * 10 + "The secret word is pumpernickel."
    add(c, "a", "u1", "s1", [{"role": "user", "content": [{"type": "text", "text": "my new teapot"},
                                                          {"type": "image_url", "image_url": {"url": img}}]}])
    add(c, "b", "u1", "s2", [{"role": "user", "content": long_text}])
    add(c, "c", "u1", "s3", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}]}])
    assert calls["describe"] == 1                       # same image described once
    data = search(c, "u1", "what colour is the teapot?").json()["data"]
    assert len(data) == 3
    assert any("green ceramic teapot" in (p.get("text", "") if isinstance(p, dict) else "")
               for p in data[0]["content"]) if isinstance(data[0]["content"], list) else True
    models = [m for m, _ in calls["rerank"]]
    assert "qwen3-rerank" in models and "qwen3-vl-rerank" in models
    assert dict(calls["rerank"])["qwen3-vl-rerank"] == 2  # two exchanges carry an image
    assert calls["expand"] == ["what colour is the teapot?"]
    assert "pumpernickel" in search(c, "u1", "secret word pumpernickel").json()["data"][0]["content"]


def test_provider_outage(model_client, monkeypatch):
    import app.models
    c, calls = model_client
    ok_describe = app.models.Models.describe_image
    ok_embed = app.models.Models.embed

    async def down(self, *a, **k):
        raise app.models.ModelUnavailable("provider down")

    monkeypatch.setattr(app.models.Models, "describe_image", down)
    msgs = [{"role": "user", "content": [{"type": "text", "text": "the blue kettle"},
                                         {"type": "image_url", "image_url": {"url": png(0, 0, 200)}}]}]
    r = add(c, "k1", "u9", "s1", msgs)
    assert r.status_code == 503 and r.headers["retry-after"]
    assert search(c, "u9", "kettle").json() == {"data": []}            # nothing half-written
    monkeypatch.setattr(app.models.Models, "describe_image", ok_describe)
    assert add(c, "k1", "u9", "s1", msgs).status_code == 200            # platform retry succeeds

    monkeypatch.setattr(app.models.Models, "embed", down)               # query embedding fails at Search
    data = search(c, "u9", "blue kettle").json()["data"]
    assert len(data) == 1                                               # lexical retrieval still answers
    monkeypatch.setattr(app.models.Models, "embed", ok_embed)


def test_whole_memory_returned_when_it_fits(client):
    for i, t in enumerate(["I adopted a cat", "My sister lives in Lyon", "I bought a red bike"]):
        add(client, f"w{i}", "u5", f"s{i}", [{"role": "user", "content": t, "timestamp": 1700000000000 + i}])
    data = search(client, "u5", "when did I adopt the cat?").json()["data"]
    assert len(data) == 3 and "cat" in data[0]["content"]          # ranked hit first, the rest still included
    assert len(search(client, "u5", "pet", top_k=2).json()["data"]) == 2


def test_reply_without_timestamp_stays_with_its_question(client):
    add(client, "t1", "u6", "s1", [{"role": "user", "content": "My bike is in garage 7", "timestamp": 1717000000000},
                                   {"role": "assistant", "content": "Noted: garage 7."}])
    add(client, "t2", "u6", "s1", [{"role": "user", "content": "I also own a kayak"}])
    data = search(client, "u6", "where is my bike?").json()["data"]
    assert len(data) == 2 and "garage 7" in data[0]["content"] and "Noted" in data[0]["content"]


def test_long_outage_degrades_then_repairs(model_client, monkeypatch):
    import asyncio
    import app.main
    import app.models
    c, calls = model_client
    object.__setattr__(app.main.settings, "add_max_defers", 2)
    ok_describe = app.models.Models.describe_image

    async def down(self, *a, **k):
        raise app.models.ModelUnavailable("provider down")

    monkeypatch.setattr(app.models.Models, "describe_image", down)
    msgs = [{"role": "user", "content": [{"type": "text", "text": "the yellow umbrella"},
                                         {"type": "image_url", "image_url": {"url": png(250, 220, 0)}}]}]
    codes = [add(c, "o1", "u7", "s1", msgs).status_code for _ in range(3)]
    assert codes == [503, 503, 200]                                   # two deferrals, then a degraded commit
    first = search(c, "u7", "umbrella").json()["data"]
    assert len(first) == 1 and "green ceramic teapot" not in json.dumps(first)
    stats = c.get("/stats", headers=H).json()
    assert stats["store"]["repair_queue"] == 1 and stats["counters"]["add_degraded"] == 1

    monkeypatch.setattr(app.models.Models, "describe_image", ok_describe)  # provider back
    assert c.portal.call(app.main.repair_once) == 1
    after = search(c, "u7", "umbrella").json()["data"]
    assert "green ceramic teapot" in json.dumps(after)                  # description filled in, index rebuilt
    assert c.get("/stats", headers=H).json()["store"]["repair_queue"] == 0
    object.__setattr__(app.main.settings, "add_max_defers", 4)


def test_reranker_internal_error_does_not_fail_search(model_client, monkeypatch):
    import app.models
    c, _ = model_client

    async def broken(self, *a, **k):
        raise ValueError("unexpected provider payload")

    monkeypatch.setattr(app.models.Models, "rerank", broken)
    monkeypatch.setattr(app.models.Models, "expand_query", broken)
    add(c, "rb", "u8", "s1", [{"role": "user", "content": "the violin lesson is on Tuesday"}])
    r = search(c, "u8", "when is the violin lesson?")
    assert r.status_code == 200 and "Tuesday" in r.json()["data"][0]["content"]


def test_odd_timestamps_are_ignored(client):
    for i, ts in enumerate([-5, 10 ** 17, 1717000000000.7, "abc"]):
        r = add(client, f"ts{i}", "u10", "s1", [{"role": "user", "content": f"note {i}", "timestamp": ts}])
        assert r.status_code == (422 if ts == "abc" else 200), (ts, r.text)
    assert len(search(client, "u10", "note").json()["data"]) == 3


def test_stats_and_health(client):
    assert client.get("/stats").status_code == 401
    s = client.get("/stats", headers=H).json()
    assert {"counters", "store", "disk_free_gb", "models"} <= set(s)
    assert client.get("/health").json() == {"status": "ok"}


def test_response_size_guard():
    from app.retrieval import fit_response
    big = "data:image/png;base64," + "A" * 4000
    items = [{"id": str(i), "content": [{"type": "text", "text": f"item {i}"},
                                        {"type": "image_url", "image_url": {"url": big}}]} for i in range(5)]
    out = fit_response(items, limit=12000)
    assert len(json.dumps({"data": out})) <= 12000 and len(out) == 5
    assert isinstance(out[-1]["content"], str) and isinstance(out[0]["content"], list)   # lowest-ranked lose images
    texts = [{"id": str(i), "content": "x" * 1000} for i in range(10)]
    assert len(fit_response(texts, limit=5000)) < 10


def test_search_skips_unusable_query_parts(client):
    add(client, "q1", "u11", "s1", [{"role": "user", "content": "my passport is in the top drawer"}])
    bad = [{"type": "text", "text": "where is my passport?"},
           {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
           {"type": "image_url", "image_url": {"url": "data:image/gif;base64,R0lGODlhAQABAAAAACw="}},
           {"type": "audio", "data": "x"}]
    r = search(client, "u11", bad)
    assert r.status_code == 200 and "drawer" in r.json()["data"][0]["content"]
    r = search(client, "u11", [{"type": "image_url", "image_url": {"url": "nope"}}])
    assert r.status_code == 200 and len(r.json()["data"]) == 1            # nothing usable: most recent memories


def test_answer_token_budget():
    from types import SimpleNamespace
    from app.config import Settings
    from app.retrieval import Unit, est_tokens, image_tokens, render
    rec = lambda i, text, imgs=(): SimpleNamespace(id=i, ts=1717000000000 + i, event_ts=None, role="user",
                                                   session_id="s", request_id=f"r{i}", msg_index=0, text=text,
                                                   images=list(imgs), parts=[["text", text]] + [["image", h] for h in imgs])
    store = SimpleNamespace(image_description=lambda h: "a red mug", image_meta=lambda h: None,
                            image_data_uri=lambda h: ("data:image/png;base64,AAAA", 3),
                            image_for_answer=lambda h, side: ("data:image/png;base64,AAAA", 3, (512, 512)))
    units = [Unit([rec(1, "short note")], "s", 0), Unit([rec(2, "word " * 4000)], "s", 1),
             Unit([rec(3, "photo of the mug", ["h1"])], "s", 2), Unit([rec(4, "another short note")], "s", 3)]
    s = Settings()
    assert image_tokens(512, 512) == 8500 and image_tokens(1024, 768) == 25501
    out = render(units, [0, 1, 2, 3], store, s, budget=5000)
    assert [it["id"] for it in out] == ["1", "3", "4"]     # oversized item skipped, later ones still fit
    assert isinstance(out[1]["content"], str) and "red mug" in out[1]["content"]   # 8.5k image gave way to text
    out = render(units, [2, 0], store, s, budget=20000)
    assert isinstance(out[0]["content"], list)               # with room, the best-ranked image keeps its pixels
    assert [it["score"] for it in out] == sorted([it["score"] for it in out], reverse=True)
    assert est_tokens("你好世界") >= 4 and est_tokens("hello world") >= 3


def test_only_top_images_carry_pixels():
    from types import SimpleNamespace
    from app.config import Settings
    from app.retrieval import Unit, render
    rec = lambda i: SimpleNamespace(id=i, ts=1717000000000 + i, event_ts=None, role="user", session_id="s",
                                    request_id=f"r{i}", msg_index=0, text=f"photo {i}", images=[f"h{i}"],
                                    parts=[["text", f"photo {i}"], ["image", f"h{i}"]])
    store = SimpleNamespace(image_description=lambda h: f"desc {h}", image_meta=lambda h: None,
                            image_data_uri=lambda h: ("data:image/png;base64,AAAA", 3),
                            image_for_answer=lambda h, side: ("data:image/png;base64,AAAA", 3, (512, 512)))
    s = Settings()
    object.__setattr__(s, "max_images", 2)
    out = render([Unit([rec(i)], "s", i) for i in range(5)], list(range(5)), store, s, budget=10 ** 6)
    with_pixels = [isinstance(it["content"], list) for it in out]
    assert with_pixels == [True, True, False, False, False]
    assert all("desc h" in (it["content"] if isinstance(it["content"], str) else str(it["content"])) for it in out)

def test_out_of_order_adds_follow_request_ordinal(client):
    # chunk-2 arrives before chunk-1, no timestamps: the request ordinal restores the order
    add(client, "run:sample:0:chunk-2", "u12", "s", [{"role": "assistant", "content": "It is under the sofa."}])
    add(client, "run:sample:0:chunk-1", "u12", "s", [{"role": "user", "content": "Where did the cat hide the toy mouse?"}])
    data = search(client, "u12", "toy mouse").json()["data"]
    assert len(data) == 1 and data[0]["content"].index("toy mouse") < data[0]["content"].index("sofa")


def test_image_question_queries_image_reranker_with_the_image(model_client, monkeypatch):
    import app.models
    c, _ = model_client
    seen = []

    async def rerank(self, query, docs, model):
        seen.append((model, "image" if isinstance(query, dict) else "text"))
        return [(i, 0.9) for i in range(len(docs))]

    monkeypatch.setattr(app.models.Models, "rerank", rerank)
    add(c, "p1", "u13", "s1", [{"role": "user", "content": [{"type": "text", "text": "foxglove in the garden"},
                                                           {"type": "image_url", "image_url": {"url": png(150, 0, 150)}}]}])
    search(c, "u13", [{"type": "text", "text": "What plant is shown in the picture?"},
                      {"type": "image_url", "image_url": {"url": png(140, 10, 140)}}])
    assert ("qwen3-vl-rerank", "image") in seen and ("qwen3-rerank", "text") in seen


def test_corrupt_or_mislabelled_images(client):
    garbage = "data:image/jpeg;base64," + base64.b64encode(b"definitely not an image" * 20).decode()
    mislabelled = png(10, 10, 10).replace("data:image/png", "data:image/jpeg")   # PNG bytes declared as JPEG
    r = add(client, "ci", "u14", "s1", [{"role": "user", "content": [
        {"type": "text", "text": "two pictures of the harbour"},
        {"type": "image_url", "image_url": {"url": garbage}},
        {"type": "image_url", "image_url": {"url": mislabelled}}]}])
    assert r.status_code == 200
    content = search(client, "u14", "harbour").json()["data"][0]["content"]
    urls = [p["image_url"]["url"] for p in content if p["type"] == "image_url"]
    assert len(urls) == 1 and urls[0].startswith("data:image/png")          # real format, garbage not returned
    assert any("unreadable image" in p.get("text", "") for p in content)


def test_arrangement_changes_order_not_membership():
    from types import SimpleNamespace
    from app.retrieval import arrange
    U = lambda sid, ts, pos: SimpleNamespace(session_id=sid, ts=ts, pos=pos)
    chosen = [(U("b", 300, 1), {"id": "1", "content": "b-late"}), (U("a", 100, 0), {"id": "2", "content": "a-early"}),
              (U("b", 200, 0), {"id": "3", "content": "b-early"})]
    assert [i["id"] for i in arrange(chosen, "relevance")] == ["1", "2", "3"]
    assert [i["id"] for i in arrange(chosen, "chrono")] == ["2", "3", "1"]
    assert [i["id"] for i in arrange(chosen + [(U("c", 50, 0), {"id": "4", "content": "c"})], "sandwich")] == \
        ["1", "3", "4", "2"]                                         # ranks 1,3 at the head, 4,2 at the tail
    sess = arrange(chosen, "session")
    assert [i["id"] for i in sess] == ["3", "2"]                  # session b first (it holds the best hit)
    assert sess[0]["content"].index("b-early") < sess[0]["content"].index("b-late")


def test_assistant_trim_beyond_top_ranks(monkeypatch, client):
    import app.main
    object.__setattr__(app.main.settings, "assistant_trim_after", 1)
    long_reply = "Here is a very long and generic piece of advice. " * 40
    for i in range(3):
        add(client, f"tr{i}", "u15", f"s{i}", [{"role": "user", "content": f"my kayak story {i}", "timestamp": 1700000000000 + i},
                                              {"role": "assistant", "content": long_reply}])
    data = search(client, "u15", "kayak").json()["data"]
    assert len(data[0]["content"]) > 1500 and all(len(d["content"]) < 500 for d in data[1:])
    object.__setattr__(app.main.settings, "assistant_trim_after", 20)


def test_request_ordinal_only_for_structured_ids():
    from app.retrieval import request_ordinal
    assert request_ordinal("run:sample:0:chunk-2") == 2 and request_ordinal("run/3/7") == 7
    assert request_ordinal("abc-12") == 12 and request_ordinal("plain") == 0
    # the platform's opaque ids: trailing digits of a hex digest are not an order
    assert request_ordinal("r_1b6b641bd57a376786ec90c26108fbc2cd64a74541b618f15e3086ba0c5b5767") == 0
    assert request_ordinal("r_24289f015b8848bb90e724d1259989f39e8d4a0b47c6cfc7940a34e4a9c3939b") == 0


def test_opaque_request_ids_keep_arrival_order(client):
    # same timestamps, hash-like ids whose trailing digits would invert the order: arrival decides
    first, second = "r_" + "a" * 60 + "9999", "r_" + "b" * 60 + "0001"
    add(client, first, "u16", "s", [{"role": "user", "content": "Where did the cat hide the toy mouse?", "timestamp": 1747774800000}])
    add(client, second, "u16", "s", [{"role": "assistant", "content": "It is under the sofa.", "timestamp": 1747774800000}])
    data = search(client, "u16", "toy mouse").json()["data"]
    assert len(data) == 1 and data[0]["content"].index("toy mouse") < data[0]["content"].index("sofa")


def test_description_context_echo_is_not_reused():
    from app.store import strip_context
    d = "Scene: a photo\nObjects: a red mug\nText: none\nContext: user: my mug from Lyon"
    assert strip_context(d) == "Scene: a photo\nObjects: a red mug\nText: none"
    md = "**Scene:** photo\n**Text:** none\n**Context:** 王景川: 这是我的杯子\n第二行"
    assert strip_context(md) == "**Scene:** photo\n**Text:** none"
    assert strip_context("Scene: x   Objects: y") == "Scene: x   Objects: y" and strip_context("") == "" and strip_context(None) is None


def test_description_cached_across_users_carries_no_other_context(model_client, monkeypatch):
    import app.models
    c, _ = model_client

    async def describe(self, uri, ctx):
        return f"Objects: a green ceramic teapot\nContext: {ctx}"

    monkeypatch.setattr(app.models.Models, "describe_image", describe)
    img = png(0, 120, 60)
    add(c, "x1", "ua", "s1", [{"role": "user", "content": [{"type": "text", "text": "secret plan of user A"},
                                                         {"type": "image_url", "image_url": {"url": img}}]}])
    add(c, "x2", "ub", "s1", [{"role": "user", "content": [{"type": "text", "text": "my teapot"},
                                                         {"type": "image_url", "image_url": {"url": img}}]}])
    data = search(c, "ub", "teapot").json()["data"]
    assert data and "secret plan" not in json.dumps(data, ensure_ascii=False)
    assert "green ceramic teapot" in json.dumps(data)


def test_provider_account_errors_are_outages_not_refusals(monkeypatch):
    import asyncio
    import httpx
    from app.config import Settings
    from app.models import ModelRejected, ModelUnavailable, Models

    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    def run(status):
        m = Models(Settings())
        m.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(status, json={})))
        try:
            asyncio.run(m._post("https://llm.example/v1/chat/completions", "k", {}, attempts=2))
        except Exception as e:
            return type(e)

    assert run(400) is ModelRejected and run(422) is ModelRejected        # the input itself
    assert run(401) is ModelUnavailable and run(403) is ModelUnavailable  # key, quota or gateway: retry, defer, repair
    assert run(429) is ModelUnavailable and run(503) is ModelUnavailable


def test_question_image_counts_against_answer_budget(client):
    import app.main
    store = app.main.state["store"]
    h = store.put_image(base64.b64decode(png(1, 2, 3, size=600).split(",", 1)[1]), "image/png")
    assert app.main.query_image_tokens(store, [h]) == 2833 + 5667 * 4
    assert app.main.query_image_tokens(store, []) == 0
