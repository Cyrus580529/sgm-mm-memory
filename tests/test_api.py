"""Contract tests for Add/Search (no external model calls: captions and dense retrieval disabled)."""
import base64
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
