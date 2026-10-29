"""Contract tests for Add/Search (no external model calls: captions and dense retrieval disabled)."""
import base64
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
    assert "Herbert" in data[0]["content"]
    assert {d["id"] for d in data} == {d["id"] for d in data[:3]} and len(data) == 3
