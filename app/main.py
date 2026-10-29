"""AML Add/Search service."""
import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import retrieval
from .config import check_competition_models, settings
from .models import Models
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("aml")

MIMES = {"image/jpeg", "image/png", "image/webp"}
DATA_URI = re.compile(r"^data:(image/[a-z]+);base64,(.*)$", re.S)
MAX_IMAGE = 10 * 1024 * 1024


class Message(BaseModel):
    role: str = Field(min_length=1)
    content: Union[str, list]
    timestamp: Optional[int] = None


class AddBody(BaseModel):
    request_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class SearchBody(BaseModel):
    query: Union[str, list]
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    options: Optional[list[Any]] = None


class BadInput(Exception):
    pass


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    check_competition_models(settings)
    if not settings.memory_api_key:
        raise RuntimeError("MEMORY_API_KEY is not set")
    state["store"] = Store(settings.data_dir)
    state["models"] = Models(settings)
    state["add_locks"] = {}
    log.info("started; captions=%s dense=%s", settings.captions_enabled, settings.dense_enabled)
    yield


app = FastAPI(title="SGM-MM-Memory", lifespan=lifespan)


def authorized(req: Request) -> bool:
    key = settings.memory_api_key
    auth = req.headers.get("authorization", "")
    cands = [req.headers.get("x-api-key", "")]
    for scheme in ("Token ", "Bearer "):
        if auth.startswith(scheme):
            cands.append(auth[len(scheme):].strip())
    return any(c and hmac.compare_digest(c, key) for c in cands)


def error(status: int, msg: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "error": msg})


def split_content(content) -> list:
    """Normalise message content into [("text", str) | ("image", (mime, bytes))]."""
    if isinstance(content, str):
        if not content.strip():
            raise BadInput("empty content")
        return [("text", content)]
    if not content:
        raise BadInput("empty content")
    out = []
    for p in content:
        if not isinstance(p, dict):
            raise BadInput("content part must be an object")
        if p.get("type") == "text":
            if isinstance(p.get("text"), str) and p["text"]:
                out.append(("text", p["text"]))
        elif p.get("type") == "image_url":
            iu = p.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            m = DATA_URI.match(url or "")
            if not m:
                raise BadInput("image_url must be a base64 data URI")
            mime = "image/jpeg" if m.group(1) == "image/jpg" else m.group(1)
            if mime not in MIMES:
                raise BadInput(f"unsupported image type {mime}")
            try:
                raw = base64.b64decode(m.group(2), validate=False)
            except (binascii.Error, ValueError):
                raise BadInput("invalid base64 image")
            if len(raw) > MAX_IMAGE:
                raise BadInput("image larger than 10 MiB")
            out.append(("image", (mime, raw)))
        else:
            raise BadInput(f"unknown content part type {p.get('type')!r}")
    if not out:
        raise BadInput("empty content")
    return out


async def describe_missing(store: Store, models: Models, jobs: list) -> None:
    """jobs: (hash, data_uri, context). Describes each new image once (cached by hash)."""
    if not settings.captions_enabled:
        return
    todo = {}
    for h, uri, ctx in jobs:
        if h not in todo and store.image_description(h) is None:
            todo[h] = (uri, ctx)

    async def one(h, uri, ctx):
        try:
            store.set_image_description(h, await models.describe_image(uri, ctx))
        except Exception as e:  # keep the Add alive; the image is still stored and returned
            log.error("image description failed for %s: %s", h[:12], e)

    await asyncio.gather(*(one(h, u, c) for h, (u, c) in todo.items()))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/memories/add")
async def add(req: Request):
    if not authorized(req):
        return error(401, "unauthorized")
    try:
        body = AddBody.model_validate(await req.json())
    except Exception as e:
        return error(422, f"invalid request: {e}")
    store: Store = state["store"]
    models: Models = state["models"]
    payload_hash = hashlib.sha256(json.dumps(
        [body.user_id, body.session_id, [m.model_dump() for m in body.messages]],
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    ok = {"success": True, "request_id": body.request_id, "user_id": body.user_id, "session_id": body.session_id}

    lock = state["add_locks"].setdefault(body.request_id, asyncio.Lock())
    async with lock:
        st = store.add_status(body.request_id, payload_hash)
        if st == "duplicate":
            return ok
        if st == "conflict":
            return error(409, "request_id already used with different content")
        try:
            parsed = [split_content(m.content) for m in body.messages]
        except BadInput as e:
            return error(400, str(e))

        rows, jobs = [], []
        for idx, (m, parts) in enumerate(zip(body.messages, parsed)):
            text = "\n".join(v for k, v in parts if k == "text")
            ref_parts, hashes = [], []
            for kind, val in parts:
                if kind == "text":
                    ref_parts.append(["text", val])
                else:
                    mime, raw = val
                    h = store.put_image(raw, mime)
                    hashes.append(h)
                    ref_parts.append(["image", h])
                    jobs.append((h, f"data:{mime};base64," + base64.b64encode(raw).decode(), text))
            rows.append({"msg_index": idx, "role": m.role, "ts": m.timestamp, "text": text,
                         "images": hashes, "parts": ref_parts})
        await describe_missing(store, models, jobs)
        for r in rows:
            descs = [store.image_description(h) for h in r["images"]]
            r["index_text"] = "\n".join([f"{r['role']}: {r['text']}"] + [f"[image] {d}" for d in descs if d])
        if settings.dense_enabled:
            vecs = await models.embed([r["index_text"] for r in rows])
            for r, v in zip(rows, vecs):
                r["vec"] = v
        else:
            for r in rows:
                r["vec"] = None
        store.commit_add(body.request_id, body.user_id, body.session_id, payload_hash, rows)
    state["add_locks"].pop(body.request_id, None)
    return ok


@app.post("/v1/memories/search")
async def search(req: Request):
    if not authorized(req):
        return error(401, "unauthorized")
    try:
        body = SearchBody.model_validate(await req.json())
        q_parts = split_content(body.query)
    except BadInput as e:
        return error(400, str(e))
    except Exception as e:
        return error(422, f"invalid request: {e}")
    store: Store = state["store"]
    models: Models = state["models"]

    records = store.user_records(body.user_id)
    if not records:
        return {"data": []}

    q_text = "\n".join(v for k, v in q_parts if k == "text")
    q_hashes, jobs = [], []
    for kind, val in q_parts:
        if kind == "image":
            mime, raw = val
            h = store.put_image(raw, mime)
            q_hashes.append(h)
            jobs.append((h, f"data:{mime};base64," + base64.b64encode(raw).decode(), q_text))
    await describe_missing(store, models, jobs)
    q_full = "\n".join([q_text] + [store.image_description(h) or "" for h in q_hashes]).strip()
    opts = " ".join(str(o) for o in (body.options or []))

    rankings = [(1.0, retrieval.bm25_rank(f"{q_full} {opts}", [r.index_text for r in records]))]
    if settings.dense_enabled and q_full:
        qv = (await models.embed([q_full]))[0]
        rankings.append((1.0, retrieval.dense_rank(qv, records)))
    if q_hashes:
        hs = set(q_hashes)
        rankings.append((2.0, [i for i, r in enumerate(records) if hs & set(r.images)]))
    fused = retrieval.rrf(rankings, settings.rrf_k)
    if not fused:  # nothing matched lexically and no dense index: fall back to most recent
        fused = list(range(len(records) - 1, -1, -1))
    order = retrieval.order_with_neighbours(fused, records, settings.seed_k, settings.neighbour_window, body.top_k)
    return {"data": retrieval.render(records, order, store, settings)}
