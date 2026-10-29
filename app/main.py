"""AML Add/Search service."""
import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import re

import numpy as np
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import retrieval
from .config import check_competition_models, settings
from .meta import meta_text
from .models import ModelRejected, ModelUnavailable, Models
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
    state["store"] = Store(settings.data_dir, settings.emb_dim)
    state["models"] = Models(settings)
    state["add_locks"] = {}
    state["indexes"] = OrderedDict()
    state["index_locks"] = {}
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


async def describe_missing(store: Store, models: Models, jobs: list, strict: bool) -> None:
    """jobs: (hash, context). Describes each new image once (cached by hash). A refused image is stored with an
    empty description. If the provider is unavailable, strict (Add) re-raises so the platform retries the Add;
    otherwise (Search) the image simply goes without a description."""
    if not settings.captions_enabled:
        return
    todo = {}
    for h, ctx in jobs:
        if h not in todo and await asyncio.to_thread(store.image_description, h) is None:
            todo[h] = ctx

    async def one(h, ctx):
        uri = (await asyncio.to_thread(store.image_data_uri, h))[0]
        try:
            desc = await models.describe_image(uri, ctx)
        except ModelRejected as e:
            log.error("image description refused for %s: %s", h[:12], e)
            desc = ""
        await asyncio.to_thread(store.set_image_description, h, desc)

    results = await asyncio.gather(*(one(h, c) for h, c in todo.items()), return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        log.error("%d image descriptions failed: %s", len(errors), errors[0])
        if strict:
            raise errors[0]


async def embed_cached(store: Store, models: Models, texts: list, text_type: str) -> list:
    """Embeddings, computed once per distinct text (the same history may be written for many samples)."""
    tag = f"{settings.emb_model}|{settings.emb_dim}|{settings.emb_style}|{text_type}|"
    keys = [hashlib.sha256((tag + t).encode()).hexdigest() for t in texts]
    have = store.cached_vecs(list(set(keys)))
    todo = {k: t for k, t in zip(keys, texts) if k not in have}
    if todo:
        new = await models.embed(list(todo.values()), text_type)
        fresh = dict(zip(todo.keys(), new))
        store.put_vecs(fresh)
        have.update(fresh)
    return [have[k] for k in keys]


async def rerank_lists(store: Store, models: Models, q_full: str, q_hashes: list, units: list, cands: list) -> list:
    """Extra (weight, unit ranking) votes from the rerankers: one over candidate texts, one over candidate images.
    A failing reranker only removes its vote."""
    if q_full:
        query = q_full[:2000]
    elif q_hashes:
        query = {"image": store.image_data_uri(q_hashes[0])[0]}
    else:
        return []
    img_units = [k for k in cands if units[k].images][:40]

    def build_docs():
        return ([{"text": retrieval.unit_text(units[k], store)[:3000] or " "} for k in cands],
                [{"image": store.image_data_uri(retrieval.first_image(units[k]))[0]} for k in img_units])

    docs, idocs = await asyncio.to_thread(build_docs)
    jobs = [(cands, 0.0, models.rerank(query, docs, settings.rerank_text_model))]
    if img_units:
        jobs.append((img_units, settings.rerank_vl_min, models.rerank(query, idocs, settings.rerank_vl_model)))
    out = []
    results = await asyncio.gather(*(j for _, _, j in jobs), return_exceptions=True)
    for (keys, min_score, _), res in zip(jobs, results):
        if isinstance(res, Exception):
            log.error("rerank failed: %s", res)
            continue
        out.append((settings.rerank_weight, [keys[i] for i, sc in res if 0 <= i < len(keys) and sc >= min_score]))
    return out


async def expansion_lists(store: Store, models: Models, cached, q_full: str) -> list:
    """Lexical and dense votes for gpt-4o-mini's rewrites of the question; any failure just drops them."""
    try:
        queries = await models.expand_query(q_full, settings.expand_n)
    except (ModelUnavailable, ModelRejected) as e:
        log.error("query expansion failed: %s", e)
        return []
    lists = [await asyncio.to_thread(cached.bm25.rank, q) for q in queries]
    if settings.dense_enabled and queries:
        try:
            vecs = await embed_cached(store, models, queries, "query")
            lists += [retrieval.to_units(retrieval.dense_rank(v, cached.records, cached.dense), cached.unit_of)
                      for v in vecs]
        except (ModelUnavailable, ModelRejected) as e:
            log.error("expansion embedding failed: %s", e)
    return [(settings.expand_weight / len(lists), l) for l in lists] if lists else []


@app.get("/health")
async def health():
    return {"status": "ok"}


class Retry(Exception):
    pass


def prepare_rows(store: Store, body: AddBody, parsed: list) -> tuple:
    """Store images (original + compact copy + EXIF) and build the rows of one Add. Runs in a worker thread."""
    rows, jobs = [], []
    for idx, (m, parts) in enumerate(zip(body.messages, parsed)):
        text = "\n".join(v for k, v in parts if k == "text")
        ref_parts, hashes = [], []
        for kind, val in parts:
            if kind == "text":
                ref_parts.append(["text", val])
            else:
                h = store.put_image(val[1], val[0])
                hashes.append(h)
                ref_parts.append(["image", h])
                jobs.append((h, text))
        taken = [t for t in ((store.image_meta(h) or {}).get("taken_ms") for h in hashes) if t]
        rows.append({"msg_index": idx, "role": m.role, "ts": m.timestamp, "text": text,
                     "event_ts": taken[0] if taken else m.timestamp, "images": hashes, "parts": ref_parts})
    return rows, jobs


def finish_rows(store: Store, rows: list) -> None:
    for r in rows:
        imgs = []
        for h in r["images"]:
            meta, desc = meta_text(store.image_meta(h)), store.image_description(h)
            imgs.append("[image] " + " ".join(x for x in (meta, desc) if x))
        dates = {retrieval.date_words(r["ts"]), retrieval.date_words(r["event_ts"])} - {""}
        r["embed_text"] = "\n".join([f"{r['role']}: {r['text']}"] + imgs)
        r["index_text"] = "\n".join(sorted(dates) + [r["embed_text"]])
        r["vec"] = None


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
    try:
        async with lock:
            st = await asyncio.to_thread(store.add_status, body.request_id, payload_hash)
            if st == "duplicate":
                return ok
            if st == "conflict":
                return error(409, "request_id already used with different content")
            try:
                parsed = [split_content(m.content) for m in body.messages]
            except BadInput as e:
                return error(400, str(e))
            rows, jobs = await asyncio.to_thread(prepare_rows, store, body, parsed)
            try:
                await describe_missing(store, models, jobs, strict=True)
                await asyncio.to_thread(finish_rows, store, rows)
                if settings.dense_enabled:
                    pieces = [retrieval.spans(r["embed_text"], settings.emb_span_chars) for r in rows]
                    vecs = await embed_cached(store, models, [p for ps in pieces for p in ps], "document")
                    k = 0
                    for r, ps in zip(rows, pieces):
                        r["vec"] = np.stack(vecs[k:k + len(ps)])
                        k += len(ps)
            except (ModelUnavailable, ModelRejected) as e:
                # nothing is committed yet; descriptions already made are cached, so a retry is cheap
                log.error("add %s deferred: %s", body.request_id, e)
                return JSONResponse(status_code=503, headers={"Retry-After": "30"},
                                    content={"success": False, "error": "model provider unavailable, retry later"})
            await asyncio.to_thread(store.commit_add, body.request_id, body.user_id, body.session_id,
                                    payload_hash, rows)
    finally:
        if not lock.locked():
            state["add_locks"].pop(body.request_id, None)
    return ok


async def user_index(store: Store, user_id: str):
    """The user's cached index, rebuilt (once, under a per-user lock) when new records arrived.
    The cache holds at most INDEX_CACHE_RECORDS records across users; least recently used users are evicted."""
    cache = state["indexes"]
    lock = state["index_locks"].setdefault(user_id, asyncio.Lock())
    async with lock:
        n = await asyncio.to_thread(store.user_count, user_id)
        if not n:
            return None
        idx = cache.get(user_id)
        if idx is None or len(idx.records) != n:
            idx = await asyncio.to_thread(lambda: retrieval.UserIndex(store.user_records(user_id)))
            cache[user_id] = idx
        cache.move_to_end(user_id)
        while len(cache) > 1 and sum(len(i.records) for i in cache.values()) > settings.index_cache_records:
            cache.popitem(last=False)
        return idx


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

    cached = await user_index(store, body.user_id)
    if cached is None:
        return {"data": []}
    records, units, unit_of = cached.records, cached.units, cached.unit_of

    q_text = "\n".join(v for k, v in q_parts if k == "text")
    q_hashes = [await asyncio.to_thread(store.put_image, val[1], val[0]) for kind, val in q_parts if kind == "image"]
    await describe_missing(store, models, [(h, q_text) for h in q_hashes], strict=False)
    q_full = "\n".join([q_text] + [store.image_description(h) or "" for h in q_hashes]).strip()
    opts = " ".join(str(o) for o in (body.options or []))

    rankings = [(1.0, await asyncio.to_thread(cached.bm25.rank, q_full))]
    if opts:  # answer options include distractors, so they vote separately and below the question itself
        rankings.append((0.5, await asyncio.to_thread(cached.bm25.rank, opts)))
    if settings.dense_enabled and q_full:
        try:
            qv = (await embed_cached(store, models, [q_full[:settings.emb_span_chars]], "query"))[0]
            rankings.append((1.0, retrieval.to_units(retrieval.dense_rank(qv, records, cached.dense), unit_of)))
        except (ModelUnavailable, ModelRejected) as e:  # degrade to lexical retrieval rather than fail the query
            log.error("query embedding failed: %s", e)
    if settings.captions_enabled and settings.expand_n and q_full:
        rankings.extend(await expansion_lists(store, models, cached, q_full))
    base = retrieval.rrf(rankings, settings.rrf_k)
    rankings.append((settings.time_weight, retrieval.time_rank(q_text, units, base)))
    if q_hashes:
        hs = set(q_hashes)
        rankings.append((2.0, [k for k, u in enumerate(units) if hs & u.images]))
    fused = retrieval.rrf(rankings, settings.rrf_k)
    if settings.rerank_enabled and fused:
        rankings.extend(await rerank_lists(store, models, q_full, q_hashes, units, fused[:settings.rerank_candidates]))
        fused = retrieval.rrf(rankings, settings.rrf_k)
    if not fused:  # nothing matched lexically and no dense index: fall back to most recent
        fused = sorted(range(len(units)), key=lambda k: -(units[k].ts or 0))
    order = retrieval.order_with_neighbours(fused, units, settings.seed_k, settings.neighbour_window, body.top_k)
    return {"data": await asyncio.to_thread(retrieval.render, units, order, store, settings)}
