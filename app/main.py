"""AML Add/Search service."""
import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import re
import shutil
import time
from collections import Counter, OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import retrieval
from .config import check_competition_models, settings
from .meta import meta_text
from .models import DESC_VERSION, ModelRejected, ModelUnavailable, Models
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("aml")

MIMES = {"image/jpeg", "image/png", "image/webp"}
DATA_URI = re.compile(r"^data:(image/[a-z]+);base64,(.*)$", re.S)
MAX_IMAGE = 10 * 1024 * 1024
TS_RANGE = (0, 4_102_444_800_000)  # 1970 .. 2100 in Unix ms; anything else is treated as "no timestamp"


class Message(BaseModel):
    role: str = Field(min_length=1)
    content: Union[str, list]
    timestamp: Optional[Union[int, float]] = None


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
stats: Counter = Counter()


@asynccontextmanager
async def lifespan(_: FastAPI):
    check_competition_models(settings)
    if not settings.memory_api_key:
        raise RuntimeError("MEMORY_API_KEY is not set")
    state["store"] = Store(settings.data_dir, settings.emb_dim, DESC_VERSION)
    state["models"] = Models(settings)
    state["add_locks"] = {}
    state["defers"] = Counter()          # 503s already given per request_id
    state["indexes"] = OrderedDict()
    state["index_locks"] = {}
    state["dirty"] = set()               # users whose records were repaired since their index was built
    state["add_gate"] = asyncio.Semaphore(settings.add_max_inflight)
    state["render_gate"] = asyncio.Semaphore(settings.render_max_inflight)
    state["started"] = time.time()
    state["last_error"] = None
    repair = asyncio.create_task(repair_loop())
    log.info("started; captions=%s dense=%s rerank=%s", settings.captions_enabled, settings.dense_enabled,
             settings.rerank_enabled)
    yield
    repair.cancel()


app = FastAPI(title="SGM-MM-Memory", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    """Unexpected failures answer 500 (the platform retries) without exposing request content."""
    stats["errors_unhandled"] += 1
    state["last_error"] = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {type(exc).__name__}: {str(exc)[:200]}"
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"success": False, "error": "internal error"})


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


def clean_ts(ts) -> Optional[int]:
    if ts is None:
        return None
    try:
        v = int(ts)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if TS_RANGE[0] < v < TS_RANGE[1] else None


def query_parts(query) -> list:
    """Search is lenient: the platform does not retry a permanent 4xx on Search, so an unusable part (bad image,
    unknown type) is skipped rather than failing the question."""
    if isinstance(query, str):
        return [("text", query)] if query.strip() else []
    out = []
    for p in query if isinstance(query, list) else []:
        try:
            out.extend(split_content([p]))
        except BadInput:
            stats["search_parts_skipped"] += 1
    return out


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
            m = DATA_URI.match(url or "") if isinstance(url, str) else None
            if not m:
                raise BadInput("image_url must be a base64 data URI")
            mime = "image/jpeg" if m.group(1) == "image/jpg" else m.group(1)
            if mime not in MIMES:
                raise BadInput(f"unsupported image type {mime}")
            try:
                raw = base64.b64decode(m.group(2), validate=False)
            except (binascii.Error, ValueError):
                raise BadInput("invalid base64 image")
            if not raw:
                raise BadInput("empty image")
            if len(raw) > MAX_IMAGE:
                raise BadInput("image larger than 10 MiB")
            out.append(("image", (mime, raw)))
        else:
            raise BadInput(f"unknown content part type {p.get('type')!r}")
    if not out:
        raise BadInput("empty content")
    return out


# ---- model-backed steps ----------------------------------------------------------------------------------
async def describe_missing(store: Store, models: Models, jobs: list, strict: bool) -> None:
    """jobs: (hash, context). Describes each new image once (cached by hash). A refused or unparseable answer is
    stored as an empty description. If the provider is unavailable, strict (Add) re-raises; otherwise (Search)
    the image simply goes without a description."""
    if not settings.captions_enabled:
        return
    todo = {}
    for h, ctx in jobs:
        if h not in todo and not await asyncio.to_thread(store.description_current, h):
            todo[h] = ctx

    async def one(h, ctx):
        got = await asyncio.to_thread(store.image_data_uri, h)
        if got is None:  # not a readable image: nothing to describe
            await asyncio.to_thread(store.set_image_description, h, "")
            return
        try:
            desc = await models.describe_image(got[0], ctx)
            stats["descriptions"] += 1
        except ModelUnavailable:
            raise
        except Exception as e:  # refused, or a response we cannot parse: no description, keep going
            stats["descriptions_refused"] += 1
            log.error("image description refused for %s: %s", h[:12], e)
            desc = ""
        await asyncio.to_thread(store.set_image_description, h, desc)

    results = await asyncio.gather(*(one(h, c) for h, c in todo.items()), return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        stats["provider_unavailable_llm"] += 1
        log.error("%d image descriptions failed: %s", len(errors), errors[0])
        if strict:
            raise ModelUnavailable(str(errors[0]))


async def embed_cached(store: Store, models: Models, texts: list, text_type: str) -> list:
    """Embeddings, computed once per distinct text (the same history may be written for many samples).
    Any provider failure surfaces as ModelUnavailable."""
    tag = f"{settings.emb_model}|{settings.emb_dim}|{settings.emb_style}|{text_type}|"
    keys = [hashlib.sha256((tag + t).encode()).hexdigest() for t in texts]
    have = await asyncio.to_thread(store.cached_vecs, list(set(keys)))
    todo = {k: t for k, t in zip(keys, texts) if k not in have}
    if todo:
        try:
            new = await models.embed(list(todo.values()), text_type)
            if len(new) != len(todo) or new.shape[1] != settings.emb_dim:
                raise ModelUnavailable(f"embedding shape {new.shape} for {len(todo)} texts")
        except ModelUnavailable:
            stats["provider_unavailable_emb"] += 1
            raise
        except Exception as e:
            stats["provider_unavailable_emb"] += 1
            raise ModelUnavailable(f"embedding failed: {type(e).__name__}: {e}")
        fresh = dict(zip(todo.keys(), new))
        await asyncio.to_thread(store.put_vecs, fresh)
        have.update(fresh)
    return [have[k] for k in keys]


def record_texts(store: Store, role: str, text: str, ts, event_ts, images: list) -> tuple:
    """(embed_text, index_text) of one message: its text plus image metadata and descriptions; index_text adds dates."""
    imgs = []
    for h in images:
        meta, desc = meta_text(store.image_meta(h)), store.image_description(h)
        imgs.append("[image] " + " ".join(x for x in (meta, desc) if x))
    dates = {retrieval.date_words(ts), retrieval.date_words(event_ts)} - {""}
    embed_text = "\n".join([f"{role}: {text}"] + imgs)
    return embed_text, "\n".join(sorted(dates) + [embed_text])


async def embed_rows(store: Store, models: Models, texts: list) -> list:
    """One (spans, dim) matrix per text."""
    pieces = [retrieval.spans(t, settings.emb_span_chars) for t in texts]
    vecs = await embed_cached(store, models, [p for ps in pieces for p in ps], "document")
    out, k = [], 0
    for ps in pieces:
        out.append(np.stack(vecs[k:k + len(ps)]))
        k += len(ps)
    return out


async def rerank_lists(store: Store, models: Models, q_full: str, q_hashes: list, units: list, cands: list) -> list:
    """Extra (weight, unit ranking) votes from the rerankers: one over candidate texts (queried with the question
    text), one over candidate images (queried with the question's own image when it has one, e.g. "what plant is
    this?", else with the text). Any failure only removes the vote."""
    try:
        q_got = await asyncio.to_thread(store.image_data_uri, q_hashes[0]) if q_hashes else None
        q_image = {"image": q_got[0]} if q_got else None
        text_query = q_full[:2000] if q_full else None
        image_query = q_image or text_query
        if image_query is None:
            return []
        img_units = [k for k in cands if units[k].images][:40] if settings.rerank_images else []

        def build_docs():
            imgs = [(k, store.image_data_uri(retrieval.first_image(units[k]))) for k in img_units]
            imgs = [(k, got) for k, got in imgs if got]
            return ([{"text": retrieval.unit_text(units[k], store)[:3000] or " "} for k in cands],
                    [k for k, _ in imgs], [{"image": got[0]} for _, got in imgs])

        docs, img_units, idocs = await asyncio.to_thread(build_docs)
        jobs = [(cands, 0.0, models.rerank(text_query, docs, settings.rerank_text_model))] if text_query else []
        if img_units:
            jobs.append((img_units, settings.rerank_vl_min, models.rerank(image_query, idocs, settings.rerank_vl_model)))
        out = []
        results = await asyncio.gather(*(j for _, _, j in jobs), return_exceptions=True)
        for (keys, min_score, _), res in zip(jobs, results):
            if isinstance(res, Exception):
                stats["rerank_failed"] += 1
                log.error("rerank failed: %s", res)
                continue
            out.append((settings.rerank_weight,
                        [keys[i] for i, sc in res if isinstance(i, int) and 0 <= i < len(keys) and sc >= min_score]))
        return out
    except Exception as e:
        stats["rerank_failed"] += 1
        log.error("rerank skipped: %s", e)
        return []


async def expansion_lists(store: Store, models: Models, cached, q_full: str) -> list:
    """Lexical and dense votes for gpt-4o-mini's rewrites of the question; any failure just drops them."""
    try:
        queries = await models.expand_query(q_full, settings.expand_n)
    except Exception as e:
        stats["expansion_failed"] += 1
        log.error("query expansion failed: %s", e)
        return []
    lists = [await asyncio.to_thread(cached.bm25.rank, q) for q in queries]
    if settings.dense_enabled and queries:
        try:
            vecs = await embed_cached(store, models, queries, "query")
            lists += [retrieval.to_units(retrieval.dense_rank(v, cached.records, cached.dense), cached.unit_of)
                      for v in vecs]
        except ModelUnavailable as e:
            log.error("expansion embedding failed: %s", e)
    return [(settings.expand_weight / len(lists), l) for l in lists] if lists else []


# ---- background repair -------------------------------------------------------------------------------------
async def repair_loop():
    """Fill in descriptions and vectors of records that were committed degraded during a provider outage."""
    while True:
        try:
            await asyncio.sleep(settings.repair_interval)
            await repair_once()
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.error("repair pass failed: %s", e)


async def repair_once(batch: int = 50) -> int:
    store: Store = state["store"]
    models: Models = state["models"]
    rows = await asyncio.to_thread(store.repair_batch, batch)
    done = 0
    for r in rows:
        try:
            await describe_missing(store, models, [(h, r["text"]) for h in r["images"]], strict=True)
            embed_text, index_text = await asyncio.to_thread(record_texts, store, r["role"], r["text"], r["ts"],
                                                             r["event_ts"], r["images"])
            vec = (await embed_rows(store, models, [embed_text]))[0] if settings.dense_enabled else None
        except ModelUnavailable as e:  # still down: try again next pass
            log.warning("repair of record %s postponed: %s", r["id"], e)
            break
        await asyncio.to_thread(store.repaired, r["id"], index_text, vec)
        state["dirty"].add(r["user_id"])
        stats["repaired"] += 1
        done += 1
    return done


# ---- endpoints ---------------------------------------------------------------------------------------------
@app.get("/health")
async def health():
    try:
        await asyncio.to_thread(state["store"].ping)
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ok"}


@app.get("/stats")
async def get_stats(req: Request):
    """Operational counters for watching an evaluation run (authenticated; no request content)."""
    if not authorized(req):
        return error(401, "unauthorized")
    store: Store = state["store"]
    disk = shutil.disk_usage(settings.data_dir)
    return {"uptime_s": int(time.time() - state["started"]), "counters": dict(stats),
            "store": await asyncio.to_thread(store.counts), "disk_free_gb": round(disk.free / 1e9, 1),
            "index_cache_users": len(state["indexes"]), "last_error": state["last_error"],
            "models": {"captions": settings.captions_enabled, "dense": settings.dense_enabled,
                       "rerank": settings.rerank_enabled, "expansion": bool(settings.captions_enabled and settings.expand_n)}}


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
        ts = clean_ts(m.timestamp)
        taken = [t for t in (clean_ts((store.image_meta(h) or {}).get("taken_ms")) for h in hashes) if t]
        rows.append({"msg_index": idx, "role": m.role, "ts": ts, "text": text,
                     "event_ts": taken[0] if taken else ts, "images": hashes, "parts": ref_parts, "vec": None})
    return rows, jobs


def finish_rows(store: Store, rows: list) -> None:
    for r in rows:
        r["embed_text"], r["index_text"] = record_texts(store, r["role"], r["text"], r["ts"], r["event_ts"],
                                                        r["images"])


@app.post("/v1/memories/add")
async def add(req: Request):
    if not authorized(req):
        return error(401, "unauthorized")
    async with state["add_gate"]:  # the body is read and decoded only once a slot is free
        return await add_inner(req)


async def add_inner(req: Request):
    try:
        body = AddBody.model_validate(await req.json())
    except Exception as e:
        stats["add_invalid"] += 1
        return error(422, f"invalid request: {str(e)[:300]}")
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
                stats["add_duplicate"] += 1
                return ok
            if st == "conflict":
                stats["add_conflict"] += 1
                return error(409, "request_id already used with different content")
            try:
                parsed = [split_content(m.content) for m in body.messages]
            except BadInput as e:
                stats["add_invalid"] += 1
                return error(400, str(e))
            rows, jobs = await asyncio.to_thread(prepare_rows, store, body, parsed)
            degraded = False
            try:
                await describe_missing(store, models, jobs, strict=True)
                await asyncio.to_thread(finish_rows, store, rows)
                if settings.dense_enabled:
                    for r, v in zip(rows, await embed_rows(store, models, [r["embed_text"] for r in rows])):
                        r["vec"] = v
            except ModelUnavailable as e:
                state["defers"][body.request_id] += 1
                if state["defers"][body.request_id] <= settings.add_max_defers:
                    # nothing committed; finished descriptions are cached, so the platform's retry is cheap
                    stats["add_deferred"] += 1
                    log.error("add %s deferred: %s", body.request_id, e)
                    return JSONResponse(status_code=503, headers={"Retry-After": str(settings.retry_after)},
                                        content={"success": False, "error": "model provider unavailable, retry later"})
                # the outage outlasts several retries: commit what we have, repair in the background
                degraded = True
                await asyncio.to_thread(finish_rows, store, rows)
                for r in rows:
                    r["vec"] = None
                stats["add_degraded"] += 1
                log.error("add %s committed degraded after %d deferrals", body.request_id,
                          state["defers"][body.request_id] - 1)
            await asyncio.to_thread(store.commit_add, body.request_id, body.user_id, body.session_id,
                                    payload_hash, rows, degraded)
            state["defers"].pop(body.request_id, None)
            stats["add_ok"] += 1
    finally:
        if not lock.locked():
            state["add_locks"].pop(body.request_id, None)
    return ok


async def user_index(store: Store, user_id: str):
    """The user's cached index, rebuilt (once, under a per-user lock) when records were added or repaired.
    The cache holds at most INDEX_CACHE_RECORDS records across users; least recently used users are evicted."""
    cache = state["indexes"]
    lock = state["index_locks"].setdefault(user_id, asyncio.Lock())
    async with lock:
        n = await asyncio.to_thread(store.user_count, user_id)
        if not n:
            return None
        idx = cache.get(user_id)
        if idx is None or len(idx.records) != n or user_id in state["dirty"]:
            state["dirty"].discard(user_id)
            idx = await asyncio.to_thread(lambda: retrieval.UserIndex(store.user_records(user_id), store))
            cache[user_id] = idx
            stats["index_builds"] += 1
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
        q_parts = query_parts(body.query)
    except Exception as e:
        stats["search_invalid"] += 1
        return error(422, f"invalid request: {str(e)[:300]}")
    store: Store = state["store"]
    models: Models = state["models"]
    stats["searches"] += 1

    cached = await user_index(store, body.user_id)
    if cached is None:
        return {"data": []}
    records, units, unit_of = cached.records, cached.units, cached.unit_of

    q_text = "\n".join(v for k, v in q_parts if k == "text")
    q_hashes = [await asyncio.to_thread(store.put_image, val[1], val[0]) for kind, val in q_parts if kind == "image"]
    await describe_missing(store, models, [(h, q_text) for h in q_hashes], strict=False)
    q_full = "\n".join([retrieval.strip_anchor(q_text)] + [store.image_description(h) or "" for h in q_hashes]).strip()
    opts = " ".join(str(o) for o in (body.options or []))

    rankings = [(1.0, await asyncio.to_thread(cached.bm25.rank, q_full))]
    if opts and settings.options_weight:  # options are mostly distractors: a separate, lower vote
        rankings.append((settings.options_weight, await asyncio.to_thread(cached.bm25.rank, opts)))
    if settings.dense_enabled and q_full:
        try:
            qv = (await embed_cached(store, models, [q_full[:settings.emb_span_chars]], "query"))[0]
            rankings.append((1.0, retrieval.to_units(retrieval.dense_rank(qv, records, cached.dense), unit_of)))
        except ModelUnavailable as e:  # degrade to lexical retrieval rather than fail the query
            log.error("query embedding failed: %s", e)
    if settings.captions_enabled and settings.expand_n and q_full:
        rankings.extend(await expansion_lists(store, models, cached, q_full))
    base = retrieval.rrf(rankings, settings.rrf_k)
    rankings.append((settings.time_weight, retrieval.time_rank(q_text, units, base)))
    rankings.append((settings.recency_weight, retrieval.recency_rank(q_text, units, base, settings.recency_depth)))
    if q_hashes:  # the question shows a picture the user shared before (exact or re-encoded copy)
        q_dh = list((await asyncio.to_thread(store.image_dhashes, q_hashes)).values())
        hs = set(q_hashes)
        exact = [k for k, u in enumerate(units) if hs & u.images]
        near = cached.image_matches(q_dh, settings.image_match_bits) if q_dh else []
        rankings.append((2.0, exact + [k for k in near if k not in set(exact)]))
    fused = retrieval.rrf(rankings, settings.rrf_k)
    if settings.rerank_enabled and fused:
        rankings.extend(await rerank_lists(store, models, q_full, q_hashes, units, fused[:settings.rerank_candidates]))
        fused = retrieval.rrf(rankings, settings.rrf_k)
    if not fused:  # nothing matched lexically and no dense index: fall back to most recent
        fused = sorted(range(len(units)), key=lambda k: -(units[k].ts or 0))
    fits = settings.return_all_if_fits and len(units) <= body.top_k
    limit = body.top_k if fits or not settings.result_limit else min(body.top_k, settings.result_limit)
    order = retrieval.order_with_neighbours(fused, units, settings.seed_k, settings.neighbour_window, limit)
    if fits and len(order) < len(units):
        seen = set(order)  # the whole memory fits: keep the ranking, then append what no signal matched
        order += sorted((k for k in range(len(units)) if k not in seen), key=lambda k: -(units[k].ts or 0))

    budget = retrieval.answer_budget(settings, q_text, opts)

    def assemble():
        items = retrieval.render(units, order, store, settings, budget)[:body.top_k]
        return retrieval.fit_response(items)

    async with state["render_gate"]:
        return {"data": await asyncio.to_thread(assemble)}
