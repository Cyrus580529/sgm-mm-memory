"""Calls to external models: gpt-4o-mini (descriptions), text-embedding-v4 (dense retrieval) and an
optional reranker (qwen3-vl-rerank / qwen3-rerank)."""
import asyncio
import json
import logging

import httpx
import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

SCHEMA_PROMPT = """You are indexing an image that a user shared, so it can be found again later and questions about it
can be answered. Describe it in these fields, one per line. Write in the language of the conversation context below
(if it is Chinese, write in Chinese; if there is no context, use the language of the text in the image, else English).
Keep field names as given. Copy text visible in the image exactly as written, never translated. Be concrete and specific: exact names of
dishes, objects, products, brands, places, species; colours, sizes, counts and positions; distinguishing details
that would tell this image apart from similar ones. Write "none" for a field that does not apply.
Scene: where it is and what kind of image it is (photo, phone screenshot of which app, slide, chart, document,
receipt, artwork ...)
Objects: the main objects, each with colour, count, material, brand and distinctive attributes
People: people or animals present, their appearance, clothing and what they are doing
Text: every piece of readable text in the image, verbatim and in its original language (titles, signs, labels,
prices, numbers, app names, chat messages and timestamps in screenshots, slide text)
Event: what is happening or what the image is about
Time: any visible clue of date, time of day or season
Context: the conversation text the image was attached to: {context}"""

EXPAND_PROMPT = """You help search a person's long-term memory (past conversations, photos with descriptions, emails,
documents) for evidence that answers a question. Write {n} short, different search queries that together cover what
the evidence would contain: the key people, objects, places, events and exact terms; for a question about a picture,
describe what the picture would show. Do not answer the question and do not add facts that are not implied by it.
Return JSON: {{"queries": ["...", "..."]}}
Question: {question}"""

DOC_INSTRUCT = "Given a question about a person's past conversations, photos and documents, retrieve the memory that helps answer it."


class ModelRejected(Exception):
    """The provider refused this input for good (4xx other than rate limits); retrying will not help."""


class ModelUnavailable(Exception):
    """The provider kept failing (timeouts, 429, 5xx); the caller may retry later."""


class Models:
    def __init__(self, s: Settings):
        self.s = s
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0))
        self.llm_sem = asyncio.Semaphore(s.llm_concurrency)
        self.emb_sem = asyncio.Semaphore(s.emb_concurrency)
        self.rerank_sem = asyncio.Semaphore(s.rerank_concurrency)  # Search runs up to 256-way concurrent

    async def _post(self, url: str, key: str, body: dict, attempts: int = 6) -> dict:
        delay = 2.0
        for attempt in range(attempts):
            try:
                r = await self.client.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in (408, 409, 425, 429) and r.status_code < 500:
                    log.error("model call %s -> %s", url, r.status_code)  # no body: it may echo input content
                    raise ModelRejected(f"{url} -> {r.status_code}")
                log.warning("model call %s -> %s, retrying", url, r.status_code)
            except httpx.TransportError as e:
                log.warning("model call %s failed (%s), retrying", url, type(e).__name__)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise ModelUnavailable(f"model call to {url} failed after {attempts} attempts")

    # ---- gpt-4o-mini ------------------------------------------------------
    async def describe_image(self, data_uri: str, context: str) -> str:
        body = {
            "model": self.s.llm_model,
            "temperature": 0,
            "max_tokens": 700,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": SCHEMA_PROMPT.format(context=context[:600] or "none")},
                {"type": "image_url", "image_url": {"url": data_uri, "detail": self.s.caption_detail}},
            ]}],
        }
        async with self.llm_sem:
            out = await self._post(f"{self.s.llm_base_url}/chat/completions", self.s.llm_api_key, body)
        return out["choices"][0]["message"]["content"].strip()

    async def expand_query(self, question: str, n: int) -> list:
        body = {"model": self.s.llm_model, "temperature": 0, "max_tokens": 200,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": EXPAND_PROMPT.format(n=n, question=question[:2000])}]}
        async with self.llm_sem:
            out = await self._post(f"{self.s.llm_base_url}/chat/completions", self.s.llm_api_key, body, attempts=3)
        try:
            qs = json.loads(out["choices"][0]["message"]["content"]).get("queries", [])
        except (ValueError, AttributeError):
            return []
        return [q.strip() for q in qs if isinstance(q, str) and q.strip()][:n]

    # ---- text-embedding-v4 ------------------------------------------------
    async def embed(self, texts: list, text_type: str = "document") -> np.ndarray:
        """L2-normalised embeddings, at most 10 inputs per call. text_type ("query"/"document") is used by the
        DashScope native API; the OpenAI-compatible API has no such field."""
        async def batch(chunk):
            chunk = [t or " " for t in chunk]
            async with self.emb_sem:
                if self.s.emb_style == "dashscope":
                    out = await self._post(f"{self.s.emb_base_url}/services/embeddings/text-embedding/text-embedding",
                                           self.s.emb_api_key,
                                           {"model": self.s.emb_model, "input": {"texts": chunk},
                                            "parameters": {"text_type": text_type, "dimension": self.s.emb_dim}})
                    rows = sorted(out["output"]["embeddings"], key=lambda d: d["text_index"])
                else:
                    out = await self._post(f"{self.s.emb_base_url}/embeddings", self.s.emb_api_key,
                                           {"model": self.s.emb_model, "input": chunk, "dimensions": self.s.emb_dim,
                                            "encoding_format": "float"})
                    rows = sorted(out["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in rows]

        parts = await asyncio.gather(*(batch(texts[i:i + 10]) for i in range(0, len(texts), 10)))
        v = np.asarray([e for p in parts for e in p], dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)

    # ---- reranker ---------------------------------------------------------
    async def rerank(self, query, documents: list, model: str) -> list:
        """(index, relevance score) of documents, most relevant first. query: str or {"image": data_uri};
        documents: list of {"text": ...} or {"image": data_uri}."""
        if self.s.rerank_style == "jina":  # OpenAI-style gateways: text only
            if not isinstance(query, str) or any("text" not in d for d in documents):
                raise ModelRejected("this reranker API takes text only")
            body = {"model": model, "query": query, "documents": [d["text"] for d in documents],
                    "top_n": len(documents)}
            async with self.rerank_sem:
                out = await self._post(f"{self.s.rerank_base_url}/rerank", self.s.rerank_api_key, body, attempts=3)
            res = sorted(out["results"], key=lambda r: -r["relevance_score"])
            return [(r["index"], r["relevance_score"]) for r in res]
        body = {"model": model, "input": {"query": query, "documents": documents},
                "parameters": {"top_n": len(documents), "return_documents": False, "instruct": DOC_INSTRUCT}}
        async with self.rerank_sem:
            out = await self._post(f"{self.s.rerank_base_url}/services/rerank/text-rerank/text-rerank",
                                   self.s.rerank_api_key, body, attempts=3)
        res = sorted(out["output"]["results"], key=lambda r: -r["relevance_score"])
        return [(r["index"], r["relevance_score"]) for r in res]
