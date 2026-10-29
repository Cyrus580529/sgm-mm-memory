"""Calls to the two permitted external models: gpt-4o-mini and text-embedding-v4."""
import asyncio
import logging

import httpx
import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

SCHEMA_PROMPT = """You are indexing a photo or image that a user shared in a conversation, so it can be found again later.
Describe it in these fields, one per line, in English. Be concrete and specific (names of dishes, objects, brands,
places, colours, counts). Write "none" for a field that does not apply.
Scene: where it is and what kind of image it is (photo, screenshot, chart, document, artwork ...)
Objects: the main objects, with colours, counts and distinctive attributes
People: people or animals present, their appearance and what they are doing
Text: all readable text in the image, verbatim (signs, labels, captions, numbers)
Event: what is happening or what the image is about
Time: any visible clue of date, time of day or season
Context: the conversation text the image was attached to: {context}"""


class Models:
    def __init__(self, s: Settings):
        self.s = s
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        self.llm_sem = asyncio.Semaphore(s.llm_concurrency)

    async def _post(self, url: str, key: str, body: dict) -> dict:
        delay = 2.0
        for attempt in range(6):
            try:
                r = await self.client.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    r.raise_for_status()
                log.warning("model call %s -> %s, retrying", url, r.status_code)
            except httpx.TransportError as e:
                log.warning("model call %s failed (%s), retrying", url, type(e).__name__)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise RuntimeError(f"model call to {url} failed after retries")

    async def describe_image(self, data_uri: str, context: str) -> str:
        body = {
            "model": self.s.llm_model,
            "temperature": 0,
            "max_tokens": 400,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": SCHEMA_PROMPT.format(context=context[:600] or "none")},
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
            ]}],
        }
        async with self.llm_sem:
            out = await self._post(f"{self.s.llm_base_url}/chat/completions", self.s.llm_api_key, body)
        return out["choices"][0]["message"]["content"].strip()

    async def embed(self, texts: list) -> np.ndarray:
        """L2-normalised embeddings; DashScope accepts at most 10 inputs per call."""
        vecs = []
        for i in range(0, len(texts), 10):
            batch = [t[:8000] or " " for t in texts[i:i + 10]]
            out = await self._post(f"{self.s.emb_base_url}/embeddings", self.s.emb_api_key,
                                   {"model": self.s.emb_model, "input": batch, "dimensions": self.s.emb_dim,
                                    "encoding_format": "float"})
            vecs.extend(d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"]))
        v = np.asarray(vecs, dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
