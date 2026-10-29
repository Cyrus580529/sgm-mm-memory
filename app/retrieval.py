"""Hybrid retrieval (BM25 + dense + exact image match, fused by RRF) and response assembly."""
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .config import Settings
from .store import Record, Store

STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being do does did
have has had i me my we our you your he him his she her it its they them their this that these those what which who
whom whose when where why how there here then than so not no can could would should will shall may might must just
about into over under again also very s t""".split())

_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> list:
    text = text.lower()
    toks = [w for w in _WORD.findall(text) if w not in STOPWORDS]
    for run in _CJK.findall(text):  # CJK: single characters plus bigrams
        toks.extend(run)
        toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


def bm25_rank(query: str, docs: list, k1: float = 1.2, b: float = 0.75) -> list:
    """Indices of docs with positive BM25 score, best first."""
    q = set(tokenize(query))
    if not q or not docs:
        return []
    tfs = [Counter(tokenize(d)) for d in docs]
    lens = [sum(tf.values()) for tf in tfs]
    avg = sum(lens) / len(lens) or 1.0
    n = len(docs)
    df = {t: sum(1 for tf in tfs if t in tf) for t in q}
    scores = []
    for i, tf in enumerate(tfs):
        s = 0.0
        for t in q:
            f = tf.get(t)
            if f:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[i] / avg))
        if s > 0:
            scores.append((s, i))
    return [i for _, i in sorted(scores, reverse=True)]


def dense_rank(qvec: np.ndarray, records: list) -> list:
    idx = [i for i, r in enumerate(records) if r.vec is not None]
    if not idx:
        return []
    sims = np.stack([records[i].vec for i in idx]) @ qvec
    return [idx[j] for j in np.argsort(-sims)]


def rrf(rankings: list, k: int) -> list:
    """rankings: list of (weight, [doc indices best first]); returns doc indices by fused score."""
    score = Counter()
    for w, ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] += w / (k + r + 1)
    return [i for i, _ in score.most_common()]


def fmt_time(ts: Optional[int]) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def order_with_neighbours(fused: list, records: list, seed_k: int, window: int, limit: int) -> list:
    """Each of the first seed_k hits is followed by its neighbouring turns in the same session;
    the rest of the fused ranking fills up to limit."""
    pos = {}  # (session, order) -> record index, order = position within the session
    by_session = {}
    for i, r in enumerate(records):
        by_session.setdefault(r.session_id, []).append(i)
    for sess, ids in by_session.items():
        ids.sort(key=lambda i: (records[i].ts if records[i].ts is not None else 0, records[i].id))
        for o, i in enumerate(ids):
            pos[i] = (sess, o)
    out, seen = [], set()

    def push(i):
        if i not in seen and len(out) < limit:
            seen.add(i)
            out.append(i)

    for i in fused[:seed_k]:
        push(i)
        sess, o = pos[i]
        ids = by_session[sess]
        for d in range(1, window + 1):
            for j in (o - d, o + d):
                if 0 <= j < len(ids):
                    push(ids[j])
    for i in fused[seed_k:]:
        push(i)
    return out


def render(records: list, order: list, store: Store, s: Settings) -> list:
    """Search response items. Original images are attached while the image budget lasts;
    after that the image's schema description stands in for it."""
    items, img_bytes, n_img = [], 0, 0
    for i in order:
        r: Record = records[i]
        header = f"[{fmt_time(r.ts)} | session {r.session_id}] {r.role}:".replace("[ | ", "[")
        parts, has_image = [{"type": "text", "text": header}], False
        for kind, val in r.parts:
            if kind == "text":
                parts.append({"type": "text", "text": val})
                continue
            desc = store.image_description(val)
            got = store.image_data_uri(val) if n_img < s.max_images else None
            if got and img_bytes + got[1] <= s.max_images_bytes:
                parts.append({"type": "image_url", "image_url": {"url": got[0]}})
                img_bytes += got[1]
                n_img += 1
                has_image = True
                if desc:
                    parts.append({"type": "text", "text": f"(image description)\n{desc}"})
            else:
                parts.append({"type": "text", "text": f"[image]\n{desc or '(no description)'}"})
        if has_image:
            content = parts
        else:
            content = "\n".join(p["text"] for p in parts)
        items.append({"id": str(r.id), "content": content, "score": round(1.0 / (1 + len(items)), 6),
                      **({"created_at": datetime.fromtimestamp(r.ts / 1000, tz=timezone.utc).isoformat()}
                         if r.ts is not None else {})})
    return items
