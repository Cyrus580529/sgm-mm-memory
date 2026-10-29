"""Hybrid retrieval (BM25 + dense + exact image match, fused by RRF) and response assembly."""
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np

from .config import Settings
from .meta import meta_text
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


class BM25:
    """BM25 over a fixed list of documents, tokenised once."""

    def __init__(self, docs: list, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens) if self.lens else 0) or 1.0
        self.postings = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.postings.setdefault(t, []).append(i)

    def rank(self, query: str) -> list:
        """Indices of documents with positive score, best first."""
        n = len(self.tfs)
        scores = Counter()
        for t in set(tokenize(query)):
            post = self.postings.get(t)
            if not post:
                continue
            idf = math.log(1 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for i in post:
                f = self.tfs[i][t]
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return [i for i, _ in scores.most_common()]


def bm25_rank(query: str, docs: list) -> list:
    return BM25(docs).rank(query)


def dense_rank(qvec: np.ndarray, records: list, matrix: Optional[tuple] = None) -> list:
    """matrix: optional precomputed (record indices, stacked vectors)."""
    idx, mat = matrix if matrix is not None else dense_matrix(records)
    if not idx:
        return []
    return [idx[j] for j in np.argsort(-(mat @ qvec))]


def dense_matrix(records: list) -> tuple:
    idx = [i for i, r in enumerate(records) if r.vec is not None]
    return idx, (np.stack([records[i].vec for i in idx]) if idx else None)


class UserIndex:
    """Everything Search needs for one user, rebuilt only when the user's record count changes."""

    def __init__(self, records: list):
        self.records = records
        self.units, self.unit_of = build_units(records)
        self.bm25 = BM25([u.index_text for u in self.units])
        self.dense = dense_matrix(records)


def rrf(rankings: list, k: int) -> list:
    """rankings: list of (weight, [doc indices best first]); returns doc indices by fused score."""
    score = Counter()
    for w, ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] += w / (k + r + 1)
    return [i for i, _ in score.most_common()]


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
_MONTH_RE = re.compile(r"\b(early|mid|late|beginning of|end of)?\s*(" + "|".join(MONTHS + ["sept"] + [m[:3] for m in MONTHS])
                       + r")\.?(?:\s+(\d{1,2})(?:st|nd|rd|th)?)?(?:,?\s+(\d{4}))?\b", re.I)
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def date_words(ts: Optional[int]) -> str:
    """Date in the forms questions use, added to the retrieval text."""
    if ts is None:
        return ""
    d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    part = "early" if d.day <= 10 else ("mid" if d.day <= 20 else "late")
    return f"{d:%Y-%m-%d} {d:%B} {d.day}, {d.year} {part} {d:%B} {d.year} {d:%A}"


_TODAY_RE = re.compile(r"\btoday(?:'s date)? is\s+(?:\w+day,?\s+)?(\S)", re.I)
_DMY_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS + ["sept"] + [m[:3] for m in MONTHS])
                     + r")\.?,?\s+(\d{4})\b", re.I)
_AGO_RE = re.compile(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month|year)s?\s+ago\b", re.I)
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10}


def _month_no(name: str) -> int:
    name = name.lower()
    return next(i + 1 for i, full in enumerate(MONTHS) if full.startswith(name[:3]))


def _first_date(text: str) -> Optional[date]:
    m = _ISO_RE.search(text)
    if m and m.group(3):
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _MONTH_RE.search(text)
    if m and m.group(3) and m.group(4):
        return date(int(m.group(4)), _month_no(m.group(2)), int(m.group(3)))
    m = _DMY_RE.search(text)
    if m:
        return date(int(m.group(3)), _month_no(m.group(2)), int(m.group(1)))
    return None


def _month_span(y: int, mo: int) -> tuple:
    start = date(y, mo, 1)
    nxt = date(y + (mo == 12), mo % 12 + 1, 1)
    return start, nxt - timedelta(days=1)


def query_periods(query: str) -> list:
    """Date predicates for periods the query names: absolute dates and months, and relative expressions
    ("last week", "3 days ago", ...) anchored on a stated "today is ..." date."""
    preds = []
    anchor, body = None, query
    t = _TODAY_RE.search(query)
    if t:
        tail = query[t.start(1):t.start(1) + 40]
        try:
            anchor = _first_date(tail)
        except ValueError:
            anchor = None
        if anchor:  # drop "today is <date>" so the anchor date is not read as an asked period
            m = _ISO_RE.search(tail) or _MONTH_RE.search(tail) or _DMY_RE.search(tail)
            body = query[:t.start()] + " " + query[t.start(1) + m.end():]

    years = [int(y) for y in _YEAR_RE.findall(body)]

    def span(lo: date, hi: date):
        preds.append(lambda d, lo=lo, hi=hi: lo <= d <= hi)

    for m in _ISO_RE.finditer(body):
        y, mo, dd = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mo <= 12:
            try:
                span(*((date(y, mo, int(dd)),) * 2 if dd else _month_span(y, mo)))
            except ValueError:
                pass
    for m in _MONTH_RE.finditer(body):
        name = m.group(2).lower()
        if name == "may" and not (m.group(1) or m.group(3) or m.group(4)):
            continue  # "may" as a verb
        mo = _month_no(name)
        if m.group(4):
            y = int(m.group(4))
        elif len(years) == 1:
            y = years[0]
        elif anchor:  # the most recent such month up to the anchor date
            y = anchor.year if mo <= anchor.month else anchor.year - 1
        else:
            y = None
        mod = (m.group(1) or "").lower()
        lo, hi = 1, 31
        if m.group(3):
            lo = hi = int(m.group(3))
        elif mod in ("early", "beginning of"):
            hi = 10
        elif mod == "mid":
            lo, hi = 11, 20
        elif mod in ("late", "end of"):
            lo = 21
        preds.append(lambda d, y=y, mo=mo, lo=lo, hi=hi: (y is None or d.year == y) and d.month == mo
                     and lo <= d.day <= hi)
    if anchor:
        q = body.lower()
        wk = anchor - timedelta(days=anchor.weekday())  # Monday of the anchor week
        if "yesterday" in q:
            span(anchor - timedelta(days=1), anchor - timedelta(days=1))
        if re.search(r"\btoday\b|\bthis (morning|afternoon|evening)\b|\btonight\b", q):
            span(anchor, anchor)
        if "last week" in q or "past week" in q:
            span(wk - timedelta(days=8), anchor - timedelta(days=1))
        if "this week" in q:
            span(wk, anchor)
        if "last weekend" in q or "past weekend" in q:
            span(wk - timedelta(days=3), wk - timedelta(days=1))
        if "last month" in q or "past month" in q:
            prev = anchor.replace(day=1) - timedelta(days=1)
            span(_month_span(prev.year, prev.month)[0], anchor - timedelta(days=1))
        if "this month" in q:
            span(anchor.replace(day=1), anchor)
        if "last year" in q or "past year" in q:
            span(date(anchor.year - 1, 1, 1), anchor - timedelta(days=1))
        if "this year" in q:
            span(date(anchor.year, 1, 1), anchor)
        for m in _AGO_RE.finditer(q):
            n = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1).lower()]
            unit = m.group(2).lower()
            days, slack = {"day": (n, 1), "week": (7 * n, 3), "month": (30 * n, 10), "year": (365 * n, 60)}[unit]
            c = anchor - timedelta(days=days)
            span(c - timedelta(days=slack), c + timedelta(days=slack))
        if not preds:  # "today is X" alone: the past up to the anchor is the plausible range
            span(date(1970, 1, 1), anchor)
    return preds


def time_rank(query: str, items: list, base: list) -> list:
    """Items (records or units) whose event date lies in a period the query names, in base-ranking order."""
    preds = query_periods(query)
    if not preds:
        return []

    def inside(it):
        ts = getattr(it, "event_ts", None) or it.ts
        if ts is None:
            return False
        d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
        return any(p(d) for p in preds)

    pos = {i: k for k, i in enumerate(base)}
    hits = [i for i, it in enumerate(items) if inside(it)]
    return sorted(hits, key=lambda i: pos.get(i, len(base) + i))


def fmt_time(ts: Optional[int]) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


@dataclass
class Unit:
    """One exchange within a session: the user message(s) of one Add and the replies that follow them."""
    records: list
    session_id: str
    pos: int = 0  # position within its session

    @property
    def ts(self):
        return self.records[0].ts

    @property
    def event_ts(self):
        return next((r.event_ts for r in self.records if r.event_ts is not None), self.ts)

    @property
    def index_text(self):
        return "\n".join(r.index_text for r in self.records)

    @property
    def images(self):
        return {h for r in self.records for h in r.images}


def build_units(records: list) -> tuple:
    """(units, unit index of each record). Sessions are ordered by timestamp, then arrival."""
    by_session = {}
    for i, r in enumerate(records):
        by_session.setdefault(r.session_id, []).append(i)
    units, unit_of = [], {}
    for sess, ids in by_session.items():
        ids.sort(key=lambda i: (records[i].ts if records[i].ts is not None else 0, records[i].id))
        cur, pos = None, 0
        for i in ids:
            r = records[i]
            # a user message opens a new exchange after a reply, or when it arrives in a separate Add
            if cur is None or (r.role == "user" and (any(x.role != "user" for x in cur.records)
                                                     or r.request_id != cur.records[-1].request_id)):
                cur = Unit([], sess, pos)
                pos += 1
                units.append(cur)
            cur.records.append(r)
            unit_of[i] = len(units) - 1
    return units, unit_of


def to_units(ranking: list, unit_of: dict) -> list:
    """Record ranking -> unit ranking (a unit ranks at its best record)."""
    out, seen = [], set()
    for i in ranking:
        u = unit_of[i]
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def order_with_neighbours(fused: list, units: list, seed_k: int, window: int, limit: int) -> list:
    """Each of the first seed_k hits is followed by its neighbouring exchanges in the same session;
    the rest of the fused ranking fills up to limit."""
    by_pos = {(u.session_id, u.pos): k for k, u in enumerate(units)}
    out, seen = [], set()

    def push(k):
        if k is not None and k not in seen and len(out) < limit:
            seen.add(k)
            out.append(k)

    for k in fused[:seed_k]:
        push(k)
        u = units[k]
        for d in range(1, window + 1):
            push(by_pos.get((u.session_id, u.pos - d)))
            push(by_pos.get((u.session_id, u.pos + d)))
    for k in fused[seed_k:]:
        push(k)
    return out


def render(units: list, order: list, store: Store, s: Settings) -> list:
    """Search response items. Original images are attached while the image budget lasts;
    after that the image's schema description stands in for it."""
    items, img_bytes, n_img = [], 0, 0
    for k in order:
        u: Unit = units[k]
        when = fmt_time(u.ts)
        parts, has_image = [{"type": "text", "text": f"[{when + ' | ' if when else ''}session {u.session_id}]"}], False
        for r in u.records:
            parts.append({"type": "text", "text": f"{r.role}:"})
            for kind, val in r.parts:
                if kind == "text":
                    parts.append({"type": "text", "text": val})
                    continue
                desc, meta = store.image_description(val), meta_text(store.image_meta(val))
                got = store.image_data_uri(val) if n_img < s.max_images else None
                if got and img_bytes + got[1] <= s.max_images_bytes:
                    parts.append({"type": "image_url", "image_url": {"url": got[0]}})
                    img_bytes += got[1]
                    n_img += 1
                    has_image = True
                    if meta or desc:
                        parts.append({"type": "text", "text": "(image " + "; ".join(x for x in (meta, desc) if x) + ")"})
                else:
                    parts.append({"type": "text", "text": "[image " + "; ".join(x for x in (meta, desc or "no description") if x) + "]"})
        content = parts if has_image else "\n".join(p["text"] for p in parts)
        item = {"id": str(u.records[0].id), "content": content, "score": round(1.0 / (1 + len(items)), 6)}
        if u.ts is not None:
            item["created_at"] = datetime.fromtimestamp(u.ts / 1000, tz=timezone.utc).isoformat()
        items.append(item)
    return items
