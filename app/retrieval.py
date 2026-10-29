"""Hybrid retrieval (BM25 + dense + exact image match, fused by RRF) and response assembly."""
import json
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
    """Records by their best-matching span. matrix: optional precomputed dense_matrix(records)."""
    owner, mat = matrix if matrix is not None else dense_matrix(records)
    if mat is None:
        return []
    best = np.full(len(records), -np.inf, dtype=np.float32)
    np.maximum.at(best, owner, mat @ qvec)
    order = np.argsort(-best)
    return [int(i) for i in order if np.isfinite(best[i])]


def dense_matrix(records: list) -> tuple:
    """(record index of each span row, stacked span vectors)."""
    owner, rows = [], []
    for i, r in enumerate(records):
        if r.vec is not None:
            owner.extend([i] * len(r.vec))
            rows.append(r.vec)
    return np.asarray(owner, dtype=np.int64), (np.concatenate(rows) if rows else None)


def spans(text: str, size: int) -> list:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


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
_ISO_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?\b")
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
    z = _ZH_TODAY.search(query)
    if z and z.group(1):
        try:
            anchor = date(int(z.group(1)), int(z.group(2)), int(z.group(3)))
            body = query[:z.start()] + " " + query[z.end():]
        except ValueError:
            anchor = None
    t = None if anchor else _TODAY_RE.search(query)
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
    zh_periods(body, anchor, preds, span)
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
    return preds


_ZH_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_ZH_DATE = re.compile(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(?:(\d{1,2})\s*[日号]|(初|上旬|中旬|下旬|底|末))?")
_ZH_YEAR = re.compile(r"(\d{4})\s*年(?!\s*\d{1,2}\s*月)")
_ZH_AGO = re.compile(r"(\d+|[一二两三四五六七八九十])\s*(天|周|个星期|星期|个月|年)(?:以)?前")
_ZH_TODAY = re.compile(r"今天是\s*(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")


def zh_periods(text: str, anchor: Optional[date], preds: list, span) -> None:
    """Chinese date expressions: 2025年5月20日 / 7月2号 / 5月初 / 2024年 and, with an anchor date,
    昨天 前天 上周 上个月 去年 N天前 ... (adds predicates in place)."""
    if not re.search(r"[\u4e00-\u9fff]", text):
        return
    for m in _ZH_DATE.finditer(text):
        mo = int(m.group(2))
        if not 1 <= mo <= 12:
            continue
        if m.group(1):
            y = int(m.group(1))
        elif anchor:
            y = anchor.year if mo <= anchor.month else anchor.year - 1
        else:
            y = None
        lo, hi, part = 1, 31, m.group(4)
        if m.group(3):
            lo = hi = int(m.group(3))
        elif part in ("初", "上旬"):
            hi = 10
        elif part == "中旬":
            lo, hi = 11, 20
        elif part in ("下旬", "底", "末"):
            lo = 21
        preds.append(lambda d, y=y, mo=mo, lo=lo, hi=hi: (y is None or d.year == y) and d.month == mo
                     and lo <= d.day <= hi)
    for m in _ZH_YEAR.finditer(text):
        y = int(m.group(1))
        preds.append(lambda d, y=y: d.year == y)
    if not anchor:
        return
    wk = anchor - timedelta(days=anchor.weekday())
    rel = {"昨天": (1, 1), "前天": (2, 2), "今天": (0, 0)}
    for w, (a, b) in rel.items():
        if w in text and not text.startswith("今天是"):
            span(anchor - timedelta(days=a), anchor - timedelta(days=b))
    if re.search("上周|上星期|上个星期|上礼拜", text):
        span(wk - timedelta(days=8), anchor - timedelta(days=1))
    if re.search("这周|本周|这个星期", text):
        span(wk, anchor)
    if "上个月" in text or "上月" in text:
        prev = anchor.replace(day=1) - timedelta(days=1)
        span(_month_span(prev.year, prev.month)[0], anchor - timedelta(days=1))
    if "这个月" in text or "本月" in text:
        span(anchor.replace(day=1), anchor)
    if "去年" in text:
        span(date(anchor.year - 1, 1, 1), date(anchor.year - 1, 12, 31))
    if "今年" in text:
        span(date(anchor.year, 1, 1), anchor)
    for m in _ZH_AGO.finditer(text):
        n = int(m.group(1)) if m.group(1).isdigit() else _ZH_NUM[m.group(1)]
        unit = m.group(2)
        days, slack = {"天": (n, 1), "周": (7 * n, 3), "个星期": (7 * n, 3), "星期": (7 * n, 3),
                       "个月": (30 * n, 10), "年": (365 * n, 60)}[unit]
        c = anchor - timedelta(days=days)
        span(c - timedelta(days=slack), c + timedelta(days=slack))


def strip_anchor(query: str) -> str:
    """The question without a leading "today is <date>" clause: the date steers time filtering, but its words
    (year, weekday, digits) would only add noise to lexical and dense matching."""
    z = _ZH_TODAY.search(query)
    if z and z.group(1):
        return (query[:z.start()] + re.sub(r"^\s*[。.,，;；:：]?", "", query[z.end():])).strip()
    t = _TODAY_RE.search(query)
    if not t:
        return query
    tail = query[t.start(1):t.start(1) + 40]
    m = _ISO_RE.search(tail) or _MONTH_RE.search(tail) or _DMY_RE.search(tail)
    if not m:
        return query
    end = t.start(1) + m.end()
    rest = re.match(r"\s*(\([A-Za-z]{3}\))?\s*(\d{1,2}:\d{2})?\s*[.,;:]?", query[end:])
    return (query[:t.start()] + query[end + (rest.end() if rest else 0):]).strip()


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


_CURRENT_RE = re.compile(r"\b(now|currently|current|these days|nowadays|lately|at the moment|at present|latest|"
                         r"most recent(?:ly)?|still|anymore|any more|go-to|these past)\b|现在|目前|如今|最近|当前|还在|"
                         r"仍然|依然|最新|这阵子|近来", re.I)


def recency_rank(query: str, items: list, base: list, depth: int) -> list:
    """For questions about the present state ("what do I currently ..."), the base ranking's top `depth`
    candidates ordered newest first: among relevant memories, the latest update is the one that counts."""
    if not _CURRENT_RE.search(query):
        return []
    top = base[:depth]
    return sorted(top, key=lambda i: -((getattr(items[i], "event_ts", None) or items[i].ts) or 0))


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


def arrival_order(records: list) -> dict:
    """Sort key per record: an Add is placed at its earliest message timestamp (an Add without timestamps at the
    previous Add's time, i.e. by arrival), and keeps its own message order. Timestamps are optional per message,
    so ordering by each message's own timestamp could split a question from its untimestamped reply."""
    by_req = {}
    for i, r in enumerate(records):
        by_req.setdefault(r.request_id, []).append(i)
    first = {req: min(records[i].id for i in ids) for req, ids in by_req.items()}
    req_time, last = {}, 0
    # Adds may arrive concurrently, so arrival is a weak order: the ordinal the platform puts at the end of
    # request_id (e.g. "...:chunk-12") comes first, arrival breaks ties
    for req in sorted(by_req, key=lambda q: (request_ordinal(q), first[q])):
        ts = [records[i].ts for i in by_req[req] if records[i].ts is not None]
        last = min(ts) if ts else last
        req_time[req] = (last, request_ordinal(req), first[req])
    return {i: (*req_time[r.request_id], r.msg_index) for i, r in enumerate(records)}


_ORDINAL = re.compile(r"(\d+)\D*$")


def request_ordinal(request_id: str) -> int:
    m = _ORDINAL.search(request_id)
    return int(m.group(1)) if m and len(m.group(1)) < 12 else 0


def build_units(records: list) -> tuple:
    """(units, unit index of each record). Sessions are ordered by timestamp, then arrival."""
    by_session = {}
    for i, r in enumerate(records):
        by_session.setdefault(r.session_id, []).append(i)
    units, unit_of = [], {}
    order = arrival_order(records)
    for sess, ids in by_session.items():
        ids.sort(key=lambda i: order[i])
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


def first_image(u: "Unit") -> str:
    return next(h for r in u.records for h in r.images)


def unit_text(u: "Unit", store: Store) -> str:
    """Plain-text view of an exchange (with image metadata and descriptions), for the text reranker."""
    lines = [fmt_time(u.ts)]
    for r in u.records:
        lines.append(f"{r.role}: {r.text}")
        for h in r.images:
            lines.append("[image] " + " ".join(x for x in (meta_text(store.image_meta(h)), store.image_description(h)) if x))
    return "\n".join(l for l in lines if l)


_CJK_CHAR = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]")


def est_tokens(text: str) -> int:
    """Rough token count, erring high: CJK about one token per character, other text about 3.6 characters."""
    cjk = len(_CJK_CHAR.findall(text))
    return cjk + int((len(text) - cjk) / 3.6) + 1


def item_tokens(item: dict, image_tokens: int) -> int:
    c = item["content"]
    if isinstance(c, str):
        return est_tokens(c) + 20
    return sum(image_tokens if p["type"] == "image_url" else est_tokens(p.get("text", "")) for p in c) + 20


def without_images(item: dict) -> dict:
    c = item["content"]
    if isinstance(c, str):
        return item
    return {**item, "content": "\n".join(p["text"] for p in c if p["type"] == "text")}


def answer_budget(s: Settings, query_text: str, options_text: str) -> int:
    return int((s.answer_window - 1024) / 1.15) - est_tokens(query_text) - est_tokens(options_text)


def render(units: list, order: list, store: Store, s: Settings, budget: Optional[int] = None) -> list:
    """Search response items in rank order. Original images are attached while the image allowance lasts (after
    that the image's description stands in). With a token budget, an item that does not fit is sent without its
    images if that fits, else skipped, so the platform's prefix cut never drops a better-ranked memory."""
    items, img_bytes, n_img = [], 0, 0
    used = 0
    for k in order:
        u: Unit = units[k]
        when = fmt_time(u.ts)
        item_bytes = item_imgs = 0  # this item's share of the image allowance, returned if its images are dropped
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
                    item_bytes += got[1]
                    item_imgs += 1
                    has_image = True
                    if meta or desc:
                        parts.append({"type": "text", "text": "(image " + "; ".join(x for x in (meta, desc) if x) + ")"})
                else:
                    parts.append({"type": "text", "text": "[image " + "; ".join(x for x in (meta, desc or "no description") if x) + "]"})
        content = parts if has_image else "\n".join(p["text"] for p in parts)
        item = {"id": str(u.records[0].id), "content": content}
        if u.ts is not None:
            item["created_at"] = datetime.fromtimestamp(u.ts / 1000, tz=timezone.utc).isoformat()
        if budget is not None:
            cost = item_tokens(item, s.image_tokens)
            if used + cost > budget and has_image:
                item = without_images(item)
                img_bytes, n_img = img_bytes - item_bytes, n_img - item_imgs
                cost = item_tokens(item, s.image_tokens)
            if used + cost > budget:
                continue
            used += cost
        items.append(item)
    for i, it in enumerate(items):  # scores follow the final order
        it["score"] = round(1.0 / (1 + i), 6)
    return items


RESPONSE_LIMIT = 29 * 1024 * 1024  # the contract allows 30 MiB per Search response


def fit_response(items: list, limit: int = RESPONSE_LIMIT) -> list:
    """Keep the serialised response under the size limit: drop original images from the lowest-ranked items first
    (their descriptions stay), and only if text alone is too large, drop the lowest-ranked items."""
    size = lambda: len(json.dumps({"data": items}, ensure_ascii=False).encode())
    # cheap upper bound first (strings dominate; CJK text is at most 3 bytes per character in UTF-8)
    rough = sum(len(p.get("text", "")) * 3 + len(p.get("image_url", {}).get("url", "")) + 64
                for it in items for p in (it["content"] if isinstance(it["content"], list)
                                          else [{"text": it["content"]}])) + 128 * len(items)
    if rough <= limit or size() <= limit:
        return items
    for it in reversed(items):
        if isinstance(it["content"], list) and any(p["type"] == "image_url" for p in it["content"]):
            parts = [p if p["type"] != "image_url" else {"type": "text", "text": "[image omitted for size]"}
                     for p in it["content"]]
            it["content"] = "\n".join(p["text"] for p in parts)
            if size() <= limit:
                return items
    while items and size() > limit:
        items.pop()
    return items
