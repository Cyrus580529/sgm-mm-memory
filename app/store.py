"""SQLite persistence: messages, Add idempotency, image files, schema descriptions, embeddings."""
import base64
import hashlib
import io
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS adds (
    request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL, payload_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL, session_id TEXT NOT NULL, request_id TEXT NOT NULL,
    msg_index INTEGER NOT NULL, role TEXT NOT NULL, ts INTEGER,
    event_ts INTEGER,            -- when the content happened: photo capture time if known, else ts
    text TEXT NOT NULL,          -- concatenated text parts
    images TEXT NOT NULL,        -- JSON list of image hashes, in content order
    parts TEXT NOT NULL,         -- JSON list of ["text", str] / ["image", hash], original order
    index_text TEXT NOT NULL,    -- text used for retrieval (text + image schema descriptions)
    vec BLOB
);
CREATE INDEX IF NOT EXISTS records_user ON records(user_id);
CREATE TABLE IF NOT EXISTS images (
    hash TEXT PRIMARY KEY, mime TEXT NOT NULL, path TEXT NOT NULL, description TEXT, meta TEXT
);
CREATE TABLE IF NOT EXISTS emb_cache (key TEXT PRIMARY KEY, vec BLOB NOT NULL);
"""


def compact_image(data: bytes, mime: str, max_side: int = 1024) -> tuple:
    """(mime, bytes) of a downscaled JPEG copy; the original if it is already small or cannot be decoded."""
    try:
        from PIL import Image, ImageOps
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im)
        if max(im.size) <= max_side and len(data) <= 300_000:
            return mime, data
        im.thumbnail((max_side, max_side))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return "image/jpeg", buf.getvalue()
    except Exception:
        return mime, data


@dataclass
class Record:
    id: int
    session_id: str
    request_id: str
    msg_index: int
    role: str
    ts: Optional[int]
    event_ts: Optional[int]
    text: str
    images: list
    parts: list
    index_text: str
    vec: Optional[np.ndarray]


class Store:
    def __init__(self, data_dir: str):
        os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
        self.data_dir = data_dir
        self.db = sqlite3.connect(os.path.join(data_dir, "memory.db"), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    # ---- Add idempotency -------------------------------------------------
    def add_status(self, request_id: str, payload_hash: str) -> str:
        """'new', 'duplicate' (same id, same payload) or 'conflict' (same id, different payload)."""
        row = self.db.execute("SELECT payload_hash FROM adds WHERE request_id=?", (request_id,)).fetchone()
        if row is None:
            return "new"
        return "duplicate" if row[0] == payload_hash else "conflict"

    # ---- images ----------------------------------------------------------
    def put_image(self, data: bytes, mime: str) -> str:
        """Keep the original and a compact copy (long side <= 1024 px JPEG) used for descriptions and Search
        responses, so that many images fit in one response."""
        h = hashlib.sha256(data).hexdigest()
        with self.lock:
            if self.db.execute("SELECT 1 FROM images WHERE hash=?", (h,)).fetchone() is None:
                orig = os.path.join(self.data_dir, "images", h + ".orig")
                with open(orig, "wb") as f:
                    f.write(data)
                small_mime, small = compact_image(data, mime)
                path = os.path.join(self.data_dir, "images", h)
                with open(path, "wb") as f:
                    f.write(small)
                from .meta import image_meta
                self.db.execute("INSERT INTO images(hash, mime, path, meta) VALUES (?,?,?,?)",
                                (h, small_mime, path, json.dumps(image_meta(data))))
                self.db.commit()
        return h

    def image_description(self, h: str) -> Optional[str]:
        row = self.db.execute("SELECT description FROM images WHERE hash=?", (h,)).fetchone()
        return row[0] if row else None

    def set_image_description(self, h: str, description: str) -> None:
        with self.lock:
            self.db.execute("UPDATE images SET description=? WHERE hash=?", (description, h))
            self.db.commit()

    def image_data_uri(self, h: str) -> Optional[tuple]:
        row = self.db.execute("SELECT mime, path FROM images WHERE hash=?", (h,)).fetchone()
        if row is None:
            return None
        with open(row[1], "rb") as f:
            raw = f.read()
        return f"data:{row[0]};base64," + base64.b64encode(raw).decode(), len(raw)

    # ---- records ---------------------------------------------------------
    def commit_add(self, request_id: str, user_id: str, session_id: str, payload_hash: str, rows: list) -> None:
        """Write all messages of one Add plus its idempotency row in a single transaction."""
        with self.lock:
            cur = self.db.cursor()
            cur.execute("BEGIN")
            for r in rows:
                cur.execute(
                    "INSERT INTO records(user_id, session_id, request_id, msg_index, role, ts, event_ts, text, images,"
                    " parts, index_text, vec) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (user_id, session_id, request_id, r["msg_index"], r["role"], r["ts"], r["event_ts"], r["text"],
                     json.dumps(r["images"]), json.dumps(r["parts"], ensure_ascii=False), r["index_text"],
                     None if r["vec"] is None else np.asarray(r["vec"], dtype=np.float32).tobytes()),
                )
            cur.execute("INSERT INTO adds(request_id, user_id, session_id, payload_hash) VALUES (?,?,?,?)",
                        (request_id, user_id, session_id, payload_hash))
            self.db.commit()

    def user_records(self, user_id: str) -> list:
        rows = self.db.execute(
            "SELECT id, session_id, request_id, msg_index, role, ts, event_ts, text, images, parts, index_text, vec"
            " FROM records WHERE user_id=? ORDER BY id", (user_id,)).fetchall()
        return [Record(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], json.loads(r[8]), json.loads(r[9]), r[10],
                       None if r[11] is None else np.frombuffer(r[11], dtype=np.float32)) for r in rows]

    def user_count(self, user_id: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM records WHERE user_id=?", (user_id,)).fetchone()[0]

    # ---- image metadata and embedding cache ------------------------------
    def image_meta(self, h: str) -> Optional[dict]:
        row = self.db.execute("SELECT meta FROM images WHERE hash=?", (h,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def set_image_meta(self, h: str, meta: dict) -> None:
        with self.lock:
            self.db.execute("UPDATE images SET meta=? WHERE hash=?", (json.dumps(meta), h))
            self.db.commit()

    def cached_vecs(self, keys: list) -> dict:
        out = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            q = "SELECT key, vec FROM emb_cache WHERE key IN (%s)" % ",".join("?" * len(chunk))
            out.update({k: np.frombuffer(v, dtype=np.float32) for k, v in self.db.execute(q, chunk)})
        return out

    def put_vecs(self, items: dict) -> None:
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO emb_cache(key, vec) VALUES (?,?)",
                                [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()])
            self.db.commit()
