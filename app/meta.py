"""Capture time and place read from an image's own EXIF data (no model involved)."""
import io
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _geocoder():
    import reverse_geocoder
    return reverse_geocoder.RGeocoder(mode=1, verbose=False)


def _country(cc: str) -> str:
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=cc)
        return getattr(c, "common_name", None) or c.name
    except Exception:
        return cc


def place_name(lat: float, lon: float) -> str:
    try:
        g = _geocoder().query([(lat, lon)])[0]
    except Exception as e:
        log.warning("reverse geocoding failed: %s", e)
        return ""
    names = [g.get("name"), g.get("admin2"), g.get("admin1"), _country(g.get("cc", ""))]
    out = []
    for n in names:
        if n and n not in out:
            out.append(n)
    return ", ".join(out)


def _deg(v) -> float:
    d, m, s = (float(x) for x in v)
    return d + m / 60 + s / 3600


def image_meta(raw: bytes) -> dict:
    """{'taken': 'YYYY-MM-DD HH:MM', 'taken_ms': utc ms, 'place': str} with whatever the EXIF provides."""
    try:
        from PIL import Image
        ex = Image.open(io.BytesIO(raw)).getexif()
    except Exception:
        return {}
    out = {}
    sub = ex.get_ifd(0x8769)
    stamp = sub.get(0x9003) or ex.get(0x0132)  # DateTimeOriginal, else DateTime
    if isinstance(stamp, str):
        try:
            local = datetime.strptime(stamp.strip(), "%Y:%m:%d %H:%M:%S")
            off = sub.get(0x9011)  # OffsetTimeOriginal, e.g. "+01:00"
            tz = timezone.utc
            if isinstance(off, str) and len(off) == 6:
                sign = 1 if off[0] == "+" else -1
                tz = timezone(sign * timedelta(hours=int(off[1:3]), minutes=int(off[4:6])))
            out["taken"] = local.strftime("%Y-%m-%d %H:%M:%S")
            # local clock time drives date questions, so keep it as the "UTC" wall-clock value
            out["taken_ms"] = int(local.replace(tzinfo=timezone.utc).timestamp() * 1000)
            out["tz"] = str(tz)
        except ValueError:
            pass
    gps = ex.get_ifd(0x8825)
    try:
        lat, lon = _deg(gps[2]), _deg(gps[4])
        if gps.get(1) == "S":
            lat = -lat
        if gps.get(3) == "W":
            lon = -lon
        if abs(lat) > 1e-6 or abs(lon) > 1e-6:
            out["place"] = place_name(lat, lon)
    except Exception:
        pass
    return out


def meta_text(m: Optional[dict]) -> str:
    if not m:
        return ""
    bits = []
    if m.get("taken"):
        d = datetime.strptime(m["taken"][:16], "%Y-%m-%d %H:%M")
        bits.append(f"photo taken {m['taken']} ({d:%A, %B} {d.day}, {d.year})")
    if m.get("place"):
        bits.append(f"at {m['place']}")
    return " ".join(bits)
