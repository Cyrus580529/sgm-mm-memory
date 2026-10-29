"""Describe a sample of benchmark images with gpt-4o-mini at "high" and "low" detail, side by side, with token
counts, to choose CAPTION_DETAIL. Writes a Markdown report.

usage: (set -a; . ./.env; set +a; python scripts/caption_compare.py OUT.md IMAGE [IMAGE ...])
       e.g. a few MemLens needle images, MemEye images and MobileMem screenshots
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings  # noqa: E402
from app.models import SCHEMA_PROMPT  # noqa: E402
from app.store import compact_image  # noqa: E402

import base64  # noqa: E402
import httpx  # noqa: E402

PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6  # gpt-4o-mini, USD per token


async def describe(client, s, uri, detail):
    r = await client.post(f"{s.llm_base_url}/chat/completions", headers={"Authorization": f"Bearer {s.llm_api_key}"},
                          json={"model": s.llm_model, "temperature": 0, "max_tokens": 700,
                                "messages": [{"role": "user", "content": [
                                    {"type": "text", "text": SCHEMA_PROMPT.format(context="none")},
                                    {"type": "image_url", "image_url": {"url": uri, "detail": detail}}]}]})
    r.raise_for_status()
    j = r.json()
    u = j.get("usage", {})
    return j["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


async def main():
    out, paths = sys.argv[1], sys.argv[2:]
    s = Settings()
    rows, cost = [], {"high": 0.0, "low": 0.0}
    async with httpx.AsyncClient(timeout=180) as client:
        for p in paths:
            raw = open(p, "rb").read()
            mime, small = compact_image(raw, "image/png" if p.lower().endswith(".png") else "image/jpeg")
            uri = f"data:{mime};base64," + base64.b64encode(small).decode()
            res = {}
            for detail in ("high", "low"):
                text, pin, pout = await describe(client, s, uri, detail)
                res[detail] = (text, pin, pout)
                cost[detail] += pin * PRICE_IN + pout * PRICE_OUT
            rows.append((p, res))
            print(f"{os.path.basename(p)}: high {res['high'][1]} tok, low {res['low'][1]} tok", file=sys.stderr)
    with open(out, "w") as f:
        n = len(rows)
        f.write(f"# Caption comparison ({n} images)\n\nAverage cost per image: high ${cost['high'] / n:.5f}, "
                f"low ${cost['low'] / n:.5f}\n\n")
        for p, res in rows:
            f.write(f"## {p}\n\n| high ({res['high'][1]} prompt tokens) | low ({res['low'][1]} prompt tokens) |\n|---|---|\n")
            f.write(f"| {res['high'][0].replace(chr(10), '<br>')} | {res['low'][0].replace(chr(10), '<br>')} |\n\n")
    print(f"wrote {out}; per image: high ${cost['high'] / max(n, 1):.5f}, low ${cost['low'] / max(n, 1):.5f}")


if __name__ == "__main__":
    asyncio.run(main())
