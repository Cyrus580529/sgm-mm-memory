"""Check the model endpoints configured in .env before running anything else.

Tests, with tiny inputs: gpt-4o-mini with an image, text-embedding-v4 through both the DashScope-native and the
OpenAI-compatible API, qwen3-rerank and qwen3-vl-rerank. Prints what works, latency, the model id the provider
reports, and which .env settings to use. Never prints keys.

usage: (set -a; . ./.env; set +a; python scripts/provider_check.py)
"""
import base64
import io
import os
import time

import httpx


def tiny_png():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (160, 60), "white")
    ImageDraw.Draw(im).text((10, 20), "GARAGE 7", fill="black")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def call(url, key, body):
    t = time.time()
    try:
        r = httpx.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60)
        return r.status_code, (r.json() if "json" in r.headers.get("content-type", "") else r.text[:200]), time.time() - t
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", time.time() - t


def report(name, ok, dt, info=""):
    print(f"[{'OK  ' if ok else 'FAIL'}] {name:38s} {dt:5.1f}s  {info}")


def main():
    env = os.environ.get
    img = tiny_png()

    llm_base, llm_key = env("LLM_BASE_URL", "https://api.openai.com/v1"), env("LLM_API_KEY")
    if not llm_key:
        print("LLM_API_KEY not set")
    else:
        st, out, dt = call(f"{llm_base}/chat/completions", llm_key, {
            "model": env("LLM_MODEL", "gpt-4o-mini"), "temperature": 0, "max_tokens": 20,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Read the text in this image."},
                                                       {"type": "image_url", "image_url": {"url": img, "detail": "high"}}]}]})
        ok = st == 200
        info = (f"model={out.get('model')} reply={out['choices'][0]['message']['content']!r} "
                f"prompt_tokens={out.get('usage', {}).get('prompt_tokens')}") if ok else f"{st} {str(out)[:160]}"
        report("gpt-4o-mini, image input", ok, dt, info)

    emb_key = env("EMB_API_KEY")
    if not emb_key:
        print("EMB_API_KEY not set")
    else:
        base = env("EMB_BASE_URL", "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
        roots = {base, "https://dashscope.aliyuncs.com/api/v1"}
        for root in sorted(roots):
            st, out, dt = call(f"{root}/services/embeddings/text-embedding/text-embedding", emb_key, {
                "model": "text-embedding-v4", "input": {"texts": ["hello"]},
                "parameters": {"text_type": "query", "dimension": 1024}})
            ok = st == 200 and "output" in out
            report(f"embedding, native ({root[:28]}...)", ok, dt,
                   f"dim={len(out['output']['embeddings'][0]['embedding'])}" if ok else f"{st} {str(out)[:120]}")
            if ok:
                print(f"       -> EMB_STYLE=dashscope EMB_BASE_URL={root}")
        for root in sorted({base, "https://dashscope.aliyuncs.com/compatible-mode/v1"}):
            st, out, dt = call(f"{root}/embeddings", emb_key, {"model": "text-embedding-v4", "input": ["hello"],
                                                              "dimensions": 1024, "encoding_format": "float"})
            ok = st == 200 and "data" in out
            report(f"embedding, compatible ({root[:24]}...)", ok, dt,
                   f"dim={len(out['data'][0]['embedding'])}" if ok else f"{st} {str(out)[:120]}")
            if ok:
                print(f"       -> EMB_STYLE=openai EMB_BASE_URL={root}")

    rk_key = env("RERANK_API_KEY") or emb_key
    if rk_key:
        root = env("RERANK_BASE_URL", "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
        url = f"{root}/services/rerank/text-rerank/text-rerank"
        st, out, dt = call(url, rk_key, {"model": env("RERANK_TEXT_MODEL", "qwen3-rerank"),
                                         "input": {"query": "where is the bike?",
                                                   "documents": ["The bike is in garage 7.", "I like tea."]},
                                         "parameters": {"top_n": 2}})
        ok = st == 200 and "output" in out
        report("qwen3-rerank", ok, dt, str(out["output"]["results"])[:120] if ok else f"{st} {str(out)[:140]}")
        st, out, dt = call(url, rk_key, {"model": env("RERANK_VL_MODEL", "qwen3-vl-rerank"),
                                         "input": {"query": "a sign saying garage 7",
                                                   "documents": [{"image": img}, {"text": "a photo of a cat"}]},
                                         "parameters": {"top_n": 2}})
        ok = st == 200 and "output" in out
        report("qwen3-vl-rerank, base64 image", ok, dt, str(out["output"]["results"])[:120] if ok else f"{st} {str(out)[:140]}")
        if not ok:
            print("       -> if reranking is unavailable, set RERANK=0 (the other signals still work)")


if __name__ == "__main__":
    main()
