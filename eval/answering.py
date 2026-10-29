"""End-to-end check: answer Mem-Gallery questions from Search results with a multimodal model, score token F1.

This only approximates the platform's Answer stage (its model and prompt are not public); it is meant for
comparing our own retrieval settings against each other.
"""
import asyncio
import collections
import json
import os
import re
import string

import httpx

SYS = ("You are an AI assistant evaluated on multimodal long-term conversational memory. Your responses must be "
       "concise, yet complete enough to accurately answer the questions. If multiple pieces of information about "
       "the same event appear in the conversation, always rely on the most recent information. Only answer using "
       "information contained within the retrieved memories. Do not hallucinate.")
FORMAT = {
    "VS": " Return the image_id of the image(s). If there are multiple images, sort them in ascending order and "
          "separate them by commas. Format example: “D2:IMG_003, D2:IMG_010, D10:IMG_002”.",
    "AR": " Only if the information about the question is not present in the conversation, reply with: "
          "“Not mentioned.”",
    "CD": " Please check whether this information conflicts with the conversation, and reply strictly with either "
          "“Yes.” or “No.”",
}


def normalize(s: str) -> list:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation or ch in ":_" else " " for ch in s)
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return [w.strip(":") for w in s.split() if w.strip(":")]


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not p or not g or not common:
        return float(p == g)
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def build_messages(qa: dict, query, items: list) -> list:
    content = [{"type": "text", "text": "The retrieved memory contents are as follows:\n"}]
    for it in items:
        c = it["content"]
        if isinstance(c, str):
            content.append({"type": "text", "text": c})
        else:
            for p in c:
                if p["type"] == "image_url":
                    content.append({"type": "image_url", "image_url": {"url": p["image_url"]["url"], "detail": "low"}})
                else:
                    content.append(p)
    q_text = qa["question"] + FORMAT.get(qa["point"], "")
    content.append({"type": "text", "text": f"\nThe current question is as follows:\n{q_text}\n"
                                            "Please only provide the content of the answer."})
    if isinstance(query, list):
        for p in query:
            if p["type"] == "image_url":
                content.append({"type": "text", "text": "Here is the attached image of the question:"})
                content.append({"type": "image_url", "image_url": {"url": p["image_url"]["url"], "detail": "low"}})
    return [{"role": "system", "content": SYS}, {"role": "user", "content": content}]


async def _run(pending: list, out_path: str) -> dict:
    base = os.environ.get("ANSWER_BASE_URL") or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    key = os.environ.get("ANSWER_API_KEY") or os.environ["LLM_API_KEY"]
    model = os.environ.get("ANSWER_MODEL", "gpt-4o-mini")
    sem = asyncio.Semaphore(int(os.environ.get("ANSWER_CONCURRENCY", "6")))
    scores = collections.defaultdict(list)
    async with httpx.AsyncClient(timeout=300) as client:
        async def one(qa, query, items):
            async with sem:
                for attempt in range(5):
                    r = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                                          json={"model": model, "temperature": 0, "max_tokens": 200,
                                                "messages": build_messages(qa, query, items)})
                    if r.status_code == 200:
                        break
                    await asyncio.sleep(2 ** attempt)
                r.raise_for_status()
                ans = r.json()["choices"][0]["message"]["content"].strip()
                usage = r.json().get("usage", {})
            s = f1(ans, str(qa["answer"]))
            scores[qa["point"]].append(s)
            scores["ALL"].append(s)
            return {"point": qa["point"], "question": qa["question"], "gold": qa["answer"], "pred": ans, "f1": s,
                    "prompt_tokens": usage.get("prompt_tokens")}
        rows = await asyncio.gather(*(one(*p) for p in pending))
    with open(out_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return scores


def answer_all(pending: list, out_path: str) -> dict:
    return asyncio.run(_run(pending, out_path))


# ---- generic helpers (used by the MemLens / MemEye / MobileMem replays) -------------------------------------
def items_to_parts(items: list, detail: str = "low") -> list:
    """Search result items -> chat content parts, in returned order (as the platform passes them to Answer)."""
    parts = []
    for i, it in enumerate(items, 1):
        parts.append({"type": "text", "text": f"\n[Memory {i}]"})
        c = it["content"]
        if isinstance(c, str):
            parts.append({"type": "text", "text": c})
            continue
        for p in c:
            if p["type"] == "image_url":
                parts.append({"type": "image_url", "image_url": {"url": p["image_url"]["url"], "detail": detail}})
            else:
                parts.append(p)
    return parts


async def chat_many(requests: list, model_env: str = "ANSWER") -> list:
    """requests: list of (messages, max_tokens). Returns reply texts ('' on failure). Endpoint from
    <model_env>_BASE_URL/_API_KEY/_MODEL, falling back to the LLM_* settings and gpt-4o-mini."""
    base = os.environ.get(f"{model_env}_BASE_URL") or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    key = os.environ.get(f"{model_env}_API_KEY") or os.environ["LLM_API_KEY"]
    model = os.environ.get(f"{model_env}_MODEL", "gpt-4o-mini")
    sem = asyncio.Semaphore(int(os.environ.get("ANSWER_CONCURRENCY", "6")))
    async with httpx.AsyncClient(timeout=300) as client:
        async def one(messages, max_tokens):
            async with sem:
                for attempt in range(5):
                    try:
                        r = await client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                                              json={"model": model, "temperature": 0, "max_tokens": max_tokens,
                                                    "messages": messages})
                        if r.status_code == 200:
                            return r.json()["choices"][0]["message"]["content"].strip()
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(2 ** attempt)
            return ""
        return await asyncio.gather(*(one(m, t) for m, t in requests))
