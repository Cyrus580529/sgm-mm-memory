"""Runtime configuration, read once from environment variables."""
import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _host(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).hostname or ""


@dataclass(frozen=True)
class Settings:
    memory_api_key: str = _env("MEMORY_API_KEY")
    data_dir: str = _env("MEMORY_DATA_DIR", "artifacts")

    # gpt-4o-mini: image schema descriptions (Add) and query-image descriptions (Search).
    llm_base_url: str = _env("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = _env("LLM_API_KEY")
    llm_model: str = _env("LLM_MODEL", "gpt-4o-mini")
    # concurrent provider calls, per kind; through the gateway 16 parallel high-detail descriptions ran ~134/min
    # (8: ~59/min; 32: throttled)
    llm_concurrency: int = int(_env("LLM_CONCURRENCY", "16"))
    emb_concurrency: int = int(_env("EMB_CONCURRENCY", "16"))
    rerank_concurrency: int = int(_env("RERANK_CONCURRENCY", "8"))

    caption_detail: str = _env("CAPTION_DETAIL", "high")  # image detail for descriptions: small text needs "high"

    # text-embedding-v4. EMB_STYLE "dashscope" = native API (query/document text types), "openai" = compatible API.
    emb_base_url: str = _env("EMB_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
    emb_api_key: str = _env("EMB_API_KEY")
    emb_model: str = _env("EMB_MODEL", "text-embedding-v4")
    emb_dim: int = int(_env("EMB_DIM", "1024"))
    emb_style: str = _env("EMB_STYLE", "dashscope")
    emb_span_chars: int = int(_env("EMB_SPAN_CHARS", "6000"))  # long texts are embedded in spans, scored by max

    # Reranker (unrestricted by the Open-source Methods rules). RERANK_STYLE "dashscope" = DashScope native API (text
    # and image documents); "jina" = the common POST {base}/rerank API of OpenAI-style gateways (text only). The
    # embedding key is reused only for a reranker on the same host, so a key never goes to another provider.
    rerank_base_url: str = _env("RERANK_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
    rerank_style: str = _env("RERANK_STYLE", "dashscope")
    rerank_api_key: str = _env("RERANK_API_KEY") or (
        _env("EMB_API_KEY") if _host(_env("RERANK_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"))
        == _host(_env("EMB_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")) else "")
    rerank_text_model: str = _env("RERANK_TEXT_MODEL", "qwen3-rerank")
    rerank_vl_model: str = _env("RERANK_VL_MODEL", "qwen3-vl-rerank")
    rerank_enabled_flag: str = _env("RERANK", "1")
    rerank_candidates: int = int(_env("RERANK_CANDIDATES", "80"))
    rerank_weight: float = float(_env("RERANK_WEIGHT", "1.5"))
    # The image reranker only scores exchanges that carry images, so it votes only for images it judges relevant
    # (score >= this); otherwise image-bearing exchanges would gain on every query.
    rerank_vl_min: float = float(_env("RERANK_VL_MIN", "0.5"))

    # Query expansion with gpt-4o-mini: extra search queries, each a lower-weight vote.
    expand_n: int = int(_env("EXPAND_N", "0"))  # 0 disables
    # total weight of all expansion votes together (the original question's lexical + dense votes weigh 2.0),
    # split evenly over them, so a misguided rewrite cannot outvote the question itself
    expand_weight: float = float(_env("EXPAND_WEIGHT", "1.0"))

    # Retrieval.
    rrf_k: int = int(_env("RRF_K", "60"))
    seed_k: int = int(_env("SEED_K", "20"))          # top results that get neighbouring turns attached
    neighbour_window: int = int(_env("NEIGHBOUR_WINDOW", "1"))
    time_weight: float = float(_env("TIME_WEIGHT", "0"))
    recency_weight: float = float(_env("RECENCY_WEIGHT", "0"))  # present-state questions: newest relevant first
    recency_depth: int = int(_env("RECENCY_DEPTH", "200"))
    options_weight: float = float(_env("OPTIONS_WEIGHT", "0"))  # answer options (mostly distractors) as a vote
    # Result count: at most top_k; RESULT_LIMIT (0 = top_k) caps it lower for large memories. When the user's whole
    # memory fits, every exchange is returned (ranked ones first, the rest after, newest first).
    result_limit: int = int(_env("RESULT_LIMIT", "0"))
    return_all_if_fits: bool = _env("RETURN_ALL_IF_FITS", "1") == "1"  # RRF weight of the "dated inside the asked period" list
    # raw image bytes per Search response; base64 adds 4/3, so 18 MiB stays under the 30 MiB response cap
    max_images_bytes: int = int(_env("MAX_IMAGES_BYTES", str(18 * 1024 * 1024)))
    # Original images returned per Search: pixels only for the best-ranked image-bearing items (the rest carry their
    # text surrogate), downscaled to RETURN_IMAGE_SIDE. Under gpt-4o-mini's image pricing a 768 px image costs
    # ~25.5k of the Answer window's 117,760 input tokens, a 512 px one ~8.5k.
    max_images: int = int(_env("MAX_IMAGES", "4"))
    # Order of the returned set (membership is decided by relevance and the token budget first): "relevance",
    # "sandwich" (best at both ends), "chrono" (oldest first) or "session" (one item per session, oldest first inside).
    arrangement: str = _env("ARRANGEMENT", "relevance")
    image_match_bits: int = int(_env("IMAGE_MATCH_BITS", "6"))
    # Beyond the first ASSISTANT_TRIM_AFTER results (0 = never), assistant replies are cut to ASSISTANT_TRIM_CHARS:
    # the user's own words carry most facts, long generic replies mostly use up the Answer window.
    assistant_trim_after: int = int(_env("ASSISTANT_TRIM_AFTER", "20"))
    assistant_trim_chars: int = int(_env("ASSISTANT_TRIM_CHARS", "300"))  # dHash distance for "same picture" (0 = exact only)
    return_image_side: int = int(_env("RETURN_IMAGE_SIDE", "512"))  # 512 px = one tile (~8.5k tokens)
    # The platform's Answer keeps only the prefix of our results that fits 117,760 input tokens (128k window minus
    # 8,192 output and 2,048 safety). We budget for it ourselves so the cut falls where we choose: images give way
    # to their text surrogates first, then items that do not fit are skipped. 1,024 tokens for instructions, 15% for
    # tokenizer differences; images are priced as gpt-4o-mini prices them (the assumed Answer model).
    answer_window: int = int(_env("ANSWER_INPUT_TOKENS", "117760"))
    image_detail_low: bool = _env("ANSWER_IMAGE_DETAIL", "auto") == "low"  # how the Answer model is assumed to view images
    # in-memory search indexes, all users: ~22 KB per record, so 50,000 ~ 1.1 GB, well inside the unit's MemoryMax=4G
    # (300,000 would need ~6.4 GB); an evicted user's index rebuilds in well under a second
    index_cache_records: int = int(_env("INDEX_CACHE_RECORDS", "50000"))

    # Provider outages during Add: answer 503 (Retry-After seconds) up to ADD_MAX_DEFERS times per request, then
    # commit without descriptions/vectors and fill them in from the background repair loop.
    # 20 x Retry-After 30 s ~ 10 min of outage before a degraded commit (the platform retries an Add up to 32 times)
    add_max_defers: int = int(_env("ADD_MAX_DEFERS", "20"))
    # Memory bounds: Adds (up to 30 MiB bodies) and Search response assembly beyond these counts wait their turn
    # instead of being decoded side by side (64 concurrent 30 MiB Adds would need several GB).
    add_max_inflight: int = int(_env("ADD_MAX_INFLIGHT", "16"))
    render_max_inflight: int = int(_env("RENDER_MAX_INFLIGHT", "32"))
    retry_after: int = int(_env("RETRY_AFTER", "30"))
    repair_interval: float = float(_env("REPAIR_INTERVAL", "60"))

    @property
    def captions_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def dense_enabled(self) -> bool:
        return bool(self.emb_api_key)

    @property
    def rerank_enabled(self) -> bool:
        return bool(self.rerank_api_key) and self.rerank_enabled_flag == "1"

    @property
    def rerank_images(self) -> bool:
        """Image documents need a multimodal reranker on the DashScope API."""
        return self.rerank_enabled and self.rerank_style == "dashscope" and bool(self.rerank_vl_model)


settings = Settings()

REQUIRED_MODELS = {"llm": "gpt-4o-mini", "embedding": "text-embedding-v4"}


def check_competition_models(s: Settings) -> None:
    """The Open-source Methods division fixes the models; refuse to start with others."""
    if os.environ.get("ALLOW_NONCOMPETITION_MODELS") == "1":
        return
    if s.llm_model != REQUIRED_MODELS["llm"] or s.emb_model != REQUIRED_MODELS["embedding"]:
        raise RuntimeError(
            f"LLM_MODEL must be gpt-4o-mini and EMB_MODEL text-embedding-v4 (got {s.llm_model}, {s.emb_model})"
        )
