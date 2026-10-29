"""Runtime configuration, read once from environment variables."""
import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    memory_api_key: str = _env("MEMORY_API_KEY")
    data_dir: str = _env("MEMORY_DATA_DIR", "artifacts")

    # gpt-4o-mini: image schema descriptions (Add) and query-image descriptions (Search).
    llm_base_url: str = _env("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = _env("LLM_API_KEY")
    llm_model: str = _env("LLM_MODEL", "gpt-4o-mini")
    llm_concurrency: int = int(_env("LLM_CONCURRENCY", "8"))

    # text-embedding-v4 (DashScope OpenAI-compatible endpoint).
    emb_base_url: str = _env("EMB_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    emb_api_key: str = _env("EMB_API_KEY")
    emb_model: str = _env("EMB_MODEL", "text-embedding-v4")
    emb_dim: int = int(_env("EMB_DIM", "1024"))

    # Retrieval.
    rrf_k: int = int(_env("RRF_K", "60"))
    seed_k: int = int(_env("SEED_K", "20"))          # top results that get neighbouring turns attached
    neighbour_window: int = int(_env("NEIGHBOUR_WINDOW", "1"))
    time_weight: float = float(_env("TIME_WEIGHT", "1.0"))  # RRF weight of the "dated inside the asked period" list
    # raw image bytes per Search response; base64 adds 4/3, so 18 MiB stays under the 30 MiB response cap
    max_images_bytes: int = int(_env("MAX_IMAGES_BYTES", str(18 * 1024 * 1024)))
    max_images: int = int(_env("MAX_IMAGES", "40"))

    @property
    def captions_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def dense_enabled(self) -> bool:
        return bool(self.emb_api_key)


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
