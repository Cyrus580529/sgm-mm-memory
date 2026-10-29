# SGM-MM-Memory

Schema-guided multimodal memory for the Agent Memory Leaderboard (AML) Cycle 2, Multimodal track,
Open-source Methods division. The service stores conversations with images and returns the original evidence for
a question; it never generates answers (the platform's Answer stage does).

**Status:** in development. The submission version and its fixed commit will be stated here before the official
evaluation.

## Method

**Write (Add).** Every message is kept verbatim, images included. For each image the service also stores
- a structured description by `gpt-4o-mini` in fixed fields (scene, objects, people, verbatim on-image text, event,
  time), written in the conversation's language and cached by image hash;
- capture time and place from the image's own EXIF data (GPS mapped to a place name with an offline GeoNames table);
- a compact copy (long side ≤ 1024 px) used in responses, so that many images fit within the response limit.

**Retrieve (Search).** The unit of retrieval is one exchange: the user message(s) of an Add and the replies after
them. Several signals each produce a ranking of exchanges, merged by reciprocal rank fusion:
- BM25 over text, image descriptions and dates (Latin words plus CJK characters and bigrams);
- dense retrieval with `text-embedding-v4` (query/document types; long texts embedded in spans, best span counts);
- rewrites of the question by `gpt-4o-mini`, together weighing less than the question itself;
- dates named in the question (absolute, relative to a stated "today", English and Chinese) and, for questions about
  the present state ("now", "currently", 现在 ...), the newest of the relevant candidates;
- exact matches of a query image;
- rerankers `qwen3-rerank` (text) and `qwen3-vl-rerank` (images; votes only for images it judges relevant).

The top exchanges are followed by their neighbouring exchanges in time. When a user's whole memory fits in `top_k`,
all of it is returned (ranked exchanges first). Each item carries its date, the original text, the image and its
description.

**Models.** LLM components use only `gpt-4o-mini` and embeddings only `text-embedding-v4`, as the Open-source Methods
rules require; rerankers are unrestricted by the rules. Without model keys the service still runs (BM25 and the
non-model signals only).

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | unauthenticated |
| POST | `/v1/memories/add` | synchronous: 200 only after the write is persisted and searchable; idempotent per `request_id` (409 for the same id with different content) |
| POST | `/v1/memories/search` | at most `top_k` items in relevance order; images as inline base64 data URIs |

Auth: `Authorization: Token <key>`, `Authorization: Bearer <key>` or `X-Api-Key: <key>`.

## Run

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env        # service key, model endpoints and keys (never commit .env)
HOST=127.0.0.1 PORT=8000 scripts/run.sh
```

All settings are environment variables, listed with their defaults in `app/config.py`.

**Deployment (Ubuntu).** `sudo deploy/install.sh` from a checkout installs the service under systemd (restart on
failure) behind Caddy (automatic TLS); then fill in `/opt/sgm-mm-memory/.env` and the domain in
`/etc/caddy/Caddyfile`. Without a domain, `<ip-with-dashes>.sslip.io` resolves to the server and works with Caddy.

## Robustness

- Image decoding, EXIF, indexing and response assembly run in a worker pool; SQLite uses one connection per thread.
- If a model provider is unavailable during Add, nothing is committed and the service answers 503 with
  `Retry-After`; work already done is cached, so the platform's retry is cheap. If the outage outlasts
  `ADD_MAX_DEFERS` retries of the same request, the Add is committed without the missing descriptions/vectors and a
  background loop fills them in once the provider is back (the user's index is then rebuilt). Inputs a provider
  refuses are stored without a description. During Search any failing model call only removes its signal.
- The platform's Answer keeps only the prefix of results that fits its 117,760 input tokens. Results are
  assembled against that budget (with a margin for tokenizer differences): an item that does not fit gives up its
  images for their descriptions, and is skipped if it still does not fit, so the cut never drops a better-ranked
  memory. `IMAGE_TOKENS` sets the per-image estimate.
- Search never fails a question on an odd query part (the platform does not retry a 4xx on Search): unusable parts
  are skipped. Adds that arrive concurrently are ordered by timestamp, then by the ordinal at the end of
  `request_id`, then by arrival.
- Memory is bounded: at most `ADD_MAX_INFLIGHT` Adds are decoded at once and `RENDER_MAX_INFLIGHT` responses
  assembled at once; others wait (64 concurrent 18 MiB Adds peaked under 1 GB).
- Provider error bodies are not logged (they may echo evaluation content); evaluation data should be deleted
  within 30 days of a run (`scripts/purge.py`).
- Search responses are kept under the 30 MiB limit: original images are dropped from the lowest-ranked items first
  (their descriptions stay). Out-of-range or malformed timestamps are treated as absent; images over 64 megapixels
  are stored as sent without decoding.
- Per-user indexes are cached in memory (bounded by total records, least recently used evicted) and rebuilt once
  when new records arrive.
- Load-tested at the contract's concurrency (64 concurrent Adds, 256 concurrent Searches) without errors.

## Checks and evaluation

| Command | What it does |
|---|---|
| `python -m pytest` | unit and API tests (no network) |
| `python scripts/provider_check.py` | checks the configured model endpoints and suggests `.env` settings |
| `python scripts/caption_compare.py out.md img...` | image descriptions at high vs low detail, with cost |
| `GET /stats` (authenticated) | counters for a running instance: Adds, Searches, deferrals, provider failures, repair queue, disk |
| `python scripts/purge.py` | deletes stored evaluation data after a run |
| `python scripts/contract_check.py <url>` | checks a running instance against the Add/Search contract |
| `python scripts/load_test.py <url> <Mem-Gallery dir> --copies 4 --add-conc 64 --search-conc 256` | concurrent load with response validation |
| `python -m eval.replay_memlens <dir> --split 256k [--answer N]` | MemLens: evidence-session recall; with `--answer`, answers graded by the official MemLens judge prompts |
| `python -m eval.replay_memgallery <MemEye dir> --grader memeye [--answer N]` | MemEye (and Mem-Gallery): evidence recall; MCQ letter match / official LLM judge |
| `python -m eval.replay_mobilemem <dir> [--answer N]` | MobileMem-Omni: evidence recall; official MobileMem judge |
| `python -m eval.replay_atm <dir>` | ATM-Bench: evidence recall |

The replays write each benchmark through the Add API and query it through Search, in process. Answering uses an
OpenAI-compatible model (`ANSWER_*`, default `gpt-4o-mini`) and grading `JUDGE_*`; official judge prompt texts are
downloaded from the benchmarks' repositories at run time. These runs compare versions of this system against each
other; they are not official scores.

## Prior work and attribution

- Structured image descriptions follow Schema-Guided Memory (SGM) from ATM-Bench: J. Mei et al., *According to Me:
  Long-Term Personalized Referential Memory QA*, arXiv:2603.01990.
- BM25 (Robertson et al.) and reciprocal rank fusion (Cormack, Clarke and Buettcher) are standard methods.
- Verification of image candidates by a reranker is in the spirit of GraphMemix's node verifier (arXiv:2608.26983).
- Offline reverse geocoding uses `reverse_geocoder` with GeoNames data (CC BY 4.0, geonames.org).

## License

MIT, see `LICENSE`.
