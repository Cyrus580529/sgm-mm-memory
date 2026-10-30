# SGM-MM-Memory

Schema-guided multimodal memory for the Agent Memory Leaderboard (AML) Cycle 2, Multimodal track,
Open-source Methods division. The service stores conversations with images and returns the original evidence for
a question; it never generates answers (the platform's Answer stage does).

**Status:** in development. The submission version and its fixed commit will be stated here before the official
evaluation.

## Method

One idea: **read images once, at write time, and deliver evidence within the Answer model's budget.** The platform's
Answer model reads our results inside a fixed 117,760-token window, and under gpt-4o-mini pricing a single image
costs 8,500–25,500 tokens. So the service turns pictures into text when they are written (paid once, outside the
window) and sends the reader verbatim evidence, best first, with pixels only where they are most likely to matter.

**Write (Add).** Every message is kept verbatim, images included. For each image the service also stores
- a structured description by `gpt-4o-mini` at high detail, in fixed fields: scene, objects (colours, counts,
  brands), people, spatial layout, verbatim on-image text, event, time; in the conversation's language, made once
  per image (cached by hash, versioned by prompt);
- capture time and place from the image's own EXIF data (GPS mapped to a place name with an offline GeoNames table);
- a perceptual hash (dHash), so the same picture sent again in another file is recognised.

**Retrieve (Search).** The unit is one exchange (the user message(s) of an Add and the replies after them).
BM25 (Latin words, CJK characters and bigrams) and `text-embedding-v4` dense retrieval are fused by reciprocal rank;
`qwen3-rerank` adds a vote over the top candidates; a picture in the question matches stored copies of the same
picture. The best 20 exchanges bring their neighbouring exchanges; a memory that fits in `top_k` is returned whole.

**Deliver.** Results stay in relevance order. Every image carries its text surrogate (capture metadata and
description); only the four best-ranked images also carry pixels, at 512 px (one tile, ~8.5k tokens). Beyond the
first 20 results long assistant replies are cut to 300 characters (the user's words are kept whole). Items are
packed against the Answer window with gpt-4o-mini's image pricing: an item that does not fit gives up its pixels, then
is skipped, so the platform's prefix cut never drops a better-ranked memory.

**What the ablations removed.** One-factor ablations against a minimal system, answered and graded end to end with
the benchmarks' official judges (MemLens, MemEye; paired per question), found no measurable gain from query
rewriting, date and present-state votes, or answer-option votes; these are off by default (`EXPAND_N`,
`TIME_WEIGHT`, `RECENCY_WEIGHT`, `OPTIONS_WEIGHT`). Chronological or per-session ordering of the returned set scored
lower than relevance order. Returning no pixels at all scored about the same as four images, i.e. the write-time
descriptions carry the visual evidence; four 512 px images are kept for fine visual detail.

**Models.** LLM components use only `gpt-4o-mini` and embeddings only `text-embedding-v4`, as the Open-source Methods
rules require; the reranker is unrestricted by the rules. Without model keys the service still runs (lexical only).

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
- Packing evidence for a fixed reader and keeping verbatim evidence first follow observations by AML cycle-1 entrant
  ActiveMemoryIndex (github.com/linxuhao/ActiveMemoryIndex); the write-time emphasis follows the "memory tax" view
  that writer quality bounds what any reader can use (arXiv:2610.29823).
- Offline reverse geocoding uses `reverse_geocoder` with GeoNames data (CC BY 4.0, geonames.org).

## License

MIT, see `LICENSE`.
