# SGM-MM-Memory

Schema-guided multimodal memory for the Agent Memory Leaderboard (AML) Cycle 2, Multimodal track, Open-source Methods division.

**Status:** v1.0 in development. The fixed submission commit will be declared before the Cycle 2 materials deadline.

## Method (summary)

- **Add:** every text/image message is stored verbatim (original image kept). Each image is additionally described with `gpt-4o-mini` into fixed schema fields (scene, objects, people, visible text, event, time), cached by image hash.
- **Search:** hybrid retrieval over the schema text and dialogue (`text-embedding-v4` dense + BM25, reciprocal rank fusion), with neighbouring-turn context; results return interleaved text and the original image.
- Models: `text-embedding-v4` for embeddings and `gpt-4o-mini` for all LLM components, as required by the Open-source Methods rules. Search never generates answers.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | unauthenticated |
| POST | `/v1/memories/add` | synchronous; 200 only after the write is persisted and searchable; idempotent per `request_id` (409 on same id with different content) |
| POST | `/v1/memories/search` | returns at most `top_k` items in relevance order; image parts are inline base64 data URIs |

Auth: `Authorization: Token <key>`, `Authorization: Bearer <key>` or `X-Api-Key: <key>`.

## Run

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env   # fill in keys
HOST=0.0.0.0 PORT=8000 scripts/run.sh
```

Put an HTTPS reverse proxy (e.g. Caddy) in front of the port. Without `LLM_API_KEY` / `EMB_API_KEY` the service still runs, using BM25 only; the declared submission configuration uses both.

Tests: `uv pip install --python .venv/bin/python -r requirements-dev.txt && .venv/bin/python -m pytest`.

## Prior work

The schema-guided representation follows Schema-Guided Memory (SGM) from ATM-Bench: J. Mei et al., *According to Me: Long-Term Personalized Referential Memory QA*, arXiv:2603.01990.

## License

MIT
