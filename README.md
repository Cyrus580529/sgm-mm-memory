# SGM-MM-Memory

Schema-guided multimodal memory for the Agent Memory Leaderboard (AML) Cycle 2, Multimodal track, Open-source Methods division.

**Status:** work in progress. Code, configuration and the fixed submission commit will be published here before the Cycle 2 materials deadline.

## Method (summary)

- **Add:** every text/image message is stored verbatim (original image kept). Each image is additionally described with `gpt-4o-mini` into fixed schema fields (scene, objects, people, visible text, event, time), cached by image hash.
- **Search:** hybrid retrieval over the schema text and dialogue (`text-embedding-v4` dense + BM25, reciprocal rank fusion), with neighbouring-turn context; results return interleaved text and the original image.
- Models: `text-embedding-v4` for embeddings and `gpt-4o-mini` for all LLM components, as required by the Open-source Methods rules. Search never generates answers.

## Prior work

The schema-guided representation follows Schema-Guided Memory (SGM) from ATM-Bench: J. Mei et al., *According to Me: Long-Term Personalized Referential Memory QA*, arXiv:2603.01990.

## License

MIT
