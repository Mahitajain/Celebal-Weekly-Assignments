# Design Decisions, Trade-offs, and Future Improvements

## Key design decisions

| Decision | Alternative considered | Why this choice |
|---|---|---|
| LangGraph `StateGraph` for orchestration | Hand-rolled `if/elif` router | The brief explicitly asks for a real multi-agent graph with shared state, not a dispatcher function. LangGraph gives conditional edges (clean hybrid fan-out), a typed shared state, and a structure that's legible as an actual graph rather than nested conditionals that grow unreadable as routes are added. |
| Admission-grain fact table, no synthetic patient entity | Invent a `patient_id` by grouping on name+age+gender | Inventing an identity the source data doesn't support would silently misrepresent data quality. Being honest about the grain (one row = one admission) is more defensible in an interview than a schema that looks more "normalized" but encodes a false assumption. |
| sqlglot AST validation over regex/keyword blocklists | Regex checks for `DROP`, `DELETE`, etc. | Regex is trivially defeated by comments, string literals containing the keyword, or case variation. Parsing into an AST and checking node *types* is the difference between "looks safe" and "is safe." |
| Rewriting a `LIMIT` into the query rather than trusting the model | Prompt instruction only ("please add LIMIT 200") | Prompts are guidance, not guarantees. The validator enforces the cap structurally so an unbounded scan can't reach the client regardless of what the LLM generated. |
| TF-IDF as the default, tested embedding backend; sentence-transformers as the documented production path | Ship only sentence-transformers | This build environment couldn't install `torch` (disk-constrained), and shipping a project that only "works" with a multi-GB dependency someone may not be able to install either would undermine "runs with minimal setup." TF-IDF is not a toy fallback here -- it's a legitimate choice for this specific corpus (small, vocabulary-stable), documented as such, with the semantic-embedding upgrade path fully implemented and one config flag away. |
| Section-aware chunking over fixed-size windows | Fixed 500-token sliding window | The source documents are already organized into short, numbered, self-contained rules. Chunking to match that structure both preserves meaning (a rule stays with its qualifying context) and enables honest section-level citations, which the brief asks for directly. |
| FAISS `IndexFlatIP` (exact search) | FAISS `IndexIVFFlat` / `IndexHNSWFlat` (approximate) or ChromaDB | At a few thousand chunks, exact search is fast enough that approximate search would trade real complexity for an unmeasurable latency win. Documented as a scale-aware choice with a one-line upgrade path, not an oversight. |
| Deterministic fallback behavior for every agent when no LLM is configured | Require an API key to run at all | A portfolio project someone else clones and runs should demonstrate its architecture even without a paid API key on hand. Every fallback is explicitly labeled as degraded mode in both the API response (`used_llm: false`) and the UI, so it's never mistaken for the primary code path. |
| In-memory conversation memory + TTL cache | Redis-backed from the start | Right-sized for a single-process portfolio deployment (zero extra infrastructure to run locally); both interfaces are already shaped so a Redis swap wouldn't touch calling code. |
| SQLite over Postgres | Postgres via docker-compose | Keeps "clone and run" to `pip install -r requirements.txt`. The schema and query patterns are portable; `docs/DATABASE_SCHEMA.md` documents the exact one-line change to move to Postgres. |

## Known limitations

- **TF-IDF embeddings won't catch true synonyms** the corpus never used
  (e.g. a query using clinical jargon the documents phrase informally, or
  vice versa). This is the single biggest quality gap versus the
  sentence-transformers path, and it's the reason that path is fully
  implemented rather than left as a TODO.
- **The rule-based SQL/routing fallbacks are intentionally narrow.** They
  cover the example questions in the brief and close variants, not
  open-ended natural language. They exist to keep the project
  runnable without an API key, not as the intended user experience.
- **Conversation memory is process-local.** Restarting the API process
  (or running multiple replicas without sticky sessions) loses history.
  Acceptable for a single-instance deployment; documented as the first
  thing to change for horizontal scaling.
- **No authentication is enabled by default.** `API_KEY` in `.env` wires
  up a simple header check point (see `app/config.py`), but it isn't
  enforced by default so the project is easy to try locally. A real
  deployment handling actual patient data would need this on, plus a
  real identity provider, not a shared static key.

## Future improvements

- Swap the TF-IDF default for sentence-transformers once a deployment
  target with more disk/GPU headroom is available; no other code changes
  needed.
- Add role-based access control (clinician vs. billing vs. admin) that
  restricts which columns the SQL agent is allowed to select -- the
  validator's table whitelist already provides the mechanism to extend
  into a column-level whitelist per role.
- Stream the LLM's response token-by-token to the frontend (FastAPI
  `StreamingResponse` + a Streamlit token-by-token render) instead of
  waiting for the full completion, for a more responsive feel on longer
  synthesis answers.
- Add a lightweight eval harness: a fixed set of question/expected-route
  pairs run against the classifier on every change, to catch routing
  regressions the way `tests/test_orchestrator.py` catches them today
  but at larger scale.
- Real patient-matching (MPI) logic if this were ever connected to a
  data source with a genuine patient identifier, replacing the
  admission-grain-only model documented in `docs/DATABASE_SCHEMA.md`.
