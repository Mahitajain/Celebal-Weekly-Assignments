# Routing Logic

## Why LLM-based classification instead of keyword matching

The brief explicitly calls for routing that is "intelligent -- not simple
keyword matching," and there's a concrete reason keyword matching breaks
down here: policy and data vocabulary overlap heavily in this domain.
"Patients," "discharge," "billing," and "surgery" are all words that show
up naturally in *both* a data question ("how many patients were
discharged yesterday") and a policy question ("what is our discharge
policy"). A keyword router would misfire constantly on exactly the kind
of question a hospital assistant needs to get right.

`app/orchestrator/classifier.py` sends the question (plus recent
conversation history) to the LLM with a system prompt that defines five
routes and asks for structured JSON back: `{route, confidence, reasoning}`.

## The five routes

| Route | Meaning | Example |
|---|---|---|
| `sql` | Answerable from the admissions database alone | "How many diabetic patients are over 60?" |
| `rag` | Answerable from policy documents alone | "What is the visitor policy for the ICU?" |
| `hybrid` | Genuinely needs both | "Which of Dr. Sharma's patients need pre-surgery clearance, and what does clearance require?" |
| `clarify` | Too ambiguous to route confidently | "Tell me about discharge." (data question about discharge dates, or policy question about discharge procedure?) |
| `unsupported` | Out of scope entirely | "What's the weather today?" |

## Why a `clarify` route, specifically

An earlier design only had `sql` / `rag` / `hybrid`, forcing every
ambiguous question into a guess. In practice, guessing wrong on an
ambiguous question is worse than asking -- a hospital staff member
who gets an answer about the wrong thing may not notice, whereas a
clarifying question is an honest signal the system doesn't yet have
enough to go on. `clarify` returns a short, concrete prompt showing one
example of each route, rather than a bare "please rephrase."

## Fallback: keyword heuristic (no LLM configured)

`_heuristic_route()` in the same file matches the question against two
small keyword lists (`_SQL_HINTS`, `_RAG_HINTS`). If both hit, route
`hybrid`; if only one hits, route that one; if neither hits, route
`clarify` rather than guessing blind. This is deliberately *not*
presented as the primary routing mechanism -- it exists so the project
runs end-to-end without an API key, and it is the reason the same
ambiguity problem described above is even more likely in fallback mode.
The documented, expected behavior is the LLM path.

## Hybrid execution

When the route is `hybrid`, LangGraph's conditional edge returns
`["sql_node", "rag_node"]`, which both execute as part of the same graph
step (not sequentially waiting on each other). Their results are merged
in a `synthesize` node -- see `docs/ARCHITECTURE.md` for how that merge
works and why it only calls the LLM when there's actually more than one
result to reconcile.

## Confidence scores

Every routing decision carries a `confidence` float (LLM self-reported,
0.5-0.55 flat for the heuristic fallback since it has no real basis for
finer-grained numbers). This is surfaced in the API response and shown
as a badge in the Streamlit UI ("🗄️ SQL Agent · 92% confidence") so a
user can see when the system is unsure, rather than presenting every
answer with uniform authority.
