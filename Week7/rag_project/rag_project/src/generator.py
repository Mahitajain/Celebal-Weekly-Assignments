"""
generator.py
------------
Stage 7 of the RAG pipeline: Answer Generation.

Takes the user's question plus the retrieved context chunks and produces
a final grounded answer. Three backends are supported:

  - "anthropic": Claude via the Anthropic API (recommended). Needs
    ANTHROPIC_API_KEY set in the environment.
  - "openai": GPT via the OpenAI API. Needs OPENAI_API_KEY.
  - "extractive": no API key / internet required. A lightweight local
    fallback that assembles an answer directly from the highest-scoring
    sentences in the retrieved chunks. Lower quality than an LLM, but
    keeps the whole pipeline runnable completely offline for testing.

The prompt instructs the model to answer ONLY from the provided context
and to say so explicitly if the answer isn't contained in it -- this is
what keeps RAG answers grounded instead of hallucinated.
"""

import os
import re

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context from the user's documents. If the answer is not "
    "contained in the context, say you don't have enough information in "
    "the document to answer, rather than guessing. Be concise and cite "
    "which part of the context you used when helpful."
)


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return (
        f"Context from the document:\n\n{context_block}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        f"Answer using only the context above:"
    )


class Generator:
    def __init__(self, backend: str = "extractive", model: str | None = None):
        self.backend = backend
        self.model = model

        if backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            print("[generator] ANTHROPIC_API_KEY not set, falling back to extractive mode.")
            self.backend = "extractive"
        if backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print("[generator] OPENAI_API_KEY not set, falling back to extractive mode.")
            self.backend = "extractive"

    def generate(self, question: str, context_chunks: list[str]) -> str:
        if self.backend == "anthropic":
            return self._generate_anthropic(question, context_chunks)
        elif self.backend == "openai":
            return self._generate_openai(question, context_chunks)
        else:
            return self._generate_extractive(question, context_chunks)

    # ---- LLM backends -----------------------------------------------

    def _generate_anthropic(self, question: str, context_chunks: list[str]) -> str:
        import anthropic
        client = anthropic.Anthropic()
        prompt = build_prompt(question, context_chunks)
        response = client.messages.create(
            model=self.model or "claude-sonnet-5",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def _generate_openai(self, question: str, context_chunks: list[str]) -> str:
        from openai import OpenAI
        client = OpenAI()
        prompt = build_prompt(question, context_chunks)
        response = client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    # ---- Offline fallback ---------------------------------------------

    def _generate_extractive(self, question: str, context_chunks: list[str]) -> str:
        """
        No LLM available: pick the sentences across the retrieved chunks
        that share the most keywords with the question, and stitch them
        into an answer. This is a *retrieval-only* baseline -- it shows
        the pipeline working end-to-end without needing an API key.
        """
        if not context_chunks:
            return "I don't have enough information in the document to answer that."

        question_words = set(re.findall(r"\w+", question.lower())) - _STOPWORDS
        sentences = []
        for chunk in context_chunks:
            sentences.extend(re.split(r"(?<=[.!?])\s+", chunk))

        scored = []
        for s in sentences:
            words = set(re.findall(r"\w+", s.lower()))
            overlap = len(words & question_words)
            if overlap > 0:
                scored.append((overlap, s.strip()))

        scored.sort(key=lambda x: -x[0])
        top_sentences = [s for _, s in scored[:3]]

        if not top_sentences:
            return ("I couldn't find a direct answer in the retrieved context. "
                    "Here's the most relevant excerpt I found:\n\n" + context_chunks[0][:400])

        return " ".join(top_sentences)


_STOPWORDS = {
    "the", "is", "are", "a", "an", "of", "to", "in", "and", "what", "how",
    "why", "does", "do", "did", "for", "on", "with", "this", "that", "it",
    "was", "were", "be", "as", "at", "by", "from", "or", "which", "who",
}
