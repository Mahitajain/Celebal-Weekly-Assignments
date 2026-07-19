"""
Unified LLM client.

Real production systems rarely hardcode a single model provider -- pricing,
rate limits, and latency requirements shift. This module exposes one
`LLMClient.complete()` call and hides Anthropic / OpenAI / Groq behind it.

If no API key is configured (`llm_provider == "none"`), the client degrades
to a `None` return rather than raising, and every caller in this codebase
is written to handle that gracefully (e.g. the SQL agent falls back to a
deterministic template-based query builder; the RAG agent falls back to
extractive answering). This keeps the project runnable end-to-end even
without paid API access, which matters for a portfolio piece someone else
will try to run locally.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMUnavailableError(RuntimeError):
    """Raised only by callers who explicitly require a live LLM."""


class LLMClient:
    """Thin, provider-agnostic wrapper around chat-completion APIs."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None
        self._provider = self.settings.llm_provider
        self._init_client()

    def _init_client(self) -> None:
        provider = self._provider
        try:
            if provider == "anthropic" and self.settings.anthropic_api_key:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            elif provider == "openai" and self.settings.openai_api_key:
                import openai

                self._client = openai.OpenAI(api_key=self.settings.openai_api_key)
            elif provider == "groq" and self.settings.groq_api_key:
                import groq

                self._client = groq.Groq(api_key=self.settings.groq_api_key)
            else:
                if provider != "none":
                    logger.warning(
                        "LLM provider '%s' selected but no API key found; "
                        "falling back to rule-based / extractive behavior.",
                        provider,
                    )
                self._provider = "none"
        except ImportError as exc:
            logger.warning("SDK for provider '%s' not installed (%s); disabling LLM calls.", provider, exc)
            self._provider = "none"

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse | None:
        """Return a completion, or None if no LLM is configured/available."""
        if not self.available:
            return None

        temperature = self.settings.llm_temperature if temperature is None else temperature
        max_tokens = self.settings.llm_max_tokens if max_tokens is None else max_tokens
        model = self.settings.llm_model

        try:
            if self._provider == "anthropic":
                resp = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(block.text for block in resp.content if block.type == "text")
                return LLMResponse(
                    text=text,
                    model=model,
                    provider="anthropic",
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                )

            if self._provider in ("openai", "groq"):
                resp = self._client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                choice = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                return LLMResponse(
                    text=choice,
                    model=model,
                    provider=self._provider,
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
                )
        except Exception:
            logger.exception("LLM call failed; caller should fall back gracefully.")
            return None

        return None


_client_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClient()
    return _client_singleton
