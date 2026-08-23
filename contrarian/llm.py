"""The single entry point for every LLM call in this project.

Only two stages ever call this module: analyze.py (stage 3) and reflect.py
(stage 4). Screening, enrichment, and reporting are deterministic Python and
must stay that way — every extra LLM call is another chance for a number to
drift away from its source.

Provider is decided entirely by config (base_url + model), so this file has no
provider-specific branching in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import openai
from openai import OpenAI

from .config import LLMConfig, load_config


class LLMError(RuntimeError):
    """An LLM call failed, with a message aimed at the person running it."""


@dataclass(frozen=True)
class ChatResult:
    """A completed call: the text, plus what it cost us."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_client: OpenAI | None = None
_client_config: LLMConfig | None = None


def get_client(config: LLMConfig | None = None) -> OpenAI:
    """Return a cached OpenAI-compatible client for the configured provider.

    Cached because constructing a client opens a connection pool; rebuilding one
    per call would be wasteful across a run of 10-20 candidates.
    """
    global _client, _client_config

    config = config or load_config()
    if _client is None or _client_config != config:
        _client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )
        _client_config = config
    return _client


def chat(
    prompt: str,
    *,
    system: str | None = None,
    config: LLMConfig | None = None,
    temperature: float | None = None,
) -> ChatResult:
    """Send a single-turn prompt and return the response.

    Single-turn by design. Stage 4 (reflection) must audit stage 3's output from
    a *fresh* context — a model shown its own prior reasoning anchors on it and
    tends to agree with itself. Keeping this function stateless makes that the
    default rather than something you have to remember.

    Args:
        prompt: The user message.
        system: Optional system message setting the model's role.
        config: Override config. Defaults to loading from .env.
        temperature: Override sampling temperature. Defaults to config value.

    Raises:
        LLMError: On any API or connection failure, with guidance on the fix.
    """
    config = config or load_config()
    client = get_client(config)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature if temperature is not None else config.temperature,
        )
    except openai.APIConnectionError as exc:
        if config.is_local:
            raise LLMError(
                f"Could not reach a local model at {config.base_url}\n\n"
                f"Is Ollama running? Start it with:\n"
                f"    ollama serve\n\n"
                f"And make sure the model is pulled:\n"
                f"    ollama pull {config.model}"
            ) from exc
        raise LLMError(
            f"Could not reach {config.provider} at {config.base_url}\n"
            f"Check your network connection and LLM_BASE_URL."
        ) from exc
    except openai.AuthenticationError as exc:
        raise LLMError(
            f"{config.provider} rejected the API key.\n"
            f"Check LLM_API_KEY in your .env file."
        ) from exc
    except openai.NotFoundError as exc:
        raise LLMError(
            f"Model {config.model!r} not found on {config.provider}.\n\n"
            + (
                f"Pull it first:\n    ollama pull {config.model}\n\n"
                f"Or list what you already have:\n    ollama list"
                if config.is_local
                else "Check LLM_MODEL against the provider's supported model list."
            )
        ) from exc
    except openai.RateLimitError as exc:
        raise LLMError(
            f"{config.provider} rate limit hit.\n"
            f"Free tiers are typically ~30 requests/min. Wait and retry, or "
            f"switch LLM_BASE_URL to a local Ollama model for unlimited calls."
        ) from exc
    except openai.APIStatusError as exc:
        raise LLMError(f"{config.provider} returned {exc.status_code}: {exc.message}") from exc

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise LLMError(
            f"{config.model} returned an empty response "
            f"(finish_reason={choice.finish_reason!r}).\n"
            f"If finish_reason is 'length', the model hit its output cap."
        )

    usage = response.usage
    return ChatResult(
        text=text,
        model=response.model or config.model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
