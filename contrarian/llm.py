"""The single entry point for every LLM call in this project.

Only two stages ever call this module: analyze.py (stage 3) and reflect.py
(stage 4). Screening, enrichment, and reporting are deterministic Python and
must stay that way — every extra LLM call is another chance for a number to
drift away from its source.

Provider is decided entirely by config (base_url + model), so this file has no
provider-specific branching in it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Type, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import LLMConfig, load_config

T = TypeVar("T", bound=BaseModel)


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


def strict_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """Turn a Pydantic model into a schema the provider will enforce.

    Two adjustments to what Pydantic emits:

    - `additionalProperties: false` everywhere, which strict mode requires and
      Pydantic does not add.
    - `$defs`/`$ref` inlined. Ollama compiles the schema to a GBNF grammar and
      does not resolve references, so a nested model would fail at generation
      time rather than validation time — the confusing kind of failure.

    **Field order in the model is generation order.** The grammar emits
    properties in schema order, so the model physically writes each field before
    the next. `analyze.py` relies on this to force the bear case out before the
    verdict; see design rule 5.
    """
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})
    return _inline_refs(schema, definitions)


def _inline_refs(node: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return _inline_refs(dict(definitions.get(name, {})), definitions)
        result = {key: _inline_refs(value, definitions) for key, value in node.items()}
        if result.get("type") == "object":
            result["additionalProperties"] = False
        return result
    if isinstance(node, list):
        return [_inline_refs(item, definitions) for item in node]
    return node


def chat_structured(
    prompt: str,
    schema: Type[T],
    *,
    system: str | None = None,
    config: LLMConfig | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[T, ChatResult]:
    """Send a prompt and get back a validated instance of `schema`.

    Uses provider-side schema enforcement (`response_format=json_schema`) rather
    than asking for JSON in the prompt and hoping. Verified against Ollama +
    qwen3:14b: the response parses first try, and the hybrid-reasoning `<think>`
    block that the Phase 1 smoke test produced does *not* leak into the output —
    constrained decoding leaves no room for it. That was the open risk
    CLAUDE.md flagged before this phase could be built.

    Raises:
        LLMError: On API failure, unparseable JSON, or a response that does not
            satisfy the schema.
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
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": strict_schema(schema),
                },
            },
        )
    except openai.BadRequestError as exc:
        raise LLMError(
            f"{config.provider} rejected the structured-output request.\n"
            f"Not every provider or model supports json_schema response format.\n"
            f"Underlying error: {exc}"
        ) from exc
    except (openai.APIConnectionError, openai.APIStatusError) as exc:
        raise _connection_error(exc, config) from exc

    choice = response.choices[0]
    text = (choice.message.content or "").strip()

    if choice.finish_reason == "length":
        raise LLMError(
            f"{config.model} hit its output cap mid-JSON, so the response cannot "
            f"be parsed.\nRaise max_tokens — for a hybrid reasoning model budget "
            f"roughly 3x the visible answer length."
        )
    if not text:
        raise LLMError(
            f"{config.model} returned an empty response "
            f"(finish_reason={choice.finish_reason!r})."
        )

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"{config.model} returned text that is not JSON, despite a schema "
            f"being enforced.\nFirst 300 chars:\n{text[:300]}"
        ) from exc

    try:
        parsed = schema.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(
            f"{config.model} returned JSON that does not satisfy {schema.__name__}:\n"
            f"{exc}\n\nPayload:\n{json.dumps(payload, indent=2)[:500]}"
        ) from exc

    usage = response.usage
    result = ChatResult(
        text=text,
        model=response.model or config.model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
    return parsed, result


def _connection_error(exc: Exception, config: LLMConfig) -> LLMError:
    """Shared connection-failure guidance for both call paths."""
    if isinstance(exc, openai.APIConnectionError):
        if config.is_local:
            return LLMError(
                f"Could not reach a local model at {config.base_url}\n\n"
                f"Is Ollama running? Start it with:\n"
                f"    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve\n\n"
                f"And make sure the model is pulled:\n"
                f"    ollama pull {config.model}"
            )
        return LLMError(
            f"Could not reach {config.provider} at {config.base_url}\n"
            f"Check your network connection and LLM_BASE_URL."
        )
    status = getattr(exc, "status_code", "?")
    message = getattr(exc, "message", str(exc))
    return LLMError(f"{config.provider} returned {status}: {message}")


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
