"""Configuration loading and validation.

Everything the app needs to reach an LLM comes from three environment variables
(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL), loaded from a project-local .env file.

Why this shape: Ollama, Groq, and OpenRouter all speak the OpenAI-compatible
protocol, so switching providers is a change to these values and nothing else.
No code change, no new dependency.

Config is validated once at startup rather than lazily at call time. A missing
key should fail immediately with a clear message, not thirty seconds into a run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid.

    Messages should tell the user exactly what to do next, not just what is wrong.
    """


@dataclass(frozen=True)
class LLMConfig:
    """Everything needed to talk to an OpenAI-compatible endpoint."""

    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0
    temperature: float = 0.2

    @property
    def provider(self) -> str:
        """Best-effort human-readable provider name, derived from the host.

        Cosmetic only — used in log lines so you can tell at a glance which
        backend a run actually used.
        """
        host = self.base_url.lower()
        if "localhost" in host or "127.0.0.1" in host:
            return "Ollama (local)"
        if "groq.com" in host:
            return "Groq"
        if "openrouter.ai" in host:
            return "OpenRouter"
        if "cerebras" in host:
            return "Cerebras"
        return host

    @property
    def is_local(self) -> bool:
        return "localhost" in self.base_url or "127.0.0.1" in self.base_url


def _require(name: str) -> str:
    """Read a required env var, or raise with actionable guidance."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set.\n\n"
            f"Create a .env file in the project root:\n"
            f"    cp .env.example .env\n\n"
            f"then edit it and uncomment one provider block."
        )
    return value


def _optional_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from exc


def load_config() -> LLMConfig:
    """Load and validate configuration from .env plus the environment.

    Real environment variables take precedence over .env, which is what you want
    for CI or one-off overrides:
        LLM_MODEL=qwen3:32b python scripts/smoke_test.py
    """
    # Load .env if present, but don't require it — exported environment variables
    # alone are a valid configuration (CI, containers, one-off overrides).
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)
    elif not os.getenv("LLM_BASE_URL"):
        raise ConfigError(
            f"No .env file found at {ENV_FILE}, and LLM_BASE_URL is not set.\n\n"
            f"Create one from the template:\n"
            f"    cp .env.example .env\n\n"
            f"then edit it and uncomment one provider block."
        )

    config = LLMConfig(
        base_url=_require("LLM_BASE_URL"),
        api_key=_require("LLM_API_KEY"),
        model=_require("LLM_MODEL"),
        timeout=_optional_float("LLM_TIMEOUT", 120.0),
        temperature=_optional_float("LLM_TEMPERATURE", 0.2),
    )

    # Guard against the Claude Code billing trap: a globally exported
    # ANTHROPIC_API_KEY causes Claude Code to bill at API rates instead of using
    # a Pro subscription. This project never uses Anthropic, so if the variable
    # is present it was almost certainly set by accident.
    if os.getenv("ANTHROPIC_API_KEY"):
        print(
            "  warning: ANTHROPIC_API_KEY is set in your environment.\n"
            "           This project does not use it, and a globally set value makes\n"
            "           Claude Code bill at API rates instead of using your Pro plan."
        )

    return config
