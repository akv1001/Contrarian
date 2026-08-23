#!/usr/bin/env python3
"""Phase 1 smoke test — proves the LLM path works end to end.

Deliberately not a "hello world". It runs a miniature version of the stage-3
judgment call on hardcoded figures, so it doubles as a cheap way to compare
providers and models on the task that actually matters: distinguishing a
temporary dip from a structural decline.

Run it:
    .venv/bin/python scripts/smoke_test.py

Swap models without editing anything:
    LLM_MODEL=qwen3:32b .venv/bin/python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrarian.config import ConfigError, load_config
from contrarian.llm import LLMError, chat

SYSTEM = (
    "You are a value-investing analyst. You reason only from figures you are "
    "given. You never introduce numbers that are not in the input. When "
    "evidence is insufficient to reach a conclusion, you say so plainly."
)

# Hardcoded so the smoke test needs no network data source. Loosely modelled on a
# real pattern: cheap on earnings, but the cash flow and debt trend are the tell.
PROMPT = """A Nifty 50 company trades at these figures:

- Price: down 34% over 12 months, currently near its 52-week low
- P/E: 9.2 (sector median: 21.4)
- Free cash flow: negative for 3 consecutive quarters
- Debt/equity: 2.1, up from 0.8 two years ago
- Revenue: flat year over year
- Recent news: a regulatory probe into the sector was announced last quarter

Answer in under 150 words:
1. Is this most likely a temporary dislocation or a structural decline?
2. What is the single strongest argument that this is a value trap?
3. What one additional data point would most change your answer?
"""


def main() -> int:
    print("ContrarianOppAIAgent — Phase 1 smoke test")
    print("=" * 60)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"\nConfiguration error:\n\n{exc}\n")
        return 1

    print(f"Provider:    {config.provider}")
    print(f"Endpoint:    {config.base_url}")
    print(f"Model:       {config.model}")
    print(f"Temperature: {config.temperature}")
    print("=" * 60)
    print("\nSending request...\n")

    try:
        result = chat(PROMPT, system=SYSTEM, config=config)
    except LLMError as exc:
        print(f"Request failed:\n\n{exc}\n")
        return 1

    print("-" * 60)
    print(result.text)
    print("-" * 60)
    print(
        f"\nmodel={result.model}  "
        f"prompt={result.prompt_tokens} tok  "
        f"completion={result.completion_tokens} tok  "
        f"total={result.total_tokens} tok"
    )
    print("\nPhase 1 verified: inference path works.")
    print(
        "\nWorth reading the answer, not just checking it ran. If the model "
        "misses that\nnegative FCF plus rising leverage is the structural tell, "
        "it will miss it on real\ndata too — try a larger model before moving to "
        "Phase 2."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
