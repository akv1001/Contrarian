#!/usr/bin/env python3
"""Phase 4 — screen, enrich, and judge. The first LLM stage.

The Phase 4 gate is "valid JSON parses into Verdict objects, and every number in
the output traces to the input", so this script reports the grounding audit as
prominently as the verdicts themselves. A clean run with an ungrounded figure in
it is a failed run.

Run it:
    .venv/bin/python scripts/run_analyze.py --limit 3

Options:
    --limit N        Analyze only the top N survivors. Each call takes ~40-60s
                     on a local 14B model, so the full nine take several minutes.
    --symbol SYM     Analyze one ticker, whether or not it passed the screen.
                     The way to test a known value trap against the pipeline.
    --show-prompt    Print the prompt before each call. Every number the model
                     could legitimately cite is in there.
    --refresh        Bypass the data cache.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrarian import analyze as analyze_stage
from contrarian import enrich as enrich_stage
from contrarian import screen
from contrarian.config import ConfigError, load_config
from contrarian.data.news_rss import default_source
from contrarian.llm import LLMError
from contrarian.schema import AnalyzedCandidate
from contrarian.universe import NIFTY_50, by_symbol

VERDICT_LABEL = {
    "TEMPORARY_DISLOCATION": "temporary dislocation",
    "VALUE_TRAP": "VALUE TRAP",
    "UNCLEAR": "unclear",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the contrarian analysis pipeline.")
    p.add_argument("--limit", type=int, default=3, help="top N survivors (default 3)")
    p.add_argument("--symbol", help="analyze one ticker, screened or not")
    p.add_argument("--show-prompt", action="store_true", help="print each prompt")
    p.add_argument("--refresh", action="store_true", help="bypass the data cache")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"\nConfiguration error:\n\n{exc}\n")
        return 1

    print("ContrarianOppAIAgent — screen, enrich, analyze")
    print("=" * 78)
    print(f"Model: {config.model} via {config.provider}")

    source = default_source()
    result = screen.run(source, NIFTY_50, refresh=args.refresh)

    selected = _select(result, args)
    if not selected:
        return 1

    print(f"Screen: {len(result.passed)} of {result.universe_size} passed. "
          f"Analyzing {len(selected)}.")
    print("=" * 78)

    analyzed: list[AnalyzedCandidate] = []
    for i, candidate in enumerate(selected, 1):
        enriched = enrich_stage.enrich(candidate, source, refresh=args.refresh)
        print(
            f"\n[{i}/{len(selected)}] {enriched.symbol} — {enriched.name}"
            f"   ({len(enriched.facts)} figures, {len(enriched.headlines)} headlines)"
        )

        if args.show_prompt:
            print("\n" + "-" * 78)
            print(analyze_stage.build_prompt(enriched))
            print("-" * 78)

        print("    thinking...", end="", flush=True)
        started = time.time()
        try:
            outcome = analyze_stage.analyze(enriched, config=config)
        except LLMError as exc:
            print(f"\r    FAILED: {str(exc).splitlines()[0]}")
            continue
        print(f"\r    done in {time.time() - started:.0f}s"
              f"   ({outcome.completion_tokens} completion tokens)")

        analyzed.append(outcome)
        _print_verdict(outcome)

    return _print_summary(analyzed, len(selected))


def _select(result, args) -> list:
    """Pick which candidates to analyze."""
    if args.symbol:
        symbol = args.symbol.upper()
        everything = list(result.passed) + list(result.rejected)
        match = next((c for c in everything if c.symbol == symbol), None)
        if match is None:
            try:
                by_symbol(symbol)
                print(f"\n  {symbol} is in the universe but was not screened "
                      f"(fetch error?).\n")
            except KeyError:
                print(f"\n  {symbol!r} is not in the Nifty 50 list.\n")
            return []
        if not match.passed:
            # Allowed on purpose: running a known trap through the pipeline is
            # how you find out whether it gets caught.
            print(f"  note: {symbol} did NOT pass the screen "
                  f"({match.failure_summary}). Analyzing anyway.")
        return [match]

    if not result.passed:
        print("\n  Nothing passed the screen. Try scripts/run_screen.py "
              "--show-rejects 10.\n")
        return []
    return list(result.passed[: args.limit])


def _print_verdict(outcome: AnalyzedCandidate) -> None:
    v = outcome.verdict
    label = VERDICT_LABEL.get(v.verdict, v.verdict)
    print(f"\n    VERDICT: {label}   confidence {v.confidence}/100")
    print(f"\n    Temporary factor:")
    print(_wrap(v.temporary_factor))
    print(f"\n    Bull case:")
    print(_wrap(v.bull_case))
    print(f"\n    Strongest value-trap case:")
    print(_wrap(v.value_trap_case))
    print(f"\n    What would change its mind:")
    print(_wrap(v.what_would_change_my_mind))

    if outcome.grounding_issues:
        print(f"\n    !! GROUNDING: {len(outcome.grounding_issues)} figure(s) "
              f"not traceable to the input")
        for issue in outcome.grounding_issues:
            print(f"       - {issue.field}: {issue.value:g}")
            print(_wrap(f'"{issue.context}"', indent=9))
    else:
        print(f"\n    grounding: clean — every figure traces to the input")

    if outcome.headline_sourced:
        cited = ", ".join(f"{v:g}" for v in outcome.headline_sourced)
        print(f"    note: {len(outcome.headline_sourced)} figure(s) cited from "
              f"headlines, not verified data ({cited})")


def _print_summary(analyzed: list[AnalyzedCandidate], attempted: int) -> int:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    if not analyzed:
        print("  Nothing analyzed successfully.")
        return 1

    for outcome in analyzed:
        flag = "" if outcome.is_grounded else f"  !! {len(outcome.grounding_issues)} ungrounded"
        print(
            f"  {outcome.symbol:12} {VERDICT_LABEL.get(outcome.verdict.verdict, '?'):22}"
            f" conf {outcome.verdict.confidence:>3}{flag}"
        )

    ungrounded = [o for o in analyzed if not o.is_grounded]
    tokens = sum(o.prompt_tokens + o.completion_tokens for o in analyzed)
    print(f"\n  {len(analyzed)}/{attempted} analyzed, {tokens:,} tokens total")

    if ungrounded:
        print(
            f"\n  {len(ungrounded)} verdict(s) cite figures not in their input. "
            f"That is the\n  hallucination the grounding check exists to catch — "
            f"treat those verdicts as\n  unreliable and read the flagged sentences."
        )
    else:
        print(
            "\n  Every figure cited traces back to the input.\n"
            "  Note what this does NOT prove: a real number can still be used to\n"
            "  support a fabricated claim. Phase 5 audits the reasoning itself."
        )
    print()
    return 0


def _wrap(text: str, indent: int = 6, width: int = 76) -> str:
    import textwrap

    pad = " " * indent
    return "\n".join(
        textwrap.fill(text, width=width, initial_indent=pad, subsequent_indent=pad)
        for _ in [0]
    )


if __name__ == "__main__":
    raise SystemExit(main())
