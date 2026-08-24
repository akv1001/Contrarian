#!/usr/bin/env python3
"""Phase 5 — screen, enrich, judge, then audit the judgment.

The Phase 5 gate is "a deliberately weak verdict gets caught and demoted", so
that case is a flag rather than something you have to set up by hand:

    .venv/bin/python scripts/run_reflect.py --weak ITC     # the gate
    .venv/bin/python scripts/run_reflect.py --limit 2      # the real pipeline

`--weak` skips the stage-3 call and substitutes a note built to fail every check:
a figure that appears nowhere in the data, a catalyst asserted with no mechanism,
a one-line brush-off of the bear case, and high confidence on all of it. If that
comes back KEEP, the stage is not working.

Options:
    --limit N        Audit the top N survivors. Two LLM calls each, so a local
                     14B model takes roughly two minutes per candidate.
    --symbol SYM     One ticker, whether or not it passed the screen.
    --weak SYM       Run the deliberately weak note against SYM's real data.
    --show-prompt    Print the audit prompt. Useful for checking the note really
                     is presented unattributed.
    --refresh        Bypass the data cache.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrarian import analyze as analyze_stage
from contrarian import enrich as enrich_stage
from contrarian import reflect as reflect_stage
from contrarian import screen
from contrarian.config import ConfigError, load_config
from contrarian.data.news_rss import default_source
from contrarian.llm import LLMError
from contrarian.schema import AnalyzedCandidate, ReflectedCandidate, Verdict
from contrarian.universe import NIFTY_50, by_symbol

VERDICT_LABEL = {
    "TEMPORARY_DISLOCATION": "temporary dislocation",
    "VALUE_TRAP": "VALUE TRAP",
    "UNCLEAR": "unclear",
}

OUTCOME_MARK = {"KEEP": "KEEP", "DEMOTE": "DEMOTE", "REJECT": "REJECT"}

# The Phase 5 gate, written to fail every check the stage makes.
#
#   - "revenue grew 34.7%" is invented; no such figure exists in any candidate's
#     data, so the mechanical check flags it and KEEP becomes unreachable.
#   - the temporary factor is a mechanism-free assertion — sentiment, rotation,
#     "should normalise" — with nothing that resolves and no date.
#   - the value-trap case is raised and dismissed in one clause, which is exactly
#     what design rule 5 exists to prevent.
#   - confidence 88 is unsupportable on any of it.
WEAK_VERDICT = Verdict(
    temporary_factor=(
        "Negative market sentiment and sector rotation have pushed the stock down. "
        "This is a temporary phase and the price should normalise once sentiment "
        "improves."
    ),
    bull_case=(
        "The company is a market leader with strong fundamentals and revenue grew "
        "34.7% last year. Management is confident and the balance sheet is solid, "
        "so the business is clearly sound."
    ),
    value_trap_case="Some might argue growth is slowing, but this seems unlikely.",
    verdict="TEMPORARY_DISLOCATION",
    confidence=88,
    what_would_change_my_mind="If the stock keeps falling.",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the contrarian pipeline through reflection.")
    p.add_argument("--limit", type=int, default=2, help="top N survivors (default 2)")
    p.add_argument("--symbol", help="one ticker, screened or not")
    p.add_argument("--weak", metavar="SYM",
                   help="audit a deliberately weak note against SYM's real data")
    p.add_argument("--show-prompt", action="store_true", help="print the audit prompt")
    p.add_argument("--refresh", action="store_true", help="bypass the data cache")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"\nConfiguration error:\n\n{exc}\n")
        return 1

    print("ContrarianOppAIAgent — screen, enrich, analyze, reflect")
    print("=" * 78)
    print(f"Model: {config.model} via {config.provider}")

    source = default_source()
    result = screen.run(source, NIFTY_50, refresh=args.refresh)

    selected = _select(result, args)
    if not selected:
        return 1

    if args.weak:
        print("\nGATE: auditing a deliberately weak note against real data.")
        print("Expected: not KEEP, confidence well below the note's 88.")
    print(f"\nScreen: {len(result.passed)} of {result.universe_size} passed. "
          f"Auditing {len(selected)}.")
    print("=" * 78)

    reflected: list[ReflectedCandidate] = []
    for i, candidate in enumerate(selected, 1):
        enriched = enrich_stage.enrich(candidate, source, refresh=args.refresh)
        print(
            f"\n[{i}/{len(selected)}] {enriched.symbol} — {enriched.name}"
            f"   ({len(enriched.facts)} figures, {len(enriched.headlines)} headlines)"
        )

        analyzed = _judge(enriched, config, weak=bool(args.weak))
        if analyzed is None:
            continue
        _print_verdict(analyzed)

        if args.show_prompt:
            print("\n" + "-" * 78)
            print(reflect_stage.build_prompt(analyzed))
            print("-" * 78)

        print("\n    auditing...", end="", flush=True)
        started = time.time()
        try:
            outcome = reflect_stage.reflect(analyzed, config=config)
        except LLMError as exc:
            print(f"\r    AUDIT FAILED: {str(exc).splitlines()[0]}")
            continue
        print(f"\r    audited in {time.time() - started:.0f}s"
              f"   ({outcome.completion_tokens} completion tokens)")

        reflected.append(outcome)
        _print_reflection(outcome)

    return _print_summary(reflected, len(selected), weak=bool(args.weak))


def _judge(enriched, config, *, weak: bool) -> AnalyzedCandidate | None:
    """Produce the note to be audited — real, or the deliberately weak one."""
    if weak:
        issues, headline_sourced = analyze_stage.check_grounding(WEAK_VERDICT, enriched)
        return AnalyzedCandidate(
            enriched=enriched,
            verdict=WEAK_VERDICT,
            grounding_issues=issues,
            headline_sourced=headline_sourced,
        )

    print("    thinking...", end="", flush=True)
    started = time.time()
    try:
        analyzed = analyze_stage.analyze(enriched, config=config)
    except LLMError as exc:
        print(f"\r    ANALYSIS FAILED: {str(exc).splitlines()[0]}")
        return None
    print(f"\r    judged in {time.time() - started:.0f}s"
          f"   ({analyzed.completion_tokens} completion tokens)")
    return analyzed


def _select(result, args) -> list:
    symbol = args.weak or args.symbol
    if symbol:
        symbol = symbol.upper()
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
        if not match.passed and not args.weak:
            print(f"  note: {symbol} did NOT pass the screen "
                  f"({match.failure_summary}). Analyzing anyway.")
        return [match]

    if not result.passed:
        print("\n  Nothing passed the screen. Try scripts/run_screen.py "
              "--show-rejects 10.\n")
        return []
    return list(result.passed[: args.limit])


def _print_verdict(analyzed: AnalyzedCandidate) -> None:
    v = analyzed.verdict
    print(f"\n    NOTE: {VERDICT_LABEL.get(v.verdict, v.verdict)}"
          f"   stated confidence {v.confidence}/100")
    print(_wrap(v.temporary_factor))
    if analyzed.grounding_issues:
        print(f"    (mechanical check: {len(analyzed.grounding_issues)} ungrounded "
              f"figure(s) — "
              + ", ".join(f"{i.value:g}" for i in analyzed.grounding_issues) + ")")


def _print_reflection(r: ReflectedCandidate) -> None:
    ref = r.reflection
    arrow = f"{r.original_confidence} -> {r.final_confidence}"
    drop = f" (-{r.confidence_drop})" if r.confidence_drop else ""
    print(f"\n    AUDIT: {OUTCOME_MARK.get(ref.outcome, ref.outcome)}"
          f"   confidence {arrow}{drop}")
    if r.final_verdict != r.original_verdict:
        print(f"    verdict withdrawn: "
              f"{VERDICT_LABEL.get(r.original_verdict, r.original_verdict)} -> "
              f"{VERDICT_LABEL.get(r.final_verdict, r.final_verdict)}")

    for label, text in (
        ("Figures", ref.figures_check),
        ("Claims", ref.claim_check),
        ("Bear case", ref.bear_case_engagement),
        ("Catalyst", ref.catalyst_concreteness),
        ("Calibration", ref.confidence_calibration),
    ):
        print(f"\n    {label}:")
        print(_wrap(text))

    if ref.weaknesses:
        print(f"\n    Weaknesses found:")
        for item in ref.weaknesses:
            print(_wrap(f"- {item}"))

    if r.overrides:
        print(f"\n    !! ASYMMETRY ENFORCED — the reflector was overruled:")
        for note in r.overrides:
            print(_wrap(f"- {note}"))

    issues, _ = reflect_stage.check_reflection_grounding(r)
    if issues:
        print(f"\n    !! the audit itself cites {len(issues)} figure(s) not in the "
              f"source data:")
        for issue in issues:
            print(_wrap(f"- {issue.field}: {issue.value:g}"))


def _print_summary(reflected: list[ReflectedCandidate], attempted: int, *, weak: bool) -> int:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    if not reflected:
        print("  Nothing audited successfully.")
        return 1

    for r in reflected:
        print(
            f"  {r.symbol:12} {VERDICT_LABEL.get(r.final_verdict, '?'):22}"
            f" {r.reflection.outcome:7} conf {r.original_confidence:>3} -> "
            f"{r.final_confidence:>3}"
            + ("  !! overruled" if r.overrides else "")
        )

    tokens = sum(r.prompt_tokens + r.completion_tokens for r in reflected)
    print(f"\n  {len(reflected)}/{attempted} audited, {tokens:,} reflection tokens")

    if weak:
        return _print_gate(reflected)

    kept = [r for r in reflected if r.reflection.outcome == "KEEP"]
    print(f"\n  {len(kept)} kept, "
          f"{sum(1 for r in reflected if r.reflection.outcome == 'DEMOTE')} demoted, "
          f"{sum(1 for r in reflected if r.reflection.outcome == 'REJECT')} rejected.")
    print("\n  Reflection can only lower conviction. Nothing here has been raised,\n"
          "  and a KEEP means the reasoning survived an audit — not that the idea\n"
          "  is correct. This is idea generation, not investment advice.")
    print()
    return 0


def _print_gate(reflected: list[ReflectedCandidate]) -> int:
    """Report the Phase 5 gate as a pass or a fail, not as output to interpret."""
    print("\n" + "-" * 78)
    failures = []
    for r in reflected:
        if r.reflection.outcome == "KEEP":
            failures.append(f"{r.symbol}: weak note came back KEEP")
        if r.final_confidence >= r.original_confidence:
            failures.append(f"{r.symbol}: confidence was not lowered")
        if r.final_confidence > 50:
            failures.append(f"{r.symbol}: confidence {r.final_confidence} still high")

    if failures:
        print("  GATE FAILED — the weak note was not caught:")
        for line in failures:
            print(f"    - {line}")
        print()
        return 1

    print("  GATE PASSED — the deliberately weak note was caught and demoted.")
    print()
    return 0


def _wrap(text: str, indent: int = 6, width: int = 76) -> str:
    pad = " " * indent
    return textwrap.fill(text, width=width, initial_indent=pad, subsequent_indent=pad)


if __name__ == "__main__":
    raise SystemExit(main())
