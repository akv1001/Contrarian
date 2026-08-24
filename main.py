#!/usr/bin/env python3
"""ContrarianOppAIAgent — scan the Nifty 50 for contrarian value picks.

Wires all five stages together and writes a Markdown report:

    screen (no LLM) -> enrich (no LLM) -> analyze (LLM) -> reflect (LLM) -> report (no LLM)

    .venv/bin/python main.py                    # everything that passed the screen
    .venv/bin/python main.py --limit 3          # just the top 3, for a quick look
    .venv/bin/python main.py --refresh          # cold cache: refetch everything
    .venv/bin/python main.py --out -            # print the report instead of writing it

Two LLM calls per candidate and no more. On a local 14B model each candidate
takes roughly two minutes, so a full run over nine survivors is a coffee break;
the script prints an estimate up front and progress as it goes.

A candidate whose LLM stages fail does not sink the run — it is recorded in the
report's appendix as skipped, which is deliberately not the same as rejected.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contrarian import analyze as analyze_stage
from contrarian import enrich as enrich_stage
from contrarian import reflect as reflect_stage
from contrarian import report as report_stage
from contrarian import screen
from contrarian.config import ConfigError, load_config
from contrarian.data.news_rss import default_source
from contrarian.llm import LLMError
from contrarian.schema import PipelineRun, ReflectedCandidate
from contrarian.universe import NIFTY_50

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Rough wall-clock per candidate on a local 14B model: two calls at ~60s each,
# plus enrichment. Only used for the up-front estimate, so precision is pointless
# — but a run that silently does nothing for twenty minutes feels broken.
SECONDS_PER_CANDIDATE = 140


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan the Nifty 50 for contrarian value picks.",
        epilog="Idea generation, not investment advice.",
    )
    p.add_argument("--limit", type=int, default=0,
                   help="analyze only the top N survivors (default: all of them)")
    p.add_argument("--refresh", action="store_true",
                   help="bypass the data cache and refetch everything")
    p.add_argument("--out", default=None,
                   help="where to write the report; '-' prints to stdout "
                        f"(default: {REPORTS_DIR.name}/contrarian-YYYY-MM-DD.md)")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()

    def say(message: str = "") -> None:
        if not args.quiet:
            print(message, flush=True)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"\nConfiguration error:\n\n{exc}\n")
        return 1

    say("ContrarianOppAIAgent")
    say("=" * 78)
    say(f"Model: {config.model} via {config.provider}")
    say("Idea generation, not investment advice.")
    say()

    source = default_source()

    say("[1/5] Screening the Nifty 50...")
    if args.refresh:
        say("      (--refresh: refetching everything, this takes a few minutes)")
    screen_result = screen.run(source, NIFTY_50, refresh=args.refresh)
    say(f"      {len(screen_result.passed)} of {screen_result.universe_size} passed "
        f"the quantitative gates.")
    if screen_result.errors:
        say(f"      {len(screen_result.errors)} ticker(s) could not be fetched.")

    selected = list(screen_result.passed)
    if args.limit > 0:
        selected = selected[: args.limit]

    if not selected:
        say("\n  Nothing passed the screen. Nothing to analyze.")
        say("  Try: .venv/bin/python scripts/run_screen.py --show-rejects 10\n")
        return 1

    estimate = len(selected) * SECONDS_PER_CANDIDATE / 60
    say(f"\n      Analyzing {len(selected)} candidate(s): two LLM calls each, "
        f"roughly {estimate:.0f} min.")
    say("=" * 78)

    reflected: list[ReflectedCandidate] = []
    skipped: list[tuple[str, str]] = []

    for i, candidate in enumerate(selected, 1):
        say(f"\n[{i}/{len(selected)}] {candidate.symbol} — {candidate.name}")

        say("      [2/5] enriching (figures + news)...")
        enriched = enrich_stage.enrich(candidate, source, refresh=args.refresh)
        say(f"            {len(enriched.facts)} figures, "
            f"{len(enriched.headlines)} headlines")

        say("      [3/5] judging...")
        step = time.time()
        try:
            analyzed = analyze_stage.analyze(enriched, config=config)
        except LLMError as exc:
            reason = str(exc).splitlines()[0]
            say(f"            FAILED: {reason}")
            skipped.append((candidate.symbol, f"analysis failed — {reason}"))
            continue
        grounding = (
            f"!! {len(analyzed.fabricated_figures)} ungrounded figure(s)"
            if analyzed.fabricated_figures
            else "grounding clean"
        )
        say(f"            {analyzed.verdict.verdict}, confidence "
            f"{analyzed.verdict.confidence} — {grounding} ({time.time() - step:.0f}s)")

        say("      [4/5] auditing...")
        step = time.time()
        try:
            outcome = reflect_stage.reflect(analyzed, config=config)
        except LLMError as exc:
            reason = str(exc).splitlines()[0]
            say(f"            FAILED: {reason}")
            skipped.append((candidate.symbol, f"audit failed — {reason}"))
            continue
        say(f"            {outcome.reflection.outcome}, confidence "
            f"{outcome.original_confidence} -> {outcome.final_confidence} "
            f"({time.time() - step:.0f}s)")
        if outcome.overrides:
            say(f"            !! reflector overruled: {outcome.overrides[0]}")

        reflected.append(outcome)

    run = PipelineRun(
        as_of=datetime.now(timezone.utc),
        screen=screen_result,
        reflected=tuple(reflected),
        model=config.model,
        provider=config.provider,
        skipped=tuple(skipped),
        duration_seconds=time.time() - started,
    )

    say("\n" + "=" * 78)
    say("[5/5] Writing the report (no LLM — every figure comes from the data layer).")

    markdown = report_stage.render(run)
    destination = _write(markdown, args.out, run.as_of)

    if destination is None:
        print(markdown)
    else:
        say(f"      {destination}")

    _print_summary(run, say)
    return 0 if reflected else 1


def _write(markdown: str, out: str | None, as_of: datetime) -> Path | None:
    """Write the report, or return None if it should go to stdout instead."""
    if out == "-":
        return None
    if out:
        destination = Path(out)
    else:
        REPORTS_DIR.mkdir(exist_ok=True)
        destination = REPORTS_DIR / report_stage.default_filename(as_of)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    return destination


def _print_summary(run: PipelineRun, say) -> None:
    say()
    say("=" * 78)
    say("SUMMARY")
    say("=" * 78)

    if not run.reflected:
        say("  Nothing completed the pipeline.")
        for symbol, why in run.skipped:
            say(f"    {symbol}: {why}")
        return

    for r in report_stage.rank(run.reflected):
        say(
            f"  {r.symbol:12} {report_stage.VERDICT_LABEL.get(r.final_verdict, '?'):22}"
            f" {r.reflection.outcome:7} conf {r.original_confidence:>3} -> "
            f"{r.final_confidence:>3}"
            + ("  !! overruled" if r.overrides else "")
        )

    outcomes = [r.reflection.outcome for r in run.reflected]
    say(f"\n  {outcomes.count('KEEP')} kept, {outcomes.count('DEMOTE')} demoted, "
        f"{outcomes.count('REJECT')} rejected.")
    say(f"  {run.llm_calls} LLM calls, {run.total_tokens:,} tokens, "
        f"{run.duration_seconds / 60:.0f} min.")
    if run.skipped:
        say(f"  {len(run.skipped)} candidate(s) skipped after a failure — see the "
            f"report appendix.")
    say("\n  Idea generation, not investment advice. Read the figures, not just the "
        "verdicts.")
    say()


if __name__ == "__main__":
    raise SystemExit(main())
