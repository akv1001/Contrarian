#!/usr/bin/env python3
"""Phase 3 — run the deterministic screen and show its work.

The Phase 3 gate is "survivors are plausibly cheap on inspection", which means
this script has to be readable enough to actually inspect. Every filter prints
the numbers it compared, so a surprising result can be traced to a threshold
rather than argued about.

Run it:
    .venv/bin/python scripts/run_screen.py

Options:
    --show-rejects N   Why the N closest misses failed. Where the tuning happens.
    --explain SYMBOL   Every filter for one ticker, pass or fail.
    --refresh          Refetch instead of reading the disk cache.

No LLM call, no network if the cache is warm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrarian import screen
from contrarian.data.news_rss import default_source
from contrarian.schema import Candidate, ScreenResult
from contrarian.universe import NIFTY_50, NIFTY_50_AS_OF


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Nifty 50 contrarian screen.")
    p.add_argument("--show-rejects", type=int, default=5, metavar="N",
                   help="show why the N closest misses failed (default 5)")
    p.add_argument("--explain", metavar="SYMBOL", help="full filter trace for one ticker")
    p.add_argument("--refresh", action="store_true", help="bypass the cache")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Contrarian screen — Nifty 50 as of {NIFTY_50_AS_OF}")
    print("=" * 78)
    print(_thresholds())

    result = screen.run(default_source(), NIFTY_50, refresh=args.refresh)

    if args.explain:
        return _explain(result, args.explain.upper())

    _print_survivors(result)
    if args.show_rejects:
        _print_rejects(result, args.show_rejects)
    _print_summary(result)
    return 0


def _thresholds() -> str:
    return (
        f"  depressed:  <= {screen.MAX_RANGE_POSITION:.0%} of 52w range "
        f"AND >= {screen.MIN_DRAWDOWN:.0%} off the high\n"
        f"  quality:    ROE >= {screen.MIN_RETURN_ON_EQUITY:.0%}, positive margin; "
        f"non-financials also need positive FCF\n"
        f"              and debt/equity <= {screen.MAX_DEBT_TO_EQUITY_PCT / 100:.1f}x\n"
        f"  value:      below the peer median multiple "
        f"(P/B for financials, P/E otherwise)\n"
        f"  score:      {screen.WEIGHT_DEPRESSION:.0%} depression + "
        f"{screen.WEIGHT_VALUATION:.0%} discount to peers + "
        f"{screen.WEIGHT_QUALITY:.0%} quality"
    )


def _print_survivors(result: ScreenResult) -> None:
    print("\nSURVIVORS")
    print("-" * 78)
    if not result.passed:
        print("  None. Either the market is expensive or a threshold is too tight —")
        print("  run with --show-rejects 10 to see which filter is doing the work.")
        return

    for rank, c in enumerate(result.passed, 1):
        f = c.fundamentals
        print(f"\n  {rank}. {c.symbol} — {c.name}  ({c.sector})")
        print(
            f"     score {c.score:.3f}"
            f"   [depression {c.depression:.2f} | discount {c.valuation_discount:.2f}"
            f" | quality {c.quality:.2f}]"
        )
        print(
            f"     {_money(f.price)}  "
            f"{_pct(f.drawdown_from_high)} off high  "
            f"{_pct(c.prices.return_12m, signed=True)} 12m  "
            f"{c.peer.metric} {_num(getattr(f, c.peer.metric))} vs peers {_num(c.peer.median)}"
        )
        for filt in c.filters:
            print(f"       . {filt.detail}")


def _print_rejects(result: ScreenResult, limit: int) -> None:
    """Closest misses first — sorted by how few filters they failed.

    This is the view that tells you whether a threshold is selecting for the
    thing you meant or just for the thing that is easy to measure.
    """
    print(f"\nCLOSEST MISSES  (top {limit} of {len(result.rejected)})")
    print("-" * 78)
    ordered = sorted(result.rejected, key=lambda c: (len(c.failures), -c.score))
    for c in ordered[:limit]:
        print(f"  {c.symbol:12} {c.sector:14} failed {len(c.failures)}: {c.failure_summary}")
        for filt in c.failures:
            print(f"       x {filt.detail}")


def _explain(result: ScreenResult, symbol: str) -> int:
    everything = list(result.passed) + list(result.rejected)
    match = next((c for c in everything if c.symbol == symbol), None)
    if match is None:
        known = ", ".join(sorted(c.symbol for c in everything))
        print(f"\n  {symbol!r} was not screened.\n\n  Known: {known}\n")
        return 1

    print(f"\n{match.symbol} — {match.name}  ({match.sector})")
    print("-" * 78)
    print(f"  verdict: {'PASSED' if match.passed else 'REJECTED'}   score {match.score:.3f}")
    print(f"  peers:   {match.peer.description}\n")
    for filt in match.filters:
        print(f"  {filt}")

    print("\n  figures the screen used (this is what Phase 4 will be given):")
    for key, value in match.figures().items():
        if value is not None:
            print(f"    {key:24} {value:,.4g}")
    return 0


def _print_summary(result: ScreenResult) -> None:
    print("\n" + "=" * 78)
    print(
        f"  {len(result.passed)} passed, {len(result.rejected)} rejected, "
        f"{len(result.errors)} fetch errors, {result.universe_size} in universe"
    )
    for symbol, message in result.errors:
        print(f"    ! {symbol}: {message[:60]}")

    from collections import Counter

    reasons: Counter[str] = Counter()
    for c in result.rejected:
        for filt in c.failures:
            reasons[filt.name] += 1
    if reasons:
        print("\n  which filter rejected how many:")
        for name, count in reasons.most_common():
            print(f"    {name:22} {count:>3}")
    print()


def _pct(v: float | None, *, signed: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:+.1%}" if signed else f"{v:.1%}"


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def _money(v: float | None) -> str:
    return "-" if v is None else f"Rs {v:,.0f}"


if __name__ == "__main__":
    raise SystemExit(main())
