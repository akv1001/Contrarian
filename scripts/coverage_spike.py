#!/usr/bin/env python3
"""Phase 2 spike — how good is free Indian market data, actually?

This is the deliverable of Phase 2, not a test. It answers the question that
decides what Phase 3 can be built out of: **which fields are reliably present
across all 50 Nifty constituents, and which are too sparse to filter on?**

Building a screen on a field that is present for 30 of 50 names does not fail
loudly. It quietly halves the universe, and the half that gets dropped is not
random — it correlates with exactly the unloved, under-covered names a
contrarian screen is supposed to surface.

Run it:
    .venv/bin/python scripts/coverage_spike.py

Options:
    --limit N     Only the first N tickers. Good for a quick check.
    --refresh     Ignore the cache and refetch everything.
    --no-news     Skip the news pass (the slow half — it is rate-limited).
    --verbose     Per-ticker detail rather than just the summary.

First run takes a few minutes and is network-bound. Later runs read the disk
cache and finish in seconds.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrarian.data import cache
from contrarian.data.news_rss import GoogleNewsRSS
from contrarian.data.source import NUMERIC_FIELDS, SCREEN_FIELDS, DataSourceError
from contrarian.data.yahoo import YahooSource
from contrarian.universe import NIFTY_50, NIFTY_50_AS_OF

# Below this, a field cannot carry a screen filter on its own.
GOOD_COVERAGE = 0.95
USABLE_COVERAGE = 0.80


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nifty 50 data coverage spike.")
    p.add_argument("--limit", type=int, default=0, help="only the first N tickers")
    p.add_argument("--refresh", action="store_true", help="bypass the cache")
    p.add_argument("--no-news", action="store_true", help="skip the news pass")
    p.add_argument("--verbose", action="store_true", help="per-ticker detail")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    constituents = NIFTY_50[: args.limit] if args.limit else NIFTY_50

    print(f"Nifty 50 coverage spike — list dated {NIFTY_50_AS_OF}")
    print(f"{len(constituents)} tickers, cache {'BYPASSED' if args.refresh else 'enabled'}")
    print("=" * 78)

    source = YahooSource()
    present: Counter[str] = Counter()
    provenance: dict[str, Counter[str]] = {f: Counter() for f in NUMERIC_FIELDS}
    # field -> sector -> (missing, total). A field can be 85% covered because the
    # scraper is flaky, or because it is meaningless for a whole sector. Those
    # call for completely different Phase 3 responses, and the aggregate
    # percentage cannot tell them apart.
    by_sector: dict[str, dict[str, list[int]]] = {f: {} for f in NUMERIC_FIELDS}
    sector_totals: Counter[str] = Counter()
    fundamental_failures: list[tuple[str, str]] = []
    price_failures: list[tuple[str, str]] = []
    screenable = 0
    ok_count = 0

    print("\nFUNDAMENTALS + PRICES")
    print("-" * 78)

    for c in constituents:
        try:
            f = source.fundamentals(c.symbol, refresh=args.refresh)
        except DataSourceError as exc:
            fundamental_failures.append((c.symbol, str(exc).split("\n")[0]))
            print(f"  {c.symbol:12} FAILED  {str(exc).splitlines()[0][:56]}")
            continue

        ok_count += 1
        sector_totals[c.sector] += 1
        for name in NUMERIC_FIELDS:
            bucket = by_sector[name].setdefault(c.sector, [0, 0])
            bucket[1] += 1
            if getattr(f, name) is not None:
                present[name] += 1
                provenance[name][f.provenance.get(name, "?")] += 1
            else:
                bucket[0] += 1

        missing = f.missing()
        if not missing:
            screenable += 1

        try:
            stats = source.price_stats(c.symbol, refresh=args.refresh)
            ret12 = stats.return_12m
        except DataSourceError as exc:
            price_failures.append((c.symbol, str(exc).splitlines()[0]))
            ret12 = None

        if args.verbose or missing:
            flag = "OK   " if not missing else "GAPS "
            print(
                f"  {c.symbol:12} {flag} "
                f"12m={_pct(ret12):>8}  "
                f"pe={_num(f.trailing_pe):>7}  "
                f"fcf={_short(f.free_cash_flow):>8}  "
                + (f"missing={','.join(missing)}" if missing else "")
            )

    _report_fields(present, provenance, ok_count)
    _report_gaps_by_sector(present, by_sector, ok_count)

    print("\nSCREENABILITY")
    print("-" * 78)
    print(f"  Tickers with every SCREEN_FIELDS value present: {screenable}/{ok_count}")
    if fundamental_failures:
        print(f"  Fundamentals fetch failed: {len(fundamental_failures)}")
        for sym, msg in fundamental_failures:
            print(f"    {sym:12} {msg[:58]}")
    if price_failures:
        print(f"  Price history failed: {len(price_failures)}")
        for sym, msg in price_failures:
            print(f"    {sym:12} {msg[:58]}")

    if not args.no_news:
        _news_pass(constituents, refresh=args.refresh, verbose=args.verbose)

    print("\nCACHE")
    print("-" * 78)
    for namespace, count in sorted(cache.stats().items()):
        print(f"  {namespace:22} {count:>4} files")

    _verdict(present, ok_count, screenable, fundamental_failures)
    return 1 if fundamental_failures else 0


def _report_fields(present: Counter[str], provenance: dict[str, Counter[str]], total: int) -> None:
    """Print the coverage matrix: the table this whole script exists to produce."""
    print("\nFIELD COVERAGE")
    print("-" * 78)
    print(f"  {'field':24} {'present':>9}  {'%':>6}  screen?  source")
    print("  " + "-" * 74)

    for name in NUMERIC_FIELDS:
        count = present[name]
        pct = count / total if total else 0.0
        mark = "yes" if name in SCREEN_FIELDS else "-"
        sources = ", ".join(f"{k}:{v}" for k, v in provenance[name].most_common(2))
        warn = " " if pct >= GOOD_COVERAGE else ("!" if pct >= USABLE_COVERAGE else "X")
        print(
            f" {warn}{name:24} {count:>4}/{total:<4} {pct:>6.0%}  {mark:^7}  {sources[:26]}"
        )
    print("\n  legend:  (blank) >=95% usable    ! 80-95% risky    X <80% do not screen on it")


def _report_gaps_by_sector(
    present: Counter[str],
    by_sector: dict[str, dict[str, list[int]]],
    total: int,
) -> None:
    """For each incompletely-covered screen field, show where the holes are.

    The distinction this draws is the useful one. A field missing uniformly at
    random is a scraper problem and you handle it by tolerating absence. A field
    missing for every company in one sector is a *modelling* fact — EV/EBITDA is
    not a defined quantity for a bank — and the right response is a different
    filter for that sector, not a looser one for everybody.
    """
    incomplete = [f for f in SCREEN_FIELDS if total and present[f] / total < 1.0]
    if not incomplete:
        return

    print("\nWHERE THE GAPS ARE  (screen fields with any missing values)")
    print("-" * 78)
    for name in incomplete:
        holes = {s: c for s, (c, _) in by_sector[name].items() if c}
        wholly = [s for s, (c, n) in by_sector[name].items() if c and c == n]
        print(f"  {name}  —  missing for {sum(holes.values())}/{total}")
        print(f"      by sector: {', '.join(f'{s} {c}' for s, c in sorted(holes.items()))}")
        if wholly:
            print(
                f"      absent for EVERY name in: {', '.join(sorted(wholly))} "
                f"— structural, not a data gap. Use a sector-specific filter."
            )


def _news_pass(constituents, *, refresh: bool, verbose: bool) -> None:
    """Compare the two news feeds on the axis that matters: on-topic volume.

    Relevance is approximated by asking whether the company's name appears in the
    headline. Crude, and it undercounts legitimate stories that refer to a
    company obliquely — but it is symmetric across both feeds, so the comparison
    holds even if the absolute numbers are pessimistic.
    """
    print("\nNEWS COVERAGE  (yahoo vs google news rss)")
    print("-" * 78)
    print(f"  {'ticker':12} {'yahoo':>12} {'relevant':>9}  {'rss':>6} {'relevant':>9}  {'newest':>8}")
    print("  " + "-" * 74)

    yahoo, rss = YahooSource(), GoogleNewsRSS()
    totals = {"y": 0, "y_rel": 0, "r": 0, "r_rel": 0}
    rss_empty: list[str] = []

    for c in constituents:
        y_items, r_items = [], []
        try:
            y_items = yahoo.news(c.symbol, limit=50, refresh=refresh)
        except DataSourceError:
            pass
        try:
            r_items = rss.news(c.symbol, company_name=c.name, limit=100, refresh=refresh)
        except DataSourceError as exc:
            print(f"  {c.symbol:12} rss failed: {str(exc).splitlines()[0][:48]}")

        y_rel = sum(1 for i in y_items if _mentions(i.title, c.name))
        r_rel = sum(1 for i in r_items if _mentions(i.title, c.name))
        totals["y"] += len(y_items)
        totals["y_rel"] += y_rel
        totals["r"] += len(r_items)
        totals["r_rel"] += r_rel
        if not r_items:
            rss_empty.append(c.symbol)

        ages = [i.age_days for i in r_items if i.age_days is not None]
        newest = f"{min(ages):.0f}d" if ages else "-"

        if verbose or not r_items or r_rel < 3:
            print(
                f"  {c.symbol:12} {len(y_items):>12} {y_rel:>9}  "
                f"{len(r_items):>6} {r_rel:>9}  {newest:>8}"
            )

    n = len(constituents)
    print("  " + "-" * 74)
    print(
        f"  {'TOTAL':12} {totals['y']:>12} {totals['y_rel']:>9}  "
        f"{totals['r']:>6} {totals['r_rel']:>9}"
    )
    print(
        f"\n  per ticker:  yahoo {totals['y'] / n:.1f} items "
        f"({_share(totals['y_rel'], totals['y'])} name-matched)   "
        f"rss {totals['r'] / n:.1f} items "
        f"({_share(totals['r_rel'], totals['r'])} name-matched)"
    )
    if rss_empty:
        print(f"  RSS returned nothing for: {', '.join(rss_empty)}")


def _verdict(present: Counter[str], total: int, screenable: int, failures: list) -> None:
    """State plainly what Phase 3 may and may not screen on."""
    print("\n" + "=" * 78)
    print("VERDICT — what Phase 3 can screen on")
    print("=" * 78)

    if not total:
        print("  No tickers fetched. Nothing to conclude.")
        return

    solid = [f for f in SCREEN_FIELDS if present[f] / total >= GOOD_COVERAGE]
    risky = [f for f in SCREEN_FIELDS if USABLE_COVERAGE <= present[f] / total < GOOD_COVERAGE]
    unusable = [f for f in SCREEN_FIELDS if present[f] / total < USABLE_COVERAGE]

    print(f"\n  Safe to filter on ({len(solid)}):")
    print(f"    {', '.join(solid) or '(none)'}")
    if risky:
        print(f"\n  Usable but must tolerate absence ({len(risky)}):")
        print(f"    {', '.join(risky)}")
    if unusable:
        print(f"\n  Do NOT build a filter on these ({len(unusable)}):")
        print(f"    {', '.join(unusable)}")

    print(f"\n  {screenable}/{total} tickers have a complete screen record.")
    if failures:
        print(f"  {len(failures)} ticker(s) failed outright — check universe.py.")
    print()


def _mentions(title: str, company_name: str) -> bool:
    """Does this headline name the company?

    Matches on the distinctive part of the name, ignoring corporate suffixes and
    words too generic to be evidence — "Tata Consumer Products" should not match
    a story purely on the word "Products".
    """
    stop = {
        "limited", "ltd", "india", "industries", "corporation", "company",
        "and", "of", "services", "products", "enterprises", "motors", "bank",
        "finance", "financial", "insurance", "life", "the",
    }
    haystack = title.lower()
    tokens = [t for t in company_name.lower().split() if t not in stop and len(t) > 2]
    return any(t in haystack for t in tokens) if tokens else company_name.lower() in haystack


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:+.1%}"


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def _short(v: float | None) -> str:
    """Format a rupee figure in crore — the unit Indian filings actually use."""
    if v is None:
        return "-"
    return f"{v / 1e7:,.0f}cr"


def _share(part: int, whole: int) -> str:
    return "-" if not whole else f"{part / whole:.0%}"


if __name__ == "__main__":
    raise SystemExit(main())
