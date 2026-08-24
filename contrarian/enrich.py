"""Stage 2 — turn screen survivors into everything the model may reason from.

Deterministic. No LLM call here, and the arithmetic is the whole point.

Every comparison the model might otherwise compute is resolved in Python and
handed over as a finished sentence: not "P/E 17.0, sector median 69.2" but "P/E
17.0 versus a Consumer sector median of 69.2 — a 75% discount". The Phase 4
spike showed exactly why. Given the raw pair, qwen3:14b wrote "P/E 17.0 is 69.2%
below sector median", folding a real input number into invented arithmetic.
Given the discount pre-computed, it reported it correctly.

That failure is the reason design rule 1 exists, and it is nastier than an
invented figure: 69.2 *is* in the input, so a naive "did this number come from
the source" check passes it. Removing the model's need to calculate is the
defence that actually works.

News is attached here too — the only signal that identifies the temporary
factor, and the reason `news_rss.py` exists.
"""

from __future__ import annotations

from .data.source import MarketDataSource, NewsItem
from .schema import Candidate, EnrichedCandidate, Fact

# How many headlines reach the prompt. The RSS feed returns ~94 per ticker; most
# are routine coverage, and 15 recent ones is enough to establish what happened
# without burying the figures in a wall of text the model will skim.
MAX_HEADLINES = 15

# Headlines older than this are dropped. The screen is looking for a *temporary*
# dislocation, so a story from eight months ago is not evidence about the current
# one — and stale news is a good way to talk the model into a narrative.
MAX_HEADLINE_AGE_DAYS = 90


def enrich(
    candidate: Candidate,
    source: MarketDataSource,
    *,
    max_headlines: int = MAX_HEADLINES,
    refresh: bool = False,
) -> EnrichedCandidate:
    """Build the complete, self-contained input for one candidate's analysis."""
    headlines = _fetch_headlines(candidate, source, max_headlines, refresh)
    return EnrichedCandidate(
        candidate=candidate,
        facts=tuple(build_facts(candidate)),
        headlines=tuple(headlines),
        news_source=getattr(source, "name", "unknown"),
    )


def enrich_all(
    candidates: tuple[Candidate, ...],
    source: MarketDataSource,
    *,
    max_headlines: int = MAX_HEADLINES,
    refresh: bool = False,
) -> tuple[EnrichedCandidate, ...]:
    return tuple(
        enrich(c, source, max_headlines=max_headlines, refresh=refresh) for c in candidates
    )


def _fetch_headlines(
    candidate: Candidate,
    source: MarketDataSource,
    limit: int,
    refresh: bool,
) -> list[NewsItem]:
    """Recent headlines, newest first.

    A news failure is not fatal. The figures still support an analysis, and the
    prompt says plainly when no headlines were available — which is far better
    than the model filling the silence with what it remembers about the company.
    """
    from .data.source import DataSourceError

    try:
        items = source.news(
            candidate.symbol,
            company_name=candidate.name,
            limit=limit * 3,
            refresh=refresh,
        )
    except DataSourceError:
        return []

    fresh = [
        item
        for item in items
        if item.age_days is None or item.age_days <= MAX_HEADLINE_AGE_DAYS
    ]
    fresh.sort(key=lambda i: i.age_days if i.age_days is not None else 1e9)
    return fresh[:limit]


def build_facts(candidate: Candidate) -> list[Fact]:
    """The figure sheet: every number the model gets, with the maths already done.

    Each `Fact` carries the numbers appearing in its own text, so the grounding
    check downstream is a set membership test rather than a re-derivation.
    """
    f, p, peer = candidate.fundamentals, candidate.prices, candidate.peer
    facts: list[Fact] = []

    def add(label: str, text: str, *numbers: float | None) -> None:
        facts.append(Fact(label, text, tuple(n for n in numbers if n is not None)))

    if f.price is not None:
        add("Price", f"Rs {f.price:,.2f}", f.price)

    if f.market_cap is not None:
        add("Market cap", f"Rs {_crore(f.market_cap)} crore", f.market_cap / 1e7)

    # --- how depressed, stated three ways so none of them needs computing ---
    if f.drawdown_from_high is not None and f.fifty_two_week_high is not None:
        add(
            "Drawdown",
            f"down {f.drawdown_from_high * 100:.1f}% from its 52-week high of "
            f"Rs {f.fifty_two_week_high:,.2f}",
            f.drawdown_from_high * 100,
            f.fifty_two_week_high,
        )
    if f.position_in_52w_range is not None and f.fifty_two_week_low is not None:
        add(
            "Position in range",
            f"sits at {f.position_in_52w_range * 100:.0f}% of its 52-week range "
            f"(low Rs {f.fifty_two_week_low:,.2f})",
            f.position_in_52w_range * 100,
            f.fifty_two_week_low,
        )

    for label, value, window in (
        ("1-month return", p.return_1m, "1 month"),
        ("3-month return", p.return_3m, "3 months"),
        ("6-month return", p.return_6m, "6 months"),
        ("12-month return", p.return_12m, "12 months"),
    ):
        if value is not None:
            direction = "up" if value >= 0 else "down"
            add(label, f"{direction} {abs(value) * 100:.1f}% over {window}", value * 100)

    # --- valuation, with the peer comparison resolved ---
    multiple = getattr(f, peer.metric, None)
    metric_name = "P/E" if peer.metric == "trailing_pe" else "price/book"
    if multiple is not None and peer.median:
        gap = (peer.median - multiple) / peer.median * 100
        relation = (
            f"a {gap:.0f}% discount to" if gap >= 0 else f"a {abs(gap):.0f}% premium to"
        )
        basis = (
            f"the {peer.label} median of {peer.median:.1f}"
            if not peer.is_fallback
            else f"the index-wide median of {peer.median:.1f} "
            f"(its own sector has too few members to compare against)"
        )
        add(
            "Valuation vs peers",
            f"{metric_name} of {multiple:.1f} — {relation} {basis}",
            multiple,
            peer.median,
            abs(gap),
        )
    elif multiple is not None:
        add("Valuation", f"{metric_name} of {multiple:.1f}", multiple)

    for label, value, fmt in (
        ("P/E (trailing)", f.trailing_pe, "{:.1f}"),
        ("P/E (forward)", f.forward_pe, "{:.1f}"),
        ("Price/book", f.price_to_book, "{:.2f}"),
        ("EV/EBITDA", f.ev_to_ebitda, "{:.1f}"),
    ):
        if value is not None:
            add(label, fmt.format(value), value)

    # --- business health ---
    if f.free_cash_flow is not None:
        sign = "positive" if f.free_cash_flow > 0 else "negative"
        add(
            "Free cash flow",
            f"{sign}, Rs {_crore(abs(f.free_cash_flow))} crore (most recent annual)",
            abs(f.free_cash_flow) / 1e7,
        )
    if f.operating_cash_flow is not None:
        add(
            "Operating cash flow",
            f"Rs {_crore(f.operating_cash_flow)} crore (most recent annual)",
            f.operating_cash_flow / 1e7,
        )
    if f.debt_to_equity is not None:
        # Converted from Yahoo's percentage to the multiple people actually quote,
        # so the model never sees the ambiguous raw number.
        add(
            "Leverage",
            f"debt/equity of {f.debt_to_equity / 100:.2f}x",
            f.debt_to_equity / 100,
        )
    for label, value in (
        ("Return on equity", f.return_on_equity),
        ("Net profit margin", f.profit_margin),
        ("Revenue growth (yoy)", f.revenue_growth),
        ("Earnings growth (yoy)", f.earnings_growth),
        ("Dividend yield", f.dividend_yield / 100 if f.dividend_yield else None),
    ):
        if value is not None:
            add(label, f"{value * 100:.1f}%", value * 100)

    if f.sector:
        facts.append(Fact("Sector", f"{f.sector} ({f.industry or 'n/a'})"))

    return facts


def _crore(rupees: float) -> str:
    """Format in crore, the unit Indian filings and headlines actually use."""
    return f"{rupees / 1e7:,.0f}"
