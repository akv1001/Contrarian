"""Stage 1 — the deterministic screen. No LLM, by design.

Answers one narrow question per ticker: *is this cheap, beaten down, and still
apparently a sound business?* Whether the cheapness is temporary or deserved is
Phase 4's job. This stage exists to get the universe from 51 names to a handful
worth spending tokens on, and to hand every figure it used to the stages that
follow.

Three gates, in order:

1. **Eligibility** — does the source actually have the fields this ticker's
   ruleset needs? Absent data excludes. It never passes by default.
2. **Depressed** — is the price genuinely beaten down, not merely mid-range?
3. **Quality and value** — positive cash generation, contained leverage, real
   returns on equity, and a valuation below its peers.

Two findings from the Phase 2 spike are load-bearing here:

- **Banks and insurers have no EV/EBITDA and no meaningful debt/equity.** Those
  fields are absent for every Financials name, and it is a modelling fact, not a
  data gap. Financials get their own ruleset — priced on book value, judged on
  return on equity. Applying the general ruleset would reject the entire largest
  sector in the index on missing data.
- **Yahoo reports `debtToEquity` as a percentage**, not a ratio: 36.65 means
  0.37x. Index median is 29.4 and the max is 315. A ceiling written as `2.0`
  would silently reject all 51 names, and the screen would look merely strict.
"""

from __future__ import annotations

import statistics
from dataclasses import replace
from datetime import datetime, timezone

from .data.source import DataSourceError, Fundamentals, MarketDataSource, PriceStats
from .schema import Candidate, FilterResult, PeerBasis, ScreenResult
from .universe import NIFTY_50, Constituent

# --- Depressed-price gate ---------------------------------------------------
# Both conditions must hold. Index p25 for range position is 0.21 and median is
# 0.53, so 0.40 takes roughly the bottom third. The drawdown floor is what stops
# a stock in a tight 52-week band qualifying just by sitting at the bottom of a
# range that never moved — position alone says nothing about magnitude.
MAX_RANGE_POSITION = 0.40
MIN_DRAWDOWN = 0.15

# --- Quality gates ----------------------------------------------------------
# Percent, not a ratio — see the module docstring. 150 is 1.5x, which sits
# between the index p75 (95) and max (315).
MAX_DEBT_TO_EQUITY_PCT = 150.0
MIN_RETURN_ON_EQUITY = 0.10  # Index p25 is 0.11, median 0.16.
MIN_PROFIT_MARGIN = 0.0

# --- Liquidity --------------------------------------------------------------
# Near-vacuous for the Nifty 50, where the thinnest name still trades ~300k
# shares a day. Kept as a guard for the day this universe widens, not because it
# bites today.
MIN_AVERAGE_VOLUME = 250_000

# --- Peer comparison --------------------------------------------------------
# Below this, a sector median is not a peer group. Nifty 50 has single-member
# sectors (Telecom, Infrastructure, Conglomerate) where the median is the stock
# itself and "cheaper than the median" is a tautology that always passes.
MIN_PEERS_FOR_SECTOR_MEDIAN = 4

# Score weights. Depression and relative cheapness are the contrarian thesis;
# quality is a tiebreak, deliberately the smallest, because a high-quality
# business that is not actually cheap is not what this pipeline is looking for.
WEIGHT_DEPRESSION = 0.40
WEIGHT_VALUATION = 0.40
WEIGHT_QUALITY = 0.20

# ROE at which the quality component saturates. Index p75 is 0.24, so 0.30 means
# only genuinely high-return businesses reach full marks.
QUALITY_ROE_CEILING = 0.30

# How much of the valuation discount survives when the peer group had to fall
# back to the whole index.
#
# This exists because the first full run got it wrong in an instructive way.
# Energy has three members, so COALINDIA and ONGC were compared against the
# index-wide P/E median of 29.7 and scored a ~75% "discount" on multiples of 8.0
# and 6.8 — enough to rank them 2nd and 3rd. But commodity producers trade on
# single-digit multiples everywhere; that gap is a sector characteristic, not
# evidence of a dislocation. Comparing a miner's P/E to a paint company's is not
# a valuation argument.
#
# The honest fix would be a wider peer universe than 51 stocks. Absent that, a
# cross-sector comparison is kept as weak evidence — it still gates, but it
# cannot dominate the ranking.
FALLBACK_DISCOUNT_DAMPING = 0.5

FINANCIALS = "Financials"


def is_financial(sector: str) -> bool:
    """Banks, NBFCs, and insurers, which need the alternative ruleset."""
    return sector == FINANCIALS


def valuation_metric(sector: str) -> str:
    """Which multiple this sector is judged on.

    Price-to-book for financials: earnings are cyclical and provision-driven,
    the balance sheet *is* the business, and P/B is what the market actually
    prices banks on. Trailing P/E for everyone else.
    """
    return "price_to_book" if is_financial(sector) else "trailing_pe"


def required_fields(sector: str) -> tuple[str, ...]:
    """Fields a ticker must have before it can be screened at all.

    Sector-specific, and short on purpose: every field here is one a missing
    value will exclude on, so anything not genuinely needed is a filter that
    rejects for the wrong reason.
    """
    common = (
        "price",
        "fifty_two_week_high",
        "fifty_two_week_low",
        "average_volume",
        "return_on_equity",
        "profit_margin",
    )
    if is_financial(sector):
        return common + ("price_to_book",)
    return common + ("trailing_pe", "free_cash_flow", "debt_to_equity")


def peer_medians(
    fundamentals: dict[str, Fundamentals],
    sectors: dict[str, str],
) -> dict[str, PeerBasis]:
    """Compute the valuation median each ticker will be compared against.

    Medians rather than means: `ev_to_ebitda` runs to 1260 and `trailing_pe` to
    713 in this universe, and one such outlier drags a mean far enough to change
    outcomes.

    The candidate is included in its own peer median. With 8-12 names a single
    member barely moves the median, and one stable published figure per sector is
    worth more for auditing than a marginally more correct leave-one-out value
    that differs for every ticker.
    """
    by_metric_sector: dict[tuple[str, str], list[float]] = {}
    by_metric_all: dict[str, list[float]] = {}

    for symbol, f in fundamentals.items():
        sector = sectors[symbol]
        metric = valuation_metric(sector)
        value = getattr(f, metric)
        if value is None or value <= 0:  # A negative multiple is not a cheap one.
            continue
        by_metric_sector.setdefault((metric, sector), []).append(value)
        by_metric_all.setdefault(metric, []).append(value)

    out: dict[str, PeerBasis] = {}
    for symbol in fundamentals:
        sector = sectors[symbol]
        metric = valuation_metric(sector)
        peers = by_metric_sector.get((metric, sector), [])

        if len(peers) >= MIN_PEERS_FOR_SECTOR_MEDIAN:
            out[symbol] = PeerBasis(
                label=sector,
                metric=metric,
                median=statistics.median(peers),
                peer_count=len(peers),
                is_fallback=False,
            )
            continue

        universe_peers = by_metric_all.get(metric, [])
        out[symbol] = PeerBasis(
            label=f"Nifty 50 ({sector} has only {len(peers)})",
            metric=metric,
            median=statistics.median(universe_peers) if universe_peers else None,
            peer_count=len(universe_peers),
            is_fallback=True,
        )
    return out


def evaluate(
    constituent: Constituent,
    fundamentals: Fundamentals,
    prices: PriceStats,
    peer: PeerBasis,
) -> Candidate:
    """Run every filter against one ticker and score it.

    All filters run even after one fails. Short-circuiting would be marginally
    faster and would throw away the thing that makes a rejection legible — "why
    did HDFCBANK not make it" is a question you ask constantly while tuning
    thresholds, and "it failed at filter 2" is not an answer.
    """
    sector = constituent.sector
    filters: list[FilterResult] = []

    missing = [name for name in required_fields(sector) if getattr(fundamentals, name) is None]
    filters.append(
        FilterResult(
            name="data_complete",
            passed=not missing,
            detail=(
                f"all {len(required_fields(sector))} required fields present"
                if not missing
                else f"missing {', '.join(missing)} — cannot screen, excluded rather than assumed"
            ),
        )
    )

    filters.extend(_depressed_filters(fundamentals))
    filters.append(_liquidity_filter(fundamentals))
    filters.extend(
        _financial_quality_filters(fundamentals)
        if is_financial(sector)
        else _general_quality_filters(fundamentals)
    )
    filters.append(_valuation_filter(fundamentals, peer))

    candidate = Candidate(
        symbol=constituent.symbol,
        name=constituent.name,
        sector=sector,
        fundamentals=fundamentals,
        prices=prices,
        peer=peer,
        filters=tuple(filters),
    )
    return _scored(candidate)


def _depressed_filters(f: Fundamentals) -> list[FilterResult]:
    position, drawdown = f.position_in_52w_range, f.drawdown_from_high
    return [
        FilterResult(
            name="near_52w_low",
            passed=position is not None and position <= MAX_RANGE_POSITION,
            detail=(
                f"sits at {position:.0%} of its 52-week range "
                f"(need <= {MAX_RANGE_POSITION:.0%})"
                if position is not None
                else "52-week range unavailable"
            ),
        ),
        FilterResult(
            name="meaningful_drawdown",
            passed=drawdown is not None and drawdown >= MIN_DRAWDOWN,
            detail=(
                f"down {drawdown:.1%} from its 52-week high "
                f"(need >= {MIN_DRAWDOWN:.0%})"
                if drawdown is not None
                else "52-week high unavailable"
            ),
        ),
    ]


def _liquidity_filter(f: Fundamentals) -> FilterResult:
    volume = f.average_volume
    return FilterResult(
        name="liquidity",
        passed=volume is not None and volume >= MIN_AVERAGE_VOLUME,
        detail=(
            f"{volume:,.0f} avg shares/day (need >= {MIN_AVERAGE_VOLUME:,})"
            if volume is not None
            else "average volume unavailable"
        ),
    )


def _general_quality_filters(f: Fundamentals) -> list[FilterResult]:
    """Quality gates for operating companies.

    Free cash flow is the one that matters most. A cheap multiple on a business
    that does not convert earnings to cash is the classic value trap, and it is
    exactly the pattern the Phase 1 smoke test asked the model to catch.
    """
    fcf, debt, roe, margin = (
        f.free_cash_flow,
        f.debt_to_equity,
        f.return_on_equity,
        f.profit_margin,
    )
    return [
        FilterResult(
            name="positive_fcf",
            passed=fcf is not None and fcf > 0,
            detail=(
                f"free cash flow {_crore(fcf)} (need > 0)"
                if fcf is not None
                else "free cash flow unavailable"
            ),
        ),
        FilterResult(
            name="leverage",
            passed=debt is not None and debt <= MAX_DEBT_TO_EQUITY_PCT,
            detail=(
                f"debt/equity {debt:.1f}% = {debt / 100:.2f}x "
                f"(need <= {MAX_DEBT_TO_EQUITY_PCT / 100:.1f}x)"
                if debt is not None
                else "debt/equity unavailable"
            ),
        ),
        *_returns_filters(roe, margin),
    ]


def _financial_quality_filters(f: Fundamentals) -> list[FilterResult]:
    """Quality gates for banks, NBFCs, and insurers.

    No free-cash-flow test and no leverage ceiling. For a bank, borrowing is the
    raw material rather than a risk signal, and "free cash flow" is an artefact
    of how financing flows are classified — AXISBANK shows -10,847cr and
    HDFCBANK +106,578cr, and neither number means what it would for a
    manufacturer. Return on equity carries the quality judgment instead.
    """
    return list(_returns_filters(f.return_on_equity, f.profit_margin))


def _returns_filters(roe: float | None, margin: float | None) -> tuple[FilterResult, ...]:
    return (
        FilterResult(
            name="return_on_equity",
            passed=roe is not None and roe >= MIN_RETURN_ON_EQUITY,
            detail=(
                f"ROE {roe:.1%} (need >= {MIN_RETURN_ON_EQUITY:.0%})"
                if roe is not None
                else "ROE unavailable"
            ),
        ),
        FilterResult(
            name="profitable",
            passed=margin is not None and margin > MIN_PROFIT_MARGIN,
            detail=(
                f"net margin {margin:.1%} (need > 0)"
                if margin is not None
                else "profit margin unavailable"
            ),
        ),
    )


def _valuation_filter(f: Fundamentals, peer: PeerBasis) -> FilterResult:
    value = getattr(f, peer.metric)
    if value is None or peer.median is None:
        return FilterResult(
            name="cheap_vs_peers",
            passed=False,
            detail=f"cannot compare: {peer.metric} or peer median unavailable",
        )
    if value <= 0:
        # Negative earnings produce no trailing P/E; Yahoo omits the field
        # entirely (TMPV). A non-positive multiple is not cheapness, it is the
        # absence of the thing being measured.
        return FilterResult(
            name="cheap_vs_peers",
            passed=False,
            detail=f"{peer.metric} is {value:.1f} — not a meaningful multiple",
        )
    return FilterResult(
        name="cheap_vs_peers",
        passed=value <= peer.median,
        detail=f"{peer.metric} {value:.1f} vs {peer.description}",
    )


def _scored(candidate: Candidate) -> Candidate:
    """Attach the rank components. Pure arithmetic on figures already fetched."""
    f, peer = candidate.fundamentals, candidate.peer

    position = f.position_in_52w_range
    depression = _clip(1.0 - position) if position is not None else 0.0

    value = getattr(f, peer.metric)
    if value is not None and value > 0 and peer.median:
        discount = _clip((peer.median - value) / peer.median)
        if peer.is_fallback:
            discount *= FALLBACK_DISCOUNT_DAMPING
    else:
        discount = 0.0

    roe = f.return_on_equity
    quality = _clip(roe / QUALITY_ROE_CEILING) if roe is not None else 0.0

    score = (
        WEIGHT_DEPRESSION * depression
        + WEIGHT_VALUATION * discount
        + WEIGHT_QUALITY * quality
    )
    return replace(
        candidate,
        depression=depression,
        valuation_discount=discount,
        quality=quality,
        score=score,
    )


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _crore(rupees: float) -> str:
    """Format in crore — the unit Indian filings and headlines actually use."""
    return f"{rupees / 1e7:,.0f}cr"


def run(
    source: MarketDataSource,
    constituents: tuple[Constituent, ...] = NIFTY_50,
    *,
    refresh: bool = False,
) -> ScreenResult:
    """Screen the universe and return survivors ranked best-first.

    Fetches everything before filtering anything, because the peer medians are a
    property of the whole universe and cannot be computed one ticker at a time.
    """
    fundamentals: dict[str, Fundamentals] = {}
    prices: dict[str, PriceStats] = {}
    sectors: dict[str, str] = {}
    errors: list[tuple[str, str]] = []

    for c in constituents:
        try:
            fundamentals[c.symbol] = source.fundamentals(c.symbol, refresh=refresh)
            prices[c.symbol] = source.price_stats(c.symbol, refresh=refresh)
            sectors[c.symbol] = c.sector
        except DataSourceError as exc:
            # A fetch failure is not a screen failure. Recorded separately so a
            # broken data layer cannot masquerade as a strict filter.
            errors.append((c.symbol, str(exc).splitlines()[0]))

    medians = peer_medians(fundamentals, sectors)

    passed: list[Candidate] = []
    rejected: list[Candidate] = []
    for c in constituents:
        if c.symbol not in fundamentals:
            continue
        candidate = evaluate(c, fundamentals[c.symbol], prices[c.symbol], medians[c.symbol])
        (passed if candidate.passed else rejected).append(candidate)

    by_score = lambda c: (-c.score, c.symbol)  # noqa: E731 — stable, ties by symbol
    return ScreenResult(
        as_of=datetime.now(timezone.utc),
        universe_size=len(constituents),
        passed=tuple(sorted(passed, key=by_score)),
        rejected=tuple(sorted(rejected, key=by_score)),
        errors=tuple(errors),
    )
