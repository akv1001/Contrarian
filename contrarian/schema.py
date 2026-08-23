"""The records that move between stages.

`Candidate` is the important one. Design rule 1 says the model never does
arithmetic and never recalls a figure from training — every number reaching it
comes from the data layer. `Candidate.figures()` is the mechanism: it returns the
complete set of numbers the screen used, and Phase 4's prompt is built from that
dict and nothing else. Any figure appearing in a verdict that is not a value in
this dict is, by definition, a hallucination.

That is also why every filter records the values it compared rather than just a
pass/fail bit. A screen you cannot audit is a screen you cannot debug, and the
expensive error here is a plausible-looking wrong number surviving a skim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .data.source import Fundamentals, PriceStats


@dataclass(frozen=True)
class FilterResult:
    """One filter's outcome, with the numbers that produced it.

    `detail` is written to be readable on its own — "P/E 15.9 <= Financials
    median 26.1" — because it ends up in the report and in the Phase 4 prompt,
    where "valuation: passed" would be useless.
    """

    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass(frozen=True)
class PeerBasis:
    """What "cheap relative to peers" was measured against.

    Carries `is_fallback` because it changes how much the comparison is worth.
    Several Nifty 50 sectors have one or two members, where a sector median is
    either the stock itself or meaningless. In those cases the screen compares
    against the whole index and says so, rather than quietly producing a
    tautology.
    """

    label: str
    metric: str
    median: float | None
    peer_count: int
    is_fallback: bool

    @property
    def description(self) -> str:
        if self.median is None:
            return f"no {self.metric} median available"
        basis = f"{self.label} median {self.metric} {self.median:.1f} (n={self.peer_count})"
        return f"{basis}, index-wide fallback" if self.is_fallback else basis


@dataclass(frozen=True)
class Candidate:
    """One company as the screen saw it: every input, every filter, and a rank.

    Kept whether it passed or failed. The rejections are as interesting as the
    survivors when you are deciding whether a threshold is doing real work — and
    they are what `--show-rejects` prints.
    """

    symbol: str
    name: str
    sector: str
    fundamentals: Fundamentals
    prices: PriceStats
    peer: PeerBasis
    filters: tuple[FilterResult, ...]

    # Score components, each 0..1 and each kept separately so a rank can be
    # explained rather than just asserted.
    depression: float = 0.0
    valuation_discount: float = 0.0
    quality: float = 0.0
    score: float = 0.0

    @property
    def passed(self) -> bool:
        return all(f.passed for f in self.filters)

    @property
    def failures(self) -> tuple[FilterResult, ...]:
        return tuple(f for f in self.filters if not f.passed)

    @property
    def failure_summary(self) -> str:
        return ", ".join(f.name for f in self.failures) or "-"

    def figures(self) -> dict[str, float | None]:
        """Every number the screen used, keyed by name.

        This is the grounding contract with Phase 4. Add a figure to a prompt
        only if it is in here; check a figure in a verdict by looking it up here.
        """
        f, p = self.fundamentals, self.prices
        return {
            "price": f.price,
            "market_cap": f.market_cap,
            "fifty_two_week_high": f.fifty_two_week_high,
            "fifty_two_week_low": f.fifty_two_week_low,
            "position_in_52w_range": f.position_in_52w_range,
            "drawdown_from_high": f.drawdown_from_high,
            "average_volume": f.average_volume,
            "return_1m": p.return_1m,
            "return_3m": p.return_3m,
            "return_6m": p.return_6m,
            "return_12m": p.return_12m,
            "trailing_pe": f.trailing_pe,
            "forward_pe": f.forward_pe,
            "price_to_book": f.price_to_book,
            "ev_to_ebitda": f.ev_to_ebitda,
            "dividend_yield": f.dividend_yield,
            "free_cash_flow": f.free_cash_flow,
            "operating_cash_flow": f.operating_cash_flow,
            "debt_to_equity_pct": f.debt_to_equity,
            "total_debt": f.total_debt,
            "return_on_equity": f.return_on_equity,
            "profit_margin": f.profit_margin,
            "revenue_growth": f.revenue_growth,
            "earnings_growth": f.earnings_growth,
            "trailing_eps": f.trailing_eps,
            "book_value": f.book_value,
            "total_revenue": f.total_revenue,
            "peer_median": self.peer.median,
        }


@dataclass(frozen=True)
class ScreenResult:
    """The output of stage 1: survivors ranked, rejections kept, errors surfaced.

    `errors` is deliberately not swallowed. A ticker that failed to fetch is not
    a ticker that failed the screen, and collapsing the two would let a broken
    data layer look like a strict filter.
    """

    as_of: datetime
    universe_size: int
    passed: tuple[Candidate, ...]
    rejected: tuple[Candidate, ...]
    errors: tuple[tuple[str, str], ...] = ()

    @property
    def evaluated(self) -> int:
        return len(self.passed) + len(self.rejected)
