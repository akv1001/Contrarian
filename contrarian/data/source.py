"""The swap seam: what a market data source must provide, in our own vocabulary.

`screen.py` and `enrich.py` depend on this module and never import yfinance.
That is the whole point. yfinance is an unofficial scraper of a site that can
change without notice; when it breaks or gets rate-limited into uselessness, the
fix should be one new file implementing `MarketDataSource`, not a rewrite of
every stage that touches a number.

Two conventions that the rest of the pipeline relies on:

1. **`None` means "the source did not give us this field."** It never means zero.
   Phase 3 must exclude on absent data rather than treat it as a failing value,
   or a ticker with patchy coverage silently looks like a bargain.
2. **Every value carries provenance.** `Fundamentals.provenance` records which
   upstream payload each figure came from. Design rule 1 says the model never
   recalls a figure from training — provenance is how we check that, by tracing
   every number in the final report back to a source payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence


class DataSourceError(RuntimeError):
    """A data source failed, or returned something that does not match its schema.

    Raised loudly on purpose. The failure mode we are guarding against is a
    scraper quietly returning `None` for everything after an upstream change,
    which would look like "no candidates passed the screen" rather than "the data
    layer is broken".
    """


# The figures Phase 3 screens on. Kept as an explicit list because the coverage
# spike reports on exactly these — a field not listed here is informational, and
# a field listed here that turns out to be sparse is a screen we cannot build.
SCREEN_FIELDS: tuple[str, ...] = (
    "price",
    "market_cap",
    "fifty_two_week_high",
    "fifty_two_week_low",
    "average_volume",
    "trailing_pe",
    "price_to_book",
    "ev_to_ebitda",
    "debt_to_equity",
    "free_cash_flow",
    "operating_cash_flow",
    "return_on_equity",
    "profit_margin",
    "revenue_growth",
)

# Every numeric field, screened-on or not. The coverage spike reports the full
# set: a field we do not currently screen on is still worth knowing about, since
# it may be what makes a filter possible.
NUMERIC_FIELDS: tuple[str, ...] = (
    "price",
    "market_cap",
    "fifty_two_week_high",
    "fifty_two_week_low",
    "average_volume",
    "trailing_pe",
    "forward_pe",
    "price_to_book",
    "ev_to_ebitda",
    "dividend_yield",
    "free_cash_flow",
    "operating_cash_flow",
    "debt_to_equity",
    "total_debt",
    "return_on_equity",
    "profit_margin",
    "revenue_growth",
    "earnings_growth",
    "trailing_eps",
    "book_value",
    "total_revenue",
)


@dataclass(frozen=True)
class Fundamentals:
    """One company's figures at a point in time.

    Flat by design. A nested structure would read better on paper, but every
    stage downstream wants "give me the P/E" and a flat record keeps that a
    single attribute access with a type the checker can see.
    """

    ticker: str
    fetched_at: datetime

    # Identity
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None

    # Price and liquidity
    price: float | None = None
    market_cap: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    average_volume: float | None = None

    # Valuation
    trailing_pe: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    ev_to_ebitda: float | None = None
    dividend_yield: float | None = None

    # Business health
    free_cash_flow: float | None = None
    operating_cash_flow: float | None = None
    debt_to_equity: float | None = None
    total_debt: float | None = None
    return_on_equity: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    trailing_eps: float | None = None
    book_value: float | None = None
    total_revenue: float | None = None

    # Where each populated field came from: "info", "cashflow", "derived", ...
    provenance: Mapping[str, str] = field(default_factory=dict)
    # The trimmed upstream payload, kept so a figure in the report can be traced
    # back to what the source actually said.
    raw: Mapping[str, Any] = field(default_factory=dict)

    def missing(self, among: Sequence[str] = SCREEN_FIELDS) -> tuple[str, ...]:
        """Which of the given fields the source did not supply."""
        return tuple(name for name in among if getattr(self, name) is None)

    @property
    def is_screenable(self) -> bool:
        """True when every field Phase 3 filters on is present."""
        return not self.missing()

    @property
    def drawdown_from_high(self) -> float | None:
        """How far below the 52-week high the price sits, as a positive fraction.

        The headline contrarian signal: 0.30 means "down 30% from its high".
        """
        if self.price is None or not self.fifty_two_week_high:
            return None
        return (self.fifty_two_week_high - self.price) / self.fifty_two_week_high

    @property
    def position_in_52w_range(self) -> float | None:
        """Where the price sits in its 52-week band: 0.0 at the low, 1.0 at the high.

        Preferred over the raw drawdown for ranking, because it accounts for how
        wide the year's range actually was.
        """
        if self.price is None or self.fifty_two_week_high is None or self.fifty_two_week_low is None:
            return None
        span = self.fifty_two_week_high - self.fifty_two_week_low
        if span <= 0:
            return None
        return (self.price - self.fifty_two_week_low) / span


@dataclass(frozen=True)
class PriceStats:
    """Trailing returns derived from a daily price series.

    Computed here rather than read from the source because Yahoo's own
    `fiftyTwoWeekChange` field is absent for NSE tickers — measured, not assumed.
    Doing the arithmetic ourselves also keeps it deterministic and auditable,
    which is the same reason the model is never allowed to do it.
    """

    ticker: str
    fetched_at: datetime
    last_close: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    return_6m: float | None = None
    return_12m: float | None = None
    high_52w: float | None = None
    low_52w: float | None = None
    avg_volume_3m: float | None = None
    trading_days: int = 0


@dataclass(frozen=True)
class NewsItem:
    """One headline about a company.

    `summary` is often empty and `published_at` is occasionally absent, so both
    are optional. `source` records which feed produced the item — the two feeds
    have very different precision, and Phase 4's prompt should be able to say so.
    """

    title: str
    url: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    summary: str | None = None
    source: str = "unknown"

    @property
    def age_days(self) -> float | None:
        if self.published_at is None:
            return None
        now = datetime.now(self.published_at.tzinfo)
        return (now - self.published_at).total_seconds() / 86400


class MarketDataSource(Protocol):
    """What every stage is allowed to ask for.

    Implementations must raise `DataSourceError` when the upstream shape changes,
    and return `None` per-field only when the source genuinely omitted a value.
    Silence and absence must be distinguishable.
    """

    name: str

    def fundamentals(self, symbol: str, *, refresh: bool = False) -> Fundamentals:
        """Figures for one company. `symbol` is an NSE symbol, e.g. `INFY`."""
        ...

    def price_stats(self, symbol: str, *, refresh: bool = False) -> PriceStats:
        """Trailing returns and the 52-week band, from daily closes."""
        ...

    def news(
        self,
        symbol: str,
        *,
        company_name: str | None = None,
        limit: int = 10,
        refresh: bool = False,
    ) -> list[NewsItem]:
        """Recent headlines. `company_name` is used by name-querying feeds."""
        ...


def as_float(value: Any) -> float | None:
    """Coerce an upstream value to a float, mapping anything unusable to None.

    Handles the specific junk these feeds emit: `None`, empty strings, NaN, and
    numpy scalars. Returning `None` rather than raising is correct here — a
    single absent field is normal, and `Fundamentals.missing()` is what surfaces
    it. Structural breakage is caught upstream, at the payload level.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return result
