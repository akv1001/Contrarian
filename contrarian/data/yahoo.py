"""`MarketDataSource` backed by yfinance.

This is the only file in the project that imports yfinance. Everything it knows
about Yahoo's field names, its NSE quirks, and its habit of changing shape stays
here.

Measured facts about NSE coverage in yfinance 1.6.0, from the Phase 2 spike —
these drove the design, so re-run the spike before assuming they still hold:

- `.info` does **not** carry `freeCashflow`, `operatingCashflow`, or
  `returnOnEquity` for NSE tickers, even though it does for US ones. All three
  are recoverable from the financial statements, which is why this module reads
  them rather than trusting `.info` alone. Screening on positive free cash flow
  is a stated Phase 3 filter, so this was not optional.
- `.info` does not carry `fiftyTwoWeekChange` either, so the 12-month return is
  computed from daily closes in `price_stats`.
- `.news` returns items nested as `{"id": ..., "content": {...}}`. Older
  tutorials assume a flat dict with `title`/`link`/`providerPublishTime`; that
  shape is gone. It is also only loosely filtered by ticker — see `news_rss.py`.

Every network call goes through `cache.py`, so a spike re-run costs nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from ..universe import yahoo_symbol
from . import cache
from .source import (
    DataSourceError,
    Fundamentals,
    NewsItem,
    PriceStats,
    as_float,
)

# Yahoo `.info` key -> our field name. Ours are snake_case and provider-neutral
# so a future source can populate the same record without renaming anything.
_INFO_MAP: dict[str, str] = {
    "currentPrice": "price",
    "marketCap": "market_cap",
    "fiftyTwoWeekHigh": "fifty_two_week_high",
    "fiftyTwoWeekLow": "fifty_two_week_low",
    "averageVolume": "average_volume",
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "priceToBook": "price_to_book",
    "enterpriseToEbitda": "ev_to_ebitda",
    "dividendYield": "dividend_yield",
    "freeCashflow": "free_cash_flow",
    "operatingCashflow": "operating_cash_flow",
    "debtToEquity": "debt_to_equity",
    "totalDebt": "total_debt",
    "returnOnEquity": "return_on_equity",
    "profitMargins": "profit_margin",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "trailingEps": "trailing_eps",
    "bookValue": "book_value",
    "totalRevenue": "total_revenue",
}

# Statement row labels we read when `.info` comes up short. Yahoo's labels are
# title-case English and have been stable, but they are still scraped text, so
# each lookup tolerates absence rather than assuming a row exists.
_CASHFLOW_ROWS = {
    "free_cash_flow": "Free Cash Flow",
    "operating_cash_flow": "Operating Cash Flow",
}
_BALANCE_ROWS = {"equity": "Stockholders Equity", "total_debt": "Total Debt"}
_INCOME_ROWS = {"net_income": "Net Income", "total_revenue": "Total Revenue"}

# A healthy `.info` payload for an NSE ticker has 100+ keys. A handful means
# Yahoo returned a stub — a delisted symbol, a rename, or a rate-limit response.
# Failing here is the point: a stub that silently becomes "all fields None" would
# read downstream as "this company has no debt and no earnings".
_MIN_INFO_KEYS = 20

# Key count alone is not enough. `TATAMOTOR.NS` (a near-miss of a real symbol)
# returns 25 keys of pure exchange metadata — timezone, locale, `marketState` —
# with no price and no market cap, and would sail past a count check. Requiring
# at least one price field is what actually distinguishes a quote from a husk.
_PRICE_KEYS = ("regularMarketPrice", "currentPrice", "previousClose", "marketCap")


class YahooSource:
    """Fetches NSE fundamentals, prices, and news from Yahoo Finance."""

    name = "yahoo"

    def __init__(self, *, news_limit: int = 10) -> None:
        self.news_limit = news_limit

    # ---------------------------------------------------------------- fundamentals

    def fundamentals(self, symbol: str, *, refresh: bool = False) -> Fundamentals:
        ysym = yahoo_symbol(symbol)

        info = cache.get_or_fetch(
            "yahoo_info", ysym, cache.TTL_QUOTE, lambda: self._fetch_info(ysym), refresh=refresh
        )
        statements = cache.get_or_fetch(
            "yahoo_statements",
            ysym,
            cache.TTL_STATEMENTS,
            lambda: self._fetch_statements(ysym),
            refresh=refresh,
        )

        return self._build_fundamentals(
            symbol=symbol,
            info=info.value,
            statements=statements.value,
            fetched_at=datetime.fromtimestamp(info.fetched_at, tz=timezone.utc),
        )

    def _fetch_info(self, ysym: str) -> dict[str, Any]:
        """Pull `.info` and keep only the keys we map, plus identity fields.

        Trimming before caching matters: the full payload is ~150 keys of mostly
        marketing copy and thumbnail URLs, and we would be storing 50 of them.
        """
        try:
            info = yf.Ticker(ysym).info
        except Exception as exc:  # yfinance raises a wide and undocumented range
            raise DataSourceError(
                f"Yahoo lookup failed for {ysym}: {type(exc).__name__}: {exc}\n"
                f"If this hits many tickers at once it is usually rate limiting — "
                f"wait a few minutes, or run against a warm cache."
            ) from exc

        if not isinstance(info, dict) or len(info) < _MIN_INFO_KEYS:
            raise DataSourceError(
                f"Yahoo returned a stub payload for {ysym} "
                f"({len(info) if hasattr(info, '__len__') else '?'} keys, expected "
                f"{_MIN_INFO_KEYS}+). The symbol may have been renamed or delisted, "
                f"or Yahoo is rate-limiting. Check contrarian/universe.py."
            )
        if not any(info.get(k) is not None for k in _PRICE_KEYS):
            raise DataSourceError(
                f"Yahoo returned exchange metadata but no price for {ysym} "
                f"(none of {', '.join(_PRICE_KEYS)}). This is what a not-quite-right "
                f"symbol looks like — check the mapping in contrarian/universe.py."
            )

        wanted = set(_INFO_MAP) | {
            "symbol",
            "shortName",
            "longName",
            "sector",
            "industry",
            "currency",
            "regularMarketPrice",
            "sharesOutstanding",
            "netIncomeToCommon",
        }
        return {k: _jsonable(v) for k, v in info.items() if k in wanted}

    def _fetch_statements(self, ysym: str) -> dict[str, Any]:
        """Pull the three annual statements, reduced to the rows we actually use.

        Each statement is fetched independently and a failure in one does not
        sink the others — Yahoo will sometimes serve a cash flow statement and an
        empty balance sheet for the same ticker.
        """
        ticker = yf.Ticker(ysym)
        out: dict[str, Any] = {}
        for attr, rows in (
            ("cashflow", _CASHFLOW_ROWS),
            ("balance_sheet", _BALANCE_ROWS),
            ("income_stmt", _INCOME_ROWS),
        ):
            out[attr] = _latest_rows(ticker, attr, rows)
        return out

    def _build_fundamentals(
        self,
        *,
        symbol: str,
        info: dict[str, Any],
        statements: dict[str, Any],
        fetched_at: datetime,
    ) -> Fundamentals:
        """Assemble a `Fundamentals` from the cached payloads, recording provenance.

        Precedence is `.info` first, statements second, derivation last. `.info`
        wins because it is the more current figure; the statements are an annual
        snapshot that can be a year old.
        """
        values: dict[str, Any] = {}
        provenance: dict[str, str] = {}

        for yahoo_key, our_key in _INFO_MAP.items():
            value = as_float(info.get(yahoo_key))
            if value is not None:
                values[our_key] = value
                provenance[our_key] = "info"

        # `currentPrice` is absent outside market hours for some NSE tickers;
        # `regularMarketPrice` is the same number and is reliably present.
        if "price" not in values:
            price = as_float(info.get("regularMarketPrice"))
            if price is not None:
                values["price"] = price
                provenance["price"] = "info:regularMarketPrice"

        cashflow = statements.get("cashflow") or {}
        for our_key in ("free_cash_flow", "operating_cash_flow"):
            if our_key not in values:
                value = as_float(cashflow.get(our_key))
                if value is not None:
                    values[our_key] = value
                    provenance[our_key] = "cashflow_stmt"

        balance = statements.get("balance_sheet") or {}
        income = statements.get("income_stmt") or {}

        if "total_debt" not in values:
            value = as_float(balance.get("total_debt"))
            if value is not None:
                values["total_debt"] = value
                provenance["total_debt"] = "balance_sheet"

        # ROE is absent from `.info` for NSE tickers. Deriving it here is safe —
        # it is our arithmetic on two sourced figures, not the model's, and
        # provenance marks it as derived so the report can say where it came from.
        if "return_on_equity" not in values:
            net_income = as_float(income.get("net_income"))
            equity = as_float(balance.get("equity"))
            if net_income is not None and equity and equity > 0:
                values["return_on_equity"] = net_income / equity
                provenance["return_on_equity"] = "derived:net_income/equity"

        if "profit_margin" not in values:
            net_income = as_float(income.get("net_income"))
            revenue = as_float(income.get("total_revenue"))
            if net_income is not None and revenue and revenue > 0:
                values["profit_margin"] = net_income / revenue
                provenance["profit_margin"] = "derived:net_income/revenue"

        return Fundamentals(
            ticker=symbol,
            fetched_at=fetched_at,
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            currency=info.get("currency"),
            provenance=provenance,
            raw={"info": info, "statements": statements},
            **values,
        )

    # ---------------------------------------------------------------- prices

    def price_stats(self, symbol: str, *, refresh: bool = False) -> PriceStats:
        ysym = yahoo_symbol(symbol)
        entry = cache.get_or_fetch(
            "yahoo_prices",
            ysym,
            cache.TTL_QUOTE,
            lambda: self._fetch_prices(ysym),
            refresh=refresh,
        )
        payload = entry.value
        return PriceStats(
            ticker=symbol,
            fetched_at=datetime.fromtimestamp(entry.fetched_at, tz=timezone.utc),
            **{k: v for k, v in payload.items()},
        )

    def _fetch_prices(self, ysym: str) -> dict[str, Any]:
        """Download a year of daily closes and reduce to trailing returns.

        Returns are computed against a fixed number of trading days back rather
        than calendar dates, which avoids having to special-case Indian market
        holidays. ~21 trading days to a month is close enough for a screen.
        """
        try:
            history = yf.Ticker(ysym).history(period="1y", interval="1d", auto_adjust=True)
        except Exception as exc:
            raise DataSourceError(
                f"Yahoo price history failed for {ysym}: {type(exc).__name__}: {exc}"
            ) from exc

        if history is None or history.empty or "Close" not in history:
            raise DataSourceError(
                f"Yahoo returned no price history for {ysym}. "
                f"Check the symbol in contrarian/universe.py."
            )

        closes = history["Close"].dropna()
        volumes = history["Volume"].dropna() if "Volume" in history else None
        n = len(closes)
        if n < 2:
            raise DataSourceError(f"Only {n} usable closes for {ysym}; expected ~250.")

        last = float(closes.iloc[-1])

        def trailing_return(days_back: int) -> float | None:
            if n <= days_back:
                return None
            past = float(closes.iloc[-1 - days_back])
            return (last - past) / past if past > 0 else None

        return {
            "last_close": last,
            "return_1m": trailing_return(21),
            "return_3m": trailing_return(63),
            "return_6m": trailing_return(126),
            "return_12m": trailing_return(min(250, n - 1)),
            "high_52w": float(closes.max()),
            "low_52w": float(closes.min()),
            "avg_volume_3m": (
                float(volumes.tail(63).mean()) if volumes is not None and len(volumes) else None
            ),
            "trading_days": n,
        }

    # ---------------------------------------------------------------- news

    def news(
        self,
        symbol: str,
        *,
        company_name: str | None = None,
        limit: int | None = None,
        refresh: bool = False,
    ) -> list[NewsItem]:
        ysym = yahoo_symbol(symbol)
        entry = cache.get_or_fetch(
            "yahoo_news", ysym, cache.TTL_NEWS, lambda: self._fetch_news(ysym), refresh=refresh
        )
        items = [_news_item(raw) for raw in entry.value]
        return [i for i in items if i is not None][: (limit or self.news_limit)]

    def _fetch_news(self, ysym: str) -> list[dict[str, Any]]:
        """Pull `.news` and flatten the `{"id", "content"}` wrapper.

        An empty list is a legitimate answer here, not an error — thin news
        coverage for NSE tickers is the known weakness this feed has, and the
        spike needs to be able to measure it rather than crash on it.
        """
        try:
            raw = yf.Ticker(ysym).news or []
        except Exception as exc:
            raise DataSourceError(
                f"Yahoo news lookup failed for {ysym}: {type(exc).__name__}: {exc}"
            ) from exc

        flattened = []
        for item in raw:
            content = item.get("content", item) if isinstance(item, dict) else None
            if not isinstance(content, dict):
                continue
            flattened.append(
                {
                    "title": content.get("title"),
                    "summary": content.get("summary") or content.get("description"),
                    "pubDate": content.get("pubDate"),
                    "publisher": (content.get("provider") or {}).get("displayName"),
                    "url": (content.get("canonicalUrl") or {}).get("url")
                    or content.get("previewUrl"),
                }
            )
        return flattened


def _news_item(raw: dict[str, Any]) -> NewsItem | None:
    title = (raw.get("title") or "").strip()
    if not title:
        return None
    return NewsItem(
        title=title,
        url=raw.get("url"),
        publisher=raw.get("publisher"),
        published_at=_parse_iso(raw.get("pubDate")),
        summary=(raw.get("summary") or "").strip() or None,
        source="yahoo",
    )


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _latest_rows(ticker: Any, attr: str, rows: dict[str, str]) -> dict[str, Any]:
    """Read the most recent column of a statement for the named rows.

    Statements come back as a DataFrame with periods as columns, newest first.
    We take the newest non-null value per row rather than strictly column 0,
    because the oldest column is routinely all-NaN and the newest occasionally is
    too when a filing is partially loaded.
    """
    try:
        frame = getattr(ticker, attr)
    except Exception:
        return {}
    if frame is None or getattr(frame, "empty", True):
        return {}

    out: dict[str, Any] = {}
    for our_key, label in rows.items():
        if label not in frame.index:
            continue
        series = frame.loc[label].dropna()
        if len(series):
            out[our_key] = float(series.iloc[0])
            out[f"{our_key}_period"] = str(series.index[0])[:10]
    return out


def _jsonable(value: Any) -> Any:
    """Reduce a scraped value to something `json.dumps` accepts.

    yfinance mixes numpy scalars and pandas timestamps into `.info`, and those
    blow up the cache write with an unhelpful TypeError.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:
            pass
    return str(value)
