"""News via Google News RSS — the primary headline source for Indian tickers.

The plan called this a fallback. The Phase 2 spike promoted it: Yahoo's `.news`
returns ~10 loosely-matched items per NSE ticker, several of which are about
other companies entirely (a RELIANCE.NS query returned a Rolls-Royce headline and
an Odessa, Texas story). Google News RSS returned ~100 dated, on-topic items from
Indian outlets for the same kind of query.

That gap matters more here than it would in most projects. News is the *only*
signal that identifies the "temporary factor" — the entire premise of the
pipeline. Feeding Phase 4 headlines about the wrong company is a direct path to
the failure mode the design exists to prevent.

No API key, no dependency: RSS over stdlib `urllib` and `xml.etree`. It is an
undocumented endpoint like everything else free, so failures are wrapped in
`DataSourceError` rather than allowed to surface as XML parse errors.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from ..universe import by_symbol
from . import cache
from .source import DataSourceError, NewsItem

# `hl`/`gl`/`ceid` pin the edition to English-language India. Without them the
# feed drifts to US coverage, which for a Nifty 50 name means either nothing or
# the wrong company with a similar name.
_ENDPOINT = "https://news.google.com/rss/search"
_EDITION = {"hl": "en-IN", "gl": "IN", "ceid": "IN:en"}

# Google returns 403 to an unset or obviously scripted User-Agent.
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Politeness delay between uncached requests. 50 tickers back to back is exactly
# the shape of traffic that gets an unofficial endpoint to stop answering.
_DELAY_SECONDS = 0.7

_last_request_at = 0.0


class GoogleNewsRSS:
    """Company-name news search against Google News RSS.

    Queries by company name, not ticker: "INFY" as a news query is meaningless,
    while "Infosys" is unambiguous. The name comes from `universe.py`, which is
    why that file carries one.
    """

    name = "google_news_rss"

    def __init__(self, *, window_days: int = 30, limit: int = 25) -> None:
        self.window_days = window_days
        self.limit = limit

    def news(
        self,
        symbol: str,
        *,
        company_name: str | None = None,
        limit: int | None = None,
        refresh: bool = False,
    ) -> list[NewsItem]:
        name = company_name or _name_for(symbol)
        entry = cache.get_or_fetch(
            "google_news",
            f"{symbol}_{self.window_days}d",
            cache.TTL_NEWS,
            lambda: self._fetch(name),
            refresh=refresh,
        )
        items = [item for item in (_to_item(raw) for raw in entry.value) if item is not None]
        return items[: (limit or self.limit)]

    def _fetch(self, company_name: str) -> list[dict[str, Any]]:
        """Fetch and parse one RSS search.

        The exact-phrase quoting is load-bearing. Unquoted, "Tata Motors" matches
        any story mentioning Tata, and the Tata group has a dozen listed entities.
        """
        query = f'"{company_name}" when:{self.window_days}d'
        params = {"q": query, **_EDITION}
        url = f"{_ENDPOINT}?{urllib.parse.urlencode(params)}"

        _throttle()
        try:
            request = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(request, timeout=20) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise DataSourceError(
                f"Google News RSS returned {exc.code} for {company_name!r}. "
                f"A 429 means we are querying too fast; increase the delay in news_rss.py."
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise DataSourceError(
                f"Could not reach Google News RSS for {company_name!r}: {exc}"
            ) from exc

        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise DataSourceError(
                f"Google News RSS returned unparseable XML for {company_name!r}. "
                f"The endpoint shape may have changed."
            ) from exc

        return [
            {
                "title": item.findtext("title"),
                "url": item.findtext("link"),
                "pubDate": item.findtext("pubDate"),
                "publisher": item.findtext("source"),
            }
            for item in root.findall("./channel/item")
        ]


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _DELAY_SECONDS:
        time.sleep(_DELAY_SECONDS - elapsed)
    _last_request_at = time.monotonic()


def _name_for(symbol: str) -> str:
    try:
        return by_symbol(symbol).name
    except KeyError:
        return symbol


def _to_item(raw: dict[str, Any]) -> NewsItem | None:
    title = (raw.get("title") or "").strip()
    if not title:
        return None

    # Google appends " - Publisher" to every headline. The publisher is already
    # in its own element, so strip it rather than feeding the model the same
    # string twice in a context we are trying to keep tight.
    publisher = (raw.get("publisher") or "").strip() or None
    if publisher and title.endswith(f" - {publisher}"):
        title = title[: -(len(publisher) + 3)].strip()

    return NewsItem(
        title=title,
        url=raw.get("url"),
        publisher=publisher,
        published_at=_parse_rfc822(raw.get("pubDate")),
        summary=None,  # RSS descriptions are boilerplate HTML link markup here.
        source="google_news_rss",
    )


def _parse_rfc822(value: Any) -> datetime | None:
    """Parse an RSS `pubDate` (RFC 822), normalised to UTC."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class CompositeSource:
    """The source the pipeline actually uses: Yahoo for figures, RSS for news.

    Composition rather than a subclass, because the two feeds are unrelated
    services that happen to both be free. If Yahoo's fundamentals degrade the way
    its news did, only the constructor argument changes.
    """

    name = "yahoo+google_news"

    def __init__(self, fundamentals_source: Any, news_source: Any) -> None:
        self._fundamentals = fundamentals_source
        self._news = news_source

    def fundamentals(self, symbol: str, *, refresh: bool = False):
        return self._fundamentals.fundamentals(symbol, refresh=refresh)

    def price_stats(self, symbol: str, *, refresh: bool = False):
        return self._fundamentals.price_stats(symbol, refresh=refresh)

    def news(
        self,
        symbol: str,
        *,
        company_name: str | None = None,
        limit: int = 10,
        refresh: bool = False,
    ) -> list[NewsItem]:
        return self._news.news(
            symbol, company_name=company_name, limit=limit, refresh=refresh
        )


def default_source() -> CompositeSource:
    """The standard wiring. One place for the rest of the project to call."""
    from .yahoo import YahooSource

    return CompositeSource(YahooSource(), GoogleNewsRSS())
