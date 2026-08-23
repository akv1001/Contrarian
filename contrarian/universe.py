"""The Nifty 50 constituent list.

Hardcoded on purpose. The index reconstitutes roughly semi-annually, so an API
dependency would buy us a live list at the cost of a network call, a parser, and
a failure mode on every single run. A dated constant that you review twice a year
is the better trade for a hobby project.

**The date below is the review date, not a guarantee.** Verify against
https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-50 when it
goes stale. `scripts/coverage_spike.py` will tell you if a symbol has stopped
resolving, which is the usual first sign that a constituent changed.

Symbols here are NSE symbols. Yahoo Finance needs a `.NS` suffix — use
`yahoo_symbol()` rather than concatenating by hand, so there is one place to fix
if the convention ever changes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Last reviewed: 2026-08-23.
NIFTY_50_AS_OF = "2026-08-23"


@dataclass(frozen=True)
class Constituent:
    """One index member.

    `name` exists for the news search, which queries by company name — ticker
    symbols are far too ambiguous for a news query to return anything useful.

    `yahoo` overrides the default `.NS` suffixing for the cases where Yahoo does
    not use the NSE symbol. Rare, but silent when it happens: you get a 404 that
    looks like a delisting.
    """

    symbol: str
    name: str
    sector: str
    yahoo: str | None = None

    @property
    def yahoo_symbol(self) -> str:
        return self.yahoo or yahoo_symbol(self.symbol)


# NSE symbol -> Yahoo symbol, for the cases where they disagree. Found by the
# Phase 2 spike, which is the argument for re-running it after any index change:
# a symbol that stops resolving is indistinguishable from a delisting until you
# go looking.
YAHOO_SYMBOL_OVERRIDES: dict[str, str] = {
    # Tata Motors demerged into a commercial-vehicle and a passenger-vehicle
    # entity. NSE kept `TATAMOTORS` for the CV business; Yahoo files it under
    # `TMCV.NS` instead. `TATAMOTORS.NS` now 404s outright.
    "TATAMOTORS": "TMCV.NS",
}


def yahoo_symbol(nse_symbol: str) -> str:
    """Convert an NSE symbol to its Yahoo Finance form (`INFY` -> `INFY.NS`).

    Checks the override table first — most symbols map by suffixing, but the
    exceptions fail in a way that reads like a bad ticker rather than a mapping
    problem, so they are worth handling in one place.
    """
    if nse_symbol.endswith(".NS"):
        return nse_symbol
    override = YAHOO_SYMBOL_OVERRIDES.get(nse_symbol.upper())
    return override if override else f"{nse_symbol}.NS"


# Sector labels are our own coarse grouping, used to compare a candidate's
# valuation against its peers in Phase 3. They are deliberately broader than
# Yahoo's `industry` field: with only 50 names, finer buckets leave too few peers
# for a median to mean anything.
#
# 51 entries, not 50 — see the Tata Motors note below.
NIFTY_50: tuple[Constituent, ...] = (
    Constituent("ADANIENT", "Adani Enterprises", "Conglomerate"),
    Constituent("ADANIPORTS", "Adani Ports and Special Economic Zone", "Infrastructure"),
    Constituent("APOLLOHOSP", "Apollo Hospitals Enterprise", "Healthcare"),
    Constituent("ASIANPAINT", "Asian Paints", "Consumer"),
    Constituent("AXISBANK", "Axis Bank", "Financials"),
    Constituent("BAJAJ-AUTO", "Bajaj Auto", "Auto"),
    Constituent("BAJAJFINSV", "Bajaj Finserv", "Financials"),
    Constituent("BAJFINANCE", "Bajaj Finance", "Financials"),
    Constituent("BEL", "Bharat Electronics", "Industrials"),
    Constituent("BHARTIARTL", "Bharti Airtel", "Telecom"),
    Constituent("CIPLA", "Cipla", "Healthcare"),
    Constituent("COALINDIA", "Coal India", "Energy"),
    Constituent("DRREDDY", "Dr Reddys Laboratories", "Healthcare"),
    Constituent("EICHERMOT", "Eicher Motors", "Auto"),
    Constituent("ETERNAL", "Eternal", "Consumer"),
    Constituent("GRASIM", "Grasim Industries", "Materials"),
    Constituent("HCLTECH", "HCL Technologies", "IT"),
    Constituent("HDFCBANK", "HDFC Bank", "Financials"),
    Constituent("HDFCLIFE", "HDFC Life Insurance", "Financials"),
    Constituent("HEROMOTOCO", "Hero MotoCorp", "Auto"),
    Constituent("HINDALCO", "Hindalco Industries", "Materials"),
    Constituent("HINDUNILVR", "Hindustan Unilever", "Consumer"),
    Constituent("ICICIBANK", "ICICI Bank", "Financials"),
    Constituent("INDUSINDBK", "IndusInd Bank", "Financials"),
    Constituent("INFY", "Infosys", "IT"),
    Constituent("ITC", "ITC", "Consumer"),
    Constituent("JIOFIN", "Jio Financial Services", "Financials"),
    Constituent("JSWSTEEL", "JSW Steel", "Materials"),
    Constituent("KOTAKBANK", "Kotak Mahindra Bank", "Financials"),
    Constituent("LT", "Larsen and Toubro", "Industrials"),
    Constituent("M&M", "Mahindra and Mahindra", "Auto"),
    Constituent("MARUTI", "Maruti Suzuki India", "Auto"),
    Constituent("NESTLEIND", "Nestle India", "Consumer"),
    Constituent("NTPC", "NTPC", "Utilities"),
    Constituent("ONGC", "Oil and Natural Gas Corporation", "Energy"),
    Constituent("POWERGRID", "Power Grid Corporation of India", "Utilities"),
    Constituent("RELIANCE", "Reliance Industries", "Energy"),
    Constituent("SBILIFE", "SBI Life Insurance", "Financials"),
    Constituent("SBIN", "State Bank of India", "Financials"),
    Constituent("SHRIRAMFIN", "Shriram Finance", "Financials"),
    Constituent("SUNPHARMA", "Sun Pharmaceutical Industries", "Healthcare"),
    Constituent("TATACONSUM", "Tata Consumer Products", "Consumer"),
    # Post-demerger both Tata Motors entities are listed here. Which of them the
    # index actually holds is the thing to check on the next review — carrying
    # both is the honest default, since dropping one on a guess silently removes
    # a large-cap auto name from the universe.
    Constituent("TATAMOTORS", "Tata Motors", "Auto", yahoo="TMCV.NS"),
    Constituent("TMPV", "Tata Motors Passenger Vehicles", "Auto"),
    Constituent("TATASTEEL", "Tata Steel", "Materials"),
    Constituent("TCS", "Tata Consultancy Services", "IT"),
    Constituent("TECHM", "Tech Mahindra", "IT"),
    Constituent("TITAN", "Titan Company", "Consumer"),
    Constituent("TRENT", "Trent", "Consumer"),
    Constituent("ULTRACEMCO", "UltraTech Cement", "Materials"),
    Constituent("WIPRO", "Wipro", "IT"),
)


def by_symbol(symbol: str) -> Constituent:
    """Look up a constituent by NSE symbol, accepting the `.NS` form too."""
    wanted = symbol.removesuffix(".NS").upper()
    for c in NIFTY_50:
        if c.symbol == wanted:
            return c
    raise KeyError(f"{symbol!r} is not in the Nifty 50 list dated {NIFTY_50_AS_OF}")


def sectors() -> dict[str, list[Constituent]]:
    """Group constituents by sector, for the peer-relative screen in Phase 3."""
    grouped: dict[str, list[Constituent]] = {}
    for c in NIFTY_50:
        grouped.setdefault(c.sector, []).append(c)
    return grouped
