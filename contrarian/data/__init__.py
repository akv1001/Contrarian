"""Market data layer.

Import from `.source` for types and the protocol; import a concrete source only
where one is being constructed. Stages must depend on `MarketDataSource`, never
on a specific implementation.
"""

from .source import (
    DataSourceError,
    Fundamentals,
    MarketDataSource,
    NewsItem,
    PriceStats,
)

__all__ = [
    "DataSourceError",
    "Fundamentals",
    "MarketDataSource",
    "NewsItem",
    "PriceStats",
]
