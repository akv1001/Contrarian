"""Disk cache with a TTL, for anything we fetch over the network.

Why this exists: a single run touches 50 tickers, and iterating on Phase 3's
filters means running it over and over. Without a cache, every run re-scrapes
Yahoo — slow, and a good way to get rate-limited into a wall of empty responses
that looks exactly like a code bug.

Deliberately boring: one JSON file per (namespace, key). No index, no eviction,
no database. `rm -rf .cache/` is a valid cache flush and that is a feature.

TTLs are per-namespace and set by the caller, because the right value differs by
an order of magnitude: quotes go stale in hours, annual financial statements in
months.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import PROJECT_ROOT

CACHE_ROOT = PROJECT_ROOT / ".cache"

# TTLs in seconds. Named constants rather than magic numbers at call sites, so
# the reasoning for each lives next to the value.
TTL_QUOTE = 6 * 3600  # Price and market cap. Intraday staleness is fine for a
                      # value strategy; free sources are delayed anyway.
TTL_STATEMENTS = 14 * 24 * 3600  # Annual/quarterly financials. These change on a
                                 # results calendar, not continuously.
TTL_NEWS = 2 * 3600  # The "temporary factor" signal. Shortest TTL of the three,
                     # but still long enough to survive a debug loop.

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class CacheError(RuntimeError):
    """The cache itself failed, as opposed to the fetch it was wrapping."""


@dataclass(frozen=True)
class CacheEntry:
    """A cached payload plus the metadata needed to judge and audit it."""

    value: Any
    fetched_at: float
    from_cache: bool

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetched_at


def _safe(part: str) -> str:
    """Make a cache key safe as a filename.

    Matters more than it looks: NSE has symbols like `M&M` and `BAJAJ-AUTO`, and
    an unescaped `&` or `/` in a path is a bug waiting for a specific ticker.
    """
    cleaned = _UNSAFE.sub("_", part)
    if not cleaned:
        raise CacheError(f"Cache key {part!r} is empty after sanitising.")
    return cleaned


def path_for(namespace: str, key: str) -> Path:
    return CACHE_ROOT / _safe(namespace) / f"{_safe(key)}.json"


def read(namespace: str, key: str, ttl: float) -> CacheEntry | None:
    """Return a cached entry if present and younger than `ttl`, else None.

    A corrupt or unreadable file is treated as a miss, not an error. The cache is
    disposable by design; a half-written file from an interrupted run should cost
    one refetch, not a crash.
    """
    file = path_for(namespace, key)
    if not file.exists():
        return None

    try:
        payload = json.loads(file.read_text())
        fetched_at = float(payload["fetched_at"])
        value = payload["value"]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None

    if time.time() - fetched_at > ttl:
        return None
    return CacheEntry(value=value, fetched_at=fetched_at, from_cache=True)


def write(namespace: str, key: str, value: Any) -> float:
    """Persist a value and return its timestamp.

    Writes to a temp file and renames, so a crash mid-write cannot leave a
    truncated file that a later read has to defend against.
    """
    file = path_for(namespace, key)
    file.parent.mkdir(parents=True, exist_ok=True)
    fetched_at = time.time()

    tmp = file.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps({"fetched_at": fetched_at, "value": value}, indent=1))
        tmp.replace(file)
    except (OSError, TypeError) as exc:
        tmp.unlink(missing_ok=True)
        raise CacheError(
            f"Could not cache {namespace}/{key}: {exc}\n"
            f"Values must be JSON-serialisable — convert pandas/numpy types before caching."
        ) from exc
    return fetched_at


def get_or_fetch(
    namespace: str,
    key: str,
    ttl: float,
    fetch: Callable[[], Any],
    *,
    refresh: bool = False,
) -> CacheEntry:
    """Return the cached value, or call `fetch` and cache the result.

    `refresh=True` forces a refetch — the escape hatch for "I think the cache is
    lying to me", without reaching for `rm -rf`.
    """
    if not refresh:
        hit = read(namespace, key, ttl)
        if hit is not None:
            return hit

    value = fetch()
    fetched_at = write(namespace, key, value)
    return CacheEntry(value=value, fetched_at=fetched_at, from_cache=False)


def clear(namespace: str | None = None) -> int:
    """Delete cached files, optionally for one namespace. Returns the count."""
    root = CACHE_ROOT / _safe(namespace) if namespace else CACHE_ROOT
    if not root.exists():
        return 0
    removed = 0
    for file in root.rglob("*.json"):
        file.unlink()
        removed += 1
    return removed


def stats() -> dict[str, int]:
    """Files per namespace. Used by the spike to report cold vs. warm runs."""
    if not CACHE_ROOT.exists():
        return {}
    return {
        d.name: len(list(d.glob("*.json")))
        for d in sorted(CACHE_ROOT.iterdir())
        if d.is_dir()
    }
