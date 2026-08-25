"""Tests for the parts of the data layer that do not touch the network.

Scope is deliberate: these cover parsing, coercion, caching, and symbol mapping —
the logic that turns a scraped payload into a `Fundamentals`. The question "is
Yahoo's coverage good enough" is not a unit test, it is `scripts/coverage_spike.py`,
because the answer changes without any code changing.

`unittest` rather than pytest: it is in the standard library, and a zero-cost
project should not take a dependency to run seven assertions.

    .venv/bin/python -m unittest discover tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrarian.data import cache
from contrarian.data.news_rss import _parse_rfc822, _to_item
from contrarian.data.source import Fundamentals, as_float
from contrarian.data.yahoo import YahooSource, _parse_iso
from contrarian.universe import NIFTY_50, by_symbol, yahoo_symbol

NOW = datetime(2026, 8, 23, tzinfo=timezone.utc)


class TestAsFloat(unittest.TestCase):
    """`as_float` is the single chokepoint where scraped junk becomes a number."""

    def test_coerces_numeric_types(self):
        self.assertEqual(as_float(3), 3.0)
        self.assertEqual(as_float("2.5"), 2.5)

    def test_unusable_values_become_none(self):
        for value in (None, "", "n/a", float("nan"), float("inf"), [], {}):
            self.assertIsNone(as_float(value), f"{value!r} should coerce to None")

    def test_booleans_are_not_numbers(self):
        # `float(True)` is 1.0, which would silently turn a flag into a figure.
        self.assertIsNone(as_float(True))
        self.assertIsNone(as_float(False))

    def test_zero_survives(self):
        # The distinction the whole layer rests on: zero is data, None is absence.
        self.assertEqual(as_float(0), 0.0)


class TestSymbolMapping(unittest.TestCase):
    def test_default_is_ns_suffix(self):
        self.assertEqual(yahoo_symbol("INFY"), "INFY.NS")

    def test_already_suffixed_is_left_alone(self):
        self.assertEqual(yahoo_symbol("INFY.NS"), "INFY.NS")

    def test_override_wins(self):
        # Tata Motors post-demerger: NSE says TATAMOTORS, Yahoo says TMCV.
        self.assertEqual(yahoo_symbol("TATAMOTORS"), "TMCV.NS")

    def test_lookup_accepts_either_form(self):
        self.assertEqual(by_symbol("INFY").name, "Infosys")
        self.assertEqual(by_symbol("infy.NS").symbol, "INFY")

    def test_unknown_symbol_raises(self):
        with self.assertRaises(KeyError):
            by_symbol("NOTREAL")

    def test_universe_has_no_duplicates(self):
        symbols = [c.symbol for c in NIFTY_50]
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_the_nifty_50_holds_exactly_fifty_names(self):
        """The only cheap check there is on a hand-maintained list.

        It held 51 for a while. A demerger and a rebalance both landed between
        reviews, and the list absorbed the new names without dropping the old
        ones — so the report said "51 tickers screened" for an index of 50, and
        two departed constituents were being analysed as current.

        That is the shape of the failure: a stale constituent does not throw. It
        fetches, screens, gets a verdict, and reaches the report looking exactly
        like a real result. Counting is the one assertion that would have caught
        it, so it is a test rather than a comment.
        """
        self.assertEqual(len(NIFTY_50), 50)

    def test_departed_constituents_are_gone(self):
        """Named individually, because the count alone cannot tell a correct
        swap from two compensating errors."""
        symbols = {c.symbol for c in NIFTY_50}
        for departed in ("INDUSINDBK", "HEROMOTOCO", "TATAMOTORS"):
            self.assertNotIn(departed, symbols)
        for current in ("INDIGO", "MAXHEALTH", "TMPV"):
            self.assertIn(current, symbols)

    def test_the_off_index_tata_motors_mapping_survives_its_removal(self):
        """`TATAMOTORS` left the index but the Yahoo override stays: the analyze
        script and the coverage spike both accept off-index tickers, and a bare
        `TATAMOTORS.NS` 404s in a way that reads like a delisting."""
        self.assertEqual(yahoo_symbol("TATAMOTORS"), "TMCV.NS")


class TestFundamentals(unittest.TestCase):
    def _sample(self, **overrides) -> Fundamentals:
        base = dict(
            ticker="TEST",
            fetched_at=NOW,
            price=80.0,
            fifty_two_week_high=100.0,
            fifty_two_week_low=60.0,
        )
        base.update(overrides)
        return Fundamentals(**base)

    def test_drawdown_and_range_position(self):
        f = self._sample()
        self.assertAlmostEqual(f.drawdown_from_high, 0.20)
        self.assertAlmostEqual(f.position_in_52w_range, 0.50)

    def test_derived_values_are_none_without_inputs(self):
        f = self._sample(fifty_two_week_high=None)
        self.assertIsNone(f.drawdown_from_high)
        self.assertIsNone(f.position_in_52w_range)

    def test_flat_range_does_not_divide_by_zero(self):
        f = self._sample(fifty_two_week_high=60.0, fifty_two_week_low=60.0)
        self.assertIsNone(f.position_in_52w_range)

    def test_missing_reports_absent_screen_fields(self):
        f = self._sample()
        self.assertIn("trailing_pe", f.missing())
        self.assertNotIn("price", f.missing())
        self.assertFalse(f.is_screenable)


class TestStubDetection(unittest.TestCase):
    """The guard that stops a husk payload becoming 'all fields None'."""

    def _fetch_with_info(self, info):
        with mock.patch("contrarian.data.yahoo.yf.Ticker") as ticker:
            ticker.return_value.info = info
            return YahooSource()._fetch_info("TEST.NS")

    def test_tiny_payload_raises(self):
        from contrarian.data.source import DataSourceError

        with self.assertRaises(DataSourceError):
            self._fetch_with_info({"symbol": "TEST.NS"})

    def test_metadata_only_payload_raises(self):
        """A real failure seen in the spike: many keys, no price.

        `TATAMOTOR.NS` returns 25 keys of exchange metadata and would pass any
        check based on key count alone.
        """
        from contrarian.data.source import DataSourceError

        metadata_only = {f"meta_{i}": "x" for i in range(30)}
        metadata_only["symbol"] = "TEST.NS"
        with self.assertRaises(DataSourceError):
            self._fetch_with_info(metadata_only)

    def test_payload_with_a_price_passes(self):
        payload = {f"meta_{i}": "x" for i in range(30)}
        payload.update({"symbol": "TEST.NS", "regularMarketPrice": 101.5})
        result = self._fetch_with_info(payload)
        self.assertEqual(result["regularMarketPrice"], 101.5)


class TestNewsParsing(unittest.TestCase):
    def test_publisher_suffix_is_stripped_from_title(self):
        item = _to_item(
            {
                "title": "Infosys beats estimates - Reuters",
                "publisher": "Reuters",
                "pubDate": "Thu, 06 Aug 2026 07:00:00 GMT",
                "url": "https://example.com/a",
            }
        )
        self.assertEqual(item.title, "Infosys beats estimates")
        self.assertEqual(item.publisher, "Reuters")

    def test_title_without_suffix_is_untouched(self):
        item = _to_item({"title": "Infosys beats estimates", "publisher": "Reuters"})
        self.assertEqual(item.title, "Infosys beats estimates")

    def test_untitled_item_is_dropped(self):
        self.assertIsNone(_to_item({"title": "   "}))

    def test_rfc822_dates_normalise_to_utc(self):
        parsed = _parse_rfc822("Thu, 06 Aug 2026 07:00:00 GMT")
        self.assertEqual(parsed.year, 2026)
        self.assertEqual(parsed.tzinfo, timezone.utc)

    def test_bad_dates_are_none_not_exceptions(self):
        self.assertIsNone(_parse_rfc822("not a date"))
        self.assertIsNone(_parse_rfc822(None))
        self.assertIsNone(_parse_iso("not a date"))
        self.assertIsNone(_parse_iso(None))

    def test_yahoo_iso_dates_parse(self):
        parsed = _parse_iso("2026-08-17T18:27:44Z")
        self.assertEqual(parsed.year, 2026)
        self.assertIsNotNone(parsed.tzinfo)


class TestCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(cache, "CACHE_ROOT", Path(self._tmp.name))
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_round_trip(self):
        cache.write("ns", "KEY", {"a": 1})
        self.assertEqual(cache.read("ns", "KEY", ttl=60).value, {"a": 1})

    def test_expired_entry_is_a_miss(self):
        cache.write("ns", "KEY", {"a": 1})
        self.assertIsNone(cache.read("ns", "KEY", ttl=-1))

    def test_missing_entry_is_a_miss(self):
        self.assertIsNone(cache.read("ns", "ABSENT", ttl=60))

    def test_corrupt_file_is_a_miss_not_a_crash(self):
        cache.write("ns", "KEY", {"a": 1})
        cache.path_for("ns", "KEY").write_text("{ truncated")
        self.assertIsNone(cache.read("ns", "KEY", ttl=60))

    def test_unsafe_symbols_are_filename_safe(self):
        # M&M and BAJAJ-AUTO are real Nifty 50 symbols.
        cache.write("ns", "M&M", {"ok": True})
        self.assertEqual(cache.read("ns", "M&M", ttl=60).value, {"ok": True})
        self.assertTrue(cache.path_for("ns", "M&M").exists())

    def test_get_or_fetch_calls_through_once(self):
        calls = []

        def fetch():
            calls.append(1)
            return {"v": len(calls)}

        first = cache.get_or_fetch("ns", "K", 60, fetch)
        second = cache.get_or_fetch("ns", "K", 60, fetch)
        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertEqual(len(calls), 1)

    def test_refresh_bypasses_the_cache(self):
        cache.get_or_fetch("ns", "K", 60, lambda: "old")
        entry = cache.get_or_fetch("ns", "K", 60, lambda: "new", refresh=True)
        self.assertEqual(entry.value, "new")
        self.assertEqual(cache.read("ns", "K", ttl=60).value, "new")

    def test_unserialisable_value_fails_loudly(self):
        with self.assertRaises(cache.CacheError):
            cache.write("ns", "K", {"bad": object()})


if __name__ == "__main__":
    unittest.main()
