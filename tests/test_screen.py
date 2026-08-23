"""Tests for the deterministic screen.

The Phase 3 gate is "tests confirm filters actually exclude", and that phrasing
is the point. A filter that never rejects anything is indistinguishable from no
filter at all until the day it matters, so each one here gets a case built
specifically to fail it, holding everything else passing. Anything that slips
through shows up as exactly one named failure.

All synthetic — no network, no cache, no yfinance.

    .venv/bin/python -m unittest discover tests -v
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrarian import screen
from contrarian.data.source import Fundamentals, PriceStats
from contrarian.schema import PeerBasis
from contrarian.universe import Constituent

NOW = datetime(2026, 8, 23, tzinfo=timezone.utc)


def make_fundamentals(**overrides) -> Fundamentals:
    """A company that passes every filter, so overrides isolate one failure."""
    base = dict(
        ticker="TEST",
        fetched_at=NOW,
        price=70.0,
        market_cap=5e11,
        fifty_two_week_high=100.0,  # 30% off the high...
        fifty_two_week_low=60.0,  # ...and 25% up the range
        average_volume=2_000_000.0,
        trailing_pe=12.0,
        price_to_book=1.5,
        free_cash_flow=5e9,
        operating_cash_flow=8e9,
        debt_to_equity=40.0,  # percent, i.e. 0.40x
        return_on_equity=0.20,
        profit_margin=0.15,
        revenue_growth=0.05,
    )
    base.update(overrides)
    return Fundamentals(**base)


def make_prices(**overrides) -> PriceStats:
    base = dict(ticker="TEST", fetched_at=NOW, last_close=70.0, return_12m=-0.25)
    base.update(overrides)
    return PriceStats(**base)


def make_peer(median: float = 20.0, metric: str = "trailing_pe", fallback: bool = False) -> PeerBasis:
    return PeerBasis(
        label="TestSector", metric=metric, median=median, peer_count=8, is_fallback=fallback
    )


def evaluate(sector: str = "Materials", peer: PeerBasis | None = None, **overrides):
    constituent = Constituent("TEST", "Test Company", sector)
    return screen.evaluate(
        constituent,
        make_fundamentals(**overrides),
        make_prices(),
        peer or make_peer(metric=screen.valuation_metric(sector)),
    )


class TestBaselinePasses(unittest.TestCase):
    """If the baseline did not pass, every exclusion test below would be vacuous."""

    def test_baseline_company_passes_everything(self):
        candidate = evaluate()
        self.assertTrue(candidate.passed, f"baseline failed: {candidate.failure_summary}")

    def test_baseline_financial_passes(self):
        candidate = evaluate(sector="Financials", peer=make_peer(2.5, "price_to_book"))
        self.assertTrue(candidate.passed, f"baseline failed: {candidate.failure_summary}")


class TestFiltersExclude(unittest.TestCase):
    """One case per filter, each failing exactly that filter and no other."""

    def assert_fails_only(self, candidate, name: str):
        failed = {f.name for f in candidate.failures}
        self.assertEqual(
            failed, {name}, f"expected only {name!r} to fail, got {failed or 'nothing'}"
        )

    def test_high_in_the_range_is_not_depressed(self):
        # Needs a wide 52-week band to isolate this filter: at 84 against a
        # 20-100 range the stock is 16% off its high (clearing the drawdown
        # gate) but sits at 80% of the range, which is not a contrarian setup.
        self.assert_fails_only(
            evaluate(price=84.0, fifty_two_week_high=100.0, fifty_two_week_low=20.0),
            "near_52w_low",
        )

    def test_shallow_drawdown_is_rejected(self):
        # At the bottom of a narrow band: low in the range, but barely off the
        # high. This is the case the drawdown floor exists to catch.
        self.assert_fails_only(
            evaluate(price=95.0, fifty_two_week_high=100.0, fifty_two_week_low=94.0),
            "meaningful_drawdown",
        )

    def test_illiquid_name_is_rejected(self):
        self.assert_fails_only(evaluate(average_volume=1_000.0), "liquidity")

    def test_negative_free_cash_flow_is_rejected(self):
        self.assert_fails_only(evaluate(free_cash_flow=-1e9), "positive_fcf")

    def test_excessive_leverage_is_rejected(self):
        self.assert_fails_only(evaluate(debt_to_equity=200.0), "leverage")

    def test_weak_return_on_equity_is_rejected(self):
        self.assert_fails_only(evaluate(return_on_equity=0.04), "return_on_equity")

    def test_loss_making_company_is_rejected(self):
        self.assert_fails_only(evaluate(profit_margin=-0.05), "profitable")

    def test_expensive_vs_peers_is_rejected(self):
        self.assert_fails_only(evaluate(trailing_pe=35.0), "cheap_vs_peers")

    def test_missing_field_excludes_rather_than_passes(self):
        # The rule that matters most: None is absence, never a passing value.
        candidate = evaluate(free_cash_flow=None)
        failed = {f.name for f in candidate.failures}
        self.assertIn("data_complete", failed)
        self.assertIn("positive_fcf", failed)
        self.assertFalse(candidate.passed)


class TestDebtToEquityIsAPercentage(unittest.TestCase):
    """Yahoo reports debtToEquity as a percentage. Getting this backwards would
    reject the entire universe while looking merely strict."""

    def test_typical_indian_leverage_passes(self):
        # RELIANCE reports 36.65, meaning 0.37x — comfortably acceptable.
        self.assertTrue(evaluate(debt_to_equity=36.65).passed)

    def test_value_that_looks_like_a_ratio_is_treated_as_a_percentage(self):
        # 1.5 as a percent is 0.015x, which passes. Were it read as a ratio it
        # would sit right on the ceiling.
        self.assertTrue(evaluate(debt_to_equity=1.5).passed)

    def test_genuinely_levered_company_is_rejected(self):
        self.assertFalse(evaluate(debt_to_equity=314.0).passed)

    def test_boundary_is_inclusive(self):
        self.assertTrue(evaluate(debt_to_equity=screen.MAX_DEBT_TO_EQUITY_PCT).passed)
        self.assertFalse(evaluate(debt_to_equity=screen.MAX_DEBT_TO_EQUITY_PCT + 0.1).passed)


class TestFinancialsRuleset(unittest.TestCase):
    """Banks are screened differently because EV/EBITDA and debt/equity do not
    exist for them — established by the Phase 2 spike, not assumed."""

    def _financial(self, **overrides):
        return evaluate(sector="Financials", peer=make_peer(2.5, "price_to_book"), **overrides)

    def test_financials_are_valued_on_price_to_book(self):
        self.assertEqual(screen.valuation_metric("Financials"), "price_to_book")
        self.assertEqual(screen.valuation_metric("IT"), "trailing_pe")

    def test_financials_do_not_need_debt_to_equity(self):
        candidate = self._financial(debt_to_equity=None)
        self.assertTrue(candidate.passed, candidate.failure_summary)

    def test_financials_are_not_screened_on_free_cash_flow(self):
        # AXISBANK reports -10,847cr FCF and HDFCBANK +106,578cr; neither means
        # what it would for a manufacturer.
        candidate = self._financial(free_cash_flow=-1e11)
        self.assertTrue(candidate.passed, candidate.failure_summary)

    def test_financials_still_face_the_returns_gates(self):
        self.assertFalse(self._financial(return_on_equity=0.01).passed)

    def test_expensive_bank_is_rejected_on_book_value(self):
        self.assertFalse(self._financial(price_to_book=6.2).passed)

    def test_non_financials_still_require_leverage_data(self):
        candidate = evaluate(debt_to_equity=None)
        self.assertIn("data_complete", {f.name for f in candidate.failures})


class TestValuationEdgeCases(unittest.TestCase):
    def test_negative_multiple_is_not_cheapness(self):
        # A loss-making company has no trailing P/E. Yahoo omits it entirely
        # (TMPV in the live run); a negative one must not read as "very cheap".
        candidate = evaluate(trailing_pe=-5.0)
        self.assert_valuation_failed(candidate)

    def test_missing_peer_median_cannot_pass(self):
        candidate = evaluate(peer=make_peer(median=None))
        self.assert_valuation_failed(candidate)

    def test_exactly_at_the_median_passes(self):
        candidate = evaluate(trailing_pe=20.0, peer=make_peer(20.0))
        self.assertNotIn("cheap_vs_peers", {f.name for f in candidate.failures})

    def assert_valuation_failed(self, candidate):
        self.assertIn("cheap_vs_peers", {f.name for f in candidate.failures})


class TestPeerMedians(unittest.TestCase):
    def _universe(self, sizes: dict[str, int]) -> tuple[dict, dict]:
        fundamentals, sectors = {}, {}
        for sector, n in sizes.items():
            for i in range(n):
                symbol = f"{sector[:3].upper()}{i}"
                fundamentals[symbol] = make_fundamentals(ticker=symbol, trailing_pe=10.0 + i)
                sectors[symbol] = sector
        return fundamentals, sectors

    def test_large_sector_uses_its_own_median(self):
        f, s = self._universe({"IT": 6})
        basis = screen.peer_medians(f, s)["IT0"]
        self.assertFalse(basis.is_fallback)
        self.assertEqual(basis.label, "IT")
        self.assertEqual(basis.median, 12.5)

    def test_tiny_sector_falls_back_to_the_index(self):
        # A one-member sector's median is the stock itself, which makes
        # "cheaper than the median" a tautology that always passes.
        f, s = self._universe({"Telecom": 1, "IT": 6})
        basis = screen.peer_medians(f, s)["TEL0"]
        self.assertTrue(basis.is_fallback)
        self.assertEqual(basis.peer_count, 7)
        self.assertIn("Telecom has only 1", basis.label)

    def test_median_ignores_non_positive_multiples(self):
        f, s = self._universe({"IT": 4})
        f["ITNEG"] = make_fundamentals(ticker="ITNEG", trailing_pe=-50.0)
        s["ITNEG"] = "IT"
        basis = screen.peer_medians(f, s)["IT0"]
        self.assertEqual(basis.peer_count, 4)
        self.assertEqual(basis.median, 11.5)

    def test_median_not_mean_survives_an_outlier(self):
        f, s = self._universe({"IT": 4})
        f["ITOUT"] = make_fundamentals(ticker="ITOUT", trailing_pe=713.0)
        s["ITOUT"] = "IT"
        basis = screen.peer_medians(f, s)["IT0"]
        self.assertEqual(basis.median, 12.0)  # a mean would be ~149


class TestScoring(unittest.TestCase):
    def test_components_stay_within_bounds(self):
        for overrides in ({}, {"trailing_pe": 0.5}, {"return_on_equity": 0.8}, {"price": 61.0}):
            c = evaluate(**overrides)
            for name in ("depression", "valuation_discount", "quality", "score"):
                value = getattr(c, name)
                self.assertGreaterEqual(value, 0.0, name)
                self.assertLessEqual(value, 1.0, name)

    def test_deeper_discount_scores_higher(self):
        cheap = evaluate(trailing_pe=8.0)
        dear = evaluate(trailing_pe=19.0)
        self.assertGreater(cheap.valuation_discount, dear.valuation_discount)
        self.assertGreater(cheap.score, dear.score)

    def test_lower_in_the_range_scores_higher(self):
        self.assertGreater(evaluate(price=62.0).depression, evaluate(price=78.0).depression)

    def test_fallback_comparison_is_damped(self):
        """Cross-sector cheapness counts for less than within-sector cheapness.

        Without this, COALINDIA and ONGC scored a ~75% 'discount' against an
        index-wide median and ranked 2nd and 3rd on what is really just how
        commodity producers are priced everywhere.
        """
        within = evaluate(trailing_pe=5.0, peer=make_peer(20.0, fallback=False))
        across = evaluate(trailing_pe=5.0, peer=make_peer(20.0, fallback=True))
        self.assertAlmostEqual(
            across.valuation_discount,
            within.valuation_discount * screen.FALLBACK_DISCOUNT_DAMPING,
        )
        self.assertLess(across.score, within.score)

    def test_quality_saturates_rather_than_running_away(self):
        self.assertEqual(evaluate(return_on_equity=0.60).quality, 1.0)


class TestFiguresContract(unittest.TestCase):
    """`figures()` is the grounding contract with Phase 4: a number not in here
    and yet in a verdict is a hallucination."""

    def test_every_screened_figure_is_exposed(self):
        figures = evaluate().figures()
        for key in (
            "price",
            "trailing_pe",
            "free_cash_flow",
            "debt_to_equity_pct",
            "return_on_equity",
            "profit_margin",
            "drawdown_from_high",
            "position_in_52w_range",
            "return_12m",
            "peer_median",
        ):
            self.assertIn(key, figures)
            self.assertIsNotNone(figures[key], key)

    def test_leverage_is_named_as_a_percentage(self):
        # The key name is the defence against the unit being misread downstream.
        figures = evaluate(debt_to_equity=40.0).figures()
        self.assertEqual(figures["debt_to_equity_pct"], 40.0)
        self.assertNotIn("debt_to_equity", figures)

    def test_absent_figures_are_none_not_zero(self):
        figures = evaluate(sector="Financials", peer=make_peer(2.5, "price_to_book"),
                           ev_to_ebitda=None).figures()
        self.assertIsNone(figures["ev_to_ebitda"])


class TestRun(unittest.TestCase):
    """End-to-end over a fake source, checking orchestration rather than filters."""

    class FakeSource:
        name = "fake"

        def __init__(self, table: dict, failing: set[str] = frozenset()):
            self.table = table
            self.failing = failing

        def fundamentals(self, symbol, *, refresh=False):
            if symbol in self.failing:
                from contrarian.data.source import DataSourceError

                raise DataSourceError(f"stub payload for {symbol}")
            return self.table[symbol]

        def price_stats(self, symbol, *, refresh=False):
            return make_prices(ticker=symbol)

        def news(self, symbol, **kwargs):
            return []

    def _universe(self):
        return (
            Constituent("CHEAP", "Cheap Co", "IT"),
            Constituent("MID", "Mid Co", "IT"),
            Constituent("DEAR", "Dear Co", "IT"),
            Constituent("RICH", "Rich Co", "IT"),
            Constituent("BROKEN", "Broken Co", "IT"),
        )

    def _table(self):
        return {
            "CHEAP": make_fundamentals(ticker="CHEAP", trailing_pe=8.0),
            "MID": make_fundamentals(ticker="MID", trailing_pe=12.0),
            "DEAR": make_fundamentals(ticker="DEAR", trailing_pe=30.0),
            "RICH": make_fundamentals(ticker="RICH", trailing_pe=40.0),
            "BROKEN": make_fundamentals(ticker="BROKEN"),
        }

    def test_survivors_are_ranked_best_first(self):
        source = self.FakeSource(self._table())
        result = screen.run(source, self._universe())
        scores = [c.score for c in result.passed]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(result.passed[0].symbol, "CHEAP")

    def test_expensive_names_are_rejected_not_dropped(self):
        result = screen.run(self.FakeSource(self._table()), self._universe())
        self.assertEqual(result.evaluated, 5)
        self.assertIn("DEAR", {c.symbol for c in result.rejected})

    def test_fetch_failure_is_an_error_not_a_rejection(self):
        """A broken data layer must not be able to masquerade as a strict screen."""
        source = self.FakeSource(self._table(), failing={"BROKEN"})
        result = screen.run(source, self._universe())
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0][0], "BROKEN")
        self.assertNotIn("BROKEN", {c.symbol for c in result.rejected})
        self.assertEqual(result.evaluated, 4)
        self.assertEqual(result.universe_size, 5)

    def test_result_carries_provenance_for_every_survivor(self):
        result = screen.run(self.FakeSource(self._table()), self._universe())
        for c in result.passed:
            self.assertTrue(c.filters)
            self.assertTrue(all(f.detail for f in c.filters))


if __name__ == "__main__":
    unittest.main()
