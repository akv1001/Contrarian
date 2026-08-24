"""Tests for enrichment, prompt construction, and the grounding audit.

No network and no model. What is testable here is the machinery around the LLM
call: that the arithmetic is done before the prompt is built, that the prompt
contains only what it should, and that the grounding check catches invented
figures while tolerating the transformations a model legitimately makes.

The grounding cases are drawn from real false positives and false negatives seen
in live runs, which is why several of them look oddly specific.

    .venv/bin/python -m unittest discover tests -v
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrarian import analyze, enrich, llm, screen
from contrarian.data.source import NewsItem
from contrarian.schema import EnrichedCandidate, Fact, Verdict
from contrarian.universe import Constituent

from test_screen import make_fundamentals, make_peer, make_prices  # noqa: E402

NOW = datetime(2026, 8, 23, tzinfo=timezone.utc)


def make_candidate(sector: str = "Materials", **overrides):
    constituent = Constituent("TEST", "Test Company", sector)
    return screen.evaluate(
        constituent,
        make_fundamentals(**overrides),
        make_prices(),
        make_peer(metric=screen.valuation_metric(sector)),
    )


def make_news(title: str, days_ago: float = 5.0) -> NewsItem:
    return NewsItem(
        title=title,
        publisher="Test Wire",
        published_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        source="test",
    )


def make_enriched(headlines=(), **overrides) -> EnrichedCandidate:
    candidate = make_candidate(**overrides)
    return EnrichedCandidate(
        candidate=candidate,
        facts=tuple(enrich.build_facts(candidate)),
        headlines=tuple(headlines),
        news_source="test",
    )


def make_verdict(**overrides) -> Verdict:
    base = dict(
        temporary_factor="A one-off event.",
        bull_case="The business is sound.",
        value_trap_case="It may be structurally impaired.",
        verdict="UNCLEAR",
        confidence=50,
        what_would_change_my_mind="Next quarter's cash flow.",
    )
    base.update(overrides)
    return Verdict(**base)


class TestFactsDoTheArithmetic(unittest.TestCase):
    """Design rule 1: the model never calculates. Enrichment is where that is
    made true rather than merely requested."""

    def _text(self, label: str, **overrides) -> str:
        facts = enrich.build_facts(make_candidate(**overrides))
        match = next((f for f in facts if f.label == label), None)
        self.assertIsNotNone(match, f"no fact labelled {label!r}")
        return match.text

    def test_peer_discount_is_precomputed(self):
        # The spike's failure: given "P/E 12.0, median 20.0" the model wrote
        # "20.0% below sector median". Handing it the resolved discount is the fix.
        text = self._text("Valuation vs peers", trailing_pe=12.0)
        self.assertIn("40% discount", text)

    def test_premium_is_named_as_a_premium(self):
        text = self._text("Valuation vs peers", trailing_pe=25.0)
        self.assertIn("premium", text)
        self.assertNotIn("discount", text)

    def test_fallback_basis_is_disclosed_in_the_text(self):
        candidate = make_candidate()
        facts = enrich.build_facts(
            screen.evaluate(
                Constituent("TEST", "Test Company", "Materials"),
                make_fundamentals(),
                make_prices(),
                make_peer(20.0, "trailing_pe", fallback=True),
            )
        )
        text = next(f for f in facts if f.label == "Valuation vs peers").text
        self.assertIn("index-wide", text)
        self.assertIn("too few members", text)

    def test_leverage_is_converted_out_of_percent(self):
        # Yahoo's 40.0 means 0.40x. The model must never see the raw percentage.
        text = self._text("Leverage", debt_to_equity=40.0)
        self.assertIn("0.40x", text)
        self.assertNotIn("40.0%", text)

    def test_drawdown_is_stated_not_implied(self):
        text = self._text("Drawdown")
        self.assertIn("30.0%", text)

    def test_negative_cash_flow_is_labelled_in_words(self):
        text = self._text("Free cash flow", free_cash_flow=-4.2e10)
        self.assertIn("negative", text)

    def test_every_fact_records_its_own_numbers(self):
        for fact in enrich.build_facts(make_candidate()):
            for number in fact.numbers:
                self.assertIsInstance(number, float)


class TestPrompt(unittest.TestCase):
    def test_prompt_contains_the_figures(self):
        prompt = analyze.build_prompt(make_enriched())
        self.assertIn("VERIFIED FIGURES", prompt)
        self.assertIn("Do not recompute", prompt)
        self.assertIn("TEST", prompt)

    def test_headlines_appear_with_age_and_publisher(self):
        prompt = analyze.build_prompt(
            make_enriched(headlines=[make_news("Profit falls on tax hike", 3)])
        )
        self.assertIn("Profit falls on tax hike", prompt)
        self.assertIn("3d ago", prompt)
        self.assertIn("Test Wire", prompt)

    def test_absent_news_is_stated_rather_than_left_silent(self):
        """An unexplained gap invites the model to fill it from memory."""
        prompt = analyze.build_prompt(make_enriched(headlines=[]))
        self.assertIn("None retrieved", prompt)
        self.assertIn("Do not speculate", prompt)

    def test_task_demands_the_bear_case_before_the_verdict(self):
        prompt = analyze.build_prompt(make_enriched())
        bear = prompt.index("value trap")
        verdict = prompt.index("Only then give your verdict")
        self.assertLess(bear, verdict)


class TestSchemaOrderEnforcesRule5(unittest.TestCase):
    """Constrained decoding emits fields in schema order, so ordering the model
    is what forces the bear case out before the verdict exists."""

    def test_value_trap_case_precedes_verdict(self):
        fields = list(Verdict.model_fields.keys())
        self.assertLess(fields.index("value_trap_case"), fields.index("verdict"))
        self.assertLess(fields.index("bull_case"), fields.index("verdict"))

    def test_generated_schema_preserves_that_order(self):
        properties = list(llm.strict_schema(Verdict)["properties"].keys())
        self.assertLess(properties.index("value_trap_case"), properties.index("verdict"))

    def test_schema_is_strict_and_flat(self):
        """Ollama compiles the schema to a grammar and does not resolve $ref."""
        import json

        schema = llm.strict_schema(Verdict)
        self.assertFalse(schema.get("additionalProperties", True))
        self.assertNotIn("$defs", schema)
        self.assertNotIn("$ref", json.dumps(schema))

    def test_confidence_range_is_enforced(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            make_verdict(confidence=150)


class TestGrounding(unittest.TestCase):
    def _issues(self, enriched, **fields):
        issues, _ = analyze.check_grounding(make_verdict(**fields), enriched)
        return issues

    def _headline_sourced(self, enriched, **fields):
        _, sourced = analyze.check_grounding(make_verdict(**fields), enriched)
        return sourced

    def test_invented_figure_is_caught(self):
        issues = self._issues(make_enriched(), bull_case="Revenue grew 847.3% last year.")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].value, 847.3)
        self.assertEqual(issues[0].field, "bull_case")

    def test_issue_records_the_sentence_for_judgment(self):
        issues = self._issues(make_enriched(), bull_case="Fine. Margin hit 91.7% though.")
        self.assertIn("91.7", issues[0].context)
        self.assertNotIn("Fine.", issues[0].context)

    def test_real_figures_are_accepted(self):
        enriched = make_enriched()
        self.assertEqual(self._issues(enriched, bull_case="ROE is 20.0%."), ())

    def test_rounding_to_fewer_decimals_is_not_a_hallucination(self):
        """Live false positive: input 0.033, model wrote 0.03. A 2% relative
        tolerance rejects that at 9% apart; decimal-place rounding accepts it."""
        enriched = make_enriched(debt_to_equity=3.3)
        self.assertEqual(self._issues(enriched, bull_case="Leverage is 0.03x."), ())

    def test_rounding_a_percentage_is_not_a_hallucination(self):
        enriched = make_enriched()
        self.assertEqual(self._issues(enriched, bull_case="Down 30% over the year."), ())

    def test_time_window_idioms_are_not_figures(self):
        """Live false positive: '52-week low' flagged 52 as an invented number."""
        enriched = make_enriched()
        for phrase in ("near its 52-week low", "Q1 was weak", "over a 3-month span",
                       "FY2026 guidance"):
            self.assertEqual(self._issues(enriched, bull_case=phrase), (), phrase)

    def test_small_integers_are_ignored(self):
        enriched = make_enriched()
        self.assertEqual(
            self._issues(enriched, value_trap_case="Negative for 3 straight quarters."), ()
        )

    def test_headline_figures_are_grounded_but_tracked(self):
        """Live case: Coal India's verdict cited 'production rises 8.44%' from a
        real headline. Legitimate, but not verified data."""
        enriched = make_enriched(
            headlines=[make_news("Coal India output rises 8.44% in July")]
        )
        self.assertEqual(self._issues(enriched, bull_case="Production rose 8.44%."), ())
        self.assertIn(8.44, self._headline_sourced(enriched, bull_case="Production rose 8.44%."))

    def test_headline_age_marker_can_be_cited(self):
        enriched = make_enriched(headlines=[make_news("Something happened", 17)])
        self.assertEqual(self._issues(enriched, temporary_factor="Per the 17d ago story."), ())

    def test_all_prose_fields_are_checked(self):
        enriched = make_enriched()
        for field_name in ("temporary_factor", "bull_case", "value_trap_case",
                           "what_would_change_my_mind"):
            issues = self._issues(enriched, **{field_name: "It moved 456.7%."})
            self.assertEqual(len(issues), 1, field_name)
            self.assertEqual(issues[0].field, field_name)

    def test_thousands_separators_are_parsed(self):
        enriched = make_enriched(free_cash_flow=5e9)  # Rs 500 crore
        self.assertEqual(self._issues(enriched, bull_case="Free cash flow of Rs 500 crore."), ())

    def test_a_clean_report_does_not_prove_a_correct_analysis(self):
        """The documented blind spot, asserted so it cannot be forgotten.

        Every number here is real; the relationship between them is fabricated.
        This is the spike's actual failure and the check passes it cleanly, which
        is why enrichment pre-computes and why Phase 5 exists.
        """
        enriched = make_enriched(trailing_pe=12.0)
        clean = self._issues(
            enriched, bull_case="A P/E of 12.0 is 20.0% below the median of 20.0."
        )
        self.assertEqual(clean, ())


class TestEnrichHeadlineSelection(unittest.TestCase):
    class FakeSource:
        name = "fake"

        def __init__(self, items):
            self.items = items

        def fundamentals(self, symbol, *, refresh=False):
            return make_fundamentals(ticker=symbol)

        def price_stats(self, symbol, *, refresh=False):
            return make_prices(ticker=symbol)

        def news(self, symbol, *, company_name=None, limit=10, refresh=False):
            return self.items

    def test_newest_headlines_come_first(self):
        source = self.FakeSource([make_news("old", 40), make_news("new", 1)])
        enriched = enrich.enrich(make_candidate(), source)
        self.assertEqual(enriched.headlines[0].title, "new")

    def test_stale_headlines_are_dropped(self):
        source = self.FakeSource([make_news("ancient", 400), make_news("recent", 2)])
        enriched = enrich.enrich(make_candidate(), source)
        self.assertEqual([h.title for h in enriched.headlines], ["recent"])

    def test_headline_count_is_capped(self):
        source = self.FakeSource([make_news(f"story {i}", i) for i in range(60)])
        enriched = enrich.enrich(make_candidate(), source, max_headlines=5)
        self.assertEqual(len(enriched.headlines), 5)

    def test_news_failure_does_not_sink_the_candidate(self):
        from contrarian.data.source import DataSourceError

        class Broken(self.FakeSource):
            def news(self, *a, **k):
                raise DataSourceError("feed down")

        enriched = enrich.enrich(make_candidate(), Broken([]))
        self.assertEqual(enriched.headlines, ())
        self.assertTrue(enriched.facts)


if __name__ == "__main__":
    unittest.main()
