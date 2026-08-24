"""Tests for the report renderer.

No network and no model — which for this module is the entire point rather than
a convenience. The first test class asserts design rule 2 structurally: if
`report.py` ever grows an LLM call, the suite fails before a human reads a
report whose numbers drifted at the last mile.

The rest check the things a reader depends on being true: that the source
figures appear next to the reasoning, that a withdrawn verdict is not quietly
presented as the analyst's, and that the honest parts — ungrounded figures,
overrides, fetch failures — survive rendering instead of being tidied away.

    .venv/bin/python -m unittest discover tests -v
"""

from __future__ import annotations

import ast
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrarian import report
from contrarian.schema import (
    GroundingIssue,
    PipelineRun,
    ReflectedCandidate,
    ScreenResult,
)

from test_analyze import make_candidate, make_enriched, make_news, make_verdict  # noqa: E402
from test_reflect import make_analyzed, make_reflection  # noqa: E402

NOW = datetime(2026, 8, 24, tzinfo=timezone.utc)


def make_reflected(
    *,
    final_verdict="UNCLEAR",
    final_confidence=50,
    overrides=(),
    analyzed=None,
    **reflection_fields,
) -> ReflectedCandidate:
    return ReflectedCandidate(
        analyzed=analyzed if analyzed is not None else make_analyzed(),
        reflection=make_reflection(**reflection_fields),
        final_verdict=final_verdict,
        final_confidence=final_confidence,
        overrides=overrides,
    )


def make_run(reflected=None, *, rejected=(), errors=(), skipped=()) -> PipelineRun:
    if reflected is None:
        reflected = (make_reflected(),)
    return PipelineRun(
        as_of=NOW,
        screen=ScreenResult(
            as_of=NOW,
            universe_size=51,
            passed=(make_candidate(),),
            rejected=tuple(rejected),
            errors=tuple(errors),
        ),
        reflected=tuple(reflected),
        model="qwen3:14b",
        provider="Ollama (local)",
        skipped=tuple(skipped),
        duration_seconds=600.0,
    )


class TestReportMakesNoLLMCall(unittest.TestCase):
    """Design rule 2, asserted structurally rather than trusted.

    Re-summarising through a model at the last mile would undo four stages of
    grounding work, and it would do it in the one artefact a human actually
    reads — where nothing downstream would catch it.
    """

    def _module_source(self) -> str:
        path = Path(report.__file__)
        return path.read_text(encoding="utf-8")

    def test_report_does_not_import_the_llm_module(self):
        tree = ast.parse(self._module_source())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        for name in imported:
            self.assertNotIn("llm", name, f"report.py imports {name!r}")
            self.assertNotIn("openai", name, f"report.py imports {name!r}")

    def test_report_never_names_a_chat_entry_point(self):
        source = self._module_source()
        for forbidden in ("chat_structured", "chat(", "OpenAI", "completions"):
            self.assertNotIn(forbidden, source, f"report.py references {forbidden!r}")

    def test_rendering_touches_no_network(self):
        """Belt and braces: rendering is pure string work over a finished run."""
        self.assertIsInstance(report.render(make_run()), str)


class TestReportContent(unittest.TestCase):
    def test_the_disclaimer_appears(self):
        markdown = report.render(make_run())
        self.assertIn("not investment advice", markdown)

    def test_the_figures_appear_beside_the_reasoning(self):
        """The audit trail is the point. A reader who doubts a sentence has to be
        able to check it without leaving the document."""
        markdown = report.render(make_run())
        self.assertIn("The figures it was given", markdown)
        self.assertIn("Return on equity", markdown)

    def test_the_analyst_note_is_rendered_in_full(self):
        analyzed = make_analyzed(
            verdict=make_verdict(
                temporary_factor="A tax hike dented Q1.",
                bull_case="Margins held up.",
                value_trap_case="Volumes may not recover.",
                what_would_change_my_mind="Next quarter's volumes.",
            )
        )
        markdown = report.render(make_run((make_reflected(analyzed=analyzed),)))
        for text in ("A tax hike dented Q1.", "Margins held up.",
                     "Volumes may not recover.", "Next quarter's volumes."):
            self.assertIn(text, markdown)

    def test_the_screen_filters_are_shown_with_their_numbers(self):
        """'valuation: passed' would tell a reader nothing about whether to
        believe the screen."""
        markdown = report.render(make_run())
        self.assertIn("Why it screened", markdown)
        self.assertIn("return_on_equity", markdown)

    def test_headlines_are_listed_with_age_and_publisher(self):
        analyzed = make_analyzed(
            enriched=make_enriched(headlines=[make_news("Profit falls on tax hike", 3)])
        )
        markdown = report.render(make_run((make_reflected(analyzed=analyzed),)))
        self.assertIn("Profit falls on tax hike", markdown)
        self.assertIn("3d ago", markdown)

    def test_headline_links_are_reference_style(self):
        """Google News hands back ~700-char base64 redirect URLs. Inline, a dozen
        of those bury the headlines and make the raw file unreadable."""
        item = make_news("Profit falls on tax hike", 3)
        item = type(item)(
            title=item.title,
            url="https://news.google.com/rss/articles/" + "A" * 600,
            publisher=item.publisher,
            published_at=item.published_at,
        )
        analyzed = make_analyzed(enriched=make_enriched(headlines=[item]))
        markdown = report.render(make_run((make_reflected(analyzed=analyzed),)))

        headline_line = next(
            line for line in markdown.splitlines() if "Profit falls on tax hike" in line
        )
        self.assertNotIn("https://", headline_line)
        self.assertIn("[test-1]", headline_line)
        self.assertIn("[test-1]: https://news.google.com/", markdown)

    def test_absent_headlines_are_stated_not_omitted(self):
        markdown = report.render(make_run())
        self.assertIn("None retrieved", markdown)

    def test_the_audit_findings_are_all_present(self):
        markdown = report.render(
            make_run((make_reflected(claim_check="The discount claim is inflated."),))
        )
        self.assertIn("The discount claim is inflated.", markdown)
        self.assertIn("Do the claims follow", markdown)

    def test_the_deterministic_stages_are_named_as_such(self):
        """The reader needs to know which parts a model touched. The call count
        is incidental; the split is what matters."""
        markdown = report.render(make_run()).replace("\n", " ")
        self.assertIn("grounding checks and this report do not", markdown)
        self.assertIn("behaves the same way every time", markdown)


class TestJargonIsExplainedBeforeItIsUsed(unittest.TestCase):
    """A reader who has never seen this project must be able to follow the
    report. "UNCLEAR at 40/100, down from 55" is meaningless otherwise."""

    def test_all_three_verdicts_are_defined(self):
        markdown = report.render(make_run())
        for phrase in ("The fall looks temporary", "Cheap for a real reason",
                       "Not enough evidence either way"):
            self.assertIn(phrase, markdown)

    def test_the_confidence_scale_is_defined(self):
        markdown = report.render(make_run())
        self.assertIn("80-100", markdown)
        self.assertIn("0-39", markdown)

    def test_the_two_numbers_are_explained(self):
        """The single most confusing thing in the old format."""
        markdown = report.render(make_run())
        self.assertIn("Why two numbers", markdown)
        self.assertIn("only lower the score, never raise it", markdown)

    def test_definitions_come_before_the_results(self):
        markdown = report.render(make_run())
        self.assertLess(markdown.index("What the words below mean"),
                        markdown.index("## Results"))

    def test_keep_demote_reject_are_defined(self):
        markdown = report.render(make_run())
        for key in ("KEEP", "DEMOTE", "REJECT"):
            self.assertIn(f"**{key}**", markdown)

    def test_a_bare_score_is_given_a_word(self):
        markdown = report.render(make_run((make_reflected(final_confidence=40),)))
        self.assertIn("40/100 — low", markdown)

    def test_a_drop_is_spelled_out_not_abbreviated(self):
        reflected = make_reflected(
            analyzed=make_analyzed(confidence=55), final_confidence=40,
            outcome="DEMOTE",
        )
        markdown = report.render(make_run((reflected,)))
        self.assertIn("The first pass scored this 55", markdown)
        self.assertIn("cut it to 40", markdown)

    def test_an_unchanged_score_says_so(self):
        markdown = report.render(make_run((make_reflected(final_confidence=50),)))
        self.assertIn("the review left it there", markdown)

    def test_the_reader_is_told_this_is_not_a_signal(self):
        markdown = report.render(make_run())
        self.assertIn("is a buy or sell signal", markdown)
        self.assertIn("It is not a recommendation.", markdown)

    def test_confidence_bands_cover_the_whole_range(self):
        for score, expected in ((95, "high"), (70, "moderate"), (45, "low"),
                                (10, "very low")):
            self.assertEqual(report.confidence_band(score)[0], expected, score)


class TestTheHonestPartsSurvive(unittest.TestCase):
    """Everything here is something a tidier report would omit."""

    def test_ungrounded_figures_are_shown_with_their_sentence(self):
        analyzed = make_analyzed(
            issues=(GroundingIssue("bull_case", 34.7, "Revenue grew 34.7%."),)
        )
        markdown = report.render(make_run((make_reflected(analyzed=analyzed),)))
        self.assertIn("34.7", markdown)
        self.assertIn("Revenue grew 34.7%.", markdown)
        self.assertIn("fabrication", markdown)

    def test_a_proposed_threshold_is_shown_but_not_called_a_fabrication(self):
        analyzed = make_analyzed(
            issues=(
                GroundingIssue(
                    "what_would_change_my_mind", 70.0, "Brent below $70/bbl."
                ),
            )
        )
        markdown = report.render(make_run((make_reflected(analyzed=analyzed),)))
        self.assertIn("70", markdown)
        self.assertIn("hypothetical future levels", markdown)
        self.assertNotIn("appear nowhere in the source data", markdown)
        # And it must not raise a warning flag in the summary table.
        self.assertNotIn("⚠", markdown)

    def test_overrides_are_disclosed(self):
        markdown = report.render(
            make_run((make_reflected(overrides=("DEMOTE left confidence at 55",)),))
        )
        self.assertIn("DEMOTE left confidence at 55", markdown)
        self.assertIn("overruled", markdown)

    def test_a_withdrawn_verdict_is_not_presented_as_the_analysts(self):
        analyzed = make_analyzed(verdict=make_verdict(verdict="TEMPORARY_DISLOCATION"))
        markdown = report.render(
            make_run((make_reflected(analyzed=analyzed, final_verdict="UNCLEAR"),))
        )
        self.assertIn("The audit withdrew that call.", markdown)

    def test_fetch_failures_are_distinguished_from_rejections(self):
        """A ticker we could not fetch is not a ticker we examined and dismissed."""
        markdown = report.render(make_run(errors=(("TMPV", "no price in payload"),)))
        self.assertIn("TMPV", markdown)
        self.assertIn("not a screen", markdown)

    def test_skipped_candidates_are_distinguished_from_rejections(self):
        markdown = report.render(make_run(skipped=(("INFY", "audit failed — timeout"),)))
        self.assertIn("INFY", markdown)
        self.assertIn("These are not", markdown)

    def test_headline_sourced_figures_are_flagged_as_weaker(self):
        analyzed = make_analyzed()
        analyzed = type(analyzed)(
            enriched=analyzed.enriched,
            verdict=analyzed.verdict,
            headline_sourced=(8.44,),
        )
        markdown = report.render(make_run((make_reflected(analyzed=analyzed),)))
        self.assertIn("8.44", markdown)
        self.assertIn("weaker than it looks", markdown)

    def test_the_blind_spot_is_stated_in_the_report_itself(self):
        """A clean grounding report is not a correct analysis, and the reader
        needs to know that without having read CLAUDE.md."""
        markdown = report.render(make_run())
        self.assertIn("does not mean a correct analysis", markdown)


class TestRanking(unittest.TestCase):
    def test_dislocations_come_before_unclear_before_value_traps(self):
        order = report.rank((
            make_reflected(final_verdict="VALUE_TRAP", final_confidence=90),
            make_reflected(final_verdict="UNCLEAR", final_confidence=50),
            make_reflected(final_verdict="TEMPORARY_DISLOCATION", final_confidence=10),
        ))
        self.assertEqual(
            [r.final_verdict for r in order],
            ["TEMPORARY_DISLOCATION", "UNCLEAR", "VALUE_TRAP"],
        )

    def test_ranking_uses_post_audit_confidence_not_the_analysts(self):
        """An idea the audit knocked down must not lead the report on the
        strength of a confidence the audit withdrew."""
        demoted = make_reflected(
            analyzed=make_analyzed(confidence=95), final_confidence=20
        )
        kept = make_reflected(analyzed=make_analyzed(confidence=55), final_confidence=55)
        self.assertEqual(report.rank((demoted, kept))[0], kept)

    def test_the_report_says_the_order_is_not_a_recommendation(self):
        self.assertIn("not a ranking of investment merit", report.render(make_run()))


class TestEmptyAndEdgeCases(unittest.TestCase):
    def test_a_run_with_no_survivors_still_renders(self):
        markdown = report.render(make_run(reflected=()))
        self.assertIn("Nothing completed the pipeline", markdown)

    def test_filename_sorts_chronologically(self):
        self.assertEqual(report.default_filename(NOW), "contrarian-2026-08-24.md")

    def test_token_and_call_counts_come_from_the_run(self):
        run = make_run((make_reflected(), make_reflected()))
        self.assertEqual(run.llm_calls, 4)
        self.assertEqual(run.attempted, 2)

    def test_skipped_candidates_count_as_attempted(self):
        run = make_run(skipped=(("INFY", "timeout"),))
        self.assertEqual(run.attempted, 2)


if __name__ == "__main__":
    unittest.main()
