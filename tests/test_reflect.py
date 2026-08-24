"""Tests for the reflection stage — prompt construction and the asymmetry.

No network and no model. What is testable without one is the part that must not
depend on the model's cooperation: that the note is presented unattributed, that
the reflector sees the same figures the analyst saw, and above all that design
rule 4 holds no matter what the reflector returns.

That last group is the important one. Every "reflection may only lower
conviction" test below is written against a *hostile* reflection — one that
raises confidence, or says DEMOTE and changes nothing — because a rule enforced
only when the model cooperates is not enforced.

    .venv/bin/python -m unittest discover tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contrarian import analyze, enrich, llm, reflect
from contrarian.schema import AnalyzedCandidate, GroundingIssue, Reflection

from test_analyze import make_enriched, make_news, make_verdict  # noqa: E402


def make_reflection(**overrides) -> Reflection:
    base = dict(
        figures_check="All figures appear in the source data.",
        claim_check="The claims follow from the figures.",
        bear_case_engagement="The bear case was argued properly.",
        catalyst_concreteness="A specific catalyst is named.",
        confidence_calibration="About right.",
        weaknesses=[],
        outcome="KEEP",
        adjusted_confidence=50,
    )
    base.update(overrides)
    return Reflection(**base)


def make_analyzed(enriched=None, verdict=None, issues=(), **verdict_fields) -> AnalyzedCandidate:
    return AnalyzedCandidate(
        enriched=enriched if enriched is not None else make_enriched(),
        verdict=verdict if verdict is not None else make_verdict(**verdict_fields),
        grounding_issues=issues,
    )


def apply(analyzed=None, **reflection_fields):
    """Run the asymmetry and return (verdict, confidence, overrides)."""
    return reflect.apply_asymmetry(
        analyzed if analyzed is not None else make_analyzed(),
        make_reflection(**reflection_fields),
    )


class TestAsymmetryIsEnforcedInPython(unittest.TestCase):
    """Design rule 4. Each case uses a reflection that does not cooperate."""

    def test_confidence_cannot_be_raised(self):
        analyzed = make_analyzed(confidence=40)
        _, confidence, overrides = apply(analyzed, adjusted_confidence=95)
        self.assertEqual(confidence, 40)
        self.assertTrue(overrides)

    def test_lowering_confidence_is_allowed(self):
        analyzed = make_analyzed(confidence=70)
        _, confidence, overrides = apply(analyzed, adjusted_confidence=30)
        self.assertEqual(confidence, 30)
        self.assertEqual(overrides, ())

    def test_demote_that_changes_nothing_is_forced_down(self):
        """Models say 'materially weakened' and return the original number."""
        analyzed = make_analyzed(confidence=80)
        _, confidence, overrides = apply(
            analyzed, outcome="DEMOTE", adjusted_confidence=80
        )
        self.assertEqual(confidence, 80 - reflect.DEMOTE_MIN_DROP)
        self.assertTrue(overrides)

    def test_demote_below_the_floor_is_left_alone(self):
        analyzed = make_analyzed(confidence=80)
        _, confidence, overrides = apply(
            analyzed, outcome="DEMOTE", adjusted_confidence=10
        )
        self.assertEqual(confidence, 10)
        self.assertEqual(overrides, ())

    def test_reject_is_capped_hard(self):
        analyzed = make_analyzed(confidence=90)
        _, confidence, _ = apply(analyzed, outcome="REJECT", adjusted_confidence=75)
        self.assertEqual(confidence, reflect.REJECT_CONFIDENCE_CEILING)

    def test_confidence_never_goes_negative(self):
        analyzed = make_analyzed(confidence=5)
        _, confidence, _ = apply(analyzed, outcome="DEMOTE", adjusted_confidence=5)
        self.assertEqual(confidence, 0)

    def test_ungrounded_figure_makes_keep_unreachable(self):
        """A note citing a number that does not exist cannot be kept, whatever
        the reflector thinks. Not a judgment call."""
        analyzed = make_analyzed(
            confidence=80,
            issues=(GroundingIssue("bull_case", 34.7, "Revenue grew 34.7%."),),
        )
        _, confidence, overrides = apply(analyzed, outcome="KEEP", adjusted_confidence=80)
        self.assertLessEqual(confidence, reflect.UNGROUNDED_CONFIDENCE_CEILING)
        self.assertTrue(any("forced to DEMOTE" in o for o in overrides))

    def test_ungrounded_figure_caps_confidence_absolutely(self):
        """The first live gate run: the reflector demolished a note claiming 88
        and returned 55. Every relative rule was satisfied — a 33-point drop —
        and a note built on a fabricated figure still read as a moderate-
        conviction idea, because the floor is anchored to the note's own inflated
        number. The cap has to be absolute."""
        analyzed = make_analyzed(
            confidence=88,
            issues=(GroundingIssue("bull_case", 34.7, "Revenue grew 34.7%."),),
        )
        _, confidence, overrides = apply(
            analyzed, outcome="DEMOTE", adjusted_confidence=55
        )
        self.assertEqual(confidence, reflect.UNGROUNDED_CONFIDENCE_CEILING)
        self.assertTrue(any("capped at" in o for o in overrides))

    def test_the_absolute_cap_does_not_raise_a_lower_confidence(self):
        analyzed = make_analyzed(
            confidence=88,
            issues=(GroundingIssue("bull_case", 34.7, "Revenue grew 34.7%."),),
        )
        _, confidence, _ = apply(analyzed, outcome="REJECT", adjusted_confidence=5)
        self.assertEqual(confidence, 5)

    def test_the_cap_applies_only_when_a_figure_is_ungrounded(self):
        analyzed = make_analyzed(confidence=70)
        _, confidence, _ = apply(analyzed, outcome="KEEP", adjusted_confidence=70)
        self.assertEqual(confidence, 70)


class TestHypotheticalThresholdsAreNotFabrications(unittest.TestCase):
    """From the first full run. ONGC's note was flagged for `70` and `20` — "a
    sustained drop in Brent below $70/bbl" and "a 20%+ rise in the share price" —
    both in `what_would_change_my_mind`, which *asks* for a future threshold.
    Treated as fabrications they tripped the absolute ceiling and gave ONGC the
    harshest treatment in the run, for answering the question it was asked."""

    def _with_hypothetical(self, confidence=55):
        return make_analyzed(
            confidence=confidence,
            issues=(
                GroundingIssue(
                    "what_would_change_my_mind", 70.0, "Brent below $70/bbl."
                ),
            ),
        )

    def test_a_proposed_threshold_does_not_trip_the_absolute_ceiling(self):
        _, confidence, overrides = apply(
            self._with_hypothetical(), outcome="KEEP", adjusted_confidence=55
        )
        self.assertEqual(confidence, 55)
        self.assertEqual(overrides, ())

    def test_a_proposed_threshold_does_not_block_keep(self):
        analyzed = self._with_hypothetical()
        self.assertEqual(analyzed.fabricated_figures, ())
        self.assertEqual(len(analyzed.hypothetical_figures), 1)

    def test_a_fabrication_in_a_factual_field_still_bites(self):
        analyzed = make_analyzed(
            confidence=88,
            issues=(
                GroundingIssue("bull_case", 34.7, "Revenue grew 34.7%."),
                GroundingIssue("what_would_change_my_mind", 70.0, "Brent below $70."),
            ),
        )
        _, confidence, overrides = apply(analyzed, outcome="KEEP", adjusted_confidence=88)
        self.assertEqual(confidence, reflect.UNGROUNDED_CONFIDENCE_CEILING)
        self.assertTrue(any("forced to DEMOTE" in o for o in overrides))

    def test_the_reflector_is_not_told_a_threshold_is_a_fabrication(self):
        prompt = reflect.build_prompt(self._with_hypothetical())
        self.assertIn("hypothetical future levels", prompt)
        self.assertNotIn("2 figure(s) in the note do NOT appear", prompt)
        self.assertIn("traces back to the source data", prompt)

    def test_clean_keep_leaves_everything_alone(self):
        analyzed = make_analyzed(confidence=55)
        verdict, confidence, overrides = apply(analyzed, adjusted_confidence=55)
        self.assertEqual((verdict, confidence, overrides), ("UNCLEAR", 55, ()))


class TestRejectWithdrawsOnlyTheOptimisticVerdict(unittest.TestCase):
    """Every rule moves conviction away from the opportunity — so a badly
    argued warning is still a warning, and stays one."""

    def test_reject_withdraws_a_dislocation_call(self):
        analyzed = make_analyzed(verdict=make_verdict(verdict="TEMPORARY_DISLOCATION"))
        final, _, overrides = apply(analyzed, outcome="REJECT")
        self.assertEqual(final, "UNCLEAR")
        self.assertTrue(any("withdrew" in o for o in overrides))

    def test_reject_does_not_rewrite_a_value_trap_call(self):
        """Flipping VALUE_TRAP to UNCLEAR would *raise* conviction — the one
        thing rule 4 forbids."""
        analyzed = make_analyzed(verdict=make_verdict(verdict="VALUE_TRAP"))
        final, _, _ = apply(analyzed, outcome="REJECT")
        self.assertEqual(final, "VALUE_TRAP")

    def test_demote_does_not_rewrite_the_verdict(self):
        analyzed = make_analyzed(verdict=make_verdict(verdict="TEMPORARY_DISLOCATION"))
        final, _, _ = apply(analyzed, outcome="DEMOTE")
        self.assertEqual(final, "TEMPORARY_DISLOCATION")


class TestAuditPrompt(unittest.TestCase):
    def test_the_note_is_presented_unattributed(self):
        """Design rule 3. Any hint that the reader wrote this, or that a model
        did, invites agreement instead of audit."""
        prompt = reflect.build_prompt(make_analyzed()).lower()
        self.assertIn("analyst note under review", prompt)
        for tell in ("your earlier", "you wrote", "you previously", "your analysis",
                     "your verdict", "ai ", "llm", "language model"):
            self.assertNotIn(tell, prompt, f"prompt hints at authorship: {tell!r}")

    def test_source_data_precedes_the_note(self):
        prompt = reflect.build_prompt(make_analyzed())
        self.assertLess(prompt.index("SOURCE DATA"), prompt.index("ANALYST NOTE"))

    def test_reflector_sees_exactly_the_analyst_figure_sheet(self):
        """The audit is only meaningful if both stages saw the same numbers."""
        enriched = make_enriched(headlines=[make_news("Profit falls on tax hike", 3)])
        sheet = enrich.figure_sheet(enriched)
        self.assertIn(sheet, analyze.build_prompt(enriched))
        self.assertIn(sheet, reflect.build_prompt(make_analyzed(enriched=enriched)))

    def test_every_part_of_the_note_is_shown(self):
        verdict = make_verdict(
            temporary_factor="A tax hike dented Q1.",
            bull_case="Margins held up.",
            value_trap_case="Volumes may not recover.",
            what_would_change_my_mind="Next quarter's volumes.",
            confidence=72,
        )
        prompt = reflect.build_prompt(make_analyzed(verdict=verdict))
        for text in ("A tax hike dented Q1.", "Margins held up.",
                     "Volumes may not recover.", "Next quarter's volumes.", "72"):
            self.assertIn(text, prompt)

    def test_mechanical_findings_are_handed_over(self):
        analyzed = make_analyzed(
            issues=(GroundingIssue("bull_case", 34.7, "Revenue grew 34.7%."),)
        )
        prompt = reflect.build_prompt(analyzed)
        self.assertIn("34.7", prompt)
        self.assertIn("do NOT appear", prompt)

    def test_a_clean_mechanical_check_is_stated_with_its_limit(self):
        """An empty findings section reads as 'not checked'; a clean one that
        does not name its blind spot invites false reassurance."""
        prompt = reflect.build_prompt(make_analyzed())
        self.assertIn("traces back to the source data", prompt)
        self.assertIn("does not prove", prompt.replace("\n  ", " "))

    def test_headline_sourced_figures_are_flagged_as_weaker(self):
        analyzed = AnalyzedCandidate(
            enriched=make_enriched(), verdict=make_verdict(), headline_sourced=(8.44,)
        )
        self.assertIn("8.44", reflect.build_prompt(analyzed))

    def test_keep_is_offered_as_a_valid_outcome(self):
        """A prompt that only rewards finding fault gets fault invented."""
        self.assertIn("KEEP is a valid outcome", reflect.build_prompt(make_analyzed()))


class TestReflectionSchema(unittest.TestCase):
    def test_findings_are_generated_before_the_outcome(self):
        """Same reason Verdict is ordered: an auditor that picks its verdict
        first and then assembles findings is not auditing."""
        fields = list(Reflection.model_fields.keys())
        for finding in ("figures_check", "claim_check", "bear_case_engagement",
                        "catalyst_concreteness", "confidence_calibration"):
            self.assertLess(fields.index(finding), fields.index("outcome"), finding)

    def test_generated_schema_preserves_that_order(self):
        properties = list(llm.strict_schema(Reflection)["properties"].keys())
        self.assertLess(properties.index("claim_check"), properties.index("outcome"))

    def test_schema_is_strict_and_flat(self):
        import json

        schema = llm.strict_schema(Reflection)
        self.assertFalse(schema.get("additionalProperties", True))
        self.assertNotIn("$ref", json.dumps(schema))

    def test_adjusted_confidence_range_is_enforced(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            make_reflection(adjusted_confidence=101)


class TestReflectionIsItselfAudited(unittest.TestCase):
    """An auditor that invents figures is no better than what it audits."""

    def test_invented_figure_in_the_audit_is_caught(self):
        reflected = reflect.ReflectedCandidate(
            analyzed=make_analyzed(),
            reflection=make_reflection(claim_check="The 913.5% growth claim is wrong."),
            final_verdict="UNCLEAR",
            final_confidence=50,
        )
        issues, _ = reflect.check_reflection_grounding(reflected)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].value, 913.5)
        self.assertEqual(issues[0].field, "claim_check")

    def test_quoting_real_figures_back_is_clean(self):
        reflected = reflect.ReflectedCandidate(
            analyzed=make_analyzed(),
            reflection=make_reflection(figures_check="ROE of 20.0% is in the data."),
            final_verdict="UNCLEAR",
            final_confidence=50,
        )
        issues, _ = reflect.check_reflection_grounding(reflected)
        self.assertEqual(issues, ())

    def test_reporting_the_notes_fabricated_figure_is_not_itself_a_fabrication(self):
        """Three live false positives in one run. The reflector was flagged for
        naming the invented 34.7 it had just caught, and for quoting "stated
        confidence 88/100" while assessing calibration. All three were shown to
        it in the prompt."""
        analyzed = make_analyzed(
            confidence=88,
            issues=(GroundingIssue("bull_case", 34.7, "Revenue grew 34.7%."),),
        )
        reflected = reflect.ReflectedCandidate(
            analyzed=analyzed,
            reflection=make_reflection(
                figures_check="The note cites revenue growth of 34.7%, which is "
                "absent from the source data.",
                confidence_calibration="88/100 is unjustified.",
            ),
            final_verdict="UNCLEAR",
            final_confidence=25,
        )
        issues, _ = reflect.check_reflection_grounding(reflected)
        self.assertEqual(issues, (), [str(i) for i in issues])

    def test_a_figure_shown_in_neither_the_data_nor_the_note_is_still_caught(self):
        """The widened pool must not become a pool that allows everything."""
        reflected = reflect.ReflectedCandidate(
            analyzed=make_analyzed(),
            reflection=make_reflection(claim_check="Peers trade at 462.9x."),
            final_verdict="UNCLEAR",
            final_confidence=50,
        )
        issues, _ = reflect.check_reflection_grounding(reflected)
        self.assertEqual([i.value for i in issues], [462.9])


class TestReflectedCandidate(unittest.TestCase):
    def _reflected(self, original: int, final: int, **fields):
        analyzed = make_analyzed(confidence=original)
        return reflect.ReflectedCandidate(
            analyzed=analyzed,
            reflection=make_reflection(**fields),
            final_verdict="UNCLEAR",
            final_confidence=final,
        )

    def test_confidence_drop_is_reported(self):
        self.assertEqual(self._reflected(80, 55).confidence_drop, 25)

    def test_survived_requires_keep_and_no_overrides(self):
        self.assertTrue(self._reflected(50, 50).survived)
        self.assertFalse(self._reflected(50, 35, outcome="DEMOTE").survived)

    def test_an_overruled_keep_did_not_survive(self):
        reflected = reflect.ReflectedCandidate(
            analyzed=make_analyzed(),
            reflection=make_reflection(outcome="KEEP"),
            final_verdict="UNCLEAR",
            final_confidence=35,
            overrides=("forced to DEMOTE",),
        )
        self.assertFalse(reflected.survived)


class TestTheWeakNoteGate(unittest.TestCase):
    """The Phase 5 gate's input, checked without a model.

    The live gate is `scripts/run_reflect.py --weak ITC`, which needs one. What
    is testable here is that the note really is weak in the ways claimed — if the
    invented figure ever became a real one, the gate would quietly stop testing
    anything.
    """

    def test_the_weak_note_contains_an_ungrounded_figure(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from run_reflect import WEAK_VERDICT

        issues, _ = analyze.check_grounding(WEAK_VERDICT, make_enriched())
        self.assertTrue(issues, "the weak note's invented figure is no longer invented")
        self.assertIn(34.7, [i.value for i in issues])

    def test_the_weak_note_cannot_be_kept(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from run_reflect import WEAK_VERDICT

        enriched = make_enriched()
        issues, _ = analyze.check_grounding(WEAK_VERDICT, enriched)
        analyzed = AnalyzedCandidate(
            enriched=enriched, verdict=WEAK_VERDICT, grounding_issues=issues
        )
        # Even a maximally generous reflector cannot leave this one intact.
        _, confidence, overrides = reflect.apply_asymmetry(
            analyzed, make_reflection(outcome="KEEP", adjusted_confidence=100)
        )
        self.assertLess(confidence, WEAK_VERDICT.confidence)
        self.assertTrue(overrides)


if __name__ == "__main__":
    unittest.main()
