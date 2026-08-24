"""Stage 4 — audit the verdicts. The second stage that uses a model.

Stage 3 produces a judgment. This stage asks a second, *uncontaminated* reader
whether that judgment survives contact with the same source data.

Three things make it work, and all three are structural rather than requested:

1. **Fresh context, third-party framing.** `llm.chat()` and `chat_structured()`
   are stateless by construction, and the note is presented unattributed. A model
   shown its own visible reasoning and asked "are you sure?" anchors on it and
   agrees. Design rule 3.
2. **The same figure sheet, byte for byte.** `enrich.figure_sheet()` is called by
   both stages, so "the note cites a figure that is not in the data" cannot mean
   "the reflector was shown a different sheet".
3. **The asymmetry is arithmetic, not instruction.** `apply_asymmetry()` clamps
   the result in Python. Design rule 4 says reflection may only lower conviction;
   a model asked politely will eventually raise a number anyway.

**What this stage is actually for.** `analyze.check_grounding()` already catches
invented figures mechanically, and does it better than a model would. The gap it
cannot close is a *real* figure supporting a *fabricated* claim — the spike's
"P/E 17.0 is 69.2% below sector median", where both numbers are real and the
relationship is invented. That is `claim_check`, and it is why this stage exists.
It is also why the figure sheet matters so much here: enrichment has already
resolved the correct comparison ("a 75% discount"), so catching a fabricated one
is a lookup against the sheet rather than arithmetic the reflector has to do.
"""

from __future__ import annotations

from .analyze import check_grounding
from .enrich import figure_sheet
from .llm import LLMError, chat_structured
from .schema import AnalyzedCandidate, Reflection, ReflectedCandidate, numbers_in

SYSTEM = """You are auditing an equity research note written by someone else. You
have the same source data the author had.

Your job is not to re-analyse the company. It is to judge whether the note's
reasoning is supported by the data in front of you.

What you are looking for, in order of how much it matters:

1. A claim that does not follow from the figures. This is the failure that
   matters most and the one that is easiest to miss, because it hides behind
   real numbers. A note can quote two genuine figures and assert a relationship
   between them that is false. Every comparison you need has already been
   calculated and is stated in the source data — check the note's comparisons
   against it. Do not calculate anything yourself.
2. A figure that is not in the source data at all.
3. A value-trap argument that was raised and waved away rather than argued.
4. A "temporary" factor with no mechanism — nothing specific, nothing that
   resolves, no reason the price should recover rather than simply being low.
5. A confidence that the evidence does not support.

Decide an outcome:
  KEEP    the reasoning is sound and the conclusion follows from the data
  DEMOTE  usable, but materially weakened by what you found
  REJECT  the reasoning does not support presenting this as an opportunity

Be specific. "The bull case is weak" is not a finding; "the bull case cites
revenue growth but ignores the negative free cash flow in the same data" is.

You may only lower confidence, never raise it. Missing an opportunity is cheap.
Recommending a value trap is not. When the two are balanced, lower it."""

# Budget. Reflection writes five prose findings plus a list, so it runs longer
# than a verdict — measured around 250-400 completion tokens. Running out
# mid-JSON is unparseable, which is a far worse failure than a long response.
MAX_COMPLETION_TOKENS = 1600

# A DEMOTE that leaves confidence untouched is not a demotion. If the reflector
# says "materially weakened" and then hands back the same number, Python applies
# the drop it declined to.
DEMOTE_MIN_DROP = 15

# REJECT means the reasoning does not support the conclusion. Whatever the
# reflector types into adjusted_confidence, the result cannot read as a
# moderately-held view.
REJECT_CONFIDENCE_CEILING = 20

# A note citing a figure that exists nowhere in the source data is capped here,
# whatever the reflector decides and whatever the note claimed for itself.
#
# This ceiling is absolute, and the first live gate run is why. The weak note
# claimed confidence 88; the reflector demolished it in detail and returned 55.
# Every relative rule was satisfied — a 33-point drop, well past DEMOTE_MIN_DROP
# — and the fabricated-figure note still came out as a moderate-conviction idea.
# The flaw is that every relative rule is anchored to a number the note chose for
# itself, so an inflated starting confidence drags the floor up with it. An
# objective defect needs an objective ceiling.
UNGROUNDED_CONFIDENCE_CEILING = 25

# Prose fields of Reflection that get the same numeric grounding audit the
# verdict gets. An auditor that invents figures is not an auditor.
REFLECTION_FIELDS = (
    "figures_check",
    "claim_check",
    "bear_case_engagement",
    "catalyst_concreteness",
    "confidence_calibration",
)


def build_prompt(analyzed: AnalyzedCandidate) -> str:
    """Assemble the audit request: source data first, then the note.

    Order is deliberate. The data comes first so it is read as the ground truth
    and the note as the claim about it, rather than the other way round.

    The note is unattributed on purpose — no "your earlier analysis", no mention
    of how it was produced. See design rule 3.
    """
    enriched = analyzed.enriched
    verdict = analyzed.verdict

    lines = [
        f"Company: {enriched.name} ({enriched.symbol}), listed on the NSE.",
        f"Sector: {enriched.sector}",
        "",
        "SOURCE DATA — this is the ground truth. Every comparison in it is already",
        "calculated. The note below was written from exactly this data and nothing",
        "else, so any figure in the note that is not here did not come from a source.",
        "",
        figure_sheet(enriched),
        "",
        "=" * 70,
        "",
        "ANALYST NOTE UNDER REVIEW",
        "",
        f"  Temporary factor claimed:",
        f"    {verdict.temporary_factor}",
        "",
        f"  Bull case:",
        f"    {verdict.bull_case}",
        "",
        f"  Value-trap case:",
        f"    {verdict.value_trap_case}",
        "",
        f"  Conclusion: {verdict.verdict}, stated confidence {verdict.confidence}/100",
        "",
        f"  What the author says would change their mind:",
        f"    {verdict.what_would_change_my_mind}",
        "",
        "=" * 70,
        "",
    ]

    lines.extend(_automated_findings(analyzed))

    lines.extend(
        [
            "",
            "YOUR TASK",
            "",
            "Audit the note against the source data. Work through the checks in "
            "order, then decide.",
            "",
            "Pay closest attention to comparisons. A note can quote two real "
            "figures and state a false relationship between them; the source data "
            "already contains the correct comparison, so check the note's against "
            "it rather than working it out yourself.",
            "",
            "If the note is sound, say so — KEEP is a valid outcome and inventing "
            "defects to look rigorous is its own failure.",
        ]
    )
    return "\n".join(lines)


def _automated_findings(analyzed: AnalyzedCandidate) -> list[str]:
    """Hand over what Python already established, so the model need not redo it.

    The mechanical grounding check is more reliable than a model at "is this
    number present in that list", and telling the reflector its result frees the
    audit to spend its attention on the part machines cannot do. Absence of
    findings is stated explicitly too — an empty section reads as "not checked",
    which would invite the model to be reassured by nothing.
    """
    lines = ["AUTOMATED CHECKS ALREADY RUN (trust these; do not repeat them)", ""]

    fabricated = analyzed.fabricated_figures
    if fabricated:
        lines.append(
            f"  {len(fabricated)} figure(s) in the note do NOT appear "
            f"anywhere in the source data:"
        )
        for issue in fabricated:
            lines.append(f'    - {issue.value:g}, in: "{issue.context}"')
        lines.append(
            "  A figure with no source is a fabrication. Weigh it accordingly."
        )
    else:
        lines.append(
            "  Every figure in the note traces back to the source data. Note what "
            "this does\n  not prove: a real figure can still support a false claim. "
            "That is your job."
        )

    if analyzed.hypothetical_figures:
        # Reported, but explicitly not as fabrications. Telling the reflector that
        # "Brent below $70/bbl" is an invented figure would have it demote a note
        # for answering the question it was asked.
        cited = ", ".join(f"{i.value:g}" for i in analyzed.hypothetical_figures)
        lines.append("")
        lines.append(
            f"  The note proposes {len(analyzed.hypothetical_figures)} threshold(s) "
            f"({cited}) when naming\n  what would change its mind. These are "
            f"hypothetical future levels, not claims about\n  the present, so they "
            f"are not expected in the source data. Judge whether they are\n  "
            f"sensible tests, not whether they are traceable."
        )

    if analyzed.headline_sourced:
        cited = ", ".join(f"{v:g}" for v in analyzed.headline_sourced)
        lines.append("")
        lines.append(
            f"  {len(analyzed.headline_sourced)} figure(s) in the note ({cited}) come "
            f"from news\n  headline text rather than verified financial data. They are "
            f"not errors, but a\n  claim resting on one is weaker than it looks."
        )
    return lines


def reflect(analyzed: AnalyzedCandidate, **kwargs) -> ReflectedCandidate:
    """Audit one verdict and apply the asymmetry.

    Raises:
        LLMError: If the call fails or the response cannot be validated.
    """
    prompt = build_prompt(analyzed)
    reflection, result = chat_structured(
        prompt,
        Reflection,
        system=SYSTEM,
        max_tokens=MAX_COMPLETION_TOKENS,
        **kwargs,
    )
    final_verdict, final_confidence, overrides = apply_asymmetry(analyzed, reflection)
    return ReflectedCandidate(
        analyzed=analyzed,
        reflection=reflection,
        final_verdict=final_verdict,
        final_confidence=final_confidence,
        overrides=overrides,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


def reflect_all(
    analyzed: tuple[AnalyzedCandidate, ...],
    *,
    on_error=None,
    **kwargs,
) -> tuple[ReflectedCandidate, ...]:
    """Audit each verdict, keeping going if one fails.

    Same reasoning as `analyze_all`: by this point every candidate represents
    minutes of screening, enrichment and a completed LLM call, and one stumble
    on a local model should not cost the rest.
    """
    out: list[ReflectedCandidate] = []
    for item in analyzed:
        try:
            out.append(reflect(item, **kwargs))
        except LLMError as exc:
            if on_error is not None:
                on_error(item, exc)
            else:
                raise
    return tuple(out)


def apply_asymmetry(
    analyzed: AnalyzedCandidate,
    reflection: Reflection,
) -> tuple[str, int, tuple[str, ...]]:
    """Enforce design rule 4 in Python: reflection lowers conviction, never raises it.

    Returns `(final_verdict, final_confidence, overrides)`, where `overrides`
    names every place this function overruled the reflector. Those are worth
    surfacing: a reflector that keeps needing to be overruled is telling you
    something about the model, and silently clamping would hide it.

    The rules, and why each one is here:

    - **Confidence can never exceed the note's.** The one rule the whole
      asymmetry rests on.
    - **An ungrounded figure blocks KEEP and caps confidence absolutely.** If the
      mechanical check found a fabricated number, the note cites something that
      does not exist — not a judgment call, so not left to the reflector. The cap
      is absolute rather than relative because every other rule here is anchored
      to the note's own stated confidence, and an inflated one drags the floor up
      with it; see `UNGROUNDED_CONFIDENCE_CEILING`.
    - **DEMOTE must actually demote.** Models say "materially weakened" and then
      return the original number.
    - **REJECT is capped hard**, so a rejected note cannot read as a
      moderately-held view further down the pipeline.

    **`VALUE_TRAP` is never rewritten.** It is the pessimistic verdict, and every
    rule here moves conviction *away* from the opportunity. Flipping a value-trap
    call to UNCLEAR because its reasoning was sloppy would raise conviction,
    which is exactly what rule 4 forbids — a badly-argued warning is still a
    warning. Only the optimistic verdict is withdrawn on REJECT.
    """
    verdict = analyzed.verdict
    outcome = reflection.outcome
    overrides: list[str] = []

    # Only figures asserting something about the present count here. A number in
    # `what_would_change_my_mind` is a proposed threshold, not a claim — see
    # `AnalyzedCandidate.fabricated_figures`.
    fabricated = analyzed.fabricated_figures

    if fabricated and outcome == "KEEP":
        outcome = "DEMOTE"
        overrides.append(
            f"reflector returned KEEP despite {len(fabricated)} "
            f"ungrounded figure(s); forced to DEMOTE"
        )

    confidence = reflection.adjusted_confidence
    if confidence > verdict.confidence:
        overrides.append(
            f"reflector raised confidence {verdict.confidence} -> {confidence}; "
            f"capped at the original"
        )
        confidence = verdict.confidence

    if outcome == "DEMOTE":
        ceiling = verdict.confidence - DEMOTE_MIN_DROP
        if confidence > ceiling:
            overrides.append(
                f"DEMOTE left confidence at {confidence}; lowered to {max(ceiling, 0)}"
            )
            confidence = ceiling
    elif outcome == "REJECT" and confidence > REJECT_CONFIDENCE_CEILING:
        overrides.append(
            f"REJECT left confidence at {confidence}; capped at "
            f"{REJECT_CONFIDENCE_CEILING}"
        )
        confidence = REJECT_CONFIDENCE_CEILING

    # Applied last, so it overrides the relative floors rather than competing
    # with them. A fabricated figure is the one defect measured without the
    # model's help, and it is not negotiable.
    if fabricated and confidence > UNGROUNDED_CONFIDENCE_CEILING:
        overrides.append(
            f"{len(fabricated)} ungrounded figure(s); confidence "
            f"capped at {UNGROUNDED_CONFIDENCE_CEILING} (was {confidence})"
        )
        confidence = UNGROUNDED_CONFIDENCE_CEILING

    confidence = max(0, min(100, confidence))

    final_verdict = verdict.verdict
    if outcome == "REJECT" and final_verdict == "TEMPORARY_DISLOCATION":
        final_verdict = "UNCLEAR"
        overrides.append("REJECT withdrew the TEMPORARY_DISLOCATION call")

    return final_verdict, confidence, tuple(overrides)


def check_reflection_grounding(reflected: ReflectedCandidate):
    """Run the numeric grounding audit over the reflection's own prose.

    An auditor that invents figures is not an auditor, and the machinery already
    exists.

    **The allowed pool is every number in the audit prompt**, not just the
    verified figures. That is the honest definition here, and the first live run
    proved it: the reflector was flagged three times for correctly quoting the
    note it was reviewing — the fabricated 34.7 it was *reporting* as fabricated,
    and the 88 and 100 from "stated confidence 88/100". All three were shown to
    it. Taking the numbers straight from the rendered prompt covers the figure
    sheet, the note, and the automated findings in one rule, and leaves a genuine
    invention with nowhere to hide.

    Reported rather than acted on. A finding here is a reason to distrust the
    audit, which is not the same as a reason to demote the candidate a second
    time for the analyst's benefit.
    """
    return check_grounding(
        reflected.reflection,
        reflected.analyzed.enriched,
        fields=REFLECTION_FIELDS,
        also_allowed=tuple(numbers_in(build_prompt(reflected.analyzed))),
    )
