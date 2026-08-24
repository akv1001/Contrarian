"""Stage 3 — the judgment call. LLM call 1 of exactly two.

Asks one question per candidate: is this price depressed by something temporary,
or is it cheap for a real structural reason?

Three defences against the failure mode this project exists to prevent, in
descending order of how much they actually help:

1. **The model is given no arithmetic to do.** `enrich.py` resolves every
   comparison before the prompt is built. This is the defence that works; the
   other two catch what it misses.
2. **The schema forces the bear case out first.** Constrained decoding emits
   fields in schema order, so `value_trap_case` is physically written before
   `verdict` exists. Design rule 5 asks for both sides to be argued — ordering
   the schema means the model cannot reach a conclusion and then rationalise it.
3. **Every number in the output is traced back to the input.** Mechanical, and
   honestly limited — see `check_grounding`.

Deliberately one call per candidate rather than one call for all of them.
Batching would be cheaper, but it lets a strong story about one company colour
the verdict on the next, and the candidates are not related to each other.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from .enrich import figure_sheet
from .llm import LLMError, chat_structured
from .schema import AnalyzedCandidate, EnrichedCandidate, GroundingIssue, Verdict

SYSTEM = """You are a sceptical value-investing analyst reviewing Indian equities.

Rules you must follow exactly:

1. Use ONLY the figures given to you. Never introduce a number from memory, and
   never state a number you were not given.
2. Do not calculate. Every comparison you need has already been worked out and
   stated in the data. If a figure you want is not present, say it is not
   available rather than deriving it.
3. Argue the bear case honestly and at full strength before reaching a verdict.
   A one-line dismissal of the value-trap argument is a failure of the task.
4. If the evidence does not support a conclusion, answer UNCLEAR. That is a
   valid and often correct answer. Reaching for a verdict you cannot support is
   worse than declining to reach one.

Calibrate confidence on this scale, and use its full range:
  80-100  the evidence points clearly one way
  60-79   the evidence leans one way, with real counter-evidence
  40-59   genuinely balanced
  0-39    you are close to guessing
Reserve UNCLEAR for cases that are actually balanced once you have weighed both
sides. If one side is stronger, say so and set confidence accordingly.

You are not giving investment advice. You are assessing evidence."""

# Budget for completion tokens. Qwen3 is a hybrid reasoning model and CLAUDE.md
# warns to allow roughly 3x the visible answer length; measured responses run
# 120-200 tokens, so this has a wide margin. Running out mid-JSON produces an
# unparseable response, which is a much worse failure than a slightly long one.
MAX_COMPLETION_TOKENS = 1200

# Numbers below this are ignored by the grounding check. Small integers are
# almost always ordinal or structural ("3 straight quarters", "the second
# factor") rather than financial claims, and flagging them buries the real
# findings in noise.
TRIVIAL_INTEGER_CEILING = 12

# Fallback tolerance for very large numbers, where a model may quote a rounded
# form ("about 3.4 lakh crore") that no decimal-place rule would match.
RELATIVE_TOLERANCE = 0.02

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# Time-window idioms, stripped before numbers are extracted. "52-week low" is a
# phrase, not a claim about a quantity, and flagging its 52 as an ungrounded
# figure buries real findings under noise. This was a live false positive on the
# first full run.
_IDIOMS = re.compile(
    r"\b\d+[-\s](?:week|month|year|day|quarter)s?\b|\bQ[1-4]\b|\bH[12]\b|\bFY\s?\d{2,4}\b",
    re.IGNORECASE,
)


def build_prompt(enriched: EnrichedCandidate) -> str:
    """Assemble the user message. Everything the model sees originates here.

    Keeping this a pure function of the candidate is what makes the grounding
    check meaningful: if a figure is not in this string, the model had no
    legitimate route to it.
    """
    lines = [
        f"Company: {enriched.name} ({enriched.symbol}), listed on the NSE.",
        f"Sector: {enriched.sector}",
        "",
        "The figures below are the only numbers you may use. Every comparison in "
        "them is already calculated. Do not recompute them.",
        "",
        figure_sheet(enriched),
    ]

    lines.extend(
        [
            "",
            "YOUR TASK",
            "",
            f"This stock passed a quantitative screen for being cheap and beaten "
            f"down. Decide whether the depressed price reflects something "
            f"temporary or a genuine structural problem.",
            "",
            "Work in this order:",
            "  1. Identify the specific factor that appears to have depressed the "
            "price. If the data does not show one, say so.",
            "  2. State the case that the business is sound.",
            "  3. State the strongest possible case that this is a value trap — "
            "cheap for a real reason. Argue it as an opponent would.",
            "  4. Only then give your verdict and a calibrated confidence.",
            "  5. Name the single data point that would most change your view.",
            "",
            "Cite figures exactly as they were given to you.",
        ]
    )
    return "\n".join(lines)


def analyze(enriched: EnrichedCandidate, **kwargs) -> AnalyzedCandidate:
    """Judge one candidate and audit the result for ungrounded figures.

    Raises:
        LLMError: If the call fails or the response cannot be validated. Not
            caught here — `analyze_all` decides what a single failure means for
            the run.
    """
    prompt = build_prompt(enriched)
    verdict, result = chat_structured(
        prompt,
        Verdict,
        system=SYSTEM,
        max_tokens=MAX_COMPLETION_TOKENS,
        **kwargs,
    )
    issues, headline_sourced = check_grounding(verdict, enriched)
    return AnalyzedCandidate(
        enriched=enriched,
        verdict=verdict,
        grounding_issues=issues,
        headline_sourced=headline_sourced,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


def analyze_all(
    candidates: tuple[EnrichedCandidate, ...],
    *,
    on_error=None,
    **kwargs,
) -> tuple[AnalyzedCandidate, ...]:
    """Analyze each candidate, keeping going if one fails.

    A local model that stumbles on the fourth of nine candidates should not cost
    the other eight, which took minutes to produce.
    """
    out: list[AnalyzedCandidate] = []
    for enriched in candidates:
        try:
            out.append(analyze(enriched, **kwargs))
        except LLMError as exc:
            if on_error is not None:
                on_error(enriched, exc)
            else:
                raise
    return tuple(out)


# Prose fields of a Verdict that carry claims, and so get audited. `verdict` and
# `confidence` are excluded: one is an enum, the other is a self-report the model
# is asked to produce rather than a figure it could have got wrong.
VERDICT_FIELDS = (
    "temporary_factor",
    "bull_case",
    "value_trap_case",
    "what_would_change_my_mind",
)


def check_grounding(
    verdict: BaseModel,
    enriched: EnrichedCandidate,
    *,
    fields: tuple[str, ...] = VERDICT_FIELDS,
    also_allowed: tuple[float, ...] = (),
) -> tuple[tuple[GroundingIssue, ...], tuple[float, ...]]:
    """Find numbers in the model's prose that do not trace back to the input.

    `fields` names which prose fields to audit, so stage 4 can run the identical
    check over a `Reflection` — an auditor that invents figures is no better than
    the analysis it is auditing.

    `also_allowed` widens the verified pool. Stage 4 needs it: the reflector is
    shown the note under review, so quoting the note's figures back — including a
    fabricated one it is reporting *as* fabricated — is the audit working, not a
    hallucination. Without it the check flags the auditor for doing its job.

    Returns `(issues, headline_sourced)`. A figure quoted from a headline is not
    an issue — the model was shown it — but it is tracked separately, because a
    number from third-party news text has not been through the data layer.
    Coal India's verdict cited "production rises 8.44%" straight from a real
    headline: legitimate, and weaker evidence than it looks.

    **What this catches:** a figure the model invented or recalled from training.
    That is the silent failure mode financial-QA benchmarks warn about, and it is
    worth catching mechanically.

    **What this does not catch, and it matters:** a real input number used to
    support a false claim. The Phase 4 spike produced "P/E 17.0 is 69.2% below
    sector median" when the median *was* 69.2 — every number traceable, the
    relationship fabricated. This check passes that cleanly.

    The primary defence against it is `enrich.py` removing the need to calculate.
    The secondary one is Phase 5, which re-reads the verdicts against the same
    data with fresh eyes. Do not mistake a clean grounding report for a correct
    analysis.
    """
    verified = enriched.allowed_numbers() + tuple(also_allowed)
    from_news = enriched.headline_numbers()
    issues: list[GroundingIssue] = []
    headline_sourced: list[float] = []

    for field_name in fields:
        text = getattr(verdict, field_name, "") or ""
        for value, decimals, context in _numbers_with_context(text):
            if _is_grounded(value, decimals, verified):
                continue
            if _is_grounded(value, decimals, from_news):
                headline_sourced.append(value)
                continue
            issues.append(GroundingIssue(field=field_name, value=value, context=context))
    return tuple(issues), tuple(headline_sourced)


def _numbers_with_context(text: str) -> list[tuple[float, int, str]]:
    """Extract numbers with their written precision and surrounding sentence.

    The decimal count is kept because it is what makes rounding checkable: "0.03"
    tells you the model rounded to two places, and an input of 0.033 is then a
    match rather than a discrepancy.
    """
    found: list[tuple[float, int, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        searchable = _IDIOMS.sub(" ", sentence)
        for match in _NUMBER.finditer(searchable):
            raw = match.group().rstrip(".").replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                continue
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            found.append((value, decimals, sentence.strip()))
    return found


def _is_grounded(value: float, decimals: int, allowed: tuple[float, ...]) -> bool:
    """Is this number traceable to something the model was shown?

    Grounded if some input value, rounded to the precision the model actually
    wrote, equals what it wrote. This models the one legitimate transformation —
    rounding — without the slack of a blanket tolerance. A relative check backs
    it up for very large figures quoted approximately.

    Signs are compared both ways: the prompt states magnitudes in words
    ("negative, Rs 4,200 crore") while the model may write the signed form.
    """
    if abs(value) <= TRIVIAL_INTEGER_CEILING and float(value).is_integer():
        return True
    for candidate in allowed:
        for signed in (candidate, -candidate):
            if round(signed, decimals) == round(value, decimals):
                return True
            if _close(value, signed):
                return True
    return False


def _close(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b))
    if scale == 0:
        return True
    return abs(a - b) / scale <= RELATIVE_TOLERANCE
