"""The records that move between stages.

`Candidate` is the important one. Design rule 1 says the model never does
arithmetic and never recalls a figure from training — every number reaching it
comes from the data layer. `Candidate.figures()` is the mechanism: it returns the
complete set of numbers the screen used, and Phase 4's prompt is built from that
dict and nothing else. Any figure appearing in a verdict that is not a value in
this dict is, by definition, a hallucination.

That is also why every filter records the values it compared rather than just a
pass/fail bit. A screen you cannot audit is a screen you cannot debug, and the
expensive error here is a plausible-looking wrong number surviving a skim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .data.source import Fundamentals, NewsItem, PriceStats


@dataclass(frozen=True)
class FilterResult:
    """One filter's outcome, with the numbers that produced it.

    `detail` is written to be readable on its own — "P/E 15.9 <= Financials
    median 26.1" — because it ends up in the report and in the Phase 4 prompt,
    where "valuation: passed" would be useless.
    """

    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass(frozen=True)
class PeerBasis:
    """What "cheap relative to peers" was measured against.

    Carries `is_fallback` because it changes how much the comparison is worth.
    Several Nifty 50 sectors have one or two members, where a sector median is
    either the stock itself or meaningless. In those cases the screen compares
    against the whole index and says so, rather than quietly producing a
    tautology.
    """

    label: str
    metric: str
    median: float | None
    peer_count: int
    is_fallback: bool

    @property
    def description(self) -> str:
        if self.median is None:
            return f"no {self.metric} median available"
        basis = f"{self.label} median {self.metric} {self.median:.1f} (n={self.peer_count})"
        return f"{basis}, index-wide fallback" if self.is_fallback else basis


@dataclass(frozen=True)
class Candidate:
    """One company as the screen saw it: every input, every filter, and a rank.

    Kept whether it passed or failed. The rejections are as interesting as the
    survivors when you are deciding whether a threshold is doing real work — and
    they are what `--show-rejects` prints.
    """

    symbol: str
    name: str
    sector: str
    fundamentals: Fundamentals
    prices: PriceStats
    peer: PeerBasis
    filters: tuple[FilterResult, ...]

    # Score components, each 0..1 and each kept separately so a rank can be
    # explained rather than just asserted.
    depression: float = 0.0
    valuation_discount: float = 0.0
    quality: float = 0.0
    score: float = 0.0

    @property
    def passed(self) -> bool:
        return all(f.passed for f in self.filters)

    @property
    def failures(self) -> tuple[FilterResult, ...]:
        return tuple(f for f in self.filters if not f.passed)

    @property
    def failure_summary(self) -> str:
        return ", ".join(f.name for f in self.failures) or "-"

    def figures(self) -> dict[str, float | None]:
        """Every number the screen used, keyed by name.

        This is the grounding contract with Phase 4. Add a figure to a prompt
        only if it is in here; check a figure in a verdict by looking it up here.
        """
        f, p = self.fundamentals, self.prices
        return {
            "price": f.price,
            "market_cap": f.market_cap,
            "fifty_two_week_high": f.fifty_two_week_high,
            "fifty_two_week_low": f.fifty_two_week_low,
            "position_in_52w_range": f.position_in_52w_range,
            "drawdown_from_high": f.drawdown_from_high,
            "average_volume": f.average_volume,
            "return_1m": p.return_1m,
            "return_3m": p.return_3m,
            "return_6m": p.return_6m,
            "return_12m": p.return_12m,
            "trailing_pe": f.trailing_pe,
            "forward_pe": f.forward_pe,
            "price_to_book": f.price_to_book,
            "ev_to_ebitda": f.ev_to_ebitda,
            "dividend_yield": f.dividend_yield,
            "free_cash_flow": f.free_cash_flow,
            "operating_cash_flow": f.operating_cash_flow,
            "debt_to_equity_pct": f.debt_to_equity,
            "total_debt": f.total_debt,
            "return_on_equity": f.return_on_equity,
            "profit_margin": f.profit_margin,
            "revenue_growth": f.revenue_growth,
            "earnings_growth": f.earnings_growth,
            "trailing_eps": f.trailing_eps,
            "book_value": f.book_value,
            "total_revenue": f.total_revenue,
            "peer_median": self.peer.median,
        }


_NUMBER_IN_TEXT = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[float]:
    """Pull every numeric literal out of a string, tolerating thousands commas."""
    out: list[float] = []
    for match in _NUMBER_IN_TEXT.finditer(text or ""):
        try:
            out.append(float(match.group().rstrip(".").replace(",", "")))
        except ValueError:
            continue
    return out


@dataclass(frozen=True)
class Fact:
    """One pre-computed statement of fact, ready to hand to the model.

    Every comparison, ratio, and percentage change is resolved *here*, in Python,
    before the model ever sees it. Design rule 1 says the model never does
    arithmetic, and this is how that is enforced rather than merely requested.

    The rule earned itself during the Phase 4 spike. Given a raw P/E of 17.0 and
    a sector median of 69.2, qwen3:14b wrote "P/E 17.0 is 69.2% below sector
    median" — reusing a real input number inside invented arithmetic. Handed the
    same comparison pre-computed as "a 75% discount", it reported it correctly.

    `numbers` lists every numeric value in `text`, which is what makes the
    grounding check in `analyze.py` mechanical.
    """

    label: str
    text: str
    numbers: tuple[float, ...] = ()

    def __str__(self) -> str:
        return f"{self.label}: {self.text}"


@dataclass(frozen=True)
class EnrichedCandidate:
    """A survivor plus everything stage 3 is allowed to reason from.

    Nothing else reaches the prompt. If a figure is not in `facts` and a headline
    is not in `headlines`, the model has no legitimate way to know it — which is
    precisely what makes an ungrounded number in the output detectable.
    """

    candidate: Candidate
    facts: tuple[Fact, ...]
    headlines: tuple[NewsItem, ...]
    news_source: str = "unknown"

    @property
    def symbol(self) -> str:
        return self.candidate.symbol

    @property
    def name(self) -> str:
        return self.candidate.name

    @property
    def sector(self) -> str:
        return self.candidate.sector

    def allowed_numbers(self) -> tuple[float, ...]:
        """Every number from the *verified* data layer. The strong whitelist."""
        seen: list[float] = []
        for fact in self.facts:
            seen.extend(fact.numbers)
        for value in self.candidate.figures().values():
            if value is not None:
                seen.append(float(value))
        return tuple(seen)

    def headline_numbers(self) -> tuple[float, ...]:
        """Numbers appearing in the headlines, and their age markers.

        Kept separate from `allowed_numbers` on purpose. These were genuinely
        shown to the model, so citing one is not a hallucination — Coal India's
        verdict quoted "production rises 8.44%" straight from a real headline.
        But a figure from third-party news text has not been through the data
        layer and carries none of its guarantees, so a verdict leaning on one is
        making a weaker claim than it appears to be. Phase 5 should weigh that.
        """
        seen: list[float] = []
        for item in self.headlines:
            seen.extend(numbers_in(item.title))
            if item.age_days is not None:
                # The prompt labels each headline "[17d ago, ...]", and models
                # quote that marker back when citing the story.
                seen.append(float(round(item.age_days)))
        return tuple(seen)


class Verdict(BaseModel):
    """Stage 3's judgment. Schema-constrained — the provider enforces the shape.

    **Field order is load-bearing.** Constrained decoding emits properties in
    schema order, so the model physically writes `value_trap_case` before it
    writes `verdict`. Design rule 5 requires the bear case to be argued before
    any recommendation, and ordering the schema is what makes that structural
    rather than a polite request the model can skip.

    `confidence` is an integer 0-100, and it needs the explicit rubric that
    `analyze.SYSTEM` carries. Without one the scale collapsed: the first full run
    returned 4, 0 and 0 across three candidates, which is not a calibrated
    judgment but a model declining to use a range it was never shown. With the
    rubric the same input scores 45-55 — "genuinely balanced", which matches the
    reasoning it wrote.

    Even calibrated, treat it as soft. The Phase 4 spike scored an identical
    value-trap case 60 and then 15 on consecutive runs. Phase 5 audits confidence
    rather than trusting it, and may only lower it.
    """

    temporary_factor: str = Field(
        description="The specific recent event or condition that appears to have "
        "depressed the price. Say 'none identified' if the data does not show one."
    )
    bull_case: str = Field(
        description="Why the business may be sound despite the price, citing only "
        "figures from the data provided."
    )
    value_trap_case: str = Field(
        description="The strongest possible argument that this is cheap for a real "
        "structural reason. Argue it properly, as an opponent would."
    )
    verdict: Literal["TEMPORARY_DISLOCATION", "VALUE_TRAP", "UNCLEAR"]
    confidence: int = Field(ge=0, le=100, description="0-100, calibrated to evidence.")
    what_would_change_my_mind: str = Field(
        description="The single data point that would most change this assessment."
    )


@dataclass(frozen=True)
class GroundingIssue:
    """A number in the model's output that does not trace to its input.

    Records the sentence it appeared in, because the number alone is rarely
    enough to judge whether it is a hallucination or a false positive.
    """

    field: str
    value: float
    context: str

    def __str__(self) -> str:
        return f"{self.field}: {self.value:g} not in input — \"{self.context}\""


# Verdict fields that are counterfactual *by construction*. `what_would_change_my_mind`
# asks the model to name a future data point that would move it — "Brent below
# $70/bbl", "a 20%+ rise over six months". Proposing a threshold requires choosing a
# number, and that number is not meant to describe the present, so it cannot be traced
# to the input and should not be. See `AnalyzedCandidate.fabricated_figures`.
HYPOTHETICAL_FIELDS = frozenset({"what_would_change_my_mind"})


@dataclass(frozen=True)
class AnalyzedCandidate:
    """A verdict bound to the candidate it judges, plus the grounding audit."""

    enriched: EnrichedCandidate
    verdict: Verdict
    grounding_issues: tuple[GroundingIssue, ...] = ()
    # Figures the verdict took from news headlines rather than the data layer.
    # Not errors — but claims resting on unverified third-party text.
    headline_sourced: tuple[float, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def symbol(self) -> str:
        return self.enriched.symbol

    @property
    def is_grounded(self) -> bool:
        return not self.grounding_issues

    @property
    def fabricated_figures(self) -> tuple[GroundingIssue, ...]:
        """Ungrounded figures that assert something about the *present*.

        The distinction earned itself on the first full run. ONGC's verdict was
        flagged for `70` and `20` — "a sustained drop in Brent below $70/bbl" and
        "a 20%+ rise in the share price" — both in `what_would_change_my_mind`,
        both proposed tests rather than claims. Treated as fabrications they
        tripped `UNGROUNDED_CONFIDENCE_CEILING` and handed ONGC the harshest
        treatment in the run, for the offence of answering the question it was
        asked.

        Only these carry a mechanical penalty. Everything is still *reported* —
        see `hypothetical_figures` — because a genuinely invented statistic could
        hide in that field too, and a human reading the report should see it.
        """
        return tuple(
            issue for issue in self.grounding_issues
            if issue.field not in HYPOTHETICAL_FIELDS
        )

    @property
    def hypothetical_figures(self) -> tuple[GroundingIssue, ...]:
        """Untraceable numbers in a field that asks for a hypothetical."""
        return tuple(
            issue for issue in self.grounding_issues
            if issue.field in HYPOTHETICAL_FIELDS
        )


class Reflection(BaseModel):
    """Stage 4's audit of a stage-3 verdict. Schema-constrained, fresh context.

    **Field order is load-bearing here too**, for the same reason `Verdict`'s is:
    constrained decoding emits properties in order, so all five findings are
    physically written before `outcome` exists. An auditor that picks its verdict
    first and then assembles findings to justify it is not auditing.

    The five checks are the plan's four, split in two where it matters:
    `figures_check` asks whether the numbers are real, `claim_check` asks whether
    the claims built on them follow. That split is the whole reason this stage
    exists. `analyze.check_grounding` already answers the first question
    mechanically and better than a model can — it is `claim_check` that closes
    the documented blind spot, where every figure is real and the relationship
    between them is invented.

    `adjusted_confidence` is a *request*, not the final word. Design rule 4 says
    reflection may only lower conviction, and `reflect.apply_asymmetry` enforces
    that in Python afterwards — a model asked politely not to raise a number will
    eventually raise one.
    """

    figures_check: str = Field(
        description="Does every figure quoted in the note appear in the source data? "
        "Name any that do not, and say so plainly if all of them do."
    )
    claim_check: str = Field(
        description="Do the note's claims actually follow from the figures? A real "
        "number can be used to support an invented relationship — check for that "
        "specifically, including any comparison or percentage the note asserts."
    )
    bear_case_engagement: str = Field(
        description="Was the value-trap case genuinely argued, or raised and dismissed? "
        "Quote or paraphrase the weakest part."
    )
    catalyst_concreteness: str = Field(
        description="Is there a specific, resolvable reason the dip should reverse, or "
        "is 'temporary' asserted without a mechanism?"
    )
    confidence_calibration: str = Field(
        description="Does the stated confidence match the strength of the evidence "
        "presented? Say whether it is too high, about right, or too low."
    )
    weaknesses: list[str] = Field(
        description="The specific defects found, one per item, each naming what is "
        "wrong rather than restating the check. Empty if the note is sound.",
        max_length=6,
    )
    outcome: Literal["KEEP", "DEMOTE", "REJECT"] = Field(
        description="KEEP: sound analysis. DEMOTE: usable but materially weakened. "
        "REJECT: the reasoning does not support presenting this as an opportunity."
    )
    adjusted_confidence: int = Field(
        ge=0, le=100, description="Confidence the evidence actually supports. May be "
        "lower than the note's, never higher."
    )


@dataclass(frozen=True)
class ReflectedCandidate:
    """A verdict after audit: what stage 3 said, what stage 4 found, what stands.

    `final_verdict` and `final_confidence` are the post-asymmetry values and the
    only ones a report should quote. `overrides` records every place Python
    overruled the reflector, so a suspiciously generous audit is visible rather
    than silently applied.
    """

    analyzed: AnalyzedCandidate
    reflection: Reflection
    final_verdict: str
    final_confidence: int
    overrides: tuple[str, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def symbol(self) -> str:
        return self.analyzed.symbol

    @property
    def name(self) -> str:
        return self.analyzed.enriched.name

    @property
    def original_verdict(self) -> str:
        return self.analyzed.verdict.verdict

    @property
    def original_confidence(self) -> int:
        return self.analyzed.verdict.confidence

    @property
    def confidence_drop(self) -> int:
        return self.original_confidence - self.final_confidence

    @property
    def survived(self) -> bool:
        """Did the analysis come through the audit intact?"""
        return self.reflection.outcome == "KEEP" and not self.overrides


@dataclass(frozen=True)
class ScreenResult:
    """The output of stage 1: survivors ranked, rejections kept, errors surfaced.

    `errors` is deliberately not swallowed. A ticker that failed to fetch is not
    a ticker that failed the screen, and collapsing the two would let a broken
    data layer look like a strict filter.
    """

    as_of: datetime
    universe_size: int
    passed: tuple[Candidate, ...]
    rejected: tuple[Candidate, ...]
    errors: tuple[tuple[str, str], ...] = ()

    @property
    def evaluated(self) -> int:
        return len(self.passed) + len(self.rejected)


@dataclass(frozen=True)
class PipelineRun:
    """One complete pass through all five stages — everything the report prints.

    `report.py` takes this and nothing else. That is what keeps design rule 2
    enforceable: a renderer that receives a finished record has no reason to
    reach for a model, and no data it would need one to obtain.

    `skipped` records candidates that screened through but whose LLM stages
    failed. Kept separate from a screen rejection for the same reason
    `ScreenResult.errors` is: a candidate we could not judge is not a candidate
    we judged and dismissed, and collapsing the two would overstate how much of
    the universe was actually examined.
    """

    as_of: datetime
    screen: ScreenResult
    reflected: tuple[ReflectedCandidate, ...]
    model: str = "unknown"
    provider: str = "unknown"
    skipped: tuple[tuple[str, str], ...] = ()
    duration_seconds: float = 0.0

    @property
    def attempted(self) -> int:
        return len(self.reflected) + len(self.skipped)

    @property
    def llm_calls(self) -> int:
        """Two per candidate that completed — the analysis and the audit.

        Hard-coded to the current two-stage shape. If a stage that calls a model is
        ever added, count the calls rather than multiplying.
        """
        return 2 * len(self.reflected)

    @property
    def total_tokens(self) -> int:
        return sum(
            r.prompt_tokens
            + r.completion_tokens
            + r.analyzed.prompt_tokens
            + r.analyzed.completion_tokens
            for r in self.reflected
        )
