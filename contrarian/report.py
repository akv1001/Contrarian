"""Stage 5 — render a finished run for a human. Markdown, deterministic.

**This module makes no LLM call, and must never make one.** Design rule 2. By
the time a `PipelineRun` reaches here every figure in it has been through the
data layer, pre-computed by `enrich.py`, cited under a schema, and audited twice.
Re-summarising that through a model at the last mile would reintroduce exactly
the drift the previous four stages exist to prevent — and it would do it in the
one artefact a human actually reads, where nothing downstream would catch it.
There is a test asserting this file imports nothing from `llm.py`.

The report's job is to make every claim auditable. That means the source figures
appear *next to* the reasoning that leans on them, not in an appendix: the
failure mode this project is built around is a confident sentence containing a
wrong number, and the only real defence is putting the number where a reader
trips over it.

Ordering is by post-audit confidence within verdict class, and the report says
plainly that this is not a ranking of investment merit. It is idea generation.

**This is read daily, so it explains as little as it can get away with.** The
rule: define a term the report actually uses, once, above the results — a reader
must be able to follow it without opening the README. Everything else about how
the pipeline works belongs in the README, or failing that in the final appendix
where a returning reader can skip it. Boilerplate repeated once per candidate is
the thing to watch for: nine copies of the same sentence is nine chances for a
reader to start skimming, and the figures are what they must not skim.
"""

from __future__ import annotations

from datetime import datetime

from .schema import (
    Candidate,
    PipelineRun,
    ReflectedCandidate,
)

# Verdicts in the order a reader should meet them. A withdrawn or dismissed idea
# is still worth printing — the rejections are where you learn whether the screen
# is doing real work — but it goes below the ones that survived.
VERDICT_ORDER = {"TEMPORARY_DISLOCATION": 0, "UNCLEAR": 1, "VALUE_TRAP": 2}

VERDICT_LABEL = {
    "TEMPORARY_DISLOCATION": "Temporary dislocation",
    "VALUE_TRAP": "Value trap",
    "UNCLEAR": "Unclear",
}

OUTCOME_NOTE = {
    "KEEP": "the audit found no material defect",
    "DEMOTE": "the audit found material weaknesses",
    "REJECT": "the audit rejected the reasoning",
}

# Plain-English gloss for every piece of jargon the report uses. A reader who has
# never seen this project should not have to infer what UNCLEAR means, or guess
# whether 40/100 is good. Each entry is (headline, what it actually means), and
# it is printed once — in the glossary above the results, never per candidate.
VERDICT_MEANING = {
    "TEMPORARY_DISLOCATION": (
        "The fall looks temporary",
        "something specific and passing knocked the price down, and the business "
        "underneath still looks sound",
    ),
    "VALUE_TRAP": (
        "Cheap for a real reason",
        "the low price reflects a genuine, lasting problem — cheap, and deservedly so",
    ),
    "UNCLEAR": (
        "Not enough evidence either way",
        "the two cases came out roughly even; a legitimate answer, but not a reason "
        "to act",
    ),
}

OUTCOME_MEANING = {
    "KEEP": "the reasoning held up",
    "DEMOTE": "real weaknesses, so the confidence was cut",
    "REJECT": "the reasoning did not hold up at all",
}

# Confidence bands, worded to match the scale the analyst was actually given.
_BANDS = (
    (80, "high", "the evidence points clearly one way"),
    (60, "moderate", "it leans one way, but there is real evidence against"),
    (40, "low", "the evidence is genuinely balanced"),
    (0, "very low", "this is close to a guess"),
)


def confidence_band(confidence: int) -> tuple[str, str]:
    """Turn a 0-100 score into a word and an explanation.

    A bare "40/100" tells a new reader nothing — is that good? The bands match the
    rubric the model was scored against, so the label and the number agree.
    """
    for floor, word, meaning in _BANDS:
        if confidence >= floor:
            return word, meaning
    return "very low", "this is close to a guess"

PURPOSE = (
    "Nifty 50 companies whose price looks depressed by a *temporary* factor rather "
    "than a broken business — contrarian value candidates, surfaced for research."
)

DISCLAIMER = (
    "**This is idea generation, not investment advice.** Every verdict below was "
    "written by a language model, and LLM equity analysis is unreliable — the "
    "figures are grounded and machine-checked, the *reasoning* is not. Treat this "
    "as a starting point for your own research and never as a decision."
)


def render(run: PipelineRun) -> str:
    """The whole report as one Markdown string.

    Returns a string rather than writing a file so the caller decides where it
    goes, and so this is testable without touching the filesystem.
    """
    parts = [
        _header(run),
        _how_to_read(),
        _summary_table(run),
    ]
    parts.extend(_candidate_section(r, i) for i, r in enumerate(rank(run.reflected), 1))
    parts.append(_screen_appendix(run))
    parts.append(_method_appendix())
    return "\n\n".join(part for part in parts if part)


def rank(reflected: tuple[ReflectedCandidate, ...]) -> list[ReflectedCandidate]:
    """Best-surviving first: verdict class, then post-audit confidence.

    Deliberately sorted on `final_confidence` rather than the analyst's stated
    number, so an idea the audit knocked down cannot lead the report on the
    strength of a confidence the audit withdrew.
    """
    return sorted(
        reflected,
        key=lambda r: (VERDICT_ORDER.get(r.final_verdict, 9), -r.final_confidence, r.symbol),
    )


def _header(run: PipelineRun) -> str:
    """Title, what this is, and the run's provenance on one line.

    The run stats used to be six bullets. They are worth keeping — a reader
    should know how much of the universe was actually examined and by what — but
    they are context, not content, so they get a single line.
    """
    when = run.as_of.strftime("%d %B %Y")
    screen = run.screen
    stats = [
        f"{screen.universe_size} tickers screened ({screen.evaluated} fetched cleanly)",
        f"{len(screen.passed)} passed the quantitative gates",
        f"{len(run.reflected)} of {run.attempted} judged and audited",
        f"{run.model} via {run.provider}",
        f"{run.llm_calls} LLM calls, {run.total_tokens:,} tokens",
    ]
    if run.duration_seconds:
        stats.append(f"{run.duration_seconds / 60:.0f} min")

    lines = [
        f"# Contrarian screen — {when}",
        "",
        PURPOSE,
        "",
        DISCLAIMER,
        "",
        " · ".join(stats),
    ]
    # Neither of these is a rejection, and the distinction is the honest part.
    misses = []
    if screen.errors:
        misses.append(f"{len(screen.errors)} ticker(s) could not be fetched at all")
    if run.skipped:
        misses.append(
            f"{len(run.skipped)} screened through but could not be judged"
        )
    if misses:
        lines.append("")
        lines.append(f"*{'; '.join(misses)} — see the appendix. Neither is a rejection.*")
    return "\n".join(lines)


def _how_to_read() -> str:
    """Define every term the report uses, once, above the results.

    Deliberately terse. This is read daily; a returning reader should be able to
    take it in at a glance or skip it entirely. It glosses the vocabulary and
    nothing else — how the verdicts are produced is in the final appendix.
    """
    lines = ["## How to read this", ""]
    for key in ("TEMPORARY_DISLOCATION", "UNCLEAR", "VALUE_TRAP"):
        headline, meaning = VERDICT_MEANING[key]
        lines.append(f"- **{VERDICT_LABEL[key]}** — *{headline}*: {meaning}.")

    outcomes = " · ".join(f"**{key}** ({OUTCOME_MEANING[key]})" for key in
                          ("KEEP", "DEMOTE", "REJECT"))
    lines.extend(
        [
            "",
            "**Confidence** runs 0-100: **80-100** high · **60-79** moderate · "
            "**40-59** low, evidence genuinely balanced · **0-39** close to a guess.",
            "",
            f"`55 → 40` means a first pass scored 55 and an independent review cut it "
            f"to 40. The review can **only lower the score, never raise it**, and "
            f"ends in {outcomes}.",
            "",
            "> **Nothing here is a buy or sell signal**, and the order is not a "
            "ranking of investment merit — it is verdict class, then whatever "
            "confidence survived the review.",
        ]
    )
    return "\n".join(lines)


def _summary_table(run: PipelineRun) -> str:
    if not run.reflected:
        return (
            "## Results\n\n"
            "Nothing completed the pipeline. Check the appendix for what failed."
        )

    rows = [
        "## Results",
        "",
        "A single confidence means the review left it unchanged.",
        "",
        "| # | Ticker | Sector | Verdict | Confidence | Review | Figures checked |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rank(run.reflected), 1):
        confidence = (
            f"{r.original_confidence} → **{r.final_confidence}**"
            if r.confidence_drop
            else f"**{r.final_confidence}**"
        )
        # Only fabrications get the warning. A proposed threshold in "what would
        # change my mind" is not an ungrounded claim about the company.
        grounding = (
            f"⚠ {len(r.analyzed.fabricated_figures)} ungrounded"
            if r.analyzed.fabricated_figures
            else "clean"
        )
        rows.append(
            f"| {i} | **{r.symbol}** | {r.analyzed.enriched.sector} | "
            f"{VERDICT_LABEL.get(r.final_verdict, r.final_verdict)} | {confidence} | "
            f"{r.reflection.outcome} | {grounding} |"
        )
    return "\n".join(rows)


def _candidate_section(r: ReflectedCandidate, index: int) -> str:
    """One company: the verdict, then the evidence for it.

    The verdict's meaning, the confidence rubric and what the audit is are all
    glossed once at the top of the report. Repeating any of it here would be the
    same three paragraphs nine times over.
    """
    verdict = r.analyzed.verdict
    label = VERDICT_LABEL.get(r.final_verdict, r.final_verdict)
    word, _band = confidence_band(r.final_confidence)

    lines = [
        "---",
        "",
        f"## {index}. {r.symbol} — {r.name}",
        "",
        f"**{label}** · **Confidence {r.final_confidence}/100 — {word}**",
    ]

    # Spell out the two-pass story rather than leaving "down from 55" to be
    # decoded. This is the single most confusing line in the old format.
    if r.confidence_drop:
        lines.append(
            f"The first pass scored this {r.original_confidence}. The review then cut "
            f"it to {r.final_confidence} — {OUTCOME_NOTE.get(r.reflection.outcome, '')}."
        )
    elif r.reflection.outcome == "KEEP":
        lines.append(
            f"The first pass scored this {r.original_confidence} and the review left "
            f"it there, finding nothing material to object to."
        )

    if r.final_verdict != r.original_verdict:
        lines.extend(
            [
                "",
                f"> The first pass called this "
                f"{VERDICT_LABEL.get(r.original_verdict, r.original_verdict).lower()}. "
                f"The audit withdrew that call.",
            ]
        )

    lines.extend(_why_it_screened(r.analyzed.enriched.candidate))
    lines.extend(
        [
            "",
            "### The analyst note",
            "",
            f"**What appears to have depressed the price.** {verdict.temporary_factor}",
            "",
            f"**The case that the business is sound.** {verdict.bull_case}",
            "",
            f"**The case that it is cheap for a real reason.** {verdict.value_trap_case}",
            "",
            f"**What would most change this view.** {verdict.what_would_change_my_mind}",
        ]
    )
    lines.extend(_audit_section(r))
    lines.extend(_figures_section(r.analyzed.enriched))
    lines.extend(_headlines_section(r.analyzed.enriched))
    return "\n".join(lines)


def _why_it_screened(candidate: Candidate) -> list[str]:
    """The quantitative case, with the numbers each gate actually compared.

    `FilterResult.detail` is written to read on its own precisely so it can be
    dropped in here — "valuation: passed" would tell a reader nothing about
    whether to believe the screen.
    """
    lines = [
        "",
        "### Why it screened",
        "",
        f"Score **{candidate.score:.2f}** — depression {candidate.depression:.2f}, "
        f"discount to peers {candidate.valuation_discount:.2f}, "
        f"quality {candidate.quality:.2f}.",
        f"Compared against {candidate.peer.description}.",
        "",
    ]
    for result in candidate.filters:
        mark = "PASS" if result.passed else "**FAIL**"
        lines.append(f"- {mark} — {result.name}: {result.detail}")
    return lines


def _audit_section(r: ReflectedCandidate) -> list[str]:
    ref = r.reflection
    lines = ["", f"### The audit — {ref.outcome}", ""]
    for label, text in (
        ("Are the figures real?", ref.figures_check),
        ("Do the claims follow from them?", ref.claim_check),
        ("Was the bear case argued?", ref.bear_case_engagement),
        ("Is the catalyst concrete?", ref.catalyst_concreteness),
        ("Is the confidence calibrated?", ref.confidence_calibration),
    ):
        lines.append(f"- **{label}** {text}")

    if ref.weaknesses:
        lines.extend(["", "**Weaknesses found:**", ""])
        lines.extend(f"- {item}" for item in ref.weaknesses)

    if r.analyzed.fabricated_figures:
        lines.extend(
            [
                "",
                f"**⚠ {len(r.analyzed.fabricated_figures)} figure(s) appear nowhere in "
                f"the source data — a fabrication, whatever else the note gets right:**",
                "",
            ]
        )
        lines.extend(
            f'- `{issue.value:g}` in "{issue.context}"'
            for issue in r.analyzed.fabricated_figures
        )

    if r.analyzed.hypothetical_figures:
        cited = ", ".join(f"`{i.value:g}`" for i in r.analyzed.hypothetical_figures)
        lines.extend(
            [
                "",
                f"*Proposes {len(r.analyzed.hypothetical_figures)} threshold(s) — "
                f"{cited} — in naming what would change its mind. Hypothetical future "
                f"levels, not claims about the present, so no penalty applies.*",
            ]
        )

    if r.analyzed.headline_sourced:
        cited = ", ".join(f"`{v:g}`" for v in r.analyzed.headline_sourced)
        lines.extend(
            [
                "",
                f"*{len(r.analyzed.headline_sourced)} figure(s) — {cited} — came from "
                f"headline text rather than verified financial data. Not errors, but a "
                f"claim resting on one is weaker than it looks.*",
            ]
        )

    if r.overrides:
        lines.extend(["", "**Overruled where the audit was too generous:**", ""])
        lines.extend(f"- {note}" for note in r.overrides)
    return lines


def _figures_section(enriched) -> list[str]:
    """Every figure the model was allowed to use, verbatim.

    This is the audit trail, and it is why the report needs no LLM: each of these
    was resolved in Python by `enrich.py`, comparisons and all, before any model
    saw it. A reader who doubts a sentence above can check it here.
    """
    lines = [
        "",
        "### The figures it was given",
        "",
        "| | |",
        "|---|---|",
    ]
    lines.extend(f"| {fact.label} | {fact.text} |" for fact in enriched.facts)
    return lines


def _headlines_section(enriched) -> list[str]:
    """The news the analyst saw. Headline text only — no links.

    The URLs are dropped deliberately. Google News RSS hands back ~700-character
    base64 redirect URLs; even as reference-style definitions they doubled the
    length of this section and made the raw file unreadable, and this report is
    read in an editor as often as rendered. What the analyst was *shown* is the
    headline text, so that is what needs to be auditable here; a reader who wants
    the article can search the headline.
    """
    lines = ["", "### The headlines it was shown", ""]
    if not enriched.headlines:
        lines.append(
            "None retrieved. The analyst was told so explicitly and instructed not "
            "to speculate about what the news might say."
        )
        return lines

    lines.append(
        f"{len(enriched.headlines)} most recent, via {enriched.news_source} "
        f"— headline text only, full articles were not retrieved."
    )
    lines.append("")

    for item in enriched.headlines:
        age = f"{item.age_days:.0f}d ago" if item.age_days is not None else "undated"
        publisher = item.publisher or "unknown source"
        lines.append(f"- *{age}, {publisher}* — {item.title}")
    return lines


def _screen_appendix(run: PipelineRun) -> str:
    """What did not make it, and why. The rejections are the honest part.

    Kept because a screen you cannot see the misses of is a screen you cannot
    judge: if a threshold is doing nothing, or doing far too much, this table is
    where it shows.
    """
    screen = run.screen
    lines = ["---", "", "## Appendix — what did not make it", ""]

    if screen.rejected:
        counts: dict[str, int] = {}
        for candidate in screen.rejected:
            for failure in candidate.failures:
                counts[failure.name] = counts.get(failure.name, 0) + 1
        tally = ", ".join(
            f"`{name}` {count}" for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        lines.extend(
            [
                f"**{len(screen.rejected)} of {screen.evaluated} were rejected by the "
                f"screen.** Failures by gate: {tally}.",
                "",
                "The ten closest misses, by score:",
                "",
                "| Ticker | Sector | Score | Failed on |",
                "|---|---|---|---|",
            ]
        )
        for candidate in screen.rejected[:10]:
            lines.append(
                f"| {candidate.symbol} | {candidate.sector} | {candidate.score:.2f} | "
                f"{candidate.failure_summary} |"
            )

    if run.skipped:
        lines.extend(
            [
                "",
                "**Screened through, but could not be judged.** These are not "
                "rejections — the pipeline failed on them:",
                "",
            ]
        )
        lines.extend(f"- `{symbol}` — {why}" for symbol, why in run.skipped)

    if screen.errors:
        lines.extend(
            [
                "",
                "**Could not be fetched at all.** A data failure, not a screen "
                "failure — these were never evaluated:",
                "",
            ]
        )
        lines.extend(f"- `{symbol}` — {why}" for symbol, why in screen.errors)
    return "\n".join(lines)


def _method_appendix() -> str:
    """How the verdicts are produced — last, because it never changes.

    This is the same every day, so it sits below the results where a returning
    reader skips it and a first-time reader can still find it. The README covers
    it at length; what is here is only what bears on trusting the output.
    """
    return "\n".join(
        [
            "---",
            "",
            "## Appendix — how this is produced",
            "",
            "Two of the five stages use a language model — the verdict and the audit. "
            "**The screen, the arithmetic, the grounding checks and this report do "
            "not.** Those are ordinary code that behaves the same way every time.",
            "",
            "- **Every number came from the data layer.** The model was handed "
            "finished comparisons — \"a 75% discount to the sector median of 69.2\" "
            "— rather than two numbers and a relationship to work out. It was never "
            "asked to calculate anything.",
            "- **Every figure in a verdict was traced back to its input.** Anything "
            "that did not trace is flagged as ungrounded in that company's section.",
            "- **A clean grounding report does not mean a correct analysis.** A real "
            "number can still support a false claim. That is what the audit is for, "
            "and the audit is a language model too.",
            "- **The audit can only lower conviction, never raise it**, enforced in "
            "code rather than asked of the model: the expensive error is "
            "recommending a value trap, not missing an opportunity.",
            "- **Headlines only.** The temporary factor is inferred from headline "
            "text; full articles are never fetched.",
        ]
    )


def default_filename(as_of: datetime) -> str:
    """`contrarian-2026-08-24.md` — sorts chronologically, one per day."""
    return f"contrarian-{as_of.strftime('%Y-%m-%d')}.md"
