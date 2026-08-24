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
        _summary_table(run),
    ]
    parts.extend(_candidate_section(r, i) for i, r in enumerate(rank(run.reflected), 1))
    parts.append(_screen_appendix(run))
    parts.append(_how_to_read())
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
    when = run.as_of.strftime("%d %B %Y")
    screen = run.screen
    lines = [
        f"# Contrarian screen — {when}",
        "",
        DISCLAIMER,
        "",
        "## This run",
        "",
        f"- **Universe:** {screen.universe_size} Nifty 50 tickers "
        f"({screen.evaluated} fetched cleanly)",
        f"- **Screen:** {len(screen.passed)} passed the quantitative gates, "
        f"{len(screen.rejected)} rejected",
        f"- **Judged and audited:** {len(run.reflected)} of {run.attempted} attempted",
        f"- **Model:** {run.model} via {run.provider}",
        f"- **LLM calls:** {run.llm_calls} — exactly two per candidate, "
        f"{run.total_tokens:,} tokens",
    ]
    if run.duration_seconds:
        lines.append(f"- **Took:** {run.duration_seconds / 60:.0f} min")
    if screen.errors:
        lines.append(
            f"- **Fetch failures:** {len(screen.errors)} ticker(s) could not be "
            f"screened — see the appendix. Not the same as being rejected."
        )
    if run.skipped:
        lines.append(
            f"- **Skipped:** {len(run.skipped)} candidate(s) screened through but "
            f"their analysis failed — see the appendix."
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
        "Confidence is shown as the analyst's figure and the figure that survived "
        "the audit. Reflection can only lower it, never raise it.",
        "",
        "| # | Ticker | Sector | Verdict | Confidence | Audit | Grounding |",
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
    verdict = r.analyzed.verdict
    lines = [
        "---",
        "",
        f"## {index}. {r.symbol} — {r.name}",
        "",
        f"**{VERDICT_LABEL.get(r.final_verdict, r.final_verdict)}**, "
        f"confidence **{r.final_confidence}/100**"
        + (
            f" — down from {r.original_confidence}, "
            f"{OUTCOME_NOTE.get(r.reflection.outcome, '')}"
            if r.confidence_drop
            else ""
        ),
    ]

    if r.final_verdict != r.original_verdict:
        lines.extend(
            [
                "",
                f"> The analyst called this "
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
    lines = [
        "",
        f"### The audit — {ref.outcome}",
        "",
        "A second reader was shown the same source data and this note, "
        "unattributed, and asked whether the reasoning holds.",
        "",
    ]
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
                f"**⚠ {len(r.analyzed.fabricated_figures)} figure(s) in this note "
                f"appear nowhere in the source data.** A figure with no source is a "
                f"fabrication, whatever else the note gets right:",
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
                f"*The note proposes {len(r.analyzed.hypothetical_figures)} threshold(s) "
                f"— {cited} — in naming what would change its mind. Those are "
                f"hypothetical future levels rather than claims about the present, so "
                f"they are not expected to appear in the data and carry no penalty.*",
            ]
        )

    if r.analyzed.headline_sourced:
        cited = ", ".join(f"`{v:g}`" for v in r.analyzed.headline_sourced)
        lines.extend(
            [
                "",
                f"*{len(r.analyzed.headline_sourced)} figure(s) — {cited} — were quoted "
                f"from news headline text rather than verified financial data. Not "
                f"errors, but a claim resting on one is weaker than it looks.*",
            ]
        )

    if r.overrides:
        lines.extend(
            [
                "",
                "**The audit was overruled where it was too generous.** Reflection may "
                "only lower conviction, and that is enforced in code rather than "
                "requested of the model:",
                "",
            ]
        )
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
        "Every number the analyst was allowed to use, exactly as supplied. "
        "Comparisons were computed in Python, not by the model.",
        "",
        "| | |",
        "|---|---|",
    ]
    lines.extend(f"| {fact.label} | {fact.text} |" for fact in enriched.facts)
    return lines


def _headlines_section(enriched) -> list[str]:
    """The news the analyst saw, with links kept out of the way.

    Reference-style links rather than inline ones, because Google News RSS hands
    back ~700-character base64 redirect URLs. Inline, a dozen of those bury the
    headlines they belong to and make the raw file unreadable — and this report
    is as likely to be read in an editor as rendered. The links still resolve;
    they just live at the bottom of the section.
    """
    lines = ["", "### The headlines it was shown", ""]
    if not enriched.headlines:
        lines.append(
            "None retrieved. The analyst was told so explicitly and instructed not "
            "to speculate about what the news might say."
        )
        return lines

    lines.append(
        f"{len(enriched.headlines)} most recent, via {enriched.news_source}. "
        f"Headlines only — full articles were not retrieved."
    )
    lines.append("")

    references: list[str] = []
    for i, item in enumerate(enriched.headlines, 1):
        age = f"{item.age_days:.0f}d ago" if item.age_days is not None else "undated"
        publisher = item.publisher or "unknown source"
        if item.url:
            # Scoped by symbol so definitions stay unique across the document.
            ref = f"{enriched.symbol.lower()}-{i}"
            title = f"[{item.title}][{ref}]"
            references.append(f"[{ref}]: {item.url}")
        else:
            title = item.title
        lines.append(f"- *{age}, {publisher}* — {title}")

    if references:
        lines.append("")
        lines.extend(references)
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


def _how_to_read() -> str:
    return "\n".join(
        [
            "---",
            "",
            "## How to read this",
            "",
            "The pipeline makes **exactly two LLM calls per candidate** and nothing "
            "else it does involves a model. The screen, the arithmetic, the "
            "grounding checks and this report are all deterministic Python.",
            "",
            "- **Every number came from the data layer.** The model was handed "
            "finished comparisons — \"a 75% discount to the sector median of 69.2\" "
            "— rather than two numbers and a relationship to work out. It was never "
            "asked to calculate anything.",
            "- **Every figure in a verdict was traced back to its input.** Anything "
            "that did not trace is flagged above as ungrounded.",
            "- **A clean grounding report does not mean a correct analysis.** A real "
            "number can still support a false claim. That is what the audit is for, "
            "and the audit is a language model too.",
            "- **The audit can only lower conviction, never raise it.** A deliberate "
            "asymmetry: the expensive error is recommending a value trap, not "
            "missing an opportunity.",
            "- **The ordering above is not a ranking of investment merit.** It is "
            "verdict class, then whatever confidence survived the audit.",
            "",
            DISCLAIMER,
        ]
    )


def default_filename(as_of: datetime) -> str:
    """`contrarian-2026-08-24.md` — sorts chronologically, one per day."""
    return f"contrarian-{as_of.strftime('%Y-%m-%d')}.md"
