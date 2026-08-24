# ContrarianOppAIAgent

Scans the **Nifty 50** for contrarian value picks: companies whose price is depressed by
a *temporary* factor — a news cycle, an earnings miss, sector rotation — but whose
fundamentals suggest the business is sound. It outputs reasoning and the figures behind
it, not just tickers.

> **This is idea generation, not investment advice.**
>
> The verdicts are written by a language model, and LLM equity analysis is unreliable.
> The *figures* are grounded and machine-checked; the *reasoning* is not. Treat the
> output as a starting point for your own research and never as a decision. Nothing
> here has been validated against whether it actually picks winners — see
> [Does it work?](#does-it-work).

## What it does

Five stages. **Two of them use a language model; three are ordinary Python that
behaves the same way every time.**

That split is the point. Anything a computer can do exactly — arithmetic,
comparisons, filtering, formatting — is kept away from the model, because the model
is the part that can be confidently wrong.

| # | Stage | What happens | LLM? |
|---|---|---|---|
| 1 | Screen | 51 tickers against quantitative gates, sector-aware | No |
| 2 | Enrich | Fundamentals + news; **all arithmetic resolved here** | No |
| 3 | Analyze | Temporary dip, or value trap? | **Yes** |
| 4 | Reflect | A fresh reader audits the verdict | **Yes** |
| 5 | Report | Markdown, figures beside the reasoning | No |

## The stages in detail

### 1. Screen — narrow 51 stocks to a handful

Downloads price history and financial statements for all 51 Nifty 50 companies,
then applies nine fixed rules. No AI involved — just comparisons against hard-coded
thresholds.

The rules test three things: is the price **beaten down** (at most 40% up its
52-week range, *and* at least 15% below its high), is the business **sound** (positive
free cash flow, debt under 1.5x equity, return on equity above 10%), and is it
**cheap against its own sector** (P/E below the sector median).

Banks get a different rule set. Free cash flow and debt ratios don't mean the same
thing for a lender, where borrowing is the raw material rather than a warning sign.

Everything is fetched before anything is filtered, because you can't know the IT
sector's median P/E until every IT stock has landed. Typical result: **9 survivors of
51.** Rejections are kept, not discarded — they're how you tell whether a threshold
is doing real work.

### 2. Enrich — assemble what the model is allowed to know

For each survivor only, this fetches recent news headlines and **does all the
arithmetic in advance**.

Every comparison is worked out in Python and handed over as a finished sentence.
The model receives:

> `Valuation vs peers: P/E of 16.9 — a 75% discount to the Consumer median of 68.9`

It does not receive "16.9" and "68.9" as two separate numbers to compare. The 75% is
already calculated. This is the single most important defence in the project, and
[it earned itself](#why-it-is-built-this-way).

News comes from Google News RSS, pinned to the Indian edition, filtered to the 15
most recent headlines within 90 days. Headlines only — no article text.

### 3. Analyze — the judgement call *(LLM)*

One call per candidate, each in isolation so a strong story about one company can't
colour the verdict on the next.

The model is given the figure sheet, the headlines, and one question: is this price
depressed by something **temporary**, or is it cheap for a **real structural
reason**?

It must answer in a fixed JSON structure, and the order of fields is deliberate.
The bear case comes *before* the verdict, so the model physically writes the argument
against the stock before it has committed to an opinion. It can't reach a conclusion
and then work backwards to justify it.

Afterwards — in Python, not by the model — every number in the answer is traced back
to its source. Anything untraceable is flagged as invented.

### 4. Reflect — audit the verdict *(LLM)*

A second call re-reads the verdict against the same source data.

Three things make this more than a rubber stamp. It runs with **no memory of the
first call**, so it isn't defending its own reasoning. The verdict is shown
**unattributed** — as a note from an unnamed analyst, not as "your earlier answer."
And it sees **byte-for-byte the same figures**, so "this cites a number that isn't in
the data" can't be a rendering difference.

It returns KEEP, DEMOTE or REJECT with an adjusted confidence. **The adjustment can
only lower conviction, never raise it** — and that's enforced by Python afterwards,
not requested in the prompt. Worth it: across a full run, 8 of 8 demotions came back
with the confidence unchanged. The model said "materially weakened" and handed back
the same number every time.

### 5. Report — write it up

Turns everything into a Markdown file. No model involved, deliberately — by this
point every figure has been sourced, computed, and checked twice, and running it
through a model again would reintroduce exactly the drift the other four stages
exist to prevent.

Each candidate gets its screen results with the numbers, the verdict, the audit's
findings, the full figure sheet, and the headlines. The figures sit **next to** the
reasoning that uses them, so a doubtful sentence can be checked without leaving the
page.

## Why it is built this way

The design assumes the model will get numbers wrong, because in financial QA it
silently does — confident wrong figures that survive a skim. So:

- **The model is never asked to calculate.** Every comparison is resolved in Python
  first and handed over finished: not "P/E 17.0, median 69.2" but "a 75% discount to the
  Consumer median of 69.2". Given the raw pair, the model wrote *"P/E 17.0 is 69.2%
  below sector median"* — folding a real number into invented arithmetic.
- **Every figure in a verdict is traced back to its input.** Anything that doesn't trace
  is flagged in the report as ungrounded.
- **The bear case is written before the verdict exists**, enforced by field order in the
  output schema rather than asked for politely.
- **The audit runs in a fresh context** and sees the note unattributed — a model shown
  its own reasoning and asked "are you sure?" tends to agree with itself.
- **The audit can only lower conviction, never raise it.** Enforced in code. Missing an
  opportunity is cheap; recommending a value trap is not.

## Setup

Requires Python 3 and a running [Ollama](https://ollama.com) (or any OpenAI-compatible
endpoint — Groq and OpenRouter both work, config-only).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env             # then uncomment one provider block
.venv/bin/python scripts/smoke_test.py
```

For the local default:

```bash
ollama pull qwen3:14b
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

Zero cost is a hard constraint — free tiers and local models only.

## Run it

```bash
.venv/bin/python main.py                # full run; writes reports/contrarian-<date>.md
.venv/bin/python main.py --limit 3      # a quicker look
.venv/bin/python main.py --out -        # print instead of writing
```

Roughly two minutes per candidate on a local 14B model. Individual stages:

```bash
.venv/bin/python scripts/run_screen.py --show-rejects 10   # tune the screen here
.venv/bin/python scripts/run_analyze.py --limit 3
.venv/bin/python scripts/run_reflect.py --weak ITC         # audit a deliberately bad note
.venv/bin/python -m unittest discover tests                # 168 tests, no network
```

## Does it work?

Unknown, honestly. It runs, it is internally consistent, and it catches a note built to
fool it. But **it has never been measured against a labelled set** of past situations —
real temporary dips versus known value traps — which is the only test of whether the
picks are any good. Everything currently tested establishes that the machinery behaves.

Current known limits: verdicts cluster at UNCLEAR 45-55 rather than committing;
confidence scores vary run to run; the peer comparison is weak because 51 stocks is a
thin universe; and it reads headlines, never article bodies.

## Layout

See [CLAUDE.md](CLAUDE.md) for the full design notes — every load-bearing decision, the
measured data-coverage findings, and the failures that produced each rule.
