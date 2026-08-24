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

Five stages, and **exactly two LLM calls per candidate**. Everything else is
deterministic Python.

| # | Stage | What happens | LLM? |
|---|---|---|---|
| 1 | Screen | 51 tickers against quantitative gates, sector-aware | No |
| 2 | Enrich | Fundamentals + news; **all arithmetic resolved here** | No |
| 3 | Analyze | Temporary dip, or value trap? | **Yes** |
| 4 | Reflect | A fresh reader audits the verdict | **Yes** |
| 5 | Report | Markdown, figures beside the reasoning | No |

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
