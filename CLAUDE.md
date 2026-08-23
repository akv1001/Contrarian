# ContrarianOppAIAgent

Scans the **Nifty 50** for contrarian value picks: companies whose price is depressed by
a *temporary* factor (news cycle, earnings miss, sector rotation) but whose fundamentals
suggest the business is sound. Outputs reasoning, not just tickers.

**This is idea generation, not investment advice.** LLM equity analysis is unreliable.
Output is a starting point for your own research, never a decision.

Full plan: `~/.claude/plans/ok-lets-use-direct-wise-candle.md`

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Scaffold + inference smoke test | **Done — verified end to end** |
| 2 | Data layer + Indian-coverage spike | Next |
| 3 | Deterministic screen | — |
| 4 | Analyze (LLM call 1) | — |
| 5 | Reflect (LLM call 2) | — |
| 6 | Report + end-to-end | — |

One phase per session. Commit at the end of each, then `/clear` before the next.
**Update this table when a phase completes** — it is what orients a fresh session.

## Pipeline

Exactly **two LLM calls** in the whole pipeline. Everything else is deterministic.

| # | Stage | Module | LLM? |
|---|---|---|---|
| 1 | Screen Nifty 50 on quantitative criteria | `contrarian/screen.py` | No |
| 2 | Enrich survivors: fundamentals + news | `contrarian/enrich.py` | No |
| 3 | Judge: temporary dip or value trap? | `contrarian/analyze.py` | **Yes** |
| 4 | Reflect: audit the verdicts | `contrarian/reflect.py` | **Yes** |
| 5 | Report | `contrarian/report.py` | No |

## Design rules

These are load-bearing. Breaking one reintroduces the failure mode the whole design
exists to prevent.

1. **The model never does arithmetic and never recalls a figure from training.** Every
   number reaching it comes from the data layer. Financial-QA benchmarks show numerical
   hallucinations are *silent* — confident wrong figures that survive a skim.
2. **`report.py` makes no LLM call.** Re-summarising through a model at the last mile
   would undo the grounding work.
3. **Stage 4 runs in a fresh context**, framed as third-party analysis. Never continue
   the stage-3 conversation and ask "are you sure?" — the model anchors on its own
   visible reasoning and agrees with itself.
4. **Reflection can only lower conviction, never raise it.** Deliberate asymmetry: the
   expensive error is recommending a value trap, not missing an opportunity.
5. **Stage 3 must argue both sides** — an explicit strongest-case-for-value-trap is
   required before any recommendation.

## Setup

```bash
python3 -m venv .venv            # required: Homebrew Python is PEP 668 managed
.venv/bin/pip install -r requirements.txt
cp .env.example .env             # then uncomment one provider block
.venv/bin/python scripts/smoke_test.py
```

Always use `.venv/bin/python`, never bare `python3`.

## Provider config

Three env vars in `.env` select the backend. Ollama, Groq, and OpenRouter all speak the
OpenAI-compatible protocol, so switching is config-only:

```
LLM_BASE_URL=http://localhost:11434/v1   # Ollama — free, unlimited, private, offline
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_BASE_URL=https://openrouter.ai/api/v1
```

Zero-cost is a hard project constraint: free tiers and local models only.

- Locally on 24 GB: **Qwen3-class at 14B/Q4** is the sweet spot. Avoid DeepSeek-R1
  distills for structured output — they deliberate instead of emitting.
- Most no-credit-card tiers train on submitted prompts. Fine for public market data;
  don't extend this pipeline to anything personal.

## Gotchas

- **Never export `ANTHROPIC_API_KEY`.** A globally set value makes Claude Code bill at
  pay-as-you-go API rates instead of using a Pro subscription. This project doesn't use
  Anthropic at all; `config.py` warns if it sees the variable.
- **NSE tickers need a `.NS` suffix** in yfinance (`RELIANCE.NS`, `INFY.NS`).
- **yfinance is an unofficial scraper.** Wrap every call so schema drift fails loudly
  rather than silently returning `None`. `jugaad-data` is the documented fallback for
  price data — it targets the current NSE site, unlike most NSE libraries.
- **News coverage for Indian tickers is thin** in yfinance, and news is exactly what
  identifies the "temporary factor." Fallback is Google News RSS (free, no key).
- Cache market data to disk with a TTL. Fundamentals move quarterly; refetching every
  run burns rate limit and slows iteration.
- **Qwen3 is a hybrid reasoning model.** In the Phase 1 smoke test it spent ~300 of 483
  completion tokens on internal thinking that Ollama stripped from the response. It did
  not leak — but if it ever leaks into Phase 4's schema-constrained JSON, parsing breaks.
  Before building on structured output in Phase 4, confirm the thinking block is
  suppressed or reliably stripped, and budget completion tokens at roughly 3x the
  visible answer length.

## Local environment (already set up)

Ollama 0.32.15 installed via the Homebrew **formula** (not the cask), with `qwen3:14b`
pulled (9.3 GB). The server is **not** a login item — start it per session with:

```bash
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

Flash attention and the q8_0 KV cache cut memory pressure at long context, which matters
once Phase 4 feeds full financial statements plus news for several candidates at once.
Run `brew services start ollama` if you want it permanent.

## Layout

```
contrarian/
  config.py      # env loading + startup validation
  llm.py         # the ONLY place LLM calls are made
  universe.py    # Nifty 50 constituents          (phase 2)
  data/          # MarketDataSource protocol      (phase 2)
  screen.py enrich.py analyze.py reflect.py report.py
  schema.py      # Candidate, Verdict, Reflection
scripts/smoke_test.py
main.py          # CLI entry point                (phase 6)
```

`data/source.py` is the swap seam: `screen.py` and `enrich.py` depend on the protocol,
never on `yfinance` directly.
