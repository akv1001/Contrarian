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
| 2 | Data layer + Indian-coverage spike | **Done — 51/51 tickers fetch clean** |
| 3 | Deterministic screen | Next |
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

## Data coverage — measured, not assumed (Phase 2)

Re-run `.venv/bin/python scripts/coverage_spike.py` before trusting any of this;
it is a scrape of a site that changes. Numbers below are from 2026-08-23, 51 tickers.

**Safe to screen on — 100% coverage:** `price`, `market_cap`, `fifty_two_week_high/low`,
`average_volume`, `price_to_book`, `free_cash_flow`, `operating_cash_flow`,
`return_on_equity`, `profit_margin`, `revenue_growth`, `total_debt`, `trailing_eps`,
`book_value`, `total_revenue`, `forward_pe`.

**Every remaining gap is one sector, not random flakiness:**

| Field | Missing | Where |
|---|---|---|
| `ev_to_ebitda` | 9/51 | all Financials |
| `debt_to_equity` | 7/51 | all Financials |
| `operating_cash_flow` | 2/51 | Financials |
| `trailing_pe` | 1/51 | TMPV — loss-making, so no trailing P/E exists |

This is **structural, not a data gap**. EV/EBITDA and debt/equity are not defined
quantities for a bank. Phase 3 must branch: screen Financials on P/B and ROE, and
everyone else on the EV and leverage filters. Treating absence as a failing value
would drop all 9-10 financials from the universe silently — and Financials is the
single largest sector in the Nifty 50.

39/51 tickers carry a complete `SCREEN_FIELDS` record; the 12 that don't are the
financials above.

### News: Google News RSS won decisively

The plan called RSS a fallback. Measured, it is the primary source:

| Feed | Items/ticker | Name-matched |
|---|---|---|
| yfinance `.news` | 9.6 | 57% |
| Google News RSS | 94.3 | 87% |

Yahoo's feed is not just thin, it is *wrong* — a `RELIANCE.NS` query returned a
Rolls-Royce headline and a story from Odessa, Texas. Since news is the only signal
that identifies the "temporary factor," feeding Phase 4 headlines about the wrong
company leads straight to the failure mode this design exists to prevent.
`news_rss.py` is the news source; `yahoo.py` keeps its `news()` only so the spike
can keep re-checking the comparison.

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
- **NSE tickers need a `.NS` suffix** in yfinance — but not always. Always go through
  `universe.yahoo_symbol()`, which consults `YAHOO_SYMBOL_OVERRIDES`. Tata Motors
  demerged: NSE kept `TATAMOTORS` for the CV entity, Yahoo files it under **`TMCV.NS`**,
  and `TATAMOTORS.NS` now 404s. `TMPV.NS` is the passenger-vehicle entity. Both are in
  `universe.py` (so it holds 51 entries, not 50) — confirm which the index actually
  holds at the next review.
- **A 404 is the friendly failure. The nasty one is a stub.** `TATAMOTOR.NS` (one letter
  off) returns 25 keys of exchange metadata with no price, which passes any key-count
  sanity check and then reads downstream as "this company has no debt and no earnings."
  `yahoo.py` requires at least one of `regularMarketPrice`/`currentPrice`/`previousClose`/
  `marketCap` before accepting a payload.
- **`.info` lacks `freeCashflow`, `operatingCashflow`, `returnOnEquity`, and
  `fiftyTwoWeekChange` for NSE tickers** even though it carries them for US ones. Don't
  conclude the data is unavailable — FCF and operating CF come from the cash flow
  statement, ROE is derived from net income / equity, and the 12-month return is computed
  from daily closes in `price_stats`. All are marked in `Fundamentals.provenance`.
- **yfinance `.news` items are nested** as `{"id", "content": {...}}` in 1.6.0. Older
  tutorials assume a flat dict with `title`/`link`/`providerPublishTime`; that shape is
  gone.
- **`None` never means zero.** Phase 3 must exclude on absent data, not treat it as a
  failing value, or patchy coverage makes a ticker look like a bargain.
- Cache market data to disk with a TTL — `contrarian/data/cache.py`, one JSON file per
  key under `.cache/`. `rm -rf .cache/` is a valid flush. A cold spike run takes minutes;
  a warm one takes seconds.
- Cache keys must survive `M&M` and `BAJAJ-AUTO` — real NSE symbols, and an unescaped
  `&` in a path is a bug waiting for one specific ticker.
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
  universe.py    # Nifty 50 constituents + Yahoo symbol overrides
  data/
    source.py    # MarketDataSource protocol, Fundamentals, PriceStats, NewsItem
    yahoo.py     # yfinance impl — the ONLY file importing yfinance
    news_rss.py  # Google News RSS + CompositeSource + default_source()
    cache.py     # disk cache w/ TTL
  screen.py enrich.py analyze.py reflect.py report.py    (phases 3-6)
  schema.py      # Candidate, Verdict, Reflection        (phase 3+)
scripts/
  smoke_test.py      # phase 1 — one real LLM call
  coverage_spike.py  # phase 2 — the coverage matrix; re-run it when data looks off
tests/test_data_layer.py
main.py          # CLI entry point                       (phase 6)
```

`data/source.py` is the swap seam: `screen.py` and `enrich.py` depend on the protocol,
never on `yfinance` directly. Build a source with `news_rss.default_source()` rather than
constructing `YahooSource` at a call site.

```bash
.venv/bin/python scripts/coverage_spike.py            # full matrix, warm cache = seconds
.venv/bin/python scripts/coverage_spike.py --limit 5 --verbose
.venv/bin/python -m unittest discover tests           # 31 tests, no network
```
