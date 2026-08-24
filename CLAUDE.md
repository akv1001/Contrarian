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
| 3 | Deterministic screen | **Done — 9 survivors of 51** |
| 4 | Analyze (LLM call 1) | **Done — grounded JSON verdicts** |
| 5 | Reflect (LLM call 2) | **Done — weak-note gate passes** |
| 6 | Report + end-to-end | Next |

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
   **`Candidate.figures()` is the mechanism.** Build the Phase 4 prompt from that dict
   and nothing else; a number in a verdict that is not a value in it is a hallucination,
   and that is a mechanical check rather than a judgment call.
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

**Every query carries a market-context clause** (`share OR shares OR stock OR
earnings OR results OR profit`) on top of the quoted company name. ITC forced this:
a bare `"ITC"` query returned headlines about *Input Tax Credit*, an ILO training
centre, and ITC Hotels — a separately listed company since the 2025 demerger. Only
about a third of the first fifteen results concerned ITC Ltd.

Note how that slipped through: the Phase 2 relevance metric scored it **87%
name-matched**, because the token "itc" does appear in every one of those
headlines. The measurement was blind to exactly the failure it existed to detect,
and it was only caught by reading a real prompt. Adding the qualifier also surfaced
ITC's genuine temporary factor — a cigarette tax hike denting Q1, with brokerages
upgrading as the concern eased — which the bare query had buried. **Read a real
prompt before trusting a coverage statistic.**

## The screen (Phase 3)

Three gates, in order: **eligibility** (does the source have the fields this
sector's ruleset needs — absent data excludes, never passes), **depressed**
(<= 40% of the 52-week range AND >= 15% off the high), then **quality + value**.

Two rulesets, because of the Phase 2 finding:

| | Non-financials | Financials |
|---|---|---|
| Valued on | trailing P/E | price/book |
| Cash test | free cash flow > 0 | *none* |
| Leverage | debt/equity <= 1.5x | *none* |
| Returns | ROE >= 10%, positive margin | same |

Financials get no FCF or leverage test because those aren't meaningful for a
bank — borrowing is the raw material, and "free cash flow" is an artefact of how
financing flows are classified (AXISBANK -10,847cr vs HDFCBANK +106,578cr; neither
means what it would for a manufacturer).

Ranked by `40% depression + 40% discount to peers + 20% quality`, each component
kept separately on the `Candidate` so a rank can be explained rather than asserted.

```bash
.venv/bin/python scripts/run_screen.py                     # ranked survivors
.venv/bin/python scripts/run_screen.py --show-rejects 10   # closest misses — tune here
.venv/bin/python scripts/run_screen.py --explain ITC       # every filter + the figures
```

### Live result, 2026-08-23

9 of 51 passed: ITC, HINDUNILVR, COALINDIA, ONGC, INFY, HDFCBANK, NTPC, WIPRO, TCS.
Rejection counts: `near_52w_low` 31, `meaningful_drawdown` 29, `cheap_vs_peers` 24,
`return_on_equity` 8, `positive_fcf` 6, `leverage` 2.

### Known weaknesses — hand these to Phase 4, don't paper over them

- **Small sectors fall back to an index-wide median, and that comparison is weak.**
  Energy has 3 members, so COALINDIA and ONGC were first scored a ~75% "discount"
  against the index P/E median of 29.7 and ranked 2nd and 3rd. But commodity
  producers trade on single-digit multiples *everywhere* — that gap is a sector
  characteristic, not a dislocation. `FALLBACK_DISCOUNT_DAMPING = 0.5` halves the
  score contribution when the basis is a fallback. It still gates; it can no
  longer dominate the ranking. The real fix is a peer universe wider than 51
  stocks, which we don't have.
- **A derated sector makes "cheap vs peers" nearly free.** 3 of 5 IT names passed,
  and TCS passed sitting *exactly* on the IT median (discount 0.00). Sector
  rotation is a legitimate temporary factor, so this is arguably the screen
  working — but the peer test carries no information when the whole peer group
  has derated together. Phase 4 should be told the peer basis, not just the gap.
- The candidate is included in its own peer median. With 8-12 names the bias is
  negligible and one stable published figure per sector is worth more for auditing.

## Enrich + analyze (Phase 4)

`enrich.py` is deterministic and does **all the arithmetic**. Every comparison is
resolved in Python and handed over as a finished sentence — not "P/E 17.0, median
69.2" but "P/E of 17.0 — a 75% discount to the Consumer median of 69.2".

This is the defence that actually works, and it earned itself. Given the raw pair,
qwen3:14b wrote *"P/E 17.0 is 69.2% below sector median"* — folding a real input
number into invented arithmetic. Given the discount pre-computed, it reported it
correctly. **Never hand the model two numbers and a relationship to infer.**

```bash
.venv/bin/python scripts/run_analyze.py --limit 3
.venv/bin/python scripts/run_analyze.py --symbol INDUSINDBK   # works on rejects too
.venv/bin/python scripts/run_analyze.py --limit 1 --show-prompt
```

### Structured output works — the Phase 1 risk is closed

`response_format=json_schema` with `strict: true` works on Ollama + qwen3:14b.
Parses first try, and **the `<think>` block does not leak** — constrained decoding
leaves no room for it. `llm.chat_structured()` is the entry point; `strict_schema()`
inlines `$ref`s because Ollama compiles the schema to a GBNF grammar and does not
resolve them.

**Field order in a Pydantic model is generation order.** `Verdict` puts
`value_trap_case` before `verdict`, so the model physically writes the bear case
before a verdict exists. That is design rule 5 enforced structurally rather than
requested politely — keep the order if you edit the model.

### Confidence needs an explicit rubric

Without one the scale collapses: the first full run returned confidence 4, 0, 0.
With the rubric in `analyze.SYSTEM` the same input scores 45-55. Still treat it as
soft — the spike scored one value-trap case 60 then 15 on consecutive runs.

Current behaviour: **everything comes back UNCLEAR at 45-55.** Self-consistent with
"genuinely balanced", and safe given rule 4, but a pipeline that never commits is
of limited use. Phase 5 improves this a little — reflection spreads the survivors
to 40/40/55 — but the spread comes from the audit, not from stage 3 committing.
Still the open question going into Phase 6.

### What the grounding check does and does not do

`check_grounding()` traces every number in the verdict back to the input. It
returns `(issues, headline_sourced)` — a figure quoted from a headline is not an
error, but it came from third-party news text rather than the data layer, so it is
tracked separately.

**It catches invented figures. It does not catch real figures used to support a
fabricated claim** — the spike's "17.0 is 69.2% below" passes cleanly, because
both numbers are real. A clean grounding report is not a correct analysis. That
gap is what Phase 5 closes via `Reflection.claim_check`, and there is a test
asserting the blind spot so it cannot be quietly forgotten. A probe handed
qwen3:14b that exact sentence and it caught the fabricated relationship first
try — which is the evidence the reflection stage earns its LLM call.

`check_grounding(verdict, enriched, *, fields=..., also_allowed=...)` — `fields`
lets stage 4 run the identical check over a `Reflection`; `also_allowed` widens
the verified pool for it. Both default to stage 3's behaviour.

Two live false positives are fixed and regression-tested: rounding is checked at
the precision the model actually wrote (input 0.033 vs written "0.03"), and
time-window idioms are stripped so "52-week low" does not report 52 as invented.
Coincidental collisions remain possible — with ~53 allowed numbers, a written "18%"
can match an unrelated 18.1% margin. Another reason not to over-trust a clean report.

## Reflect (Phase 5)

Stage 4 re-reads each verdict against the same source data with fresh eyes, and
answers the question `check_grounding` cannot: do the *claims* follow, not just
the figures. Output is `KEEP`/`DEMOTE`/`REJECT` plus an adjusted confidence.

```bash
.venv/bin/python scripts/run_reflect.py --weak ITC     # the phase gate
.venv/bin/python scripts/run_reflect.py --limit 3      # the real pipeline
.venv/bin/python scripts/run_reflect.py --show-prompt --limit 1
```

`--weak` substitutes a note built to fail every check — an invented 34.7% growth
figure, a mechanism-free catalyst, a one-clause brush-off of the bear case, and
confidence 88 on all of it. It runs against a real candidate's real data and
prints GATE PASSED/FAILED rather than output you have to interpret. **Keep it
working**; it is the only thing that tests whether this stage does anything.

### Both prompts must be built from `enrich.figure_sheet()`

The audit only means something if stage 4 saw exactly what stage 3 saw —
otherwise "the note cites a figure that isn't in the data" is ambiguous between a
hallucination and a rendering difference. One function, called by both, and a
test asserts the rendered block appears verbatim in both prompts.

### The asymmetry is arithmetic, not instruction

`apply_asymmetry()` clamps the result in Python and records every override on
`ReflectedCandidate.overrides`. Design rule 4 is not left to the model, and the
live runs show why: **2 of 2 DEMOTEs returned the confidence unchanged** — the
reflector wrote "materially weakened" and handed back 55 of 55. `DEMOTE_MIN_DROP`
exists because of exactly that.

- Confidence can never exceed the note's.
- An ungrounded figure blocks `KEEP` outright.
- `DEMOTE` must drop at least 15.
- `REJECT` caps at 20, and withdraws a `TEMPORARY_DISLOCATION` call to `UNCLEAR`.
- **`VALUE_TRAP` is never rewritten.** Every rule moves conviction *away* from the
  opportunity, so flipping a value-trap call to UNCLEAR for sloppy reasoning would
  *raise* conviction — the one thing rule 4 forbids. A badly argued warning is
  still a warning.

**Relative floors are not enough on their own, and the first gate run proved it.**
The weak note claimed 88; the reflector demolished it in detail and returned 55.
Every relative rule was satisfied — a 33-point drop, twice `DEMOTE_MIN_DROP` — and
a note resting on a fabricated figure still came out a moderate-conviction idea.
The floor is anchored to a number *the note chose for itself*, so an inflated
starting confidence drags it up. `UNGROUNDED_CONFIDENCE_CEILING = 25` is absolute
for that reason. Watch for the same anchoring anywhere else conviction is derived
from a model-supplied number.

### Live behaviour, 2026-08-24

Three real candidates: COALINDIA `KEEP` 55, ITC and HINDUNILVR `DEMOTE` 55->40
(both overruled). So the stage discriminates rather than reflexively demoting —
worth re-checking, since a reflector that demotes everything is as useless as one
that keeps everything. Post-reflection confidences finally spread (40/40/55)
where stage 3 alone returns 45-55 for everything.

Run-to-run variance is real here too: the same weak note scored `DEMOTE`/55 and
then `REJECT`/0 on consecutive runs. The gate passes on both because the clamps
are mechanical — which is the argument for having them.

### The reflection is itself grounding-checked

`check_reflection_grounding()` runs the same numeric audit over the audit.
**Its allowed pool is every number in the audit prompt**, not just the verified
figures — the first live run flagged the reflector three times for correctly
quoting the note it was reviewing: the fabricated 34.7 it had just caught, and
the 88 and 100 from "stated confidence 88/100". Reported, never acted on; a bad
audit is a reason to distrust the audit, not to demote the candidate twice.

Known wart: qwen3:14b sometimes bleeds content across the five finding fields —
one COALINDIA run put calibration reasoning in `catalyst_concreteness`. Harmless
for now, but don't build a report that assumes each field stays in its lane.

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
- **`None` never means zero.** The screen excludes on absent data rather than treating
  it as a failing value — otherwise patchy coverage makes a ticker look like a bargain.
- **Yahoo's `debtToEquity` is a percentage, not a ratio.** `36.65` means 0.37x. Index
  median is 29.4, max 315. A ceiling written as `2.0` would silently reject all 51 names
  and the screen would look merely strict. `Candidate.figures()` exposes it as
  `debt_to_equity_pct` so the unit cannot be misread downstream.
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
  screen.py      # stage 1 — deterministic, no LLM
  enrich.py      # stage 2 — deterministic; does ALL the arithmetic + figure_sheet()
  analyze.py     # stage 3 — LLM call 1, + the grounding audit
  reflect.py     # stage 4 — LLM call 2, + the asymmetry clamps
  schema.py      # Candidate, Fact, EnrichedCandidate, Verdict, AnalyzedCandidate,
                 #   Reflection, ReflectedCandidate
  report.py                                              (phase 6)
scripts/
  smoke_test.py      # phase 1 — one real LLM call
  coverage_spike.py  # phase 2 — the coverage matrix; re-run it when data looks off
  run_screen.py      # phase 3 — ranked survivors, with every filter's numbers
  run_analyze.py     # phase 4 — screen -> enrich -> judge, with grounding report
  run_reflect.py     # phase 5 — the above plus the audit; --weak is the gate
tests/  test_data_layer.py  test_screen.py  test_analyze.py  test_reflect.py
                 # 137 tests, no network
main.py          # CLI entry point                       (phase 6)
```

`data/source.py` is the swap seam: `screen.py` and `enrich.py` depend on the protocol,
never on `yfinance` directly. Build a source with `news_rss.default_source()` rather than
constructing `YahooSource` at a call site.

```bash
.venv/bin/python scripts/coverage_spike.py            # full matrix, warm cache = seconds
.venv/bin/python scripts/coverage_spike.py --limit 5 --verbose
.venv/bin/python -m unittest discover tests           # 137 tests, no network
```
