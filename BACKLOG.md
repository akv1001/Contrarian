# Backlog

Wanted, deliberately not built. Both are picked up independently — neither blocks
the other, and neither blocks using the pipeline as it stands.

Each entry records *why it matters*, *what has already been established* (so the
groundwork is not redone), the *caveats*, and a *cheapest first step*.

---

## 1. Validation — the labelled value-trap set

**Status:** not started. Design notes and feasibility measured 2026-08-24.

### Why this is the item that matters

The original plan's Verification section says it plainly:

> Assemble a small labelled set of past Nifty 50 situations — genuine temporary dips
> vs. known value traps — and measure whether the pipeline flags the traps. **This is
> the only test of whether the thing actually works. Everything else tests that it
> runs.**

It was never a numbered phase, so it never got built. 168 unit tests, the weak-note
gate and a clean cold run all establish that the machinery behaves correctly. **None
of them establish that the stock picks are any good.** That distinction is the whole
of the remaining risk, and it is worth not blurring.

### Measured feasibility — the two blockers are cleared

I assumed a point-in-time backtest was impossible on a free stack. Measured, it
isn't. **These numbers are scrapes of sites that change — re-verify before relying on
them** — but the shape of the answer is what matters:

| What | Availability | Verdict |
|---|---|---|
| `Ticker.history(start, end)` | full daily history | price, 52-week band, trailing returns are **exact** at any past date |
| annual `financials` / `balance_sheet` | **5 periods, FY2022-FY2026** | enough to derive ROE, margin, revenue growth, FCF, leverage |
| `quarterly_balance_sheet` | **2 periods only** | quarterly point-in-time is **not** viable; annual is |
| `.info` | **current only** | not the blocker it looks like — see below |
| Google News RSS `after:` / `before:` | **works** | a 2023-Q1 ITC query returned 52 items, pubDates correctly inside the window |

Two of those deserve emphasis, because they are the ones that reverse the obvious
conclusion:

- **`.info` being current-only does not sink this.** P/E, P/B and market cap are
  *derivable* from price-at-date plus statement figures. Don't read "no historical
  `.info`" as "no historical fundamentals".
- **Historical news works.** This was the assumed hard blocker — the "temporary
  factor" signal is the one thing that cannot be reconstructed from a price series,
  so if news had been unavailable the whole exercise would have been pointless.
  `after:`/`before:` on the RSS query solves it.

### Design sketch

- **`contrarian/data/asof.py`** — an `AsOfSource` implementing the existing
  `MarketDataSource` protocol (`contrarian/data/source.py`) for a given as-of date.
  `screen.run(source, ...)` already takes the source as an argument, so **no stage
  needs modifying**. This is exactly the swap seam `source.py` was built for; use it
  rather than threading a date parameter through the pipeline. Follow the composition
  pattern in `news_rss.CompositeSource`.
- **Label mechanically from forward returns**, not from hindsight judgement about
  which names were "obviously" traps. Recovered against the index => temporary dip;
  stayed down or fell further => value trap; in between => excluded as ambiguous.
  Thresholds as named constants, recorded with every result, so the labelling is
  reproducible and arguable rather than a matter of opinion.
- **Evaluate the screen first, with zero LLM calls.** Run the deterministic screen at
  past dates and check whether its survivors beat the index. Free, minutes rather
  than hours. Two reasons this comes first: if the screen carries no signal, verdict
  quality cannot rescue it; and the question actually worth answering — *does the LLM
  add anything over the screen alone?* — needs the screen-only number as its
  baseline. Only then spend ~2.5 min/case on the full pipeline.

### Caveats to state in the write-up, not discover later

- `universe.NIFTY_50` is **today's** constituents, so any historical run carries
  survivorship bias. The index reconstitutes roughly semi-annually.
- Financials as filed today may have been **restated** since the as-of date.
- Google's index of old news is not what was **visible at the time**, and coverage of
  older stories decays.
- **Small-n.** Even 60 cases is thin for a claim about equity returns. The honest
  output is a signal and a direction, not proof.

### Cheapest first step

The screen-only evaluation. It costs no inference, takes minutes, and may settle the
question before any LLM budget is committed.

---

## 2. Try a free-tier hosted model

**Status:** not started. No code change required.

### Why

Stage 3 does not commit. Measured over a full run: **9 of 9 candidates came back
UNCLEAR, 8 of them at exactly 55.** All the spread in the final report comes from the
audit, not from the analyst reaching a view.

It is currently unknown whether that is the **prompt** (which offers UNCLEAR as an
explicitly blessed answer, and gets taken up on it every time) or a **14B capability
ceiling**. Running a stronger model against the identical prompt is the fastest way
to tell those apart — and it is the cheaper experiment to run first, because if a
better model commits, the prompt is fine and the ceiling was real.

### Already true, so don't rebuild it

- The switch is **config-only**: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in
  `.env`. `contrarian/llm.py` has no provider branching — that was the point of
  writing against the OpenAI-compatible protocol.
- `.env.example` already carries commented **Groq** and **OpenRouter** blocks.

### Caveats

- **Structured output is the real risk.** `chat_structured()` relies on
  `response_format=json_schema` with `strict: true`, and not every provider or model
  supports it. `llm.py` already raises a specific `LLMError` naming this case, so the
  failure will be legible rather than mysterious — but expect to have to shop for a
  model, not just a provider.
- Free tiers run ~30 requests/min. Irrelevant for an 18-call run; relevant if it is
  ever pointed at a large evaluation sweep (item 1).
- Most no-credit-card tiers **train on submitted prompts**. Fine for public market
  data — already noted in `CLAUDE.md` under Provider config — but it is the reason
  not to extend this pipeline to anything personal.
- Zero cost is a hard project constraint. Free tiers only; no pay-as-you-go.

### Cheapest first step

```bash
.venv/bin/python scripts/run_analyze.py --limit 3
```

Run it hosted, and compare the verdict and confidence spread against the recorded
local baseline (9/9 UNCLEAR, 8 at exactly 55). One command, no code change, and it
answers the prompt-vs-ceiling question directly.

---

## 3. Fetch article text for a subset of headlines

**Status:** not started. Shape agreed, blockers measured 2026-08-24.

### The idea

Keep Google News RSS as the headline source. It is free, unlimited, pinned to the
Indian edition, and supports `after:`/`before:` for item 1. Nothing replaces it.

Then, for **only the 3-4 headlines that look like the temporary factor**, fetch the
actual article text via a search API (Tavily or similar) and give the model that
instead of a bare title.

Right now the model infers the whole temporary factor from 15 titles. `summary` is
`None` — the RSS descriptions are boilerplate link markup, so there is genuinely no
article text anywhere in the pipeline.

### Blocker 1 (measured): you cannot follow the RSS link

Google News gives a link like `news.google.com/rss/articles/CBMimAFBV...`. Fetching
it returns **HTTP 200 and a 580KB Google page**, not the article — the redirect to
the publisher happens in JavaScript. Tested directly; it does not resolve.

**This is why a search API is the right tool and a fetcher/extractor is not.** You
cannot retrieve the URL you already hold. You have to take the headline text and
search for the article again. Anyone scoping this will assume it is a two-line
`urlopen` and lose an afternoon; it isn't.

### Blocker 2 (unsolved): which 3-4 headlines?

Something must choose. Two options, both with a real cost:

- **A keyword rule** — match "results", "downgrade", "tax", "fine", "guidance".
  Cheap, deterministic, no extra LLM call. Crude, and will miss unusual phrasing,
  which is often the interesting case.
- **Ask the model** — much better selection, but it is a **third LLM call**, and
  "exactly two LLM calls" is a load-bearing design rule. Breaking it is a decision to
  take deliberately, not a detail to slip in.

No obvious right answer. This is the part to think hardest about.

### The cost that is easy to miss

Article bodies enlarge the **untrusted number pool**. Measured on ITC: the pool is
currently 30 values (15 from titles, 15 age markers), against 53 verified values (43
unique). Three or four full articles could triple the untrusted side.

That directly weakens `check_grounding()` — with more numbers allowed, coincidental
matches get likelier, and a written "18%" matching an unrelated 18.1% is already a
documented problem. **Richer context is bought with a weaker hallucination check.**
Worth it only if the articles measurably improve verdicts.

Also note the precision issue compounds. 3 of ITC's 15 headlines were about **ITC
Hotels**, a separately listed company since the 2025 demerger. Fetching the full text
of a wrong-company headline is worse than showing the wrong title — more words, more
numbers, more conviction behind a false story.

### Tavily specifics worth knowing up front

- **1,000 free credits/month**, no credit card. Basic search 1 credit, advanced 2.
- Fine for normal runs (~9 news fetches each). **Not fine for item 1's backtest** —
  51 tickers x 8 dates = 408 queries per sweep. Another reason RSS stays.
- **`country` targeting only works when `topic` is `general`, not `news`.** So you
  cannot have news-optimised ranking *and* India targeting at once. Given that the
  India edition pinning is what keeps US "Reliance" (Reliance Steel, NYSE: RS) out of
  the results, do not assume a Western index will do better on NSE names.
- Free tiers are a dependency with a shelf life — Brave removed its 5,000/month free
  plan in Feb 2026. Weigh that against RSS's different failure mode (unofficial, may
  break, but free forever).

### How to evaluate it

The seams already exist. `MarketDataSource` is a Protocol and `CompositeSource`
composes fundamentals and news independently, so a `TavilyNews` class is a drop-in
with no stage changes. `scripts/coverage_spike.py` already compares sources
head-to-head — `yahoo.py` keeps its `news()` *only* so that comparison can be re-run.
Add a third source to that spike.

**Then ignore the summary statistic and read a real prompt.** The Phase 2 metric
scored the broken ITC query at 87% "name-matched" while a third of the results were
about Input Tax Credit and ITC Hotels. The measurement was blind to the exact failure
it existed to detect.

### Priority

**Below item 2.** The model currently answers UNCLEAR to all nine candidates. Better
news would be improving the input to a stage that is not using the input it already
has. Fix the clustering first, then find out whether news depth is what is still
missing.
