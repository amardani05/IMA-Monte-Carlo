# Roadmap

Status as of 2026-10-14. Live at **https://ima-monte-carlo.vercel.app**

The five phases of the October roadmap ("From Status Quo to Prediction") are
built, and the harness has already changed the plan once: see "What the
harness decided" below. Numbers quoted are from `tools/backtest.py`; the run
files are in `backtests/` (ignored by git, reproducible).

**Nothing in this project calls a model or inference service.** Every figure is
a number filed with the SEC, a Yahoo Finance close, arithmetic over those, or
output from the simulation engine, and each auto-filled field carries its
source and as-of date.

---

## Phase 0, backtest harness: done

`montecarlo/backtest.py`, `tools/backtest.py`. Point-in-time auto-fill from
EDGAR `filed` dates (frames derived from period dates, since EDGAR attaches its
own frame label only to the latest filed copy of a fact), realised prices from
Yahoo, PIT and CRPS per snapshot, paired comparison between runs, pluggable
model and prefill variants.

Baseline, the model as it stood on 6 September, 462 snapshots across 15 names,
two-year horizon, 2016 to 2024:

| | Result | Target |
| --- | --- | --- |
| Coverage of the 50 / 80 / 90% bands | 27% / 44% / 53% | 50 / 80 / 90% |
| PIT in the two outer deciles | 56% | 20% |
| Mean log error of the median | +0.18 | 0 |

A deep U-shaped PIT histogram: the distributions were far too narrow, and on
balance too low.

## Phase 1, price history: done

`montecarlo/prices.py`. Daily closes via yfinance locally or the Yahoo chart
endpoint on Vercel (one attempt, fails fast on a 429); year-end EV/EBITDA from
closes, shares, net debt and FY EBITDA; five-year weekly beta against SPY and a
CAPM cost of equity with stated defaults. The public-float multiple proxy is
gone wherever a price series exists.

## Phase 2, priors that move: done, and tuned on the harness

- Metalog and log-metalog drivers specified as P10 / P50 / P90, sampled by
  closed-form inverse CDF; the three-term metalog's logit tails are held flat
  beyond P0.5 and P99.5 (`distributions.py`).
- Student-t copula, 5 degrees of freedom, correlations estimated from pooled
  year-over-year changes across the peer panel.
- Sector panels (`panel.py`): a curated 102-name universe with SIC codes from
  EDGAR, hand-picked comparables for MYRG, TDS and QuinStreet, cached by
  `tools/build_panels.py`. Panel statistics: margin, growth and multiple
  quantiles, a pooled partial-adjustment reversion rate, change correlations.
- `calibrate_drivers_v2`: quantiles shrunk toward the panel, centres on a
  reverting path, a multiple struck on sub-2% margins refused as an anchor.

### What the harness decided

The first v2 as written in the roadmap (centres blended toward peer medians,
multiple reverting toward the peer median, width inflated by the square root
of the horizon) was **rejected**: calibration improved (90% coverage 53% to
81%) but CRPS got worse, because it was wide and ran high. A parameter sweep on
84 snapshots (MYRG, TDS, QuinStreet, 2017 to 2024) after the minority-interest
correction:

| Variant | CRPS / spot | 90% coverage | PIT tails |
| --- | --- | --- | --- |
| Own history only (baseline) | 0.590 | 71% | 37% |
| v2, centres blended to peers | 0.617 | 96% | 12% |
| v2, own target, tight widths | 0.569 | 83% | 26% |
| **v2, own target, own growth centre, tight widths (adopted)** | **0.561** | **83%** | **25%** |

The finding: the peer panel earns its keep through **width**, not by moving the
centre. A company's own ten years understate what can happen to it, and the
panel fixes that; but pulling its margin or growth toward the peer median made
every variant worse. Centres now stay on the company's own record, reverting
toward its own long-run median. These are the production defaults
(`V2_DEFAULTS` in `marketdata.py`). The CRPS gain is inside one standard error
on 84 snapshots; the coverage gain is not. The full-panel run is in progress.

## Phase 3, mechanisms: done

In `engine.py`, each switched on per name from the UI, proposed from filings by
`marketdata.mechanisms`:

- **Cost structure**: gross margin from CostOfRevenue, fixed cost defined so
  that revenue x gross margin less fixed cost equals filed EBITDA, growing at
  its own history's rate. EBITDA margin becomes an output; a COGS thesis is a
  gross-margin P50.
- **Backlog**: year-one revenue = backlog x conversion + the uncovered share at
  the CAGR, from the remaining-performance-obligation tag (MYRG $2.8B, Quanta
  $33.6B). The conversion rate is an assumption and says so.
- **FCF-to-equity bridge**: FCF margin x EV/FCF, a second price reported beside
  the EBITDA bridge; their disagreement is shown, not hidden.
- **Justified multiple**: the drawn multiple becomes a function of drawn growth,
  ROIC from NOPAT over invested capital, tax rate from the filing, D&A share and
  a WACC; today's premium to the justified level halves over the horizon.

Stacking all four on MYRG widens P5 to P95 to $58 to $797, which is why the UI
asks for one at a time.

## Phase 4, expectations: done

- Market-implied margin, growth and multiple at spot (each holding the other
  two at P50), shown beside the analyst's P10 / P50 / P90 and flagged when
  outside them.
- The analyst's three-statement path as a percentile of the distribution.
- A calibration log endpoint and "Log this run" button recording every stated
  P10 / P50 / P90 for later scoring. Ephemeral on Vercel until
  `CALIBRATION_LOG_PATH` points somewhere durable.

## Data corrections the harness surfaced

- Minority interest and preferred stock now come off enterprise value. TDS
  carries $1.07B of preferred and $232M of minority interest; net debt moved
  from -$1,507M to -$201M.
- Filers without the dei share-count tag fall back to us-gaap counts.
- Any multiple struck on a sub-2% margin is excluded from history and refused as
  an anchor (QuinStreet 2017: 330x).
- Margin reversion rates estimated from a thin panel are shrunk toward 0.38 and
  capped at 0.5 (the telecom panel estimated 0.72 from 47 pairs).

## Open

- **TDS remains a sum-of-parts story.** The bridge now takes preferred and
  minority interest off, but an EV/EBITDA bridge on a holding company with a
  tower business, a wireless stake and $2.2B of cash is the wrong instrument.
  The FCF bridge and the warning are the mitigation; a sum-of-parts is not in
  this tool.
- **Yahoo from Vercel.** The chart endpoint rate-limits Vercel's egress, so the
  deployed lookup usually has no quote and no price-based multiple history (it
  falls back to the public-float proxy and the panel). `FINNHUB_API_KEY` fixes
  the quote; history needs a keyed provider or a scheduled job that writes the
  series into the repository.
- **Panels are a snapshot.** `tools/build_panels.py` should run on a schedule
  with the ticker-map refresh.
- **Backlog conversion** is a hand-set 0.55 / 0.70 / 0.85 until the 10-K's
  "expected within twelve months" figure is parsed.
- **Scheduled refresh** of peers, panels and the ticker map once the repository
  has a CI runner.
