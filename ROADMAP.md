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
(`V2_DEFAULTS` in `marketdata.py`).

**Full panel, 488 paired snapshots across 17 names, 2016 to 2024** (after two
more corrections the run forced: the bridge now refuses names with an EBITDA
margin under 2%, where EverQuote produced a $27,000 P95 on a $7 stock; and a
peer panel may widen a company's own range by at most a factor of two):

| | Own history (baseline) | v2 adopted |
| --- | --- | --- |
| CRPS / spot | 0.790 | 0.789 |
| Better on snapshots | | 53% |
| Better on names | | 10 of 17 |
| 90% coverage | 48% | 65% |
| PIT in outer deciles | 65% | 46% |

The sharpness gate is met only at the margin: CRPS is a tie (delta -0.001,
standard error 0.012). The calibration gate is met clearly. v2 ships because it
is as sharp as the baseline and much better calibrated, not because it is more
accurate at the centre. Two things the numbers say about what is left:

- Coverage at 65% is still well short of 90%, and the by-year breakdown says
  where: 90% coverage was 87 to 92% on snapshots taken in 2016, 2017 and 2019,
  fell to the mid-60s in 2018 and 2020, and collapsed to about 50% for 2022 to
  2024, with the median running 40 to 50% low. The model is calibrated in an
  ordinary regime and badly too narrow in the 2022 to 2024 engineering and
  construction re-rating, when several names rose two to ten times. That is the
  regime risk a filings-only prior cannot see coming, and it argues for a
  market-state input (sector multiple momentum, rate level) in a later phase.
- The median still runs low (mean log error +0.18). Part of that is the sample:
  2017 to 2024 was an exceptional stretch for this sector list, and a model that
  would have called it from filings alone does not exist. Part is the model.
  Separating the two needs a longer window and a broader universe, which the
  curated 102-name list and the panel builder now make a scheduled job rather
  than a project.

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

## Phase 3, continued on 15 October

- **Justified multiple on names with negative filed ROIC.** TDS FY2025: EBITDA
  $206M but D&A $303M, so operating income is -$97M, ROIC -2.0% and EVA -$487M
  on $4.6B of invested capital. A justified multiple needs a positive return,
  so the block used to be withheld. It is now offered with those figures in the
  note and the ROIC field left for the analyst to underwrite (normalised EBIT,
  or the FCF bridge instead). EVA is reported wherever invested capital exists.
- **Pitch assessor.** One PNG per run for the deck or the Excel model: verdict,
  targets, distribution, probabilities with sampling error, driver quantiles,
  market-implied values, the FCF bridge and the three-statement percentile,
  with the histogram and CDF. The same figures are copied to the clipboard as a
  tab-separated block for pasting into a sheet.
- **Auto-fill price.** A price left in the field from the previous name was
  sent along with the lookup and echoed back as the new name's price. The quote
  now always wins, with a badge naming its source.

## Phase 4, expectations: done

- Market-implied margin, growth and multiple at spot (each holding the other
  two at P50), shown beside the analyst's P10 / P50 / P90 and flagged when
  outside them.
- The analyst's three-statement path as a percentile of the distribution.
- A calibration log endpoint and "Log this run" button recording every stated
  P10 / P50 / P90 for later scoring. Ephemeral on Vercel until
  `CALIBRATION_LOG_PATH` points somewhere durable.

## Phase 3 and Phase 5 on the harness

Queued on 15 October, 2012 onward, two-year horizon, default names, each
compared with v2 on the same snapshots: every mechanism switched on where
filings support it (`phase3_jm`, `phase3_cost`, `phase3_backlog`), the FCF
bridge scored as the price (`phase3_fcf`), and a Phase 5 prototype that carries
sector multiple momentum into the multiple centre (`p5_mom05`, `p5_mom10`),
built on per-year peer multiple medians the panels now record. The full
universe at one- and two-year horizons runs after them.

**Phase 3 results, 2012 onward, default names, paired against v2:**

| Variant | CRPS/spot delta | Better on | 90% coverage |
| --- | --- | --- | --- |
| Justified multiple on, as proposed | +0.085 (se 0.023) | 32% | 71% to 69% |
| Cost structure on, as proposed | +2.12 (se 0.42) | 35% | 71% to 77% |
| Backlog on, as proposed | +0.005 (se 0.002) | 24% | 71% to 70% |
| FCF bridge scored as the price | +0.119 (se 0.026) | 52% | 71% to 73% |

None of the mechanisms improves the forecast when switched on with the values
proposed from filings and no analyst view. That is the expected result for
backlog (its conversion rate is an assumption) and for the FCF bridge (a
different valuation, not a sharper one), and it is the point of the
mechanisms: they are where a thesis enters, not an automatic improvement. Two
were mis-defaulted and are being rerun: the cost model's fixed-cost growth
spread came from a noisy implied series (std 16% on MYRG, P5 to P95 at eleven
times spot) and is now shrunk and capped at 5%; the justified multiple's
premium reversion of 0.5 over two years is more than the market gives back and
is now 0.25. The UI defaults stay off; auto-fill proposes, the analyst decides.
With the tightened default the cost model lands at +0.113 (se 0.036), still
behind plain v2, with 90% coverage 71% to 74%.

**Phase 5 prototype, rejected.** Carrying sector multiple momentum (the
change in the peer median multiple over the prior year, from per-year peer
medians the panels now record) into the multiple centre: half the move,
CRPS/spot +0.015 (se 0.008), better on 38% of snapshots; the full move, +0.074
(se 0.016). Momentum in peer multiples does not predict the next two years
well enough to shift the centre; the 2022 to 2024 miss is not fixed this way.
A market-state input, if one exists, will have to work through width or the
rate level rather than through momentum in the centre, and that is the next
hypothesis to put through the harness, not a change to ship.

On the same 2012-onward default-name set, v2 against the own-history baseline:
588 paired snapshots, CRPS/spot a tie (delta -0.001, se 0.011), 90% coverage
52% to 71%, median CRPS/spot 0.43.

## Data corrections the harness surfaced

- Minority interest and preferred stock now come off enterprise value. TDS
  carries $1.07B of preferred and $232M of minority interest; net debt moved
  from -$1,507M to -$201M.
- Filers without the dei share-count tag fall back to us-gaap counts.
- Any multiple struck on a sub-2% margin is excluded from history and refused as
  an anchor (QuinStreet 2017: 330x), and the bridge itself is refused for names
  whose latest margin is under 2% (EverQuote).
- A peer panel can widen a company's own P10 to P90 by at most a factor of two;
  the ad-tech group's -7% to +46% growth range otherwise swamped QuinStreet.
- Margin reversion rates estimated from a thin panel are shrunk toward 0.38 and
  capped at 0.5 (the telecom panel estimated 0.72 from 47 pairs).

## Open

- **TDS remains a sum-of-parts story.** The bridge now takes preferred and
  minority interest off, but an EV/EBITDA bridge on a holding company with a
  tower business, a wireless stake and $2.2B of cash is the wrong instrument.
  The FCF bridge and the warning are the mitigation; a sum-of-parts is not in
  this tool.
- **Yahoo from Vercel** is now routed around: twelve years of daily closes for
  the universe ship in `montecarlo/data/prices`, refreshed weekly by a workflow,
  so the deployed lookup has multiple history, beta and a labelled last close
  without a network call. `FINNHUB_API_KEY` adds a live quote on top.
- **Panels are a snapshot.** `tools/build_panels.py` should run on a schedule
  with the ticker-map refresh.
- **Backlog conversion** is a hand-set 0.55 / 0.70 / 0.85 until the 10-K's
  "expected within twelve months" figure is parsed.
- **Scheduled refresh** now exists as GitHub Actions: tests on every push,
  prices, panels and the ticker map weekly (committed back), and a weekly
  backtest whose report is kept as an artifact. The ticker-map step needs a
  `SEC_USER_AGENT` repository secret.
