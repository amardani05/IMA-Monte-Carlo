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

**S&P 600, 12,639 paired snapshots across 429 names, 2013 to 2024, two-year
horizon (15 October; `--names sp600`)**. The first run of this chain read
`gate: FAIL` with a delta of NaN, a verdict about the data, not the model. Five
filer tagging patterns had to be handled before the gate meant anything, each
of which had put a median price somewhere between $5,900 and $100M a share:

- a share count of zero (dual-class filers tag the undimensioned total as 0;
  SHAK, CXM, ROAD, FOXF), which divided equity by zero in 48 snapshots;
- a placeholder or stale count (TRIP's 1,000 pre-spin-off shares from 2011,
  AMRX's 1,000 holding-company shares);
- a count tagged in thousands as shares, or 1,000 times too large (HUBG, ANIP,
  PAHC, SEI; CPK, SKY, STAA), now caught against net income over diluted EPS;
- a revenue sub-line under `Revenues` shadowing the `SalesRevenueNet` total
  (ANDE's $0.9M beside $5.6B, PRGO, MWA, NPO), which made a 161x margin.

| | Own history (baseline) | v2 adopted |
| --- | --- | --- |
| CRPS / spot, mean | 0.933 | 1.512 |
| CRPS / spot, median | 0.330 | 0.312 |
| Better on snapshots | | 54% |
| 50% / 90% coverage | 27% / 56% | 42% / 80% |
| PIT in outer deciles | 53% | 31% |

**Gate: FAIL** on the mean (delta +0.578, standard error 0.307); coverage
passes. VCEL's four 2021 snapshots contribute +0.590 of the delta: its filed
margin history includes the near-zero revenue years before its products
launched (-1,416x), the baseline's robust bounds trim that year, and v2 carries
it straight into a margin metalog with a P10 of -1,416 and a P95 price of
$4.4M. Without VCEL the delta is -0.012 (standard error 0.074), a tie again;
the median paired delta is -0.010 and the 5% trimmed delta -0.032. The
defaults are unchanged. The fix the numbers point to is a margin-history guard
in v2 of the kind v1 has, and it goes through the harness like anything else.
2022 to 2024 coverage is 79%, 75% and 74% under v2 against 58%, 58% and 56%:
better, still short.

**Horizon term structure (15 October; S&P 600, 2012 onward, paired).** Under
two years the price distribution did not open out from spot. MYRG at $291.17
(v2, backlog on, seed 42): six months and one year gave the same distribution
(P5 198.7, median 308.2, P95 507.0), and at 0.05 years P5 to P95 was still
$308 wide. Three causes:

- Backlog booked a full year of revenue at any horizon under one. A part year
  now books that share of year one, which is what trailing revenue at T is.
  A bug fix, on; no one- or two-year number moves.
- Margins and exit multiples are levels, and both calibrations give them a
  width with no horizon in it (v2's `width_scale` is 1.0), so their full width
  applies at any T.
- Dilution is calibrated as one year's share-count change (MYRG: -1.6%, std
  6.0%) and applied as the change over the whole horizon. Net debt is scaled
  (drift x T, std x sqrt T); dilution is not.

The engine now has `level_width_horizon` H: under H, each margin and multiple
draw is pulled toward its median by sqrt(T / H). Medians do not move, and at or
beyond H prices are bit-identical. Dilution over the horizon was scored as a
calibration variant (mean x T, std x sqrt T), not changed in the engine, since
the service's mode path and implied values read the same field.

| Change | Horizon | CRPS/spot delta (se) | Median delta | 90% coverage | Gate |
| --- | --- | --- | --- | --- | --- |
| Level widths full at two years (H = 2) | 1 year | -0.184 (0.051) | -0.004 | 85% to 77% | FAIL |
| Dilution stated over the horizon | 2 years | -0.016 (0.003) | -0.001 | 79% to 79% | PASS |
| Level widths full at one year (H = 1) | 0.5 year | -0.020 (0.002) | -0.011 | 90% to 84% | FAIL |
| H = 1 and dilution over the horizon | 0.5 year | -0.020 (0.002) | -0.012 | 90% to 84% | FAIL |

What it says: **v2's stated widths are one-year widths.** At one year v2 is
nearly calibrated (50 / 80 / 90% coverage 51% / 77% / 85%, PIT tails 23%), at
two years too narrow (42% / 69% / 79%, 31%). Pulling the one-year distribution
in toward a two-year reference costs eight points of 90% coverage, and some in
every snapshot year; its mean CRPS gain is a few extreme-width names (TGTX,
SUPN), and the median paired delta is -0.004. At half a year the stated widths are
too wide in the body (50% coverage 62%, PIT tails 15%: a hump) and on target
in the 90% band. H = 1 sharpens it on two snapshots in three (ten standard
errors on the mean, median delta -0.011) and puts the 50% band on target, but
root-T narrows the tails as much as the body, and 90% coverage falls six
points: a fail on coverage. Dilution over the horizon makes no difference at
half a year.

**Defaults unchanged.** `level_width_horizon` stays off; the backlog fix is
on. Dilution over the horizon, the one change that passed, belongs in
`calibrate_drivers` (the variant is `phase2_dilT`) and goes in there next. For
the short end the next hypothesis is a width that narrows the body but keeps
the tails, which root-T scaling of a level distribution cannot do; an H
between half a year and one year would only trade the CRPS gain against the
coverage tolerance. Run files: `backtests/h05_*`, `h1_*`, `h2_*`. Every run
in this entry reads one data definition, main as of c48ae3e (impairments added
back to EBITDA, short-term investments in net debt), and none is paired
against the earlier `sp600fix3_*` files. A first one-year pair launched before
c48ae3e is kept in `backtests/pre_c48ae3e/`; on its own, c48ae3e moves v2's
one-year CRPS/spot by -0.006 (se 0.002) and touches 11% of snapshots.

The chain as in the handoff, rerun on today's data: v2 against the own-history
baseline at one year (14,455 snapshots) fails on the mean (+0.360, se 0.170;
TGTX's 2024 snapshots alone are +1,000 to +1,500 each) and wins on the median
(-0.009) and the 5% trimmed mean (-0.037), with 90% coverage 63% to 85%. At
two years it fails the same way (+0.539, se 0.282; VCEL), median -0.012,
coverage 55% to 79%. COLL's 2022 snapshots are the same margin-history
pathology as VCEL: median $0 and P95 $100,000 to $154,000 on a $16 to $23
stock.

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

## The universe is now the S&P 600

Until 15 October the "universe" was a hand-curated list of 102 tickers in
`tools/build_peers.py`, chosen so MYRG, TDS and QuinStreet had peer groups,
plus the harness names. It served both as the pool the SIC peer panels are
drawn from and as the backtest set, and it was an arbitrary sample. It is now
the S&P SmallCap 600, the index these names sit in: `montecarlo/data/sp600.txt`
holds the constituents (603, from the public list, with GICS sector), every
member resolves to a CIK and SIC code from EDGAR, bundled prices cover all of
them (weekly closes plus every quarter-end close and the last sixty days, about
9 MB), and the backtest runs over the whole index at a two-year horizon from
2012. Exact-SIC peer groups become possible for most members instead of the
two-digit fallback; the hand-picked comparables for the three names in use
stay in force. The curated names outside the index (Quanta, Dycom, Apple and
the rest) remain in `peers.json` as comparables but are no longer the universe.

## Pitch assessor and the path chart

The results page now opens with the price history, the forecast fan (P5 to P95
and P25 to P75 from spot to the horizon) and the terminal distribution drawn
vertically at the horizon, the way a volume profile sits beside a price chart,
with the targets and median across it. The pitch assessor image carries that
chart, the scenario probability bars and the histogram beside the figures, with
no calibration or copula lines and no run subtitle.

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
