# Roadmap

Status as of 2026-09-02. Live at **https://ima-monte-carlo.vercel.app**

The engine works and is now deployed. The gaps that matter are not in the
plumbing — they are in whether the numbers it produces will survive a committee
question. Phase 1 is the blocker for real use; everything after it is leverage.

---

## Phase 0 — Shipped

- Engine extracted to `montecarlo/engine.py`, shared by the CLI and the web app
- `POST /api/simulate` with payload validation (distribution ordering, positive
  std, symmetric + positive-definite correlation matrix, 250k path cap)
- Browser front end: assumption form, stat cards, four SVG panels, sensitivity
  tables, JSON export
- 100k paths in ~180 ms server-side, ~350 ms round trip, ~12 KB response
- Horizon is now a parameter instead of a hardcoded `** 2`

---

## Phase 1 — Make the numbers defensible

**This is the blocker.** Running the shipped MYRG assumptions surfaces three
problems that would be asked about in the first two minutes of a pitch.

### 1.1 Reconcile the simulation against the case targets — *highest priority*

| | |
| --- | --- |
| Base target (from DCF/comps) | **$323.01** |
| Simulated median | **$298.90** (−7.5%) |
| P(≥ base) | **17.9%** |

The simulation says the base case is a roughly 1-in-6 outcome. That is a
coherent thing to believe, but right now it is unintentional: the DCF and the
Monte Carlo bridge are two different models that were never reconciled. Either

- **calibrate** — back-solve the driver assumptions so the median lands on base,
  and report the implied drivers as the pitch's actual assumption set, or
- **decide the divergence is the finding** — and say so explicitly, with the
  DCF-vs-simulation delta shown as a first-class output.

Ship a reconciliation panel that shows the gap and its driver attribution
either way. Silently presenting both numbers is the thing to avoid.

### 1.2 The distributions encode directional bets as certainties

Measured over 200k paths on the shipped assumptions:

| Driver | Current | Sampled range | Paths on the "no change" side |
| --- | --- | --- | --- |
| EBITDA margin | 6.1% | 7.10% – 7.70% | **0.00%** |
| EV/EBITDA multiple | 20.1x | 12.4x – 18.5x | **100% below current** |
| Net debt change | — | mean +0.00% of EV | no FCF deleveraging modelled |

Every path assumes 100–160 bps of margin expansion **and** multiple compression.
Those may both be right, but they are assumptions, not uncertainty — and folding
them into the "distribution" hides them. Consequences downstream:

- P90/P10 terminal price ratio is **1.29x**. For a two-year equity forecast that
  is implausibly tight.
- P(below bear) is **0.2%** — the model claims near-certainty of clearing the
  bear case.

**Work:** calibrate triangular ranges to realized historical dispersion
(revenue growth, margin volatility, multiple range through a cycle) rather than
to the analyst's point-estimate band. Separate *directional view* (the mode)
from *uncertainty* (the width) in the UI so the two are argued independently.

### 1.3 Precision is currently overstating accuracy

Monte Carlo sampling error is negligible and not the problem:

| | |
| --- | --- |
| SE on P(≥base) @ 100k paths | 0.12 pp (95% CI ±0.24 pp) |
| Median across 8 seeds | $298.67 – $299.14 (spread $0.48) |

So the tool reports `17.8%` to a tenth of a point while the real uncertainty —
assumption uncertainty — is orders of magnitude larger. **Work:** report the MC
standard error alongside each probability, and add an assumption-uncertainty
band (re-run across a grid of plausible driver ranges) so the headline number
carries an honest error bar. Round displayed probabilities accordingly.

### 1.4 Test suite

There are currently **no tests** on code whose output goes into investment
pitches. Minimum bar:

- Golden-value regression on the MYRG assumptions (catches silent numeric drift)
- Sampler property tests — triangular inverse-CDF recovers its analytic mean and
  mode; Gaussian copula recovers the target rank correlations within tolerance
- Validation tests for every `ValidationError` branch
- Bridge arithmetic checked against a hand-computed single path

### 1.5 Model mechanics worth revisiting

- **Net debt** moves as a zero-mean shock scaled by *EV*. A cash-generative
  business should deleverage mechanically over the horizon; tie the change to
  modelled FCF instead of an EV-scaled random walk.
- **No discounting.** `expected_return` compares a two-year terminal price to
  today's price — it is a cumulative, undiscounted, non-annualised return.
  Label it precisely, and add annualised and PV variants.
- **`np.maximum(price, 0)`** truncates the left tail rather than modelling
  distress. It never binds on the current assumptions (minimum simulated price
  is $204), so this is a latent issue that appears only once the distributions
  are widened in 1.2 — but then it matters.
- **Mean vs median.** The distribution is *left*-skewed (skew −0.31), so the
  mean ($296.15) sits below the median ($298.87) and `expected_return`, which is
  computed off the mean, slightly understates the central outcome. Lead with the
  median — the UI already does — and label the mean-based figure as such.
- **Correlation matrix is invented.** The "default cross-sector assumptions" are
  not estimated from anything. Estimate them, or run the output's sensitivity to
  the correlation assumption and show it.

---

## Phase 2 — Make it a tool rather than a demo

- **Save and share a scenario.** Today every run is ephemeral. Start with
  URL-encoded state (no backend needed), then a small store for named pitches.
- **Edit the correlation matrix in the UI.** The API already validates symmetry
  and positive-definiteness; the front end just needs a 5×5 grid with live
  feedback on why a matrix was rejected.
- **Export charts for decks.** JSON export exists; analysts need the PNG/PDF.
  The matplotlib panel already exists in `MonteCarlo.py` — either expose it
  behind an endpoint or render SVG → PNG client-side.
- **Compare scenarios side by side** — two pitches, or one pitch under two
  assumption sets, on shared axes. This is what makes the distribution argument
  legible to a committee.
- **Preset library** per sector, so driver ranges start from something
  defensible instead of blank fields.

---

## Phase 3 — Stop typing financials by hand

Current price, shares outstanding, net debt and EV are all manual entry and go
stale the day they are typed.

- Pull quote + share count + balance-sheet items from a market data provider,
  keyed on ticker
- Show the as-of date on every fetched figure
- Fall back cleanly to manual entry when a ticker is unavailable
- Derive `current_ebitda_margin` and `current_ev` rather than asking for them,
  with an override

---

## Phase 4 — Team readiness

- **Git-connected deploys.** The Vercel↔GitHub link failed during setup, so
  production currently ships via `vercel --prod` from a laptop. Install the
  Vercel GitHub app on `amardani05/IMA-Monte-Carlo` so pushes to `main` deploy.
- **Access control.** The production URL is public. Nothing is persisted
  server-side today, so the exposure is the tool rather than the data — but that
  changes the moment Phase 2 saves scenarios. Decide before then.
- **Error monitoring** on the function, and analytics on which drivers people
  actually edit.
- **Housekeeping:** `results/*.png` are build artifacts committed to git;
  `scipy` is pulled in for `norm.cdf`/`norm.ppf` alone and could be replaced
  with a numpy `erf` implementation to cut bundle size and cold-start time.
- **Accessibility:** the SVG panels have no text alternative. The sensitivity
  tables partly cover this; the four charts do not.

---

## Suggested order

1. **1.4 tests** — cheap, and everything else is safer once they exist
2. **1.1 reconciliation** — the credibility blocker
3. **1.2 distribution calibration** — the accuracy blocker
4. **2.1 save/share + 2.3 chart export** — what makes it get used
5. **1.3 error bars**, then Phase 3 and 4 as the tool gets real users
