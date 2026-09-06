# Roadmap

Status as of 2026-09-06. Live at **https://ima-monte-carlo.vercel.app**

The build-out is complete. What remains needs a decision or a credential from
the operator rather than code, and is listed at the end.

**Nothing in this project calls a model or inference service.** Every figure is
a number filed with the SEC, arithmetic over those numbers, or output from the
simulation engine. Each auto-filled field carries its source and as-of date.

---

## Why terminal prices sit near spot, and what to do about it

This came up on TDS and QuinStreet and it is worth understanding rather than
tuning away, because the behaviour is mostly correct.

**1. The mode is held at today's value.** Auto-fill puts the historical width
around margin and multiple but leaves the mode at the current figure. The median
path is therefore "revenue grows at its historical median; margin and multiple
stay put." That is the right null hypothesis. It is not a forecast, and a pitch
has a thesis. The thesis enters through the mode. The chips under each driver
now make that a one-click choice between the current value, the historical
median, or a typed view, and the label on the middle input says whose number it is.

**2. Triangular means sit below their modes when the range is left-skewed.**
Ten filed years almost always carry more downside than upside: loss years,
impairments, multiple compression in drawdowns. So even with every mode at
current, the mean of each driver is below current and the simulated median lands
below the mode path. The reconciliation panel shows both numbers together. For
QuinStreet the mode path is $20.66 and the median $16.32; the 21% gap is skew,
not a bug.

**3. Single events were setting the tails.** TDS's 2023 impairment put a -34%
margin into the range and the UScellular sale put a -53% two-year CAGR there,
each dominating the triangular mean on its own. Bounds now trim one extreme per
side once eight observations exist. The chips still show the full filed range so
nothing is hidden.

**4. The multiple band was biased.** The old 0.70x to 1.15x anchor had a mean 5%
below current, a systematic drag on every run. It is symmetric now, and where
six or more years of filed public float exist the width comes from the company's
own multiple history, bounded to half and one-and-a-half times today.

What "more accurate" means here: a Monte Carlo turns assumptions into a
distribution. It cannot supply the view. With status-quo assumptions the honest
answer is "status quo plus noise", and that answer near spot is correct.
Accuracy comes from putting the thesis in the mode explicitly and keeping the
width honest. The reconciliation table then states what each target implicitly
assumes, which is exactly the thing that has to be defended out loud.

**TDS specifically.** Net cash is 80% of market cap after the UScellular sale.
An EV/EBITDA bridge values the operating business and carries the cash across
unchanged, so what management does with $1.5B is most of the equity story and
the model cannot see it. Run it, but read the warning it now raises and put a
sum-of-parts beside it.

---

## Shipped

### Engine and API
- Engine in `montecarlo/engine.py`, shared by the CLI and the web app
- `POST /api/simulate` with full payload validation; 100k paths in ~180 ms
- Response carries a histogram, a 400-point CDF, tornado bounds, summary
  statistics, Monte Carlo standard errors on every probability, annualised
  median and mean returns, and a reconciliation block. Never the raw price vector.
- Horizon is a parameter, not a hardcoded exponent

### Auto-fill from SEC EDGAR
- `GET /api/company/<ticker>`: revenue, EBITDA margin, share count, cash and debt
  from XBRL company facts, with provenance on every field
- Ticker to CIK resolves against a bundled 10,391-entry map; no call to
  `www.sec.gov` on the hot path. `tools/refresh_tickers.py` regenerates it.
- Candidate XBRL tags are merged, not first-hit, so filers that changed tagging
  keep their early years
- TTM rolls four quarters, reconstructing the Q4 most filers leave untagged;
  revenue and margin are then taken from the same fiscal year so their product
  is an EBITDA the company actually filed
- EBITDA falls back from operating income to gross profit less opex to pre-tax
  plus interest, which is what makes Dycom resolve
- Banks, brokers and insurers rejected by SIC; REITs warned; implausible margins
  warned; net cash above half of market cap warned

### Calibration from filed history
- Revenue CAGR width from every realised N-year CAGR in the record
- Margin width from filed years; multiple width from public-float history where
  six or more usable years exist; net-debt drift from filed free cash flow
- One extreme trimmed per side at eight or more observations
- Every driver returns current value, historical median and raw range; the UI
  renders them as chips that set the mode

### Front end
- Light institutional theme: cool off-white ground, deep navy accent, IBM Plex
  Sans and Mono, tabular numerics. No em dashes anywhere in the UI.
- Reconciliation panel: mode-path price against the simulated median, and for
  each case target the multiple needed at the mode margin and the margin needed
  at the mode multiple, flagged red when outside the range given
- Share links: full payload including the correlation matrix encoded in the URL
  fragment, restored with an auto-run on load
- Correlation matrix editor, symmetric by construction, positive-definiteness
  checked server-side
- PNG export composites the four panels for decks; JSON export carries inputs
  and results together
- Pin-and-compare strip with deltas against a pinned run
- aria-labels on every chart describing the numbers they show

### Tests
- 54 tests, offline by default; SEC parsing runs against a synthetic fixture
- Golden-value regressions pin the published MYRG numbers
- Reconciliation tests prove each implied driver reproduces its target exactly
- `IMA_LIVE_SEC=1` additionally exercises EDGAR

---

## Needs the operator

| Item | Why it needs you |
| --- | --- |
| `FINNHUB_API_KEY` | Sign up for the free tier and set it in Vercel. Removes the one remaining manual field. |
| `SEC_USER_AGENT` | Set to `"Your Name you@example.com"`. Only needed for tickers missing from the bundled map, but SEC asks for it. |
| Push to GitHub | Every commit is local. The repository remote exists and nothing has been pushed. |
| Vercel GitHub app | Install on `amardani05/IMA-Monte-Carlo` so pushes to `main` deploy. The CLI link failed. |
| Access control | The production URL is public. Nothing is persisted server-side, so the exposure is the tool, not the data. Decide before anything saves state. |

## Deliberately not done

- **SciPy removal.** It is used for `norm.cdf` and `norm.ppf` only and adds
  ~40 MB to the bundle. Replacing it with an `erf` approximation changes the
  numerical core at the 1e-7 level for a cold-start gain. Not worth the risk for
  a tool whose value is in its numbers agreeing with themselves.
- **Estimating the correlation matrix.** It is editable now. Estimating it from
  data needs return series the tool deliberately does not ingest.
- **Present-value discounting.** Returns are annualised, which is the honest
  comparable. A discounted variant needs a cost-of-equity input, which is a
  DCF concern rather than a scenario-analysis one.
- **Exxon and other custom-taxonomy filers.** Need per-filer tag overrides.
  Fails with a clear message.
- **Sector presets.** Superseded by auto-fill.
- **Scheduled ticker-map refresh.** A GitHub Action, once the repository is pushed.

## Latent

- The zero floor on terminal price never binds on current assumptions but will
  once distributions are widened further.
- Free-cash-flow drift is capped at 60% of EV and 15% standard deviation. On a
  cash-dominated EV like TDS the cap binds and the note says so.
