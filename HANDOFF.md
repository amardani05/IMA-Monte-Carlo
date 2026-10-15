# Handoff

State of the IMA Monte Carlo project as of 2026-10-15. For the model's
history and the harness verdicts read `ROADMAP.md`; for the method and the
reading list, the roadmap document ("IMA Monte Carlo: From Status Quo to
Prediction"). This file is what you need to pick the work up.

## What it is

A Monte Carlo scenario model for equity pitches. Five correlated drivers
(revenue CAGR, EBITDA margin, exit EV/EBITDA, net-debt change, dilution)
through an exit-multiple bridge to a terminal share price; probabilities of
reaching bear, base and bull targets. Live at https://ima-monte-carlo.vercel.app
(Vercel, Flask, Python; `app.py`). No model or inference service anywhere:
every number is filed with the SEC, a Yahoo or Finnhub close, arithmetic over
those, or engine output.

## Layout

| Path | What |
| --- | --- |
| `montecarlo/engine.py` | the bridge; metalog drivers, t-copula, the four mechanisms |
| `montecarlo/distributions.py` | metalog, log-metalog, triangular, normal samplers |
| `montecarlo/marketdata.py` | EDGAR extraction, point-in-time filtering, calibration v1 and v2, mechanisms |
| `montecarlo/panel.py` | SIC peer panels: quantiles, reversion rate, change correlations |
| `montecarlo/prices.py` | closes (bundle, yfinance, Yahoo chart), beta, cost of equity |
| `montecarlo/service.py` | validation, summary, reconciliation, price-implied, DCF percentile |
| `montecarlo/backtest.py`, `variants.py` | the harness and the model variants it scores |
| `montecarlo/data/` | `sp600.txt`, `peers.json`, `peer_overrides.json`, `panels/`, `prices/`, `sec_tickers.json` |
| `tools/` | `backtest.py`, `build_peers.py`, `build_panels.py`, `refresh_prices.py`, `refresh_tickers.py` |
| `public/` | the front end: `index.html`, `app.js`, `styles.css` |
| `tests/` | 99 tests, offline; `IMA_LIVE_SEC=1` adds live EDGAR |

## Run it

```bash
pip install -r requirements-dev.txt yfinance
python -m unittest discover -s tests -t .
python app.py                                   # or: flask --app app run
python tools/backtest.py --names MYRG,TDS,QNST --start 2017-12-31 --label smoke
python tools/backtest.py --model phase2 --label v2 --compare smoke --gate
```

Env: `FINNHUB_API_KEY` (set in Vercel; live quotes), `SEC_USER_AGENT`
("Name email@domain", needed only for `www.sec.gov`), `RISK_FREE`,
`EQUITY_RISK_PREMIUM`, `CALIBRATION_LOG_PATH`, `IMA_CACHE_DIR`.

## The rule

No modelling change ships unless `tools/backtest.py --gate` passes against
the current default on paired CRPS and 90% coverage. The harness has rejected
more than it has accepted; see "What the harness decided" in `ROADMAP.md`.
Production calibration defaults are `V2_DEFAULTS` in `marketdata.py`.

## Last S&P 600 verdict

`python tools/backtest.py --names sp600 --start 2012-12-31` then the same with
`--model phase2 --compare <base label> --gate`; about ten minutes each on a warm
EDGAR cache (the cache expires after 24 hours; a cold run downloads about
180 MB). `--names all` is the S&P 600 plus about 90 large caps from the
curated peer pool; `sp600` is the index alone. Latest run files are
`backtests/sp600fix3_*` (git-ignored). The gate **fails** on mean CRPS, and
VCEL's four snapshots are the whole of it; see the S&P 600 entry under "What
the harness decided" in `ROADMAP.md`. Defaults unchanged. The machine's disk
runs within a few hundred MB of full; a full disk stops every command,
including the tool's own output.

## Pitch assessor and added drivers

The assessor is a results tab (top bar button or the tab beside Analysis),
drawn as one 1600 x 1200 SVG and saved as a PNG with the IBM Plex fonts
inlined. Figures on the left; the terminal price probability (per-bin share
coloured by scenario band, cumulative on the right axis, FCF bridge overlaid,
axis cut at P98 with the share beyond stated) and the driver sensitivity on
the right; the price history with the horizon densities along the bottom.

- **Added drivers.** Backlog, FCF, ROIC and EVA, and cost structure are
  driver groups added beside the base five, each uncertain input a P10 / P50 /
  P90. `service.propose_extra_drivers` fills them from `marketdata.mechanisms`
  and offers every group even when the filings cannot support it, leaving the
  missing inputs for the analyst (TDS's ROIC, QuinStreet's backlog). ROIC,
  WACC and growth fade accept a P50 alone (a point input) or a full triple (a
  drawn driver, independent of the copula); the proposed widths are stated
  (ROIC 0.75x to 1.25x the filed figure, WACC one point either side, fade
  0.25 / 0.50 / 0.75). EVA at P50 is (ROIC - WACC) x filed invested capital.
- **Driver sensitivity** (`engine.swing_tornado`): each driver held at its own
  P10, then its P90, on the same paths with every other draw unchanged, for
  the median and P(at least base), per bridge. `engine.tornado` (conditional
  deciles) is kept for the CLI. The draw and bridge split that makes this
  exact reproduces earlier runs bit for bit.
- **Horizon densities** (`service.horizon_densities`): smoothed per-dollar
  densities at quarters of the horizon on one price grid, Gaussian KDE in log
  space. The intermediate horizons are interpolated from the terminal run
  (log x_u = log spot + u log(median / spot) + sqrt(u)(log x_T - log median))
  and labelled so; swap in real runs once the engine simulates those dates.

## Open

- Two data fixes landed after the last S&P 600 run and are not in its
  numbers: goodwill and intangible impairments are added back to EBITDA
  (RAL's FY2025 margin was -51.6% from a $1.44B write-down), and short-term
  investments count against debt (Cal-Maine's EV was $3.3B against $2.5B).
  Rerun the chain to record what they do to CRPS and coverage. The historical
  multiple's EV proxy still uses cash alone.
- v2 carries extreme margin-history years straight into its margin metalog
  (VCEL, -1,416x from pre-launch revenue); the baseline trims them. A guard
  there is the likeliest way to pass the S&P 600 gate.
- Dual-class share counts are understated: the SEC's company facts drop the
  per-class cover counts, so SHAK, AMRX and MIR bridge on Class A alone and
  CXM on a stale count. LMAT's cover count is off by ten. Fixing it means
  reading the filing's own XBRL instance.
- The 2022 to 2024 miss (90% coverage about 50% on those snapshot years) is
  unexplained by anything tried: mean reversion, peer widths, multiple
  momentum. Next hypothesis: a market-state input through width or the rate
  level. It goes through the harness before the app.
- TDS is a sum-of-parts story the bridge cannot represent; preferred and
  minority interest are taken off, the FCF bridge is offered, and the ROIC
  field is left for the analyst because the filed ROIC is negative.
- Backlog conversion is a stated assumption (0.55/0.70/0.85) until the 10-K's
  twelve-month figure is parsed. At horizons of a year or more, more
  conversion gives a lower price whenever the CAGR is positive: year-one
  revenue is (1 + g) R - g x backlog x conversion, so backlog-covered revenue
  does not grow. MYRG on the 2-year run: median $371 at P10 conversion, $360
  at P90. The sensitivity chart shows it; the maths is unchanged here and a
  change goes through the harness.
- The calibration log is ephemeral on Vercel until `CALIBRATION_LOG_PATH`
  points at a durable store.
- Weekly GitHub Actions refresh prices, panels and the ticker map and run a
  backtest; the ticker-map step needs a `SEC_USER_AGENT` repository secret.

## People

Amar Dani (amardani05) owns it, pitches TDS and QuinStreet with it, and
decides what ships. Preferences that matter: no em dashes anywhere in the UI,
light institutional styling, mechanisms offered rather than withheld even when
the filed number is ugly, and the universe is the S&P 600, not a curated list.
