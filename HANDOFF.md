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

## Open

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
  twelve-month figure is parsed.
- The calibration log is ephemeral on Vercel until `CALIBRATION_LOG_PATH`
  points at a durable store.
- Weekly GitHub Actions refresh prices, panels and the ticker map and run a
  backtest; the ticker-map step needs a `SEC_USER_AGENT` repository secret.

## People

Amar Dani (amardani05) owns it, pitches TDS and QuinStreet with it, and
decides what ships. Preferences that matter: no em dashes anywhere in the UI,
light institutional styling, mechanisms offered rather than withheld even when
the filed number is ugly, and the universe is the S&P 600, not a curated list.
