# IMA Monte Carlo

Monte Carlo scenario analysis for equity pitches. Simulates five correlated
value drivers through a revenue → EBITDA → EV → equity-per-share bridge, and
reports the probability of landing at or beyond your bear / base / bull targets.

Type a ticker and the assumptions fill themselves in from SEC filings. Set the
three case targets and the driver modes, and run.

| Surface | Entry point | Output |
| --- | --- | --- |
| Web app | `public/` + `app.py` | Ticker auto-fill, interactive form, live SVG charts, JSON export |
| Local CLI | `python MonteCarlo.py` | Console summary + 4-panel PNG in `results/` |

Both run the **same engine** (`montecarlo/engine.py`).

## No model in the loop

Nothing in this project calls an LLM or any inference service. Every figure the
app shows is either a number the company filed with the SEC, plain arithmetic
over those numbers, or output from the simulation engine, and each auto-filled
field carries its source and as-of date so it can be checked against the filing.
The dependency list is `flask`, `numpy`, `scipy`.

## Model

Drivers are sampled through a **Gaussian copula** so they move together rather
than independently, a good revenue year tends to come with better margins and
a higher multiple.

| Driver | Distribution |
| --- | --- |
| Revenue CAGR | triangular (low, mode, high) |
| EBITDA margin | triangular (low, mode, high) |
| EV / EBITDA multiple | triangular (low, mode, high) |
| Net debt change (% of EV) | normal (mean, std) |
| Share dilution | normal (mean, std), clipped to −10% / +15% |

Bridge, per path:

```
terminal_revenue = current_revenue × (1 + rev_cagr) ^ horizon_years
terminal_ebitda  = terminal_revenue × ebitda_margin
terminal_ev      = terminal_ebitda × ev_ebitda_multiple
terminal_equity  = terminal_ev − (current_net_debt + current_ev × net_debt_chg)
terminal_price   = terminal_equity / (shares × (1 + dilution)),  floored at 0
```

Correlations use a built-in cross-sector matrix. The API accepts a custom 5×5
matrix and validates it is symmetric and positive definite before use.

## Auto-fill

`GET /api/company/<ticker>` reads SEC EDGAR's XBRL company-facts API and returns
a ready-to-run set of assumptions, each driver as a P10, P50 and P90.

**Pulled from filings:** revenue, EBITDA margin (operating income + D&A, with
fallbacks for filers that tag neither), share count, cash, debt, minority
interest and preferred stock. Year-end EV/EBITDA history comes from Yahoo
Finance closes against the balance sheet.

**Calibrated from filed history and a peer panel (calibration v2).** The
company's own quantiles are shrunk toward its SIC peer group
(`montecarlo/data/peers.json`, panels cached by `tools/build_panels.py`); the
margin P50 follows a partial-adjustment path toward the company's own long-run
median at a rate estimated from the panel and shrunk toward Fama-French's 0.38;
the copula is Student-t with correlations estimated from pooled year-over-year
changes. The defaults were chosen on the backtest harness, where the peer panel
turned out to earn its keep through width rather than by moving the centre.

**Mechanisms** (`mechanisms` in the response, switched on per name in the UI):
cost structure from CostOfRevenue and the implied fixed base, backlog from the
remaining-performance-obligation tag, an FCF-to-equity second bridge, and a
justified multiple from ROIC, a CAPM cost of equity and a stated WACC.

**Left to you:** bear, base and bull targets, and the P50 of any driver you
hold a view on. The results page shows what spot already implies for each
driver, and where your three-statement path sits in the distribution.

### Backtest harness

```bash
python tools/backtest.py                                # default names, baseline
python tools/backtest.py --model phase2 --label v2 --compare baseline
python tools/backtest.py --names MYRG,TDS,QNST --start 2017-12-31 --model v2_tight
```

For each name and past quarter-end the auto-fill is rebuilt from filings
available on that date and the close that day, the model runs at the horizon,
and the realised price is scored: PIT (uniform if the distributions are right)
and CRPS in dollars. `montecarlo/variants.py` registers model variants; every
modelling change is accepted or rejected on these numbers. `--gate` exits
non-zero when a run fails the paired CRPS and coverage test against
`--compare`, and each run writes `backtests/<label>.html` with the PIT
histogram and per-name and per-year tables. `--names all` runs the whole
universe. EDGAR documents are cached on disk under `~/.cache/ima-monte-carlo`.
Needs `yfinance` locally.

### Bundled prices

`montecarlo/data/prices/` holds twelve years of daily closes for the universe
and SPY, written by `tools/refresh_prices.py`. The deployed function reads
these first, so it has multiple history and beta without a network call to
Yahoo, which rate-limits Vercel's egress. Refresh weekly and commit; a series
older than seven days is used only as a last resort and labelled with its date.

### Not every filer works

Banks, brokers and insurers are rejected outright: they book interest as
revenue, so EBITDA is not meaningful and the bridge would return confident
nonsense. REITs warn. Filers using custom taxonomies for revenue (Exxon, for
one) cannot be read generically and fail with a clear message.

### Configuration

| Variable | Effect |
| --- | --- |
| `FINNHUB_API_KEY` | Quote provider. Without it Yahoo Finance is tried once; on a 429 the price is typed by hand and everything else still fills. |
| `SEC_USER_AGENT` | `"Name email@domain"`. Only needed for tickers missing from the bundled map. |
| `RISK_FREE`, `EQUITY_RISK_PREMIUM` | CAPM inputs for the cost of equity (defaults 4.2% and 4.5%). |
| `CALIBRATION_LOG_PATH` | Where logged runs are appended. Ephemeral on Vercel unless pointed at a mounted store. |

## Tests

```bash
python -m unittest discover -s tests -t .
```

80 tests, no network needed. SEC parsing, calibration v2 and the harness math
run against synthetic fixtures; golden-value regressions pin the published MYRG
numbers. Set `IMA_LIVE_SEC=1` to additionally exercise EDGAR for real.

## Local development

```bash
pip install -r requirements-dev.txt
python MonteCarlo.py          # edit the assumptions at the bottom of the file
```

To exercise the web app locally, run any static server over `public/` alongside
a process that serves `POST /api/simulate`, or just use `vercel dev`:

```bash
vercel dev
```

## Deployment

Deployed on Vercel as a static front end plus one Python serverless function.

- `app.py`, Flask entrypoint; serves `public/` and the two API routes
- `montecarlo/`, engine, request layer, and the SEC data layer
- `requirements.txt`, flask + numpy + scipy (**no matplotlib**; charts are SVG)

The function never returns the raw price vector. It sends a 100-bin histogram,
a 400-point CDF, tornado bounds and summary stats, roughly 13 KB per run.

```bash
vercel --prod
```

## Known gaps

See [ROADMAP.md](ROADMAP.md). The build-out is complete; what remains needs a
credential or a decision from the operator.

## Disclaimer

Model output, not investment advice. Results are only as good as the
assumptions you feed in.
