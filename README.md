# IMA Monte Carlo

Monte Carlo scenario analysis for equity pitches. Simulates five correlated
value drivers through a revenue → EBITDA → EV → equity-per-share bridge, and
reports the probability of landing at or beyond your bear / base / bull targets.

Two front ends run the **same engine** (`montecarlo/engine.py`):

| Surface | Entry point | Output |
| --- | --- | --- |
| Web app | `public/` + `api/simulate.py` | Interactive form, live SVG charts, JSON export |
| Local CLI | `python MonteCarlo.py` | Console summary + 4-panel PNG in `results/` |

## Model

Drivers are sampled through a **Gaussian copula** so they move together rather
than independently — a good revenue year tends to come with better margins and
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

## Local development

```bash
pip install -r requirements-dev.txt
python MonteCarlo.py          # edit the assumptions at the bottom of the file
```

To exercise the web app locally, run any static server over `public/` alongside
a process that serves `POST /api/simulate` — or just use `vercel dev`:

```bash
vercel dev
```

## Deployment

Deployed on Vercel as a static front end plus one Python serverless function.

- `public/` — static assets (`outputDirectory` in `vercel.json`)
- `api/simulate.py` — `POST /api/simulate`, runs the engine, returns aggregates
- `requirements.txt` — function deps only (numpy + scipy; **no matplotlib**)

The function never returns the raw price vector. It sends a 100-bin histogram,
a 400-point CDF, tornado bounds and summary stats — roughly 13 KB per run.

```bash
vercel --prod
```

## Known gaps

See [ROADMAP.md](ROADMAP.md). The headline items: no persistence, no test
suite, no market-data ingestion, and no way to edit the correlation matrix from
the UI.

## Disclaimer

Model output, not investment advice. Results are only as good as the
assumptions you feed in.
