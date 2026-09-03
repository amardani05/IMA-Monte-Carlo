# IMA Monte Carlo

Monte Carlo scenario analysis for equity pitches. Simulates five correlated
value drivers through a revenue → EBITDA → EV → equity-per-share bridge, and
reports the probability of landing at or beyond your bear / base / bull targets.

Type a ticker and the assumptions fill themselves in from SEC filings.

| Surface | Entry point | Output |
| --- | --- | --- |
| Web app | `public/` + `app.py` | Ticker auto-fill, interactive form, live SVG charts, JSON export |
| Local CLI | `python MonteCarlo.py` | Console summary + 4-panel PNG in `results/` |

Both run the **same engine** (`montecarlo/engine.py`).

## No model in the loop

Nothing in this project calls an LLM or any inference service. Every figure the
app shows is either a number the company filed with the SEC, plain arithmetic
over those numbers, or output from the simulation engine — and each auto-filled
field carries its source and as-of date so it can be checked against the filing.
The dependency list is `flask`, `numpy`, `scipy`.

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

## Auto-fill

`GET /api/company/<ticker>` reads SEC EDGAR's XBRL company-facts API and returns
a ready-to-run set of assumptions.

**Pulled from filings:** revenue, EBITDA margin (operating income + D&A, with
fallbacks for filers that tag neither), share count, cash and debt.

**Calibrated from filed history:** revenue CAGR from every realised N-year CAGR
in the record, and the EBITDA margin range from the years actually filed. This
matters — hand-set ranges tend to encode a directional view as though it were
uncertainty. On MYRG the hand-typed 7.1–7.7% margin band sits entirely above a
company that has ranged 3.55–6.82% over ten filed years.

**Left to you:** bear, base and bull targets are your DCF output — the thing the
simulation exists to test — and the exit multiple is your comps view, so it is
anchored on the current multiple and flagged as uncalibrated.

Ticker → CIK resolves against a map bundled at `montecarlo/data/sec_tickers.json`,
so the common case costs no network call. Refresh it occasionally:

```bash
SEC_USER_AGENT="Your Name you@example.com" python tools/refresh_tickers.py
```

### Not every filer works

Banks, brokers and insurers are rejected outright: they book interest as
revenue, so EBITDA is not meaningful and the bridge would return confident
nonsense. REITs warn. Filers using custom taxonomies for revenue (Exxon, for
one) cannot be read generically and fail with a clear message.

### Configuration

| Variable | Effect |
| --- | --- |
| `FINNHUB_API_KEY` | Fetches the share price too. Without it every other field still fills and you type the price; enterprise value recomputes as you do. |
| `SEC_USER_AGENT` | `"Name email@domain"`. Only needed for tickers missing from the bundled map — `www.sec.gov` returns 403 without a contact. |

No keyless quote feed proved reliable enough to ship: Stooq now sits behind a
JavaScript proof-of-work wall and Yahoo's undocumented endpoints rate-limit
datacenter IPs. Serving a stale price silently is worse than asking for one number.

## Tests

```bash
python -m unittest discover -s tests -t .
```

42 tests, no network needed — SEC parsing runs against a synthetic fixture.
Golden-value regressions pin the published MYRG numbers. Set `IMA_LIVE_SEC=1`
to additionally exercise EDGAR for real.

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

- `app.py` — Flask entrypoint; serves `public/` and the two API routes
- `montecarlo/` — engine, request layer, and the SEC data layer
- `requirements.txt` — flask + numpy + scipy (**no matplotlib**; charts are SVG)

The function never returns the raw price vector. It sends a 100-bin histogram,
a 400-point CDF, tornado bounds and summary stats — roughly 13 KB per run.

```bash
vercel --prod
```

## Known gaps

See [ROADMAP.md](ROADMAP.md). The headline items now: no persistence, no way to
edit the correlation matrix from the UI, and no chart export for decks.

## Disclaimer

Model output, not investment advice. Results are only as good as the
assumptions you feed in.
