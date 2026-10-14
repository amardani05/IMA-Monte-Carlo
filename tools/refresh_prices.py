#!/usr/bin/env python3
"""
Refresh the bundled price series in montecarlo/data/prices/.

    python tools/refresh_prices.py                 # every name in peers.json plus the harness list
    python tools/refresh_prices.py MYRG TDS QNST   # just these

Writes one JSON per ticker: {"as_of", "symbol", "closes": [[date, close], ...]}
with daily closes for 12 years, unadjusted for dividends. The deployed app reads
these first, so run this weekly (a cron job once the repository has CI) and
commit the result. Needs yfinance locally.
"""

import argparse
import json
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from montecarlo import prices, panel, backtest  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "montecarlo", "data", "prices")

ap = argparse.ArgumentParser()
ap.add_argument("tickers", nargs="*")
ap.add_argument("--years", type=int, default=12)
args = ap.parse_args()

names = [t.upper() for t in args.tickers] or sorted(set(panel.peers_map()) | set(backtest.DEFAULT_NAMES) | {"SPY"})
os.makedirs(OUT, exist_ok=True)
ok = 0
for t in names:
    try:
        series = prices._fetch_yfinance(t, args.years) or prices._fetch_chart(t, f"{args.years}y")
    except prices.PriceError as exc:
        print(f"  {t:<6} skipped: {exc}")
        continue
    if not series:
        print(f"  {t:<6} skipped: empty")
        continue
    doc = {"as_of": date.today().isoformat(), "symbol": t, "closes": [[d, round(c, 4)] for d, c in sorted(series)]}
    with open(os.path.join(OUT, f"{t}.json"), "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    ok += 1
    time.sleep(0.2)
print(f"wrote {ok} series to {os.path.normpath(OUT)}")
