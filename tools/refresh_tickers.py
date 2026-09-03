#!/usr/bin/env python3
"""
Refresh the bundled SEC ticker -> CIK map.

Run occasionally (say quarterly) so newly listed companies resolve without the
live www.sec.gov fallback, which rate-limits hard and rejects User-Agents that
carry no contact details.

    SEC_USER_AGENT="Your Name you@example.com" python tools/refresh_tickers.py
"""

import json
import os
import sys
import urllib.request

URL = "https://www.sec.gov/files/company_tickers.json"
OUT = os.path.join(os.path.dirname(__file__), "..", "montecarlo", "data", "sec_tickers.json")

ua = os.environ.get("SEC_USER_AGENT")
if not ua or "@" not in ua:
    sys.exit(
        "Set SEC_USER_AGENT to a name and contact email, e.g.\n"
        '  SEC_USER_AGENT="Jane Doe jane@example.com" python tools/refresh_tickers.py\n'
        "www.sec.gov returns 403 without one."
    )

req = urllib.request.Request(URL, headers={"User-Agent": ua, "Accept": "application/json"})
with urllib.request.urlopen(req, timeout=30) as resp:
    raw = json.loads(resp.read().decode("utf-8"))

mapping = {}
for row in raw.values():
    ticker = str(row.get("ticker", "")).upper().strip()
    if ticker:
        mapping[ticker] = [int(row["cik_str"]), str(row.get("title", "")).strip()]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(mapping, fh, separators=(",", ":"), sort_keys=True)

print(f"Wrote {len(mapping):,} tickers to {os.path.normpath(OUT)}")
