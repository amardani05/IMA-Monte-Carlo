#!/usr/bin/env python3
"""
Build montecarlo/data/peers.json: {ticker: {"cik", "sic", "sic_description", "name"}}
for a curated universe, using data.sec.gov submissions (no email-bearing
User-Agent needed). The sector panel groups names by SIC from this file.

    python tools/build_peers.py            # curated universe
    python tools/build_peers.py --add XYZ  # append a name

With SEC_USER_AGENT set to "Name email@domain", --sic 1731 pulls every filer
in that SIC from www.sec.gov's company browse instead.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from montecarlo import marketdata as md  # noqa: E402

UNIVERSE = """
MYRG PWR DY EME FIX MTZ STRL IESC PRIM ACM J FLR KBR TTEK ROAD GVA APG
TDS SHEN ATNI CNSL LUMN FYBR T VZ TMUS CHTR CMCSA CABO WOW
QNST ZD MGNI PUBM CARG EVER TTD DV IAS CRTO TBLA OB PERI APP
CAT DE PCAR CMI ETN EMR ROK AME DOV ITW PH IR GWW FAST
AAPL MSFT NVDA AMD INTC QCOM TXN AVGO MU ADI
COST WMT TGT DG DLTR KR HD LOW BBY ULTA
TSLA GM F RIVN
WM RSG CLH CWST
UNP CSX NSC JBHT ODFL XPO SAIA KNX
NKE LULU CROX DECK VFC
ADBE CRM NOW INTU ORCL WDAY PANW CRWD
""".split()

OUT = os.path.join(os.path.dirname(__file__), "..", "montecarlo", "data", "peers.json")

ap = argparse.ArgumentParser()
ap.add_argument("--add", nargs="*", default=[])
ap.add_argument("--from-file", default=None, help="tickers, one per line (TICKER|name|sector lines accepted)")
ap.add_argument("--replace", action="store_true", help="drop names not in the file or the curated list")
args = ap.parse_args()
FILE_NAMES, FILE_META = [], {}
if args.from_file:
    for line in open(args.from_file):
        line = line.strip()
        if not line or line.startswith("COUNT") or line.startswith("#"):
            continue
        parts = line.split("|")
        t = parts[0].strip().upper().replace(".", "-")
        FILE_NAMES.append(t)
        if len(parts) >= 3:
            FILE_META[t] = {"index_name": parts[1].strip(), "gics_sector": parts[2].strip()}

existing = {}
if os.path.exists(OUT):
    existing = json.load(open(OUT))

names = sorted(set(UNIVERSE) | set(t.upper() for t in args.add) | set(FILE_NAMES) | (set() if args.replace else set(existing)))
out = {t: v for t, v in existing.items() if t in names} if args.replace else dict(existing)
for t in names:
    if t in out and out[t].get("sic"):
        continue
    try:
        cik, title = md.resolve_cik(t)
        meta = md.company_meta(cik)
        out[t] = {"cik": cik, "name": meta.get("name") or title, "sic": meta.get("sic"),
                  "sic_description": meta.get("sic_description"), "exchanges": meta.get("exchanges"),
                  **FILE_META.get(t, {}), "in_sp600": t in FILE_NAMES}
        print(f"  {t:<6} SIC {meta.get('sic')}  {meta.get('sic_description')}")
    except md.DataError as exc:
        print(f"  {t:<6} skipped: {exc}")
    time.sleep(0.12)
    if len(out) % 50 == 0:
        print(f"  {len(out)} resolved", flush=True)

with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print(f"wrote {len(out)} names to {os.path.normpath(OUT)}")
