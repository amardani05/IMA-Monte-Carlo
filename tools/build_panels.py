#!/usr/bin/env python3
"""
Build and cache sector panels for a list of tickers.

    python tools/build_panels.py MYRG TDS QNST
    python tools/build_panels.py --all      # every SIC present in peers.json

Each panel is written to montecarlo/data/panels/<sic>.json and is what the
web app reads; nothing at request time fetches peers.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from montecarlo import marketdata as md, panel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tickers", nargs="*")
ap.add_argument("--all", action="store_true")
ap.add_argument("--horizon", type=float, default=2.0)
args = ap.parse_args()

targets = [t.upper() for t in args.tickers]
if args.all:
    targets = sorted(panel.peers_map())
seen_keys = set()
for t in targets:
    try:
        cik, _ = md.resolve_cik(t)
        sic = md.company_meta(cik).get("sic")
    except md.DataError as exc:
        print(f"{t}: {exc}"); continue
    peers, basis = panel.peers_for(t, sic)
    key = t if basis == "hand-picked comparables" else sic
    if key in seen_keys:
        continue
    seen_keys.add(key)
    t0 = time.time()
    print(f"{t} (SIC {sic}, {basis}): {len(peers)} peers", flush=True)
    pnl = panel.build_panel(t, sic, horizon_years=args.horizon, verbose=True)
    if basis == "hand-picked comparables":
        pnl["sic"] = f"{sic}-{t}"
    path = panel.save_cached(pnl)
    m, c, mu, r = pnl["margin"], pnl["cagr"], pnl["multiple"], pnl["margin_reversion"]
    print(f"  used {len(pnl['peers_used'])} | margin p10/50/90 {m['p10']*100:.1f}/{m['p50']*100:.1f}/{m['p90']*100:.1f}% (n={m['n']})"
          if m else "  no margin data")
    if c: print(f"  {args.horizon:g}y CAGR p10/50/90 {c['p10']*100:.1f}/{c['p50']*100:.1f}/{c['p90']*100:.1f}% (n={c['n']})")
    if mu: print(f"  EV/EBITDA p10/50/90 {mu['p10']:.1f}/{mu['p50']:.1f}/{mu['p90']:.1f}x (n={mu['n']})")
    print(f"  margin reversion lambda {r['lambda']:.2f}/yr (n_pairs={r['n_pairs']}) | corr {pnl['corr_changes']} | {time.time()-t0:.0f}s -> {path}")
