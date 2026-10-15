#!/usr/bin/env python3
"""
Score the model against realised prices.

    python tools/backtest.py                       # default names, 2-year horizon
    python tools/backtest.py --names MYRG,TDS,QNST --start 2018-12-31
    python tools/backtest.py --names sp600         # the index constituents only
    python tools/backtest.py --label phase2 --compare baseline

Writes backtests/<label>.json. --compare prints a paired CRPS comparison
against an earlier run's file.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from montecarlo import backtest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--names", default=",".join(backtest.DEFAULT_NAMES))
ap.add_argument("--start", default="2016-12-31")
ap.add_argument("--end", default=None)
ap.add_argument("--horizon", type=float, default=2.0)
ap.add_argument("--label", default="baseline")
ap.add_argument("--model", default="default", help="model variant registered in montecarlo.variants")
ap.add_argument("--compare", default=None, help="label of an earlier run to compare against")
ap.add_argument("--gate", action="store_true", help="exit 1 if the run fails the gate against --compare")
ap.add_argument("--n-sims", type=int, default=20000)
args = ap.parse_args()
if args.names.strip().lower() == "all":
    # the S&P 600 plus the curated peer pool in peers.json (about 90 large caps)
    from montecarlo import panel as _panel  # noqa: E402
    args.names = ",".join(sorted(set(_panel.peers_map()) | set(backtest.DEFAULT_NAMES)))
elif args.names.strip().lower() == "sp600":
    # the index constituents and nothing else
    _path = os.path.join(os.path.dirname(__file__), "..", "montecarlo", "data", "sp600.txt")
    with open(_path) as fh:
        args.names = ",".join(sorted({ln.split("|")[0].strip().upper().replace(".", "-")
                                      for ln in fh if ln.strip() and not ln.startswith("#")}))

from montecarlo import variants  # noqa: E402
model_fn, prefill_fn = variants.get(args.model)

res = backtest.run(names=[n.strip().upper() for n in args.names.split(",") if n.strip()],
                   start=args.start, end=args.end, horizon=args.horizon,
                   model_fn=model_fn, label=args.label, prefill_fn=prefill_fn)

prev = None
if args.compare:
    with open(os.path.join("backtests", f"{args.compare}.json")) as fh:
        prev = json.load(fh)
    c = backtest.compare(prev, res)
    if c["n"]:
        print(f"\nPaired on {c['n']} snapshots: CRPS/spot {c['a']} {c['mean_crps_over_spot_a']:.4f} -> {c['b']} {c['mean_crps_over_spot_b']:.4f}")
        print(f"  delta {c['delta']:+.4f} (se {c['delta_se']:.4f}); {c['b']} better on {c['b_better_share']:.0%} of snapshots; cov90 {c['coverage_90_a']:.0%} -> {c['coverage_90_b']:.0%}")
rb = backtest.robust_summary(res["rows"])
if rb:
    print(f"  median CRPS/spot {rb['median_crps']:.4f}, 10% trimmed mean {rb['trimmed_mean_crps']:.4f}")
html_path = os.path.join("backtests", f"{args.label}.html")
with open(html_path, "w") as fh:
    fh.write(backtest.report_html(res, prev))
print(f"  report: {html_path}")
if args.gate and prev:
    g = backtest.gate(prev, res)
    print(f"  gate: {'PASS' if g['pass'] else 'FAIL'}")
    sys.exit(0 if g["pass"] else 1)
