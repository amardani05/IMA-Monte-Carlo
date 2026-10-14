#!/usr/bin/env python3
"""
Score the model against realised prices.

    python tools/backtest.py                       # default names, 2-year horizon
    python tools/backtest.py --names MYRG,TDS,QNST --start 2018-12-31
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
args = ap.parse_args()

from montecarlo import variants  # noqa: E402
model_fn, prefill_fn = variants.get(args.model)

res = backtest.run(names=[n.strip().upper() for n in args.names.split(",") if n.strip()],
                   start=args.start, end=args.end, horizon=args.horizon,
                   model_fn=model_fn, label=args.label, prefill_fn=prefill_fn)

if args.compare:
    with open(os.path.join("backtests", f"{args.compare}.json")) as fh:
        prev = json.load(fh)
    c = backtest.compare(prev, res)
    if c["n"]:
        print(f"\nPaired on {c['n']} snapshots: CRPS/spot {c['a']} {c['mean_crps_over_spot_a']:.4f} -> {c['b']} {c['mean_crps_over_spot_b']:.4f}")
        print(f"  delta {c['delta']:+.4f} (se {c['delta_se']:.4f}); {c['b']} better on {c['b_better_share']:.0%} of snapshots; cov90 {c['coverage_90_a']:.0%} -> {c['coverage_90_b']:.0%}")
