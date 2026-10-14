"""
Backtest harness: scores a model's forecast distributions against what the
share price actually did.

For each name and each past quarter-end, the auto-fill is rebuilt using only
filings available on that date (EDGAR `filed` dates) and the close on that
date (Yahoo Finance). The model runs at the chosen horizon, the realised price
at the end of the horizon is looked up, and two things are recorded:

- PIT, the probability integral transform: the forecast distribution's
  percentile at the realised price. If the distributions are right, PIT is
  uniform on (0, 1) across snapshots (Diebold, Gunther and Tay, 1998).
- CRPS, the continuous ranked probability score in dollars, and divided by
  spot so names can be pooled (Gneiting and Raftery, 2007). Lower is better.

Any model variant is pluggable as `model_fn(payload) -> np.ndarray` of
terminal prices, so each roadmap change is accepted or rejected on the same
numbers. No part of this uses a model or inference service.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from typing import Callable, Optional

import numpy as np

from . import marketdata as md
from . import prices
from .engine import MonteCarloEngine, PitchAssumptions

DEFAULT_NAMES = [
    # the names in use, then their nearest filers by business
    "MYRG", "PWR", "DY", "EME", "FIX", "MTZ", "STRL", "IESC", "PRIM",
    "TDS", "SHEN", "ATNI",
    "QNST", "ZD", "MGNI", "PUBM", "CARG", "EVER",
]


def quarter_ends(start: str, end: str) -> list:
    """ISO dates of calendar quarter-ends in [start, end]."""
    out = []
    y, m = int(start[:4]), int(start[5:7])
    m = ((m - 1) // 3 + 1) * 3
    while True:
        d = date(y, m, [31, 30, 30, 31][m // 3 - 1]).isoformat()
        if d > end:
            break
        if d >= start:
            out.append(d)
        m += 3
        if m > 12:
            m, y = 3, y + 1
    return out


def crps_from_samples(samples: np.ndarray, y: float) -> float:
    """CRPS = E|X - y| - 0.5 E|X - X'| from a sorted sample (energy form)."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    term1 = float(np.mean(np.abs(x - y)))
    idx = np.arange(1, n + 1)
    term2 = float(2.0 / (n * n) * np.sum((2 * idx - n - 1) * x))
    return term1 - 0.5 * term2


def pit(samples: np.ndarray, y: float) -> float:
    return float(np.mean(np.asarray(samples) <= y))


def default_model(payload: dict, n_sims: int = 20_000, seed: int = 42) -> np.ndarray:
    """Run the engine on a prefill payload; Phase 2 keys pass straight through when present."""
    kw = {k: v for k, v in payload.items()
          if k in PitchAssumptions.__dataclass_fields__}
    for k in ("rev_cagr", "ebitda_margin", "ev_ebitda_multiple", "net_debt_change_pct", "share_dilution_pct"):
        kw[k] = tuple(kw[k])
    if kw.get("correlation_matrix") is not None:
        kw["correlation_matrix"] = np.asarray(kw["correlation_matrix"], dtype=float)
    kw.update(n_simulations=n_sims, random_seed=seed)
    engine = MonteCarloEngine(PitchAssumptions(**kw))
    return engine.run()


def snapshot(ticker: str, as_of: str, series: list, horizon: float,
             model_fn: Callable[[dict], np.ndarray], history_years: int = 10,
             prefill_fn: Callable = None) -> Optional[dict]:
    window = prices.forward_return_window(series, as_of, horizon)
    if not window:
        return None
    spot, realised, realised_date = window
    try:
        if prefill_fn:
            pre = prefill_fn(ticker, spot, horizon, as_of, history_years)
        else:
            pre = md.build_prefill(ticker, price=spot, horizon_years=horizon,
                                   history_years=history_years, as_of=as_of, calibration="v1")
    except md.DataError as exc:
        return {"ticker": ticker, "as_of": as_of, "error": str(exc)}
    f = dict(pre["fields"])
    if f.get("current_net_debt") is None:
        # no cash or debt tagged at that date: treat as zero and carry the flag
        f["current_net_debt"] = 0.0
        f["current_ev"] = spot * f["shares_outstanding"]
    f.update(bear_price=spot * 0.8, base_price=spot * 1.15, bull_price=spot * 1.4)
    for k in ("driver_specs", "copula", "copula_df", "correlation_matrix"):
        if k in pre:
            f[k] = pre[k]
    f.pop("horizon_years", None)
    f["horizon_years"] = horizon
    t0 = time.perf_counter()
    try:
        samples = model_fn(f)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "as_of": as_of, "error": f"model: {type(exc).__name__}: {exc}"}
    q = np.percentile(samples, [5, 10, 25, 50, 75, 90, 95])
    return {
        "ticker": ticker, "as_of": as_of, "spot": spot,
        "realised": realised, "realised_date": realised_date,
        "median": float(q[3]), "p5": float(q[0]), "p10": float(q[1]), "p25": float(q[2]),
        "p75": float(q[4]), "p90": float(q[5]), "p95": float(q[6]),
        "pit": pit(samples, realised),
        "crps": crps_from_samples(samples, realised),
        "crps_over_spot": crps_from_samples(samples, realised) / spot,
        "log_error_median": float(np.log(realised / q[3])) if q[3] > 0 else None,
        "in_50": bool(q[2] <= realised <= q[4]),
        "in_80": bool(q[1] <= realised <= q[5]),
        "in_90": bool(q[0] <= realised <= q[6]),
        "latest_fy": pre["company"]["latest_fy"],
        "fields": {k: f[k] for k in ("current_revenue", "current_ebitda_margin", "current_ev",
                                     "rev_cagr", "ebitda_margin", "ev_ebitda_multiple")},
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def summarize(rows: list) -> dict:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n": 0, "errors": len(rows)}
    pits = np.array([r["pit"] for r in ok])
    hist, _ = np.histogram(pits, bins=10, range=(0, 1))
    return {
        "n": len(ok),
        "errors": len(rows) - len(ok),
        "names": sorted({r["ticker"] for r in ok}),
        "mean_crps": float(np.mean([r["crps"] for r in ok])),
        "mean_crps_over_spot": float(np.mean([r["crps_over_spot"] for r in ok])),
        "median_abs_log_error": float(np.median([abs(r["log_error_median"]) for r in ok if r["log_error_median"] is not None])),
        "mean_log_error": float(np.mean([r["log_error_median"] for r in ok if r["log_error_median"] is not None])),
        "coverage_50": float(np.mean([r["in_50"] for r in ok])),
        "coverage_80": float(np.mean([r["in_80"] for r in ok])),
        "coverage_90": float(np.mean([r["in_90"] for r in ok])),
        "pit_hist": hist.tolist(),
        "pit_mean": float(pits.mean()),
        # a U-shaped histogram = too narrow; a hump = too wide; a slope = biased
        "pit_tail_share": float(np.mean((pits < 0.1) | (pits > 0.9))),
    }


def run(names: list = None, start: str = "2016-12-31", end: str = None, horizon: float = 2.0,
        model_fn: Callable[[dict], np.ndarray] = None, label: str = "baseline",
        out_dir: str = "backtests", verbose: bool = True, prefill_fn: Callable = None) -> dict:
    names = names or DEFAULT_NAMES
    end = end or date.today().isoformat()
    model_fn = model_fn or default_model
    rows = []
    for t in names:
        try:
            series = prices.price_history(t, 12)
        except prices.PriceError as exc:
            rows.append({"ticker": t, "as_of": None, "error": f"prices: {exc}"})
            continue
        n_before = len(rows)
        for d in quarter_ends(start, end):
            r = snapshot(t, d, series, horizon, model_fn, prefill_fn=prefill_fn)
            if r:
                rows.append(r)
        if verbose:
            done = [r for r in rows[n_before:] if "error" not in r]
            errs = [r for r in rows[n_before:] if "error" in r]
            msg = f"  {t:<6} {len(done):>3} snapshots"
            if done:
                msg += f"  mean CRPS/spot {np.mean([r['crps_over_spot'] for r in done]):.3f}  cov90 {np.mean([r['in_90'] for r in done]):.0%}"
            if errs:
                msg += f"  ({len(errs)} skipped: {errs[0]['error'][:60]})"
            print(msg, flush=True)
    summary = summarize(rows)
    result = {"label": label, "horizon_years": horizon, "start": start, "end": end,
              "run_at": date.today().isoformat(), "summary": summary, "rows": rows}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1)
    if verbose:
        print_summary(summary, label)
        print(f"  written to {path}")
    return result


def print_summary(s: dict, label: str = "") -> None:
    if not s.get("n"):
        print(f"{label}: no scored snapshots ({s.get('errors', 0)} errors)")
        return
    print(f"\n{label}: {s['n']} snapshots across {len(s['names'])} names ({s['errors']} skipped)")
    print(f"  mean CRPS / spot        {s['mean_crps_over_spot']:.4f}   (lower is better)")
    print(f"  median |log error|      {s['median_abs_log_error']:.3f}   mean log error {s['mean_log_error']:+.3f} (negative = model too high)")
    print(f"  coverage  50/80/90      {s['coverage_50']:.0%} / {s['coverage_80']:.0%} / {s['coverage_90']:.0%}   (target 50/80/90)")
    print(f"  PIT tails (<0.1 or >0.9) {s['pit_tail_share']:.0%}   (target 20%; higher = too narrow)")
    bars = " ".join(f"{h:>3}" for h in s["pit_hist"])
    print(f"  PIT histogram (10 bins)  {bars}")


def compare(a: dict, b: dict) -> dict:
    """Paired comparison of two runs on the snapshots both scored."""
    ka = {(r["ticker"], r["as_of"]): r for r in a["rows"] if "error" not in r}
    kb = {(r["ticker"], r["as_of"]): r for r in b["rows"] if "error" not in r}
    keys = sorted(set(ka) & set(kb))
    if not keys:
        return {"n": 0}
    da = np.array([ka[k]["crps_over_spot"] for k in keys])
    db = np.array([kb[k]["crps_over_spot"] for k in keys])
    diff = db - da
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return {
        "n": len(keys),
        "a": a["label"], "b": b["label"],
        "mean_crps_over_spot_a": float(da.mean()), "mean_crps_over_spot_b": float(db.mean()),
        "delta": float(diff.mean()), "delta_se": se,
        "b_better_share": float(np.mean(diff < 0)),
        "coverage_90_a": float(np.mean([ka[k]["in_90"] for k in keys])),
        "coverage_90_b": float(np.mean([kb[k]["in_90"] for k in keys])),
    }
