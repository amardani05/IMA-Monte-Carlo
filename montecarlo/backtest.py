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


# ─────────────────────────────────────────────────────────────────────────────
# Reporting and the regression gate
# ─────────────────────────────────────────────────────────────────────────────
def breakdown(rows: list, key) -> dict:
    """Mean CRPS/spot, coverage and count per group, where key(row) names the group."""
    groups: dict = {}
    for r in rows:
        if "error" in r:
            continue
        groups.setdefault(key(r), []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        c = np.array([r["crps_over_spot"] for r in rs])
        out[g] = {"n": len(rs), "mean_crps": float(c.mean()), "median_crps": float(np.median(c)),
                  "coverage_90": float(np.mean([r["in_90"] for r in rs])),
                  "coverage_50": float(np.mean([r["in_50"] for r in rs])),
                  "mean_log_error": float(np.mean([r["log_error_median"] for r in rs if r["log_error_median"] is not None]))}
    return out


def robust_summary(rows: list) -> dict:
    """Outlier-aware companions to the mean CRPS: median and 10% trimmed mean."""
    c = np.sort(np.array([r["crps_over_spot"] for r in rows if "error" not in r]))
    if len(c) == 0:
        return {}
    k = int(len(c) * 0.1)
    return {"median_crps": float(np.median(c)), "trimmed_mean_crps": float(c[k:len(c) - k].mean()) if len(c) > 2 * k else float(c.mean())}


def gate(prev: dict, cur: dict, max_crps_worsening: float = 0.0, min_coverage_gain: float = -0.02) -> dict:
    """
    Accept `cur` over `prev` only if paired CRPS did not worsen beyond the
    tolerance and 90% coverage did not fall by more than the tolerance. Both
    conditions, because a model can win CRPS by being narrow and wrong less
    often than it is right, and win coverage by being uselessly wide.
    """
    c = compare(prev, cur)
    if not c.get("n"):
        return {"pass": False, "reason": "no paired snapshots"}
    crps_ok = c["delta"] <= max_crps_worsening
    cov_ok = (c["coverage_90_b"] - c["coverage_90_a"]) >= min_coverage_gain
    return {"pass": bool(crps_ok and cov_ok), "paired": c, "crps_ok": crps_ok, "coverage_ok": cov_ok}


def report_html(result: dict, prev: Optional[dict] = None) -> str:
    """A self-contained HTML report: summary, PIT histogram, per-name and per-year tables."""
    s = result["summary"]; rows = result["rows"]
    rb = robust_summary(rows)
    by_name = breakdown(rows, lambda r: r["ticker"])
    by_year = breakdown(rows, lambda r: r["as_of"][:4])
    hist = s["pit_hist"]; hmax = max(hist) or 1
    bars = "".join(
        f'<rect x="{20 + i * 34}" y="{140 - 120 * h / hmax:.1f}" width="28" height="{120 * h / hmax:.1f}" fill="#2a63b0" opacity=".85"/>'
        f'<text x="{34 + i * 34}" y="156" font-size="9" text-anchor="middle" fill="#64707e">{i / 10:.1f}</text>'
        for i, h in enumerate(hist))
    expected = 140 - 120 * (s["n"] / 10) / hmax
    pit_svg = (f'<svg viewBox="0 0 380 170" width="380" height="170"><line x1="20" y1="{expected:.1f}" x2="360" y2="{expected:.1f}" '
               f'stroke="#b1271f" stroke-dasharray="4 3"/>{bars}<text x="366" y="{expected:.1f}" font-size="9" fill="#b1271f">uniform</text></svg>')
    def tbl(d, label):
        head = f"<tr><th>{label}</th><th>n</th><th>mean CRPS/spot</th><th>median</th><th>cov 50</th><th>cov 90</th><th>mean log err</th></tr>"
        body = "".join(f"<tr><td>{k}</td><td>{v['n']}</td><td>{v['mean_crps']:.3f}</td><td>{v['median_crps']:.3f}</td>"
                       f"<td>{v['coverage_50']:.0%}</td><td>{v['coverage_90']:.0%}</td><td>{v['mean_log_error']:+.2f}</td></tr>" for k, v in d.items())
        return f"<table>{head}{body}</table>"
    cmp_html = ""
    if prev:
        g = gate(prev, result); c = g.get("paired", {})
        if c.get("n"):
            cmp_html = (f"<h2>Against {c['a']}</h2><p>Paired on {c['n']} snapshots. CRPS/spot {c['mean_crps_over_spot_a']:.4f} to "
                        f"{c['mean_crps_over_spot_b']:.4f} (delta {c['delta']:+.4f}, se {c['delta_se']:.4f}); better on {c['b_better_share']:.0%}; "
                        f"90% coverage {c['coverage_90_a']:.0%} to {c['coverage_90_b']:.0%}. Gate: <b>{'PASS' if g['pass'] else 'FAIL'}</b> "
                        f"(CRPS {'ok' if g['crps_ok'] else 'worse'}, coverage {'ok' if g['coverage_ok'] else 'worse'}).</p>")
    return f"""<!doctype html><meta charset="utf-8"><title>Backtest {result['label']}</title>
<style>body{{font:14px/1.5 -apple-system,Segoe UI,sans-serif;color:#101820;max-width:980px;margin:32px auto;padding:0 20px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:8px 0 20px}}td,th{{padding:4px 10px;border-bottom:1px solid #e9edf2;text-align:right}}
td:first-child,th:first-child{{text-align:left}}h1{{font-size:20px}}h2{{font-size:15px;margin-top:26px}}.k{{color:#64707e}}</style>
<h1>Backtest: {result['label']}</h1>
<p class="k">{s['n']} snapshots across {len(s['names'])} names, {result['horizon_years']:g}-year horizon, {result['start']} to {result['end']}, run {result['run_at']}. {s['errors']} skipped.</p>
<table><tr><th>mean CRPS/spot</th><th>median</th><th>10% trimmed</th><th>cov 50</th><th>cov 80</th><th>cov 90</th><th>PIT tails</th><th>mean log err</th></tr>
<tr><td>{s['mean_crps_over_spot']:.4f}</td><td>{rb.get('median_crps', float('nan')):.4f}</td><td>{rb.get('trimmed_mean_crps', float('nan')):.4f}</td>
<td>{s['coverage_50']:.0%}</td><td>{s['coverage_80']:.0%}</td><td>{s['coverage_90']:.0%}</td><td>{s['pit_tail_share']:.0%}</td><td>{s['mean_log_error']:+.3f}</td></tr></table>
<h2>PIT histogram</h2><p class="k">Uniform if the distributions are right. A U means too narrow; a hump, too wide; a slope, biased.</p>{pit_svg}
{cmp_html}
<h2>By name</h2>{tbl(by_name, 'name')}
<h2>By snapshot year</h2>{tbl(by_year, 'year')}"""
