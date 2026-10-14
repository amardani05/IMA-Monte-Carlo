"""
Sector reference class: the outside view.

A company's own ten filed years are a sample of one path. The Base Rate Book
argument (Mauboussin and Callahan, 2016) is that the distribution of outcomes
across comparable companies is the right prior, blended with the inside view
in proportion to how much the company's own record tells you. This module
builds that reference class from EDGAR and Yahoo Finance for the names in
montecarlo/data/peers.json, grouped by SIC code, and estimates:

- the cross-sectional median and spread of EBITDA margin, N-year revenue CAGR
  and EV/EBITDA, pooled over the panel's filed years;
- the rate at which margins revert toward the panel median, as a pooled
  partial-adjustment regression (the Fama-French 2000 construction), with
  0.38 a year as the fallback when the panel is too thin to estimate;
- pairwise correlations between year-over-year changes in growth, margin and
  log multiple, which replace the hard-coded copula matrix.

Panels are cached to montecarlo/data/panels/<sic>.json by tools/build_panels.py
so the web app never fetches fifteen company-facts documents on a request.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np

from . import marketdata as md

PEERS_PATH = os.path.join(os.path.dirname(__file__), "data", "peers.json")
PANEL_DIR = os.path.join(os.path.dirname(__file__), "data", "panels")
FAMA_FRENCH_LAMBDA = 0.38
MIN_PEERS = 4
_PEERS: Optional[dict] = None


def peers_map() -> dict:
    global _PEERS
    if _PEERS is None:
        try:
            with open(PEERS_PATH) as fh:
                _PEERS = json.load(fh)
        except (OSError, json.JSONDecodeError):
            _PEERS = {}
    return _PEERS


OVERRIDES_PATH = os.path.join(os.path.dirname(__file__), "data", "peer_overrides.json")


def peers_for(ticker: str, sic: Optional[int]) -> tuple:
    """(peer tickers, basis). A hand-picked list wins; then same SIC; then the 2-digit SIC group."""
    ticker = ticker.upper()
    pm = peers_map()
    try:
        with open(OVERRIDES_PATH) as fh:
            ov = json.load(fh)
        if ticker in ov and isinstance(ov[ticker], list):
            return [t.upper() for t in ov[ticker] if t.upper() != ticker], "hand-picked comparables"
    except (OSError, json.JSONDecodeError):
        pass
    if sic is None:
        return [], "no SIC"
    same = sorted(t for t, v in pm.items() if v.get("sic") == sic and t != ticker)
    if len(same) >= MIN_PEERS:
        return same, f"SIC {sic}"
    group = sorted(t for t, v in pm.items() if v.get("sic") and v["sic"] // 100 == sic // 100 and t != ticker)
    if len(group) >= MIN_PEERS:
        return group, f"SIC group {sic // 100}xx"
    return same or group, f"SIC {sic} (thin)"


def _q(values, qs=(10, 50, 90)):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None
    return {f"p{q}": float(np.percentile(v, q)) for q in qs} | {"n": int(len(v))}


def _reversion_rate(margin_series: list, center: float) -> Optional[dict]:
    """
    Pooled partial adjustment: (m_{t+1} - c) = (1 - lambda)(m_t - c) + e.
    Returns lambda per year and the fit's n. None when fewer than 12 pairs.
    """
    x, y = [], []
    for series in margin_series:
        ys = sorted(series)
        for a, b in zip(ys, ys[1:]):
            if b - a == 1:
                x.append(series[a] - center)
                y.append(series[b] - center)
    if len(x) < 12:
        return None
    x, y = np.asarray(x), np.asarray(y)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return None
    rho = float(np.dot(x, y) / denom)           # (1 - lambda), no intercept
    lam = min(max(1.0 - rho, 0.0), 1.0)
    return {"lambda": lam, "n_pairs": int(len(x))}


def _yoy_changes(series: dict) -> dict:
    ys = sorted(series)
    return {b: series[b] - series[a] for a, b in zip(ys, ys[1:]) if b - a == 1}


def _corr(a: dict, b: dict) -> Optional[float]:
    ks = sorted(set(a) & set(b))
    if len(ks) < 8:
        return None
    x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks])
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def build_panel(ticker: str, sic: Optional[int], horizon_years: float = 2.0,
                history_years: int = 10, as_of: Optional[str] = None,
                with_prices: bool = True, verbose: bool = False) -> dict:
    """Reference-class statistics for a ticker's SIC group. Pure arithmetic on filed data."""
    peers, basis = peers_for(ticker, sic)
    span = max(1, int(round(horizon_years)))
    margins, cagrs, mults = [], [], []
    margin_series, growth_series, mult_series = [], [], []
    used = []
    for p in peers:
        try:
            prof = md.build_profile(p, history_years=history_years, as_of=as_of)
        except md.DataError as exc:
            if verbose:
                print(f"    {p}: skipped ({str(exc)[:50]})")
            continue
        mh = {int(y): v for y, v in prof["margin_history"].items()}
        rh = {int(y): v for y, v in prof["revenue_history_m"].items()}
        if len(mh) < 3:
            continue
        used.append(p)
        margins.extend(mh.values())
        margin_series.append(mh)
        c = md.rolling_cagrs(prof["revenue_history_m"], span)
        cagrs.extend(c)
        # one-year growth per year, for correlation
        ys = sorted(rh)
        growth_series.append({b: rh[b] / rh[a] - 1 for a, b in zip(ys, ys[1:]) if b - a == 1 and rh[a] > 0})
        if with_prices:
            try:
                from .prices import price_history
                mh_p = price_based_multiples(prof, price_history(p, 12), as_of=as_of)
            except Exception:
                mh_p = {}
            if mh_p:
                mults.extend(mh_p.values())
                mult_series.append({y: math.log(v) for y, v in mh_p.items() if v > 0})
    center = float(np.median(margins)) if margins else None
    rev = _reversion_rate(margin_series, center) if center is not None else None

    # correlations of year-over-year changes, pooled across peers
    pairs = {"g_m": [], "g_mult": [], "m_mult": []}
    for i, g in enumerate(growth_series):
        m = margin_series[i] if i < len(margin_series) else {}
        dm = _yoy_changes(m)
        lm = mult_series[i] if i < len(mult_series) else {}
        dl = _yoy_changes(lm)
        for key, (a, b) in {"g_m": (g, dm), "g_mult": (g, dl), "m_mult": (dm, dl)}.items():
            ks = sorted(set(a) & set(b))
            pairs[key].extend((a[k], b[k]) for k in ks)
    corr = {}
    for key, pts in pairs.items():
        if len(pts) >= 12:
            x, y = np.array(pts).T
            corr[key] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
        else:
            corr[key] = None

    return {
        "ticker": ticker.upper(), "sic": sic, "basis": basis, "peers_used": used,
        "as_of": as_of, "horizon_years": horizon_years,
        "margin": _q(margins), "cagr": _q(cagrs), "multiple": _q(mults),
        "margin_reversion": rev or {"lambda": FAMA_FRENCH_LAMBDA, "n_pairs": 0, "fallback": "Fama-French 2000"},
        "corr_changes": corr,
    }


def price_based_multiples(profile: dict, series: list, as_of: Optional[str] = None) -> dict:
    """EV/EBITDA at each fiscal year-end from the close that day, shares and net debt on the balance sheet."""
    from .prices import close_on
    out = {}
    sh = profile.get("shares_history_m") or {}
    nd = profile.get("net_debt_history_m") or {}
    rev = profile.get("revenue_history_m") or {}
    mg = profile.get("margin_history") or {}
    for y_key in rev:
        y = int(y_key)
        if y_key not in mg or y_key not in sh:
            continue
        ebitda = rev[y_key] * mg[y_key]
        if ebitda <= 0 or rev[y_key] <= 0 or mg[y_key] < 0.02:
            continue
        date = f"{y}-12-31"
        if as_of and date > as_of:
            continue
        c = close_on(series, date)
        if not c:
            continue
        ev = c[1] * sh[y_key] + nd.get(y_key, 0.0)
        if ev <= 0:
            continue
        m = ev / ebitda
        if m <= 60:
            out[y] = m
    return out


def panel_path(key) -> str:
    return os.path.join(PANEL_DIR, f"{key}.json")


def load_cached(sic: Optional[int], ticker: Optional[str] = None) -> Optional[dict]:
    """A hand-picked panel for the ticker wins over the SIC panel."""
    candidates = []
    if ticker and sic is not None:
        candidates.append(f"{sic}-{ticker.upper()}")
    if sic is not None:
        candidates.append(sic)
    for key in candidates:
        try:
            with open(panel_path(key)) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def save_cached(panel: dict) -> str:
    os.makedirs(PANEL_DIR, exist_ok=True)
    path = panel_path(panel["sic"])
    with open(path, "w") as fh:
        json.dump(panel, fh, indent=1)
    return path
