"""Model variants for the backtest harness. Each is payload -> terminal prices."""

from __future__ import annotations

import numpy as np

from . import backtest, marketdata as md, panel as pnl

_PANEL_CACHE: dict = {}


def _panel_as_of(ticker: str, as_of: str, horizon: float) -> dict:
    """Sector panel rebuilt with filings available at the snapshot date, cached per year."""
    key = (ticker, as_of[:4], horizon)
    if key not in _PANEL_CACHE:
        cik, _ = md.resolve_cik(ticker)
        sic = md.company_meta(cik).get("sic")
        _PANEL_CACHE[key] = pnl.build_panel(ticker, sic, horizon_years=horizon, as_of=as_of, with_prices=True)
    return _PANEL_CACHE[key]


def phase2_prefill(ticker, price, horizon, as_of, history_years=10, params=None):
    """A prefill using calibration v2 with a point-in-time panel."""
    panel = _panel_as_of(ticker, as_of, horizon)
    return md.build_prefill(ticker, price=price, horizon_years=horizon, history_years=history_years,
                            as_of=as_of, calibration="v2", panel=panel, params=params)


def _v2(params):
    return lambda t, p, h, a, hy=10: phase2_prefill(t, p, h, a, hy, params=params)


REGISTRY = {
    # name: (model_fn, prefill_fn or None)
    "default": (backtest.default_model, None),
    "t_copula_only": (lambda p: backtest.default_model(dict(p, copula="t", copula_df=5.0)), None),
    "phase2": (backtest.default_model, phase2_prefill),       # current V2_DEFAULTS
    "v2_blend_all": (backtest.default_model, _v2({"k_shrink": 8.0, "lam_mult": 0.2, "width_scale": None, "growth_centre": "blend", "target": "blend"})),
    "phase2_gaussian": (lambda p: backtest.default_model(dict(p, copula="gaussian")), phase2_prefill),
    # sweep: trust own history more, no multiple reversion, no horizon width inflation, own growth centre
    "v2_k3": (backtest.default_model, _v2({"k_shrink": 3.0})),
    "v2_nomultrev": (backtest.default_model, _v2({"lam_mult": 0.0})),
    "v2_noscale": (backtest.default_model, _v2({"width_scale": 1.0})),
    "v2_owngrowth": (backtest.default_model, _v2({"growth_centre": "own"})),
    "v2_tight": (backtest.default_model, _v2({"k_shrink": 3.0, "lam_mult": 0.0, "width_scale": 1.0, "growth_centre": "own"})),
    "v2_tight_rev": (backtest.default_model, _v2({"k_shrink": 3.0, "lam_mult": 0.0, "width_scale": 1.0})),
    "v2_owntarget": (backtest.default_model, _v2({"target": "own"})),
    "v2_owntarget_tight": (backtest.default_model, _v2({"target": "own", "k_shrink": 3.0, "lam_mult": 0.0, "width_scale": 1.0})),
    "v2_owntarget_tight_owng": (backtest.default_model, _v2({"target": "own", "k_shrink": 3.0, "lam_mult": 0.0, "width_scale": 1.0, "growth_centre": "own"})),
}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]
