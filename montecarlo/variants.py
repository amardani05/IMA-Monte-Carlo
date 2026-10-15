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


def _with_mechanism(kind, params=None):
    """v2 prefill with one Phase 3 block lifted into the payload when filings support it."""
    def f(t, p, h, a, hy=10):
        pre = phase2_prefill(t, p, h, a, hy, params=params)
        m = pre.get("mechanisms") or {}
        block = m.get(kind)
        if m.get("available", {}).get(kind) and block and not (kind == "justified_multiple" and block.get("roic") is None):
            pre[kind] = block
        return pre
    return f


def _dilution_over_horizon(t, p, h, a, hy=10):
    """
    v2 prefill with dilution stated over the horizon, the way net debt already
    is: calibration hands over one year's share-count change (mean, std) and
    the engine applies it as the change over the whole horizon. Mean x T, std
    x sqrt(T), the same scaling the net-debt drift gets.
    """
    pre = phase2_prefill(t, p, h, a, hy)
    mean, std = pre["fields"]["share_dilution_pct"]
    mean, std = mean * h, std * h ** 0.5
    pre["fields"]["share_dilution_pct"] = [mean, std]
    spec = (pre.get("driver_specs") or {}).get("share_dilution_pct")
    if spec:
        pre["driver_specs"]["share_dilution_pct"] = dict(spec, mean=mean, std=std)
    return pre


def _level_width(horizon):
    """Engine with level-driver width scaled by sqrt(T / horizon) under that horizon."""
    return lambda p: backtest.default_model(dict(p, level_width_horizon=horizon))


def _fcf_scored(payload):
    """Score the FCF bridge's price where the payload carries one, else the EBITDA bridge."""
    from .engine import MonteCarloEngine, PitchAssumptions
    kw = {k: v for k, v in payload.items() if k in PitchAssumptions.__dataclass_fields__}
    for k in ("rev_cagr", "ebitda_margin", "ev_ebitda_multiple", "net_debt_change_pct", "share_dilution_pct"):
        kw[k] = tuple(kw[k])
    if kw.get("correlation_matrix") is not None:
        kw["correlation_matrix"] = np.asarray(kw["correlation_matrix"], dtype=float)
    kw.update(n_simulations=20_000, random_seed=42)
    e = MonteCarloEngine(PitchAssumptions(**kw)); e.run()
    return e.terminal_prices_fcf if e.terminal_prices_fcf is not None else e.terminal_prices


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
    # Phase 3 on the harness: each mechanism switched on where filings support it
    "phase3_jm": (backtest.default_model, _with_mechanism("justified_multiple")),
    "phase3_cost": (backtest.default_model, _with_mechanism("cost_model")),
    "phase3_backlog": (backtest.default_model, _with_mechanism("backlog")),
    "phase3_fcf": (_fcf_scored, _with_mechanism("fcf_bridge")),
    "phase3_cost2": (backtest.default_model, _with_mechanism("cost_model")),     # tightened fixed-cost growth
    "phase3_jm2": (backtest.default_model, _with_mechanism("justified_multiple")),  # premium reversion 0.25
    # Phase 5 prototype: sector multiple momentum carried into the multiple centre
    "p5_mom05": (backtest.default_model, _v2({"mult_momentum": 0.5})),
    "p5_mom10": (backtest.default_model, _v2({"mult_momentum": 1.0})),
    # Horizon term structure: level widths full at 1 or 2 years and narrower under it;
    # dilution scaled to the horizon like net debt; the one-year widths and dilution together
    "phase2_lw1": (_level_width(1.0), phase2_prefill),
    "phase2_lw2": (_level_width(2.0), phase2_prefill),
    "phase2_dilT": (backtest.default_model, _dilution_over_horizon),
    "phase2_term": (_level_width(1.0), _dilution_over_horizon),
}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]
