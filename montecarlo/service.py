"""
Request layer for the simulation API. Transport agnostic.

Validates an incoming assumption payload, runs the Monte Carlo engine, and
returns summary stats plus pre-aggregated chart series. The raw price vector is
never sent over the wire: only a histogram, a downsampled CDF, tornado bounds
and scalar statistics, which keeps a 100k-path run at roughly 13 KB.
"""

import time
from typing import Optional

import numpy as np

from .engine import PitchAssumptions, MonteCarloEngine

MAX_SIMULATIONS = 250_000
HIST_BINS = 100
CDF_POINTS = 400


class ValidationError(ValueError):
    """Raised when the incoming payload is not a runnable set of assumptions."""


def _num(payload, key, *, positive=False, nonzero=False):
    if key not in payload or payload[key] is None or payload[key] == "":
        raise ValidationError(f"Missing required field: {key}")
    try:
        val = float(payload[key])
    except (TypeError, ValueError):
        raise ValidationError(f"Field '{key}' must be a number")
    if not np.isfinite(val):
        raise ValidationError(f"Field '{key}' must be a finite number")
    if positive and val <= 0:
        raise ValidationError(f"Field '{key}' must be greater than zero")
    if nonzero and val == 0:
        raise ValidationError(f"Field '{key}' must not be zero")
    return val


def _triangular(payload, key):
    raw = payload.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValidationError(f"Field '{key}' must be [low, mode, high]")
    try:
        low, mode, high = (float(v) for v in raw)
    except (TypeError, ValueError):
        raise ValidationError(f"Field '{key}' must contain three numbers")
    if not all(np.isfinite(v) for v in (low, mode, high)):
        raise ValidationError(f"Field '{key}' must contain finite numbers")
    if not (low <= mode <= high):
        raise ValidationError(f"Field '{key}' requires low <= mode <= high")
    if high == low:
        raise ValidationError(f"Field '{key}' requires high > low (zero-width range)")
    return (low, mode, high)


def _normal(payload, key):
    raw = payload.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValidationError(f"Field '{key}' must be [mean, std]")
    try:
        mean, std = (float(v) for v in raw)
    except (TypeError, ValueError):
        raise ValidationError(f"Field '{key}' must contain two numbers")
    if not all(np.isfinite(v) for v in (mean, std)):
        raise ValidationError(f"Field '{key}' must contain finite numbers")
    if std <= 0:
        raise ValidationError(f"Field '{key}' requires std > 0")
    return (mean, std)


def build_assumptions(payload: dict) -> PitchAssumptions:
    ticker = str(payload.get("ticker") or "TICKER").strip()[:12] or "TICKER"
    company_name = str(payload.get("company_name") or ticker).strip()[:80]

    bear = _num(payload, "bear_price")
    base = _num(payload, "base_price")
    bull = _num(payload, "bull_price")
    if not (bear <= base <= bull):
        raise ValidationError("Case targets must satisfy bear <= base <= bull")

    n_sims = int(payload.get("n_simulations") or 100_000)
    if n_sims < 1_000:
        raise ValidationError("n_simulations must be at least 1,000")
    if n_sims > MAX_SIMULATIONS:
        raise ValidationError(f"n_simulations is capped at {MAX_SIMULATIONS:,}")

    horizon = float(payload.get("horizon_years") or 2.0)
    if not (0 < horizon <= 10):
        raise ValidationError("horizon_years must be between 0 and 10")

    corr = payload.get("correlation_matrix")
    if corr is not None:
        corr = np.asarray(corr, dtype=float)
        if corr.shape != (5, 5):
            raise ValidationError("correlation_matrix must be 5x5")
        if not np.allclose(corr, corr.T, atol=1e-8):
            raise ValidationError("correlation_matrix must be symmetric")
        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            raise ValidationError(
                "correlation_matrix is not positive definite, so the copula cannot be built from it"
            )

    driver_specs = payload.get("driver_specs")
    if driver_specs is not None:
        if not isinstance(driver_specs, dict):
            raise ValidationError("driver_specs must be an object keyed by driver name")
        from .distributions import make_sampler
        for name, spec in driver_specs.items():
            if name not in ("rev_cagr", "ebitda_margin", "ev_ebitda_multiple",
                            "net_debt_change_pct", "share_dilution_pct"):
                raise ValidationError(f"driver_specs has an unknown driver '{name}'")
            try:
                make_sampler(spec)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValidationError(f"driver_specs['{name}'] is not a valid distribution: {exc}")

    def _spec(block, key, where):
        try:
            make_sampler(block[key])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValidationError(f"{where}.{key} is not a valid distribution: {exc}")

    from .distributions import make_sampler
    cost_model = payload.get("cost_model")
    if cost_model is not None:
        if not isinstance(cost_model, dict):
            raise ValidationError("cost_model must be an object")
        _spec(cost_model, "gross_margin", "cost_model")
        fc = float(cost_model.get("fixed_cost_m", 0.0))
        if not np.isfinite(fc) or fc < 0:
            raise ValidationError("cost_model.fixed_cost_m must be a non-negative number")
        if cost_model.get("fixed_cost_growth") is not None:
            _spec(cost_model, "fixed_cost_growth", "cost_model")
    backlog = payload.get("backlog")
    if backlog is not None:
        if not isinstance(backlog, dict):
            raise ValidationError("backlog must be an object")
        b = float(backlog.get("backlog_m", 0.0))
        if not np.isfinite(b) or b < 0:
            raise ValidationError("backlog.backlog_m must be a non-negative number")
        _spec(backlog, "conversion", "backlog")
    fcf_bridge = payload.get("fcf_bridge")
    if fcf_bridge is not None:
        if not isinstance(fcf_bridge, dict):
            raise ValidationError("fcf_bridge must be an object")
        _spec(fcf_bridge, "fcf_margin", "fcf_bridge")
        _spec(fcf_bridge, "ev_fcf_multiple", "fcf_bridge")
    jm = payload.get("justified_multiple")
    if jm is not None:
        if not isinstance(jm, dict):
            raise ValidationError("justified_multiple must be an object")
        for k in ("roic", "wacc"):
            v = float(jm.get(k, float("nan")))
            if not np.isfinite(v) or v <= 0:
                raise ValidationError(f"justified_multiple.{k} must be a positive number")
        if float(jm["wacc"]) <= float(jm.get("long_run_growth", 0.03)):
            raise ValidationError("justified_multiple.wacc must exceed long_run_growth")

    copula = str(payload.get("copula") or "gaussian").lower()
    if copula not in ("gaussian", "t"):
        raise ValidationError("copula must be 'gaussian' or 't'")
    copula_df = float(payload.get("copula_df") or 5.0)
    if not (2.0 < copula_df <= 100.0):
        raise ValidationError("copula_df must be between 2 and 100")

    return PitchAssumptions(
        ticker=ticker,
        company_name=company_name,
        current_price=_num(payload, "current_price", positive=True),
        bear_price=bear,
        base_price=base,
        bull_price=bull,
        current_revenue=_num(payload, "current_revenue", positive=True),
        current_ebitda_margin=_num(payload, "current_ebitda_margin"),
        current_ev=_num(payload, "current_ev"),
        current_net_debt=_num(payload, "current_net_debt"),
        shares_outstanding=_num(payload, "shares_outstanding", positive=True),
        rev_cagr=_triangular(payload, "rev_cagr"),
        ebitda_margin=_triangular(payload, "ebitda_margin"),
        ev_ebitda_multiple=_triangular(payload, "ev_ebitda_multiple"),
        net_debt_change_pct=_normal(payload, "net_debt_change_pct"),
        share_dilution_pct=_normal(payload, "share_dilution_pct"),
        correlation_matrix=corr,
        n_simulations=n_sims,
        random_seed=int(payload.get("random_seed") or 42),
        horizon_years=horizon,
        driver_specs=driver_specs,
        copula=copula,
        copula_df=copula_df,
        cost_model=cost_model,
        backlog=backlog,
        fcf_bridge=fcf_bridge,
        justified_multiple=jm,
    )


def _driver_quantiles(a: PitchAssumptions) -> dict:
    """P10/P50/P90 of every driver as actually sampled, for display."""
    from .distributions import describe
    specs = a.driver_specs or {}
    out = {}
    for name, legacy in (("rev_cagr", a.rev_cagr), ("ebitda_margin", a.ebitda_margin),
                         ("ev_ebitda_multiple", a.ev_ebitda_multiple)):
        out[name] = describe(specs.get(name, tuple(legacy)))
    for name, legacy in (("net_debt_change_pct", a.net_debt_change_pct),
                         ("share_dilution_pct", a.share_dilution_pct)):
        spec = specs.get(name) or {"type": "normal", "mean": legacy[0], "std": legacy[1]}
        out[name] = describe(spec)
    return out


def _fcf_stats(engine: MonteCarloEngine) -> Optional[dict]:
    p = engine.terminal_prices_fcf
    if p is None:
        return None
    a = engine.a
    n = len(p)
    q = np.percentile(p, [5, 25, 50, 75, 95])
    return {
        "median_price": float(q[2]), "p5": float(q[0]), "p25": float(q[1]), "p75": float(q[3]), "p95": float(q[4]),
        "mean_price": float(p.mean()),
        "p_at_least_bear": float((p >= a.bear_price).sum() / n),
        "p_at_least_base": float((p >= a.base_price).sum() / n),
        "p_at_least_bull": float((p >= a.bull_price).sum() / n),
        "median_return": float(q[2] / a.current_price - 1),
        "gap_to_ebitda_bridge": float(q[2] / np.median(engine.terminal_prices) - 1) if np.median(engine.terminal_prices) > 0 else None,
    }


def _se(p: float, n: int) -> float:
    """Monte Carlo standard error of a probability estimate."""
    return (max(p * (1 - p), 0.0) / n) ** 0.5


def mode_path_price(a: PitchAssumptions, g=None, m=None, mult=None) -> float:
    """Price along the path with every driver at its mode (or P50), overridable per driver."""
    from .distributions import describe
    specs = a.driver_specs or {}
    g = g if g is not None else (describe(specs["rev_cagr"])["p50"] if "rev_cagr" in specs else a.rev_cagr[1])
    m = m if m is not None else (describe(specs["ebitda_margin"])["p50"] if "ebitda_margin" in specs else a.ebitda_margin[1])
    mult = mult if mult is not None else (describe(specs["ev_ebitda_multiple"])["p50"] if "ev_ebitda_multiple" in specs else a.ev_ebitda_multiple[1])
    nd_chg = specs["net_debt_change_pct"]["mean"] if "net_debt_change_pct" in specs else a.net_debt_change_pct[0]
    dil = specs["share_dilution_pct"]["mean"] if "share_dilution_pct" in specs else a.share_dilution_pct[0]
    rev = a.current_revenue * (1 + g) ** a.horizon_years
    nd = a.current_net_debt + a.current_ev * nd_chg
    sh = a.shares_outstanding * (1 + dil)
    return float((rev * m * mult - nd) / sh)


def _bisect(f, lo, hi, target, iters=60):
    flo, fhi = f(lo) - target, f(hi) - target
    if flo * fhi > 0:
        return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        fm = f(mid) - target
        if abs(fm) < 1e-9:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def price_implied(a: PitchAssumptions) -> dict:
    """
    Expectations investing, inverted: the margin (holding growth and multiple
    at their modes) and the growth (holding margin and multiple) at which the
    mode path lands exactly on the current price. What the market is pricing.
    """
    spot = a.current_price
    m_star = _bisect(lambda m: mode_path_price(a, m=m), -0.5, 0.9, spot)
    g_star = _bisect(lambda g: mode_path_price(a, g=g), -0.6, 1.5, spot)
    mult_star = _bisect(lambda x: mode_path_price(a, mult=x), 0.1, 200.0, spot)
    return {
        "spot": spot,
        "implied_margin_at_mode_growth_and_multiple": m_star,
        "implied_growth_at_mode_margin_and_multiple": g_star,
        "implied_multiple_at_mode_growth_and_margin": mult_star,
    }


def dcf_path_percentile(engine: MonteCarloEngine, dcf: dict) -> Optional[dict]:
    """Where the analyst's three-statement path sits inside the simulated distribution."""
    a = engine.a
    try:
        price = mode_path_price(a, g=float(dcf["rev_cagr"]), m=float(dcf["ebitda_margin"]),
                                mult=float(dcf["ev_ebitda_multiple"]))
    except (KeyError, TypeError, ValueError):
        return None
    prices = engine.terminal_prices
    pct = float((prices <= price).mean())
    return {"price": price, "percentile": pct, "p_at_least": 1 - pct,
            "drivers": {k: float(dcf[k]) for k in ("rev_cagr", "ebitda_margin", "ev_ebitda_multiple")}}


def reconcile(engine: MonteCarloEngine) -> dict:
    """
    What each case target implicitly assumes.

    The DCF produced the targets; the simulation produced a distribution. When
    they disagree, this shows the disagreement in driver terms: holding revenue
    growth and margin at their modes, what exit multiple does each target need,
    and holding the multiple at its mode, what margin? Those are the numbers an
    analyst has to be able to defend out loud.
    """
    a = engine.a
    g = a.rev_cagr[1]
    m = a.ebitda_margin[1]
    mult = a.ev_ebitda_multiple[1]
    nd_chg = a.net_debt_change_pct[0]
    dil = a.share_dilution_pct[0]

    terminal_rev = a.current_revenue * (1 + g) ** a.horizon_years
    terminal_net_debt = a.current_net_debt + a.current_ev * nd_chg
    terminal_shares = a.shares_outstanding * (1 + dil)
    prices = engine.terminal_prices
    n = len(prices)

    def implied(target):
        needed_ev = target * terminal_shares + terminal_net_debt
        needed_ebitda_at_mode_mult = needed_ev / mult if mult else None
        return {
            "target": target,
            "implied_multiple_at_mode_margin": (
                needed_ev / (terminal_rev * m) if terminal_rev * m > 0 else None
            ),
            "implied_margin_at_mode_multiple": (
                needed_ebitda_at_mode_mult / terminal_rev
                if needed_ebitda_at_mode_mult is not None and terminal_rev > 0 else None
            ),
            "p_at_least": float((prices >= target).sum() / n),
            "vs_spot": target / a.current_price - 1,
        }

    mode_path_price = (terminal_rev * m * mult - terminal_net_debt) / terminal_shares
    median = float(np.median(prices))
    return {
        "mode_path_price": float(mode_path_price),
        "median_price": median,
        "base_gap_pct": median / a.base_price - 1,
        "cases": {
            "bear": implied(a.bear_price),
            "base": implied(a.base_price),
            "bull": implied(a.bull_price),
        },
        "mode_drivers": {
            "rev_cagr": g, "ebitda_margin": m, "ev_ebitda_multiple": mult,
            "net_debt_change_pct": nd_chg, "share_dilution_pct": dil,
        },
    }


def summarize(engine: MonteCarloEngine) -> dict:
    """Aggregate the price vector into wire-sized chart series."""
    a = engine.a
    prices = engine.terminal_prices
    stats_out = engine.compute_case_probabilities()
    n = len(prices)

    # Sampling error on every probability, so precision is not mistaken for accuracy
    stats_out["se"] = {
        k: _se(stats_out[k], n)
        for k in ("below_bear", "bear_to_base", "base_to_bull", "above_bull",
                  "p_at_least_bear", "p_at_least_base", "p_at_least_bull")
    }
    h = a.horizon_years
    stats_out["median_return"] = stats_out["median_price"] / a.current_price - 1
    stats_out["median_return_annualized"] = (
        (stats_out["median_price"] / a.current_price) ** (1 / h) - 1 if h > 0 else None
    )
    stats_out["mean_return_annualized"] = (
        (stats_out["mean_price"] / a.current_price) ** (1 / h) - 1 if h > 0 else None
    )

    # Histogram. mirrors the matplotlib panel (density, clipped at P99.5)
    upper = float(np.percentile(prices, 99.5))
    if upper <= 0:
        upper = max(float(prices.max()), 1.0)
    edges = np.linspace(0.0, upper, HIST_BINS + 1)
    density, _ = np.histogram(prices, bins=edges, density=True)

    # CDF. downsample the sorted vector to a fixed number of points
    sorted_prices = np.sort(prices)
    idx = np.linspace(0, len(sorted_prices) - 1, CDF_POINTS).astype(int)
    cdf_x = sorted_prices[idx]
    cdf_y = (idx + 1) / len(sorted_prices)

    sensitivity = {
        name: engine.sensitivity_analysis(name, n_buckets=5)
        for name in engine.driver_samples
    }

    return {
        "stats": stats_out,
        "histogram": {
            "edges": [round(float(e), 4) for e in edges],
            "density": [round(float(d), 8) for d in density],
        },
        "cdf": {
            "x": [round(float(v), 4) for v in cdf_x],
            "y": [round(float(v), 6) for v in cdf_y],
        },
        "tornado": engine.tornado(),
        "sensitivity": sensitivity,
        "reconciliation": reconcile(engine),
        "price_implied": price_implied(a),
        "fcf_bridge": _fcf_stats(engine),
        "meta": {
            "ticker": a.ticker,
            "company_name": a.company_name,
            "current_price": a.current_price,
            "bear_price": a.bear_price,
            "base_price": a.base_price,
            "bull_price": a.bull_price,
            "n_simulations": a.n_simulations,
            "horizon_years": a.horizon_years,
            "random_seed": a.random_seed,
            "copula": a.copula,
            "copula_df": a.copula_df if a.copula == "t" else None,
            "driver_quantiles": _driver_quantiles(a),
        },
    }


def run_simulation(payload: dict) -> dict:
    started = time.perf_counter()
    assumptions = build_assumptions(payload)
    engine = MonteCarloEngine(assumptions)
    engine.run()
    result = summarize(engine)
    if isinstance(payload.get("dcf_path"), dict):
        result["dcf_path"] = dcf_path_percentile(engine, payload["dcf_path"])
    result["meta"]["mechanisms"] = {k: bool(getattr(assumptions, k)) for k in
                                    ("cost_model", "backlog", "fcf_bridge", "justified_multiple")}
    result["meta"]["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result
