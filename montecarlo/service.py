"""
Request layer for the simulation API. Transport agnostic.

Validates an incoming assumption payload, runs the Monte Carlo engine, and
returns summary stats plus pre-aggregated chart series. The raw price vector is
never sent over the wire: only a histogram, a downsampled CDF, tornado bounds
and scalar statistics, which keeps a 100k-path run at roughly 13 KB.
"""

import time

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
    )


def _se(p: float, n: int) -> float:
    """Monte Carlo standard error of a probability estimate."""
    return (max(p * (1 - p), 0.0) / n) ** 0.5


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
        },
    }


def run_simulation(payload: dict) -> dict:
    started = time.perf_counter()
    assumptions = build_assumptions(payload)
    engine = MonteCarloEngine(assumptions)
    engine.run()
    result = summarize(engine)
    result["meta"]["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result
