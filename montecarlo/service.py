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


def _number_or_spec(raw, where) -> float:
    """A scalar input or a distribution spec; returns the central value (the number, or the spec's P50)."""
    from .distributions import describe
    if isinstance(raw, dict):
        try:
            return describe(raw)["p50"]
        except (KeyError, ValueError, TypeError) as exc:
            raise ValidationError(f"{where} is not a valid distribution: {exc}")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ValidationError(f"{where} must be a number or a distribution")
    if not np.isfinite(v):
        raise ValidationError(f"{where} must be a finite number")
    return v


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
        central = {}
        for k in ("roic", "wacc", "fade"):
            if k == "fade" and jm.get(k) is None:
                continue
            central[k] = _number_or_spec(jm.get(k), f"justified_multiple.{k}")
        for k in ("roic", "wacc"):
            if central[k] <= 0:
                raise ValidationError(f"justified_multiple.{k} must be a positive number"
                                      + (" at its P50" if isinstance(jm.get(k), dict) else ""))
        if "fade" in central and not 0 <= central["fade"] <= 1:
            raise ValidationError("justified_multiple.fade must be between 0 and 1")
        if central["wacc"] <= float(jm.get("long_run_growth", 0.03)):
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


def _extra_quantiles(a: PitchAssumptions) -> dict:
    """P10/P50/P90 of the drivers added beside the base five; scalar inputs as {"value": x}."""
    from .distributions import describe
    out = {}
    if a.backlog:
        out["backlog_conversion"] = describe(a.backlog["conversion"])
        out["backlog_m"] = {"value": float(a.backlog["backlog_m"])}
    if a.cost_model:
        out["gross_margin"] = describe(a.cost_model["gross_margin"])
        out["fixed_cost_m"] = {"value": float(a.cost_model["fixed_cost_m"])}
        if a.cost_model.get("fixed_cost_growth"):
            out["fixed_cost_growth"] = describe(a.cost_model["fixed_cost_growth"])
    if a.fcf_bridge:
        out["fcf_margin"] = describe(a.fcf_bridge["fcf_margin"])
        out["ev_fcf_multiple"] = describe(a.fcf_bridge["ev_fcf_multiple"])
    if a.justified_multiple:
        jm = a.justified_multiple
        for k in ("roic", "wacc", "fade"):
            v = jm.get(k, 0.5 if k == "fade" else None)
            out[k] = describe(v) if isinstance(v, dict) else {"value": float(v)}
    return out


def _eva(a: PitchAssumptions, q: dict) -> Optional[dict]:
    """
    EVA at the central inputs: (ROIC - WACC) x invested capital. Invested
    capital is the filed figure the UI passes along with the block; without it
    only the spread is reported.
    """
    jm = a.justified_multiple
    if not jm:
        return None
    centre = lambda d: d["p50"] if "p50" in d else d["value"]  # noqa: E731
    roic, wacc = centre(q["roic"]), centre(q["wacc"])
    ic = jm.get("invested_capital_m")
    ic = float(ic) if ic is not None and np.isfinite(float(ic)) else None
    return {"roic": roic, "wacc": wacc, "spread": roic - wacc, "invested_capital_m": ic,
            "eva_m": (roic - wacc) * ic if ic is not None else None}


DENSITY_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DENSITY_POINTS = 300


def _horizon_label(years: float) -> str:
    months = years * 12
    if abs(months - round(months)) < 1e-6:
        months = int(round(months))
        return f"{months // 12}y" if months % 12 == 0 else f"{months}m"
    return f"{months:.1f}m"


def _interpolated_prices(engine: MonteCarloEngine, u: float) -> Optional[np.ndarray]:
    """
    Prices at fraction u of the horizon, from the terminal sample alone: the
    median drifts geometrically from spot and each path's log distance from the
    median shrinks with sqrt(u), the transform the path chart's cone used.
        log x_u = log spot + u log(median / spot) + sqrt(u) (log x_T - log median)
    Paths that finish at zero stay at zero. A stand-in until the engine
    simulates the intermediate dates; returns None when the median is zero.
    """
    x = engine.terminal_prices
    spot, med = engine.a.current_price, float(np.median(x))
    if u >= 1.0:
        return x
    if med <= 0 or spot <= 0:
        return None
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = np.exp(np.log(spot) + u * np.log(med / spot) + np.sqrt(u) * (np.log(x[pos]) - np.log(med)))
    return out


def _log_kde_per_dollar(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Gaussian KDE in log space with Silverman's bandwidth, binned (2,048 bins)
    and convolved so 100,000 paths cost milliseconds, then read back per
    dollar: f(x) = f_log(log x) / x. Paths at zero carry no density, so the
    curve integrates to the share of paths above zero.
    """
    pos = x[x > 0]
    if len(pos) < 2:
        return np.zeros_like(grid)
    lx = np.log(pos)
    q75, q25 = np.percentile(lx, [75, 25])
    spread = min(float(np.std(lx)), (q75 - q25) / 1.34) if q75 > q25 else float(np.std(lx))
    h = 0.9 * spread * len(lx) ** -0.2 if spread > 0 else 1e-3
    g = np.log(grid)
    lo, hi = g[0] - 5 * h, g[-1] + 5 * h
    counts, edges = np.histogram(lx, bins=2048, range=(lo, hi))
    centres, dx = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    half = int(np.ceil(4 * h / dx))
    kern = np.exp(-0.5 * (np.arange(-half, half + 1) * dx / h) ** 2)
    smooth = np.convolve(counts, kern / kern.sum(), mode="same") / (len(x) * dx)
    return np.interp(g, centres, smooth) / grid


def horizon_densities(engine: MonteCarloEngine, fractions=DENSITY_FRACTIONS, points: int = DENSITY_POINTS) -> Optional[dict]:
    """
    Smoothed price densities at fractions of the horizon on one price grid and
    one per-dollar scale, with each horizon's median and P(at least) each target.
    The intermediate horizons are interpolated from the terminal run (see
    `_interpolated_prices`), and the result says so.
    """
    a, T = engine.a, engine.a.horizon_years
    samples = [(u, _interpolated_prices(engine, u)) for u in fractions]
    if any(x is None for _, x in samples):
        return None
    lows = [np.percentile(x[x > 0], 0.5) for _, x in samples if (x > 0).sum() > 1]
    highs = [np.percentile(x, 99.5) for _, x in samples]
    if not lows:
        return None
    # past the extreme quantiles so every curve has thinned out at both ends of the grid
    grid = np.linspace(max(0.6 * min(lows), 1e-6), 1.05 * max(highs), points)
    horizons = []
    for u, x in samples:
        dens = _log_kde_per_dollar(x, grid)
        horizons.append({
            "fraction": u, "years": u * T, "label": _horizon_label(u * T), "interpolated": u < 1.0,
            "median": float(np.median(x)),
            "p_at_least_bear": float(np.mean(x >= a.bear_price)),
            "p_at_least_base": float(np.mean(x >= a.base_price)),
            "p_at_least_bull": float(np.mean(x >= a.bull_price)),
            "density": [float(f"{v:.6g}") for v in dens],
        })
    inter = [h["label"] for h in horizons if h["interpolated"]]
    t_label = f"{T:g}-year"
    note = (f"{inter[0]} to {inter[-1]} interpolated from the {t_label} run" if len(inter) > 1
            else f"{inter[0]} interpolated from the {t_label} run" if inter else "")
    return {"grid": [round(float(v), 4) for v in grid], "horizons": horizons, "note": note}


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
    histogram = {"edges": [round(float(e), 4) for e in edges], "density": [round(float(d), 8) for d in density]}
    if engine.terminal_prices_fcf is not None:
        # same bins, normalised over all FCF paths so the two overlay on one axis
        counts, _ = np.histogram(engine.terminal_prices_fcf, bins=edges)
        histogram["density_fcf"] = [round(float(c / (n * (edges[1] - edges[0]))), 8) for c in counts]

    # CDF. downsample the sorted vector to a fixed number of points
    sorted_prices = np.sort(prices)
    idx = np.linspace(0, len(sorted_prices) - 1, CDF_POINTS).astype(int)
    cdf_x = sorted_prices[idx]
    cdf_y = (idx + 1) / len(sorted_prices)

    sensitivity = {
        name: engine.sensitivity_analysis(name, n_buckets=5)
        for name in engine.driver_samples
    }

    extra_q = _extra_quantiles(a)
    return {
        "stats": stats_out,
        "histogram": histogram,
        "cdf": {
            "x": [round(float(v), 4) for v in cdf_x],
            "y": [round(float(v), 6) for v in cdf_y],
        },
        # conditional: median price among paths in the driver's bottom and top decile
        "tornado": engine.tornado(),
        # one at a time: the driver held at its P10, then its P90, every other draw unchanged
        "swing": engine.swing_tornado(),
        # the path chart's curves: smoothed densities at quarters of the horizon
        "horizon_densities": horizon_densities(engine),
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
            "extra_quantiles": extra_q,
            "eva": _eva(a, extra_q),
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


# ─────────────────────────────────────────────────────────────────────────────
# Drivers beyond the base five, proposed for the auto-fill
# ─────────────────────────────────────────────────────────────────────────────
ROIC_WIDTH = 0.25           # P10 and P90 at 0.75x and 1.25x the filed ROIC: a stated width, not a filed one
WACC_WIDTH = 0.01           # P10 and P90 one point either side of the computed WACC
FADE_QUANTILES = (0.25, 0.50, 0.75)
BACKLOG_CONVERSION = {"type": "metalog", "p10": 0.55, "p50": 0.70, "p90": 0.85}


def _metalog(p10, p50, p90, kind="metalog"):
    spec = {"type": kind, "p10": round(float(p10), 5), "p50": round(float(p50), 5), "p90": round(float(p90), 5)}
    if kind == "log_metalog":
        spec["lower"] = 0.0
    return spec


def propose_extra_drivers(mechanisms: Optional[dict]) -> dict:
    """
    The four mechanisms recast as drivers the analyst adds beside the base
    five, every uncertain quantity a P10/P50/P90.

    Values come from `marketdata.mechanisms`: filed figures, arithmetic over
    them, or the stated assumptions named in each note. A group the filings
    cannot support is still offered, with the missing inputs left as None for
    the analyst to enter; `filed` says which case it is. Nothing is switched on.
    """
    m = mechanisms or {}
    avail, notes = m.get("available") or {}, m.get("notes") or {}
    out = {}

    bl = m.get("backlog") or {}
    out["backlog"] = {
        "filed": bool(avail.get("backlog") and bl), "note": notes.get("backlog", ""),
        "backlog_m": bl.get("backlog_m"), "conversion": bl.get("conversion") or dict(BACKLOG_CONVERSION),
    }

    fb = m.get("fcf_bridge") or {}
    out["fcf_bridge"] = {
        "filed": bool(avail.get("fcf_bridge") and fb), "note": notes.get("fcf_bridge", ""),
        "fcf_margin": fb.get("fcf_margin"), "ev_fcf_multiple": fb.get("ev_fcf_multiple"),
    }

    cm = m.get("cost_model") or {}
    out["cost_model"] = {
        "filed": bool(avail.get("cost_model") and cm), "note": notes.get("cost_model", ""),
        "gross_margin": cm.get("gross_margin"), "fixed_cost_m": cm.get("fixed_cost_m"),
        "fixed_cost_growth": cm.get("fixed_cost_growth") or {"type": "normal", "mean": 0.03, "std": 0.03},
    }

    jm = m.get("justified_multiple") or {}
    rc = m.get("returns") or {}
    roic, wacc = jm.get("roic"), jm.get("wacc")
    width_note = []
    roic_spec = None
    if roic is not None and roic > 0:
        roic_spec = _metalog(roic * (1 - ROIC_WIDTH), roic, roic * (1 + ROIC_WIDTH))
        width_note.append(f"ROIC P10 and P90 are {1 - ROIC_WIDTH:.2f}x and {1 + ROIC_WIDTH:.2f}x the filed {roic * 100:.1f}%")
    wacc_spec = None
    if wacc is not None:
        wacc_spec = _metalog(wacc - WACC_WIDTH, wacc, wacc + WACC_WIDTH)
        width_note.append(f"WACC P10 and P90 are {WACC_WIDTH * 100:.0f} point either side")
    fade = min(max(float(jm.get("fade", FADE_QUANTILES[1])), 0.05), 0.95)
    if abs(fade - FADE_QUANTILES[1]) < 1e-12:
        fade_spec = _metalog(*FADE_QUANTILES)
    else:
        fade_spec = _metalog(max(fade - 0.25, fade / 2), fade, min(fade + 0.25, (1 + fade) / 2))
    width_note.append(f"growth fade {fade_spec['p10']:.2f} / {fade_spec['p50']:.2f} / {fade_spec['p90']:.2f}")
    out["justified_multiple"] = {
        "filed": bool(avail.get("justified_multiple") and jm), "note": notes.get("justified_multiple", ""),
        "roic": roic_spec, "wacc": wacc_spec, "fade": fade_spec,
        "tax_rate": jm.get("tax_rate"), "da_share": jm.get("da_share"),
        "long_run_growth": jm.get("long_run_growth", 0.03), "premium_reversion": jm.get("premium_reversion", 0.25),
        "current_growth": jm.get("current_growth"), "current_multiple": jm.get("current_multiple"),
        "filed_roic": rc.get("roic"), "invested_capital_m": rc.get("invested_capital_m"),
        "nopat_m": rc.get("nopat_m"), "eva_m": rc.get("eva_m"),
        "width_note": "Stated widths, not filed: " + "; ".join(width_note) + ". Set them to your view.",
    }
    return out
