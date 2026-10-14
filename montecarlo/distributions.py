"""
Quantile-parameterised distributions for the drivers.

The triangular distribution is kept for backward compatibility. The default
for new work is Keelin's metalog (Decision Analysis, 2016): it is specified by
the quantiles an analyst can actually defend (P10, P50, P90), has a closed-form
inverse CDF, and does not put equal weight on the two extremes the way a
triangular does. Everything here is pure numpy; the copula sampler feeds a
uniform in and gets a driver value out, so the engine is unchanged.
"""

from __future__ import annotations

import math

import numpy as np

# Keelin (2016) feasibility bound for the three-term metalog: |a3| / a2 < 1.66711
_FEAS = 1.66711
_EPS = 1e-6


def _logit(y: np.ndarray) -> np.ndarray:
    return np.log(y / (1.0 - y))


def fit_metalog3(p10: float, p50: float, p90: float) -> np.ndarray:
    """
    Coefficients (a1, a2, a3) of the three-term metalog through the three
    quantiles. If the triple is infeasible (too asymmetric for three terms),
    the skew term is capped at the feasibility bound and the median is kept,
    so the result is the nearest feasible distribution rather than an error.
    """
    if not (p10 < p50 < p90):
        raise ValueError("Quantiles must satisfy p10 < p50 < p90")
    ys = np.array([0.1, 0.5, 0.9])
    L = _logit(ys)
    A = np.column_stack([np.ones(3), L, (ys - 0.5) * L])
    a = np.linalg.solve(A, np.array([p10, p50, p90], dtype=float))
    a1, a2, a3 = a
    if a2 <= 0 or abs(a3) / a2 >= _FEAS:
        # cap the skew term, keep the median, refit a2 to the wider side
        sign = 1.0 if a3 >= 0 else -1.0
        half = max(p90 - p50, p50 - p10)
        L90 = float(L[2])
        a2 = half / (L90 * (1 + 0.4 * (_FEAS - _EPS) * 0 + 0.4 * (_FEAS - _EPS) / 2))
        a2 = max(a2, _EPS)
        a3 = sign * a2 * (_FEAS - 1e-3)
        a1 = p50
    return np.array([a1, a2, a3], dtype=float)


TAIL_CLIP = 0.005   # beyond P0.5 and P99.5 the quantile function is held flat


def metalog3_ppf(u: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Inverse CDF of the three-term metalog, with u held inside [0.5%, 99.5%].

    The logit tails of a three-term metalog extrapolate far past the assessed
    P10 and P90; nobody stated a P0.1, so the sampler does not invent one.
    """
    u = np.clip(np.asarray(u, dtype=float), TAIL_CLIP, 1 - TAIL_CLIP)
    L = _logit(u)
    return a[0] + a[1] * L + a[2] * (u - 0.5) * L


def fit_log_metalog3(p10: float, p50: float, p90: float, lower: float = 0.0) -> tuple:
    """Metalog on log(x - lower): a positive quantity such as a multiple."""
    if p10 <= lower:
        raise ValueError("p10 must exceed the lower bound")
    a = fit_metalog3(math.log(p10 - lower), math.log(p50 - lower), math.log(p90 - lower))
    return a, lower


def log_metalog3_ppf(u: np.ndarray, a: np.ndarray, lower: float = 0.0) -> np.ndarray:
    return lower + np.exp(metalog3_ppf(u, a))


def triangular_ppf(u: np.ndarray, low: float, mode: float, high: float) -> np.ndarray:
    """Inverse CDF of the triangular distribution (the legacy sampler)."""
    c = (mode - low) / (high - low)
    return np.where(
        u < c,
        low + np.sqrt(u * (high - low) * (mode - low)),
        high - np.sqrt((1 - u) * (high - low) * (high - mode)),
    )


def triangular_quantiles(low: float, mode: float, high: float) -> tuple:
    """(p10, p50, p90) of a triangular, for converting legacy inputs."""
    u = np.array([0.1, 0.5, 0.9])
    q = triangular_ppf(u, low, mode, high)
    return float(q[0]), float(q[1]), float(q[2])


def make_sampler(spec):
    """
    Turn a driver specification into a function u -> values.

    Accepted forms:
      (low, mode, high)                       triangular (legacy)
      {"type": "triangular", "low", "mode", "high"}
      {"type": "metalog", "p10", "p50", "p90"}             unbounded
      {"type": "log_metalog", "p10", "p50", "p90", "lower"} bounded below
      {"type": "normal", "mean", "std", "clip_low", "clip_high"}
    """
    from scipy import stats

    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        low, mode, high = (float(v) for v in spec)
        return lambda u: triangular_ppf(u, low, mode, high)
    if not isinstance(spec, dict):
        raise ValueError(f"Unrecognised driver spec: {spec!r}")
    t = spec.get("type", "triangular")
    if t == "triangular":
        low, mode, high = float(spec["low"]), float(spec["mode"]), float(spec["high"])
        return lambda u: triangular_ppf(u, low, mode, high)
    if t == "metalog":
        a = fit_metalog3(float(spec["p10"]), float(spec["p50"]), float(spec["p90"]))
        return lambda u: metalog3_ppf(u, a)
    if t == "log_metalog":
        a, lower = fit_log_metalog3(float(spec["p10"]), float(spec["p50"]), float(spec["p90"]),
                                    float(spec.get("lower", 0.0)))
        return lambda u: log_metalog3_ppf(u, a, lower)
    if t == "normal":
        mean, std = float(spec["mean"]), float(spec["std"])
        lo, hi = spec.get("clip_low"), spec.get("clip_high")

        def f(u):
            x = stats.norm.ppf(np.clip(u, 1e-9, 1 - 1e-9), loc=mean, scale=std)
            if lo is not None:
                x = np.maximum(x, lo)
            if hi is not None:
                x = np.minimum(x, hi)
            return x
        return f
    raise ValueError(f"Unknown distribution type: {t}")


def describe(spec) -> dict:
    """The P10/P50/P90 of any spec, for display."""
    f = make_sampler(spec)
    q = f(np.array([0.1, 0.5, 0.9]))
    return {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}
