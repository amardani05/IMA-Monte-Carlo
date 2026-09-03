"""
Vercel serverless function: POST /api/simulate

Accepts a pitch assumption payload, runs the Monte Carlo engine, and returns
summary stats plus pre-aggregated chart series. The raw price vector is never
sent over the wire — only a histogram, a downsampled CDF, and tornado bounds.
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from montecarlo import PitchAssumptions, MonteCarloEngine

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
                "correlation_matrix is not positive definite — the copula cannot be built from it"
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


def summarize(engine: MonteCarloEngine) -> dict:
    """Aggregate the price vector into wire-sized chart series."""
    a = engine.a
    prices = engine.terminal_prices
    stats_out = engine.compute_case_probabilities()

    # Histogram — mirrors the matplotlib panel (density, clipped at P99.5)
    upper = float(np.percentile(prices, 99.5))
    if upper <= 0:
        upper = max(float(prices.max()), 1.0)
    edges = np.linspace(0.0, upper, HIST_BINS + 1)
    density, _ = np.histogram(prices, bins=edges, density=True)

    # CDF — downsample the sorted vector to a fixed number of points
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


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._send(200, {"ok": True, "endpoint": "POST /api/simulate"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 1_000_000:
            self._send(400, {"error": "Request body must be a JSON object under 1MB"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(400, {"error": "Request body is not valid JSON"})
            return

        if not isinstance(payload, dict):
            self._send(400, {"error": "Request body must be a JSON object"})
            return

        try:
            self._send(200, run_simulation(payload))
        except ValidationError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 — surface engine failures as 500s
            self._send(500, {"error": f"Simulation failed: {type(exc).__name__}: {exc}"})

    def log_message(self, *args):  # silence default stderr access logging
        pass
