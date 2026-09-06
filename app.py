"""
Vercel entrypoint.

Serves the static front end out of public/ and exposes the simulation engine at
POST /api/simulate. Flask owns transport only; validation and the model live in
montecarlo/service.py, so the local CLI (MonteCarlo.py) runs identical code.
"""

from flask import Flask, jsonify, request, send_from_directory

from montecarlo.marketdata import DataError, build_prefill
from montecarlo.service import ValidationError, run_simulation

MAX_BODY_BYTES = 1_000_000

app = Flask(__name__, static_folder="public", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(ok=True, endpoint="POST /api/simulate")


@app.get("/api/company/<ticker>")
def company(ticker: str):
    """
    Auto-fill a pitch from SEC filings.

    Optional `price` (a hand-entered quote) and `horizon` (years, which sets the
    CAGR window used for calibration) refine the result.
    """
    price = request.args.get("price", type=float)
    if price is not None and (price <= 0 or price > 1_000_000):
        return jsonify(error="Price must be a positive number"), 400

    horizon = request.args.get("horizon", default=2.0, type=float)
    if not 0 < horizon <= 10:
        return jsonify(error="Horizon must be between 0 and 10 years"), 400

    try:
        result = build_prefill(ticker, price=price, horizon_years=horizon)
    except DataError as exc:
        return jsonify(error=str(exc)), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"Lookup failed: {type(exc).__name__}: {exc}"), 502

    response = jsonify(result)
    response.headers["Cache-Control"] = "public, max-age=1800"
    return response


@app.post("/api/simulate")
def simulate():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object"), 400

    try:
        result = run_simulation(payload)
    except ValidationError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:  # noqa: BLE001, surface engine failures as 500s
        return jsonify(error=f"Simulation failed: {type(exc).__name__}: {exc}"), 500

    response = jsonify(result)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="Not found"), 404
