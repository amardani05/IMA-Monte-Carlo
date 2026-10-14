"""
Share price history.

Source is Yahoo Finance, the same feed yfinance wraps. The deployed app talks to
the chart endpoint directly so the function bundle does not carry pandas; the
backtest and tools use yfinance when it is installed because it handles
cookies and retries better for long histories. Both return the same shape:
a list of (ISO date, close) tuples, oldest first, dividend-unadjusted so that
EV arithmetic uses the price that actually traded.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from bisect import bisect_right
from datetime import date, datetime, timedelta
from typing import Optional

CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range={rng}&interval=1d&events=div"
try:
    import yfinance as _yf  # noqa: F401
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
HTTP_TIMEOUT = 12.0
_CACHE: dict = {}
CACHE_TTL = 6 * 3600


class PriceError(Exception):
    """Raised when no usable price series can be fetched."""


def _fetch_chart(symbol: str, rng: str, retries: int = 3) -> list:
    url = CHART_URL.format(symbol=symbol.upper(), rng=rng)
    last_exc = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            result = (data.get("chart") or {}).get("result")
            if not result:
                err = (data.get("chart") or {}).get("error") or {}
                raise PriceError(err.get("description") or f"No chart data for {symbol}")
            r = result[0]
            stamps = r.get("timestamp") or []
            closes = ((r.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
            out = []
            for ts, c in zip(stamps, closes):
                if c is None:
                    continue
                d = datetime.utcfromtimestamp(ts).date().isoformat()
                out.append((d, float(c)))
            return out
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                if attempt + 1 < retries:
                    time.sleep(1.5 * (attempt + 1))
                continue
            if exc.code == 404:
                raise PriceError(f"{symbol} is not a known Yahoo Finance symbol")
            raise PriceError(f"Price request failed ({exc.code})")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            time.sleep(0.5)
    raise PriceError(f"Could not reach Yahoo Finance: {last_exc}")


def _fetch_yfinance(symbol: str, years: int) -> Optional[list]:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        return None
    try:
        h = yf.Ticker(symbol).history(period=f"{years}y", interval="1d", auto_adjust=False)
    except Exception:
        return None
    if h is None or h.empty:
        return None
    return [(idx.date().isoformat(), float(c)) for idx, c in h["Close"].items() if c == c]


BUNDLE_DIR = os.path.join(os.path.dirname(__file__), "data", "prices")
BUNDLE_MAX_AGE_DAYS = 7


def bundled_history(symbol: str) -> Optional[list]:
    """
    A price series shipped with the app (tools/refresh_prices.py). The deployed
    function cannot rely on Yahoo, which rate-limits its egress; a weekly
    refresh committed to the repository gives it year-end closes for the
    multiple history and weekly closes for beta without a network call.
    Returns None when absent or older than BUNDLE_MAX_AGE_DAYS; the caller then
    tries the network and falls back to a stale bundle last.
    """
    path = os.path.join(BUNDLE_DIR, f"{symbol.upper()}.json")
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    series = [(d, float(c)) for d, c in doc.get("closes", [])]
    if not series:
        return None
    age = (date.today() - date.fromisoformat(doc.get("as_of", "1970-01-01"))).days
    bundled_history.last_age = age
    return series


def price_history(symbol: str, years: int = 10, prefer_yfinance: bool = True) -> list:
    """
    [(ISO date, close)], oldest first. Cached for six hours.

    yfinance is tried first when installed: it carries a session cookie and
    crumb, which the bare chart endpoint lacks, and Yahoo rate-limits bare
    bursts at 429. The deployed function has no yfinance and falls through.
    """
    key = (symbol.upper(), years)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    bundle = bundled_history(symbol)
    if bundle and getattr(bundled_history, "last_age", 999) <= BUNDLE_MAX_AGE_DAYS:
        _CACHE[key] = (time.time(), bundle)
        return bundle
    series = _fetch_yfinance(symbol, years) if prefer_yfinance else None
    if not series:
        try:
            series = _fetch_chart(symbol, f"{years}y", retries=1 if not _HAS_YFINANCE else 3)
        except PriceError:
            if bundle:
                _CACHE[key] = (time.time(), bundle)
                return bundle          # stale is better than nothing, and says so via as_of
            raise
    if not series:
        raise PriceError(f"No price history returned for {symbol}")
    series.sort()
    _CACHE[key] = (time.time(), series)
    return series


def latest_price(symbol: str) -> Optional[dict]:
    """
    Most recent close. yfinance first when installed; otherwise one bare
    attempt at the chart endpoint with no retry sleeps, because a 429 here
    must cost milliseconds, not the nine seconds a backoff loop would.
    """
    series = _fetch_yfinance(symbol, 1)
    source = "Yahoo Finance"
    if not series:
        try:
            series = _fetch_chart(symbol, "5d", retries=1)
        except PriceError:
            series = None
    if not series:
        series = bundled_history(symbol)
        source = "bundled close (refresh with tools/refresh_prices.py)"
    if not series:
        return None
    d, c = series[-1]
    return {"value": c, "source": source, "as_of": d}


def close_on(series: list, as_of: str) -> Optional[tuple]:
    """Last close on or before as_of. Returns (date, close) or None."""
    dates = [d for d, _ in series]
    i = bisect_right(dates, as_of)
    if i == 0:
        return None
    return series[i - 1]


def forward_return_window(series: list, start: str, years: float) -> Optional[tuple]:
    """(start_close, end_close, end_date) for a horizon from start, or None if end is in the future."""
    s = close_on(series, start)
    if not s:
        return None
    end_date = (date.fromisoformat(start) + timedelta(days=int(round(365.25 * years)))).isoformat()
    if end_date > series[-1][0]:
        return None
    e = close_on(series, end_date)
    if not e:
        return None
    return s[1], e[1], e[0]


def beta(symbol: str, benchmark: str = "SPY", years: int = 5) -> Optional[dict]:
    """
    Regression beta of weekly log returns against the benchmark.

    Weekly rather than daily to dampen asynchronous-close noise; five years is
    the usual window. Returns None when fewer than 52 overlapping weeks exist.
    """
    try:
        a = dict(price_history(symbol, years))
        b = dict(price_history(benchmark, years))
    except PriceError:
        return None
    import math
    common = sorted(set(a) & set(b))
    weekly = [d for i, d in enumerate(common) if i % 5 == 0]
    if len(weekly) < 53:
        return None
    ra = [math.log(a[weekly[i]] / a[weekly[i - 1]]) for i in range(1, len(weekly))]
    rb = [math.log(b[weekly[i]] / b[weekly[i - 1]]) for i in range(1, len(weekly))]
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (n - 1)
    var = sum((y - mb) ** 2 for y in rb) / (n - 1)
    if var <= 0:
        return None
    return {"beta": cov / var, "weeks": n, "benchmark": benchmark}


def cost_of_equity(symbol: str, rf: float = None, erp: float = None) -> Optional[dict]:
    """
    CAPM cost of equity. Risk-free and ERP are explicit inputs with stated
    defaults (10-year Treasury near 4.2%, ERP 4.5%), overridable via RISK_FREE
    and EQUITY_RISK_PREMIUM environment variables.
    """
    rf = float(os.environ.get("RISK_FREE", 0.042)) if rf is None else rf
    erp = float(os.environ.get("EQUITY_RISK_PREMIUM", 0.045)) if erp is None else erp
    b = beta(symbol)
    if not b:
        return None
    return {"cost_of_equity": rf + b["beta"] * erp, "beta": b["beta"], "rf": rf, "erp": erp, "weeks": b["weeks"]}
