"""
Deterministic market-data layer.

Pulls company fundamentals straight from SEC EDGAR's XBRL company-facts API and
derives a complete set of pitch assumptions from filed history. There is no
model, no inference service and no third-party analytics in this path — every
number returned is either a figure the company filed with the SEC or plain
arithmetic over those figures, and each one is returned with its source and
as-of date so it can be checked against the filing.

The one figure EDGAR cannot supply is a live share price. If a quote provider is
configured (see PRICE_PROVIDERS) it is used; otherwise price is returned as
null and the caller supplies it. Everything else auto-fills either way.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

# SEC's fair-access policy asks for a descriptive User-Agent. data.sec.gov (the
# hot path) accepts this as-is; www.sec.gov additionally demands an email-style
# contact and is only consulted when the bundled ticker map misses. Set
# SEC_USER_AGENT to "Your Name your@email.com" to enable that fallback.
DEFAULT_UA = "IMA Monte Carlo scenario analysis tool"
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", DEFAULT_UA)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# An EV/EBITDA bridge is meaningless for balance-sheet businesses: banks book
# interest as revenue, so "EBITDA margin" comes out near 100% and the model
# produces confident nonsense. Reject by SIC rather than let that through.
FINANCIALS_SIC = (6020, 6411)
REIT_SIC = (6500, 6799)

HTTP_TIMEOUT = 12.0
TICKER_MAP_TTL = 24 * 3600
FACTS_TTL = 6 * 3600

# XBRL tags in priority order. Filers tag the same economics differently, and
# a tag can be present but sparse, so candidates are merged rather than the
# first hit winning outright — see _merged_series.
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueServicesNet",
]
OPERATING_INCOME_TAGS = ["OperatingIncomeLoss"]
GROSS_PROFIT_TAGS = ["GrossProfit"]
OPERATING_EXPENSE_TAGS = ["OperatingExpenses", "CostsAndExpenses"]
PRETAX_INCOME_TAGS = [
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
]
INTEREST_EXPENSE_TAGS = ["InterestExpense", "InterestExpenseDebt", "InterestIncomeExpenseNet"]
DA_TAGS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "DepreciationAndAmortization",
    "Depreciation",
]
CASH_TAGS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
]
DEBT_COMPONENT_TAGS = [
    ["LongTermDebtNoncurrent", "LongTermDebtCurrent"],
    ["LongTermDebtNoncurrent", "ShortTermBorrowings"],
    ["LongTermDebtAndCapitalLeaseObligations", "ShortTermBorrowings"],
    ["LongTermDebtAndCapitalLeaseObligationsCurrent", "LongTermDebtAndCapitalLeaseObligations"],
    ["LongTermDebt"],
    ["DebtLongtermAndShorttermCombinedAmount"],
    ["LongTermDebtNoncurrent"],
]
SHARES_TAGS = ["EntityCommonStockSharesOutstanding"]

_CACHE: dict = {}


class DataError(Exception):
    """Raised when a ticker cannot be resolved or its filings lack what we need."""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────
def _get_json(url: str, ttl: float) -> dict:
    hit = _CACHE.get(url)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": SEC_USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip

                raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise DataError("SEC has no filings at that address")
        raise DataError(f"SEC request failed ({exc.code})")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DataError(f"Could not reach SEC EDGAR: {exc}")
    except json.JSONDecodeError:
        raise DataError("SEC returned a malformed response")

    _CACHE[url] = (time.time(), data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Ticker → CIK
# ─────────────────────────────────────────────────────────────────────────────
_BUNDLED_MAP_PATH = os.path.join(os.path.dirname(__file__), "data", "sec_tickers.json")


def _bundled_map() -> dict:
    """The ticker->CIK map shipped with the app. Refresh with tools/refresh_tickers.py."""
    cached = _CACHE.get("__bundled__")
    if cached is None:
        try:
            with open(_BUNDLED_MAP_PATH) as fh:
                cached = json.load(fh)
        except (OSError, json.JSONDecodeError):
            cached = {}
        _CACHE["__bundled__"] = cached
    return cached


def resolve_cik(ticker: str) -> tuple[int, str]:
    """
    Ticker -> (CIK, registrant name).

    Resolves against the bundled map first so the common case costs no network
    call at all. www.sec.gov rate-limits aggressively and rejects User-Agents
    without contact details, so it is only consulted for symbols the bundled
    map has not seen — a recent listing, typically.
    """
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", ticker):
        raise DataError(f"'{ticker}' is not a valid ticker symbol")

    hit = _bundled_map().get(ticker)
    if hit:
        return int(hit[0]), hit[1] or ticker

    try:
        data = _get_json(TICKER_MAP_URL, TICKER_MAP_TTL)
    except DataError:
        raise DataError(
            f"{ticker} is not in the bundled SEC ticker list, and EDGAR could not "
            "be reached to check for a recent listing"
        )
    for row in data.values():
        if str(row.get("ticker", "")).upper() == ticker:
            return int(row["cik_str"]), row.get("title", ticker)
    raise DataError(
        f"{ticker} is not an SEC registrant — foreign-listed names, ADRs and "
        "private companies do not file XBRL with EDGAR"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Series extraction
# ─────────────────────────────────────────────────────────────────────────────
_ANNUAL = re.compile(r"^CY(\d{4})$")
_QUARTER = re.compile(r"^CY(\d{4})Q(\d)$")


def _tag_frames(facts: dict, tag: str, taxonomy: str = "us-gaap", unit: str = "USD") -> dict:
    """{frame_label: value} using EDGAR's normalized calendar frames."""
    node = facts.get("facts", {}).get(taxonomy, {}).get(tag)
    if not node:
        return {}
    out = {}
    for unit_name, rows in node.get("units", {}).items():
        if unit_name != unit:
            continue
        for row in rows:
            frame = row.get("frame")
            if frame:
                out[frame] = row["val"]
    return out


def _merged_series(facts: dict, tags: list, pattern, unit: str = "USD") -> dict:
    """
    Merge candidate tags into one series keyed by period.

    Earlier tags win on conflict, but later tags fill periods the earlier ones
    never covered — a filer that switched tagging mid-history (MYRG moved from
    `Revenues` to `RevenueFromContractWithCustomer...`) otherwise loses years.
    """
    merged: dict = {}
    for tag in reversed(tags):
        for frame, val in _tag_frames(facts, tag, unit=unit).items():
            m = pattern.match(frame)
            if m:
                merged[frame] = val
    return merged


def annual_series(facts: dict, tags: list) -> dict:
    """{year:int -> value:float}, most reliable basis (audited full years)."""
    raw = _merged_series(facts, tags, _ANNUAL)
    return {int(_ANNUAL.match(k).group(1)): float(v) for k, v in raw.items()}


def quarterly_series(facts: dict, tags: list) -> dict:
    """{(year, quarter) -> value:float}."""
    raw = _merged_series(facts, tags, _QUARTER)
    out = {}
    for k, v in raw.items():
        m = _QUARTER.match(k)
        out[(int(m.group(1)), int(m.group(2)))] = float(v)
    return out


def _point_in_time(facts: dict, tags: list, taxonomy: str = "us-gaap", unit: str = "USD"):
    """Latest instant value for a balance-sheet concept, with its date."""
    best = None
    for tag in tags:
        node = facts.get("facts", {}).get(taxonomy, {}).get(tag)
        if not node:
            continue
        for unit_name, rows in node.get("units", {}).items():
            if unit_name != unit:
                continue
            for row in rows:
                end = row.get("end")
                if not end:
                    continue
                if best is None or end > best[0]:
                    best = (end, float(row["val"]), tag, row.get("form"), row.get("filed"))
        if best:
            break
    return best


def trailing_twelve_months(annual: dict, quarterly: dict) -> Optional[tuple]:
    """
    Roll a TTM figure: latest four consecutive quarters.

    Filers routinely omit Q4 (it is implied by the 10-K), so a missing Q4 is
    reconstructed as full year minus the three tagged quarters. Returns
    (value, label) or None when four quarters cannot be assembled.
    """
    if not quarterly:
        return None
    latest = max(quarterly)
    year, qtr = latest

    needed, y, q = [], year, qtr
    for _ in range(4):
        needed.append((y, q))
        q -= 1
        if q == 0:
            y, q = y - 1, 4

    values = []
    for (yy, qq) in needed:
        if (yy, qq) in quarterly:
            values.append(quarterly[(yy, qq)])
        elif qq == 4 and yy in annual and all((yy, k) in quarterly for k in (1, 2, 3)):
            values.append(annual[yy] - sum(quarterly[(yy, k)] for k in (1, 2, 3)))
        else:
            return None
    return sum(values), f"TTM through {year}Q{qtr}"


# ─────────────────────────────────────────────────────────────────────────────
# Share price — the one figure EDGAR cannot provide
# ─────────────────────────────────────────────────────────────────────────────
def fetch_price(ticker: str) -> Optional[dict]:
    """
    Try each configured quote provider in turn.

    Every provider needs an API key, so with none configured this returns None
    and the caller collects the price by hand. That is deliberate: the keyless
    quote feeds (Stooq, Yahoo's undocumented endpoints) either sit behind bot
    walls or rate-limit datacenter IPs, and silently serving a stale or wrong
    price is worse than asking for one number.
    """
    key = os.environ.get("FINNHUB_API_KEY")
    if key:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker.upper()}&token={key}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            price = data.get("c")
            if price:
                return {
                    "value": float(price),
                    "source": "Finnhub",
                    "as_of": "live quote",
                    "previous_close": data.get("pc"),
                }
        except Exception:
            pass  # fall through to manual entry rather than failing the lookup
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Profile
# ─────────────────────────────────────────────────────────────────────────────
FINANCE_LEASE_TAGS = ["FinanceLeaseLiability"]


def _debt(facts: dict):
    """
    Total debt from the first component group that fully resolves.

    Finance leases are added when tagged; operating leases are deliberately
    excluded, since EBITDA here is struck after operating lease expense and
    counting them both ways would double-charge the bridge.
    """
    for group in DEBT_COMPONENT_TAGS:
        parts = [_point_in_time(facts, [tag]) for tag in group]
        if all(parts):
            total = sum(part[1] for part in parts)
            as_of = max(part[0] for part in parts)
            components = {tag: part[1] / 1e6 for tag, part in zip(group, parts)}

            lease = _point_in_time(facts, FINANCE_LEASE_TAGS)
            if lease:
                total += lease[1]
                as_of = max(as_of, lease[0])
                components["FinanceLeaseLiability"] = lease[1] / 1e6

            return total, as_of, components
    return None


def _share_history(facts: dict) -> dict:
    """{year -> share count}, taking the last observation filed in each year."""
    node = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding")
    if not node:
        return {}
    by_year: dict = {}
    for rows in node.get("units", {}).values():
        for row in rows:
            end = row.get("end")
            if not end:
                continue
            year = int(end[:4])
            if year not in by_year or end > by_year[year][0]:
                by_year[year] = (end, float(row["val"]))
    return {y: v for y, (_, v) in by_year.items()}


def _share_change_stats(facts: dict) -> Optional[dict]:
    """Mean and standard deviation of year-over-year share-count change."""
    hist = _share_history(facts)
    years = sorted(hist)
    changes = [
        hist[b] / hist[a] - 1
        for a, b in zip(years, years[1:])
        if b - a == 1 and hist[a] > 0
    ]
    if len(changes) < 3:
        return None
    n = len(changes)
    mean = sum(changes) / n
    var = sum((c - mean) ** 2 for c in changes) / (n - 1)
    return {"mean": mean, "std": var ** 0.5, "n": n, "changes": changes}


def company_meta(cik: int) -> dict:
    """Registrant metadata — SIC, exchange, fiscal year end — from EDGAR submissions."""
    try:
        data = _get_json(SUBMISSIONS_URL.format(cik=cik), FACTS_TTL)
    except DataError:
        return {}
    return {
        "name": data.get("name"),
        "sic": int(data["sic"]) if str(data.get("sic", "")).isdigit() else None,
        "sic_description": data.get("sicDescription"),
        "exchanges": data.get("exchanges") or [],
        "fiscal_year_end": data.get("fiscalYearEnd"),
    }


def _check_modelable(ticker: str, meta: dict) -> list:
    """Reject industries the bridge cannot represent; warn where it is a stretch."""
    sic = meta.get("sic")
    if sic is None:
        return []
    desc = meta.get("sic_description") or "this industry"
    if FINANCIALS_SIC[0] <= sic <= FINANCIALS_SIC[1]:
        raise DataError(
            f"{ticker} is classified as {desc} (SIC {sic}). Banks, brokers and "
            "insurers book interest and premiums as revenue, so EBITDA and "
            "EV/EBITDA are not meaningful for them — this model does not fit."
        )
    if REIT_SIC[0] <= sic <= REIT_SIC[1]:
        return [
            f"{ticker} is classified as {desc} (SIC {sic}). Real-estate names are "
            "normally valued on FFO or cap rates rather than EV/EBITDA — treat the "
            "multiple with care."
        ]
    return []


def _ebitda_history(facts: dict, rev_annual: dict, da_annual: dict) -> tuple:
    """
    EBITDA per year, trying progressively looser reconstructions.

    Not every filer tags operating income — Dycom, for one, does not — so fall
    back to gross profit less operating expenses, and then to pre-tax income
    plus interest. Returns ({year: ebitda}, description of the basis used).
    """
    oi = annual_series(facts, OPERATING_INCOME_TAGS)
    if oi:
        years = set(oi) & set(da_annual) & set(rev_annual)
        if years:
            return {y: oi[y] + da_annual[y] for y in sorted(years)}, "operating income + D&A"

    gp = annual_series(facts, GROSS_PROFIT_TAGS)
    opex = annual_series(facts, OPERATING_EXPENSE_TAGS)
    if gp and opex:
        years = set(gp) & set(opex) & set(da_annual) & set(rev_annual)
        if years:
            return (
                {y: gp[y] - opex[y] + da_annual[y] for y in sorted(years)},
                "gross profit − operating expenses + D&A",
            )

    pretax = annual_series(facts, PRETAX_INCOME_TAGS)
    interest = annual_series(facts, INTEREST_EXPENSE_TAGS)
    if pretax and interest:
        years = set(pretax) & set(interest) & set(da_annual) & set(rev_annual)
        if years:
            return (
                {y: pretax[y] + interest[y] + da_annual[y] for y in sorted(years)},
                "pre-tax income + interest + D&A",
            )
    if pretax:
        years = set(pretax) & set(da_annual) & set(rev_annual)
        if years:
            return (
                {y: pretax[y] + da_annual[y] for y in sorted(years)},
                "pre-tax income + D&A (interest not tagged)",
            )

    return {}, "unavailable"


def build_profile(ticker: str, history_years: int = 10) -> dict:
    """
    Everything needed to populate a pitch, with provenance on every figure.

    Monetary values are returned in $M to match PitchAssumptions.
    """
    cik, title = resolve_cik(ticker)
    meta = company_meta(cik)
    warnings = _check_modelable(ticker.upper(), meta)
    facts = _get_json(COMPANY_FACTS_URL.format(cik=cik), FACTS_TTL)

    rev_annual = annual_series(facts, REVENUE_TAGS)
    if not rev_annual:
        raise DataError(f"No revenue figures tagged in {ticker}'s filings")
    da_annual = annual_series(facts, DA_TAGS)
    ebitda_annual, ebitda_basis = _ebitda_history(facts, rev_annual, da_annual)
    if not ebitda_annual:
        raise DataError(
            f"{ticker}'s filings do not carry the tags needed to derive EBITDA "
            "(operating income or pre-tax income, plus D&A). Banks, insurers and "
            "filers using custom taxonomies often fall into this case, and an "
            "EV/EBITDA bridge would not be the right model for them anyway."
        )

    latest_fy = max(ebitda_annual)
    cutoff = latest_fy - history_years + 1
    rev_hist = {y: v for y, v in rev_annual.items() if y >= cutoff}
    margin_hist = {
        y: ebitda_annual[y] / rev_annual[y]
        for y in ebitda_annual
        if y >= cutoff and rev_annual.get(y)
    }

    ttm = trailing_twelve_months(rev_annual, quarterly_series(facts, REVENUE_TAGS))

    shares = _point_in_time(facts, SHARES_TAGS, taxonomy="dei", unit="shares")
    if not shares:
        raise DataError(f"No share count tagged in {ticker}'s filings")
    cash = _point_in_time(facts, CASH_TAGS)
    debt = _debt(facts)

    net_debt_m = None
    net_debt_meta = None
    if cash and debt:
        net_debt_m = (debt[0] - cash[1]) / 1e6
        net_debt_meta = {
            "total_debt_m": debt[0] / 1e6,
            "cash_m": cash[1] / 1e6,
            "as_of": max(debt[1], cash[0]),
            "components": debt[2],
            "note": "total debt (incl. finance leases) less cash; operating leases excluded",
        }

    margin_now = ebitda_annual[latest_fy] / rev_annual[latest_fy]
    if not -0.5 < margin_now < 0.75:
        warnings.append(
            f"Derived EBITDA margin of {margin_now * 100:.1f}% is outside the range "
            "this bridge normally sees. Check it against the filing before relying on it."
        )

    price = fetch_price(ticker)
    display_name = meta.get("name") or title

    return {
        "ticker": ticker.upper(),
        "company_name": display_name.title() if display_name.isupper() else display_name,
        "warnings": warnings,
        "sic_description": meta.get("sic_description"),
        "exchanges": meta.get("exchanges"),
        "cik": cik,
        "filings_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik:010d}&type=10-K",
        "price": price,
        "shares_outstanding_m": shares[1] / 1e6,
        "shares_meta": {"as_of": shares[0], "form": shares[3], "filed": shares[4]},
        "revenue_fy_m": rev_annual[latest_fy] / 1e6,
        "revenue_ttm_m": (ttm[0] / 1e6) if ttm else None,
        "revenue_ttm_label": ttm[1] if ttm else None,
        "latest_fy": latest_fy,
        "ebitda_fy_m": ebitda_annual[latest_fy] / 1e6,
        "ebitda_margin_fy": ebitda_annual[latest_fy] / rev_annual[latest_fy],
        "ebitda_basis": ebitda_basis,
        "net_debt_m": net_debt_m,
        "net_debt_meta": net_debt_meta,
        "revenue_history_m": {str(y): rev_hist[y] / 1e6 for y in sorted(rev_hist)},
        "margin_history": {str(y): margin_hist[y] for y in sorted(margin_hist)},
        "share_change_stats": _share_change_stats(facts),
        "source": "SEC EDGAR XBRL company facts",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Driver calibration from filed history
# ─────────────────────────────────────────────────────────────────────────────
def _triple(values: list) -> tuple:
    """(low, mode, high) = (min, median, max) of observed values."""
    vals = sorted(values)
    n = len(vals)
    mid = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return (vals[0], mid, vals[-1])


def _widen(low: float, mode: float, high: float, floor: float = 1e-4) -> tuple:
    """Guarantee low < mode < high so the triangular sampler stays well defined."""
    if high - low < floor:
        pad = max(abs(mode) * 0.05, floor)
        low, high = mode - pad, mode + pad
    mode = min(max(mode, low + floor / 2), high - floor / 2)
    return (round(low, 6), round(mode, 6), round(high, 6))


def rolling_cagrs(history: dict, years: int) -> list:
    """Every realised `years`-long CAGR in the revenue history."""
    pts = sorted((int(y), v) for y, v in history.items() if v and v > 0)
    out = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            span = pts[j][0] - pts[i][0]
            if span == years:
                out.append((pts[j][1] / pts[i][1]) ** (1 / span) - 1)
    return out


def calibrate_drivers(profile: dict, horizon_years: float = 2.0) -> dict:
    """
    Derive driver distributions from what the company actually filed.

    Revenue growth and EBITDA margin come straight from the realised history —
    that is the whole point, since hand-set ranges tend to encode a directional
    view as if it were uncertainty. The exit multiple genuinely is an analyst
    judgement (it is the comps view, and EDGAR carries no price history), so it
    is anchored on the current multiple and flagged as uncalibrated.
    """
    notes = {}

    span = max(1, int(round(horizon_years)))
    cagrs = rolling_cagrs(profile["revenue_history_m"], span)
    if len(cagrs) >= 3:
        rev_cagr = _widen(*_triple(cagrs))
        notes["rev_cagr"] = (
            f"{len(cagrs)} realised {span}-year CAGRs, FY"
            f"{min(profile['revenue_history_m'])}–FY{max(profile['revenue_history_m'])}"
        )
    else:
        rev_cagr = _widen(0.0, 0.03, 0.06)
        notes["rev_cagr"] = "insufficient revenue history — generic range, set this yourself"

    margins = list(profile["margin_history"].values())
    if len(margins) >= 3:
        lo, mid, hi = _triple(margins)
        ebitda_margin = _widen(lo, profile["ebitda_margin_fy"], hi)
        notes["ebitda_margin"] = (
            f"{len(margins)} filed years, {min(margins)*100:.2f}–{max(margins)*100:.2f}%; "
            f"mode set to latest FY"
        )
    else:
        m = profile["ebitda_margin_fy"]
        ebitda_margin = _widen(m * 0.8, m, m * 1.2)
        notes["ebitda_margin"] = "insufficient margin history — ±20% around latest FY"

    current_multiple = profile.get("current_ev_ebitda")
    if current_multiple and current_multiple > 0:
        ev_ebitda = _widen(current_multiple * 0.70, current_multiple, current_multiple * 1.15)
        notes["ev_ebitda_multiple"] = (
            f"anchored on the current {current_multiple:.1f}× — NOT calibrated from "
            "filings; replace with your comps range"
        )
    else:
        ev_ebitda = _widen(8.0, 11.0, 14.0)
        notes["ev_ebitda_multiple"] = "no price, so no current multiple — placeholder, set from comps"

    dilution = profile.get("share_change_stats")
    if dilution:
        share_dilution = (round(dilution["mean"], 6), round(max(dilution["std"], 0.001), 6))
        notes["share_dilution_pct"] = f"{dilution['n']} year-over-year share-count changes"
    else:
        share_dilution = (0.01, 0.01)
        notes["share_dilution_pct"] = "no usable share-count history — generic assumption"

    notes["net_debt_change_pct"] = (
        "generic assumption — EDGAR gives the balance-sheet level, not a forward path"
    )

    return {
        "rev_cagr": rev_cagr,
        "ebitda_margin": ebitda_margin,
        "ev_ebitda_multiple": ev_ebitda,
        "net_debt_change_pct": (0.0, 0.015),
        "share_dilution_pct": share_dilution,
        "notes": notes,
    }


def apply_price(profile: dict, price: float) -> dict:
    """
    Fill in the price-dependent figures: market cap, enterprise value and the
    current EV/EBITDA multiple. Kept separate from build_profile so a
    hand-entered price produces exactly the same result as a fetched one.
    """
    profile = dict(profile)
    market_cap = price * profile["shares_outstanding_m"]
    net_debt = profile.get("net_debt_m") or 0.0
    ev = market_cap + net_debt

    profile["market_cap_m"] = market_cap
    profile["current_ev_m"] = ev
    profile["current_ev_ebitda"] = ev / profile["ebitda_fy_m"] if profile["ebitda_fy_m"] else None
    return profile


def build_prefill(ticker: str, price: Optional[float] = None,
                  horizon_years: float = 2.0, history_years: int = 10) -> dict:
    """
    Everything the assumption form needs for one ticker.

    Returns the field values, the calibrated driver distributions, the history
    they were derived from, and a provenance entry per field. The three case
    targets are deliberately absent: bear/base/bull are the analyst's DCF
    output, and the point of the tool is to test them against the simulation
    rather than to generate them.
    """
    profile = build_profile(ticker, history_years=history_years)

    price_meta = profile.get("price")
    if price is None and price_meta:
        price = price_meta["value"]
    if price is not None:
        profile = apply_price(profile, float(price))

    drivers = calibrate_drivers(profile, horizon_years)

    # Revenue and margin must come from the same period, or revenue x margin
    # equals no EBITDA the company ever filed. TTM revenue rolls cleanly from
    # quarterly tags, but TTM D&A does not — most filers tag it annually only —
    # so the fiscal year is the one basis where both halves are real.
    revenue = profile["revenue_fy_m"]
    revenue_basis = f"FY{profile['latest_fy']}"

    fields = {
        "ticker": profile["ticker"],
        "company_name": profile["company_name"],
        "current_price": price,
        "current_revenue": round(revenue, 1),
        "current_ebitda_margin": round(profile["ebitda_margin_fy"], 5),
        "current_ev": round(profile["current_ev_m"], 1) if price is not None else None,
        "current_net_debt": round(profile["net_debt_m"], 1) if profile["net_debt_m"] is not None else None,
        "shares_outstanding": round(profile["shares_outstanding_m"], 4),
        "rev_cagr": list(drivers["rev_cagr"]),
        "ebitda_margin": list(drivers["ebitda_margin"]),
        "ev_ebitda_multiple": list(drivers["ev_ebitda_multiple"]),
        "net_debt_change_pct": list(drivers["net_debt_change_pct"]),
        "share_dilution_pct": list(drivers["share_dilution_pct"]),
        "horizon_years": horizon_years,
    }

    nd = profile.get("net_debt_meta") or {}
    provenance = {
        "current_price": (
            f"{price_meta['source']}, {price_meta['as_of']}" if price_meta
            else ("entered by hand" if price is not None else "not set — no quote provider configured")
        ),
        "current_revenue": (
            f"SEC EDGAR, {revenue_basis}. TTM revenue is "
            f"${profile['revenue_ttm_m']:,.0f}M ({profile['revenue_ttm_label']}) but TTM D&A "
            "is not tagged, so the fiscal year keeps revenue and margin on one basis"
            if profile.get("revenue_ttm_m") else f"SEC EDGAR, {revenue_basis}"
        ),
        "current_ebitda_margin": (
            f"SEC EDGAR, FY{profile['latest_fy']}: {profile['ebitda_basis']}, over revenue "
            f"— pairs with revenue above to give filed EBITDA of ${profile['ebitda_fy_m']:,.1f}M"
        ),
        "current_ev": (
            "derived: price × shares + net debt" if price is not None
            else "needs a price"
        ),
        "current_net_debt": (
            f"SEC EDGAR, {nd.get('as_of', 'n/a')}: debt "
            f"${nd.get('total_debt_m', 0):,.1f}M less cash ${nd.get('cash_m', 0):,.1f}M"
            if nd else "not tagged in filings — set by hand"
        ),
        "shares_outstanding": (
            f"SEC EDGAR, {profile['shares_meta']['as_of']} "
            f"({profile['shares_meta']['form']} filed {profile['shares_meta']['filed']})"
        ),
        **drivers["notes"],
    }

    return {
        "fields": fields,
        "provenance": provenance,
        "history": {
            "revenue_m": profile["revenue_history_m"],
            "margin": profile["margin_history"],
        },
        "company": {
            "cik": profile["cik"],
            "name": profile["company_name"],
            "latest_fy": profile["latest_fy"],
            "filings_url": profile["filings_url"],
            "ebitda_fy_m": round(profile["ebitda_fy_m"], 1),
            "ebitda_basis": profile["ebitda_basis"],
            "revenue_ttm_m": round(profile["revenue_ttm_m"], 1) if profile.get("revenue_ttm_m") else None,
            "revenue_ttm_label": profile.get("revenue_ttm_label"),
            "market_cap_m": round(profile["market_cap_m"], 1) if price is not None else None,
            "current_ev_ebitda": (
                round(profile["current_ev_ebitda"], 2)
                if price is not None and profile.get("current_ev_ebitda") else None
            ),
        },
        "needs_price": price is None,
        "source": "SEC EDGAR XBRL company facts — no third-party model or inference in this path",
    }
