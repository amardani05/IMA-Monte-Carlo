"""
Deterministic market-data layer.

Pulls company fundamentals straight from SEC EDGAR's XBRL company-facts API and
derives a complete set of pitch assumptions from filed history. There is no
model, no inference service and no third-party analytics in this path: every
number returned is either a figure the company filed with the SEC or plain
arithmetic over those figures, and each one is returned with its source and
as-of date so it can be checked against the filing.

The one figure EDGAR cannot supply is a live share price. If a quote provider is
configured (see PRICE_PROVIDERS) it is used; otherwise price is returned as
null and the caller supplies it. Everything else auto-fills either way.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
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
# first hit winning outright. see _merged_series.
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
PUBLIC_FLOAT_TAGS = ["EntityPublicFloat"]
CFO_TAGS = ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
CAPEX_TAGS = ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"]

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
    map has not seen, typically a recent listing.
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
        f"{ticker} is not an SEC registrant. Foreign-listed names, ADRs and "
        "private companies do not file XBRL with EDGAR"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Series extraction
# ─────────────────────────────────────────────────────────────────────────────
def facts_as_of(facts: dict, as_of: Optional[str]) -> dict:
    """
    The company-facts document as it stood on a date: every row filed after
    `as_of` is dropped. This is what makes a backtest honest; EDGAR carries the
    `filed` date on every fact, so no look-ahead is needed to reconstruct it.
    """
    if not as_of:
        return facts
    out = {"facts": {}}
    for k in ("entityName", "cik", "_ticker"):
        if k in facts:
            out[k] = facts[k]
    for taxonomy, tags in (facts.get("facts") or {}).items():
        kept_tags = {}
        for tag, node in tags.items():
            units = {}
            for unit, rows in (node.get("units") or {}).items():
                keep = [r for r in rows if (r.get("filed") or "9999-99-99") <= as_of]
                if keep:
                    units[unit] = keep
            if units:
                kept_tags[tag] = {**{k: v for k, v in node.items() if k != "units"}, "units": units}
        if kept_tags:
            out["facts"][taxonomy] = kept_tags
    return out


_ANNUAL = re.compile(r"^CY(\d{4})$")
_QUARTER = re.compile(r"^CY(\d{4})Q(\d)$")


def _derive_frame(row: dict) -> Optional[str]:
    """
    A calendar-frame label for a fact that carries none.

    EDGAR attaches `frame` to only one copy of each fact, normally the most
    recently filed one. Filter the document to a past date and that copy is
    gone, so the period must be rebuilt from start/end. Durations of about a
    year become CY<year>, about a quarter become CY<year>Q<n>, instants become
    CY<year>Q<n>I, following EDGAR's own alignment rule closely enough for
    annual and quarterly series.
    """
    end = row.get("end")
    if not end:
        return None
    try:
        e = datetime.date.fromisoformat(end)
    except ValueError:
        return None
    start = row.get("start")
    if not start:
        q = (e.month - 1) // 3 + 1
        return f"CY{e.year}Q{q}I"
    try:
        st = datetime.date.fromisoformat(start)
    except ValueError:
        return None
    days = (e - st).days
    if 350 <= days <= 380:
        year = e.year if e.month >= 6 else e.year - 1
        return f"CY{year}"
    if 80 <= days <= 100:
        q = (e.month - 1) // 3 + 1
        return f"CY{e.year}Q{q}"
    return None


def _tag_frames(facts: dict, tag: str, taxonomy: str = "us-gaap", unit: str = "USD") -> dict:
    """
    {frame_label: value}. Uses EDGAR's frame where present and derives one
    otherwise; when several filings carry the same period, the latest filed
    copy wins, which is the restatement that was known at the time.
    """
    node = facts.get("facts", {}).get(taxonomy, {}).get(tag)
    if not node:
        return {}
    best: dict = {}
    for unit_name, rows in node.get("units", {}).items():
        if unit_name != unit:
            continue
        for row in rows:
            frame = row.get("frame") or _derive_frame(row)
            if not frame:
                continue
            filed = row.get("filed") or ""
            prev = best.get(frame)
            if prev is None or filed >= prev[0]:
                best[frame] = (filed, row["val"])
    return {k: v for k, (_, v) in best.items()}


def _merged_series(facts: dict, tags: list, pattern, unit: str = "USD") -> dict:
    """
    Merge candidate tags into one series keyed by period.

    Earlier tags win on conflict, but later tags fill periods the earlier ones
    never covered: a filer that switched tagging mid-history (MYRG moved from
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


_INSTANT = re.compile(r"^CY(\d{4})Q(\d)I$")


def _instant_series(facts: dict, tags: list, taxonomy: str = "us-gaap", unit: str = "USD") -> dict:
    """{year -> value} for a balance-sheet concept, taking the latest quarter-end instant in each year."""
    out: dict = {}
    for tag in reversed(tags):
        node = facts.get("facts", {}).get(taxonomy, {}).get(tag)
        if not node:
            continue
        for unit_name, rows in node.get("units", {}).items():
            if unit_name != unit:
                continue
            for row in rows:
                m = _INSTANT.match(row.get("frame") or _derive_frame(row) or "")
                if not m:
                    continue
                year, q = int(m.group(1)), int(m.group(2))
                filed = row.get("filed") or ""
                cur = out.get(year)
                if cur is None or q > cur[0] or (q == cur[0] and filed >= cur[2]):
                    out[year] = (q, float(row["val"]), filed)
    return {y: v for y, (_, v, _f) in out.items()}


def _public_float_series(facts: dict) -> dict:
    """{year -> public float USD}. Filed once a year as of the second fiscal quarter's end."""
    node = facts.get("facts", {}).get("dei", {}).get("EntityPublicFloat")
    if not node:
        return {}
    out: dict = {}
    for rows in node.get("units", {}).values():
        for row in rows:
            end = row.get("end")
            if not end:
                continue
            year = int(end[:4])
            if year not in out or end > out[year][0]:
                out[year] = (end, float(row["val"]))
    return {y: v for y, (_, v) in out.items()}


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
# Share price. the one figure EDGAR cannot provide
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
    if not key:
        try:
            from .prices import latest_price
            q = latest_price(ticker)
            if q:
                return q
        except Exception:
            pass
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
NCI_TAGS = ["MinorityInterest", "StockholdersEquityAttributableToNoncontrollingInterest"]
PREFERRED_TAGS = ["PreferredStockValue"]


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
        gaap = facts.get("facts", {}).get("us-gaap", {})
        node = gaap.get("CommonStockSharesOutstanding") or gaap.get("WeightedAverageNumberOfDilutedSharesOutstanding")
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
    """Registrant metadata. SIC, exchange, fiscal year end. from EDGAR submissions."""
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
            "EV/EBITDA are not meaningful for them. This model does not fit."
        )
    if REIT_SIC[0] <= sic <= REIT_SIC[1]:
        return [
            f"{ticker} is classified as {desc} (SIC {sic}). Real-estate names are "
            "normally valued on FFO or cap rates rather than EV/EBITDA. Treat the "
            "multiple with care."
        ]
    return []


def _ebitda_history(facts: dict, rev_annual: dict, da_annual: dict) -> tuple:
    """
    EBITDA per year, trying progressively looser reconstructions.

    Not every filer tags operating income (Dycom, for one, does not), so fall
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


def build_profile(ticker: str, history_years: int = 10, as_of: Optional[str] = None) -> dict:
    """
    Everything needed to populate a pitch, with provenance on every figure.

    Monetary values are returned in $M to match PitchAssumptions.
    """
    cik, title = resolve_cik(ticker)
    meta = company_meta(cik)
    warnings = _check_modelable(ticker.upper(), meta)
    facts = facts_as_of(_get_json(COMPANY_FACTS_URL.format(cik=cik), FACTS_TTL), as_of)

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

    # Historical EV/EBITDA on a public-float basis. Public float undercounts
    # market cap by the insider stake, so the LEVEL is biased low; the
    # year-to-year DISPERSION and the ratio of today to history are what we use.
    float_hist = _public_float_series(facts)
    cash_hist = _instant_series(facts, CASH_TAGS)
    nci_hist = _instant_series(facts, NCI_TAGS)
    debt_hist: dict = {}
    for group in DEBT_COMPONENT_TAGS:
        parts = [_instant_series(facts, [t]) for t in group]
        years = set.intersection(*(set(p) for p in parts)) if parts else set()
        if years:
            debt_hist = {y: sum(p[y] for p in parts) for y in years}
            break
    # A multiple on near-zero EBITDA is noise, not valuation: QuinStreet's 0.2%
    # margin year would otherwise contribute a 447x observation.
    MIN_MARGIN_FOR_MULTIPLE, MAX_SANE_MULTIPLE = 0.02, 50.0
    multiple_hist = {}
    for y in sorted(set(float_hist) & set(ebitda_annual) & set(cash_hist)):
        if y < cutoff or ebitda_annual[y] <= 0 or not rev_annual.get(y):
            continue
        if ebitda_annual[y] / rev_annual[y] < MIN_MARGIN_FOR_MULTIPLE:
            continue
        ev_proxy = float_hist[y] + debt_hist.get(y, 0.0) - cash_hist[y] + nci_hist.get(y, 0.0)
        if ev_proxy > 0:
            m = ev_proxy / ebitda_annual[y]
            if m <= MAX_SANE_MULTIPLE:
                multiple_hist[y] = m

    # Free cash flow history for the net-debt drift
    cfo_annual = annual_series(facts, CFO_TAGS)
    capex_annual = annual_series(facts, CAPEX_TAGS)
    fcf_hist = {
        y: cfo_annual[y] - capex_annual[y]
        for y in sorted(set(cfo_annual) & set(capex_annual))
        if y >= cutoff
    }

    shares = _point_in_time(facts, SHARES_TAGS, taxonomy="dei", unit="shares")
    if not shares:
        # some filers carry the count only in us-gaap, or only as a diluted weighted average
        shares = _point_in_time(facts, ["CommonStockSharesOutstanding", "CommonStockSharesIssued",
                                        "WeightedAverageNumberOfDilutedSharesOutstanding"], unit="shares")
    if not shares:
        raise DataError(f"No share count tagged in {ticker}'s filings")
    cash = _point_in_time(facts, CASH_TAGS)
    debt = _debt(facts)

    net_debt_m = None
    net_debt_meta = None
    if cash and debt:
        # Minority interest and preferred stock are claims on enterprise value
        # ahead of the common. TDS consolidates UScellular at about 82%; the
        # 18% it does not own is not the shareholders' and comes off here.
        nci = _point_in_time(facts, NCI_TAGS)
        pref = _point_in_time(facts, PREFERRED_TAGS)
        other_claims = (nci[1] if nci else 0.0) + (pref[1] if pref else 0.0)
        net_debt_m = (debt[0] - cash[1] + other_claims) / 1e6
        net_debt_meta = {
            "total_debt_m": debt[0] / 1e6,
            "cash_m": cash[1] / 1e6,
            "minority_interest_m": (nci[1] / 1e6) if nci else 0.0,
            "preferred_m": (pref[1] / 1e6) if pref else 0.0,
            "as_of": max(debt[1], cash[0]),
            "components": debt[2],
            "note": "total debt (incl. finance leases) plus minority interest and preferred, less cash; operating leases excluded",
        }

    margin_now = ebitda_annual[latest_fy] / rev_annual[latest_fy]
    if not -0.5 < margin_now < 0.75:
        warnings.append(
            f"Derived EBITDA margin of {margin_now * 100:.1f}% is outside the range "
            "this bridge normally sees. Check it against the filing before relying on it."
        )

    price = None if as_of else fetch_price(ticker)
    display_name = meta.get("name") or title

    facts["_ticker"] = ticker.upper()
    return {
        "_facts": facts,
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
        "multiple_history": {str(y): multiple_hist[y] for y in sorted(multiple_hist)},
        "multiple_history_basis": "public float",
        "shares_history_m": {str(y): v / 1e6 for y, v in sorted(_share_history(facts).items()) if y >= cutoff},
        "net_debt_history_m": {str(y): (debt_hist.get(y, 0.0) - cash_hist[y] + nci_hist.get(y, 0.0)) / 1e6
                               for y in sorted(cash_hist) if y >= cutoff},
        "public_float_latest_m": (
            float_hist[max(float_hist)] / 1e6 if float_hist else None
        ),
        "fcf_history_m": {str(y): fcf_hist[y] / 1e6 for y in sorted(fcf_hist)},
        "share_change_stats": _share_change_stats(facts),
        "source": "SEC EDGAR XBRL company facts",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Driver calibration from filed history
# ─────────────────────────────────────────────────────────────────────────────
def _median(vals: list) -> float:
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _robust_bounds(values: list, trim_at: int = 8) -> tuple:
    """
    (low, median, high) of a history, trimming one extreme each side once there
    are enough observations to spare them.

    A single impairment year or divestiture otherwise defines the whole tail:
    TDS's 2023 write-down put a -34% margin into the range, and the UScellular
    sale put a -53% revenue CAGR there. Trimming is not a claim those years did
    not happen; it is a claim that one event should not set the mode-weighted
    mean of a triangular distribution.
    """
    v = sorted(values)
    trimmed = v[1:-1] if len(v) >= trim_at else v
    return trimmed[0], _median(trimmed), trimmed[-1]


def _widen(low: float, mode: float, high: float, floor: float = 1e-4) -> tuple:
    """Guarantee low < mode < high so the triangular sampler stays well defined."""
    if high - low < floor:
        pad = max(abs(mode) * 0.05, floor)
        low, high = mode - pad, mode + pad
    low, high = min(low, mode), max(high, mode)
    mode = min(max(mode, low + floor / 2), high - floor / 2)
    return (round(low, 6), round(mode, 6), round(high, 6))


def rolling_cagrs(history: dict, years: int) -> list:
    """Every realised `years`-long CAGR in the revenue history."""
    pts = sorted((int(y), v) for y, v in history.items() if v and v > 0)
    out = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if pts[j][0] - pts[i][0] == years:
                out.append((pts[j][1] / pts[i][1]) ** (1 / years) - 1)
    return out


def calibrate_drivers(profile: dict, horizon_years: float = 2.0) -> dict:
    """
    Derive driver distributions from what the company actually filed.

    The WIDTH of each range comes from history; that is the uncertainty. The
    MODE is where the analyst's view belongs, and by default it is held at
    today's value, which makes the untouched model a status-quo projection:
    revenue grows, margin and multiple stay put, so the median lands near spot.
    That is the honest null, not a forecast. Every driver therefore returns a
    `reference` block with the current value and the historical median so the
    view can be set deliberately rather than inherited.
    """
    notes, refs = {}, {}

    # ── Revenue growth ──
    span = max(1, int(round(horizon_years)))
    cagrs = rolling_cagrs(profile["revenue_history_m"], span)
    if len(cagrs) >= 3:
        lo, med, hi = _robust_bounds(cagrs)
        rev_cagr = _widen(lo, med, hi)
        trimmed = " (one extreme trimmed each side)" if len(cagrs) >= 8 else ""
        notes["rev_cagr"] = (
            f"{len(cagrs)} realised {span}-year CAGRs, FY{min(profile['revenue_history_m'])}"
            f"-FY{max(profile['revenue_history_m'])}{trimmed}. Mode is the historical median."
        )
        refs["rev_cagr"] = {"hist_median": round(med, 4), "raw_min": round(min(cagrs), 4),
                            "raw_max": round(max(cagrs), 4), "n": len(cagrs)}
    else:
        rev_cagr = _widen(0.0, 0.03, 0.06)
        notes["rev_cagr"] = "Insufficient revenue history. Generic range; set this yourself."
        refs["rev_cagr"] = {}

    # ── EBITDA margin ──
    margins = list(profile["margin_history"].values())
    current_margin = profile["ebitda_margin_fy"]
    if len(margins) >= 3:
        lo, med, hi = _robust_bounds(margins)
        ebitda_margin = _widen(lo, current_margin, hi)
        loss_years = sum(1 for m in margins if m <= 0)
        trimmed = " (one extreme trimmed each side)" if len(margins) >= 8 else ""
        notes["ebitda_margin"] = (
            f"{len(margins)} filed years, {min(margins)*100:.1f}% to {max(margins)*100:.1f}%{trimmed}. "
            f"Mode held at latest FY ({current_margin*100:.2f}%); historical median is {med*100:.2f}%."
            + (f" {loss_years} loss year(s) in the record." if loss_years else "")
        )
        refs["ebitda_margin"] = {"current": round(current_margin, 5), "hist_median": round(med, 5),
                                 "raw_min": round(min(margins), 5), "raw_max": round(max(margins), 5),
                                 "n": len(margins)}
    else:
        ebitda_margin = _widen(current_margin * 0.8, current_margin, current_margin * 1.2)
        notes["ebitda_margin"] = "Insufficient margin history. Plus or minus 20% around latest FY."
        refs["ebitda_margin"] = {"current": round(current_margin, 5)}

    # ── Exit multiple ──
    current_multiple = profile.get("current_ev_ebitda")
    if current_multiple and (current_multiple > 40 or profile.get("ebitda_margin_fy", 1.0) < 0.02):
        current_multiple = None
    mult_hist = list(profile.get("multiple_history", {}).values())
    float_basis_current = None
    if profile.get("public_float_latest_m") and profile.get("ebitda_fy_m"):
        nd = profile.get("net_debt_m") or 0.0
        float_basis_current = (profile["public_float_latest_m"] + nd) / profile["ebitda_fy_m"]

    if current_multiple and current_multiple > 0:
        if len(mult_hist) >= 6 and float_basis_current and float_basis_current > 0:
            # Dispersion and rich/cheap ratio on the float basis, applied to the
            # true current multiple so the insider-stake bias cancels. Bounded to
            # half and one-and-a-half times today: a multiple history built on
            # depressed earnings (QuinStreet at 30-40x on 2% margins) is not a
            # forecast of where a re-earning business will trade.
            lo_f, med_f, hi_f = _robust_bounds(mult_hist)
            scale = current_multiple / float_basis_current
            lo, med, hi = lo_f * scale, med_f * scale, hi_f * scale
            lo, hi = min(lo, med, current_multiple * 0.85), max(hi, med, current_multiple * 1.15)
            lo = max(lo, current_multiple * 0.5)
            hi = min(hi, current_multiple * 1.5)
            ev_ebitda = _widen(lo, current_multiple, hi)
            rich = current_multiple / med - 1
            notes["ev_ebitda_multiple"] = (
                f"Width from {len(mult_hist)} years of filed public float and EBITDA. Mode held at the "
                f"current {current_multiple:.1f}x; the historical median on the same basis is {med:.1f}x, "
                f"so the stock is {abs(rich)*100:.0f}% {'richer' if rich > 0 else 'cheaper'} than its own history. "
                "Your comps view belongs in the mode."
            )
            refs["ev_ebitda_multiple"] = {"current": round(current_multiple, 2), "hist_median": round(med, 2),
                                          "raw_min": round(min(mult_hist) * scale, 2),
                                          "raw_max": round(max(mult_hist) * scale, 2), "n": len(mult_hist),
                                          "basis": "public float, rescaled to today's true multiple"}
        else:
            ev_ebitda = _widen(current_multiple * 0.80, current_multiple, current_multiple * 1.20)
            notes["ev_ebitda_multiple"] = (
                f"Anchored on the current {current_multiple:.1f}x, plus or minus 20%. "
                + (f"Only {len(mult_hist)} usable years of filed multiple history, too few to set a width. "
                   if mult_hist else "No usable filed multiple history. ")
                + "Your comps view belongs in the mode."
            )
            refs["ev_ebitda_multiple"] = {"current": round(current_multiple, 2)}
            if mult_hist:
                refs["ev_ebitda_multiple"]["hist_median"] = round(
                    _median(mult_hist) * (current_multiple / float_basis_current if float_basis_current else 1), 2)
                refs["ev_ebitda_multiple"]["n"] = len(mult_hist)
    else:
        ev_ebitda = _widen(8.0, 11.0, 14.0)
        notes["ev_ebitda_multiple"] = "No price, so no current multiple. Placeholder; set from comps."
        refs["ev_ebitda_multiple"] = {}

    # ── Net debt drift from free cash flow ──
    fcf = list(profile.get("fcf_history_m", {}).values())
    ev_m = profile.get("current_ev_m")
    if len(fcf) >= 3 and ev_m and ev_m > 0:
        recent = fcf[-3:]
        yields = [f / ev_m for f in recent]
        mean_yield = sum(yields) / len(yields)
        n = len(yields)
        var = sum((y - mean_yield) ** 2 for y in yields) / max(n - 1, 1)
        std_yield = var ** 0.5
        # Cash generated over the horizon reduces net debt; the engine scales this by EV.
        drift = -mean_yield * horizon_years
        std = max(std_yield * horizon_years ** 0.5, 0.005)
        MAX_DRIFT, MAX_STD = 0.60, 0.15
        capped = abs(drift) > MAX_DRIFT or std > MAX_STD
        drift = max(-MAX_DRIFT, min(MAX_DRIFT, drift))
        std = min(std, MAX_STD)
        net_debt_change = (round(drift, 5), round(std, 5))
        notes["net_debt_change_pct"] = (
            f"Free cash flow (operating cash flow less capex) averaged {mean_yield*100:.1f}% of EV over the "
            f"last {n} filed years, so net debt drifts {drift*100:+.1f}% of EV over {horizon_years:g} years "
            "before any buybacks, dividends or M&A, which are yours to add."
            + (" Capped: the cash-flow history is large relative to enterprise value, which the bridge "
               "cannot represent well." if capped else "")
        )
        refs["net_debt_change_pct"] = {"fcf_yield_mean": round(mean_yield, 4),
                                       "fcf_recent_m": [round(f, 1) for f in recent], "capped": capped}
    else:
        net_debt_change = (0.0, 0.015)
        notes["net_debt_change_pct"] = "Generic assumption. Cash-flow tags were not usable for this filer."
        refs["net_debt_change_pct"] = {}

    # ── Dilution ──
    dil = profile.get("share_change_stats")
    if dil:
        share_dilution = (round(dil["mean"], 6), round(max(dil["std"], 0.001), 6))
        notes["share_dilution_pct"] = (
            f"{dil['n']} year-over-year share-count changes; "
            f"{'net buybacks' if dil['mean'] < 0 else 'net issuance'} on average."
        )
        refs["share_dilution_pct"] = {"hist_mean": round(dil["mean"], 4), "hist_std": round(dil["std"], 4)}
    else:
        share_dilution = (0.01, 0.01)
        notes["share_dilution_pct"] = "No usable share-count history. Generic assumption."
        refs["share_dilution_pct"] = {}

    return {
        "rev_cagr": rev_cagr,
        "ebitda_margin": ebitda_margin,
        "ev_ebitda_multiple": ev_ebitda,
        "net_debt_change_pct": net_debt_change,
        "share_dilution_pct": share_dilution,
        "notes": notes,
        "reference": refs,
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
                  horizon_years: float = 2.0, history_years: int = 10,
                  as_of: Optional[str] = None, calibration: str = "auto",
                  panel: Optional[dict] = None, params: Optional[dict] = None) -> dict:
    """
    Everything the assumption form needs for one ticker.

    Returns the field values, the calibrated driver distributions, the history
    they were derived from, and a provenance entry per field. The three case
    targets are deliberately absent: bear/base/bull are the analyst's DCF
    output, and the point of the tool is to test them against the simulation
    rather than to generate them.
    """
    profile = build_profile(ticker, history_years=history_years, as_of=as_of)

    price_meta = profile.get("price")
    if price is None and price_meta:
        price = price_meta["value"]
    if price is not None:
        profile = apply_price(profile, float(price))

    # Price-based multiple history when a price series is reachable (Phase 1)
    try:
        from .prices import price_history
        from .panel import price_based_multiples
        mh = price_based_multiples(profile, price_history(ticker, 12), as_of=as_of)
        if len(mh) >= 4:
            profile["multiple_history"] = {str(y): v for y, v in sorted(mh.items())}
            profile["multiple_history_basis"] = "year-end close x shares + net debt, over FY EBITDA"
    except Exception:
        pass

    use_v2 = calibration == "v2" or (calibration == "auto" and panel is not None)
    if calibration == "auto" and panel is None:
        from .panel import load_cached
        sic = None
        try:
            sic = company_meta(profile["cik"]).get("sic")
        except Exception:
            sic = None
        panel = load_cached(sic, ticker)
        use_v2 = panel is not None
    drivers = calibrate_drivers_v2(profile, panel, horizon_years, params) if use_v2 else calibrate_drivers(profile, horizon_years)

    # Revenue and margin must come from the same period, or revenue x margin
    # equals no EBITDA the company ever filed. TTM revenue rolls cleanly from
    # quarterly tags, but TTM D&A does not. most filers tag it annually only -
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
            else ("entered by hand" if price is not None else "not set. No quote provider configured")
        ),
        "current_revenue": (
            f"SEC EDGAR, {revenue_basis}. TTM revenue is "
            f"${profile['revenue_ttm_m']:,.0f}M ({profile['revenue_ttm_label']}) but TTM D&A "
            "is not tagged, so the fiscal year keeps revenue and margin on one basis"
            if profile.get("revenue_ttm_m") else f"SEC EDGAR, {revenue_basis}"
        ),
        "current_ebitda_margin": (
            f"SEC EDGAR, FY{profile['latest_fy']}: {profile['ebitda_basis']}, over revenue "
            f". Pairs with revenue above to give filed EBITDA of ${profile['ebitda_fy_m']:,.1f}M"
        ),
        "current_ev": (
            "derived: price × shares + net debt" if price is not None
            else "needs a price"
        ),
        "current_net_debt": (
            f"SEC EDGAR, {nd.get('as_of', 'n/a')}: debt ${nd.get('total_debt_m', 0):,.1f}M"
            + (f" plus minority interest ${nd['minority_interest_m']:,.1f}M" if nd.get("minority_interest_m") else "")
            + (f" plus preferred ${nd['preferred_m']:,.1f}M" if nd.get("preferred_m") else "")
            + f" less cash ${nd.get('cash_m', 0):,.1f}M"
            if nd else "not tagged in filings. Set by hand"
        ),
        "shares_outstanding": (
            f"SEC EDGAR, {profile['shares_meta']['as_of']} "
            f"({profile['shares_meta']['form']} filed {profile['shares_meta']['filed']})"
        ),
        **drivers["notes"],
    }

    warnings = list(profile.get("warnings") or [])
    nd, mc = profile.get("net_debt_m"), profile.get("market_cap_m")
    if price is not None and nd is not None and mc and nd < -0.5 * mc:
        warnings.append(
            f"Net cash is {abs(nd)/mc*100:.0f}% of market cap. An EV/EBITDA bridge values the operating "
            "business and carries the cash across unchanged, so what management does with that cash "
            "(buybacks, special dividend, acquisitions) is most of the equity story and this model "
            "cannot see it. Consider a sum-of-parts alongside."
        )
    if profile.get("ebitda_fy_m") and price is not None and profile.get("current_ev_ebitda") and profile["current_ev_ebitda"] < 4:
        warnings.append(
            f"Current EV/EBITDA of {profile['current_ev_ebitda']:.1f}x is unusually low, which usually means "
            "the enterprise value is dominated by cash or the EBITDA is transitional. Check the filing."
        )

    mech = mechanisms(profile, fields, drivers, horizon_years)
    extra = {"mechanisms": mech}
    if drivers.get("version") == "v2":
        extra.update({"driver_specs": drivers["driver_specs"], "copula": drivers["copula"],
                      "copula_df": drivers["copula_df"], "correlation_matrix": drivers["correlation_matrix"],
                      "panel": drivers.get("panel")})
        provenance["correlation_matrix"] = drivers["notes"]["correlation_matrix"]
    return {
        **extra,
        "calibration": drivers.get("version", "v1"),
        "fields": fields,
        "provenance": provenance,
        "reference": drivers["reference"],
        "warnings": warnings,
        "history": {
            "revenue_m": profile["revenue_history_m"],
            "margin": profile["margin_history"],
            "multiple": profile.get("multiple_history", {}),
            "fcf_m": profile.get("fcf_history_m", {}),
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
        "source": "SEC EDGAR XBRL company facts. No third-party model or inference in this path",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calibration v2: priors that move
# ─────────────────────────────────────────────────────────────────────────────
def _nearest_psd(m: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """Clip eigenvalues so a pooled correlation estimate is a valid copula matrix."""
    sym = (m + m.T) / 2
    w, v = np.linalg.eigh(sym)
    w = np.maximum(w, floor)
    out = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def _blend_quantiles(own: Optional[dict], ref: Optional[dict], n_own: int, k: float = 8.0) -> tuple:
    """
    Shrink the company's P10/P50/P90 toward the reference class.

    Weight on the company's own record is n / (n + k): ten filed years count
    for about 55%, three years for about 27%. k = 8 is the Base Rate Book's
    spirit (lean outside when the record is short) made numeric; it is a
    parameter, and the backtest is the place to tune it.
    """
    if own and ref:
        w = n_own / (n_own + k)
        return tuple(w * own[q] + (1 - w) * ref[q] for q in ("p10", "p50", "p90")) + (w,)
    if own:
        return own["p10"], own["p50"], own["p90"], 1.0
    if ref:
        return ref["p10"], ref["p50"], ref["p90"], 0.0
    return None


def _ensure_order(p10, p50, p90, min_gap):
    lo = min(p10, p50 - min_gap)
    hi = max(p90, p50 + min_gap)
    return float(lo), float(p50), float(hi)


# Defaults chosen on the backtest harness (tools/backtest.py, 84 snapshots on MYRG,
# TDS and QuinStreet, 2017 to 2024, after the minority-interest fix). Against the
# own-history baseline: CRPS/spot 0.590 -> 0.561, 90% coverage 71% -> 83%, PIT tails
# 37% -> 25%. The finding behind the numbers: the peer panel earns its keep through
# WIDTH (the company's own ten years understate what can happen), not by moving
# the centre. Centres stay on the company's own record, reverting toward its own
# long-run median; blending centres toward peer medians made every variant worse.
V2_DEFAULTS = {
    "k_shrink": 3.0,        # own-history weight is n / (n + k); ten years -> 77%
    "lam_margin": None,     # None = panel estimate shrunk toward Fama-French 0.38 and capped at 0.5
    "lam_mult": 0.0,        # no multiple reversion toward peers; the anchor is the company's own level
    "width_scale": 1.0,     # no horizon inflation of the cross-sectional width
    "growth_centre": "own", # growth P50 is the company's own realised median
    "target": "own",        # margin reverts toward the company's own long-run median
}


def calibrate_drivers_v2(profile: dict, panel: Optional[dict], horizon_years: float = 2.0,
                         params: Optional[dict] = None) -> dict:
    """
    Driver distributions with an outside view and mean reversion.

    Differences from calibrate_drivers:
    - Margin and multiple modes follow a partial-adjustment path toward the
      panel median over the horizon, at the panel's estimated reversion rate
      (Fama-French 0.38 a year when the panel cannot estimate it). The
      analyst's view is a deviation from that path, not from today.
    - Every driver is returned as a metalog spec (P10/P50/P90), with the
      company's quantiles shrunk toward the panel's.
    - The copula is Student-t with 5 degrees of freedom and a correlation
      matrix estimated from pooled year-over-year changes where the panel has
      enough pairs, falling back to the generic matrix cell by cell.
    Legacy triangular tuples are still produced so older clients keep working.
    """
    from .distributions import triangular_quantiles

    notes, refs, specs = {}, {}, {}
    pn = panel or {}
    T = float(horizon_years)
    span = max(1, int(round(T)))
    P = {**V2_DEFAULTS, **(params or {})}
    K = float(P["k_shrink"])

    # ── Revenue growth: own history shrunk toward the panel ──
    cagrs = rolling_cagrs(profile["revenue_history_m"], span)
    own = _q_of(cagrs) if len(cagrs) >= 3 else None
    ref = pn.get("cagr")
    blended = _blend_quantiles(own, ref, len(cagrs), k=K)
    if blended:
        p10, p50, p90, w = blended
        if P["growth_centre"] == "own" and own:
            p50 = own["p50"]
        p10, p50, p90 = _ensure_order(p10, p50, p90, 0.01)
        specs["rev_cagr"] = {"type": "metalog", "p10": round(p10, 5), "p50": round(p50, 5), "p90": round(p90, 5)}
        notes["rev_cagr"] = (
            f"P10/P50/P90 from {len(cagrs)} realised {span}-year CAGRs"
            + (f", weight {w:.0%}, blended with {ref['n']} peer observations ({pn.get('basis')})" if ref else "")
            + ". Mode is the blended median; your view goes in P50."
        )
    else:
        specs["rev_cagr"] = {"type": "metalog", "p10": 0.0, "p50": 0.03, "p90": 0.08}
        notes["rev_cagr"] = "No usable growth history or panel. Generic range; set this yourself."
    refs["rev_cagr"] = {"own": own, "panel": ref}

    # ── Margin: reverting path for the centre, blended width ──
    margins = list(profile["margin_history"].values())
    m0 = profile["ebitda_margin_fy"]
    rev_info = pn.get("margin_reversion") or {}
    if P["lam_margin"] is None:
        lam_est = float(rev_info.get("lambda", 0.38))
        n_pairs = int(rev_info.get("n_pairs", 0))
        # a rate from 47 noisy pairs is not a rate; shrink toward Fama-French and cap
        w_l = n_pairs / (n_pairs + 60.0)
        lam = min(w_l * lam_est + (1 - w_l) * 0.38, 0.5)
    else:
        lam = float(P["lam_margin"])
    ref_m = pn.get("margin")
    if ref_m:
        m_bar = ref_m["p50"]
        m_T = m0 + (1 - (1 - lam) ** T) * (m_bar - m0)
        path_note = f"centre moves from {m0*100:.2f}% toward the peer median {m_bar*100:.2f}% at {lam:.2f}/yr, landing at {m_T*100:.2f}% in {T:g} years"
    else:
        m_T = m0
        path_note = f"no peer panel, so the centre stays at today's {m0*100:.2f}%"
    own_m = _q_of(margins) if len(margins) >= 3 else None
    if ref_m and own_m:
        # The reversion target is not the raw peer median: a company that has
        # earned 3% for ten years in a group that earns 11% is told by its own
        # record that it is structurally different. Shrink the target toward
        # its long-run median with the same n/(n+k) weight used everywhere.
        w_t = 1.0 if P.get("target") == "own" else len(margins) / (len(margins) + K)
        m_bar = w_t * own_m["p50"] + (1 - w_t) * ref_m["p50"]
        m_T = m0 + (1 - (1 - lam) ** T) * (m_bar - m0)
        path_note = (f"centre moves from {m0*100:.2f}% toward {m_bar*100:.2f}% (own long-run median {own_m['p50']*100:.2f}% "
                     f"weighted {w_t:.0%}, peer median {ref_m['p50']*100:.2f}%) at {lam:.2f}/yr, landing at {m_T*100:.2f}% in {T:g} years")
    # width: own dispersion around own median, panel dispersion around panel median, blended
    if own_m:
        own_w = {"lo": own_m["p50"] - own_m["p10"], "hi": own_m["p90"] - own_m["p50"]}
    else:
        own_w = None
    if ref_m:
        ref_w = {"lo": ref_m["p50"] - ref_m["p10"], "hi": ref_m["p90"] - ref_m["p50"]}
    else:
        ref_w = None
    n_m = len(margins)
    if own_w and ref_w:
        w = n_m / (n_m + K)
        lo_w = w * own_w["lo"] + (1 - w) * ref_w["lo"]
        hi_w = w * own_w["hi"] + (1 - w) * ref_w["hi"]
    elif own_w:
        lo_w, hi_w, w = own_w["lo"], own_w["hi"], 1.0
    elif ref_w:
        lo_w, hi_w, w = ref_w["lo"], ref_w["hi"], 0.0
    else:
        lo_w = hi_w = max(abs(m0) * 0.25, 0.01); w = 0.0
    # widen with horizon: dispersion of a T-year-ahead margin is larger than one year's cross-section
    scale = T ** 0.5 if P["width_scale"] is None else float(P["width_scale"])
    p10, p50, p90 = _ensure_order(m_T - lo_w * scale, m_T, m_T + hi_w * scale, 0.002)
    specs["ebitda_margin"] = {"type": "metalog", "p10": round(p10, 5), "p50": round(p50, 5), "p90": round(p90, 5)}
    notes["ebitda_margin"] = (
        f"Mean reversion: {path_note} (reversion rate from {rev_info.get('n_pairs', 0)} peer year-pairs"
        + (f"; fallback {rev_info['fallback']}" if rev_info.get("fallback") else "") + "). "
        f"Width from {n_m} own years (weight {w:.0%}) and the panel, scaled by sqrt({T:g}). Your view goes in P50."
    )
    refs["ebitda_margin"] = {"current": round(m0, 5), "path_end": round(m_T, 5), "panel_median": (round(m_bar, 5) if ref_m else None),
                             "lambda": lam, "own": own_m, "panel": ref_m}

    # ── Exit multiple: current, reverting toward the panel median; log-metalog ──
    cur = profile.get("current_ev_ebitda")
    anchor_note = ""
    if cur and (cur > 40 or cur <= 0 or profile.get("ebitda_margin_fy", 1.0) < 0.02):
        # QuinStreet in 2017: 0.2% margin, 330x. The multiple on transitional
        # EBITDA says nothing about where a re-earning business will trade, so
        # the anchor is the panel or own median instead, and the note says so.
        fallback = (own_mu_pre := _q_of(list((profile.get("multiple_history") or {}).values()))) or pn.get("multiple")
        anchor_note = f" Today's {cur:.0f}x is struck on a {profile.get('ebitda_margin_fy', 0)*100:.1f}% margin and is not used as the anchor."
        cur = fallback["p50"] if fallback else None
    ref_mu = pn.get("multiple")
    lam_mu = float(P["lam_mult"])  # multiples revert more slowly than margins; backtest-tuned
    own_mu = _q_of(list((profile.get("multiple_history") or {}).values()))
    if cur and cur > 0:
        if ref_mu or own_mu:
            n_mu = own_mu["n"] if own_mu else 0
            w_mu = n_mu / (n_mu + K) if (own_mu and ref_mu) else (1.0 if own_mu else 0.0)
            target = (w_mu * own_mu["p50"] + (1 - w_mu) * ref_mu["p50"]) if (own_mu and ref_mu) else (own_mu or ref_mu)["p50"]
            mu_T = cur + (1 - (1 - lam_mu) ** T) * (target - cur)
            def ratios(q):
                return max(q["p10"] / q["p50"], 0.5), min(q["p90"] / q["p50"], 1.8)
            if own_mu and ref_mu:
                lo_o, hi_o = ratios(own_mu); lo_p, hi_p = ratios(ref_mu)
                lo_r, hi_r = w_mu * lo_o + (1 - w_mu) * lo_p, w_mu * hi_o + (1 - w_mu) * hi_p
            else:
                lo_r, hi_r = ratios(own_mu or ref_mu)
            p10, p50, p90 = mu_T * lo_r, mu_T, mu_T * hi_r
            mnote = (f"centre moves from today's {cur:.1f}x toward {target:.1f}x "
                     + (f"(own {n_mu}-year median {own_mu['p50']:.1f}x weighted {w_mu:.0%}, " if own_mu else "(")
                     + (f"peer median {ref_mu['p50']:.1f}x over {ref_mu['n']} year-ends)" if ref_mu else "own history)")
                     + f" at {lam_mu:.2f}/yr, landing at {mu_T:.1f}x; width is the blended P10/P90 ratio, capped at 0.5x to 1.8x")
        else:
            p10, p50, p90 = cur * 0.8, cur, cur * 1.25
            mnote = f"anchored on today's {cur:.1f}x, no panel; plus or minus 20 to 25%"
        p10, p50, p90 = _ensure_order(p10, p50, p90, 0.25)
        specs["ev_ebitda_multiple"] = {"type": "log_metalog", "p10": round(p10, 4), "p50": round(p50, 4), "p90": round(p90, 4), "lower": 0.0}
        notes["ev_ebitda_multiple"] = f"Exit multiple: {mnote}.{anchor_note} Your comps view goes in P50."
        refs["ev_ebitda_multiple"] = {"current": round(cur, 2), "path_end": round(p50, 2), "panel": ref_mu, "lambda": lam_mu}
    else:
        specs["ev_ebitda_multiple"] = {"type": "log_metalog", "p10": 8.0, "p50": 11.0, "p90": 15.0, "lower": 0.0}
        notes["ev_ebitda_multiple"] = "No price, so no current multiple. Placeholder; set from comps."
        refs["ev_ebitda_multiple"] = {}

    # ── Net debt drift and dilution: reuse v1 logic ──
    v1 = calibrate_drivers(profile, horizon_years)
    specs["net_debt_change_pct"] = {"type": "normal", "mean": v1["net_debt_change_pct"][0], "std": v1["net_debt_change_pct"][1]}
    specs["share_dilution_pct"] = {"type": "normal", "mean": v1["share_dilution_pct"][0], "std": v1["share_dilution_pct"][1],
                                   "clip_low": -0.10, "clip_high": 0.15}
    notes["net_debt_change_pct"] = v1["notes"]["net_debt_change_pct"]
    notes["share_dilution_pct"] = v1["notes"]["share_dilution_pct"]
    refs["net_debt_change_pct"] = v1["reference"]["net_debt_change_pct"]
    refs["share_dilution_pct"] = v1["reference"]["share_dilution_pct"]

    # ── Correlation matrix: pooled year-over-year changes where estimable ──
    base = np.array([
        [1.00, 0.40, 0.35, -0.15, 0.10],
        [0.40, 1.00, 0.25, -0.10, 0.05],
        [0.35, 0.25, 1.00, -0.05, 0.00],
        [-0.15, -0.10, -0.05, 1.00, 0.20],
        [0.10, 0.05, 0.00, 0.20, 1.00],
    ])
    cc = pn.get("corr_changes") or {}
    est = []
    for key, (i, j) in {"g_m": (0, 1), "g_mult": (0, 2), "m_mult": (1, 2)}.items():
        v = cc.get(key)
        if v is not None and np.isfinite(v):
            v = float(np.clip(v, -0.9, 0.9))
            base[i, j] = base[j, i] = v
            est.append(f"{key}={v:+.2f}")
    corr = _nearest_psd(base)
    notes["correlation_matrix"] = (
        "Estimated from pooled year-over-year changes across the panel: " + ", ".join(est)
        if est else "Generic cross-sector matrix; the panel had too few year-pairs to estimate one."
    )

    # legacy triangular tuples so older clients still run (quantile-matched)
    def tri_from(spec):
        p10, p50, p90 = spec["p10"], spec["p50"], spec["p90"]
        return (round(p10 - 0.5 * (p50 - p10), 6), round(p50, 6), round(p90 + 0.5 * (p90 - p50), 6))

    return {
        "rev_cagr": tri_from(specs["rev_cagr"]),
        "ebitda_margin": tri_from(specs["ebitda_margin"]),
        "ev_ebitda_multiple": tri_from(specs["ev_ebitda_multiple"]),
        "net_debt_change_pct": v1["net_debt_change_pct"],
        "share_dilution_pct": v1["share_dilution_pct"],
        "driver_specs": specs,
        "copula": "t",
        "copula_df": 5.0,
        "correlation_matrix": [[round(float(x), 4) for x in row] for row in corr],
        "notes": notes,
        "reference": refs,
        "panel": {k: pn.get(k) for k in ("basis", "peers_used", "as_of")} if pn else None,
        "version": "v2",
        "params": P,
    }


def _q_of(values) -> Optional[dict]:
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return None
    v = sorted(v)
    if len(v) >= 8:
        v = v[1:-1]
    arr = np.asarray(v)
    return {"p10": float(np.percentile(arr, 10)), "p50": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)), "n": int(len(v))}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: mechanisms behind the five numbers
# ─────────────────────────────────────────────────────────────────────────────
COGS_TAGS = ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"]
SGA_TAGS = ["SellingGeneralAndAdministrativeExpense"]
OPEX_PART_TAGS = ["GeneralAndAdministrativeExpense", "SellingAndMarketingExpense", "ResearchAndDevelopmentExpense"]
RPO_TAGS = ["RevenueRemainingPerformanceObligation"]
TAX_TAGS = ["IncomeTaxExpenseBenefit"]
EQUITY_TAGS = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]


def cost_structure(facts: dict, rev_annual: dict, ebitda_annual: dict, cutoff: int) -> Optional[dict]:
    """
    Gross margin history and the implied cash fixed-cost base.

    Fixed cost is defined so that revenue x gross margin - fixed cost equals
    the EBITDA the company filed in the latest year; it is "the cash operating
    cost below gross profit", not a tagged line, and that is stated in the note.
    """
    cogs = annual_series(facts, COGS_TAGS)
    years = sorted(y for y in cogs if y in rev_annual and y >= cutoff and rev_annual[y] > 0)
    if len(years) < 3:
        return None
    gm = {y: 1 - cogs[y] / rev_annual[y] for y in years}
    sga = annual_series(facts, SGA_TAGS)
    if not sga:
        parts = [annual_series(facts, [t]) for t in OPEX_PART_TAGS]
        sga = {}
        for y in years:
            vals = [p[y] for p in parts if y in p]
            if vals:
                sga[y] = sum(vals)
    opex_ratio = {y: sga[y] / rev_annual[y] for y in years if y in sga}
    latest = max(y for y in years if y in ebitda_annual) if any(y in ebitda_annual for y in years) else None
    if latest is None:
        return None
    fixed_latest = rev_annual[latest] * gm[latest] - ebitda_annual[latest]
    # fixed-cost growth history: year-over-year change in the implied fixed base
    fixed_hist = {y: rev_annual[y] * gm[y] - ebitda_annual[y] for y in years if y in ebitda_annual}
    ys = sorted(fixed_hist)
    growth = [fixed_hist[b] / fixed_hist[a] - 1 for a, b in zip(ys, ys[1:]) if b - a == 1 and fixed_hist[a] > 0]
    return {
        "gross_margin_history": {str(y): gm[y] for y in years},
        "opex_ratio_history": {str(y): opex_ratio[y] for y in sorted(opex_ratio)},
        "fixed_cost_m": fixed_latest / 1e6,
        "fixed_cost_history_m": {str(y): fixed_hist[y] / 1e6 for y in ys},
        "fixed_cost_growth": growth,
        "latest_fy": latest,
        "cogs_basis": next((t for t in COGS_TAGS if t in facts.get("facts", {}).get("us-gaap", {})), None),
    }


def backlog_from_filings(facts: dict) -> Optional[dict]:
    """Remaining performance obligation (ASC 606), the closest standard tag to backlog."""
    rpo = _point_in_time(facts, RPO_TAGS)
    if not rpo:
        return None
    return {"backlog_m": rpo[1] / 1e6, "as_of": rpo[0], "form": rpo[3], "tag": "RevenueRemainingPerformanceObligation"}


def returns_on_capital(facts: dict, rev_annual: dict, ebitda_annual: dict, da_annual: dict,
                       latest_fy: int, market_cap_m: Optional[float], net_debt_m: Optional[float]) -> dict:
    """ROIC, tax rate, D&A share of EBITDA and a WACC with stated defaults."""
    oi = annual_series(facts, OPERATING_INCOME_TAGS)
    tax = annual_series(facts, TAX_TAGS)
    pretax = annual_series(facts, PRETAX_INCOME_TAGS)
    y = latest_fy
    rate = None
    if y in tax and y in pretax and pretax[y] > 0:
        rate = float(np.clip(tax[y] / pretax[y], 0.05, 0.40))
    tax_rate = rate if rate is not None else 0.25
    ebit = oi.get(y, ebitda_annual.get(y, 0.0) - da_annual.get(y, 0.0))
    nopat = ebit * (1 - tax_rate)
    eq = _instant_series(facts, EQUITY_TAGS).get(y)
    invested = None
    if eq is not None and net_debt_m is not None:
        invested = eq + net_debt_m * 1e6
    roic = (nopat / invested) if invested and invested > 0 else None
    da_share = (da_annual[y] / ebitda_annual[y]) if y in da_annual and ebitda_annual.get(y, 0) > 0 else 0.15
    ke = None
    try:
        from .prices import cost_of_equity
        ke_info = cost_of_equity(_ticker_from_facts(facts))
        ke = ke_info["cost_of_equity"] if ke_info else None
    except Exception:
        ke = None
    ke = ke if ke is not None else 0.10
    kd = 0.06
    debt = max((net_debt_m or 0.0), 0.0)
    E = market_cap_m or 0.0
    V = E + debt
    wacc = (E / V) * ke + (debt / V) * kd * (1 - tax_rate) if V > 0 else ke
    return {
        "roic": roic, "tax_rate": tax_rate, "tax_rate_source": "filed" if rate is not None else "default 25%",
        "da_share": float(np.clip(da_share, 0.0, 0.8)), "cost_of_equity": ke, "cost_of_debt": kd, "wacc": wacc,
        "nopat_m": nopat / 1e6, "invested_capital_m": (invested / 1e6) if invested else None,
    }


def _ticker_from_facts(facts: dict) -> str:
    return facts.get("_ticker", "")


def mechanisms(profile: dict, fields: dict, drivers: dict, horizon_years: float) -> dict:
    """
    Suggested specs for each Phase 3 mechanism, with availability and notes.
    Nothing here is switched on by default; the analyst picks the KPI that
    carries the thesis and the UI sends only that block.
    """
    facts = profile.get("_facts") or {}
    out = {"available": {}, "notes": {}}
    T = float(horizon_years)
    rev_hist = {int(y): v * 1e6 for y, v in profile["revenue_history_m"].items()}
    mg_hist = {int(y): v for y, v in profile["margin_history"].items()}
    ebitda_hist = {y: rev_hist[y] * mg_hist[y] for y in rev_hist if y in mg_hist}
    cutoff = min(rev_hist) if rev_hist else 0
    da = annual_series(facts, DA_TAGS)

    # cost structure
    cs = cost_structure(facts, rev_hist, ebitda_hist, cutoff) if facts else None
    if cs:
        gm = list(cs["gross_margin_history"].values())
        g0 = gm[-1]
        q = _q_of(gm) or {"p10": g0 - 0.02, "p50": g0, "p90": g0 + 0.02}
        lo_w, hi_w = max(q["p50"] - q["p10"], 0.005), max(q["p90"] - q["p50"], 0.005)
        fg = cs["fixed_cost_growth"]
        fg_mean = float(np.median(fg)) if fg else 0.03
        fg_std = float(np.std(fg)) if len(fg) >= 3 else 0.05
        out["cost_model"] = {
            "gross_margin": {"type": "metalog", "p10": round(g0 - lo_w, 5), "p50": round(g0, 5), "p90": round(g0 + hi_w, 5)},
            "fixed_cost_m": round(cs["fixed_cost_m"], 1),
            "fixed_cost_growth": {"type": "normal", "mean": round(fg_mean, 4), "std": round(max(fg_std, 0.01), 4)},
        }
        out["available"]["cost_model"] = True
        out["notes"]["cost_model"] = (
            f"Gross margin from {cs['cogs_basis']}: {len(gm)} filed years, {min(gm)*100:.1f}% to {max(gm)*100:.1f}%, "
            f"centre at latest FY{cs['latest_fy']} {g0*100:.2f}%. Fixed cost ${cs['fixed_cost_m']:,.0f}M is the cash cost below gross "
            f"profit implied by filed EBITDA, growing {fg_mean*100:+.1f}%/yr (own history). A COGS thesis goes in the gross-margin P50."
        )
        out["cost_structure_history"] = {"gross_margin": cs["gross_margin_history"], "fixed_cost_m": cs["fixed_cost_history_m"]}
    else:
        out["available"]["cost_model"] = False
        out["notes"]["cost_model"] = "Cost of revenue is not tagged in enough years to split the margin."

    # backlog
    bl = backlog_from_filings(facts) if facts else None
    if bl:
        out["backlog"] = {"backlog_m": round(bl["backlog_m"], 1),
                          "conversion": {"type": "metalog", "p10": 0.55, "p50": 0.70, "p90": 0.85}}
        out["available"]["backlog"] = True
        cov = bl["backlog_m"] / fields["current_revenue"] if fields.get("current_revenue") else None
        out["notes"]["backlog"] = (
            f"Remaining performance obligation ${bl['backlog_m']:,.0f}M as of {bl['as_of']} ({bl['form']}), "
            f"{cov*100:.0f}% of FY revenue. " if cov else f"Remaining performance obligation ${bl['backlog_m']:,.0f}M as of {bl['as_of']}. "
        ) + "The 12-month conversion rate (P10/P50/P90 0.55/0.70/0.85) is an assumption, not a filed figure; the 10-K's backlog note usually states the share expected within a year, so set P50 to it."
    else:
        out["available"]["backlog"] = False
        out["notes"]["backlog"] = "No remaining-performance-obligation tag. Enter backlog by hand from the 10-K if the company discloses it."

    # FCF bridge
    fcf = profile.get("fcf_history_m") or {}
    if len(fcf) >= 3 and fields.get("current_revenue"):
        fm = {int(y): v / profile["revenue_history_m"][y] for y, v in fcf.items() if y in profile["revenue_history_m"]}
        vals = list(fm.values())
        q = _q_of(vals) or {"p10": vals[-1] - 0.02, "p50": vals[-1], "p90": vals[-1] + 0.02}
        latest_fm = vals[-1]
        lo_w, hi_w = max(q["p50"] - q["p10"], 0.005), max(q["p90"] - q["p50"], 0.005)
        ev = fields.get("current_ev")
        latest_fcf = list(fcf.values())[-1]
        if ev and latest_fcf > 0:
            ev_fcf = ev / latest_fcf
            mult_spec = {"type": "log_metalog", "p10": round(ev_fcf * 0.7, 2), "p50": round(ev_fcf, 2), "p90": round(ev_fcf * 1.4, 2), "lower": 0.0}
            mnote = f"EV/FCF today {ev_fcf:.1f}x on FY free cash flow ${latest_fcf:,.0f}M, width 0.7x to 1.4x of that"
        else:
            mult_spec = {"type": "log_metalog", "p10": 14.0, "p50": 20.0, "p90": 28.0, "lower": 0.0}
            mnote = "no positive FCF to anchor on; EV/FCF placeholder 14/20/28x"
        out["fcf_bridge"] = {
            "fcf_margin": {"type": "metalog", "p10": round(latest_fm - lo_w, 5), "p50": round(latest_fm, 5), "p90": round(latest_fm + hi_w, 5)},
            "ev_fcf_multiple": mult_spec,
        }
        out["available"]["fcf_bridge"] = True
        out["notes"]["fcf_bridge"] = (
            f"FCF margin (operating cash flow less capex, over revenue): {len(vals)} filed years, "
            f"{min(vals)*100:.1f}% to {max(vals)*100:.1f}%, centre at latest {latest_fm*100:.2f}%. {mnote}. "
            "Reported beside the EBITDA bridge; disagreement between the two is information."
        )
        out["fcf_margin_history"] = {str(y): v for y, v in sorted(fm.items())}
    else:
        out["available"]["fcf_bridge"] = False
        out["notes"]["fcf_bridge"] = "Cash-flow tags are not usable for this filer."

    # justified multiple
    latest = profile["latest_fy"]
    rc = returns_on_capital(facts, rev_hist, ebitda_hist, da, latest, profile.get("market_cap_m"), profile.get("net_debt_m")) if facts else {}
    if rc.get("roic") and rc["roic"] > 0 and profile.get("current_ev_ebitda"):
        g_mode = (drivers.get("driver_specs", {}).get("rev_cagr", {}) or {}).get("p50")
        if g_mode is None:
            g_mode = drivers["rev_cagr"][1]
        out["justified_multiple"] = {
            "roic": round(rc["roic"], 4), "wacc": round(rc["wacc"], 4), "tax_rate": round(rc["tax_rate"], 3),
            "da_share": round(rc["da_share"], 3), "long_run_growth": 0.03, "fade": 0.5,
            "current_growth": round(float(g_mode), 4), "current_multiple": round(profile["current_ev_ebitda"], 3),
            "premium_reversion": 0.5,
        }
        out["available"]["justified_multiple"] = True
        out["notes"]["justified_multiple"] = (
            f"ROIC {rc['roic']*100:.1f}% (NOPAT ${rc['nopat_m']:,.0f}M over invested capital ${rc['invested_capital_m']:,.0f}M, FY{latest}), "
            f"tax {rc['tax_rate']*100:.0f}% ({rc['tax_rate_source']}), WACC {rc['wacc']*100:.1f}% "
            f"(cost of equity {rc['cost_of_equity']*100:.1f}% from a 5-year weekly beta against SPY, risk-free 4.2%, ERP 4.5%; debt at 6%). "
            "The exit multiple becomes a function of drawn growth and these returns; today's premium to the justified multiple halves over the horizon. "
            "A ROIC or EVA thesis changes the multiple through this block."
        )
        out["returns"] = rc
    else:
        out["available"]["justified_multiple"] = False
        out["notes"]["justified_multiple"] = "Equity or operating income is not tagged in a form that yields a return on capital."
    return out
