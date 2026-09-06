"""
Market-data tests. Offline by default — the SEC parsing runs against a
synthetic company-facts fixture so the suite does not depend on EDGAR being
reachable or on a filer's tagging staying put.

Set IMA_LIVE_SEC=1 to additionally hit EDGAR for real.
"""

import os
import unittest

from montecarlo import marketdata as md


def facts(**tags):
    """Build a minimal companyfacts document from {tag: {frame: value}}."""
    us_gaap, dei = {}, {}
    for tag, frames in tags.items():
        target, unit = (dei, "shares") if tag.startswith("Entity") else (us_gaap, "USD")
        target[tag] = {
            "units": {unit: [{"frame": f, "val": v, "end": _end(f)} for f, v in frames.items()]}
        }
    return {"facts": {"us-gaap": us_gaap, "dei": dei}}


def _end(frame):
    if "Q" in frame:
        year, q = frame[2:].split("Q")
        return f"{year}-{int(q) * 3:02d}-28"
    return f"{frame[2:]}-12-31"


class SeriesExtraction(unittest.TestCase):
    def test_annual_series_parses_calendar_frames(self):
        f = facts(Revenues={"CY2023": 100.0, "CY2024": 110.0, "CY2024Q1": 25.0})
        self.assertEqual(md.annual_series(f, ["Revenues"]), {2023: 100.0, 2024: 110.0})

    def test_later_tags_fill_gaps_left_by_earlier_ones(self):
        """A filer that switched tags mid-history must not lose the early years."""
        f = facts(
            RevenueFromContractWithCustomerExcludingAssessedTax={"CY2023": 300.0, "CY2024": 310.0},
            Revenues={"CY2021": 100.0, "CY2022": 200.0, "CY2023": 999.0},
        )
        series = md.annual_series(f, md.REVENUE_TAGS)
        self.assertEqual(series[2021], 100.0)          # only the legacy tag has it
        self.assertEqual(series[2023], 300.0)          # preferred tag wins on conflict
        self.assertEqual(sorted(series), [2021, 2022, 2023, 2024])

    def test_quarterly_series_keys_on_year_and_quarter(self):
        f = facts(Revenues={"CY2024Q1": 10.0, "CY2024Q2": 12.0})
        self.assertEqual(md.quarterly_series(f, ["Revenues"]), {(2024, 1): 10.0, (2024, 2): 12.0})


class TrailingTwelveMonths(unittest.TestCase):
    def test_sums_four_tagged_quarters(self):
        q = {(2024, 3): 10.0, (2024, 4): 11.0, (2025, 1): 12.0, (2025, 2): 13.0}
        value, label = md.trailing_twelve_months({}, q)
        self.assertEqual(value, 46.0)
        self.assertIn("2025Q2", label)

    def test_reconstructs_a_missing_q4_from_the_full_year(self):
        """Filers usually leave Q4 untagged; it is the year less the first three."""
        annual = {2024: 100.0}
        q = {(2024, 1): 20.0, (2024, 2): 25.0, (2024, 3): 25.0,
             (2025, 1): 30.0, (2025, 2): 35.0}
        value, _ = md.trailing_twelve_months(annual, q)
        self.assertAlmostEqual(value, 25.0 + 30.0 + 30.0 + 35.0)  # Q4 2024 = 100-70 = 30

    def test_returns_none_when_a_quarter_cannot_be_assembled(self):
        self.assertIsNone(md.trailing_twelve_months({}, {(2025, 1): 10.0, (2025, 2): 12.0}))
        self.assertIsNone(md.trailing_twelve_months({}, {}))


class EbitdaFallbacks(unittest.TestCase):
    REV = {2023: 1000.0, 2024: 1100.0}
    DA = {2023: 50.0, 2024: 55.0}

    def test_prefers_operating_income(self):
        f = facts(OperatingIncomeLoss={"CY2023": 100.0, "CY2024": 120.0})
        hist, basis = md._ebitda_history(f, self.REV, self.DA)
        self.assertEqual(hist[2024], 175.0)
        self.assertIn("operating income", basis)

    def test_falls_back_to_pretax_plus_interest(self):
        """Dycom tags no operating income at all."""
        f = facts(
            IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest={
                "CY2023": 80.0, "CY2024": 90.0},
            InterestExpense={"CY2023": 10.0, "CY2024": 12.0},
        )
        hist, basis = md._ebitda_history(f, self.REV, self.DA)
        self.assertEqual(hist[2024], 90.0 + 12.0 + 55.0)
        self.assertIn("pre-tax", basis)

    def test_reports_unavailable_when_nothing_resolves(self):
        hist, basis = md._ebitda_history(facts(), self.REV, self.DA)
        self.assertEqual(hist, {})
        self.assertEqual(basis, "unavailable")


class Calibration(unittest.TestCase):
    def test_rolling_cagrs_span_the_requested_window(self):
        hist = {"2020": 100.0, "2021": 110.0, "2022": 121.0, "2023": 133.1}
        two_year = md.rolling_cagrs(hist, 2)
        self.assertEqual(len(two_year), 2)
        for c in two_year:
            self.assertAlmostEqual(c, 0.10, places=6)

    def test_rolling_cagrs_ignore_non_positive_revenue(self):
        self.assertEqual(md.rolling_cagrs({"2020": 0.0, "2022": 100.0}, 2), [])

    def test_widen_guarantees_a_usable_triangle(self):
        low, mode, high = md._widen(0.05, 0.05, 0.05)
        self.assertLess(low, mode)
        self.assertLess(mode, high)

    def test_calibration_uses_filed_history_not_a_guess(self):
        profile = {
            "revenue_history_m": {"2019": 100.0, "2020": 110.0, "2021": 121.0,
                                  "2022": 133.1, "2023": 146.4, "2024": 161.1},
            "margin_history": {"2022": 0.04, "2023": 0.05, "2024": 0.06},
            "ebitda_margin_fy": 0.06,
            "current_ev_ebitda": 12.0,
            "share_change_stats": None,
        }
        drivers = md.calibrate_drivers(profile, horizon_years=2.0)
        low, mode, high = drivers["ebitda_margin"]
        self.assertAlmostEqual(low, 0.04)     # the worst year actually filed
        self.assertAlmostEqual(high, 0.06)
        self.assertIn("filed years", drivers["notes"]["ebitda_margin"])
        # the multiple is analyst judgement and must say so
        self.assertIn("comps view", drivers["notes"]["ev_ebitda_multiple"])
        self.assertAlmostEqual(drivers["ev_ebitda_multiple"][1], 12.0)
        # reference block exposes both anchors so the mode can be chosen deliberately
        ref = drivers["reference"]["ebitda_margin"]
        self.assertAlmostEqual(ref["current"], 0.06)
        self.assertAlmostEqual(ref["hist_median"], 0.05)

    def test_thin_history_falls_back_and_says_so(self):
        profile = {
            "revenue_history_m": {"2023": 100.0},
            "margin_history": {"2023": 0.05},
            "ebitda_margin_fy": 0.05,
            "current_ev_ebitda": None,
            "share_change_stats": None,
        }
        drivers = md.calibrate_drivers(profile, horizon_years=2.0)
        self.assertIn("Insufficient", drivers["notes"]["rev_cagr"])
        self.assertIn("Placeholder", drivers["notes"]["ev_ebitda_multiple"])


class RobustBounds(unittest.TestCase):
    def test_trims_one_extreme_each_side_when_history_is_long_enough(self):
        vals = [-0.34, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.60]
        lo, med, hi = md._robust_bounds(vals)
        self.assertAlmostEqual(lo, 0.05)     # the -34% impairment year is gone
        self.assertAlmostEqual(hi, 0.12)     # so is the +60% outlier
        self.assertAlmostEqual(med, 0.085)

    def test_keeps_everything_when_history_is_short(self):
        lo, med, hi = md._robust_bounds([0.01, 0.05, 0.09])
        self.assertEqual((lo, hi), (0.01, 0.09))

    def test_widen_extends_range_to_include_an_out_of_range_mode(self):
        """Current margin above every filed year must still sit inside the triangle."""
        low, mode, high = md._widen(0.03, 0.08, 0.06)
        self.assertLessEqual(low, 0.08)
        self.assertGreaterEqual(high, 0.08)
        self.assertLess(low, mode)
        self.assertLess(mode, high)


class DriftAndMultiple(unittest.TestCase):
    def _profile(self, **over):
        base = {
            "revenue_history_m": {str(2016 + i): 100.0 * 1.05 ** i for i in range(10)},
            "margin_history": {str(2016 + i): 0.10 for i in range(10)},
            "ebitda_margin_fy": 0.10,
            "current_ev_ebitda": 12.0,
            "current_ev_m": 1000.0,
            "ebitda_fy_m": 83.0,
            "public_float_latest_m": None,
            "multiple_history": {},
            "fcf_history_m": {},
            "share_change_stats": None,
        }
        base.update(over)
        return base

    def test_fcf_drift_is_negative_for_a_cash_generative_business(self):
        d = md.calibrate_drivers(self._profile(fcf_history_m={"2023": 50.0, "2024": 60.0, "2025": 70.0}), 2.0)
        mean, std = d["net_debt_change_pct"]
        self.assertLess(mean, 0)                    # cash builds, net debt falls
        self.assertAlmostEqual(mean, -0.06 * 2, places=3)  # 6% of EV a year for two years
        self.assertFalse(d["reference"]["net_debt_change_pct"]["capped"])

    def test_fcf_drift_is_capped_when_ev_is_cash_dominated(self):
        """TDS: $500M a year of FCF against a $387M EV."""
        d = md.calibrate_drivers(self._profile(current_ev_m=387.0,
                                               fcf_history_m={"2023": 498.0, "2024": 780.0, "2025": 199.0}), 2.0)
        mean, std = d["net_debt_change_pct"]
        self.assertGreaterEqual(mean, -0.60)
        self.assertLessEqual(std, 0.15)
        self.assertTrue(d["reference"]["net_debt_change_pct"]["capped"])
        self.assertIn("Capped", d["notes"]["net_debt_change_pct"])

    def test_multiple_width_is_symmetric_without_history(self):
        d = md.calibrate_drivers(self._profile(), 2.0)
        lo, mode, hi = d["ev_ebitda_multiple"]
        self.assertAlmostEqual(mode, 12.0)
        self.assertAlmostEqual(mode - lo, hi - mode, places=4)

    def test_multiple_history_is_bounded_to_half_and_one_and_a_half_times_today(self):
        """Depressed-earnings years must not set a 40x ceiling on a 15x stock."""
        hist = {str(2016 + i): v for i, v in enumerate([30.0, 35.0, 40.0, 38.0, 33.0, 36.0, 31.0])}
        d = md.calibrate_drivers(self._profile(multiple_history=hist, public_float_latest_m=900.0,
                                               net_debt_m=0.0), 2.0)
        lo, mode, hi = d["ev_ebitda_multiple"]
        self.assertLessEqual(hi, 12.0 * 1.5 + 1e-6)
        self.assertGreaterEqual(lo, 12.0 * 0.5 - 1e-6)


class TickerResolution(unittest.TestCase):
    def test_bundled_map_is_present_and_populated(self):
        self.assertGreater(len(md._bundled_map()), 5000)

    def test_known_ticker_resolves_without_network(self):
        cik, name = md.resolve_cik("myrg")
        self.assertEqual(cik, 700923)
        self.assertIn("MYR", name.upper())

    def test_malformed_symbols_are_rejected(self):
        for bad in ["", "  ", "NOT A TICKER", "TOOLONGTICKER", "../etc/passwd"]:
            with self.assertRaises(md.DataError):
                md.resolve_cik(bad)


class IndustryGuard(unittest.TestCase):
    def test_banks_are_rejected(self):
        with self.assertRaises(md.DataError) as ctx:
            md._check_modelable("JPM", {"sic": 6021, "sic_description": "National Commercial Banks"})
        self.assertIn("not meaningful", str(ctx.exception))

    def test_reits_warn_rather_than_fail(self):
        warnings = md._check_modelable("AMT", {"sic": 6798, "sic_description": "REIT"})
        self.assertEqual(len(warnings), 1)
        self.assertIn("FFO", warnings[0])

    def test_ordinary_industrials_pass_clean(self):
        self.assertEqual(md._check_modelable("MYRG", {"sic": 1731, "sic_description": "Electrical Work"}), [])


@unittest.skipUnless(os.environ.get("IMA_LIVE_SEC"), "set IMA_LIVE_SEC=1 to hit EDGAR")
class LiveEdgar(unittest.TestCase):
    def test_myrg_prefill_is_internally_consistent(self):
        data = md.build_prefill("MYRG", price=271.0)
        f, c = data["fields"], data["company"]
        # revenue x margin must equal the EBITDA actually filed
        self.assertAlmostEqual(f["current_revenue"] * f["current_ebitda_margin"],
                               c["ebitda_fy_m"], delta=0.5)
        self.assertGreater(len(data["history"]["revenue_m"]), 5)
        self.assertFalse(data["needs_price"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
