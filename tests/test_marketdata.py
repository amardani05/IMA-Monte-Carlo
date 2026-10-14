"""
Market-data tests. Offline by default — the SEC parsing runs against a
synthetic company-facts fixture so the suite does not depend on EDGAR being
reachable or on a filer's tagging staying put.

Set IMA_LIVE_SEC=1 to additionally hit EDGAR for real.
"""

import os
import unittest

import numpy as np

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
        year, q = frame[2:].rstrip("I").split("Q")
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


class CalibrationV2(unittest.TestCase):
    PROFILE = {
        "revenue_history_m": {str(2016 + i): 100.0 * 1.06 ** i for i in range(10)},
        "margin_history": {str(2016 + i): m for i, m in enumerate([.05, .055, .06, .058, .062, .06, .064, .063, .065, .064])},
        "ebitda_margin_fy": 0.064,
        "current_ev_ebitda": 18.0, "current_ev_m": 1000.0, "ebitda_fy_m": 55.0,
        "public_float_latest_m": None, "multiple_history": {}, "fcf_history_m": {"2023": 30.0, "2024": 35.0, "2025": 40.0},
        "share_change_stats": None,
    }
    PANEL = {
        "basis": "test panel", "peers_used": ["A", "B", "C", "D"], "as_of": None,
        "margin": {"p10": 0.03, "p50": 0.09, "p90": 0.14, "n": 40},
        "cagr": {"p10": -0.02, "p50": 0.05, "p90": 0.15, "n": 40},
        "multiple": {"p10": 7.0, "p50": 11.0, "p90": 17.0, "n": 40},
        "margin_reversion": {"lambda": 0.30, "n_pairs": 36},
        "corr_changes": {"g_m": 0.25, "g_mult": -0.10, "m_mult": -0.30},
    }

    BLEND = {"target": "blend", "k_shrink": 8.0, "lam_mult": 0.2, "growth_centre": "blend", "width_scale": None}

    def test_margin_centre_reverts_toward_the_panel_median(self):
        d = md.calibrate_drivers_v2(self.PROFILE, self.PANEL, horizon_years=2.0, params=self.BLEND)
        p50 = d["driver_specs"]["ebitda_margin"]["p50"]
        m0 = 0.064
        w_l = 36 / (36 + 60.0)                        # the panel's rate is shrunk toward Fama-French by pair count
        lam = min(w_l * 0.30 + (1 - w_l) * 0.38, 0.5)
        own_med = md._q_of(list(self.PROFILE["margin_history"].values()))["p50"]
        w = 10 / (10 + 8.0)
        m_bar = w * own_med + (1 - w) * 0.09          # target shrinks toward the company's own long-run median
        self.assertAlmostEqual(p50, m0 + (1 - (1 - lam) ** 2) * (m_bar - m0), places=5)
        self.assertGreater(p50, m0)       # a below-target margin is lifted

    def test_growth_is_shrunk_toward_the_panel(self):
        d = md.calibrate_drivers_v2(self.PROFILE, self.PANEL, horizon_years=2.0, params=self.BLEND)
        own = md._q_of(md.rolling_cagrs(self.PROFILE["revenue_history_m"], 2))
        p50 = d["driver_specs"]["rev_cagr"]["p50"]
        self.assertLess(min(own["p50"], 0.05) - 1e-9, p50)
        self.assertLess(p50, max(own["p50"], 0.05) + 1e-9)   # between own and panel medians

    def test_multiple_moves_toward_the_panel_median_and_is_log_bounded(self):
        d = md.calibrate_drivers_v2(self.PROFILE, self.PANEL, horizon_years=2.0, params=self.BLEND)
        spec = d["driver_specs"]["ev_ebitda_multiple"]
        self.assertEqual(spec["type"], "log_metalog")
        self.assertLess(spec["p50"], 18.0); self.assertGreater(spec["p50"], 11.0)

    def test_correlation_matrix_is_estimated_and_valid(self):
        d = md.calibrate_drivers_v2(self.PROFILE, self.PANEL, horizon_years=2.0)
        c = np.array(d["correlation_matrix"])
        np.testing.assert_allclose(c, c.T)
        np.linalg.cholesky(c)                                   # positive definite
        self.assertAlmostEqual(c[0, 1], 0.25, delta=0.02)
        self.assertEqual(d["copula"], "t")

    def test_production_defaults_keep_centres_on_own_record(self):
        """The harness decision: the panel sets width, the company's own record sets the centre."""
        d = md.calibrate_drivers_v2(self.PROFILE, self.PANEL, horizon_years=2.0)
        own_g = md._q_of(md.rolling_cagrs(self.PROFILE["revenue_history_m"], 2))["p50"]
        self.assertAlmostEqual(d["driver_specs"]["rev_cagr"]["p50"], own_g, places=5)
        self.assertAlmostEqual(d["driver_specs"]["ev_ebitda_multiple"]["p50"], 18.0, places=3)   # no reversion toward peers
        own_m = md._q_of(list(self.PROFILE["margin_history"].values()))["p50"]
        lam = min((36 / 96) * 0.30 + (1 - 36 / 96) * 0.38, 0.5)
        self.assertAlmostEqual(d["driver_specs"]["ebitda_margin"]["p50"], 0.064 + (1 - (1 - lam) ** 2) * (own_m - 0.064), places=5)
        # but the panel still widens the range beyond the company's own dispersion
        own_q = md._q_of(list(self.PROFILE["margin_history"].values()))
        spec = d["driver_specs"]["ebitda_margin"]
        self.assertGreater(spec["p50"] - spec["p10"], own_q["p50"] - own_q["p10"])
        self.assertGreater(spec["p90"] - spec["p50"], own_q["p90"] - own_q["p50"])

    def test_without_a_panel_it_falls_back_cleanly(self):
        d = md.calibrate_drivers_v2(self.PROFILE, None, horizon_years=2.0)
        self.assertAlmostEqual(d["driver_specs"]["ebitda_margin"]["p50"], 0.064, places=6)
        self.assertIn("no peer panel", d["notes"]["ebitda_margin"])


class Harness(unittest.TestCase):
    def test_crps_of_a_point_mass_is_absolute_error(self):
        from montecarlo.backtest import crps_from_samples
        self.assertAlmostEqual(crps_from_samples(np.full(1000, 10.0), 13.0), 3.0, places=9)

    def test_crps_rewards_sharpness_when_centred(self):
        from montecarlo.backtest import crps_from_samples
        rng = np.random.default_rng(0)
        narrow = crps_from_samples(rng.normal(10, 1, 50_000), 10.0)
        wide = crps_from_samples(rng.normal(10, 5, 50_000), 10.0)
        self.assertLess(narrow, wide)

    def test_pit_is_the_percentile_of_the_outcome(self):
        from montecarlo.backtest import pit
        self.assertAlmostEqual(pit(np.arange(1, 101, dtype=float), 50.0), 0.5)

    def test_quarter_ends(self):
        from montecarlo.backtest import quarter_ends
        self.assertEqual(quarter_ends("2024-01-15", "2024-12-31"), ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"])

    def test_facts_as_of_drops_later_filings(self):
        f = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
            {"start": "2022-01-01", "end": "2022-12-31", "val": 100, "filed": "2023-02-20"},
            {"start": "2023-01-01", "end": "2023-12-31", "val": 120, "filed": "2024-02-20"},
        ]}}}}}
        self.assertEqual(sorted(md.annual_series(md.facts_as_of(f, "2023-06-30"), ["Revenues"])), [2022])
        self.assertEqual(sorted(md.annual_series(f, ["Revenues"])), [2022, 2023])

    def test_derived_frames_match_edgar_alignment(self):
        self.assertEqual(md._derive_frame({"start": "2024-01-28", "end": "2025-01-25"}), "CY2024")   # Dycom-style January FYE
        self.assertEqual(md._derive_frame({"start": "2024-01-01", "end": "2024-12-31"}), "CY2024")
        self.assertEqual(md._derive_frame({"start": "2024-04-01", "end": "2024-06-30"}), "CY2024Q2")
        self.assertEqual(md._derive_frame({"end": "2024-06-30"}), "CY2024Q2I")


class MultipleAnchorGuard(unittest.TestCase):
    def test_transitional_ebitda_is_not_an_anchor(self):
        prof = dict(CalibrationV2.PROFILE, current_ev_ebitda=330.0, ebitda_margin_fy=0.002,
                    margin_history={str(2016 + i): 0.002 for i in range(10)})
        d = md.calibrate_drivers_v2(prof, CalibrationV2.PANEL, horizon_years=2.0)
        spec = d["driver_specs"]["ev_ebitda_multiple"]
        self.assertLess(spec["p50"], 20.0)
        self.assertIn("not used as the anchor", d["notes"]["ev_ebitda_multiple"])


class ClaimsAheadOfCommon(unittest.TestCase):
    def test_minority_interest_is_added_to_net_debt(self):
        f = facts(LongTermDebtNoncurrent={"CY2024Q4I": 1000.0}, LongTermDebtCurrent={"CY2024Q4I": 0.0},
                  CashAndCashEquivalentsAtCarryingValue={"CY2024Q4I": 300.0}, MinorityInterest={"CY2024Q4I": 250.0})
        for tag in ("LongTermDebtNoncurrent", "LongTermDebtCurrent", "CashAndCashEquivalentsAtCarryingValue", "MinorityInterest"):
            for r in f["facts"]["us-gaap"][tag]["units"]["USD"]:
                r["end"] = "2024-12-31"; r.pop("frame", None)
        debt = md._debt(f); cash = md._point_in_time(f, md.CASH_TAGS); nci = md._point_in_time(f, md.NCI_TAGS)
        self.assertEqual(debt[0], 1000.0); self.assertEqual(cash[1], 300.0); self.assertEqual(nci[1], 250.0)


class WidthCaps(unittest.TestCase):
    def test_panel_cannot_widen_more_than_twice_own(self):
        prof = dict(CalibrationV2.PROFILE)
        wide = dict(CalibrationV2.PANEL, margin={"p10": -0.30, "p50": 0.09, "p90": 0.60, "n": 40},
                    cagr={"p10": -0.40, "p50": 0.05, "p90": 0.90, "n": 40})
        d = md.calibrate_drivers_v2(prof, wide, horizon_years=2.0)
        own_m = md._q_of(list(prof["margin_history"].values()))
        spec = d["driver_specs"]["ebitda_margin"]
        self.assertLessEqual(spec["p90"] - spec["p50"], 2 * (own_m["p90"] - own_m["p50"]) + 1e-9)
        own_g = md._q_of(md.rolling_cagrs(prof["revenue_history_m"], 2))
        g = d["driver_specs"]["rev_cagr"]
        # the fixture's revenue is exactly geometric, so own width is zero and the sampler's 1pp minimum gap applies
        self.assertLessEqual(g["p90"] - g["p50"], 2 * (own_g["p90"] - own_g["p50"]) + 0.01 + 1e-9)
