"""Engine and validation tests. No network — pure numerics."""

import unittest

import numpy as np

from montecarlo import MonteCarloEngine, PitchAssumptions
from montecarlo.service import ValidationError, run_simulation

MYRG = dict(
    ticker="MYRG", company_name="MYR Group", current_price=271.0,
    bear_price=222.32, base_price=323.01, bull_price=362.32,
    current_revenue=3510.0, current_ebitda_margin=0.061, current_ev=4310.0,
    current_net_debt=110.0, shares_outstanding=15.5,
    rev_cagr=(0.057, 0.069, 0.075), ebitda_margin=(0.071, 0.075, 0.077),
    ev_ebitda_multiple=(12.4, 17.0, 18.5), net_debt_change_pct=(0.0, 0.015),
    share_dilution_pct=(0.01, 0.01), n_simulations=100_000, random_seed=42,
)


class GoldenValues(unittest.TestCase):
    """Pins the published MYRG numbers so silent numeric drift fails loudly."""

    @classmethod
    def setUpClass(cls):
        cls.engine = MonteCarloEngine(PitchAssumptions(**MYRG))
        cls.engine.run()
        cls.probs = cls.engine.compute_case_probabilities()

    def test_median(self):
        self.assertAlmostEqual(self.probs["median_price"], 298.76, places=1)

    def test_case_probabilities(self):
        self.assertAlmostEqual(self.probs["p_at_least_base"], 0.1784, places=3)
        self.assertAlmostEqual(self.probs["below_bear"], 0.0020, places=3)

    def test_probabilities_partition(self):
        buckets = ["below_bear", "bear_to_base", "base_to_bull", "above_bull"]
        self.assertAlmostEqual(sum(self.probs[b] for b in buckets), 1.0, places=9)

    def test_seed_is_reproducible(self):
        again = MonteCarloEngine(PitchAssumptions(**MYRG))
        np.testing.assert_array_equal(again.run(), self.engine.terminal_prices)

    def test_left_skewed_so_mean_sits_below_median(self):
        self.assertLess(self.probs["mean_price"], self.probs["median_price"])


class SingleUnitBridge(unittest.TestCase):
    """The bridge arithmetic, checked against a hand-computed path."""

    def test_degenerate_drivers_reproduce_hand_calculation(self):
        eps = 1e-9
        a = PitchAssumptions(
            ticker="T", company_name="T", current_price=100.0,
            bear_price=1.0, base_price=2.0, bull_price=3.0,
            current_revenue=1000.0, current_ebitda_margin=0.10,
            current_ev=5000.0, current_net_debt=500.0, shares_outstanding=100.0,
            rev_cagr=(0.10 - eps, 0.10, 0.10 + eps),
            ebitda_margin=(0.20 - eps, 0.20, 0.20 + eps),
            ev_ebitda_multiple=(10.0 - eps, 10.0, 10.0 + eps),
            net_debt_change_pct=(0.0, 1e-9),
            share_dilution_pct=(0.0, 1e-9),
            correlation_matrix=np.eye(5),
            n_simulations=2000, random_seed=7, horizon_years=2.0,
        )
        prices = MonteCarloEngine(a).run()
        # revenue 1000*1.1^2 = 1210; EBITDA 242; EV 2420; equity 2420-500 = 1920
        self.assertAlmostEqual(float(np.median(prices)), 1920.0 / 100.0, places=4)

    def test_horizon_compounds_revenue(self):
        base = dict(MYRG, n_simulations=20_000)
        one = MonteCarloEngine(PitchAssumptions(**base, horizon_years=1.0)).run()
        three = MonteCarloEngine(PitchAssumptions(**base, horizon_years=3.0)).run()
        self.assertLess(np.median(one), np.median(three))


class SamplerProperties(unittest.TestCase):
    def test_triangular_recovers_analytic_mean(self):
        engine = MonteCarloEngine(PitchAssumptions(**MYRG))
        u = np.linspace(1e-9, 1 - 1e-9, 400_000)
        low, mode, high = 2.0, 5.0, 11.0
        drawn = engine._sample_triangular(u, low, mode, high)
        self.assertAlmostEqual(float(drawn.mean()), (low + mode + high) / 3, places=2)
        self.assertGreaterEqual(float(drawn.min()), low - 1e-6)
        self.assertLessEqual(float(drawn.max()), high + 1e-6)

    def test_copula_recovers_target_correlation(self):
        engine = MonteCarloEngine(PitchAssumptions(**MYRG))
        u = engine._generate_correlated_uniforms(200_000)
        target = MYRG_CORR = PitchAssumptions(**MYRG).correlation_matrix
        # Gaussian copula preserves rank order, so compare on the normal scale.
        from scipy import stats as st
        z = st.norm.ppf(u)
        observed = np.corrcoef(z, rowvar=False)
        np.testing.assert_allclose(observed, target, atol=0.02)

    def test_dilution_is_clipped(self):
        a = PitchAssumptions(**dict(MYRG, share_dilution_pct=(0.0, 0.5), n_simulations=50_000))
        engine = MonteCarloEngine(a)
        engine.run()
        d = engine.driver_samples["Share Dilution (%)"]
        self.assertGreaterEqual(float(d.min()), -0.10 - 1e-9)
        self.assertLessEqual(float(d.max()), 0.15 + 1e-9)

    def test_prices_never_negative(self):
        a = PitchAssumptions(**dict(MYRG, ev_ebitda_multiple=(0.01, 0.02, 0.03), n_simulations=20_000))
        self.assertGreaterEqual(float(MonteCarloEngine(a).run().min()), 0.0)


class Validation(unittest.TestCase):
    def _err(self, **overrides):
        payload = dict(MYRG, n_simulations=2000)
        payload.update(overrides)
        payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in payload.items()}
        with self.assertRaises(ValidationError) as ctx:
            run_simulation(payload)
        return str(ctx.exception)

    def test_triangular_ordering(self):
        self.assertIn("low <= mode <= high", self._err(rev_cagr=[0.1, 0.05, 0.2]))

    def test_zero_width_triangular(self):
        self.assertIn("high > low", self._err(rev_cagr=[0.05, 0.05, 0.05]))

    def test_std_must_be_positive(self):
        self.assertIn("std > 0", self._err(share_dilution_pct=[0.01, 0.0]))

    def test_case_target_ordering(self):
        self.assertIn("bear <= base <= bull", self._err(bull_price=100.0))

    def test_simulation_cap(self):
        self.assertIn("capped", self._err(n_simulations=10_000_000))

    def test_missing_field(self):
        payload = {k: v for k, v in MYRG.items() if k != "bear_price"}
        payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in payload.items()}
        with self.assertRaises(ValidationError) as ctx:
            run_simulation(payload)
        self.assertIn("bear_price", str(ctx.exception))

    def test_non_positive_definite_correlation_matrix(self):
        bad = [[1, .99, .99, .99, .99], [.99, 1, -.99, .99, .99], [.99, -.99, 1, .99, .99],
               [.99, .99, .99, 1, .99], [.99, .99, .99, .99, 1]]
        self.assertIn("positive definite", self._err(correlation_matrix=bad))

    def test_asymmetric_correlation_matrix(self):
        bad = np.eye(5).tolist()
        bad[0][1] = 0.5
        self.assertIn("symmetric", self._err(correlation_matrix=bad))

    def test_shares_must_be_positive(self):
        self.assertIn("greater than zero", self._err(shares_outstanding=0))


class Reconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in MYRG.items()}
        cls.result = run_simulation(payload)

    def test_implied_multiple_reproduces_the_target(self):
        """Plugging the implied multiple back into the mode path must land on the target."""
        a = PitchAssumptions(**MYRG)
        rc = self.result["reconciliation"]
        base = rc["cases"]["base"]
        rev = a.current_revenue * (1 + a.rev_cagr[1]) ** a.horizon_years
        nd = a.current_net_debt + a.current_ev * a.net_debt_change_pct[0]
        sh = a.shares_outstanding * (1 + a.share_dilution_pct[0])
        price = (rev * a.ebitda_margin[1] * base["implied_multiple_at_mode_margin"] - nd) / sh
        self.assertAlmostEqual(price, a.base_price, places=6)

    def test_implied_margin_reproduces_the_target(self):
        a = PitchAssumptions(**MYRG)
        bull = self.result["reconciliation"]["cases"]["bull"]
        rev = a.current_revenue * (1 + a.rev_cagr[1]) ** a.horizon_years
        nd = a.current_net_debt + a.current_ev * a.net_debt_change_pct[0]
        sh = a.shares_outstanding * (1 + a.share_dilution_pct[0])
        price = (rev * bull["implied_margin_at_mode_multiple"] * a.ev_ebitda_multiple[1] - nd) / sh
        self.assertAlmostEqual(price, a.bull_price, places=6)

    def test_case_probabilities_agree_with_stats(self):
        rc, st = self.result["reconciliation"], self.result["stats"]
        self.assertAlmostEqual(rc["cases"]["base"]["p_at_least"], st["p_at_least_base"], places=9)

    def test_standard_errors_are_reported_and_small(self):
        se = self.result["stats"]["se"]
        p = self.result["stats"]["p_at_least_base"]
        self.assertAlmostEqual(se["p_at_least_base"], (p * (1 - p) / 100_000) ** 0.5, places=9)
        self.assertLess(se["p_at_least_base"], 0.002)

    def test_annualised_return_compounds_to_total(self):
        st = self.result["stats"]
        h = MYRG.get("horizon_years", 2.0)
        self.assertAlmostEqual((1 + st["median_return_annualized"]) ** h - 1, st["median_return"], places=9)


class ResponseShape(unittest.TestCase):
    def test_payload_is_aggregated_not_raw(self):
        payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in MYRG.items()}
        result = run_simulation(payload)
        self.assertEqual(len(result["histogram"]["density"]), 100)
        self.assertEqual(len(result["cdf"]["x"]), 400)
        self.assertEqual(len(result["tornado"]), 5)
        # the 100k-element price vector must never be serialised
        self.assertNotIn("terminal_prices", result)
        import json
        self.assertLess(len(json.dumps(result)), 60_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class Phase2Options(unittest.TestCase):
    """Quantile-parameterised drivers and the t-copula, behind flags."""

    def test_legacy_defaults_unchanged(self):
        a = PitchAssumptions(**MYRG)
        self.assertIsNone(a.driver_specs)
        self.assertEqual(a.copula, "gaussian")

    def test_metalog_driver_hits_its_quantiles(self):
        specs = {"ebitda_margin": {"type": "metalog", "p10": 0.04, "p50": 0.065, "p90": 0.09}}
        a = PitchAssumptions(**dict(MYRG, n_simulations=200_000), driver_specs=specs)
        e = MonteCarloEngine(a); e.run()
        m = e.driver_samples["EBITDA Margin"]
        q = np.percentile(m, [10, 50, 90])
        np.testing.assert_allclose(q, [0.04, 0.065, 0.09], atol=0.002)

    def test_t_copula_keeps_correlation_and_fattens_joint_tail(self):
        g = MonteCarloEngine(PitchAssumptions(**dict(MYRG, n_simulations=300_000)))
        t = MonteCarloEngine(PitchAssumptions(**dict(MYRG, n_simulations=300_000), copula="t", copula_df=4.0))
        ug, ut = g._generate_correlated_uniforms(300_000), t._generate_correlated_uniforms(300_000)
        from scipy import stats as st
        rg = st.spearmanr(ug[:, 0], ug[:, 1]).correlation
        rt = st.spearmanr(ut[:, 0], ut[:, 1]).correlation
        self.assertAlmostEqual(rg, rt, delta=0.03)       # same rank correlation
        joint_g = np.mean((ug[:, 0] < 0.05) & (ug[:, 1] < 0.05))
        joint_t = np.mean((ut[:, 0] < 0.05) & (ut[:, 1] < 0.05))
        self.assertGreater(joint_t, joint_g * 1.3)       # but more joint downside

    def test_service_validates_specs_and_copula(self):
        payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in MYRG.items()}
        payload["n_simulations"] = 2000
        with self.assertRaises(ValidationError):
            run_simulation(dict(payload, copula="clayton"))
        with self.assertRaises(ValidationError):
            run_simulation(dict(payload, driver_specs={"ebitda_margin": {"type": "metalog", "p10": 0.1, "p50": 0.05, "p90": 0.2}}))
        ok = run_simulation(dict(payload, copula="t", copula_df=5,
                                 driver_specs={"rev_cagr": {"type": "metalog", "p10": 0.0, "p50": 0.06, "p90": 0.15}}))
        self.assertEqual(ok["meta"]["copula"], "t")
        self.assertAlmostEqual(ok["meta"]["driver_quantiles"]["rev_cagr"]["p50"], 0.06, places=6)


class Phase3Mechanisms(unittest.TestCase):
    EPS = 1e-9

    def _deg(self, **over):
        """All drivers pinned so mechanisms can be checked by hand."""
        e = self.EPS
        base = dict(
            ticker="T", company_name="T", current_price=100.0, bear_price=1.0, base_price=2.0, bull_price=3.0,
            current_revenue=1000.0, current_ebitda_margin=0.10, current_ev=5000.0, current_net_debt=500.0,
            shares_outstanding=100.0,
            rev_cagr=(0.10 - e, 0.10, 0.10 + e), ebitda_margin=(0.20 - e, 0.20, 0.20 + e),
            ev_ebitda_multiple=(10.0 - e, 10.0, 10.0 + e), net_debt_change_pct=(0.0, 1e-9),
            share_dilution_pct=(0.0, 1e-9), correlation_matrix=np.eye(5), n_simulations=2000,
            random_seed=3, horizon_years=2.0,
        )
        base.update(over)
        return PitchAssumptions(**base)

    def test_backlog_covers_year_one(self):
        # backlog 660 at 100% conversion covers 60% of organic year-one revenue (1100)
        a = self._deg(backlog={"backlog_m": 660.0, "conversion": {"type": "normal", "mean": 1.0, "std": 1e-9}})
        e = MonteCarloEngine(a); e.run()
        rev1 = 660.0 + 1100.0 * (1 - 0.66)      # coverage = 660/1000
        self.assertAlmostEqual(float(np.median(e.terminal_revenue)), rev1 * 1.10, places=3)

    def test_cost_structure_makes_margin_an_output(self):
        a = self._deg(cost_model={"gross_margin": {"type": "normal", "mean": 0.40, "std": 1e-9},
                                  "fixed_cost_m": 200.0,
                                  "fixed_cost_growth": {"type": "normal", "mean": 0.0, "std": 1e-9}})
        e = MonteCarloEngine(a); e.run()
        rev = 1000 * 1.1 ** 2
        self.assertAlmostEqual(float(np.median(e.driver_samples["EBITDA Margin"])), (rev * 0.4 - 200) / rev, places=6)
        self.assertAlmostEqual(float(np.median(e.terminal_prices)), ((rev * 0.4 - 200) * 10 - 500) / 100, places=4)

    def test_fcf_bridge_is_a_second_price(self):
        a = self._deg(fcf_bridge={"fcf_margin": {"type": "normal", "mean": 0.08, "std": 1e-9},
                                  "ev_fcf_multiple": {"type": "normal", "mean": 20.0, "std": 1e-9}})
        e = MonteCarloEngine(a); e.run()
        rev = 1000 * 1.1 ** 2
        self.assertAlmostEqual(float(np.median(e.terminal_prices_fcf)), (rev * 0.08 * 20 - 500) / 100, places=4)
        self.assertAlmostEqual(float(np.median(e.terminal_prices)), (rev * 0.2 * 10 - 500) / 100, places=4)

    def test_justified_multiple_moves_with_growth(self):
        jm = {"roic": 0.15, "wacc": 0.09, "tax_rate": 0.25, "da_share": 0.15, "long_run_growth": 0.03,
              "fade": 0.5, "current_growth": 0.10, "current_multiple": 10.0, "premium_reversion": 1.0}
        lo = MonteCarloEngine(self._deg(rev_cagr=(0.02 - 1e-9, 0.02, 0.02 + 1e-9), justified_multiple=jm)); lo.run()
        hi = MonteCarloEngine(self._deg(rev_cagr=(0.10 - 1e-9, 0.10, 0.10 + 1e-9), justified_multiple=jm)); hi.run()
        m_lo = float(np.median(lo.driver_samples["EV/EBITDA Multiple"]))
        m_hi = float(np.median(hi.driver_samples["EV/EBITDA Multiple"]))
        self.assertGreater(m_hi, m_lo)
        # with full premium reversion the multiple is the justified one; check the formula at g = 10%
        g_term = 0.03 + 0.5 * (0.10 - 0.03)
        expect = (1 - g_term / 0.15) * 0.75 * 0.85 / (0.09 - g_term)
        self.assertAlmostEqual(m_hi, expect, places=4)

    def test_golden_path_untouched_by_new_fields(self):
        a = PitchAssumptions(**MYRG)
        self.assertIsNone(a.cost_model); self.assertIsNone(a.backlog)
        self.assertIsNone(a.fcf_bridge); self.assertIsNone(a.justified_multiple)


class Phase4Service(unittest.TestCase):
    def _payload(self, **over):
        p = {k: (list(v) if isinstance(v, tuple) else v) for k, v in MYRG.items()}
        p["n_simulations"] = 20_000
        p.update(over)
        return p

    def test_price_implied_margin_reproduces_spot_on_the_mode_path(self):
        from montecarlo.service import mode_path_price, price_implied
        a = PitchAssumptions(**MYRG)
        pi = price_implied(a)
        m = pi["implied_margin_at_mode_growth_and_multiple"]
        self.assertAlmostEqual(mode_path_price(a, m=m), a.current_price, places=4)
        g = pi["implied_growth_at_mode_margin_and_multiple"]
        self.assertAlmostEqual(mode_path_price(a, g=g), a.current_price, places=4)

    def test_dcf_path_percentile_is_consistent_with_distribution(self):
        r = run_simulation(self._payload(dcf_path={"rev_cagr": 0.069, "ebitda_margin": 0.075, "ev_ebitda_multiple": 17.0}))
        d = r["dcf_path"]
        # the modes themselves: price equals the mode-path price from reconciliation
        self.assertAlmostEqual(d["price"], r["reconciliation"]["mode_path_price"], places=6)
        self.assertTrue(0 <= d["percentile"] <= 1)
        self.assertAlmostEqual(d["percentile"] + d["p_at_least"], 1.0, places=9)

    def test_fcf_bridge_reported_when_requested(self):
        r = run_simulation(self._payload(fcf_bridge={"fcf_margin": {"type": "metalog", "p10": 0.04, "p50": 0.06, "p90": 0.08},
                                                     "ev_fcf_multiple": {"type": "log_metalog", "p10": 14, "p50": 18, "p90": 24}}))
        fb = r["fcf_bridge"]
        self.assertIsNotNone(fb)
        self.assertGreater(fb["median_price"], 0)
        self.assertTrue(r["meta"]["mechanisms"]["fcf_bridge"])
        self.assertIsNone(run_simulation(self._payload())["fcf_bridge"])

    def test_mechanism_validation(self):
        with self.assertRaises(ValidationError):
            run_simulation(self._payload(backlog={"backlog_m": -5, "conversion": {"type": "normal", "mean": .7, "std": .1}}))
        with self.assertRaises(ValidationError):
            run_simulation(self._payload(justified_multiple={"roic": 0.1, "wacc": 0.02, "long_run_growth": 0.03}))
        with self.assertRaises(ValidationError):
            run_simulation(self._payload(cost_model={"gross_margin": {"type": "metalog", "p10": 0.3, "p50": 0.2, "p90": 0.4}, "fixed_cost_m": 10}))


class HorizonTermStructure(unittest.TestCase):
    """
    The distribution must open out from spot as the horizon grows. A status-quo
    pitch (centres at today's margin and multiple, EV / EBITDA = 1,200 / 100,
    spot 100) with the flows stated over the horizon, as calibration states
    net debt.
    """
    SPOT = 100.0

    def _pitch(self, T, **over):
        base = dict(
            ticker="T", company_name="T", current_price=self.SPOT, bear_price=80.0, base_price=115.0, bull_price=140.0,
            current_revenue=1000.0, current_ebitda_margin=0.10, current_ev=1200.0, current_net_debt=200.0,
            shares_outstanding=10.0,
            rev_cagr=(0.0, 0.06, 0.12), ebitda_margin=(0.07, 0.10, 0.13), ev_ebitda_multiple=(8.0, 12.0, 17.0),
            net_debt_change_pct=(-0.03 * T, 0.03 * T ** 0.5), share_dilution_pct=(0.01 * T, 0.01 * T ** 0.5),
            driver_specs={"rev_cagr": {"type": "metalog", "p10": 0.0, "p50": 0.06, "p90": 0.12},
                          "ebitda_margin": {"type": "metalog", "p10": 0.08, "p50": 0.10, "p90": 0.12},
                          "ev_ebitda_multiple": {"type": "log_metalog", "p10": 9.0, "p50": 12.0, "p90": 16.0, "lower": 0.0}},
            copula="t", n_simulations=50_000, random_seed=7, horizon_years=T, level_width_horizon=2.0,
        )
        base.update(over)
        return PitchAssumptions(**base)

    def _run(self, T, **over):
        e = MonteCarloEngine(self._pitch(T, **over))
        e.run()
        return e

    def test_width_grows_with_horizon(self):
        backlog = {"backlog_m": 600.0, "conversion": {"type": "metalog", "p10": 0.55, "p50": 0.70, "p90": 0.85}}
        for over in ({}, {"backlog": backlog}):
            widths = []
            for T in (0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
                p5, p95 = np.percentile(self._run(T, **over).terminal_prices, [5, 95])
                widths.append(p95 - p5)
            self.assertTrue(all(b > a for a, b in zip(widths, widths[1:])), (over.keys(), widths))

    def test_short_horizon_collapses_to_spot(self):
        # T -> 0 is the limit: at 1e-4 years root-T leaves 0.7% of the level width
        p5, p50, p95 = np.percentile(self._run(1e-4).terminal_prices, [5, 50, 95])
        # (the median of a product of skewed draws sits off the product of medians by O(s) too)
        self.assertAlmostEqual(p50 / self.SPOT, 1.0, delta=0.005)
        self.assertGreater(p5, 0.99 * self.SPOT)
        self.assertLess(p95, 1.01 * self.SPOT)
        # the stated width at every horizon (legacy) does not close on spot at all
        l5, l95 = np.percentile(self._run(1e-4, level_width_horizon=None).terminal_prices, [5, 95])
        self.assertGreater(l95 - l5, 50 * (p95 - p5))
        # and on the way down it closes at the root-T rate: 0.01 years is about a tenth of 1 year
        w = {T: np.subtract(*np.percentile(self._run(T).terminal_prices, [95, 5])) for T in (0.01, 1.0)}
        self.assertLess(w[0.01] / w[1.0], 0.15)

    def test_stated_width_holds_at_and_beyond_the_reference(self):
        for T in (2.0, 3.0):
            self.assertTrue(np.array_equal(self._run(T).terminal_prices,
                                           self._run(T, level_width_horizon=None).terminal_prices))

    def test_level_drivers_keep_their_medians_and_shrink_by_root_t(self):
        short, legacy = self._run(0.5), self._run(0.5, level_width_horizon=None)
        for name in ("EBITDA Margin", "EV/EBITDA Multiple"):
            s, l = short.driver_samples[name], legacy.driver_samples[name]
            self.assertAlmostEqual(float(np.median(s)), float(np.median(l)), places=10)
            ws, wl = np.subtract(*np.percentile(s, [90, 10])), np.subtract(*np.percentile(l, [90, 10]))
            self.assertAlmostEqual(ws / wl, 0.5, places=9)       # sqrt(0.5 / 2)
        # revenue already compounds over T and is left alone
        self.assertTrue(np.array_equal(short.terminal_revenue, legacy.terminal_revenue))

    def test_mechanisms_scale_and_stay_finite(self):
        over = dict(cost_model={"gross_margin": {"type": "metalog", "p10": 0.35, "p50": 0.40, "p90": 0.45},
                                "fixed_cost_m": 300.0},
                    fcf_bridge={"fcf_margin": {"type": "metalog", "p10": 0.04, "p50": 0.06, "p90": 0.08},
                                "ev_fcf_multiple": {"type": "log_metalog", "p10": 14.0, "p50": 20.0, "p90": 28.0, "lower": 0.0}})
        short, legacy = self._run(0.25, **over), self._run(0.25, level_width_horizon=None, **over)
        for attr in ("terminal_prices", "terminal_prices_fcf"):
            s, l = getattr(short, attr), getattr(legacy, attr)
            self.assertTrue(np.isfinite(s).all())
            self.assertLess(np.subtract(*np.percentile(s, [95, 5])), 0.5 * np.subtract(*np.percentile(l, [95, 5])))

    def test_backlog_books_a_share_of_year_one(self):
        backlog = {"backlog_m": 600.0, "conversion": {"type": "metalog", "p10": 0.55, "p50": 0.70, "p90": 0.85}}
        flows = dict(net_debt_change_pct=(0.0, 0.01), share_dilution_pct=(0.0, 0.01))   # a zero std at T = 0 is not a distribution
        rev = {T: self._run(T, backlog=backlog, **flows).terminal_revenue for T in (0.0, 0.25, 0.5, 1.0)}
        # same seed, same draws: a part year is that share of the way to year one, path by path
        for T in (0.25, 0.5):
            np.testing.assert_allclose(rev[T] - 1000.0, T * (rev[1.0] - 1000.0), rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(rev[0.0], 1000.0)
        self.assertLess(np.median(rev[0.5]), np.median(rev[1.0]))     # it used to be the full year at any T < 1
