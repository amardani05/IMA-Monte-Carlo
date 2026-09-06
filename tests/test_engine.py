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
