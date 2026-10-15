"""Service-layer chart series: the horizon densities behind the assessor's path chart. Offline."""

import unittest

import numpy as np
from scipy import stats

from montecarlo import MonteCarloEngine, PitchAssumptions
from montecarlo.service import (DENSITY_POINTS, _horizon_label, _log_kde_per_dollar, horizon_densities,
                                run_simulation)

MYRG = dict(
    ticker="MYRG", company_name="MYR Group", current_price=271.0,
    bear_price=222.32, base_price=323.01, bull_price=362.32,
    current_revenue=3510.0, current_ebitda_margin=0.061, current_ev=4310.0,
    current_net_debt=110.0, shares_outstanding=15.5,
    rev_cagr=(0.057, 0.069, 0.075), ebitda_margin=(0.071, 0.075, 0.077),
    ev_ebitda_multiple=(12.4, 17.0, 18.5), net_debt_change_pct=(0.0, 0.015),
    share_dilution_pct=(0.01, 0.01), n_simulations=40_000, random_seed=42,
)
SPECS = {"rev_cagr": {"type": "metalog", "p10": 0.06, "p50": 0.157, "p90": 0.22},
         "ebitda_margin": {"type": "metalog", "p10": 0.05, "p50": 0.062, "p90": 0.07},
         "ev_ebitda_multiple": {"type": "log_metalog", "p10": 13.4, "p50": 18.9, "p90": 28.4, "lower": 0.0}}


def _engine(**over):
    e = MonteCarloEngine(PitchAssumptions(**{**MYRG, **over}))
    e.run()
    return e


class HorizonDensities(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = _engine(driver_specs=SPECS)
        cls.hd = horizon_densities(cls.engine)

    def test_quarters_of_a_two_year_run(self):
        self.assertEqual([h["label"] for h in self.hd["horizons"]], ["6m", "1y", "18m", "2y"])
        self.assertEqual([h["interpolated"] for h in self.hd["horizons"]], [True, True, True, False])
        self.assertEqual(self.hd["note"], "6m to 18m interpolated from the 2-year run")
        self.assertEqual(len(self.hd["grid"]), DENSITY_POINTS)

    def test_medians_drift_geometrically_from_spot(self):
        spot, med = MYRG["current_price"], float(np.median(self.engine.terminal_prices))
        for h in self.hd["horizons"]:
            self.assertAlmostEqual(h["median"], spot * (med / spot) ** h["fraction"], delta=1e-6 * spot)

    def test_terminal_horizon_is_the_run_itself(self):
        last = self.hd["horizons"][-1]
        p = self.engine.terminal_prices
        self.assertAlmostEqual(last["median"], float(np.median(p)), places=9)
        self.assertAlmostEqual(last["p_at_least_base"], float(np.mean(p >= MYRG["base_price"])), places=12)
        self.assertAlmostEqual(last["p_at_least_bear"], float(np.mean(p >= MYRG["bear_price"])), places=12)

    def test_curves_integrate_to_one_on_a_common_scale(self):
        g = np.asarray(self.hd["grid"])
        peaks = []
        for h in self.hd["horizons"]:
            d = np.asarray(h["density"])
            self.assertTrue(np.all(d >= 0))
            self.assertAlmostEqual(float(np.trapezoid(d, g)), 1.0, delta=0.02)
            peaks.append(d.max())
        # the near horizon is narrow and tall, the far one broad and flat
        self.assertEqual(peaks, sorted(peaks, reverse=True))

    def test_dispersion_shrinks_with_the_square_root_of_time(self):
        x_T = self.engine.terminal_prices
        from montecarlo.service import _interpolated_prices
        x_q = _interpolated_prices(self.engine, 0.25)
        sd_T = np.std(np.log(x_T[x_T > 0])); sd_q = np.std(np.log(x_q[x_q > 0]))
        self.assertAlmostEqual(sd_q / sd_T, 0.5, places=6)

    def test_paths_at_zero_carry_no_density(self):
        # a thin margin and heavy net debt put a share of paths at zero
        e = _engine(current_net_debt=1500.0, ebitda_margin=(0.01, 0.04, 0.08))
        share_pos = float(np.mean(e.terminal_prices > 0))
        self.assertLess(share_pos, 0.99)
        hd = horizon_densities(e)
        g = np.asarray(hd["grid"])
        last = np.asarray(hd["horizons"][-1]["density"])
        self.assertAlmostEqual(float(np.trapezoid(last, g)), share_pos, delta=0.02)

    def test_no_curves_when_the_median_is_zero(self):
        e = _engine(current_net_debt=20_000.0)
        self.assertEqual(float(np.median(e.terminal_prices)), 0.0)
        self.assertIsNone(horizon_densities(e))

    def test_labels_for_other_horizons(self):
        self.assertEqual([_horizon_label(u * 3) for u in (0.25, 0.5, 0.75, 1)], ["9m", "18m", "27m", "3y"])
        self.assertEqual([_horizon_label(u * 1) for u in (0.25, 0.5, 0.75, 1)], ["3m", "6m", "9m", "1y"])
        self.assertEqual(_horizon_label(0.625), "7.5m")
        hd = horizon_densities(_engine(horizon_years=1.0))
        self.assertEqual(hd["note"], "3m to 9m interpolated from the 1-year run")

    def test_log_kde_matches_a_lognormal(self):
        rng = np.random.default_rng(7)
        mu, sigma = np.log(100.0), 0.3
        x = np.exp(rng.normal(mu, sigma, 200_000))
        grid = np.linspace(40, 250, 300)
        est = _log_kde_per_dollar(x, grid)
        true = stats.lognorm.pdf(grid, s=sigma, scale=np.exp(mu))
        core = (grid > 60) & (grid < 180)
        self.assertLess(float(np.max(np.abs(est[core] - true[core]) / true[core])), 0.04)

    def test_on_the_wire(self):
        p = {k: (list(v) if isinstance(v, tuple) else v) for k, v in MYRG.items()}
        r = run_simulation(p)
        hd = r["horizon_densities"]
        self.assertEqual(len(hd["horizons"]), 4)
        self.assertTrue(all(len(h["density"]) == len(hd["grid"]) for h in hd["horizons"]))
        self.assertAlmostEqual(hd["horizons"][-1]["median"], r["stats"]["median_price"], places=9)


if __name__ == "__main__":
    unittest.main()
