"""The example pitches in public/examples.json: well formed, and accepted by the service as written."""
import json
import os
import unittest

from montecarlo.service import build_assumptions

PATH = os.path.join(os.path.dirname(__file__), "..", "public", "examples.json")


def _payload(ex):
    """The example on top of a minimal filed base, as the front end sends it after auto-fill."""
    p = {"ticker": ex["ticker"], "company_name": ex["company_name"], "current_price": ex["current_price"],
         "bear_price": ex["targets"]["bear"], "base_price": ex["targets"]["base"], "bull_price": ex["targets"]["bull"],
         "horizon_years": ex["horizon_years"], "current_revenue": 1000.0, "current_ebitda_margin": 0.1,
         "current_ev": 1000.0, "current_net_debt": 0.0, "shares_outstanding": 10.0,
         "rev_cagr": [0.0, 0.05, 0.1], "ebitda_margin": [0.08, 0.1, 0.12], "ev_ebitda_multiple": [8.0, 10.0, 12.0],
         "net_debt_change_pct": [0.0, 0.02], "share_dilution_pct": [0.0, 0.01], "n_simulations": 1000, "random_seed": 1}
    p.update(ex.get("mechanisms", {}))
    if ex.get("dcf_path"):
        p["dcf_path"] = {k: ex["dcf_path"][k] for k in ("rev_cagr", "ebitda_margin", "ev_ebitda_multiple")}
    return p


class Examples(unittest.TestCase):
    def setUp(self):
        with open(PATH) as fh:
            self.examples = json.load(fh)["examples"]

    def test_ids_are_unique(self):
        ids = [e["id"] for e in self.examples]
        self.assertEqual(len(ids), len(set(ids)))

    def test_targets_are_ordered_and_priced(self):
        for e in self.examples:
            t = e["targets"]
            self.assertGreater(e["current_price"], 0, e["id"])
            self.assertLess(t["bear"], t["base"], e["id"])
            self.assertLess(t["base"], t["bull"], e["id"])
            self.assertTrue(0 < e["horizon_years"] <= 10, e["id"])

    def test_every_example_names_its_source_and_horizon_basis(self):
        for e in self.examples:
            self.assertTrue(e.get("source"), e["id"])
            self.assertTrue(e.get("horizon_basis"), e["id"])
            self.assertTrue(e.get("pitch_date"), e["id"])

    def test_the_service_accepts_each_example(self):
        for e in self.examples:
            with self.subTest(e["id"]):
                build_assumptions(_payload(e))


if __name__ == "__main__":
    unittest.main()
