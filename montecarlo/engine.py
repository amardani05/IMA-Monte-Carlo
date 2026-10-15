"""
Core Monte Carlo engine — pure numpy/scipy, no plotting dependencies.

Extracted verbatim from the original MonteCarlo.py so the same code powers
both the local CLI and the deployed serverless API.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


@dataclass
class PitchAssumptions:
    """
    All inputs for a single company pitch.
    No defaults on core fields — you must provide every field.

    Triangular distributions: (low, mode, high)
    Normal distributions:     (mean, std)
    """

    # ── Company identifiers ──────────────────────────────────────────────
    ticker: str
    company_name: str
    current_price: float                    # Current share price

    # ── Case targets (your DCF / comps output) ───────────────────────────
    bear_price: float
    base_price: float
    bull_price: float

    # ── Current financials (TTM or NTM, be consistent) ───────────────────
    current_revenue: float                  # $M
    current_ebitda_margin: float            # decimal (e.g., 0.06 = 6%)
    current_ev: float                       # $M  (market cap + net debt)
    current_net_debt: float                 # $M  (negative = net cash)
    shares_outstanding: float               # M shares

    # ── Driver distributions ─────────────────────────────────────────────
    rev_cagr: tuple                         # triangular(low, mode, high)
    ebitda_margin: tuple                    # triangular(low, mode, high)
    ev_ebitda_multiple: tuple               # triangular(low, mode, high)
    net_debt_change_pct: tuple              # normal(mean, std)
    share_dilution_pct: tuple               # normal(mean, std)

    # ── Correlation matrix (order: rev, margin, multiple, debt, dilution)
    correlation_matrix: Optional[np.ndarray] = None

    # ── Simulation parameters ────────────────────────────────────────────
    n_simulations: int = 100_000
    random_seed: int = 42
    horizon_years: float = 2.0

    # ── Phase 2 options (all default to the legacy behaviour) ────────────
    # driver_specs: per-driver distribution specs that override the tuples
    # above, e.g. {"ebitda_margin": {"type": "metalog", "p10": .., "p50": .., "p90": ..}}.
    # copula: "gaussian" (legacy) or "t"; copula_df is the t-copula's degrees of freedom.
    driver_specs: Optional[dict] = None
    copula: str = "gaussian"
    copula_df: float = 5.0

    # ── Phase 3 mechanisms (each None = not used) ────────────────────────
    # cost_model: {"gross_margin": spec, "fixed_cost_m": $M today, "fixed_cost_growth": spec}
    #   EBITDA = revenue x gross margin - fixed cost; margin becomes an output.
    # backlog: {"backlog_m": $M, "conversion": spec}
    #   year-one revenue = backlog x conversion + uncovered part grown at the CAGR;
    #   a horizon under a year books that share of year one's revenue.
    # fcf_bridge: {"fcf_margin": spec, "ev_fcf_multiple": spec}
    #   a second valuation path, FCF x EV/FCF, reported beside the EBITDA bridge.
    # justified_multiple: {"roic", "wacc", "tax_rate", "da_share", "long_run_growth", "fade"}
    #   ties the drawn multiple to drawn growth through a justified EV/EBITDA.
    #   roic, wacc and fade each take a number or a distribution spec; a spec
    #   is drawn per path, independently of the five copula drivers.
    cost_model: Optional[dict] = None
    backlog: Optional[dict] = None
    fcf_bridge: Optional[dict] = None
    justified_multiple: Optional[dict] = None

    # ── Horizon term structure ───────────────────────────────────────────
    # level_width_horizon: years at which the level drivers (EBITDA margin,
    #   gross margin, FCF margin, the exit multiples) carry their full stated
    #   width. Under it, every draw is pulled toward its driver's median by
    #   sqrt(T / level_width_horizon), so the price distribution narrows to
    #   the stated centres as T goes to zero; at or beyond it, the stated
    #   width. Medians do not move. None = the stated width at every horizon.
    level_width_horizon: Optional[float] = None

    def __post_init__(self):
        if self.correlation_matrix is None:
            # Default cross-sector correlation assumptions
            # Order: rev_cagr, ebitda_margin, ev_ebitda, net_debt_chg, dilution
            self.correlation_matrix = np.array([
                [ 1.00,  0.40,  0.35, -0.15,  0.10],   # rev_cagr
                [ 0.40,  1.00,  0.25, -0.10,  0.05],   # ebitda_margin
                [ 0.35,  0.25,  1.00, -0.05,  0.00],   # ev_ebitda
                [-0.15, -0.10, -0.05,  1.00,  0.20],   # net_debt_change
                [ 0.10,  0.05,  0.00,  0.20,  1.00],   # dilution
            ])
        else:
            self.correlation_matrix = np.asarray(self.correlation_matrix, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
NUM_DRIVERS = 5

DRIVER_ORDER = [
    "Revenue CAGR",
    "EBITDA Margin",
    "EV/EBITDA Multiple",
    "Net Debt Chg (% EV)",
    "Share Dilution (%)",
]


class MonteCarloEngine:
    def __init__(self, assumptions: PitchAssumptions):
        self.a = assumptions
        self.rng = np.random.default_rng(self.a.random_seed)
        self.terminal_prices = None
        self.terminal_prices_fcf = None
        self.terminal_revenue = None
        self.driver_samples = None

    def _generate_correlated_uniforms(self, n: int) -> np.ndarray:
        """
        Generate correlated uniform(0,1) samples via Gaussian copula.
        Returns shape (n, NUM_DRIVERS) — one column per driver.
        """
        L = np.linalg.cholesky(self.a.correlation_matrix)
        z = self.rng.standard_normal((n, NUM_DRIVERS))
        correlated_z = z @ L.T
        if self.a.copula == "t":
            # Student-t copula: same correlation, but a shared chi-square
            # scaling makes extreme moves arrive together. The Gaussian copula
            # has zero tail dependence, which understates joint drawdowns.
            nu = float(self.a.copula_df)
            w = self.rng.chisquare(nu, size=(n, 1)) / nu
            t_draws = correlated_z / np.sqrt(w)
            return stats.t.cdf(t_draws, df=nu)
        u = stats.norm.cdf(correlated_z)
        return u

    def _sample_triangular(self, u: np.ndarray, low: float, mode: float, high: float) -> np.ndarray:
        """Inverse CDF of triangular distribution given uniform samples."""
        c = (mode - low) / (high - low)
        result = np.where(
            u < c,
            low + np.sqrt(u * (high - low) * (mode - low)),
            high - np.sqrt((1 - u) * (high - low) * (high - mode))
        )
        return result

    def _sample_normal_clipped(self, u: np.ndarray, mean: float, std: float,
                                clip_low: float = None, clip_high: float = None) -> np.ndarray:
        """Inverse CDF of normal, with optional clipping."""
        samples = stats.norm.ppf(u, loc=mean, scale=std)
        if clip_low is not None:
            samples = np.maximum(samples, clip_low)
        if clip_high is not None:
            samples = np.minimum(samples, clip_high)
        return samples

    def _draw(self) -> dict:
        """
        Every random draw of a run, keyed by driver, in a fixed order.

        The order of generator calls is the legacy one (copula, then backlog
        conversion, then fixed-cost growth, then any distributional ROIC, WACC
        and fade), so a run with the same inputs and seed reproduces earlier
        releases exactly.
        """
        from .distributions import make_sampler
        n = self.a.n_simulations
        u = self._generate_correlated_uniforms(n)
        specs = self.a.driver_specs or {}

        def draw(idx, name, legacy):
            spec = specs.get(name)
            if spec is None:
                return legacy()
            return make_sampler(spec)(u[:, idx])

        d = {
            "rev_cagr": draw(0, "rev_cagr", lambda: self._sample_triangular(u[:, 0], *self.a.rev_cagr)),
            "ebitda_margin": draw(1, "ebitda_margin", lambda: self._sample_triangular(u[:, 1], *self.a.ebitda_margin)),
            "ev_ebitda": draw(2, "ev_ebitda_multiple", lambda: self._sample_triangular(u[:, 2], *self.a.ev_ebitda_multiple)),
            "net_debt_chg": draw(3, "net_debt_change_pct", lambda: self._sample_normal_clipped(u[:, 3], *self.a.net_debt_change_pct)),
            "dilution": draw(4, "share_dilution_pct", lambda: self._sample_normal_clipped(
                u[:, 4], *self.a.share_dilution_pct, clip_low=-0.10, clip_high=0.15)),
        }
        d["ev_ebitda"] = np.maximum(d["ev_ebitda"], 0.0)
        if self.a.backlog:
            d["backlog_conv"] = make_sampler(self.a.backlog["conversion"])(self.rng.random(n))
        if self.a.cost_model:
            cm = self.a.cost_model
            d["gross_margin"] = make_sampler(cm["gross_margin"])(u[:, 1])
            d["fc_growth"] = make_sampler(cm["fixed_cost_growth"])(self.rng.random(n)) if cm.get("fixed_cost_growth") else 0.0
        if self.a.justified_multiple:
            jm = self.a.justified_multiple
            for key, default in (("roic", None), ("wacc", None), ("fade", 0.5)):
                v = jm.get(key, default)
                if isinstance(v, dict):
                    x = make_sampler(v)(self.rng.random(n))
                    d[key] = np.clip(x, 0.0, 1.0) if key == "fade" else x
                else:
                    d[key] = float(v)
        if self.a.fcf_bridge:
            fb = self.a.fcf_bridge
            d["fcf_margin"] = make_sampler(fb["fcf_margin"])(u[:, 1])        # moves with the margin draw
            d["ev_fcf"] = make_sampler(fb["ev_fcf_multiple"])(u[:, 2])       # moves with the multiple draw
        return d

    LEVEL_DRIVERS = ("ebitda_margin", "gross_margin", "fcf_margin", "ev_ebitda", "ev_fcf")

    def _scale_levels(self, d: dict) -> dict:
        """
        Level drivers at a horizon shorter than level_width_horizon. Revenue,
        net debt and fixed cost already compound or drift over T; a margin or
        multiple draw is a level, so without this it has its full width at any
        T and the distribution never closes on today. Each draw becomes
        median + s x (draw - median), s = sqrt(T / level_width_horizon): a
        convex combination, so the median, the sign and any floor at zero hold.
        """
        H = self.a.level_width_horizon
        if not H:
            return d
        s = min(self.a.horizon_years / float(H), 1.0) ** 0.5
        if s >= 1.0:
            return d
        d = dict(d)
        for key in self.LEVEL_DRIVERS:
            if key in d and np.ndim(d[key]):
                med = float(np.median(d[key]))
                d[key] = med + s * (d[key] - med)
        return d

    def _bridge(self, d: dict, ref: dict) -> dict:
        """
        Draws to prices. Pure arithmetic, no randomness: the same draws give
        the same prices, which is what lets `swing` re-price one driver held
        fixed against everything else exactly as it was drawn. `ref` holds the
        medians of the unperturbed run the justified multiple is anchored to.
        """
        n = self.a.n_simulations
        T = self.a.horizon_years
        rev_cagr, ebitda_margin, ev_ebitda = d["rev_cagr"], d["ebitda_margin"], d["ev_ebitda"]
        extra = {}

        # ── Revenue path, with backlog covering year one when supplied ───
        if self.a.backlog:
            conv = d["backlog_conv"]
            covered = float(self.a.backlog["backlog_m"]) * conv
            organic_y1 = self.a.current_revenue * (1 + rev_cagr)
            coverage = np.clip(covered / np.maximum(self.a.current_revenue, 1e-9), 0.0, 1.0)
            revenue_y1 = covered + organic_y1 * (1.0 - coverage)
            if T < 1.0:
                # trailing revenue at T is (1 - T) of today's year and T of year one
                terminal_revenue = self.a.current_revenue + T * (revenue_y1 - self.a.current_revenue)
            else:
                terminal_revenue = revenue_y1 * (1 + rev_cagr) ** (T - 1.0)
            extra["Backlog Conversion"] = conv
        else:
            terminal_revenue = self.a.current_revenue * (1 + rev_cagr) ** T

        # ── Margin: a single draw, or a cost structure ───────────────────
        if self.a.cost_model:
            cm = self.a.cost_model
            gross_margin, fc_growth = d["gross_margin"], d["fc_growth"]
            fixed_cost = float(cm["fixed_cost_m"]) * (1 + fc_growth) ** T
            terminal_ebitda = terminal_revenue * gross_margin - fixed_cost
            ebitda_margin = terminal_ebitda / np.maximum(terminal_revenue, 1e-9)
            extra["Gross Margin"] = gross_margin
            if cm.get("fixed_cost_growth"):
                extra["Fixed Cost Growth"] = np.broadcast_to(fc_growth, (n,))
        else:
            terminal_ebitda = terminal_revenue * ebitda_margin

        # ── Exit multiple: free draw, or justified by growth and returns ─
        if self.a.justified_multiple:
            jm = self.a.justified_multiple
            g_lr = float(jm.get("long_run_growth", 0.03))
            fade, roic, wacc = d["fade"], d["roic"], d["wacc"]
            g_term = g_lr + (1 - fade) * (rev_cagr - g_lr)          # drawn growth fades toward long run
            tax, da_share = float(jm.get("tax_rate", 0.25)), float(jm.get("da_share", 0.15))
            spread = np.maximum(wacc - g_term, 0.01)
            reinvest = np.clip(g_term / np.maximum(roic, 1e-6), 0.0, 0.95)
            justified = (1 - reinvest) * (1 - tax) * (1 - da_share) / spread
            # the company's premium to its own justified multiple today, decayed toward 1;
            # with drawn ROIC, WACC or fade, today's justified multiple is per path too
            g_mode = float(jm["current_growth"]) if jm.get("current_growth") is not None else ref["rev_cagr_median"]
            g_term0 = g_lr + (1 - fade) * (g_mode - g_lr)
            j0 = (1 - np.clip(g_term0 / np.maximum(roic, 1e-6), 0, 0.95)) * (1 - tax) * (1 - da_share) / np.maximum(wacc - g_term0, 0.01)
            cm0 = float(jm["current_multiple"]) if jm.get("current_multiple") is not None else j0
            prem0 = cm0 / np.maximum(j0, 1e-6)
            prem_T = 1.0 + (prem0 - 1.0) * (1 - float(jm.get("premium_reversion", 0.5))) ** T
            # the drawn multiple keeps only its relative dispersion around its median
            rel_noise = ev_ebitda / max(ref["ev_ebitda_median"], 1e-6)
            ev_ebitda = justified * prem_T * rel_noise
            extra["Justified Multiple"] = np.broadcast_to(justified * prem_T, (n,))
            for key, label in (("roic", "ROIC"), ("wacc", "WACC"), ("fade", "Growth Fade")):
                if np.ndim(d[key]):
                    extra[label] = d[key]
        terminal_ev = terminal_ebitda * ev_ebitda

        # ── Bridge to equity value per share ─────────────────────────────
        net_debt_delta = self.a.current_ev * d["net_debt_chg"]
        terminal_net_debt = self.a.current_net_debt + net_debt_delta
        terminal_equity_value = terminal_ev - terminal_net_debt

        terminal_shares = self.a.shares_outstanding * (1 + d["dilution"])
        terminal_price = terminal_equity_value / terminal_shares

        terminal_price = np.maximum(terminal_price, 0.0)

        # ── Second bridge: free cash flow x EV/FCF ───────────────────────
        price_fcf = None
        if self.a.fcf_bridge:
            fcf_ev = terminal_revenue * d["fcf_margin"] * d["ev_fcf"]
            price_fcf = np.maximum((fcf_ev - terminal_net_debt) / terminal_shares, 0.0)
            extra["FCF Margin"] = d["fcf_margin"]
            extra["EV/FCF Multiple"] = d["ev_fcf"]

        return {
            "price": terminal_price, "price_fcf": price_fcf, "revenue": terminal_revenue,
            "samples": {
                "Revenue CAGR": rev_cagr,
                "EBITDA Margin": ebitda_margin,
                "EV/EBITDA Multiple": ev_ebitda,
                "Net Debt Chg (% EV)": d["net_debt_chg"],
                "Share Dilution (%)": d["dilution"],
                **extra,
            },
        }

    def run(self) -> np.ndarray:
        """
        Run the full Monte Carlo simulation.
        Returns array of terminal share prices, shape (n_simulations,).
        """
        d = self._scale_levels(self._draw())
        self._draws = d
        self._ref = {"rev_cagr_median": float(np.median(d["rev_cagr"])),
                     "ev_ebitda_median": float(np.median(d["ev_ebitda"]))}
        out = self._bridge(d, self._ref)
        self.terminal_prices = out["price"]
        self.terminal_prices_fcf = out["price_fcf"]
        self.terminal_revenue = out["revenue"]
        self.driver_samples = out["samples"]
        return self.terminal_prices

    # ─────────────────────────────────────────────────────────────────────
    # One-at-a-time sensitivity on common random numbers
    # ─────────────────────────────────────────────────────────────────────
    def swing_drivers(self) -> list:
        """
        (key, label, bases) for every driver this run draws. `bases` names the
        prices the driver reaches: "ebitda" is the EBITDA bridge, "fcf" the
        FCF bridge. Scalar inputs are not drivers and are left out.
        """
        a, d = self.a, self._draws
        both = ("ebitda", "fcf") if a.fcf_bridge else ("ebitda",)
        rows = [("rev_cagr", "Revenue CAGR", both)]
        if a.backlog:
            rows.append(("backlog_conv", "Backlog conversion", both))
        if a.cost_model:
            rows.append(("gross_margin", "Gross margin", ("ebitda",)))
            if np.ndim(d["fc_growth"]):
                rows.append(("fc_growth", "Fixed cost growth", ("ebitda",)))
        else:
            rows.append(("ebitda_margin", "EBITDA margin", ("ebitda",)))
        rows.append(("ev_ebitda", "Multiple vs justified" if a.justified_multiple else "EV/EBITDA exit multiple", ("ebitda",)))
        if a.justified_multiple:
            for key, label in (("roic", "ROIC"), ("wacc", "WACC"), ("fade", "Growth fade")):
                if np.ndim(d[key]):
                    rows.append((key, label, ("ebitda",)))
        rows += [("net_debt_chg", "Net debt change", both), ("dilution", "Share dilution", both)]
        if a.fcf_bridge:
            rows += [("fcf_margin", "FCF margin", ("fcf",)), ("ev_fcf", "EV/FCF exit multiple", ("fcf",))]
        return rows

    def swing(self, key: str, value: float) -> dict:
        """Prices with one driver held at `value` on every path and every other draw unchanged."""
        if self.terminal_prices is None:
            self.run()
        d = dict(self._draws)
        d[key] = np.full(self.a.n_simulations, float(value))
        out = self._bridge(d, self._ref)
        return {"ebitda": out["price"], "fcf": out["price_fcf"]}

    def swing_tornado(self, lo_q: float = 10.0, hi_q: float = 90.0) -> dict:
        """
        How far each driver moves the median and P(at least base) between
        its P10 and its P90, the others drawn as before. Rows per bridge,
        widest median swing first. The fixed driver loses its correlation
        with the rest; that is what one-at-a-time means.
        """
        if self.terminal_prices is None:
            self.run()
        base = self.a.base_price
        out = {}
        prices = {"ebitda": self.terminal_prices, "fcf": self.terminal_prices_fcf}
        for basis, p in prices.items():
            if p is not None:
                out[basis] = {"median": float(np.median(p)), "p_base": float(np.mean(p >= base)), "rows": []}
        if self.a.justified_multiple:
            mult_med = self._ref["ev_ebitda_median"]
        for key, label, bases in self.swing_drivers():
            lo, hi = (float(v) for v in np.percentile(self._draws[key], [lo_q, hi_q]))
            at_lo, at_hi = self.swing(key, lo), self.swing(key, hi)
            shown_lo, shown_hi, unit = lo, hi, None
            if key == "ev_ebitda" and self.a.justified_multiple:
                # the drawn multiple is only noise around the justified one: show it as a ratio
                shown_lo, shown_hi, unit = lo / max(mult_med, 1e-6), hi / max(mult_med, 1e-6), "ratio"
            for basis in bases:
                pl, ph = at_lo[basis], at_hi[basis]
                out[basis]["rows"].append({
                    "key": key, "name": label, "lo": shown_lo, "hi": shown_hi, "unit": unit,
                    "median_lo": float(np.median(pl)), "median_hi": float(np.median(ph)),
                    "p_base_lo": float(np.mean(pl >= base)), "p_base_hi": float(np.mean(ph >= base)),
                })
        for b in out.values():
            b["rows"].sort(key=lambda r: -abs(r["median_hi"] - r["median_lo"]))
        return out

    def compute_case_probabilities(self) -> dict:
        """Compute probability of landing at or beyond each case target."""
        if self.terminal_prices is None:
            self.run()

        prices = self.terminal_prices
        n = len(prices)

        p_below_bear = np.sum(prices < self.a.bear_price) / n
        p_bear_to_base = np.sum((prices >= self.a.bear_price) & (prices < self.a.base_price)) / n
        p_base_to_bull = np.sum((prices >= self.a.base_price) & (prices < self.a.bull_price)) / n
        p_above_bull = np.sum(prices >= self.a.bull_price) / n

        p_at_least_bear = np.sum(prices >= self.a.bear_price) / n
        p_at_least_base = np.sum(prices >= self.a.base_price) / n
        p_at_least_bull = np.sum(prices >= self.a.bull_price) / n

        return {
            "below_bear": float(p_below_bear),
            "bear_to_base": float(p_bear_to_base),
            "base_to_bull": float(p_base_to_bull),
            "above_bull": float(p_above_bull),
            "p_at_least_bear": float(p_at_least_bear),
            "p_at_least_base": float(p_at_least_base),
            "p_at_least_bull": float(p_at_least_bull),
            "median_price": float(np.median(prices)),
            "mean_price": float(np.mean(prices)),
            "std_price": float(np.std(prices)),
            "p5": float(np.percentile(prices, 5)),
            "p25": float(np.percentile(prices, 25)),
            "p75": float(np.percentile(prices, 75)),
            "p95": float(np.percentile(prices, 95)),
            "expected_return": float(np.mean(prices) / self.a.current_price - 1),
        }

    def sensitivity_analysis(self, driver_name: str, n_buckets: int = 5) -> list:
        """
        Bucket a driver into quantiles and show median terminal price per bucket.
        """
        if self.driver_samples is None:
            self.run()

        driver_vals = self.driver_samples[driver_name]
        percentiles = np.linspace(0, 100, n_buckets + 1)
        edges = np.percentile(driver_vals, percentiles)

        results = []
        for i in range(n_buckets):
            mask = (driver_vals >= edges[i]) & (driver_vals < edges[i + 1])
            if i == n_buckets - 1:
                mask = (driver_vals >= edges[i]) & (driver_vals <= edges[i + 1])
            bucket_prices = self.terminal_prices[mask]
            results.append({
                "bucket": f"{edges[i]:.3f} - {edges[i+1]:.3f}",
                "count": int(mask.sum()),
                "median_price": float(np.median(bucket_prices)) if len(bucket_prices) > 0 else 0,
                "p_above_base": float(np.sum(bucket_prices >= self.a.base_price) / max(len(bucket_prices), 1)),
            })
        return results

    def tornado(self) -> list:
        """P10 vs P90 median terminal price per driver, sorted by spread."""
        if self.driver_samples is None:
            self.run()

        prices = self.terminal_prices
        impacts = []
        for name, vals in self.driver_samples.items():
            low_mask = vals <= np.percentile(vals, 10)
            high_mask = vals >= np.percentile(vals, 90)
            median_low = float(np.median(prices[low_mask]))
            median_high = float(np.median(prices[high_mask]))
            impacts.append({
                "name": name,
                "low": median_low,
                "high": median_high,
                "spread": median_high - median_low,
            })
        impacts.sort(key=lambda x: x["spread"])
        return impacts
