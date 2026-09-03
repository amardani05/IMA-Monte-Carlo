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
        self.driver_samples = None

    def _generate_correlated_uniforms(self, n: int) -> np.ndarray:
        """
        Generate correlated uniform(0,1) samples via Gaussian copula.
        Returns shape (n, NUM_DRIVERS) — one column per driver.
        """
        L = np.linalg.cholesky(self.a.correlation_matrix)
        z = self.rng.standard_normal((n, NUM_DRIVERS))
        correlated_z = z @ L.T
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

    def run(self) -> np.ndarray:
        """
        Run the full Monte Carlo simulation.
        Returns array of terminal share prices, shape (n_simulations,).
        """
        n = self.a.n_simulations
        u = self._generate_correlated_uniforms(n)

        # ── Sample each driver ───────────────────────────────────────────
        rev_cagr = self._sample_triangular(u[:, 0], *self.a.rev_cagr)
        ebitda_margin = self._sample_triangular(u[:, 1], *self.a.ebitda_margin)
        ev_ebitda = self._sample_triangular(u[:, 2], *self.a.ev_ebitda_multiple)
        net_debt_chg = self._sample_normal_clipped(u[:, 3], *self.a.net_debt_change_pct)
        dilution = self._sample_normal_clipped(u[:, 4], *self.a.share_dilution_pct,
                                                clip_low=-0.10, clip_high=0.15)

        # ── Terminal financials ──────────────────────────────────────────
        terminal_revenue = self.a.current_revenue * (1 + rev_cagr) ** self.a.horizon_years
        terminal_ebitda = terminal_revenue * ebitda_margin
        terminal_ev = terminal_ebitda * ev_ebitda

        # ── Bridge to equity value per share ─────────────────────────────
        net_debt_delta = self.a.current_ev * net_debt_chg
        terminal_net_debt = self.a.current_net_debt + net_debt_delta
        terminal_equity_value = terminal_ev - terminal_net_debt

        terminal_shares = self.a.shares_outstanding * (1 + dilution)
        terminal_price = terminal_equity_value / terminal_shares

        terminal_price = np.maximum(terminal_price, 0.0)

        self.terminal_prices = terminal_price
        self.driver_samples = {
            "Revenue CAGR": rev_cagr,
            "EBITDA Margin": ebitda_margin,
            "EV/EBITDA Multiple": ev_ebitda,
            "Net Debt Chg (% EV)": net_debt_chg,
            "Share Dilution (%)": dilution,
        }
        return terminal_price

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
