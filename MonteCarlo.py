import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from dataclasses import dataclass, field
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


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
NUM_DRIVERS = 5

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
        terminal_revenue = self.a.current_revenue * (1 + rev_cagr) ** 2
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
            "below_bear": p_below_bear,
            "bear_to_base": p_bear_to_base,
            "base_to_bull": p_base_to_bull,
            "above_bull": p_above_bull,
            "p_at_least_bear": p_at_least_bear,
            "p_at_least_base": p_at_least_base,
            "p_at_least_bull": p_at_least_bull,
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


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(engine: MonteCarloEngine, save_path: str = None):
    """Generate a publication-quality results chart. Saves as {TICKER}_monte_carlo.png by default."""
    a = engine.a
    prices = engine.terminal_prices
    probs = engine.compute_case_probabilities()

    if save_path is None:
        save_path = f"results/{a.ticker}_monte_carlo.png"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#f0f6fc")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # ── 1. Price Distribution Histogram ──────────────────────────────────
    ax1 = axes[0, 0]
    bins = np.linspace(0, np.percentile(prices, 99.5), 100)
    ax1.hist(prices, bins=bins, color="#1f6feb", alpha=0.7, edgecolor="none", density=True)
    ax1.axvline(a.bear_price, color="#f85149", linestyle="--", linewidth=2, label=f"Bear: ${a.bear_price:.0f}")
    ax1.axvline(a.base_price, color="#d29922", linestyle="--", linewidth=2, label=f"Base: ${a.base_price:.0f}")
    ax1.axvline(a.bull_price, color="#3fb950", linestyle="--", linewidth=2, label=f"Bull: ${a.bull_price:.0f}")
    ax1.axvline(a.current_price, color="#f0f6fc", linestyle="-", linewidth=1.5, alpha=0.6, label=f"Current: ${a.current_price:.0f}")
    ax1.axvline(probs["median_price"], color="#bc8cff", linestyle=":", linewidth=2, label=f"Median: ${probs['median_price']:.0f}")
    ax1.set_xlabel("Terminal Share Price ($)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{a.ticker} — Terminal Price Distribution (2-Year, {a.n_simulations:,} sims)")
    ax1.legend(fontsize=9, facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")

    # ── 2. Case Probability Bar Chart ────────────────────────────────────
    ax2 = axes[0, 1]
    categories = [f"Below Bear\n(<${a.bear_price:.0f})",
                  f"Bear-Base\n(${a.bear_price:.0f}-${a.base_price:.0f})",
                  f"Base-Bull\n(${a.base_price:.0f}-${a.bull_price:.0f})",
                  f"Above Bull\n(>${a.bull_price:.0f})"]
    values = [probs["below_bear"], probs["bear_to_base"],
              probs["base_to_bull"], probs["above_bull"]]
    colors = ["#f85149", "#d29922", "#1f6feb", "#3fb950"]
    bars = ax2.bar(categories, values, color=colors, edgecolor="none", alpha=0.85)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.1%}", ha="center", va="bottom", color="#f0f6fc", fontweight="bold", fontsize=12)
    ax2.set_ylabel("Probability")
    ax2.set_title("Scenario Probability Breakdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_ylim(0, max(values) * 1.25)

    # ── 3. Cumulative Probability (CDF) ──────────────────────────────────
    ax3 = axes[1, 0]
    sorted_prices = np.sort(prices)
    cdf = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)
    ax3.plot(sorted_prices, cdf, color="#1f6feb", linewidth=1.5)
    ax3.axvline(a.bear_price, color="#f85149", linestyle="--", linewidth=1.5)
    ax3.axvline(a.base_price, color="#d29922", linestyle="--", linewidth=1.5)
    ax3.axvline(a.bull_price, color="#3fb950", linestyle="--", linewidth=1.5)
    ax3.axvline(a.current_price, color="#f0f6fc", linestyle="-", linewidth=1, alpha=0.5)

    for price, label, color in [(a.bear_price, "Bear", "#f85149"),
                                 (a.base_price, "Base", "#d29922"),
                                 (a.bull_price, "Bull", "#3fb950")]:
        p_at_least = np.sum(prices >= price) / len(prices)
        ax3.annotate(f"P(>={label}) = {p_at_least:.1%}",
                     xy=(price, 1 - p_at_least), fontsize=9,
                     color=color, fontweight="bold",
                     xytext=(15, 10), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    ax3.set_xlabel("Terminal Share Price ($)")
    ax3.set_ylabel("Cumulative Probability")
    ax3.set_title("CDF — Probability of Price <= X")
    ax3.set_xlim(0, np.percentile(prices, 99.5))
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # ── 4. Driver Sensitivity (Tornado-style) ────────────────────────────
    ax4 = axes[1, 1]
    driver_names = list(engine.driver_samples.keys())
    impacts = []
    for name in driver_names:
        vals = engine.driver_samples[name]
        low_mask = vals <= np.percentile(vals, 10)
        high_mask = vals >= np.percentile(vals, 90)
        median_low = np.median(prices[low_mask])
        median_high = np.median(prices[high_mask])
        impacts.append((median_high - median_low, median_low, median_high, name))

    impacts.sort(key=lambda x: x[0])
    y_pos = np.arange(len(impacts))
    for i, (spread, low, high, name) in enumerate(impacts):
        ax4.barh(i, high - low, left=low, color="#1f6feb", alpha=0.75, height=0.6)
        ax4.text(low - 2, i, f"${low:.0f}", ha="right", va="center", color="#f85149", fontsize=9)
        ax4.text(high + 2, i, f"${high:.0f}", ha="left", va="center", color="#3fb950", fontsize=9)

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([x[3] for x in impacts], color="#c9d1d9")
    ax4.axvline(probs["median_price"], color="#bc8cff", linestyle=":", linewidth=1.5)
    ax4.set_xlabel("Median Terminal Price ($)")
    ax4.set_title("Driver Sensitivity (P10 vs P90)")

    fig.suptitle(f"{a.company_name} ({a.ticker}) — Monte Carlo Scenario Analysis",
                 fontsize=16, fontweight="bold", color="#f0f6fc", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Chart saved to {save_path}")


def print_summary(engine: MonteCarloEngine):
    """Print a formatted summary table."""
    a = engine.a
    probs = engine.compute_case_probabilities()

    print("\n" + "=" * 70)
    print(f"  {a.company_name} ({a.ticker}) — Monte Carlo Results")
    print(f"  Simulations: {a.n_simulations:,}  |  Horizon: 2 Years")
    print(f"  Current Price: ${a.current_price:.2f}")
    print("=" * 70)

    def fmt_price(val):
        return f"${val:.2f}"

    print(f"\n  {'Metric':<32} {'Value':>15}")
    print(f"  {'-'*32} {'-'*15}")
    print(f"  {'Median Terminal Price':<32} {fmt_price(probs['median_price']):>15}")
    print(f"  {'Mean Terminal Price':<32} {fmt_price(probs['mean_price']):>15}")
    print(f"  {'Std Dev':<32} {fmt_price(probs['std_price']):>15}")
    print(f"  {'5th Percentile':<32} {fmt_price(probs['p5']):>15}")
    print(f"  {'25th Percentile':<32} {fmt_price(probs['p25']):>15}")
    print(f"  {'75th Percentile':<32} {fmt_price(probs['p75']):>15}")
    print(f"  {'95th Percentile':<32} {fmt_price(probs['p95']):>15}")
    print(f"  {'Expected 2yr Return':<32} {probs['expected_return']:>14.1%}")

    bear_label = f"Below Bear (<${a.bear_price:.0f})"
    bb_label = f"Bear to Base (${a.bear_price:.0f}-${a.base_price:.0f})"
    bbu_label = f"Base to Bull (${a.base_price:.0f}-${a.bull_price:.0f})"
    abv_label = f"Above Bull (>${a.bull_price:.0f})"

    print(f"\n  {'Scenario':<32} {'Probability':>15}")
    print(f"  {'-'*32} {'-'*15}")
    print(f"  {bear_label:<32} {probs['below_bear']:>14.1%}")
    print(f"  {bb_label:<32} {probs['bear_to_base']:>14.1%}")
    print(f"  {bbu_label:<32} {probs['base_to_bull']:>14.1%}")
    print(f"  {abv_label:<32} {probs['above_bull']:>14.1%}")

    print(f"\n  {'Cumulative Thresholds':<32} {'Probability':>15}")
    print(f"  {'-'*32} {'-'*15}")
    print(f"  {'P(>= Bear Target)':<32} {probs['p_at_least_bear']:>14.1%}")
    print(f"  {'P(>= Base Target)':<32} {probs['p_at_least_base']:>14.1%}")
    print(f"  {'P(>= Bull Target)':<32} {probs['p_at_least_bull']:>14.1%}")
    print("=" * 70)

    # Driver sensitivity summary
    print(f"\n  Driver Sensitivity (P10 vs P90 median terminal price):")
    print(f"  {'-'*55}")
    for name in engine.driver_samples:
        sens = engine.sensitivity_analysis(name, n_buckets=5)
        spread = sens[-1]["median_price"] - sens[0]["median_price"]
        low_p = f"${sens[0]['median_price']:.0f}"
        high_p = f"${sens[-1]['median_price']:.0f}"
        print(f"  {name:<28} Spread: ${spread:>8.2f}  ({low_p} -> {high_p})")
    print()


if __name__ == "__main__":
    # =====================================================================
    # MODIFY THESE ASSUMPTIONS FOR EACH PITCH
    # =====================================================================
    pitch = PitchAssumptions(
        ticker = "MYRG",
        company_name = "MYR Group",
        current_price = 271.00,

        bear_price = 222.32,
        base_price = 323.01,
        bull_price = 362.32,

        current_revenue=3510.0,             # $M LTM through Q3 2025
        current_ebitda_margin=0.061,         # LTM EBITDA ~$214M / $3,510M
        current_ev=4310.0,                   # $M
        current_net_debt=110.0,              # $M
        shares_outstanding=15.5,             # M

        # Triangular distributions: (low, mode, high)
        rev_cagr=(0.057, 0.069, 0.075),
        ebitda_margin=(0.071, 0.075, 0.077),
        ev_ebitda_multiple=(12.4, 17.0, 18.5),

        # Normal distributions: (mean, std)
        net_debt_change_pct=(0.00, 0.015),
        share_dilution_pct=(0.01, 0.01),

        n_simulations = 100000,
    )

    engine = MonteCarloEngine(pitch)
    engine.run()

    print_summary(engine)
    plot_results(engine)  # saves as MYRG_monte_carlo.png