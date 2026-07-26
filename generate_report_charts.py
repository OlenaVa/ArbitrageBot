"""
generate_report_charts.py - turns the committed results/*.csv files into
the PNG charts embedded in README.md.

This does NOT touch the network or re-run the model: it only reads
results/oos_daily_returns.csv (already produced by oos_evaluation.py) and
config.py, and writes PNGs to results/charts/. Run it any time after
regenerating the CSVs so the README images stay in sync with the numbers:

    python oos_evaluation.py
    python generate_report_charts.py

Charts produced:
    results/charts/oos_equity_drawdown.png  - equity + drawdown, one column
                                               per OOS period (each period
                                               resets to 1.0 - see README
                                               "OOS evaluation: a bug found
                                               and fixed" for why these are
                                               NOT chained into one curve)
    results/charts/beta_over_time.png       - Kalman hedge ratio, full history
    results/charts/spread_zscore.png        - spread + z-score with the
                                               configured entry/exit bands
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import StrategyConfig

CHARTS_DIR = Path("results/charts")

PERIOD_LABELS = {
    "development_2019_2021": "Development",
    "validation_2022_2023": "Validation",
    "final_test_2024_2026": "Final test",
}
PERIOD_ORDER = list(PERIOD_LABELS.keys())

NAVY = "#1b2a4a"
STEEL = "#2c5f8a"
MAROON = "#a33d3d"
GOLD = "#c9922c"
GRID = "#dddddd"


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _load_daily_returns() -> pd.DataFrame:
    df = pd.read_csv("results/oos_daily_returns.csv", parse_dates=["Date"])
    df["mr_regime"] = df["mr_regime"].astype(str).isin(["True", "1", "1.0"])
    return df


def _period_bounds(df: pd.DataFrame) -> dict:
    return {
        name: (g["Date"].min(), g["Date"].max())
        for name, g in df.groupby("period")
    }


def plot_equity_and_drawdown(df: pd.DataFrame, results_summary: pd.DataFrame | None) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey="row")

    # Shared y-limits per row so the three periods are visually comparable.
    eq_lo, eq_hi = 1.0, 1.0
    dd_lo = 0.0
    for period in PERIOD_ORDER:
        sub = df[df["period"] == period]
        if sub.empty:
            continue
        eq_lo = min(eq_lo, sub["equity"].min())
        eq_hi = max(eq_hi, sub["equity"].max())
        dd = sub["equity"] / sub["equity"].cummax() - 1.0
        dd_lo = min(dd_lo, dd.min())
    eq_pad = (eq_hi - eq_lo) * 0.08 or 0.01
    dd_pad = abs(dd_lo) * 0.15 or 0.005

    for col, period in enumerate(PERIOD_ORDER):
        sub = df[df["period"] == period]
        label = PERIOD_LABELS[period]
        if sub.empty:
            axes[0, col].set_title(f"{label}\n(no data)")
            continue

        start, end = sub["Date"].min(), sub["Date"].max()
        dd = sub["equity"] / sub["equity"].cummax() - 1.0

        ax_eq = axes[0, col]
        ax_eq.plot(sub["Date"], sub["equity"], color=NAVY, lw=1.5)
        ax_eq.axhline(1.0, color="#aaaaaa", lw=0.8, ls="--")
        ax_eq.set_title(f"{label}\n{start:%Y-%m-%d} \u2192 {end:%Y-%m-%d}")
        ax_eq.set_ylim(eq_lo - eq_pad, eq_hi + eq_pad)
        if col == 0:
            ax_eq.set_ylabel("Equity (start = 1.0)")

        if results_summary is not None and period in results_summary.index:
            row = results_summary.loc[period]
            ax_eq.text(
                0.03, 0.06,
                f"Sharpe {row['sharpe']:.2f}  |  Return {row['return'] * 100:.1f}%",
                transform=ax_eq.transAxes, fontsize=8.5, color="#333333",
                va="bottom", ha="left",
            )

        ax_dd = axes[1, col]
        ax_dd.fill_between(sub["Date"], dd * 100, 0, color=MAROON, alpha=0.35, lw=0)
        ax_dd.plot(sub["Date"], dd * 100, color=MAROON, lw=1.0)
        ax_dd.set_ylim((dd_lo - dd_pad) * 100, 1.0)
        if col == 0:
            ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.tick_params(axis="x", labelrotation=30)

    fig.suptitle(
        "Out-of-sample performance by period (each period is an independent test - "
        "equity resets to 1.0 at the start of each, not chained across periods)",
        fontsize=10.5, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "oos_equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beta_over_time(df: pd.DataFrame, config: StrategyConfig) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df["Date"], df["beta"], color=NAVY, lw=1.1)
    ax.axhline(config.beta_warn_min, color=GOLD, lw=1.0, ls="--",
               label=f"Diagnostic band [{config.beta_warn_min}, {config.beta_warn_max}] (not a clip)")
    ax.axhline(config.beta_warn_max, color=GOLD, lw=1.0, ls="--")

    for name, (start, _end) in _period_bounds(df).items():
        if name == "development_2019_2021":
            continue
        ax.axvline(start, color="#888888", lw=0.9, ls=":")
        ax.text(start, ax.get_ylim()[1], f" {PERIOD_LABELS[name]}", fontsize=8,
                color="#666666", va="top", ha="left")

    ax.set_title("Kalman hedge ratio (beta) - full history, pure posterior, no clipping")
    ax.set_ylabel("beta")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "beta_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spread_and_zscore(df: pd.DataFrame, config: StrategyConfig) -> None:
    fig, (ax_s, ax_z) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ax_s.plot(df["Date"], df["spread"], color=NAVY, lw=0.9)
    ax_s.axhline(0, color="#aaaaaa", lw=0.8)
    ax_s.set_title("Kalman spread (log-residual y - beta*x) and z-score, full history")
    ax_s.set_ylabel("spread (log space)")

    ax_z.plot(df["Date"], df["z"], color=STEEL, lw=0.8)
    ax_z.axhspan(-config.exit_threshold, config.exit_threshold, color=GRID, alpha=0.6,
                 label=f"Exit band (\u00b1{config.exit_threshold})")
    ax_z.axhline(config.entry_threshold, color=MAROON, lw=1.0, ls="--",
                 label=f"Entry (\u00b1{config.entry_threshold})")
    ax_z.axhline(-config.entry_threshold, color=MAROON, lw=1.0, ls="--")
    ax_z.set_ylabel("z-score")
    ax_z.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)

    for ax in (ax_s, ax_z):
        for name, (start, _end) in _period_bounds(df).items():
            if name == "development_2019_2021":
                continue
            ax.axvline(start, color="#888888", lw=0.9, ls=":")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "spread_zscore.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    _apply_style()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    config = StrategyConfig()

    df = _load_daily_returns()

    results_summary = None
    results_path = Path("results/oos_results.csv")
    if results_path.exists():
        results_summary = pd.read_csv(results_path).set_index("period")

    plot_equity_and_drawdown(df, results_summary)
    plot_beta_over_time(df, config)
    plot_spread_and_zscore(df, config)

    print(f"Wrote 3 charts to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
