"""
Transaction-cost stress testing.

The strategy's PnL step charges `cost x turnover` whenever the position
size changes. The original single flat assumption (5 bps / 0.0005, still
used as the primary reported number in main.py - see config.cost_per_turnover)
is explicitly a placeholder, not derived from real CL/BZ bid-ask quotes.

Rather than defending one guessed number, this module re-runs the SAME
already-computed position path (turnover doesn't depend on the cost
assumption - only the final PnL subtraction does) at several cost levels
and reports how much of the strategy's edge survives. "Does this still work
if real costs turn out higher than assumed" is a more honest question than
"is 5bps exactly right", and much cheaper to answer than building a full
market-impact model without real order-book data to calibrate it against
(see README - a stylized stress test, not a realistic execution simulation).
"""
from __future__ import annotations

import pandas as pd

import spread_model

# label -> cost in basis points (1 bp = 0.0001 = 0.01%)
COST_SCENARIOS_BPS = {
    "low": 0.5,
    "base": 1.0,
    "stress": 2.0,
}


def run_cost_stress_test(df: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
    """
    df must already contain: Brent, WTI, beta, and `position_col` (the
    final risk-scaled position path from the primary run). Reuses that
    exact position path for every scenario - only the cost assumption
    changes between rows.
    """
    rows = []
    for label, bps in COST_SCENARIOS_BPS.items():
        cost = bps / 10_000.0
        perf = spread_model.compute_performance(df, df[position_col], cost)
        rows.append({
            "scenario": label,
            "cost_bps": bps,
            "sharpe": perf["sharpe"],
            "max_dd_pct": perf["max_dd"] * 100 if perf["max_dd"] == perf["max_dd"] else float("nan"),
            "n_trades": perf["n_trades"],
        })
    return pd.DataFrame(rows)


def summarize_cost_stress(stress_df: pd.DataFrame, reference_cost_bps: float) -> str:
    """One-line summary, plus where the model's own primary assumption
    (config.cost_per_turnover, reported separately in main.py) sits
    relative to this scenario range."""
    survives_stress = bool((stress_df["sharpe"] > 0).all())
    base_row = stress_df[stress_df["scenario"] == "base"]
    base_sharpe = base_row["sharpe"].iloc[0] if not base_row.empty else float("nan")
    stress_row = stress_df[stress_df["scenario"] == "stress"]
    stress_sharpe = stress_row["sharpe"].iloc[0] if not stress_row.empty else float("nan")

    verdict = (
        "Sharpe stays positive across all tested cost levels."
        if survives_stress else
        "Sharpe turns negative at one or more tested cost levels - "
        "the edge is cost-sensitive."
    )
    return (
        f"base (1.0 bps) Sharpe={base_sharpe:.3f}, stress (2.0 bps) Sharpe={stress_sharpe:.3f}. "
        f"{verdict} For reference, the flat cost used elsewhere in this "
        f"backtest (config.cost_per_turnover) is {reference_cost_bps:.1f} bps, "
        f"outside the top of this stress range - the numbers here are "
        f"deliberately testing a lower-cost regime, not replacing that assumption."
    )