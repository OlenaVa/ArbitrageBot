"""
Transaction-cost stress testing.

The strategy's PnL step charges `cost x turnover` whenever the position
size changes. The primary reported number in main.py (5 bps / 0.0005,
see config.cost_per_turnover) is explicitly a placeholder, not derived
from real CL/BZ bid-ask quotes.

Rather than defending one guessed number, this module re-runs the SAME
already-computed position path (turnover doesn't depend on the cost
assumption - only the final PnL subtraction does) at several cost levels
and reports how much of the strategy's edge survives.

Scenario levels, and why: "base" is set equal to config.cost_per_turnover
(5 bps) so this module's own baseline row reproduces the primary run's
number as a sanity check, rather than silently testing a different range
than the one actually used elsewhere. "low" (half of base) is an
optimistic case - tighter spreads / passive fills. "stress" (2x base) is
the actual stress case this module is named for: costs coming in worse
than assumed, e.g. from wider bid-ask spreads or slippage on size. This
fixes an earlier version of this file where "stress" (2 bps) was
actually LOWER than the 5 bps primary assumption - i.e. testing an
easier scenario while calling it a stress test. See README - still a
stylized stress test, not a realistic execution/market-impact
simulation.
"""
from __future__ import annotations

import pandas as pd

import spread_model

# label -> cost in basis points (1 bp = 0.0001 = 0.01%)
# "base" intentionally matches config.cost_per_turnover (5 bps) - see
# module docstring for why "low"/"stress" are defined relative to it.
COST_SCENARIOS_BPS = {
    "low": 2.5,
    "base": 5.0,
    "stress": 10.0,
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
    """
    One-line summary, plus where the model's own primary assumption
    (config.cost_per_turnover, reported separately in main.py) sits
    relative to this scenario range. Reads scenario values from
    `stress_df` itself rather than hardcoding bps numbers in the message,
    so this text can't silently go stale if COST_SCENARIOS_BPS changes.
    """
    survives_stress = bool((stress_df["sharpe"] > 0).all())

    def _sharpe_for(label: str) -> float:
        row = stress_df[stress_df["scenario"] == label]
        return float(row["sharpe"].iloc[0]) if not row.empty else float("nan")

    def _bps_for(label: str) -> float:
        row = stress_df[stress_df["scenario"] == label]
        return float(row["cost_bps"].iloc[0]) if not row.empty else float("nan")

    base_bps, stress_bps = _bps_for("base"), _bps_for("stress")
    base_sharpe, stress_sharpe = _sharpe_for("base"), _sharpe_for("stress")

    verdict = (
        "Sharpe stays positive across all tested cost levels."
        if survives_stress else
        "Sharpe turns negative at one or more tested cost levels - "
        "the edge is cost-sensitive."
    )

    min_bps, max_bps = stress_df["cost_bps"].min(), stress_df["cost_bps"].max()
    if min_bps <= reference_cost_bps <= max_bps:
        placement = (
            f"matches the 'base' scenario exactly, so that row reproduces the primary run's Sharpe as a sanity check."
            if abs(reference_cost_bps - base_bps) < 1e-9 else
            "falls inside the tested range."
        )
    else:
        placement = "falls outside the tested range - treat these scenarios as relative, not calibrated to it."

    return (
        f"base ({base_bps:.1f} bps) Sharpe={base_sharpe:.3f}, "
        f"stress ({stress_bps:.1f} bps) Sharpe={stress_sharpe:.3f}. "
        f"{verdict} For reference, the flat cost used elsewhere in this "
        f"backtest (config.cost_per_turnover) is {reference_cost_bps:.1f} bps, which {placement}"
    )