"""
Stylized futures roll-cost drag.

Why this exists: `main.py` and `market_check.py` trade CL=F/BZ=F, which
yfinance serves as continuous front-month series - there is no per-contract
curve data in this repo to compute a real roll yield from actual
contango/backwardation on each roll date. That was previously listed in
README "Known limitations" as the single largest unmodeled gap between this
backtest and live trading. This module does not close that gap - a true fix
needs real futures-curve data this project doesn't have - but it stops the
gap from being silently absent from the numbers: it applies a configurable
annualized drag (`config.roll_drag_annual_bps`) whenever a position is held,
and reports Sharpe with and without it, the same "how much of the edge
survives" framing already used for `costs.py`'s transaction-cost stress test.

Treat this as a sensitivity check, not a roll simulation: it tells you
whether the strategy's edge is fragile or robust to a plausible order of
magnitude of roll cost, not what the exact roll cost would be.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import spread_model


def apply_roll_drag(df: pd.DataFrame, position_col: str, annual_bps: float) -> pd.Series:
    """
    Returns a daily log-return drag series: -annual_bps/10000/252 on every
    day a (risk-scaled) position is held, scaled by the position's
    absolute size (a 2x-leveraged day is charged twice the drag of a
    1x day, consistent with how `spread_model.compute_performance`
    already scales transaction costs by turnover rather than a flat
    per-trade fee). Zero on days with no position - flat/no exposure
    means nothing is being rolled.
    """
    daily_drag = annual_bps / 10_000.0 / 252.0
    position = df[position_col].shift(1).fillna(0.0)  # yesterday's held position, consistent with the pos_lag convention elsewhere
    return -daily_drag * position.abs()


def compute_performance_with_roll_drag(
        df: pd.DataFrame, position_col: str, cost_per_turnover: float, roll_drag_annual_bps: float,
) -> dict:
    """
    Re-derives net_log_ret starting from spread_model.compute_performance's
    own transaction-cost-adjusted return, then subtracts the roll drag on
    top - so this never duplicates or drifts from how transaction costs are
    charged elsewhere; it only adds one further, clearly-labeled cost.
    """
    base_perf = spread_model.compute_performance(df, df[position_col], cost_per_turnover)
    drag = apply_roll_drag(df, position_col, roll_drag_annual_bps)
    net_with_drag = (base_perf["net_log_ret"] + drag).dropna()

    if len(net_with_drag) == 0 or net_with_drag.std() == 0:
        return {"net_log_ret": net_with_drag, "sharpe": np.nan, "sortino": np.nan, "max_dd": np.nan}

    r = np.exp(net_with_drag) - 1
    sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252)
    down = r[r < 0]
    sortino = (r.mean() / (down.std() + 1e-12)) * np.sqrt(252) if len(down) > 0 else np.nan
    equity = np.exp(net_with_drag.cumsum())
    max_dd = (equity / equity.cummax() - 1).min()

    return {"net_log_ret": net_with_drag, "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd}


def summarize_roll_drag(df: pd.DataFrame, position_col: str, cost_per_turnover: float, roll_drag_annual_bps: float) -> str:
    """One-line before/after comparison, framed as a sensitivity check."""
    without = spread_model.compute_performance(df, df[position_col], cost_per_turnover)
    with_drag = compute_performance_with_roll_drag(df, position_col, cost_per_turnover, roll_drag_annual_bps)

    delta = with_drag["sharpe"] - without["sharpe"]
    verdict = (
        "Sharpe stays positive even with the roll-drag assumption added."
        if with_drag["sharpe"] > 0 else
        "Sharpe turns negative once the roll-drag assumption is added - "
        "the edge may not survive real futures roll costs."
    )
    return (
        f"Without roll drag: Sharpe={without['sharpe']:.3f}. "
        f"With {roll_drag_annual_bps:.0f} bps/yr roll-drag assumption: Sharpe={with_drag['sharpe']:.3f} "
        f"(delta={delta:+.3f}). {verdict} This is a stylized sensitivity check, not a "
        f"real per-contract roll simulation - see roll_costs.py and README."
    )