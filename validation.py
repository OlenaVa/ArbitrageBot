"""
Diagnostics that ask "should this parameter/result be trusted", as opposed
to spread_model.py which defines what the parameters/results ARE.

Two tools live here:
  1. estimate_ou_half_life  - a data-derived justification for the z-score
     window, instead of "chosen because it seemed reasonable".
  2. run_parameter_sensitivity - a grid over entry/exit thresholds, to check
     whether the strategy only works at the one specific (entry, exit) pair
     that happens to be configured, or across a broad plausible range.

IMPORTANT SCOPING RULE for run_parameter_sensitivity: only ever call it on
the DEVELOPMENT slice of the data (see main.py - currently 2019-2021).
Running it on the validation or final-test slice and picking whichever
thresholds score best there turns this diagnostic into exactly the kind of
in-sample parameter search it exists to detect. If you ever change that,
say so explicitly in the README - silently expanding the sweep's data
window is the single easiest way to quietly overfit this strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import StrategyConfig
import spread_model


# =====================================================================
# OU / AR(1) half-life
# =====================================================================

def estimate_ou_half_life(spread: pd.Series) -> dict:
    """
    Fits a discrete-time Ornstein-Uhlenbeck / AR(1) process to the spread by
    OLS regression of its change on its own lagged level:

        spread[t] - spread[t-1] = alpha + theta_coef * spread[t-1] + e[t]

    theta = -theta_coef is the (discrete-time-approximated) mean-reversion
    speed; half_life = ln(2) / theta is the number of days it takes a
    deviation to decay halfway back to the mean. This is the standard
    "OU half-life" estimator used in the pairs-trading literature (e.g.
    Ernest Chan's mean-reversion half-life test).

    Also reports the p-value on theta_coef: if it's not significantly
    negative, there is no statistically detectable mean reversion in this
    sample, and a half-life number would be spurious - that case is
    reported explicitly rather than returning a misleadingly precise number.
    """
    s = spread.dropna()
    lagged = s.shift(1)
    delta = s.diff()
    data = pd.concat([lagged, delta], axis=1, keys=["lagged", "delta"]).dropna()

    if len(data) < 30:
        return {
            "half_life_days": None,
            "mean_reverting": False,
            "note": f"Only {len(data)} usable observations - too few to fit an OU process.",
        }

    X = sm.add_constant(data["lagged"])
    model = sm.OLS(data["delta"], X).fit()

    theta_coef = float(model.params["lagged"])
    theta_pvalue = float(model.pvalues["lagged"])
    theta = -theta_coef

    result = {
        "theta_coef": theta_coef,
        "theta": theta,
        "theta_pvalue": theta_pvalue,
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
    }

    if theta <= 0:
        result.update({
            "half_life_days": None,
            "mean_reverting": False,
            "note": "OLS coefficient on the lagged spread is >= 0: no mean "
                    "reversion detected over this sample (or too weak/noisy "
                    "to estimate a finite half-life).",
        })
        return result

    half_life = float(np.log(2) / theta)
    result.update({
        "half_life_days": half_life,
        "mean_reverting": theta_pvalue < 0.05,
        "suggested_window_1_5x_half_life": round(1.5 * half_life),
        "suggested_window_2x_half_life": round(2.0 * half_life),
        "note": None if theta_pvalue < 0.05 else
        "theta is negative (mean-reverting direction) but NOT "
        "statistically significant at 5% - treat the half-life "
        "figure as indicative, not a precise estimate.",
    })
    return result


# =====================================================================
# Parameter sensitivity sweep
# =====================================================================

def run_parameter_sensitivity(
        df_dev: pd.DataFrame,
        config: StrategyConfig,
        entry_grid=(1.5, 1.8, 2.0, 2.2, 2.5),
        exit_grid=(0.0, 0.3, 0.5, 0.7),
) -> pd.DataFrame:
    """
    Sweeps entry/exit thresholds on a FIXED df_dev (development slice only),
    reusing the SAME beta/spread/z/regime/risk_scale columns already on
    df_dev (those don't depend on entry/exit) and only re-running the
    position + PnL step per (entry, exit) combination via
    spread_model.compute_positions / compute_performance - the identical
    functions used for the primary backtest, so this sweep can't drift from
    how the "real" run is scored.

    df_dev must already have: WTI, Brent, beta, z, mr_regime, risk_scale.
    Combinations where exit >= entry are skipped (nonsensical: you'd never
    reach the exit condition after entering).
    """
    rows = []
    for entry in entry_grid:
        for exit_ in exit_grid:
            if exit_ >= entry:
                continue
            positions = spread_model.compute_positions(df_dev, entry, exit_)
            position = positions * df_dev["risk_scale"].values
            perf = spread_model.compute_performance(df_dev, position, config.cost_per_turnover)
            rows.append({
                "entry": entry,
                "exit": exit_,
                "sharpe": perf["sharpe"],
                "max_dd_pct": perf["max_dd"] * 100 if perf["max_dd"] == perf["max_dd"] else np.nan,
                "n_trades": perf["n_trades"],
            })

    result = pd.DataFrame(rows).sort_values(["entry", "exit"]).reset_index(drop=True)
    return result


def summarize_sensitivity(sensitivity_df: pd.DataFrame, configured_entry: float, configured_exit: float) -> str:
    """One-line, honest summary: does the strategy only work at the
    configured (entry, exit) pair, or across a broad range?"""
    valid = sensitivity_df.dropna(subset=["sharpe"])
    if valid.empty:
        return "No valid (entry, exit) combination produced a Sharpe ratio on this slice."

    positive = valid[valid["sharpe"] > 0]
    frac_positive = len(positive) / len(valid)
    configured_row = valid[
        np.isclose(valid["entry"], configured_entry) & np.isclose(valid["exit"], configured_exit)
        ]
    configured_sharpe = configured_row["sharpe"].iloc[0] if not configured_row.empty else float("nan")

    return (
            f"Configured (entry={configured_entry}, exit={configured_exit}) Sharpe on dev period: "
            f"{configured_sharpe:.3f}. Sharpe range across the grid: "
            f"[{valid['sharpe'].min():.3f}, {valid['sharpe'].max():.3f}]. "
            f"{frac_positive:.0%} of tested combinations were Sharpe > 0. "
            + ("Broad support across parameters - lower overfitting risk on this axis."
               if frac_positive >= 0.7 else
               "Performance is concentrated in a narrow parameter region - treat the "
               "configured thresholds with more caution.")
    )