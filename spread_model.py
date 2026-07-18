"""
Shared spread model: everything about defining the traded WTI/Brent spread
and turning it into a position and a P&L, used identically by main.py's
backtest and market_check.py's "should we trade now" checker.

Why this file exists: main.py and market_check.py used to each contain
their own copy of the Kalman filter (identical Q/R/clip/smoothing
constants, pasted twice). If one were edited and the other forgotten,
the backtest and the live checker would silently stop agreeing on what
"the spread" even is. Everything here is now called from exactly one
place by both scripts.

v2 methodology change - PURE KALMAN FILTER (previously a hybrid):
Earlier versions clipped beta to [beta_warn_min, beta_warn_max] and then
applied an extra 0.98/0.02 exponential smoothing pass AFTER the Kalman
update, while the uncertainty term P continued to evolve as though the
raw (unclipped, unsmoothed) beta were what propagated forward. That was
an internal inconsistency: P no longer described the uncertainty of the
value actually used downstream.

This version removes the clip and the smoothing entirely. beta[t] is
exactly the Kalman posterior mean and P[t] its matching posterior
variance. The old range is kept ONLY as a diagnostic band: if beta
wanders outside it, this is logged and counted (see KalmanDiagnostics)
but beta itself is never altered.

Practical consequence, expected and not a bug: without the smoothing
pass, beta reacts more sharply in volatile periods (Q_t and R_t both
already scale up with volatility, so the Kalman gain is largest exactly
when the old smoothing used to damp it most). See README.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import StrategyConfig

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op fallback decorator so this module works correctly (just
        without the JIT speedup) if numba isn't installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func
        return decorator


# =====================================================================
# 1. DYNAMIC HEDGE RATIO - PURE KALMAN FILTER
# =====================================================================

@dataclass
class KalmanDiagnostics:
    """What actually happened during the Kalman run, so README/CV claims
    can cite exact figures instead of 'it might sometimes drift'."""
    n_obs: int
    n_beta_below_min: int
    n_beta_above_max: int
    beta_min_seen: float
    beta_max_seen: float
    first_out_of_band_index: Optional[object]

    @property
    def n_out_of_band(self) -> int:
        return self.n_beta_below_min + self.n_beta_above_max

    @property
    def pct_out_of_band(self) -> float:
        return self.n_out_of_band / self.n_obs if self.n_obs else 0.0

    def __str__(self) -> str:
        return (
                f"n_obs={self.n_obs}, out_of_band={self.n_out_of_band} "
                f"({self.pct_out_of_band:.2%}), "
                f"beta range seen=[{self.beta_min_seen:.4f}, {self.beta_max_seen:.4f}]"
                + (f", first out-of-band on {self.first_out_of_band_index}" if self.n_out_of_band else "")
        )


@njit(cache=True)
def _kalman_core(x_arr, y_arr, brent_pct, beta_init, p_init, q, r, q_vol_mult, r_vol_mult):
    """Pure Kalman recursion - no clip, no smoothing. Precompiled numeric
    core; all pandas/diagnostics handling lives in compute_beta_and_spread."""
    n = len(x_arr)
    beta = np.zeros(n)
    p = np.zeros(n)
    beta[0] = beta_init
    p[0] = p_init

    for t in range(1, n):
        x = x_arr[t]
        y = y_arr[t]
        bp = brent_pct[t]
        vol = abs(bp) if (t > 1 and not np.isnan(bp)) else 0.0

        q_t = q * (1.0 + q_vol_mult * vol)
        r_t = r * (1.0 + r_vol_mult * vol)
        p_pred = p[t - 1] + q_t

        err = y - beta[t - 1] * x
        s = x * p_pred * x + r_t
        k = p_pred * x / (s + 1e-12)

        beta[t] = beta[t - 1] + k * err          # pure posterior mean - no clip, no smoothing
        p[t] = (1.0 - k * x) * p_pred             # matching posterior variance

    return beta, p


def compute_beta_and_spread(df: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, KalmanDiagnostics]:
    """
    Pure Kalman filter for the dynamic hedge ratio beta in y = beta * x + noise,
    where x = log(WTI), y = log(Brent) must already be columns on df.

    Returns (df_with_beta_P_spread, diagnostics). `diagnostics` is a
    DIAGNOSTIC-ONLY report of how often beta left the historically-plausible
    [beta_warn_min, beta_warn_max] band - nothing here alters beta based on it.
    A warning is also raised (Python `warnings`) if the band was ever left,
    so it can't be missed even if the diagnostics object itself is ignored.
    """
    x_arr = np.asarray(df["x"].values, dtype=np.float64)
    y_arr = np.asarray(df["y"].values, dtype=np.float64)
    brent_pct = np.asarray(df["Brent"].pct_change().values, dtype=np.float64)  # computed ONCE - O(n), not O(n^2)

    beta, p = _kalman_core(
        x_arr, y_arr, brent_pct,
        float(config.beta_init), float(config.p_init),
        float(config.kalman_q), float(config.kalman_r),
        float(config.kalman_q_vol_multiplier), float(config.kalman_r_vol_multiplier),
    )

    df = df.copy()
    df["beta"] = beta
    df["P"] = p
    df["spread"] = df["y"] - df["beta"] * df["x"]

    below_mask = beta < config.beta_warn_min
    above_mask = beta > config.beta_warn_max
    n_below = int(below_mask.sum())
    n_above = int(above_mask.sum())
    first_idx = None
    out_positions = np.where(below_mask | above_mask)[0]
    if len(out_positions) > 0:
        first_idx = df.index[out_positions[0]]

    diagnostics = KalmanDiagnostics(
        n_obs=len(beta),
        n_beta_below_min=n_below,
        n_beta_above_max=n_above,
        beta_min_seen=float(np.min(beta)),
        beta_max_seen=float(np.max(beta)),
        first_out_of_band_index=first_idx,
    )

    if diagnostics.n_out_of_band > 0:
        warnings.warn(
            f"beta left the diagnostic band [{config.beta_warn_min}, {config.beta_warn_max}] "
            f"on {diagnostics.n_out_of_band}/{diagnostics.n_obs} days "
            f"({diagnostics.pct_out_of_band:.2%}); range seen "
            f"[{diagnostics.beta_min_seen:.4f}, {diagnostics.beta_max_seen:.4f}]. "
            f"Diagnostic only - beta was NOT clipped or altered.",
            stacklevel=2,
        )

    return df, diagnostics


# =====================================================================
# 2. Z-SCORE SIGNAL
# =====================================================================

def compute_zscore(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Rolling z-score of the spread. mean/std use .shift(1) so today's
    z-score never uses today's own value in the window - avoids look-ahead."""
    df = df.copy()
    mean = df["spread"].rolling(config.z_window).mean().shift(1)
    std = df["spread"].rolling(config.z_window).std().shift(1)
    df["z"] = (df["spread"] - mean) / (std + 1e-12)
    return df


# =====================================================================
# 3. REGIME FILTER
# =====================================================================

def compute_regime_filter(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """True when recent spread volatility sits below its own trailing
    historical `regime_quantile`. Shifted by 1 so the label is known
    before the day starts (uses no same-day information)."""
    df = df.copy()
    ret = df["spread"].diff()
    vol = ret.rolling(config.regime_vol_window).std()
    threshold = vol.rolling(config.regime_lookback).quantile(config.regime_quantile)
    df["mr_regime"] = (vol < threshold).shift(1).fillna(0)
    return df


# =====================================================================
# 4. RISK SCALING (volatility targeting)
# =====================================================================

def compute_risk_scale(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """EWMA-covariance-based volatility targeting for the HEDGED
    Brent-vs-beta*WTI portfolio (not each leg individually):
    Var(Brent - beta*WTI) = Var(Brent) + beta^2*Var(WTI) - 2*beta*Cov(Brent,WTI).
    """
    df = df.copy()
    ret_b = np.log(df["Brent"]).diff()
    ret_w = np.log(df["WTI"]).diff()

    var_b = ret_b.ewm(span=config.ewma_span).var()
    var_w = ret_w.ewm(span=config.ewma_span).var()
    cov = ret_b.ewm(span=config.ewma_span).cov(ret_w)

    portfolio_var = var_b + (df["beta"] ** 2) * var_w - 2 * df["beta"] * cov
    portfolio_vol = np.sqrt(np.clip(portfolio_var, 1e-12, None)) * np.sqrt(252)

    risk_scale = (config.target_annual_vol / (portfolio_vol + 1e-12)).shift(1)
    df["risk_scale"] = risk_scale.clip(config.risk_scale_min, config.risk_scale_max).fillna(1.0)
    return df


# =====================================================================
# 5. ENTRY/EXIT WITH HYSTERESIS
# =====================================================================

@njit(cache=True)
def _positions_core(z_arr, mr_arr, entry, exit_):
    n = len(z_arr)
    pos = 0.0
    positions = np.zeros(n)
    for i in range(n):
        zi = z_arr[i]
        mri = mr_arr[i]
        if pos == 0.0:
            if mri > 0.0:
                if zi > entry:
                    pos = -1.0
                elif zi < -entry:
                    pos = 1.0
        else:
            if pos == -1.0 and zi < exit_:
                pos = 0.0
            elif pos == 1.0 and zi > -exit_:
                pos = 0.0
        positions[i] = pos
    return positions


def compute_positions(df: pd.DataFrame, entry: float, exit_: float) -> np.ndarray:
    """
    Entry/exit hysteresis state machine over z-score + regime filter.
    pos=+1: long the spread; pos=-1: short the spread - the sign only
    matters via the PnL formula in compute_performance, there is no
    explicit order logic here. Two different thresholds (hysteresis)
    stop the position flipping open/closed every time z oscillates near
    a single boundary.

    NaN z-values (the warmup period) always compare False, so no signal
    fires until enough history exists - unchanged behaviour from before.
    """
    z_arr = np.asarray(df["z"].values, dtype=np.float64)
    mr_arr = np.asarray(df["mr_regime"].values, dtype=np.float64)
    return _positions_core(z_arr, mr_arr, float(entry), float(exit_))


# =====================================================================
# 6. PERFORMANCE: position path (+ a cost) -> PnL and metrics
# =====================================================================

def compute_performance(df: pd.DataFrame, position, cost: float) -> dict:
    """
    Turns a (risk-scaled) position path into PnL and standard metrics.
    This is the SINGLE place performance is computed from a position path -
    used by main.py's headline numbers, its walk-forward sub-periods,
    validation.run_parameter_sensitivity, and costs.run_cost_stress_test -
    so all four report numbers computed exactly the same way, varying only
    what each of them is actually testing (thresholds, or cost, or window).

    `position` may be a pd.Series or np.ndarray aligned to df.index - the
    already risk-scaled position, BEFORE the one-day lag applied here.
    Uses yesterday's position and beta against today's return
    (`pos_lag`, `beta_lag`) - today's own position/beta must never be used,
    or this leaks future information into the backtest.
    """
    position = pd.Series(np.asarray(position, dtype=np.float64), index=df.index)

    dB = np.log(df["Brent"]).diff()
    dW = np.log(df["WTI"]).diff()
    pos_lag = position.shift(1)
    beta_lag = df["beta"].shift(1)
    gross = 1.0 + np.abs(beta_lag)  # normalizes by total gross exposure (1 WTI-leg unit + beta Brent-leg units)

    strategy_log_ret = pos_lag * (dB - beta_lag * dW) / gross
    turnover = position.diff().abs()
    net_log_ret = strategy_log_ret - turnover * cost

    valid = net_log_ret.dropna()
    n_trades = int((turnover.dropna() > 1e-9).sum())

    if len(valid) == 0 or valid.std() == 0:
        return {
            "net_log_ret": net_log_ret, "equity": None,
            "sharpe": np.nan, "sortino": np.nan, "max_dd": np.nan,
            "n_trades": n_trades, "n_obs": len(valid),
        }

    r = np.exp(valid) - 1
    sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252)
    down = r[r < 0]
    sortino = (r.mean() / (down.std() + 1e-12)) * np.sqrt(252) if len(down) > 0 else np.nan
    equity = np.exp(valid.cumsum())
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "net_log_ret": net_log_ret,
        "equity": equity,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "n_obs": len(valid),
    }