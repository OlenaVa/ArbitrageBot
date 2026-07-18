"""
Single source of truth for every tunable constant in this project.

Before this file existed, the same numbers (Kalman Q/R, beta clip bounds,
z-score window, entry/exit thresholds, regime parameters, risk target,
cost) were hardcoded independently in main.py and market_check.py. That's
exactly the kind of drift that lets a backtest and its "live" checker
quietly stop agreeing with each other. Now both import StrategyConfig from
here, so there is exactly one place to change a parameter.

None of these were fit by optimizing the backtest's own Sharpe ratio -
see README "Hardcoded parameters" for the reasoning behind each one, and
`validation.py` for the data-derived checks (OU half-life, parameter
sensitivity sweep) that test whether they hold up.
"""
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    # ---- Kalman filter (pure - no clip, no smoothing; see spread_model.py) ----
    kalman_q: float = 1e-6              # process noise: how fast beta may drift
    kalman_r: float = 0.01              # observation noise: trust in a single day's price pair
    kalman_q_vol_multiplier: float = 10.0   # Q_t = kalman_q * (1 + kalman_q_vol_multiplier * vol)
    kalman_r_vol_multiplier: float = 5.0    # R_t = kalman_r * (1 + kalman_r_vol_multiplier * vol)
    beta_init: float = 1.0
    p_init: float = 1.0

    # Diagnostic-only plausibility band. NOT applied to beta (no clipping) -
    # only used to log/count how often the filter's own posterior beta
    # wanders outside a historically-plausible WTI/Brent ratio.
    beta_warn_min: float = 0.5
    beta_warn_max: float = 2.0

    # ---- Signal generation ----
    z_window: int = 30                  # see validation.estimate_ou_half_life for the data-derived check
    entry_threshold: float = 1.8
    exit_threshold: float = 0.3

    # ---- Regime filter ----
    regime_vol_window: int = 20
    regime_lookback: int = 100
    regime_quantile: float = 0.7

    # ---- Risk scaling (volatility targeting) ----
    target_annual_vol: float = 0.10
    risk_scale_min: float = 0.1
    risk_scale_max: float = 2.0
    ewma_span: int = 30

    # ---- Transaction costs ----
    # Base case used for the primary reported Sharpe/Sortino/MaxDD in main.py.
    # See costs.py for the low/base/stress sensitivity scenarios around this.
    cost_per_turnover: float = 0.0005   # 5 bps - flat assumption, not derived from real quotes