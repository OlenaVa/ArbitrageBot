"""
market_check.py - "should we trade right now?"

v3 change: the Kalman filter / z-score / regime filter that used to be
duplicated here (identical Q/R/clip/smoothing constants, pasted from
main.py) now come from spread_model.py + config.py. This file can no longer
silently drift from what main.py's backtest actually did - both call the
exact same functions with the exact same StrategyConfig.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

from config import StrategyConfig
import spread_model as sm

DEFAULT_CONFIG = StrategyConfig()


def load_recent_data(days=250):
    """
    Loads enough history to compute a stable Kalman beta and a rolling ADF
    test, without re-running the full 2019-present backtest.
    """
    raw = yf.download(["CL=F", "BZ=F"], period=f"{days}d", progress=False, auto_adjust=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("Close", axis=1, level=0)

    raw = raw.rename(columns={"CL=F": "WTI", "BZ=F": "Brent"})
    raw = raw.dropna()

    invalid = (raw <= 0).any(axis=1)
    if invalid.any():
        raw = raw[~invalid]

    return raw


def check_market_health(days=250, rolling_adf_window=90, config: StrategyConfig = None):
    """
    Answers "should we trade right now" - using spread_model.py, the SAME
    beta/spread/z/regime definitions as main.py's backtest, on a recent
    (short) window.
    """
    config = config or DEFAULT_CONFIG

    raw = load_recent_data(days=days)
    if len(raw) < rolling_adf_window + 30:
        print(f"Not enough data: need at least {rolling_adf_window + 30} days")
        return None

    df = raw.copy()
    df["x"] = np.log(df["WTI"])
    df["y"] = np.log(df["Brent"])
    df, kalman_diag = sm.compute_beta_and_spread(df, config)
    df = sm.compute_zscore(df, config)
    df = sm.compute_regime_filter(df, config)
    df = sm.compute_risk_scale(df, config)

    # rolling ADF: is the spread stationary on a recent window, not just
    # over the multi-year backtest - this is the actual "trade now?" check
    adf_pvalues = [np.nan] * rolling_adf_window
    spread_vals = df["spread"].values
    for i in range(rolling_adf_window, len(df)):
        window_data = spread_vals[i - rolling_adf_window:i]
        try:
            p = adfuller(window_data)[1]
        except Exception:
            p = np.nan
        adf_pvalues.append(p)
    df["rolling_adf_pvalue"] = adf_pvalues

    return {
        "date": df.index[-1],
        "wti": df["WTI"].iloc[-1],
        "brent": df["Brent"].iloc[-1],
        "beta": df["beta"].iloc[-1],
        "spread": df["spread"].iloc[-1],
        "z": df["z"].iloc[-1],
        "mr_regime": bool(df["mr_regime"].iloc[-1]),
        "risk_scale": df["risk_scale"].iloc[-1],
        "rolling_adf_pvalue": df["rolling_adf_pvalue"].iloc[-1],
        "locally_stationary": df["rolling_adf_pvalue"].iloc[-1] < 0.05,
        "kalman_diagnostics": kalman_diag,
    }


def describe_trade_action(health, capital_usd=100_000, config: StrategyConfig = None):
    """
    Translates the abstract position signal (+1/-1/0) into concrete futures
    contracts: how many WTI (CL=F) and Brent (BZ=F) contracts, and which
    side of each. 1 contract = 1000 barrels for both CL and BZ.
    """
    config = config or DEFAULT_CONFIG
    CONTRACT_SIZE = 1000  # barrels per futures contract

    z = health["z"]
    beta = health["beta"]
    wti_price = health["wti"]
    entry = config.entry_threshold
    target_annual_vol = config.target_annual_vol

    if not health["locally_stationary"] or not health["mr_regime"] or abs(z) < entry:
        return None  # no entry signal - nothing to size

    # direction: z > entry means the spread (Brent side) is "too expensive"
    # relative to beta*WTI -> short the spread: sell Brent leg, buy WTI leg.
    # z < -entry is the mirror case.
    if z > 0:
        brent_side, wti_side = "SELL (short)", "BUY (long)"
    else:
        brent_side, wti_side = "BUY (long)", "SELL (short)"

    # Position sizing: reuses the SAME risk_scale that check_market_health
    # already computed via spread_model.compute_risk_scale (identical EWMA
    # portfolio-variance vol-targeting to main.py's backtest), instead of a
    # flat capital fraction. target_annual_vol is the notional-to-capital
    # ratio at risk_scale == 1.0 (i.e. when current realized vol sits
    # exactly at target); risk_scale scales that baseline up or down
    # depending on whether today's vol is below or above target - same
    # relationship as `position = raw_signal * risk_scale` in main.py.
    # Still a research signal check, not a decision-grade sizing model -
    # see "Known limitations" (no contract specs, margin, or staleness
    # checks here).
    notional = capital_usd * target_annual_vol * health["risk_scale"]
    wti_contracts = notional / (wti_price * CONTRACT_SIZE)
    brent_contracts = wti_contracts * beta

    if round(wti_contracts) == 0:
        min_capital = wti_price * CONTRACT_SIZE / (target_annual_vol * health["risk_scale"])
        return {
            "error": f"Computed size ({wti_contracts:.3f} contracts) is below 1 - "
                     f"with capital ${capital_usd:,} this strategy cannot be executed "
                     f"in whole contracts. Minimum capital needed: ${min_capital:,.0f}."
        }

    return {
        "brent_action": brent_side,
        "brent_contracts": round(brent_contracts),
        "wti_action": wti_side,
        "wti_contracts": round(wti_contracts),
    }


if __name__ == "__main__":
    health = check_market_health()
    if health:
        print(f"Date: {health['date'].date()}")
        print(f"WTI: ${health['wti']:.2f}   Brent: ${health['brent']:.2f}")
        print(f"Beta: {health['beta']:.4f}")
        print(f"Kalman diagnostics: {health['kalman_diagnostics']}")
        print(f"Spread (log): {health['spread']:.4f}")
        print(f"Z-score: {health['z']:.2f}")
        print(f"Regime (calm market): {health['mr_regime']}")
        print(f"Risk scale (vol-target leverage): {health['risk_scale']:.2f}")
        print(f"Rolling ADF p-value (90d): {health['rolling_adf_pvalue']:.4f}")
        print(f"Locally stationary: {health['locally_stationary']}")

        print("\n=== CONCLUSION ===")
        if not health["locally_stationary"]:
            print("Spread is NOT stationary on the short window - "
                  "the short-horizon mean-reversion hypothesis is not "
                  "currently supported.")
        elif not health["mr_regime"]:
            print("Market is currently too volatile (regime filter is off).")
        elif abs(health["z"]) < DEFAULT_CONFIG.entry_threshold:
            print(f"Spread is within its normal range (z={health['z']:.2f}, "
                  f"entry threshold {DEFAULT_CONFIG.entry_threshold}) - no entry signal.")
        else:
            direction = "short-spread (short Brent-leg, long beta*WTI-leg)" if health["z"] > 0 \
                else "long-spread (long Brent-leg, short beta*WTI-leg)"
            print(f"Conditions met: locally stationary, calm regime, "
                  f"z={health['z']:.2f} beyond entry threshold -> signal: {direction}")

            trade = describe_trade_action(health)
            if trade:
                if "error" in trade:
                    print(f"\n=== SPECIFIC ACTION ===\n{trade['error']}")
                else:
                    print("\n=== SPECIFIC ACTION ===")
                    print(f"Brent (BZ=F): {trade['brent_action']} {trade['brent_contracts']} contracts")
                    print(f"WTI   (CL=F): {trade['wti_action']} {trade['wti_contracts']} contracts")
                    print("(sized for $100,000 capital and a 10%/yr risk target - "
                          "pass capital_usd=... to describe_trade_action() to match your own capital)")