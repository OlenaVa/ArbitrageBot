
"""
main.py - full Brent/WTI stat-arb backtest.

v3 changes (this version):
- Uses shared spread_model.py / config.py instead of duplicating the Kalman
  filter, z-score, regime filter and risk-scaling logic that also lives in
  market_check.py. See spread_model.py for why that mattered.
- Pure Kalman filter: no clip, no smoothing (see spread_model.py docstring).
  The old [0.5, 2.0] range is now a diagnostic-only warning band.
- Adds an OU/AR(1) half-life estimate, to justify the z-score window with
  data instead of "chosen because it seemed reasonable" (section 2.6).
- Adds a parameter sensitivity sweep over entry/exit thresholds, run ONLY
  on the 2019-2021 development slice (section 8.6).
- Adds a transaction-cost stress test at 0.5 / 1.0 / 2.0 bps (section 8.7).
- The walk-forward split is now explicitly labeled development /
  validation / final-test (section 8.5) - see README "Known limitations"
  for what this does and does not yet guarantee.
- Numba-accelerated core loops in spread_model.py (falls back to pure
  Python automatically if numba isn't installed - see the printed notice
  below).
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from config import StrategyConfig
import spread_model as sm
import validation as val
import costs as ct

config = StrategyConfig()

print(f"Numba JIT: {'enabled' if sm.NUMBA_AVAILABLE else 'not installed - falling back to pure Python (still correct, just slower; pip install numba to enable)'}")

# =====================================================
# 1. DATA LOADING
# =====================================================
raw = yf.download(["CL=F", "BZ=F"], start="2019-01-01", progress=False)

# CL=F (WTI) went negative in April 2020 (famous negative-price event).
# log() of a non-positive price is undefined, so those rows are dropped.
if (raw <= 0).any().any():
    print("Warning: zero or negative prices detected (e.g. WTI negative-price day) - dropping those rows")
    raw = raw[raw > 0]

# yfinance sometimes returns a MultiIndex (field, ticker); flatten to just Close prices
if isinstance(raw.columns, pd.MultiIndex):
    raw = raw.xs("Close", axis=1, level=0)

raw = raw.dropna()
raw = raw.rename(columns={"CL=F": "WTI", "BZ=F": "Brent"})

df = pd.DataFrame(index=raw.index)
df["WTI"] = raw["WTI"]
df["Brent"] = raw["Brent"]

# Work in log-prices: makes beta a ratio of percentage moves, not dollar
# moves - standard for cointegration analysis.
df["x"] = np.log(df["WTI"])
df["y"] = np.log(df["Brent"])

# =====================================================
# 2. DYNAMIC HEDGE RATIO - PURE KALMAN FILTER (spread_model.py)
# =====================================================
df, kalman_diag = sm.compute_beta_and_spread(df, config)
print(f"\nKalman diagnostics: {kalman_diag}")

# NOTE on methodology change: this used to also clip beta to [0.5, 2.0] and
# apply an extra 0.98/0.02 smoothing pass. Both are removed - see
# spread_model.py and README "Known limitations / methodology changes".
# The line above tells you, with real numbers from THIS run, how often the
# unclipped filter actually left that historically-plausible range.

# =====================================================
# 2.5. HYPOTHESIS TEST: is the traded spread actually mean-reverting?
# =====================================================
adf_result = adfuller(df["spread"].dropna())
adf_pvalue = adf_result[1]
print(f"\nADF test on the traded spread: p-value = {adf_pvalue:.4f}")
print(f"Stationary (p<0.05): {adf_pvalue < 0.05}")

# =====================================================
# 2.6. OU HALF-LIFE: data-derived justification for the z-score window
# =====================================================
ou = val.estimate_ou_half_life(df["spread"])
print("\n=== OU MEAN-REVERSION HALF-LIFE ===")
if ou.get("half_life_days") is None:
    print(f"No usable half-life estimate: {ou.get('note')}")
    print(f"Falling back to the configured z_window={config.z_window} "
          f"(not data-derived on this run).")
else:
    print(f"theta={ou['theta']:.5f} (p={ou['theta_pvalue']:.4f}), "
          f"half-life={ou['half_life_days']:.1f} days, "
          f"significant at 5%: {ou['mean_reverting']}")
    print(f"Suggested window (1.5x-2x half-life): "
          f"{ou['suggested_window_1_5x_half_life']}-{ou['suggested_window_2x_half_life']} days "
          f"vs. configured z_window={config.z_window}")
    if ou.get("note"):
        print(f"Caveat: {ou['note']}")

# =====================================================
# 3. SIGNAL: Z-SCORE OF THE SPREAD (spread_model.py)
# =====================================================
df = sm.compute_zscore(df, config)

# =====================================================
# 4. REGIME FILTER: only trade when the spread itself is calm (spread_model.py)
# =====================================================
df = sm.compute_regime_filter(df, config)

# =====================================================
# 5. RISK SCALING: target a constant portfolio volatility (spread_model.py)
# =====================================================
df = sm.compute_risk_scale(df, config)

# =====================================================
# 6. ENTRY/EXIT LOGIC WITH HYSTERESIS (spread_model.py, numba core)
# =====================================================
positions = sm.compute_positions(df, config.entry_threshold, config.exit_threshold)
df["position"] = positions * df["risk_scale"]

# =====================================================
# 7. PnL + 8. PERFORMANCE METRICS (spread_model.compute_performance)
# =====================================================
perf = sm.compute_performance(df, df["position"], config.cost_per_turnover)
df["net_log_ret"] = perf["net_log_ret"]
# full-length equity curve for plotting (treats the pre-warmup NaN period as
# flat at 1.0, same convention as before - compute_performance's own equity
# series is only over valid rows, which is what feeds the metrics below)
df["equity"] = np.exp(df["net_log_ret"].fillna(0).cumsum())

sharpe, sortino, dd = perf["sharpe"], perf["sortino"], perf["max_dd"]

print(f"\n=== BACKTEST RESULTS (base cost = {config.cost_per_turnover*10000:.1f} bps) ===")
print(f"Sharpe:  {sharpe:.3f}")
print(f"Sortino: {sortino:.3f}")
print(f"Max DD:  {dd*100:.2f}%")
print(f"Trades:  {perf['n_trades']}")

df_valid = df.dropna(subset=["net_log_ret"])
r = np.exp(df_valid["net_log_ret"]) - 1

# =====================================================
# 8.5. WALK-FORWARD: development / validation / final-test sub-periods
# =====================================================
# Labeled explicitly (not just "sub-period 1/2/3") because that labeling is
# the point: 2019-2021 is where thresholds may be inspected and reasoned
# about; 2024-2026 is meant to be looked at once, honestly, and not tuned
# against afterward. See README "Known limitations" for what this framing
# does and does not yet guarantee (it is NOT yet a re-fit-per-period
# out-of-sample test - that's a planned follow-up, not implemented here).
splits = {
    "2019-2021 (development)": ("2019", "2021"),
    "2022-2023 (validation)": ("2022", "2023"),
    "2024-2026 (final test - do not tune against this)": ("2024", "2026"),
}

print("\n=== WALK-FORWARD BY SUB-PERIOD ===")
for name, (start, end) in splits.items():
    sub = df[start:end]
    if len(sub) < 30:
        print(f"{name}: not enough data ({len(sub)} days)")
        continue
    sub_perf = sm.compute_performance(sub, sub["position"], config.cost_per_turnover)
    print(f"{name}: Sharpe={sub_perf['sharpe']:.3f}, MaxDD={sub_perf['max_dd']*100:.2f}%, N={len(sub)}")

print(f"\nWorst single day: {r.min()*100:.2f}%")
print(f"Best single day:  {r.max()*100:.2f}%")
print(f"Median leverage (risk_scale): {df['risk_scale'].median():.2f}")
print(f"Max leverage (risk_scale):    {df['risk_scale'].max():.2f}")
print(f"Realized annualized volatility: {r.std()*np.sqrt(252)*100:.2f}%")

# =====================================================
# 8.6. PARAMETER SENSITIVITY SWEEP - DEVELOPMENT PERIOD ONLY
# =====================================================
# Only ever run on 2019-2021. Running this on validation/final-test data and
# picking whichever thresholds score best there would turn this diagnostic
# into exactly the in-sample parameter search it exists to catch.
print("\n=== PARAMETER SENSITIVITY: entry/exit grid, 2019-2021 (development) ONLY ===")
dev_df = df["2019":"2021"]
sensitivity = val.run_parameter_sensitivity(dev_df, config)
print(sensitivity.to_string(index=False))
print(val.summarize_sensitivity(sensitivity, config.entry_threshold, config.exit_threshold))

# =====================================================
# 8.7. TRANSACTION COST STRESS TEST
# =====================================================
print("\n=== TRANSACTION COST STRESS TEST (full history) ===")
stress = ct.run_cost_stress_test(df)
print(stress.to_string(index=False))
print(ct.summarize_cost_stress(stress, config.cost_per_turnover * 10000))

# =====================================================
# 9. EQUITY CURVE
# =====================================================
plt.figure(figsize=(12, 5))
plt.plot(df["equity"], lw=1.6, color="black")
plt.title("Brent-WTI Stat-Arb Backtest")
plt.grid(alpha=0.3)
plt.show()

# =====================================================
# 10. SUMMARY: what to check before trusting this backtest
# =====================================================
print("\n=== HOW MUCH TO TRUST THIS BACKTEST ===")
print(f"1. Traded spread is stationary over full history (ADF p<0.05): "
      f"{adf_pvalue < 0.05} (p={adf_pvalue:.4f})")
print(f"2. Mean-reversion half-life is statistically significant: "
      f"{ou.get('mean_reverting', 'N/A')}")
print(f"3. Sharpe stable across development/validation/final-test: see "
      f"'WALK-FORWARD' output above - compare manually whether the result "
      f"depends on one specific window")
print(f"4. Sharpe not concentrated in one specific (entry, exit) pair "
      f"(development period only): see 'PARAMETER SENSITIVITY' above")
print(f"5. Sharpe survives higher assumed transaction costs: see "
      f"'TRANSACTION COST STRESS TEST' above")
print(f"6. Kalman beta stayed within its historically-plausible band: "
      f"{kalman_diag.pct_out_of_band:.2%} of days out-of-band (diagnostic, "
      f"not enforced - see point 2 above)")
print("7. NOT modeled: futures contract roll mechanics (CL=F/BZ=F are "
      "continuous series) - see README 'Known limitations'.")