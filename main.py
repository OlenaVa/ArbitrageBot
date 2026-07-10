import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

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

# Work in log-prices: makes the beta below a ratio of percentage moves,
# not dollar moves, which is standard for cointegration analysis
df["x"] = np.log(df["WTI"])
df["y"] = np.log(df["Brent"])

n = len(df)

# =====================================================
# 2. DYNAMIC HEDGE RATIO VIA KALMAN FILTER
# =====================================================
# Instead of a single fixed beta (Brent = beta * WTI) fit once on the whole
# history, this estimates beta day by day, letting the relationship drift
# slowly over time (the true ratio between Brent and WTI is not constant).
beta = np.zeros(n)
beta[0] = 1.0

P = 1.0       # estimate uncertainty (variance of beta)
Q = 1e-6      # process noise: how much beta is allowed to drift per step
R = 0.01      # observation noise: how much we trust today's price pair

for t in range(1, n):
    x = df["x"].iloc[t]
    y = df["y"].iloc[t]

    # scale both noise terms up on volatile days, so beta reacts faster
    # when the market is moving a lot, and stays stable when it's quiet
    vol = abs(df["Brent"].pct_change().iloc[t]) if t > 1 else 0.0
    Q_t = Q * (1 + 10 * vol)
    R_t = R * (1 + 5 * vol)

    P_pred = P + Q_t

    err = y - beta[t-1] * x          # prediction error (innovation)
    S = x * P_pred * x + R_t         # innovation variance
    K = P_pred * x / (S + 1e-12)     # Kalman gain

    raw_beta = beta[t-1] + K * err

    # hard bounds: beta is not allowed outside this range regardless of
    # what the filter computes (an arbitrary safety clamp, not derived
    # from data - worth revisiting if the true ratio ever moves outside it)
    raw_beta = np.clip(raw_beta, 0.5, 2.0)

    # extra smoothing on top of the Kalman update, to reduce day-to-day jitter
    beta[t] = 0.98 * beta[t-1] + 0.02 * raw_beta

    P = (1 - K * x) * P_pred

df["beta"] = beta
# the traded spread: how far Brent actually is from "beta * WTI"
df["spread"] = df["y"] - df["beta"] * df["x"]

# =====================================================
# 2.5. HYPOTHESIS TEST: is the traded spread actually mean-reverting?
# =====================================================
# The whole strategy assumes the spread is stationary (i.e. it reverts to a
# local mean instead of drifting freely). This is the direct check on the
# actual object being traded, not on some other proxy spread.
adf_result = adfuller(df["spread"].dropna())
adf_pvalue = adf_result[1]
print(f"ADF test on the traded spread: p-value = {adf_pvalue:.4f}")
print(f"Stationary (p<0.05): {adf_pvalue < 0.05}")

# =====================================================
# 3. SIGNAL: Z-SCORE OF THE SPREAD
# =====================================================
window = 30

# mean/std use .shift(1) so today's z-score never uses today's own value -
# avoids look-ahead bias
mean = df["spread"].rolling(window).mean().shift(1)
std = df["spread"].rolling(window).std().shift(1)

df["z"] = (df["spread"] - mean) / (std + 1e-12)

# =====================================================
# 4. REGIME FILTER: only trade when the spread itself is calm
# =====================================================
# Mean-reversion signals are less reliable when the spread is unusually
# volatile (e.g. during a real structural break). This restricts trading to
# periods where recent spread volatility is below its own historical 70th
# percentile. Uses .shift(1) so the regime label is known before the day starts.
ret = df["spread"].diff()
vol = ret.rolling(20).std()
df["mr_regime"] = (vol < vol.rolling(100).quantile(0.7)).shift(1).fillna(0)

# =====================================================
# 5. RISK SCALING: target a constant portfolio volatility
# =====================================================
# Estimates the variance of the hedged Brent-vs-beta*WTI portfolio itself
# (not just each leg individually), then sizes the position so the expected
# annualized volatility stays near a fixed target (10%), shrinking size when
# markets get choppy and growing it when things are calm.
ret_b = np.log(df["Brent"]).diff()
ret_w = np.log(df["WTI"]).diff()

var_b = ret_b.ewm(span=30).var()
var_w = ret_w.ewm(span=30).var()
cov = ret_b.ewm(span=30).cov(ret_w)

portfolio_var = (
        var_b
        + (df["beta"] ** 2) * var_w
        - 2 * df["beta"] * cov
)

portfolio_vol = np.sqrt(np.clip(portfolio_var, 1e-12, None)) * np.sqrt(252)

TARGET_ANNUAL_VOL = 0.10  # arbitrary target: 10% annualized portfolio volatility
df["risk_scale"] = (TARGET_ANNUAL_VOL / (portfolio_vol + 1e-12)).shift(1)
df["risk_scale"] = df["risk_scale"].clip(0.1, 2.0).fillna(1.0)  # cap leverage between 0.1x and 2x

# =====================================================
# 6. ENTRY/EXIT LOGIC WITH HYSTERESIS
# =====================================================
# Different thresholds for entering (1.8) vs exiting (0.3) a position, so the
# strategy doesn't flip in and out every time z hovers near zero.
# pos = +1 means: long the spread (implicitly: long Brent-leg, short beta*WTI-leg)
# pos = -1 means: the opposite. This sign convention only exists implicitly,
# baked into the PnL formula in section 7 - there is no explicit order logic here.
entry, exit = 1.8, 0.3

pos = 0.0
positions = np.zeros(n)

for i in range(n):
    z = df["z"].iloc[i]
    mr = df["mr_regime"].iloc[i]

    if pos == 0:
        if mr:
            if z > entry:
                pos = -1
            elif z < -entry:
                pos = 1
    else:
        if pos == -1 and z < exit:
            pos = 0
        elif pos == 1 and z > -exit:
            pos = 0

    positions[i] = pos

df["position"] = positions * df["risk_scale"]

# =====================================================
# 7. PnL: hedged log-return of the pair, minus trading costs
# =====================================================
dB = np.log(df["Brent"]).diff()
dW = np.log(df["WTI"]).diff()

# use yesterday's position and beta to compute today's return -
# today's own position/beta must never be used, or this leaks future info
pos_lag = df["position"].shift(1)
beta_lag = df["beta"].shift(1)

# normalizes the return by total gross exposure (1 unit WTI leg + beta units
# Brent leg), so a bigger beta doesn't mechanically inflate the return
gross = 1.0 + np.abs(beta_lag)

strategy_log_ret = pos_lag * (dB - beta_lag * dW) / gross

# cost charged whenever position size changes (proxy for bid-ask spread /
# commissions); this is a flat assumption, not derived from real quotes
turnover = df["position"].diff().abs()
cost = 0.0005

df["net_log_ret"] = strategy_log_ret - turnover * cost

# cumulative equity curve, built in log-return space for numerical stability
df["equity"] = np.exp(df["net_log_ret"].fillna(0).cumsum())

# =====================================================
# 8. PERFORMANCE METRICS
# =====================================================
df = df.dropna()

r = np.exp(df["net_log_ret"]) - 1

sharpe = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252)

down = r[r < 0]
sortino = (r.mean() / (down.std() + 1e-12)) * np.sqrt(252)

dd = (df["equity"] / df["equity"].cummax() - 1).min()

print("=== BACKTEST RESULTS ===")
print(f"Sharpe:  {sharpe:.3f}")
print(f"Sortino: {sortino:.3f}")
print(f"Max DD:  {dd*100:.2f}%")

# =====================================================
# 8.5. WALK-FORWARD: is the result stable across sub-periods,
# or is it driven by one lucky window?
# =====================================================
splits = {
    "2019-2021": df["2019":"2021"],
    "2022-2023": df["2022":"2023"],
    "2024-2026": df["2024":"2026"],
}

print("\n=== WALK-FORWARD BY SUB-PERIOD ===")
for name, sub in splits.items():
    if len(sub) < 30:
        print(f"{name}: not enough data ({len(sub)} days)")
        continue
    r_sub = np.exp(sub["net_log_ret"]) - 1
    sharpe_sub = (r_sub.mean() / (r_sub.std() + 1e-12)) * np.sqrt(252)
    dd_sub = (sub["equity"] / sub["equity"].cummax() - 1).min()
    print(f"{name}: Sharpe={sharpe_sub:.3f}, MaxDD={dd_sub*100:.2f}%, N={len(sub)}")

print(f"Worst single day: {r.min()*100:.2f}%")
print(f"Best single day:  {r.max()*100:.2f}%")
print(f"Median leverage (risk_scale): {df['risk_scale'].median():.2f}")
print(f"Max leverage (risk_scale):    {df['risk_scale'].max():.2f}")
print(f"Realized annualized volatility: {r.std()*np.sqrt(252)*100:.2f}%")

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
print(f"1. Traded spread is stationary (ADF p<0.05): {adf_pvalue < 0.05} (p={adf_pvalue:.4f})")
print(f"2. Sharpe stable across sub-periods: see 'WALK-FORWARD' output above")
print(f"   -> compare manually whether the result depends on one specific window")