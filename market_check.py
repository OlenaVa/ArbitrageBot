import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller


def load_recent_data(days=250):
    """
    Loads enough history to compute a stable Kalman beta and a rolling ADF
    test, without re-running the full 2019-present backtest.
    """
    raw = yf.download(["CL=F", "BZ=F"], period=f"{days}d", progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("Close", axis=1, level=0)

    raw = raw.rename(columns={"CL=F": "WTI", "BZ=F": "Brent"})
    raw = raw.dropna()

    invalid = (raw <= 0).any(axis=1)
    if invalid.any():
        raw = raw[~invalid]

    return raw


def compute_beta_and_spread(df):
    """Same Kalman filter as main.py, so the traded object is identical."""
    n = len(df)
    beta = np.zeros(n)
    beta[0] = 1.0
    P = 1.0
    Q = 1e-6
    R = 0.01

    x_arr = df["x"].values
    y_arr = df["y"].values
    brent_pct = df["Brent"].pct_change().values

    for t in range(1, n):
        x = x_arr[t]
        y = y_arr[t]
        vol = abs(brent_pct[t]) if t > 1 and not np.isnan(brent_pct[t]) else 0.0

        Q_t = Q * (1 + 10 * vol)
        R_t = R * (1 + 5 * vol)
        P_pred = P + Q_t

        err = y - beta[t - 1] * x
        S = x * P_pred * x + R_t
        K = P_pred * x / (S + 1e-12)

        raw_beta = beta[t - 1] + K * err
        raw_beta = np.clip(raw_beta, 0.5, 2.0)
        beta[t] = 0.98 * beta[t - 1] + 0.02 * raw_beta

        P = (1 - K * x) * P_pred

    df["beta"] = beta
    df["spread"] = df["y"] - df["beta"] * df["x"]
    return df


def check_market_health(days=250, z_window=30, rolling_adf_window=90):
    """
    Answers "should we trade right now" - using the SAME beta/spread
    definition as main.py's backtest, on a recent (short) window.
    """
    raw = load_recent_data(days=days)
    if len(raw) < rolling_adf_window + 30:
        print(f"Замало даних: потрібно принаймні {rolling_adf_window + 30} днів")
        return None

    df = raw.copy()
    df["x"] = np.log(df["WTI"])
    df["y"] = np.log(df["Brent"])
    df = compute_beta_and_spread(df)

    # short-horizon z-score, same window the strategy actually trades on
    mean = df["spread"].rolling(z_window).mean().shift(1)
    std = df["spread"].rolling(z_window).std().shift(1)
    df["z"] = (df["spread"] - mean) / (std + 1e-12)

    # regime filter, same construction as main.py
    ret = df["spread"].diff()
    vol = ret.rolling(20).std()
    df["mr_regime"] = (vol < vol.rolling(100).quantile(0.7)).shift(1).fillna(0)

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
        "rolling_adf_pvalue": df["rolling_adf_pvalue"].iloc[-1],
        "locally_stationary": df["rolling_adf_pvalue"].iloc[-1] < 0.05,
    }


def describe_trade_action(health, capital_usd=100_000, target_annual_vol=0.10):
    """
    Translates the abstract position signal (+1/-1/0) into concrete futures
    contracts: how many WTI (CL=F) and Brent (BZ=F) contracts, and which
    side of each. 1 contract = 1000 barrels for both CL and BZ.
    """
    CONTRACT_SIZE = 1000  # barrels per futures contract

    z = health["z"]
    beta = health["beta"]
    wti_price = health["wti"]

    if not health["locally_stationary"] or not health["mr_regime"] or abs(z) < 1.8:
        return None  # no entry signal - nothing to size

    # direction: z > entry means spread (Brent side) is "too expensive"
    # relative to beta*WTI -> short the spread: sell Brent leg, buy WTI leg.
    # z < -entry is the mirror case.
    if z > 0:
        brent_side, wti_side = "SELL (short)", "BUY (long)"
    else:
        brent_side, wti_side = "BUY (long)", "SELL (short)"

    # Position sizing: rough approximation of main.py's risk_scale idea
    # (target vol / portfolio vol), using capital and a fixed fraction since
    # full EWMA portfolio vol isn't recomputed in this lightweight check.
    # Treat this as a starting point, not a precise sizing model.
    notional = capital_usd * target_annual_vol
    wti_contracts = notional / (wti_price * CONTRACT_SIZE)
    brent_contracts = wti_contracts * beta

    if round(wti_contracts) == 0:
        return {"error": f"Розрахований розмір ({wti_contracts:.3f} контракти) менший за 1 -"
                         f" при капіталі ${capital_usd:,} ця стратегія не може бути виконана"
                         f" на цілих контрактах. Потрібен капітал щонайменше "
                         f"${(wti_price * CONTRACT_SIZE / target_annual_vol):,.0f}."}

    return {
        "brent_action": brent_side,
        "brent_contracts": round(brent_contracts),
        "wti_action": wti_side,
        "wti_contracts": round(wti_contracts),
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
        print(f"Дата: {health['date'].date()}")
        print(f"WTI: ${health['wti']:.2f}   Brent: ${health['brent']:.2f}")
        print(f"Бета: {health['beta']:.4f}")
        print(f"Спред (log): {health['spread']:.4f}")
        print(f"Z-score: {health['z']:.2f}")
        print(f"Режим (спокійний ринок): {health['mr_regime']}")
        print(f"Rolling ADF p-value (90d): {health['rolling_adf_pvalue']:.4f}")
        print(f"Локально стаціонарний: {health['locally_stationary']}")

        print("\n=== ВИСНОВОК ===")
        if not health["locally_stationary"]:
            print("Спред НЕ стаціонарний на короткому вікні -"
                  " короткострокова mean-reversion гіпотеза зараз не підтверджена.")
        elif not health["mr_regime"]:
            print("Ринок зараз занадто волатильний (regime filter вимкнений).")
        elif abs(health["z"]) < 1.8:
            print(f"Спред у нормальному діапазоні (z={health['z']:.2f}, поріг входу 1.8) -"
                  " сигналу на вхід немає.")
        else:
            direction = "шорт-спред (short Brent-leg, long beta*WTI-leg)" if health["z"] > 0 \
                else "лонг-спред (long Brent-leg, short beta*WTI-leg)"
            print(f"Умови виконані: локально стаціонарний, спокійний режим, "
                  f"z={health['z']:.2f} за межею входу -> сигнал: {direction}")

            trade = describe_trade_action(health)
            if trade:
                if "error" in trade:
                    print(f"\n=== КОНКРЕТНА ДІЯ ===\n{trade['error']}")
                else:
                    print("\n=== КОНКРЕТНА ДІЯ ===")
                    print(f"Brent (BZ=F): {trade['brent_action']} {trade['brent_contracts']} контрактів")
                    print(f"WTI   (CL=F): {trade['wti_action']} {trade['wti_contracts']} контрактів")