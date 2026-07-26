import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def run_monte_carlo_analysis(days_ahead=10, simulations=10000, volatility=None, seed=None):
    """
    Runs a Monte Carlo simulation for WTI price 10 days ahead, based on
    current market data. Returns the forecasted low/median/high price range.

    Naming note: this module reports `brent_wti_diff_usd`, the plain
    dollar difference between the two benchmarks on the latest available
    day. This is deliberately NOT called "spread" here - that word is
    reserved elsewhere in this repo (main.py, market_check.py,
    spread_model.py) for the Kalman-hedge-ratio log-residual
    (y - beta*x), a different quantity computed a different way. Reusing
    "spread" for this plain difference was a past source of confusion
    when reading the two modules side by side; the field is renamed to
    make that distinction explicit rather than documented-but-ambiguous.

    volatility=None -> computed from the last 30 days of real WTI returns
    instead of a hardcoded number.
    seed=None -> every run gives a different result, since no seed is set.
    Pass an explicit seed for a reproducible run (e.g. a specific dated
    report, or a test) - reproducibility and Monte Carlo sampling aren't
    in tension, they're just two different use cases.
    """
    # 1. Load recent data for both tickers from ONE aligned download, so
    # "the latest price" for WTI and Brent always refers to the SAME
    # calendar day. Fetching them separately (e.g. two different period=
    # windows, one call per ticker) can silently compare two different
    # days if the exchanges' calendars/data feeds ever diverge - a
    # holiday observed by one but not the other, or a data delay on just
    # one ticker. dropna() keeps only rows where both are available.
    raw = yf.download(["CL=F", "BZ=F"], period="30d", progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("Close", axis=1, level=0)
    raw = raw.rename(columns={"CL=F": "WTI", "BZ=F": "Brent"}).dropna()

    wti_hist = raw["WTI"]
    current_price = wti_hist.iloc[-1]
    brent = raw["Brent"].iloc[-1]
    brent_wti_diff_usd = brent - current_price

    # 2. Volatility: estimated from real recent WTI returns rather than a
    # hardcoded constant, so the simulation reflects current market conditions
    if volatility is None:
        log_ret = np.log(wti_hist).diff().dropna()
        volatility = log_ret.std() * np.sqrt(252)

    dt = 1 / 252
    daily_vol = volatility * np.sqrt(dt)

    # 3. Geometric Brownian motion (no drift term - the simulation assumes
    # no expected directional move, only random daily noise). A local
    # Generator avoids mutating NumPy's global random state - default_rng
    # already behaves correctly for seed=None (fresh, unseeded draws).
    rng = np.random.default_rng(seed)

    price_paths = np.zeros((simulations, days_ahead + 1))
    price_paths[:, 0] = current_price

    for t in range(1, days_ahead + 1):
        shocks = rng.normal(0, daily_vol, simulations)
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(shocks)

    # 4. Percentiles: the "from - to" range for each day, including day `days_ahead`
    p5 = np.percentile(price_paths, 5, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)

    return {
        "current_price": current_price,
        "brent": brent,
        "brent_wti_diff_usd": brent_wti_diff_usd,
        "volatility_used": volatility,
        "days_ahead": days_ahead,
        "p50": p50,
        "p5": p5,
        "p95": p95,
    }


def plot_analysis(results):
    """Visualizes the simulation: median forecast plus 5th-95th percentile range."""
    plt.figure(figsize=(10, 6))
    days_ahead = len(results["p50"]) - 1

    plt.plot(results["p50"], color='black', label='WTI Forecast (P50)', linewidth=2)
    plt.fill_between(range(days_ahead + 1), results["p5"], results["p95"],
                     color='gray', alpha=0.2, label='WTI Risk Range (5th-95th pct)')
    plt.axhline(y=results["brent"], color='red', linestyle='--',
                label=f'Brent (${results["brent"]:.2f})')

    plt.title(f"WTI {days_ahead}-Day Forecast (vol: {results['volatility_used']*100:.1f}%/yr, "
              f"current WTI-Brent $ difference: ${results['brent_wti_diff_usd']:.2f})")
    plt.ylabel("Price (USD/bbl)")
    plt.xlabel("Days Ahead")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    data = run_monte_carlo_analysis(days_ahead=10)

    day = data["days_ahead"]
    print(f"Current WTI: ${data['current_price']:.2f}, Brent: ${data['brent']:.2f}")
    print(f"WTI-Brent $ difference (NOT the Kalman-hedge spread used in main.py): "
          f"${data['brent_wti_diff_usd']:.2f}")
    print(f"Volatility used: {data['volatility_used']*100:.1f}%/yr (from last 30 days of real returns)")
    print(f"\n=== WTI PRICE FORECAST, {day} DAYS AHEAD ===")
    print(f"Low  (5th percentile):  ${data['p5'][-1]:.2f}")
    print(f"Mid  (median):          ${data['p50'][-1]:.2f}")
    print(f"High (95th percentile): ${data['p95'][-1]:.2f}")
    print(f"\nRange: ${data['p5'][-1]:.2f} - ${data['p95'][-1]:.2f}")

    plot_analysis(data)