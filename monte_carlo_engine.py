import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def run_monte_carlo_analysis(days_ahead=10, simulations=10000, volatility=None, seed=None):
    """
    Runs a Monte Carlo simulation for WTI price 10 days ahead, based on
    current market data. Returns the forecasted low/median/high price range.

    volatility=None -> computed from the last 30 days of real WTI returns
    instead of a hardcoded number.
    seed=None -> every run gives a different result (a fixed seed would
    defeat the purpose of a Monte Carlo simulation).
    """
    # 1. Load recent data. period="3d" can fail on weekends/holidays when
    # markets are closed - "3d"/"30d" always has at least one valid recent row.
    wti_hist = yf.Ticker("CL=F").history(period="30d")['Close']
    brent_hist = yf.Ticker("BZ=F").history(period="3d")['Close']

    current_price = wti_hist.iloc[-1]
    brent = brent_hist.iloc[-1]
    spread = brent - current_price

    # 2. Volatility: estimated from real recent WTI returns rather than a
    # hardcoded constant, so the simulation reflects current market conditions
    if volatility is None:
        log_ret = np.log(wti_hist).diff().dropna()
        volatility = log_ret.std() * np.sqrt(252)

    dt = 1 / 252
    daily_vol = volatility * np.sqrt(dt)

    # 3. Geometric Brownian motion (no drift term - the simulation assumes
    # no expected directional move, only random daily noise)
    if seed is not None:
        np.random.seed(seed)

    price_paths = np.zeros((simulations, days_ahead + 1))
    price_paths[:, 0] = current_price

    for t in range(1, days_ahead + 1):
        shocks = np.random.normal(0, daily_vol, simulations)
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(shocks)

    # 4. Percentiles: the "from - to" range for each day, including day `days_ahead`
    p5 = np.percentile(price_paths, 5, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)

    return {
        "current_price": current_price,
        "brent": brent,
        "spread": spread,
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
              f"current WTI-Brent spread: ${results['spread']:.2f})")
    plt.ylabel("Price (USD/bbl)")
    plt.xlabel("Days Ahead")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    data = run_monte_carlo_analysis(days_ahead=10)

    day = data["days_ahead"]
    print(f"Current WTI: ${data['current_price']:.2f}, Brent: ${data['brent']:.2f}")
    print(f"Volatility used: {data['volatility_used']*100:.1f}%/yr (from last 30 days of real returns)")
    print(f"\n=== WTI PRICE FORECAST, {day} DAYS AHEAD ===")
    print(f"Low  (5th percentile):  ${data['p5'][-1]:.2f}")
    print(f"Mid  (median):          ${data['p50'][-1]:.2f}")
    print(f"High (95th percentile): ${data['p95'][-1]:.2f}")
    print(f"\nRange: ${data['p5'][-1]:.2f} - ${data['p95'][-1]:.2f}")

    plot_analysis(data)