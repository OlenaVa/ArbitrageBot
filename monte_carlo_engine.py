import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def run_monte_carlo_analysis(days_ahead=10, simulations=10000, volatility=0.519):
    """
    Виконує симуляцію Монте-Карло для WTI на основі поточних ринкових даних.
    Повертає прогнозні значення та об'єкт графіка.
    """
    # 1. Завантаження живих даних
    wti_data = yf.Ticker("CL=F").history(period="1d")
    brent_data = yf.Ticker("BZ=F").history(period="1d")

    current_price = wti_data['Close'].iloc[-1]
    brent = brent_data['Close'].iloc[-1]
    spread = brent - current_price

    # 2. Параметри симуляції
    dt = 1/252
    daily_vol = volatility * np.sqrt(dt)

    # 3. Геометричний броунівський рух
    np.random.seed(42)
    price_paths = np.zeros((simulations, days_ahead + 1))
    price_paths[:, 0] = current_price

    for t in range(1, days_ahead + 1):
        shocks = np.random.normal(0, daily_vol, simulations)
        price_paths[:, t] = price_paths[:, t-1] * np.exp(shocks)

    # 4. Розрахунок перцентилів
    p5 = np.percentile(price_paths, 5, axis=0)
    p50 = np.percentile(price_paths, 50, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)

    return {
        "current_price": current_price,
        "brent": brent,
        "spread": spread,
        "p50": p50,
        "p5": p5,
        "p95": p95
    }

def plot_analysis(results):
    """Візуалізація результатів симуляції"""
    plt.figure(figsize=(10, 6))
    days_ahead = len(results["p50"]) - 1

    plt.plot(results["p50"], color='black', label='WTI Forecast (P50)', linewidth=2)
    plt.fill_between(range(days_ahead + 1), results["p5"], results["p95"], color='gray', alpha=0.2, label='WTI Risk Range')
    plt.axhline(y=results["brent"], color='red', linestyle='--', label=f'Brent (${results["brent"]:.2f})')

    plt.title(f"Market Analysis: WTI vs Brent (Spread: ${results['spread']:.2f})")
    plt.ylabel("Price (USD/bbl)")
    plt.xlabel("Days Ahead")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Блок для швидкого тестування самого файлу
    data = run_monte_carlo_analysis()
    print(f"Поточна WTI: ${data['current_price']:.2f}, Brent: ${data['brent']:.2f}")
    print(f"Прогноз WTI (P50) на 10-й день: {data['p50'][-1]:.2f}")
    plot_analysis(data)