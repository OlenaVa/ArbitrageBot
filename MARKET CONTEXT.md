# Market Context for the Brent–WTI Backtest

The rest of this repository is a statistical research framework: cointegration testing, a Kalman-filtered hedge ratio, a mean-reversion signal, and an out-of-sample evaluation of whether that signal survives realistic costs. None of that explains *why* Brent and WTI move the way they do relative to each other. This document is the economic context the statistics sit inside — read together with `README.md`, not instead of it.

## Two different pricing systems, one crude quality

Brent and WTI are both light sweet crude benchmarks, which is why they track each other closely enough for a mean-reversion strategy to be worth testing at all. But they price different things. Brent reflects globally traded, seaborne crude — supply/demand balances, OPEC+ policy, international refinery demand, and freight and geopolitical conditions affecting seaborne flows. WTI is priced for delivery at Cushing, Oklahoma — an inland hub more directly exposed to US production growth, regional storage levels, and pipeline takeaway capacity, even as US export growth has increasingly linked it to the global market.

A move in the spread is often less about oil quality and more about how well-connected the US inland market is to the global seaborne one at that moment. A narrowing spread can reflect improving connectivity; a widening one can reflect a temporary bottleneck, or a genuine shift in how the two markets relate.

## What actually moves it

None of the following are inputs to the statistical model in this repo — the regime filter, for example, is a purely statistical volatility check (`spread_model.compute_regime_filter`), not a fundamentals filter. They're the economic backdrop a statistical signal should be read against:

* Cushing inventory levels
* pipeline utilization and takeaway capacity out of the Permian and other inland basins
* US crude export infrastructure and capacity
* freight economics for seaborne grades
* refinery maintenance seasons and regional demand
* geopolitical supply disruptions affecting seaborne flows specifically
* shifts in global production balances (OPEC+ decisions, non-OPEC supply growth)

## Two episodes, for concreteness

**April 2020.** The large spike visible in `results/charts/spread_zscore.png` around April 2020 is WTI's brief move to a negative price: Cushing storage was running close to full as COVID-era demand collapsed, and holders of the expiring futures contract paid to avoid taking physical delivery they had nowhere to store. This is a Cushing-storage event specific to WTI's inland delivery point — Brent, priced on seaborne crude with no equivalent storage constraint, didn't see the same dislocation. `main.py` explicitly drops non-positive prices for exactly this reason (see "Resolved in this version" in `README.md`).

**2022.** Russia's invasion of Ukraine and the resulting disruption to Russian seaborne crude exports widened Brent's premium to WTI for a period — a geopolitical, seaborne-specific shock, the opposite kind of driver from April 2020's inland-storage event. Directionally well documented; treat any precise number here as something to verify against a primary source (EIA, ICE) before citing it, since this document doesn't.

## Why a dynamic hedge ratio, economically

`README.md`'s Methodology section explains the Kalman filter mechanically. The economic case for letting the hedge ratio move at all: a fixed beta assumes the relationship between the two benchmarks is constant, which is hard to justify when pipeline capacity, export infrastructure, and storage conditions are all things that change over the multi-year horizon this backtest covers. The filter doesn't know *why* the ratio is moving — it just tracks that it is. Letting it adapt is a way of not assuming away exactly the kind of change described above, not a claim that the filter understands the cause.

## Reading a signal in context

A statistically significant z-score means the spread's recent behavior differs from its own trailing history — nothing more. What that's worth depends on the environment it shows up in:

| Environment                              | How to weigh the signal                                              |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| Stable logistics, balanced inventories    | Historical relationship more likely still informative                 |
| Elevated Cushing storage                  | Reversion may be slower than the historical average implies           |
| Pipeline or takeaway constraints          | The equilibrium relationship may have shifted, temporarily or not     |
| Export infrastructure expanding           | The long-run relationship may be gradually moving, not just noisy     |
| Freight disruption                        | Physical arbitrage gets more expensive; the statistical signal alone is less reliable |
| Major geopolitical supply shock           | Physical risk premia can dominate the historical statistical relationship entirely |

## Scope

This is a statistical robustness study — does a Brent–WTI mean-reversion signal survive realistic transaction costs, roll drag, and out-of-sample testing — not a crude-oil price forecast, not a complete fundamental model, and not a substitute for checking the conditions above before acting on a signal. See `README.md`'s "What this is not" and "Known limitations" for the code-level version of the same caution.

Statistical arbitrage assumes prices eventually reflect an underlying equilibrium. Commodity markets are a reminder that the equilibrium itself can move — infrastructure, regulation, logistics, and geopolitics continuously redefine what "fair value" even means. The aim here isn't to discover a fixed statistical relationship, but to see how a statistical signal behaves while the economic structure generating it keeps changing.