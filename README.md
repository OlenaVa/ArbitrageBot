# Brent–WTI Statistical Arbitrage: Kalman-Filtered Pairs Trading Backtest

A research backtest of a Brent/WTI crude oil statistical arbitrage strategy: a
Kalman-filtered dynamic hedge ratio, a cointegration-based mean-reversion signal,
a volatility regime filter, and volatility-targeted position sizing. Includes a
lightweight "should we trade today" market checker and a standalone Monte Carlo
price forecast.

> **What this is not:** there is no order execution, broker connection, or live
> capital involved anywhere in this repo. Everything here is a historical
> backtest and a set of diagnostic tools — treat the results as a research
> exercise, not a performance guarantee.

## Contents

| File | What it does |
|---|---|
| `main.py` | End-to-end backtest: data loading, Kalman hedge ratio, ADF stationarity test, z-score signal, regime filter, risk scaling, PnL, walk-forward check, equity curve |
| `market_check.py` | "Should we trade right now?" — recomputes the same beta/spread on recent data and, if there's a signal, translates it into concrete WTI/Brent futures contract counts |
| `monte_carlo_engine.py` | Standalone 10-day GBM Monte Carlo forecast for WTI, independent of the stat-arb signal |

## Methodology

1. **Dynamic hedge ratio (Kalman filter).** Rather than fitting a single beta once
   on the whole history, `Brent ≈ beta × WTI` is re-estimated day by day, so the
   hedge ratio can drift slowly as the true relationship between the two
   benchmarks changes.
2. **Stationarity check (ADF test)**, run directly on the *traded* spread — not a
   proxy — both over the full history (`main.py`) and on a rolling recent window
   (`market_check.py`), since a spread that was stationary in 2019 may not be
   stationary today.
3. **Z-score entry/exit signal with hysteresis**: a 30-day rolling z-score of the
   spread, entering at `|z| > 1.8` and exiting at `|z| < 0.3`, so the position
   doesn't flip open/closed every time z drifts near a single threshold.
4. **Regime filter**: trading is switched off whenever recent spread volatility
   sits in the top 30% of its trailing 100-day range — mean-reversion signals are
   least trustworthy exactly when the spread might be breaking down structurally
   rather than temporarily deviating.
5. **Volatility targeting**: position size is scaled so the *hedged pair*
   (Brent leg + beta × WTI leg together, via an EWMA covariance estimate) targets
   a constant annualized volatility — not each leg individually.
6. **Transaction costs** are charged on every change in position size.
7. **Walk-forward check**: Sharpe and max drawdown are also reported per
   sub-period (2019–2021 / 2022–2023 / 2024–2026), to see whether the result
   depends on one specific window rather than holding up across regimes.

Every signal input — z-score, regime label, risk scale, and the position itself
— is computed with `.shift(1)`, and the PnL step explicitly uses **yesterday's**
position and beta (`pos_lag`, `beta_lag`) against **today's** return. This is
what keeps the backtest from leaking future information into itself, which is
the single most common way an amateur backtest inflates its own results.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py                  # full backtest, prints metrics, shows equity curve
python market_check.py          # is there a signal right now? + suggested contract sizes
python monte_carlo_engine.py    # 10-day WTI price forecast
```

## Hardcoded parameters — what they are and why

None of these were fit by optimizing the backtest's own Sharpe ratio — they're
priors and safety bounds, chosen for the reasons below and then checked, not
tuned. That distinction matters for anyone evaluating this: a parameter set
that was searched over the backtest itself tells you much less than one that
was fixed in advance for independent reasons.

**Kalman filter — hedge ratio dynamics**

| Parameter | Value | Why |
|---|---|---|
| `Q` (process noise) | `1e-6` | How fast the filter is allowed to believe beta is drifting. A very small value encodes the assumption that the true WTI/Brent relationship moves slowly (driven by refining capacity, transport costs, regional supply/demand) rather than jumping day to day. |
| `R` (observation noise) | `0.01` | How much a single day's price pair is trusted. Day-to-day price noise is treated as non-trivial relative to signal, so beta tracks the trend rather than each day's tick. |
| Vol-scaling multipliers | `×10` on Q, `×5` on R | Let both noise terms rise on volatile days, so beta adapts faster when markets move and stays stable when they're quiet. Q reacting roughly twice as aggressively as R is a heuristic, not a fitted ratio. |
| Beta clip | `[0.5, 2.0]` | A hard safety bound, independent of what the filter computes — already flagged in the code as arbitrary. Stops a bad tick or filter glitch from pushing the hedge ratio somewhere nonsensical. Worth revisiting only if the true ratio ever structurally moves outside this band. |
| Smoothing weights | `0.98` / `0.02` | An extra exponential smoothing pass on top of the Kalman update, to damp day-to-day jitter in the *traded* beta. Heavier weight on yesterday's value means beta moves slowly; a convenience choice, not fit to minimize a specific error metric. |
| Initial `beta`, `P` | `1.0`, `1.0` | Before seeing any data: assume a neutral 1:1 hedge ratio (both are crude benchmarks) and a moderately large starting uncertainty, so the filter adapts quickly to real data in the first few observations before settling into the Q/R-driven steady state. |

**Signal generation**

| Parameter | Value | Why |
|---|---|---|
| Z-score window | `30` days | About six trading weeks — long enough to estimate a stable mean/std for the spread, short enough to stay responsive as the relationship drifts. |
| Entry threshold | `1.8` | For a roughly normal variable this is close to the 93rd/7th percentile — an uncommon deviation, chosen deliberately high so the strategy doesn't open positions on ordinary noise. |
| Exit threshold | `0.3` | Close to zero, so a position is closed as soon as the spread has meaningfully reverted, rather than waiting for a full round-trip back through zero (which risks giving back profit if the spread overshoots the other way). |

The important design choice is using *two* different thresholds (hysteresis)
instead of one — that's what stops the position from flipping open/closed every
time z oscillates near a single boundary, which would rack up transaction costs
for no real signal. `1.8` / `0.3` themselves are a reasonable, conventional
choice for this kind of z-score strategy, not a fitted optimum.

**Regime filter**

| Parameter | Value | Why |
|---|---|---|
| Spread-volatility window | `20` days | A one-month proxy for how turbulent the spread currently is. |
| Percentile lookback | `100` days | The recent history against which that 20-day volatility is ranked. |
| Percentile threshold | `0.7` | Trading is allowed only in the calmer 70% of recent regimes. `0.7` is a middle ground: strict enough to skip real turbulence, loose enough to still trade most of the time — a judgment call, not something derived by optimizing the backtest's own Sharpe. |

**Risk scaling**

| Parameter | Value | Why |
|---|---|---|
| Target annualized volatility | `10%` | A conservative, round-number risk budget for a moderate-risk portfolio. Flagged in the code as an arbitrary target rather than something derived from a specific capital or risk-appetite constraint. |
| Leverage clip | `[0.1×, 2.0×]` | A safety fuse: the vol-targeting formula alone has no natural bound, so this stops it from scaling the position to near-zero or to extreme leverage if estimated portfolio volatility is briefly unusual (e.g. a data glitch). |
| EWMA span | `30` | Used for the variance/covariance estimate behind risk scaling; same order of magnitude as the z-score window for consistency, not separately tuned. |

**Transaction costs**

| Parameter | Value | Why |
|---|---|---|
| Cost per unit of turnover | `0.0005` (5 bps) | A flat, round-number stand-in for the real bid-ask spread and commissions on CL/BZ futures. Explicitly an assumption, not derived from real quotes — a live version of this strategy should replace it with the venue's actual cost structure. |

**Monte Carlo forecast**

| Parameter | Value | Why |
|---|---|---|
| Horizon | `10` trading days | About two calendar weeks: long enough to say something useful about whether the current WTI/Brent deviation is likely to persist or fade, short enough that a no-drift GBM assumption doesn't compound into an uninformatively wide range. 1 day would be mostly noise; 90 days would be too wide to act on and a worse fit for the no-drift assumption. |
| Volatility, random seed | *not* hardcoded | Volatility is estimated from the last 30 days of realized WTI returns, and the seed is left unset by default, so every run reflects current conditions and normal Monte Carlo variation. Listed here for contrast with the horizon above, which is fixed. |

## Results

_Fill in after running `main.py` end-to-end with a live network connection to
Yahoo Finance:_

- Sharpe:
- Sortino:
- Max drawdown:
- Walk-forward Sharpe by sub-period:
- Equity curve:

## Known limitations

- **No true out-of-sample test.** Every constant above (window sizes, thresholds,
  target vol, regime quantile, Q/R) was chosen once and checked in-sample. The
  walk-forward split in `main.py` checks *stability across periods*, not a
  train/test split with refit parameters — worth being upfront about this if asked.
- **The Kalman filter is technically a heuristic hybrid, not a textbook Kalman
  filter.** After the standard update, beta is clipped and smoothed, but the
  uncertainty term `P` is updated as if the raw (unclipped, unsmoothed) beta is
  what continues forward. Common in practice, but more accurate to describe as
  "Kalman-based estimate with heuristic smoothing" than a pure Kalman filter.
- **The Kalman filter core is duplicated** between `main.py` and
  `market_check.py` (identical Q/R/clip/smoothing). Fine today since both are
  kept in sync by hand, but a natural next step is to extract it into a shared
  module (e.g. `spread_model.py`) so the two can't silently drift apart.
- **`main.py` is a flat script**, unlike the function-based `market_check.py` —
  no `if __name__ == "__main__":`, no functions. Not a problem for a personal
  backtest, but worth restructuring if this becomes a shared/imported module.
- **"Spread" means two different things across files.** In `main.py` and
  `market_check.py` it's the log-residual of the cointegration relationship;
  in `monte_carlo_engine.py` it's the plain dollar difference between Brent and
  WTI. Easy to mix up if the code is ever consolidated.

## Performance fix in this version

`main.py`'s Kalman loop previously recalculated `df["Brent"].pct_change()` — a
pass over the *entire* series — on every single iteration, making that step
`O(n²)` instead of `O(n)`. `market_check.py` already had the fix (compute
`pct_change()` once, index into the resulting array); this version carries that
same fix back into `main.py`.

Verified on synthetic data sized to match the real history (1,966 rows, ≈
2019–2026): the new loop produces **numerically identical** beta values to the
original (`max abs difference: 0.0`), and ran **~39× faster** in this sandbox
(0.40s → 0.01s). Exact multiplier will vary by machine, but the direction and
order of magnitude are real, not a rough guess.
