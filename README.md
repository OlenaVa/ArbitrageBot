# Brent–WTI Statistical Arbitrage: Kalman-Filtered Pairs Trading Backtest

A research backtest of a Brent/WTI crude oil statistical arbitrage strategy: a
Kalman-filtered dynamic hedge ratio, a cointegration-based mean-reversion signal,
a volatility regime filter, and volatility-targeted position sizing. Includes a
lightweight "should we trade today" market checker, a standalone Monte Carlo
price forecast, and a frozen-parameter out-of-sample evaluation.

> **What this is not:** there is no order execution, broker connection, or live
> capital involved anywhere in this repo. Everything here is a historical
> backtest and a set of diagnostic tools — treat the results as a research
> exercise, not a performance guarantee. It also does not model futures
> contract roll mechanics (see "Known limitations" below) — `CL=F`/`BZ=F` are
> continuous front-month series, not a single tradable instrument.

## Contents

| File | What it does |
|---|---|
| `main.py` | Full-history backtest: data loading, Kalman hedge ratio, ADF stationarity test, OU half-life, z-score signal, regime filter, risk scaling, PnL, walk-forward check, parameter sensitivity sweep, cost stress test, equity curve |
| `market_check.py` | "Should we trade right now?" — recomputes the same beta/spread on recent data (via `spread_model.py`) and, if there's a signal, translates it into concrete WTI/Brent futures contract counts |
| `monte_carlo_engine.py` | Standalone 10-day GBM Monte Carlo forecast for WTI, independent of the stat-arb signal |
| `config.py` | Single, frozen source of truth for every tunable constant (`StrategyConfig`) and the three OOS period boundaries |
| `spread_model.py` | Shared model logic: Kalman hedge ratio, z-score, regime filter, risk scaling, position logic, and performance metrics — imported identically by every script below, so none of them can silently disagree about what "the spread" is |
| `validation.py` | OU/AR(1) mean-reversion half-life estimate; parameter sensitivity sweep (development period only — see below) |
| `costs.py` | Transaction-cost stress test (low / base / stress bps scenarios) |
| `oos_evaluation.py` | Frozen-parameter evaluation across development / validation / final-test periods. See "OOS evaluation: a bug found and fixed" below before trusting any earlier version of this file's output |
| `parameter_sensitivity.py` | Standalone entry/exit sensitivity sweep on real downloaded data, development period only |

## Methodology

1. **Dynamic hedge ratio — pure Kalman filter.** Rather than fitting a single
   beta once on the whole history, `Brent ≈ beta × WTI` is re-estimated day by
   day in log-price space, so the hedge ratio can drift slowly as the true
   relationship between the two benchmarks changes. This is a **pure**
   posterior estimate: no clipping, no post-hoc smoothing. The old `[0.5, 2.0]`
   plausibility range is now a **diagnostic-only** band: `spread_model.compute_beta_and_spread`
   logs and counts how often beta leaves it, without ever altering beta.
2. **Stationarity check (ADF test)**, run directly on the *traded* spread —
   not a proxy — both over the full history (`main.py`) and on a rolling
   recent window (`market_check.py`), since a spread that was stationary in
   2019 may not be stationary today.
3. **OU / AR(1) half-life estimate**, fit on the traded spread, gives a
   data-derived mean-reversion speed. The z-score window is chosen with this
   as a reference point (roughly 1.5–2× the estimated half-life) rather than
   being an arbitrary round number.
4. **Z-score entry/exit signal with hysteresis**: a rolling z-score of the
   spread, entering at `|z| > entry_threshold` and exiting at
   `|z| < exit_threshold`, so the position doesn't flip open/closed every
   time z drifts near a single threshold.
5. **Regime filter**: trading is switched off whenever recent spread
   volatility sits in the top 30% of its trailing 100-day range.
6. **Volatility targeting**: position size is scaled so the *hedged pair*
   (Brent leg + beta × WTI leg together, via an EWMA covariance estimate)
   targets a constant annualized volatility — not each leg individually.
7. **Transaction costs** are charged on every change in position size.

Every signal input — z-score, regime label, risk scale, and the position
itself — is computed with `.shift(1)`, and the PnL step explicitly uses
**yesterday's** position and beta (`pos_lag`, `beta_lag`) against **today's**
return. This is what keeps the backtest from leaking future information into
itself.

## Validation & robustness checks

- **ADF stationarity test** on the traded spread.
- **OU half-life** estimate, cross-checking the configured z-score window.
- **Frozen-parameter OOS evaluation** across development (2019–2021),
  validation (2022–2023) and final-test (2024–2026) — see `oos_evaluation.py`
  and the dedicated section below.
- **Parameter sensitivity sweep** over entry/exit thresholds, run only on the
  development slice.
- **Transaction cost stress test** at low/base/stress bps scenarios, reusing
  the exact position path from the primary run.
- **Kalman diagnostics**: how often, and by how much, beta left the
  `[0.5, 2.0]` historically-plausible band on real data.

None of these prove the strategy works. They narrow down *how* it could be
fooling itself, which is a more answerable question.

## OOS evaluation: a bug found and fixed

The first version of `oos_evaluation.py` sliced the raw price data into three
independent chunks (development / validation / final-test) **before**
computing the model, and ran the Kalman filter, z-score, regime filter, and
risk scaling fresh on each chunk. This silently reset beta to its initial
guess (1.0) and wiped every rolling window at the start of both the
validation and final-test periods — verified directly on the exported
`oos_daily_returns.csv`: beta reset to exactly `1.0000` on the first day of
each of those periods, and the regime filter showed "too volatile to trade"
on **100% of the first 120 days** of both periods — not because the market
was actually turbulent, but because the 100-day regime lookback simply had no
history yet.

Measured impact, same real data, before vs. after fixing this:

| Period | Reset-per-period (bug) | Continuous, sliced only for reporting (fixed) |
|---|---|---|
| Development 2019–2021 | 1.009 | 1.009 (unaffected — this period starts at the true beginning of history) |
| Validation 2022–2023 | 0.659 | 1.070 |
| Final test 2024–2026 | 1.045 | 1.212 |

Two causal implementations were written and cross-checked against each other
on the same data: (1) compute the model once over the full history and slice
only for reporting, and (2) compute each evaluation period over an expanding
history ending at that period's end date, then slice performance to the
target period — this is what `oos_evaluation.py` currently does. They agree
to three decimal places because every function in `spread_model.py` is
strictly causal (`.shift(1)`, recursive, or backward-looking rolling windows
only) — a model like this needs continuous history to be in the state a real,
continuously-run strategy would actually be in on any given date; only
*performance reporting* should be period-scoped, never the model computation
itself. (Approach (2) does recompute some history redundantly — development
gets recomputed inside validation's expanding window, and again inside
final-test's — harmless, but see "Known limitations".)

**Why this is documented here instead of quietly corrected:** the reset
version wasn't a different, equally-valid test condition — it measured a
different (and unrealistic) strategy that forgets three years of tracked
history on an arbitrary calendar boundary, and it understated validation
Sharpe by 0.41 as a direct result. Calling that "different test conditions"
would be shading the truth; calling it a bug that was caught, quantified, and
fixed is both more accurate and a stronger thing to be able to say out loud.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py                  # full backtest: metrics, validation checks, equity curve
python market_check.py          # is there a signal right now? + suggested contract sizes
python monte_carlo_engine.py    # 10-day WTI price forecast
python oos_evaluation.py        # frozen-parameter OOS evaluation -> results/oos_*.csv
python parameter_sensitivity.py # entry/exit sensitivity on real data (development period)
```

`config.py`, `spread_model.py`, `validation.py` and `costs.py` are not meant
to be run directly — they're imported by the scripts above.

## Hardcoded parameters — what they are and why

None of these were fit by optimizing the backtest's own Sharpe ratio — they're
priors and safety bounds, chosen for the reasons below and then checked, not
tuned.

**Kalman filter — hedge ratio dynamics (pure filter, no clip/smoothing)**

| Parameter | Value | Why |
|---|---|---|
| `kalman_q` (process noise) | `1e-6` | How fast the filter is allowed to believe beta is drifting — a very small value encodes a slowly-moving true relationship. |
| `kalman_r` (observation noise) | `0.01` | How much a single day's price pair is trusted. |
| Vol-scaling multipliers | `×10` on Q, `×5` on R | Let both noise terms rise on volatile days, so beta adapts faster when markets move. |
| `beta_warn_min` / `beta_warn_max` | `[0.5, 2.0]` | Diagnostic only — not applied to beta. Logged and counted via `KalmanDiagnostics` when left. |
| Initial `beta`, `P` | `1.0`, `1.0` | Neutral 1:1 prior before seeing any data. |

**Signal generation**

| Parameter | Value | Why |
|---|---|---|
| Z-score window | `30` days | Cross-checked against `validation.estimate_ou_half_life` each run — a common rule of thumb is 1.5–2× the estimated OU half-life. |
| Entry threshold | `1.8` | Close to the 93rd/7th percentile for a roughly normal variable — an uncommon deviation. Sensitivity checked directly (see below). |
| Exit threshold | `0.3` | Close to zero, so a position closes as soon as the spread has meaningfully reverted. |

**Regime filter**

| Parameter | Value | Why |
|---|---|---|
| Spread-volatility window | `20` days | Proxy for current spread turbulence. |
| Percentile lookback | `100` days | Recent history the 20-day volatility is ranked against. |
| Percentile threshold | `0.7` | Trade only in the calmer 70% of recent regimes. |

**Risk scaling**

| Parameter | Value | Why |
|---|---|---|
| Target annualized volatility | `10%` | Conservative, round-number risk budget. |
| Leverage clip | `[0.1×, 2.0×]` | Safety fuse on the vol-targeting formula. |
| EWMA span | `30` | Same order of magnitude as the z-score window. |

**Transaction costs**

| Parameter | Value | Why |
|---|---|---|
| Cost per unit of turnover (primary/reported) | `0.0005` (5 bps) | Flat stand-in for real bid-ask + commissions — an assumption, not derived from real quotes. |
| Stress-test scenarios | `0.5 / 1.0 / 2.0 bps` | Tests how much of the edge survives across a lower-cost range — a stylized sensitivity check, not a realistic execution simulation. |

## Results

Frozen config version `frozen_step_1_v1`, frozen `2026-07-22`. From
`oos_evaluation.py`'s corrected (causal expanding-history) computation, live
run:

| Period | Return | Sharpe | Sortino | Max DD | Trades |
|---|---|---|---|---|---|
| Development 2019–2021 | 9.77% | 1.009 | 1.198 | -2.61% | 206 |
| Validation 2022–2023 | 7.79% | 1.070 | 1.385 | -3.37% | 187 |
| Final test 2024–2026 | 12.29% | 1.181 | 0.886 | -3.10% | 195 |

Beta stayed inside the `[0.5, 2.0]` diagnostic band on 100% of days in every
period (no warnings raised).

Worth flagging rather than glossing over: final-test Sortino (0.886) is
*lower* than its own Sharpe (1.181), unlike development and validation where
Sortino exceeds Sharpe as usual. That means downside deviation is
proportionally larger relative to mean return in final-test than in the
other two periods — i.e. this period's losing days are relatively more
severe/asymmetric, even though headline risk-adjusted return is still the
best of the three. Not a red flag on its own, but a reason to look at the
drawdown shape in final-test specifically before citing this number without
qualification.

Parameter sensitivity (development period, real data): configured (1.8, 0.3)
Sharpe = 1.009; grid range [0.702, 1.472] across tested entry/exit
combinations, **100% positive**. The best grid point (entry=2.0, exit=0.4,
Sharpe=1.47) is not the configured pair — a mild point in favor of the
configured thresholds not having been quietly cherry-picked after the fact.

_Still to fill in: cost stress test summary; block bootstrap CI on Sharpe
(planned); equity curve screenshot._

## Resolved in this version

- Duplicated Kalman filter core (`main.py` / `market_check.py`) → extracted
  into `spread_model.py`.
- Kalman filter was a heuristic hybrid (clip + post-hoc smoothing,
  inconsistent with its own uncertainty term `P`) → removed; pure Kalman
  posterior, diagnostic-only warning band.
- Z-score window was an unjustified round number → cross-checked against a
  data-derived OU half-life estimate.
- No parameter sensitivity check → added, scoped to the development period.
- Single flat transaction cost, never stress-tested → three-scenario stress
  test added.
- `main.py`'s `pct_change()` recomputed every Kalman iteration (O(n²)) →
  fixed.
- **OOS evaluation reset the model at every period boundary** → fixed; see
  dedicated section above.
- `.gitignore` was UTF-16-encoded and silently did nothing (git doesn't parse
  that encoding) → re-saved as UTF-8, verified with a real `git init`; added
  `*.iml`.

## Known limitations (still open)

- **Futures contract roll mechanics are not modeled.** `CL=F`/`BZ=F` are
  continuous front-month series; this backtest does not account for the
  cost/benefit of rolling a position across contracts. Probably the single
  largest gap between what's reported here and what live trading would
  realize.
- **No re-fit-per-window OOS test yet.** Parameters are fixed once across
  the whole history, not re-estimated on rolling/expanding windows.
- **`main.py` is still a flat orchestration script** — the *model* logic it
  used to duplicate now lives in importable, tested modules; what's left
  inline is sequencing and printing/plotting.
- **"Spread" means two different things across files** — log-residual in
  `main.py`/`market_check.py`, plain dollar difference in
  `monte_carlo_engine.py`.
- **Cost stress test is stylized**, not a realistic execution/market-impact
  simulation (would need real order-book data this project doesn't have).
- **`oos_evaluation.py` recomputes some periods' history redundantly**
  (development is recomputed inside validation's expanding window, and again
  inside final-test's) — harmless (verified numerically identical to a
  compute-once version) but not the most efficient structure; a candidate
  for simplification, not a correctness issue.
- **`parameter_sensitivity.py` still has its own copy of `load_data()`**,
  separate from `oos_evaluation.py`'s — no correctness impact (development
  period starts at the true beginning of history either way), but worth
  consolidating if these scripts are touched again.

## Reproducing this project's history

Git commit history is the source of truth for how this evolved, including
the bug above — there's no `archive/` folder of superseded scripts in this
repo, and there shouldn't be: keeping old, known-incorrect versions sitting
in the working tree invites someone running the wrong file by accident. Old,
non-current results are likewise not committed as parallel CSVs; only the
current `results/oos_results.csv` / `results/oos_daily_returns.csv` are
checked in, and the before/after numbers that matter for the record are the
table above, not a second copy of the (incorrect) data. Commit messages for
methodology-affecting changes state the numeric impact, e.g.: *"Fix: OOS
evaluation reset Kalman/z-score/regime state at each period boundary,
understating validation Sharpe by ~0.41 (see README)."*