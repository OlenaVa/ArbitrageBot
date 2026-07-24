# Brent–WTI Statistical Arbitrage: Kalman-Filtered Pairs Trading Backtest

A research backtest of a Brent/WTI crude oil statistical arbitrage strategy: a volatility-adaptive Kalman-filtered dynamic hedge ratio, a cointegration-inspired mean-reversion signal, a volatility regime filter, and volatility-targeted position sizing. Includes a lightweight "should we trade today" market checker, a standalone Monte Carlo price forecast, and a frozen-parameter out-of-sample evaluation.

**What this is not:** there is no order execution, broker connection, or live capital involved anywhere in this repo. Everything here is a historical backtest and a set of diagnostic tools — treat the results as a research exercise, not a performance guarantee. `market_check.py`'s contract-count suggestion is a research signal check, not a trade-ready instruction: it does not verify contract specifications, exact multipliers, currency, margin requirements, or data staleness before suggesting a size — see "Known limitations". It also does not model futures contract roll mechanics (see "Known limitations" below) — `CL=F`/`BZ=F` are continuous front-month series, not a single tradable instrument.

## Contents

| File                       | What it does                                                                                                                                                                                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `main.py`                  | Full-history backtest: data loading, Kalman hedge ratio, ADF stationarity test, OU half-life, z-score signal, regime filter, risk scaling, PnL, walk-forward check, parameter sensitivity sweep, cost stress test, roll-drag sensitivity, equity curve                         |
| `market_check.py`          | "Should we trade right now?" — recomputes the same beta/spread on recent data (via `spread_model.py`) and, if there's a signal, translates it into concrete WTI/Brent futures contract counts. A research signal check, not a trade-ready instruction — see "What this is not" |
| `monte_carlo_engine.py`    | Standalone 10-day GBM Monte Carlo forecast for WTI, independent of the stat-arb signal                                                                                                                                                                                         |
| `config.py`                | Single, frozen source of truth for every tunable constant (`StrategyConfig`) and the three OOS period boundaries                                                                                                                                                               |
| `spread_model.py`          | Shared model logic: Kalman hedge ratio, z-score, regime filter, risk scaling, position logic, and performance metrics — imported identically by every script below, so none of them can silently disagree about what "the spread" is                                           |
| `validation.py`            | OU/AR(1) mean-reversion half-life estimate (diagnostic / parameter-justification only — see Methodology); parameter sensitivity sweep (development period only — see below)                                                                                                    |
| `costs.py`                 | Transaction-cost stress test (low / base / stress bps scenarios, base fixed to match `config.cost_per_turnover`)                                                                                                                                                               |
| `roll_costs.py`            | Stylized futures roll-cost drag sensitivity check — applies a configurable annualized bps drag while a position is held and reports Sharpe with/without it. Not a real per-contract roll simulation (`CL=F`/`BZ=F` have no curve data behind them); see "Known limitations"    |
| `oos_evaluation.py`        | Frozen-parameter evaluation across development / validation / final-test periods. See "OOS evaluation: a bug found and fixed" below before trusting any earlier version of this file's output                                                                                  |
| `parameter_sensitivity.py` | Standalone entry/exit sensitivity sweep on real downloaded data, development period only                                                                                                                                                                                       |

## Methodology

* **Dynamic hedge ratio — a volatility-adaptive Kalman filter.** Rather than fitting a single beta once on the whole history, `Brent ≈ beta × WTI` is re-estimated day by day in log-price space, so the hedge ratio can drift slowly as the true relationship between the two benchmarks changes. This is a pure posterior estimate: no clipping, no post-hoc smoothing. Both the process noise (Q) and observation noise (R) are scaled up on volatile days — this is a heuristic adaptive-noise variant, not a textbook constant-Q/R Kalman filter, which is why it's named that way here rather than just "Kalman filter." Raising Q and R together does not, by construction, guarantee beta becomes more responsive on volatile days — the net effect runs through the Kalman gain (`P_pred·x² / (P_pred·x² + R)`), which depends on both terms jointly, not on either one in isolation. Treat "beta adapts faster in volatile regimes" as an observed tendency in this parameter range (see the printed Kalman diagnostics each run), not a property the Q/R formula guarantees on its own. The old `[0.5, 2.0]` plausibility range is now a diagnostic-only band: `spread_model.compute_beta_and_spread` logs and counts how often beta leaves it, without ever altering beta.
* **Stationarity check (ADF test)**, run directly on the traded spread — not a proxy — both over the full history (`main.py`) and on a rolling recent window (`market_check.py`), since a spread that was stationary in 2019 may not be stationary today. This tests stationarity of the dynamically reconstructed (time-varying-beta) spread actually traded, which is a different and looser claim than a classical Engle–Granger cointegration test with a single fixed hedge coefficient over the full sample — the two are not interchangeable evidence.
* **OU / AR(1) half-life estimate**, fit on the traded spread, gives a data-derived mean-reversion speed. The z-score window is chosen with this as a reference point (roughly 1.5–2× the estimated half-life) rather than being an arbitrary round number. This is a diagnostic / parameter-justification check computed on the same data the strategy trades — not an independent out-of-sample validation that mean reversion exists.
* **Z-score entry/exit signal with hysteresis:** a rolling z-score of the spread against its own trailing local mean (not a fixed fundamental value), entering at `|z| > entry_threshold` and exiting at `|z| < exit_threshold`, so the position doesn't flip open/closed every time z drifts near a single threshold.
* **Regime filter:** trading is switched off whenever recent spread volatility sits in the top 30% of its trailing 100-day range — a volatility-gating rule, not a full regime-detection model.
* **Volatility targeting:** position size is scaled so the hedged pair (Brent leg + beta × WTI leg together, via an EWMA covariance estimate) targets a constant annualized volatility — not each leg individually.
* **Transaction costs** are charged on every change in position size.

Every signal input — z-score, regime label, risk scale, and the position itself — is computed with `.shift(1)`, and the PnL step explicitly uses yesterday's position and beta (`pos_lag`, `beta_lag`) against today's return. This is what keeps the backtest from leaking future information into itself. (`risk_scale` is shifted inside `compute_risk_scale`, while `compute_performance` uses `position.shift(1)` and `df["beta"].shift(1)`.)

## Validation & robustness checks

* ADF stationarity test on the traded spread.
* OU half-life estimate, cross-checking the configured z-score window (diagnostic, computed on the same data the strategy trades — see Methodology).
* Frozen-parameter OOS evaluation across development (2019–2021), validation (2022–2023) and final-test (2024–2026 target window — see the note on this label under "Results") — see `oos_evaluation.py` and the dedicated section below.
* Parameter sensitivity sweep over entry/exit thresholds, run only on the development slice. This checks robustness to that specific pair of parameters — it does not sweep the Kalman noise terms, regime thresholds, or risk target, so "robust to parameter choice" should be read as scoped to entry/exit, not the model as a whole.
* Transaction cost stress test at low/base/stress bps scenarios, reusing the exact position path from the primary run.
* Futures roll-drag sensitivity check (stylized annualized bps assumption — see `roll_costs.py`).
* Kalman diagnostics: how often, and by how much, beta left the `[0.5, 2.0]` historically-plausible band on real data.

None of these prove the strategy works. They narrow down how it could be fooling itself, which is a more answerable question.

One thing code alone cannot certify: `config.FROZEN_DATE` records when these parameters were locked, but that timestamp is a discipline aid, not proof of ex-ante blindness — it doesn't by itself establish that the final-test period's results were genuinely unseen before the freeze. That's a claim about research process, not something a config file can enforce or verify on its own.

## OOS evaluation: a bug found and fixed

The first version of `oos_evaluation.py` sliced the raw price data into three independent chunks (development / validation / final-test) before computing the model, and ran the Kalman filter, z-score, regime filter, and risk scaling fresh on each chunk. This silently reset beta to its initial guess (`1.0`) and wiped every rolling window at the start of both the validation and final-test periods — verified directly on the exported `oos_daily_returns.csv`: beta reset to exactly `1.0000` on the first day of each of those periods, and the regime filter showed "too volatile to trade" on 100% of the first 120 days of both periods — not because the market was actually turbulent, but because the 100-day regime lookback simply had no history yet.

Measured impact, same real data, before vs. after fixing this:

| Period                | Reset-per-period (bug) | Continuous, sliced only for reporting (fixed) |
| --------------------- | ---------------------- | --------------------------------------------- |
| Development 2019–2021 | 1.009                  | 1.009                                         |
| Validation 2022–2023  | 0.659                  | 1.070                                         |
| Final test 2024–2026  | 1.045                  | 1.212                                         |

Two causal implementations were written and cross-checked against each other on the same data:

1. Compute the model once over the full history and slice only for reporting.
2. Compute each evaluation period over an expanding history ending at that period's end date, then slice performance to the target period.

This is what `oos_evaluation.py` currently does. They agree to three decimal places because every function in `spread_model.py` is strictly causal (`.shift(1)`, recursive, or backward-looking rolling windows only). A model like this needs continuous history to be in the state a real, continuously-run strategy would actually be in on any given date; only performance reporting should be period-scoped, never the model computation itself.

Approach (2) does recompute some history redundantly — development gets recomputed inside validation's expanding window, and again inside final-test's — harmless, but see "Known limitations".

Why this is documented here instead of quietly corrected: the reset version wasn't a different, equally-valid test condition — it measured a different and unrealistic strategy that forgets three years of tracked history on an arbitrary calendar boundary, and it understated validation Sharpe by approximately `0.41` as a direct result. Calling that "different test conditions" would be shading the truth; calling it a bug that was caught, quantified, and fixed is both more accurate and a stronger thing to be able to say out loud.

A note on the "final test 2024–2026" label, and the two different Sharpe numbers for it in this README (`1.212` above vs. `1.181` in "Results" below): **2024–2026 is a configured target window, not a claim that the period has already finished.** As of any given run, Yahoo Finance can only return data through the latest available trading day. In practice, "final test 2024–2026" means "2024-01-01 through whatever the latest trading day was when `oos_evaluation.py` was run," not a completed three-year window, until `2026-12-31` actually passes.

That is also why the two numbers differ and aren't a typo: development (1.009) and validation (1.070) agree exactly between the two tables because those are closed historical windows that cannot change between runs. Final-test is the one period whose end boundary is "today" — every additional trading day of real data pulled in by a later run can shift it slightly. The two tables above were generated on different days. Treat the "Results" section below as the current number; the `1.212` in the bug-impact table is only meant to demonstrate the size of the fix, not to stand as the latest headline figure — and neither should be read as a final, complete-period result until the window has actually closed.

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

`config.py`, `spread_model.py`, `validation.py`, `costs.py`, and `roll_costs.py` are not meant to be run directly — they're imported by the scripts above.

## Hardcoded parameters — what they are and why

None of these were fit by optimizing the backtest's own Sharpe ratio — they're priors and safety bounds, chosen for the reasons below and then checked, not tuned.

### Kalman filter — hedge ratio dynamics

| Parameter                         | Value             | Why                                                                                                                                                                                                                                                                                                            |
| --------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kalman_q` (process noise)        | `1e-6`            | How fast the filter is allowed to believe beta is drifting — a very small value encodes a slowly moving true relationship.                                                                                                                                                                                     |
| `kalman_r` (observation noise)    | `0.01`            | How much a single day's price pair is trusted.                                                                                                                                                                                                                                                                 |
| Vol-scaling multipliers           | ×10 on Q, ×5 on R | Both noise terms rise together on volatile days. Empirically this tends to let beta move faster in volatile periods in this parameter range, but the exact effect runs through the Kalman gain — which depends on Q and R jointly, not either alone — rather than being guaranteed just by raising both terms. |
| `beta_warn_min` / `beta_warn_max` | `[0.5, 2.0]`      | Diagnostic only — not applied to beta. Logged and counted via `KalmanDiagnostics` when left.                                                                                                                                                                                                                   |
| Initial beta, P                   | `1.0`, `1.0`      | Neutral 1:1 prior before seeing any data.                                                                                                                                                                                                                                                                      |

### Signal generation

| Parameter       | Value   | Why                                                                                                                              |
| --------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Z-score window  | 30 days | Cross-checked against `validation.estimate_ou_half_life` each run — a common rule of thumb is 1.5–2× the estimated OU half-life. |
| Entry threshold | 1.8     | An uncommon deviation for a roughly normal variable. Sensitivity checked directly.                                               |
| Exit threshold  | 0.3     | Close to zero, so a position closes as the spread meaningfully reverts.                                                          |

### Regime filter

| Parameter                | Value    | Why                                                     |
| ------------------------ | -------- | ------------------------------------------------------- |
| Spread-volatility window | 20 days  | Proxy for current spread turbulence.                    |
| Percentile lookback      | 100 days | Recent history the 20-day volatility is ranked against. |
| Percentile threshold     | 0.7      | Trade only in the calmer 70% of recent regimes.         |

### Risk scaling

| Parameter                    | Value          | Why                                            |
| ---------------------------- | -------------- | ---------------------------------------------- |
| Target annualized volatility | 10%            | Conservative, round-number risk budget.        |
| Leverage clip                | `[0.1×, 2.0×]` | Safety fuse on the vol-targeting formula.      |
| EWMA span                    | 30             | Same order of magnitude as the z-score window. |

### Transaction costs

| Parameter                                    | Value                                    | Why                                                                                                                                                                                                                                                                                            |
| -------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cost per unit of turnover (primary/reported) | `0.0005` (5 bps)                         | Flat stand-in for real bid-ask + commissions — an assumption, not derived from real quotes.                                                                                                                                                                                                    |
| Stress-test scenarios                        | `2.5 / 5.0 / 10.0 bps` (low/base/stress) | "Base" intentionally matches the primary 5 bps assumption above, so that row reproduces the primary run's Sharpe as a sanity check. "Stress" is 2× base — a genuine stress case, fixing an earlier version where the labeled stress scenario was actually cheaper than the primary assumption. |

### Roll-drag assumption

| Parameter              | Value     | Why                                                                                                                                                                                                      |
| ---------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `roll_drag_annual_bps` | 25 bps/yr | A stylized order-of-magnitude drag applied while a position is held, standing in for futures roll cost this project has no curve data to compute directly — see `roll_costs.py` and "Known limitations". |

## Results

Frozen config version `frozen_step_1_v1`, frozen `2026-07-22`. From `oos_evaluation.py`'s corrected causal expanding-history computation, live run:

| Period                | Return | Sharpe | Sortino | Max DD | Trades |
| --------------------- | ------ | ------ | ------- | ------ | ------ |
| Development 2019–2021 | 9.77%  | 1.009  | 1.198   | -2.61% | 206    |
| Validation 2022–2023  | 7.79%  | 1.070  | 1.385   | -3.37% | 187    |
| Final test 2024–2026* | 12.29% | 1.181  | 0.886   | -3.10% | 195    |

* `2024–2026` is a configured target window. These figures reflect data through the latest run date, not a completed 2024–2026 window.

Beta stayed inside the `[0.5, 2.0]` diagnostic band on 100% of days in every period (no warnings raised).

Worth flagging rather than glossing over: final-test Sortino (`0.886`) is lower than its own Sharpe (`1.181`), unlike development and validation where Sortino exceeds Sharpe as usual. That means downside deviation is proportionally larger relative to mean return in final-test than in the other two periods — i.e. this period's losing days are relatively more severe/asymmetric, even though headline risk-adjusted return is still the best of the three. Not a red flag on its own, but a reason to look at the drawdown shape in final-test specifically before citing this number without qualification.

**Parameter sensitivity (development period, real data):** configured `(1.8, 0.3)` Sharpe = `1.009`; the tested grid ranged from `0.702` to `1.472`, and all tested combinations produced positive Sharpe. The best grid point (`entry=2.0`, `exit=0.4`, Sharpe=`1.47`) is not the configured pair. This shows that the configured thresholds are not simply the in-sample maximum of this tested entry/exit grid. It does **not**, by itself, prove that the thresholds were chosen without prior exposure to the development results, or that other model parameters were not selected using the data.

Still to fill in: block bootstrap CI on Sharpe (planned); equity curve screenshot.

## Resolved in this version

* Duplicated Kalman filter core (`main.py` / `market_check.py`) → extracted into `spread_model.py`.
* Kalman filter was a heuristic hybrid (clip + post-hoc smoothing, inconsistent with its own uncertainty term `P`) → removed; pure Kalman posterior, diagnostic-only warning band.
* Z-score window was an unjustified round number → cross-checked against a data-derived OU half-life estimate.
* No parameter sensitivity check → added, scoped to the development period.
* Single flat transaction cost, never stress-tested → three-scenario stress test added.
* `main.py`'s `pct_change()` recomputed every Kalman iteration (`O(n²)`) → fixed.
* OOS evaluation reset the model at every period boundary → fixed; see dedicated section above.
* `.gitignore` was UTF-16-encoded and silently did nothing (git doesn't parse that encoding) → re-saved as UTF-8, verified with a real `git init`; added `*.iml`.
* `oos_evaluation.py` / `parameter_sensitivity.py` had inconsistent, heavily fragmented formatting unlike the rest of the codebase → reformatted to match the rest of the codebase; `parameter_sensitivity.py` now imports `load_data` from `oos_evaluation.py` instead of keeping a second copy.
* `costs.py`'s "stress" scenario (`2 bps`) was actually lower than the `5 bps` primary cost assumption → scenarios redefined so `base` matches `config.cost_per_turnover` exactly and `stress` is 2× that.
* "Spread" meant two different things across files (Kalman log-residual vs. plain dollar difference) → `monte_carlo_engine.py`'s field renamed to `brent_wti_diff_usd` to make the distinction explicit instead of documented-but-ambiguous.
* Added `roll_costs.py`: futures roll-drag sensitivity check, previously the single largest unmodeled gap with no mitigation at all.
* Terminology precision pass: Kalman filter explicitly labeled volatility-adaptive rather than implying textbook constant-Q/R behavior; ADF-on-dynamic-spread distinguished from classical fixed-coefficient cointegration testing; OU half-life labeled diagnostic/parameter-justification rather than independent validation; "final test 2024–2026" clarified as a target window, not a completed period, before `2026-12-31`; added a caveat that `FROZEN_DATE` is a discipline aid, not proof of ex-ante blindness; `market_check.py` explicitly flagged as a research signal check, not a trade-ready instruction.

## Known limitations (still open)

* **Futures contract roll mechanics are still not truly modeled.** `roll_costs.py` adds a stylized annualized drag as a sensitivity check, but that is an assumed order-of-magnitude figure, not a real per-contract roll simulation — `CL=F`/`BZ=F` are continuous front-month series with no futures-curve data behind them in this project. Probably still the single largest gap between what's reported here and what live trading would realize.
* **`market_check.py` is a research signal check, not a decision-grade tool.** It does not verify exact contract multipliers/tick sizes, currency, margin requirements, data staleness, or roll status before suggesting a position size. Turning it into something safe to act on would need those checks added first.
* **No re-fit-per-window OOS test yet.** Parameters are fixed once across the whole history, not re-estimated on rolling/expanding windows.
* **`main.py` is still a flat orchestration script.** The model logic it used to duplicate now lives in importable modules; what's left inline is sequencing and printing/plotting.
* **Cost stress test and roll-drag check are both stylized**, not realistic execution/market-impact or futures-curve simulations. They would need real order-book and futures-curve data, which this project does not have.
* **`oos_evaluation.py` recomputes some periods' history redundantly** — development is recomputed inside validation's expanding window, and again inside final-test's — harmless (verified numerically identical to a compute-once version) but not the most efficient structure; a candidate for simplification, not a correctness issue.
* **Ex-ante blindness for the final-test period is a research-process claim, not something this repo's code can prove on its own** — see the `FROZEN_DATE` note under "Validation & robustness checks."

## Reproducing this project's history

Git commit history is the source of truth for how this evolved, including the bug above — there's no `archive/` folder of superseded scripts in this repo, and there shouldn't be: keeping old, known-incorrect versions sitting in the working tree invites someone running the wrong file by accident. Old, non-current results are likewise not committed as parallel CSVs; only the current `results/oos_results.csv` / `results/oos_daily_returns.csv` are checked in, and the before/after numbers that matter for the record are the table above, not a second copy of the incorrect data. Commit messages for methodology-affecting changes state the numeric impact, e.g.:

```text
Fix: OOS evaluation reset Kalman/z-score/regime state at each period boundary, understating validation Sharpe by ~0.41 (see README).
```
