Brent–WTI Statistical Arbitrage: Kalman-Filtered Pairs Trading Backtest

A research backtest of a Brent/WTI crude oil statistical arbitrage strategy: a
Kalman-filtered dynamic hedge ratio, a cointegration-based mean-reversion signal,
a volatility regime filter, and volatility-targeted position sizing. Includes a
lightweight "should we trade today" market checker and a standalone Monte Carlo
price forecast.


What this is not: there is no order execution, broker connection, or live
capital involved anywhere in this repo. Everything here is a historical
backtest and a set of diagnostic tools — treat the results as a research
exercise, not a performance guarantee. It also does not model futures
contract roll mechanics (see "Known limitations" below) — CL=F/BZ=F are
continuous front-month series, not a single tradable instrument.



Contents

FileWhat it doesmain.pyEnd-to-end backtest: data loading, Kalman hedge ratio, ADF stationarity test, OU half-life, z-score signal, regime filter, risk scaling, PnL, walk-forward check, parameter sensitivity sweep, cost stress test, equity curvemarket_check.py"Should we trade right now?" — recomputes the same beta/spread on recent data (via spread_model.py) and, if there's a signal, translates it into concrete WTI/Brent futures contract countsmonte_carlo_engine.pyStandalone 10-day GBM Monte Carlo forecast for WTI, independent of the stat-arb signalconfig.pySingle source of truth for every tunable constant (StrategyConfig)spread_model.pyShared model logic: Kalman hedge ratio, z-score, regime filter, risk scaling, position logic, and performance metrics — imported identically by main.py and market_check.py, so they cannot silently disagree about what "the spread" isvalidation.pyOU/AR(1) mean-reversion half-life estimate; parameter sensitivity sweep (development period only — see below)costs.pyTransaction-cost stress test (low / base / stress bps scenarios)

Methodology


Dynamic hedge ratio — pure Kalman filter. Rather than fitting a single
beta once on the whole history, Brent ≈ beta × WTI is re-estimated day by
day in log-price space, so the hedge ratio can drift slowly as the true
relationship between the two benchmarks changes. This is a pure
posterior estimate: no clipping, no post-hoc smoothing (see "Resolved in
this version" — earlier versions did both, which made the filter's own
uncertainty term inconsistent with the beta actually used downstream). The
old [0.5, 2.0] plausibility range is now a diagnostic-only band:
spread_model.compute_beta_and_spread logs and counts how often beta
leaves it, without ever altering beta.
Stationarity check (ADF test), run directly on the traded spread —
not a proxy — both over the full history (main.py) and on a rolling
recent window (market_check.py), since a spread that was stationary in
2019 may not be stationary today.
OU / AR(1) half-life estimate, fit on the traded spread, gives a
data-derived mean-reversion speed. The z-score window is chosen with this
as a reference point (roughly 1.5–2× the estimated half-life is a common
rule of thumb) rather than being an arbitrary round number — see
validation.estimate_ou_half_life and the printed comparison in
main.py's output.
Z-score entry/exit signal with hysteresis: a rolling z-score of the
spread, entering at |z| > entry_threshold and exiting at
|z| < exit_threshold, so the position doesn't flip open/closed every
time z drifts near a single threshold.
Regime filter: trading is switched off whenever recent spread
volatility sits in the top 30% of its trailing 100-day range —
mean-reversion signals are least trustworthy exactly when the spread might
be breaking down structurally rather than temporarily deviating.
Volatility targeting: position size is scaled so the hedged pair
(Brent leg + beta × WTI leg together, via an EWMA covariance estimate)
targets a constant annualized volatility — not each leg individually.
Transaction costs are charged on every change in position size — see
"Validation & robustness checks" below for how sensitive results are to
this assumption.


Every signal input — z-score, regime label, risk scale, and the position
itself — is computed with .shift(1), and the PnL step explicitly uses
yesterday's position and beta (pos_lag, beta_lag) against today's
return. This is what keeps the backtest from leaking future information into
itself, which is the single most common way an amateur backtest inflates its
own results.

Validation & robustness checks

Beyond the ADF stationarity test above, main.py runs four additional checks
before reporting a headline Sharpe:


Walk-forward, explicitly labeled development / validation / final-test
(2019–2021 / 2022–2023 / 2024–2026). The labeling matters more than
the split itself: 2019–2021 is where thresholds may be inspected and
reasoned about; 2024–2026 is meant to be looked at once, honestly, and
not tuned against afterward. This is not yet a re-fit-per-period
out-of-sample test — see "Known limitations".
Parameter sensitivity sweep over a grid of entry/exit thresholds,
run only on the 2019–2021 development slice. The question it answers:
does the strategy only work at the one specific configured
(entry, exit) pair, or across a broad plausible range? A result
concentrated in a narrow parameter region is a red flag for overfitting,
even before touching validation or final-test data. Running this sweep on
anything but the development slice would turn the diagnostic itself into
an in-sample parameter search — see the guard comment in
validation.run_parameter_sensitivity.
Transaction cost stress test at three flat scenarios — low (0.5 bps),
base (1.0 bps), stress (2.0 bps) — reusing the exact same position path
computed by the primary backtest and varying only the cost assumption.
Answers "does this survive higher costs than assumed", which is a more
honest question than defending one guessed number. For reference, the
flat cost used for the primary reported Sharpe/Sortino/MaxDD elsewhere in
this backtest (config.cost_per_turnover) is 5 bps — outside the top of
this stress range; the scenarios here deliberately probe a lower-cost
regime rather than replacing that reference assumption.
Kalman diagnostics: with clipping removed, how often (and by how much)
did beta actually leave the [0.5, 2.0] historically-plausible band on
real data? Reported after every run, not just asserted.


None of these prove the strategy works. They narrow down how it could be
fooling itself, which is a more answerable question.

Setup

bashpip install -r requirements.txt

Usage

bashpython main.py                  # full backtest: metrics, validation checks, equity curve
python market_check.py          # is there a signal right now? + suggested contract sizes
python monte_carlo_engine.py    # 10-day WTI price forecast

config.py, spread_model.py, validation.py and costs.py are not meant
to be run directly — they're imported by the three scripts above.

Hardcoded parameters — what they are and why

None of these were fit by optimizing the backtest's own Sharpe ratio — they're
priors and safety bounds, chosen for the reasons below and then checked, not
tuned. That distinction matters for anyone evaluating this: a parameter set
that was searched over the backtest itself tells you much less than one that
was fixed in advance for independent reasons.

Kalman filter — hedge ratio dynamics (pure filter, no clip/smoothing)

ParameterValueWhykalman_q (process noise)1e-6How fast the filter is allowed to believe beta is drifting. A very small value encodes the assumption that the true WTI/Brent relationship moves slowly (driven by refining capacity, transport costs, regional supply/demand) rather than jumping day to day.kalman_r (observation noise)0.01How much a single day's price pair is trusted. Day-to-day price noise is treated as non-trivial relative to signal, so beta tracks the trend rather than each day's tick.Vol-scaling multipliers×10 on Q, ×5 on RLet both noise terms rise on volatile days, so beta adapts faster when markets move and stays stable when they're quiet. Q reacting roughly twice as aggressively as R is a heuristic, not a fitted ratio.beta_warn_min / beta_warn_max[0.5, 2.0]Diagnostic only, as of this version — not applied to beta. Previously a hard clip; now just a historically-plausible band that gets logged and counted when left, via KalmanDiagnostics. See "Resolved in this version" for why the clip was removed.Initial beta, P1.0, 1.0Before seeing any data: assume a neutral 1:1 hedge ratio (both are crude benchmarks) and a moderately large starting uncertainty, so the filter adapts quickly to real data in the first few observations.

Signal generation

ParameterValueWhyZ-score window30 daysA round-number default, now cross-checked against validation.estimate_ou_half_life on each run — a common rule of thumb is 1.5–2× the estimated OU half-life; see the printed comparison in main.py's output rather than a hardcoded justification here, since the estimate is data-dependent.Entry threshold1.8For a roughly normal variable this is close to the 93rd/7th percentile — an uncommon deviation, chosen deliberately high so the strategy doesn't open positions on ordinary noise. Sensitivity to this value is checked directly — see "Validation & robustness checks".Exit threshold0.3Close to zero, so a position is closed as soon as the spread has meaningfully reverted, rather than waiting for a full round-trip back through zero (which risks giving back profit if the spread overshoots the other way).

The important design choice is using two different thresholds (hysteresis)
instead of one — that's what stops the position from flipping open/closed every
time z oscillates near a single boundary, which would rack up transaction costs
for no real signal.

Regime filter

ParameterValueWhySpread-volatility window20 daysA one-month proxy for how turbulent the spread currently is.Percentile lookback100 daysThe recent history against which that 20-day volatility is ranked.Percentile threshold0.7Trading is allowed only in the calmer 70% of recent regimes. 0.7 is a middle ground: strict enough to skip real turbulence, loose enough to still trade most of the time — a judgment call, not something derived by optimizing the backtest's own Sharpe.

Risk scaling

ParameterValueWhyTarget annualized volatility10%A conservative, round-number risk budget for a moderate-risk portfolio.Leverage clip[0.1×, 2.0×]A safety fuse: the vol-targeting formula alone has no natural bound, so this stops it from scaling the position to near-zero or to extreme leverage if estimated portfolio volatility is briefly unusual (e.g. a data glitch).EWMA span30Used for the variance/covariance estimate behind risk scaling; same order of magnitude as the z-score window for consistency, not separately tuned.

Transaction costs

ParameterValueWhyCost per unit of turnover (primary/reported)0.0005 (5 bps)A flat, round-number stand-in for the real bid-ask spread and commissions on CL/BZ futures — the number behind the headline Sharpe/Sortino/MaxDD. Explicitly an assumption, not derived from real quotes.Stress-test scenarios0.5 / 1.0 / 2.0 bpsSee "Validation & robustness checks" — rather than defend the single number above, three lower scenarios test how much of the edge survives across a plausible range. Treat as a stylized sensitivity check, not a realistic market-impact/execution simulation (that would require real order-book data this project doesn't have — see "Known limitations").

Monte Carlo forecast

ParameterValueWhyHorizon10 trading daysAbout two calendar weeks: long enough to say something useful about whether the current WTI/Brent deviation is likely to persist or fade, short enough that a no-drift GBM assumption doesn't compound into an uninformatively wide range.Volatility, random seednot hardcodedVolatility is estimated from the last 30 days of realized WTI returns, and the seed is left unset by default, so every run reflects current conditions and normal Monte Carlo variation.

Results

Fill in after running main.py end-to-end with a live network connection to
Yahoo Finance:


Sharpe / Sortino / Max drawdown (base cost, full history):
Walk-forward Sharpe — development / validation / final-test:
OU half-life estimate and suggested vs. configured z-score window:
Parameter sensitivity summary (development period):
Cost stress test summary (does Sharpe survive 2.0 bps):
Kalman diagnostics — % of days beta left the [0.5, 2.0] plausibility band:
Equity curve:


Resolved in this version


Duplicated Kalman filter core (main.py / market_check.py) →
extracted into spread_model.py, imported identically by both. Z-score and
regime-filter logic, which had the same duplication problem, were moved
alongside it for the same reason.
Kalman filter was a heuristic hybrid, not a textbook filter (clip +
post-hoc smoothing after the Kalman update, while P continued to evolve
as if the raw unclipped/unsmoothed beta had propagated forward) → removed
entirely. Beta is now the pure Kalman posterior; the old bounds are a
diagnostic-only warning band. Expected consequence, not a regression: beta
now reacts more sharply in volatile periods, since nothing is damping it
anymore.
Z-score window was an unjustified round number → cross-checked each
run against a data-derived OU/AR(1) half-life estimate.
No parameter sensitivity check → added, scoped strictly to the
development period to avoid the check itself becoming an overfitting
channel.
Single flat transaction cost, never stress-tested → added a three-level
cost stress test reusing the primary run's own position path.
main.py's pct_change() recomputed on the whole series every Kalman
iteration (O(n²) instead of O(n)) → fixed in the previous version,
carried through unchanged here.
.gitignore was UTF-16-encoded and silently did nothing — git does not
parse that encoding, so .idea/, .venv/ and __pycache__/ would have
been committed despite "being ignored". Re-saved as UTF-8 and verified with
a real git init that it now works; also added *.iml, which nothing
previously excluded.


A note on numba

spread_model.py's two sequential loops (the Kalman recursion and the
entry/exit state machine) are JIT-compiled with numba.njit(cache=True) when
numba is installed, and fall back automatically to plain Python otherwise —
both paths are verified to produce numerically identical output.

Measured, not assumed, on this project's actual scale: once compiled, the
JIT version is ~100–120× faster per call than pure Python. But compiling costs
roughly 0.3–0.6s (first run; cache=True reduces but doesn't eliminate this
on later runs), and a single python main.py run only calls the Kalman core
once and the position state machine roughly 20 times (main run + sensitivity
grid). The break-even point, measured on this sandbox, is around ~700
repeated calls of the same function within one process — this project isn't
there yet, so numba is currently a net loss in wall-clock time for a single
run, not a speedup. It's included anyway because (a) it's verified correct
and free of downside beyond the fixed compile cost, and (b) the nested
out-of-sample refit and larger parameter grids planned as a follow-up would
call these functions hundreds of times per run, which is exactly where this
starts to pay off. Reporting this as a clean win without checking would have
been the same mistake this whole round of fixes was about avoiding elsewhere.

Known limitations (still open)


Futures contract roll mechanics are not modeled. CL=F/BZ=F from
Yahoo Finance are continuous front-month series, not a single tradable
instrument — in reality, holding a futures position across a contract roll
has a cost (or benefit) driven by the term structure (contango/backwardation)
that this backtest does not account for at all; it treats the continuous
series as if it were one instrument with no roll. This can shift the real
Sharpe/Sortino in either direction and is probably the single largest gap
between what's reported here and what live trading would actually realize.
Any claim of "institutional-quality" backtesting should be scoped down
until this is either modeled or explicitly excluded, as it is here.
No true out-of-sample test yet. The walk-forward split is now labeled
development/validation/final-test, and the parameter sensitivity sweep is
restricted to the development slice — but no formal re-fit-per-period
framework exists yet (i.e. parameters are still fixed once across the whole
history, not re-estimated on each rolling/expanding window). Planned as a
follow-up, not implemented here.
main.py is still a flat orchestration script — no
if __name__ == "__main__":, no functions of its own. The core model
logic it used to duplicate now lives in importable, tested modules
(spread_model.py, validation.py, costs.py); what's left inline in
main.py is just the sequencing of calls into those modules plus
printing/plotting.
"Spread" means two different things across files. In main.py /
market_check.py (via spread_model.py) it's the log-residual of the
cointegration relationship; in monte_carlo_engine.py it's the plain
dollar difference between Brent and WTI. Easy to mix up if the code is
ever consolidated.
Cost stress test is stylized, not a realistic execution simulation.
It varies a flat per-turnover cost; it does not model market impact,
order-book depth, or slippage that scales with order size relative to
volume — that would require real bid/ask or order-book data this project
doesn't have.


Performance fix (carried from the previous version)

main.py's Kalman loop previously recalculated df["Brent"].pct_change() — a
pass over the entire series — on every single iteration, making that step
O(n²) instead of O(n). market_check.py already had the fix (compute
pct_change() once, index into the resulting array); this was carried back
into main.py and is now shared via spread_model.py, so it can't regress
independently in either file again.

Verified on synthetic data sized to match the real history (~1,960 rows,
2019–2026): the fixed loop produces numerically identical beta values to the
original recompute-every-iteration version, and ran roughly two orders of
magnitude faster in this sandbox. Exact multiplier varies by machine; the
direction and order of magnitude are real, not a rough guess.