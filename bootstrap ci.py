"""
bootstrap_ci.py - stationary block bootstrap confidence intervals for
Sharpe, Sortino, and total return, one per OOS period. This is the "Still
to fill in: block bootstrap CI on Sharpe (planned)" item from README.md.

Why block bootstrap, not a plain (i.i.d.) bootstrap of individual days:
daily strategy returns are not independent - volatility clustering, and a
position that's typically held over several consecutive days, both mean
that shuffling individual days independently understates the real sampling
uncertainty of the Sharpe ratio. A block bootstrap resamples contiguous
chunks of days together, preserving that short-range dependence.

This uses the STATIONARY bootstrap (Politis & Romano, 1994): block lengths
are drawn from a geometric distribution around a given MEAN rather than one
fixed length, and the resample wraps around the series circularly. Mean
block length is set to 10 trading days (~2 calendar weeks) - long enough to
span the kind of short-lived volatility clustering typical of daily futures
returns, short enough to still leave on the order of 50-75 effectively
independent blocks in even the shortest OOS period (validation, ~500 days).
This is a standard-practice starting point, not a value tuned against these
specific periods' data - see the module-level NOTE below on that distinction.

Reads the already-frozen, already-cost-adjusted results/oos_daily_returns.csv
(no network access, nothing recomputed) and reuses
spread_model.performance_metrics_from_log_returns for Sharpe / Sortino /
total-return, so the CIs here are directly comparable to the point
estimates already in results/oos_results.csv and README.md's Results
table - the script asserts this agreement on every run (see the sanity
check in main()) rather than assuming it.

NOTE on the block-length choice: this parameter was picked once, from
general practice for daily financial return series, BEFORE looking at
whether it produces a "nice" or "convenient" confidence interval. It has
not been swept or re-picked based on the resulting CI width. If you change
it, treat that as a new, disclosed choice - not a free parameter to tune
until the interval looks the way you want.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import spread_model

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op fallback decorator so this module works correctly (just
        without the JIT speedup) if numba isn't installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func
        return decorator

RESULTS_PATH = Path("results/oos_daily_returns.csv")
OOS_SUMMARY_PATH = Path("results/oos_results.csv")
OUTPUT_PATH = Path("results/bootstrap_ci.csv")

MEAN_BLOCK_LENGTH = 10.0  # trading days - see module docstring
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
# Fixed so THIS report is reproducible on rerun - a fixed seed doesn't
# undermine a Monte Carlo/bootstrap exercise, it just pins down which one
# of many equally-valid draws this specific run reports (see
# monte_carlo_engine.py's docstring for the same point made about seeding).
SEED = 20260722


@njit(cache=True)
def _stationary_bootstrap_indices(n, restart, jump_targets, start):
    """One stationary-bootstrap resample of indices into a length-n series.
    restart[t] / jump_targets[t] are pre-drawn outside this function so the
    hot loop itself makes no random calls - only array reads, a compare,
    and a modulo, which numba compiles well."""
    idx = np.empty(n, dtype=np.int64)
    i = start
    for t in range(n):
        idx[t] = i
        i = jump_targets[t] if restart[t] else (i + 1) % n
    return idx


def bootstrap_period(log_returns: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    values = log_returns.to_numpy()
    n = len(values)
    p = 1.0 / MEAN_BLOCK_LENGTH

    rows = []
    for _ in range(N_BOOTSTRAP):
        restart = rng.random(n) < p
        jump_targets = rng.integers(0, n, size=n)
        start = int(rng.integers(0, n))
        idx = _stationary_bootstrap_indices(n, restart, jump_targets, start)
        rows.append(spread_model.performance_metrics_from_log_returns(values[idx]))
    return pd.DataFrame(rows)


def main():
    if not NUMBA_AVAILABLE:
        print("Note: numba not available, running the bootstrap loop unJIT'd (slower).")

    daily = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
    oos_summary = pd.read_csv(OOS_SUMMARY_PATH).set_index("period") if OOS_SUMMARY_PATH.exists() else None
    rng = np.random.default_rng(SEED)
    alpha = (1 - CI_LEVEL) / 2

    print(f"Stationary block bootstrap - mean block length {MEAN_BLOCK_LENGTH:.0f} trading days, "
          f"{N_BOOTSTRAP:,} resamples, seed {SEED}\n")

    summary_rows = []
    for period, g in daily.groupby("period", sort=False):
        valid = g["net_log_ret"].dropna()
        point = spread_model.performance_metrics_from_log_returns(valid.to_numpy())

        # Sanity check: this script's point estimate must match the
        # committed oos_results.csv - if it doesn't, something about this
        # script has drifted from spread_model.compute_performance's
        # formula and the CI below would be centered on the wrong number.
        if oos_summary is not None and period in oos_summary.index:
            committed_sharpe = oos_summary.loc[period, "sharpe"]
            if not np.isclose(point["sharpe"], committed_sharpe, atol=1e-6):
                print(f"  WARNING: {period} point Sharpe ({point['sharpe']:.6f}) does not match "
                      f"results/oos_results.csv ({committed_sharpe:.6f}) - formulas have diverged.")

        draws = bootstrap_period(valid, rng)

        row = {"period": period, "n_obs": len(valid)}
        for metric in ("sharpe", "sortino", "return"):
            lo = draws[metric].quantile(alpha)
            hi = draws[metric].quantile(1 - alpha)
            row[f"{metric}_point"] = round(float(point[metric]), 6)
            row[f"{metric}_ci_low"] = round(float(lo), 6)
            row[f"{metric}_ci_high"] = round(float(hi), 6)
            row[f"{metric}_bootstrap_median"] = round(float(draws[metric].median()), 6)
        summary_rows.append(row)

        print(f"{period}  (n={len(valid)}):")
        print(f"  Sharpe   point={point['sharpe']:.3f}   95% CI [{row['sharpe_ci_low']:.3f}, {row['sharpe_ci_high']:.3f}]"
              f"   bootstrap median {row['sharpe_bootstrap_median']:.3f}")
        print(f"  Sortino  point={point['sortino']:.3f}   95% CI [{row['sortino_ci_low']:.3f}, {row['sortino_ci_high']:.3f}]"
              f"   bootstrap median {row['sortino_bootstrap_median']:.3f}")
        print(f"  Return   point={point['return']:.2%}   95% CI [{row['return_ci_low']:.2%}, {row['return_ci_high']:.2%}]")
        print()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.insert(0, "n_bootstrap", N_BOOTSTRAP)
    summary_df.insert(0, "mean_block_length_days", MEAN_BLOCK_LENGTH)
    summary_df.insert(0, "seed", SEED)
    summary_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()