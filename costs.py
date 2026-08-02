"""
Transaction-cost sensitivity: two separate analyses.

1. run_cost_stress_test() - varies the single flat cost_per_turnover
   assumption (see config.py) across a small range, on the live position
   path passed in (normally main.py's full-history run).

2. run_bid_ask_layer() - a narrower, ADDITIONAL cost, layered on top of the
   already-committed, already-cost_per_turnover-adjusted OOS numbers in
   results/oos_daily_returns.csv: the cost of actually crossing the
   bid-ask spread on each leg (WTI, Brent) every time the position turns
   over, using each contract's real exchange-set minimum tick as the unit.
   See that function's docstring for what's verified vs. assumed.

The strategy's PnL step charges `cost x turnover` whenever the position
size changes. The primary reported number in main.py (5 bps / 0.0005,
see config.cost_per_turnover) is explicitly a placeholder, not derived
from real CL/BZ bid-ask quotes.

Rather than defending one guessed number, this module re-runs the SAME
already-computed position path (turnover doesn't depend on the cost
assumption - only the final PnL subtraction does) at several cost levels
and reports how much of the strategy's edge survives.

Scenario levels, and why: "base" is set equal to config.cost_per_turnover
(5 bps) so this module's own baseline row reproduces the primary run's
number as a sanity check, rather than silently testing a different range
than the one actually used elsewhere. "low" (half of base) is an
optimistic case - tighter spreads / passive fills. "stress" (2x base) is
the actual stress case this module is named for: costs coming in worse
than assumed, e.g. from wider bid-ask spreads or slippage on size. This
fixes an earlier version of this file where "stress" (2 bps) was
actually LOWER than the 5 bps primary assumption - i.e. testing an
easier scenario while calling it a stress test. See README - still a
stylized stress test, not a realistic execution/market-impact
simulation.
"""
from __future__ import annotations

import pandas as pd

import spread_model

# label -> cost in basis points (1 bp = 0.0001 = 0.01%)
# "base" intentionally matches config.cost_per_turnover (5 bps) - see
# module docstring for why "low"/"stress" are defined relative to it.
COST_SCENARIOS_BPS = {
    "low": 2.5,
    "base": 5.0,
    "stress": 10.0,
}


def run_cost_stress_test(df: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
    """
    df must already contain: Brent, WTI, beta, and `position_col` (the
    final risk-scaled position path from the primary run). Reuses that
    exact position path for every scenario - only the cost assumption
    changes between rows.
    """
    rows = []
    for label, bps in COST_SCENARIOS_BPS.items():
        cost = bps / 10_000.0
        perf = spread_model.compute_performance(df, df[position_col], cost)
        rows.append({
            "scenario": label,
            "cost_bps": bps,
            "sharpe": perf["sharpe"],
            "max_dd_pct": perf["max_dd"] * 100 if not pd.isna(perf["max_dd"]) else float("nan"),
            "n_trades": perf["n_trades"],
        })
    return pd.DataFrame(rows)


def summarize_cost_stress(stress_df: pd.DataFrame, reference_cost_bps: float) -> str:
    """
    One-line summary, plus where the model's own primary assumption
    (config.cost_per_turnover, reported separately in main.py) sits
    relative to this scenario range. Reads scenario values from
    `stress_df` itself rather than hardcoding bps numbers in the message,
    so this text can't silently go stale if COST_SCENARIOS_BPS changes.
    """
    survives_stress = bool((stress_df["sharpe"] > 0).all())

    def _sharpe_for(label: str) -> float:
        row = stress_df[stress_df["scenario"] == label]
        return float(row["sharpe"].iloc[0]) if not row.empty else float("nan")

    def _bps_for(label: str) -> float:
        row = stress_df[stress_df["scenario"] == label]
        return float(row["cost_bps"].iloc[0]) if not row.empty else float("nan")

    base_bps, stress_bps = _bps_for("base"), _bps_for("stress")
    base_sharpe, stress_sharpe = _sharpe_for("base"), _sharpe_for("stress")

    verdict = (
        "Sharpe stays positive across all tested cost levels."
        if survives_stress else
        "Sharpe turns negative at one or more tested cost levels - "
        "the edge is cost-sensitive."
    )

    min_bps, max_bps = stress_df["cost_bps"].min(), stress_df["cost_bps"].max()
    if min_bps <= reference_cost_bps <= max_bps:
        placement = (
            f"matches the 'base' scenario exactly, so that row reproduces the primary run's Sharpe as a sanity check."
            if abs(reference_cost_bps - base_bps) < 1e-9 else
            "falls inside the tested range."
        )
    else:
        placement = "falls outside the tested range - treat these scenarios as relative, not calibrated to it."

    return (
        f"base ({base_bps:.1f} bps) Sharpe={base_sharpe:.3f}, "
        f"stress ({stress_bps:.1f} bps) Sharpe={stress_sharpe:.3f}. "
        f"{verdict} For reference, the flat cost used elsewhere in this "
        f"backtest (config.cost_per_turnover) is {reference_cost_bps:.1f} bps, which {placement}"
    )


# =====================================================================
# Bid-ask spread layer: a different, narrower cost, ADDED ON TOP of the
# already-committed OOS numbers above (which already have
# config.cost_per_turnover subtracted) - not a replacement for them.
#
# Verified, not assumed: both CL (WTI, NYMEX/CME) and BZ (Brent, ICE/CME)
# have a minimum tick of $0.01 per barrel = $10 per contract (1,000
# barrels) - confirmed from each exchange's own contract specification.
#
# Assumed, not verified: how many ticks WIDE the actual bid-ask spread
# typically is. yfinance provides no historical bid/ask data - only
# OHLCV - so this can't be measured from anything already in this repo.
# Both contracts are among the most actively traded futures globally (CL
# alone routinely sees 600k+ contracts/day), which is normally consistent
# with a spread at or near the 1-tick minimum in ordinary conditions -
# but "normally" is not "always" (spreads widen near contract roll,
# around scheduled data releases, and in stressed markets). Rather than
# assert one number, this is tested across a small range (1/2/3 ticks) -
# the same "state the range, don't defend one guess" approach as
# COST_SCENARIOS_BPS above.
#
# The extra daily cost RATE (in return space, same units as cost_bps
# above) on day t is:
#     spread_ticks * BID_ASK_TICK_USD * (1/WTI_t + |beta_t|/Brent_t)
# multiplied by that day's turnover, then subtracted from the
# already-committed net_log_ret. Stated plainly: this does not replicate
# compute_performance's exact gross-exposure normalization (its
# 1 + |beta| denominator) - it is order-of-magnitude correct, not a
# third-decimal-place claim, consistent with everything else in this
# section being explicitly a stylized layer. What a flat bps cost CANNOT
# capture, and this does: a fixed-dollar tick cost is proportionally
# HEAVIER when the price level is low (e.g. April 2020) and lighter when
# it's high - the opposite of how a flat bps assumption behaves.

BID_ASK_TICK_USD = 0.01  # verified: CL and BZ minimum tick, both $0.01/bbl
BID_ASK_SPREAD_TICKS = (1, 2, 3)  # assumed range - see note above


def run_bid_ask_layer(daily: pd.DataFrame) -> pd.DataFrame:
    """
    daily: results/oos_daily_returns.csv, already loaded - needs Date,
    WTI, Brent, beta, position, net_log_ret, period. For each OOS period
    and each tested spread width, layers the stylized extra cost above on
    top of that period's already-committed net_log_ret, then recomputes
    Sharpe/Sortino/return via spread_model.performance_metrics_from_log_returns
    (the same formula results/oos_results.csv's numbers already use).
    """
    rows = []
    for period, g in daily.groupby("period", sort=False):
        g = g.sort_values("Date")
        turnover = g["position"].diff().abs().fillna(0.0)
        for n_ticks in BID_ASK_SPREAD_TICKS:
            extra_rate = n_ticks * BID_ASK_TICK_USD * (1.0 / g["WTI"] + g["beta"].abs() / g["Brent"])
            extra_cost = turnover * extra_rate
            adjusted = (g["net_log_ret"] - extra_cost).dropna()
            metrics = spread_model.performance_metrics_from_log_returns(adjusted.to_numpy())
            rows.append({
                "period": period,
                "spread_ticks": n_ticks,
                "avg_extra_cost_bps_per_day": round(float(extra_cost.mean() * 10_000), 4),
                "sharpe": round(float(metrics["sharpe"]), 6),
                "return": round(float(metrics["return"]), 6),
            })
    return pd.DataFrame(rows)


def summarize_bid_ask_layer(bid_ask_df: pd.DataFrame, oos_summary: pd.DataFrame) -> str:
    """
    One line per period, comparing the already-committed Sharpe (from
    oos_results.csv, indexed by period) against this layer's Sharpe at
    the middle tested width (2 ticks) - the same "here's the delta"
    framing as summarize_cost_stress above.
    """
    lines = []
    for period in bid_ask_df["period"].unique():
        sub = bid_ask_df[bid_ask_df["period"] == period]
        mid = sub[sub["spread_ticks"] == 2].iloc[0]
        base_sharpe = float(oos_summary.loc[period, "sharpe"]) if period in oos_summary.index else float("nan")
        sharpe_at_min_ticks = sub[sub["spread_ticks"] == sub["spread_ticks"].min()].iloc[0]["sharpe"]
        sharpe_at_max_ticks = sub[sub["spread_ticks"] == sub["spread_ticks"].max()].iloc[0]["sharpe"]
        lines.append(
            f"{period}: Sharpe {base_sharpe:.3f} -> {mid['sharpe']:.3f} at 2 ticks "
            f"(range {sharpe_at_min_ticks:.3f} at 1 tick to {sharpe_at_max_ticks:.3f} at 3 ticks), "
            f"avg extra cost {mid['avg_extra_cost_bps_per_day']:.2f} bps/day"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Standalone entry point for the bid-ask layer specifically: unlike
    # run_cost_stress_test above (called from within main.py's live df),
    # this reads the already-committed OOS CSVs directly - no network
    # access, nothing recomputed.
    from pathlib import Path

    daily = pd.read_csv(Path("results/oos_daily_returns.csv"), parse_dates=["Date"])
    oos_summary = pd.read_csv(Path("results/oos_results.csv")).set_index("period")

    bid_ask_df = run_bid_ask_layer(daily)
    bid_ask_df.to_csv(Path("results/bid_ask_layer.csv"), index=False)

    print("Bid-ask spread cost layer (stylized, on top of the committed OOS numbers):\n")
    print(summarize_bid_ask_layer(bid_ask_df, oos_summary))
    print("\nWrote results/bid_ask_layer.csv")