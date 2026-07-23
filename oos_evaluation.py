"""
Causal out-of-sample evaluation.

Frozen parameters are used throughout the entire evaluation.

Periods:
    2019-2021: development
    2022-2023: validation
    2024-2026: final test

The model is calculated causally:
    - no future data after the evaluated date is used;
    - rolling indicators use only past/current available data;
    - Kalman beta is calculated sequentially from the beginning
      of available history.

Important:
    The final test is not used for parameter tuning.

Results are saved to:
    results/oos_results.csv
    results/oos_daily_returns.csv
"""


from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import spread_model as sm

from config import (
    OOS_PERIODS,
    CONFIG_VERSION,
    FROZEN_DATE,
    get_frozen_config,
    print_frozen_config,
)


RESULTS_DIR = Path("results")


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Download Brent and WTI data.

    Data source:
        CL=F -> WTI
        BZ=F -> Brent

    Important:
        WTI traded below zero in April 2020.

        Since the model uses logarithmic prices, non-positive
        observations cannot be transformed using np.log().

        Therefore those rows are removed explicitly.
    """

    raw = yf.download(
        ["CL=F", "BZ=F"],
        start="2019-01-01",
        progress=False,
        auto_adjust=False,
    )

    if raw.empty:
        raise RuntimeError(
            "Yahoo Finance returned no data."
        )

    # yfinance can return MultiIndex columns:
    #
    #             Close
    #             CL=F   BZ=F
    #
    if isinstance(
            raw.columns,
            pd.MultiIndex,
    ):
        raw = raw.xs(
            "Close",
            axis=1,
            level=0,
        )

    raw = raw.rename(
        columns={
            "CL=F": "WTI",
            "BZ=F": "Brent",
        }
    )

    raw = raw[
        [
            "WTI",
            "Brent",
        ]
    ].dropna()

    # --------------------------------------------------------
    # Handle non-positive prices
    # --------------------------------------------------------

    invalid = (
            raw <= 0
    ).any(
        axis=1
    )

    if invalid.any():

        print(
            f"Removing {int(invalid.sum())} rows "
            "with non-positive WTI/Brent prices."
        )

        raw = raw.loc[
            ~invalid
        ]

    return raw


# ============================================================
# CAUSAL PERIOD PREPARATION
# ============================================================

def prepare_period(
        raw: pd.DataFrame,
        start: str,
        end: str,
        config,
) -> tuple[pd.DataFrame, object, object]:
    """
    Causal state-carrying preparation.

    The model is run from the beginning of the available
    historical data up to `end`.

    Therefore:

        development:
            2019 -> 2021

        validation:
            2019 -> 2023

        final test:
            2019 -> 2026

    The model never sees data after `end`.

    All model calculations are performed before slicing the
    evaluation period:

        prices
          ↓
        log prices
          ↓
        Kalman beta
          ↓
        spread
          ↓
        z-score
          ↓
        regime filter
          ↓
        risk scaling
          ↓
        position state
          ↓
        performance

    Only after all causal calculations are completed do we
    slice [start, end] for reporting.

    This preserves historical information needed by:

        - Kalman filter
        - rolling z-score
        - regime filter
        - volatility scaling
        - position state

    Parameters remain frozen.
    """

    # --------------------------------------------------------
    # 1. Use only information available up to `end`
    # --------------------------------------------------------

    history = raw.loc[
        :end
    ].copy()

    if len(history) < 150:

        raise ValueError(
            f"Not enough history before {end}. "
            f"Only {len(history)} rows available."
        )

    # --------------------------------------------------------
    # 2. Log prices
    # --------------------------------------------------------

    history["x"] = np.log(
        history["WTI"]
    )

    history["y"] = np.log(
        history["Brent"]
    )

    # --------------------------------------------------------
    # 3. Dynamic hedge ratio and spread
    # --------------------------------------------------------

    history, kalman_diag = (
        sm.compute_beta_and_spread(
            history,
            config,
        )
    )

    # --------------------------------------------------------
    # 4. Z-score
    # --------------------------------------------------------

    history = (
        sm.compute_zscore(
            history,
            config,
        )
    )

    # --------------------------------------------------------
    # 5. Regime filter
    # --------------------------------------------------------

    history = (
        sm.compute_regime_filter(
            history,
            config,
        )
    )

    # --------------------------------------------------------
    # 6. Risk scaling
    # --------------------------------------------------------

    history = (
        sm.compute_risk_scale(
            history,
            config,
        )
    )

    # --------------------------------------------------------
    # 7. Position state machine
    # --------------------------------------------------------

    positions = (
        sm.compute_positions(
            history,
            config.entry_threshold,
            config.exit_threshold,
        )
    )

    history["position"] = (
            positions
            * history["risk_scale"]
    )

    # --------------------------------------------------------
    # 8. Calculate performance on full causal history
    #
    # This ensures turnover and position changes are calculated
    # with the complete historical context.
    # --------------------------------------------------------

    full_perf = (
        sm.compute_performance(
            history,
            history["position"],
            config.cost_per_turnover,
        )
    )

    history["net_log_ret"] = (
        full_perf["net_log_ret"]
    )

    # --------------------------------------------------------
    # 9. Slice only the requested evaluation period
    # --------------------------------------------------------

    period = (
        history.loc[
            start:end
        ].copy()
    )

    # --------------------------------------------------------
    # 10. Recalculate reported performance for the period
    # --------------------------------------------------------

    period_perf = (
        sm.compute_performance(
            period,
            period["position"],
            config.cost_per_turnover,
        )
    )

    period["net_log_ret"] = (
        period_perf["net_log_ret"]
    )

    period["equity"] = np.exp(
        period["net_log_ret"]
        .fillna(0)
        .cumsum()
    )

    return (
        period,
        period_perf,
        kalman_diag,
    )


# ============================================================
# EVALUATE ALL PERIODS
# ============================================================

def evaluate_all_periods(
        raw: pd.DataFrame,
        config,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    summary_rows = []

    daily_rows = []

    for (
            period_name,
            (
                    start,
                    end,
            ),
    ) in OOS_PERIODS.items():

        print()

        print(
            "=" * 70
        )

        print(
            period_name
        )

        print(
            f"{start} -> {end}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Prepare period causally
        # ----------------------------------------------------

        (
            period,
            perf,
            kalman_diag,
        ) = prepare_period(
            raw=raw,
            start=start,
            end=end,
            config=config,
        )

        # ----------------------------------------------------
        # Return statistics
        # ----------------------------------------------------

        valid_returns = (
            period[
                "net_log_ret"
            ]
            .dropna()
        )

        if len(
                valid_returns
        ) > 0:

            simple_returns = (
                    np.exp(
                        valid_returns
                    )
                    - 1
            )

            total_return = (
                    np.exp(
                        valid_returns.sum()
                    )
                    - 1
            )

            realized_vol = (
                    simple_returns.std()
                    * np.sqrt(252)
            )

            worst_day = (
                simple_returns.min()
            )

            best_day = (
                simple_returns.max()
            )

        else:

            total_return = np.nan

            realized_vol = np.nan

            worst_day = np.nan

            best_day = np.nan

        # ----------------------------------------------------
        # Beta diagnostics ONLY for evaluation period
        # ----------------------------------------------------

        beta_min = (
            period["beta"].min()
        )

        beta_max = (
            period["beta"].max()
        )

        beta_pct_out_of_band = (
            (
                    (
                            period["beta"]
                            < config.beta_warn_min
                    )
                    |
                    (
                            period["beta"]
                            > config.beta_warn_max
                    )
            )
            .mean()
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Observations: "
            f"{len(period)}"
        )

        print(
            f"Return:      "
            f"{total_return:.2%}"
        )

        print(
            f"Sharpe:      "
            f"{perf['sharpe']:.3f}"
        )

        print(
            f"Sortino:     "
            f"{perf['sortino']:.3f}"
        )

        print(
            f"Max DD:      "
            f"{perf['max_dd']:.2%}"
        )

        print(
            f"Trades:      "
            f"{perf['n_trades']}"
        )

        print(
            f"Realized vol:"
            f"{realized_vol:.2%}"
        )

        print(
            f"Beta range:  "
            f"{beta_min:.4f} -> "
            f"{beta_max:.4f}"
        )

        print(
            f"Beta outside diagnostic band: "
            f"{beta_pct_out_of_band:.2%}"
        )

        # ----------------------------------------------------
        # Summary row
        # ----------------------------------------------------

        summary_rows.append(
            {
                "period": period_name,

                "start": start,

                "end": end,

                "config_version":
                    CONFIG_VERSION,

                "frozen_date":
                    FROZEN_DATE,

                "n_observations":
                    len(period),

                "return":
                    total_return,

                "sharpe":
                    perf["sharpe"],

                "sortino":
                    perf["sortino"],

                "max_dd":
                    perf["max_dd"],

                "n_trades":
                    perf["n_trades"],

                "realized_vol":
                    realized_vol,

                "worst_day":
                    worst_day,

                "best_day":
                    best_day,

                "beta_min":
                    beta_min,

                "beta_max":
                    beta_max,

                "beta_pct_out_of_band":
                    beta_pct_out_of_band,
            }
        )

        # ----------------------------------------------------
        # Daily data
        # ----------------------------------------------------

        daily = period[
            [
                "WTI",
                "Brent",
                "beta",
                "spread",
                "z",
                "mr_regime",
                "risk_scale",
                "position",
                "net_log_ret",
                "equity",
            ]
        ].copy()

        daily["period"] = (
            period_name
        )

        daily["config_version"] = (
            CONFIG_VERSION
        )

        daily["frozen_date"] = (
            FROZEN_DATE
        )

        daily_rows.append(
            daily.reset_index()
        )

    # --------------------------------------------------------
    # Build output DataFrames
    # --------------------------------------------------------

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
    )

    daily_df = (
        pd.concat(
            daily_rows,
            ignore_index=True,
        )
    )

    return (
        summary_df,
        daily_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load frozen configuration
    # --------------------------------------------------------

    config = (
        get_frozen_config()
    )

    print_frozen_config(
        config
    )

    # --------------------------------------------------------
    # Load market data
    # --------------------------------------------------------

    print()

    print(
        "Loading market data..."
    )

    raw = (
        load_data()
    )

    print(
        f"Loaded {len(raw)} total rows."
    )

    # --------------------------------------------------------
    # Evaluate all periods
    # --------------------------------------------------------

    (
        summary_df,
        daily_df,
    ) = evaluate_all_periods(
        raw=raw,
        config=config,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
            RESULTS_DIR
            / "oos_results.csv"
    )

    daily_path = (
            RESULTS_DIR
            / "oos_daily_returns.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    daily_df.to_csv(
        daily_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "FINAL OOS SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary_df[
            [
                "period",
                "return",
                "sharpe",
                "sortino",
                "max_dd",
                "n_trades",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved: "
        f"{summary_path}"
    )

    print(
        f"Saved: "
        f"{daily_path}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Parameters remain frozen."
    )

    print(
        "Do NOT retune the strategy after observing "
        "the 2024-2026 final-test results."
    )


if __name__ == "__main__":

    main()