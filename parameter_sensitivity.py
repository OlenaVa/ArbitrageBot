"""
Diagnostic parameter sensitivity analysis.

IMPORTANT:
This is a diagnostic only.

The frozen strategy configuration is NOT changed.

The analysis tests whether the frozen thresholds:
    entry = 1.8
    exit  = 0.3

sit inside a reasonably stable local region of nearby parameters.

The sweep is performed only on the development period:
    2019-01-01 -> 2021-12-31

The 2022-2023 validation period and
2024-2026 final test period are NOT used
for parameter selection.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import spread_model as sm

from config import (
    OOS_PERIODS,
    get_frozen_config,
)


RESULTS_DIR = Path("results")


def load_data() -> pd.DataFrame:
    """
    Load WTI and Brent prices.

    Non-positive prices are removed because
    logarithmic prices are required by the model.
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

    if isinstance(raw.columns, pd.MultiIndex):
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
        ["WTI", "Brent"]
    ].dropna()

    invalid = (
            raw <= 0
    ).any(axis=1)

    if invalid.any():
        print(
            f"Removing {int(invalid.sum())} rows "
            "with non-positive prices."
        )

        raw = raw.loc[
            ~invalid
        ]

    return raw


def prepare_development_data(
        raw: pd.DataFrame,
        config,
) -> tuple[pd.DataFrame, object]:

    start, end = OOS_PERIODS[
        "development_2019_2021"
    ]

    df = raw.loc[
        start:end
    ].copy()

    if len(df) < 150:
        raise ValueError(
            "Development period is too short."
        )

    df["x"] = np.log(
        df["WTI"]
    )

    df["y"] = np.log(
        df["Brent"]
    )

    df, kalman_diag = (
        sm.compute_beta_and_spread(
            df,
            config,
        )
    )

    df = sm.compute_zscore(
        df,
        config,
    )

    df = sm.compute_regime_filter(
        df,
        config,
    )

    df = sm.compute_risk_scale(
        df,
        config,
    )

    return df, kalman_diag


def run_local_sensitivity(
        df_dev: pd.DataFrame,
        config,
) -> pd.DataFrame:

    entry_grid = (
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
        2.1,
        2.2,
    )

    exit_grid = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    )

    rows = []

    for entry in entry_grid:

        for exit_ in exit_grid:

            if exit_ >= entry:
                continue

            positions = (
                sm.compute_positions(
                    df_dev,
                    entry,
                    exit_,
                )
            )

            position = (
                    positions
                    * df_dev[
                        "risk_scale"
                    ].values
            )

            perf = (
                sm.compute_performance(
                    df_dev,
                    position,
                    config.cost_per_turnover,
                )
            )

            rows.append(
                {
                    "entry": entry,
                    "exit": exit_,
                    "sharpe": perf[
                        "sharpe"
                    ],
                    "sortino": perf[
                        "sortino"
                    ],
                    "max_dd": perf[
                        "max_dd"
                    ],
                    "n_trades": perf[
                        "n_trades"
                    ],
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "entry",
                "exit",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def summarize(
        result: pd.DataFrame,
        config,
) -> None:

    valid = result.dropna(
        subset=[
            "sharpe"
        ]
    )

    configured = valid[
        np.isclose(
            valid["entry"],
            config.entry_threshold,
        )
        &
        np.isclose(
            valid["exit"],
            config.exit_threshold,
        )
        ]

    if configured.empty:
        print(
            "Configured pair not found."
        )

        return

    configured_sharpe = (
        configured[
            "sharpe"
        ].iloc[0]
    )

    positive_fraction = (
            valid["sharpe"] > 0
    ).mean()

    print()
    print(
        "=" * 70
    )

    print(
        "LOCAL PARAMETER SENSITIVITY"
    )

    print(
        "=" * 70
    )

    print(
        f"Frozen entry: "
        f"{config.entry_threshold}"
    )

    print(
        f"Frozen exit:  "
        f"{config.exit_threshold}"
    )

    print()

    print(
        f"Frozen pair Sharpe: "
        f"{configured_sharpe:.3f}"
    )

    print(
        f"Sharpe range: "
        f"{valid['sharpe'].min():.3f}"
        f" -> "
        f"{valid['sharpe'].max():.3f}"
    )

    print(
        f"Positive Sharpe combinations: "
        f"{positive_fraction:.1%}"
    )

    print()

    best = valid.sort_values(
        "sharpe",
        ascending=False,
    ).head(10)

    print(
        "TOP 10 COMBINATIONS"
    )

    print(
        best.to_string(
            index=False
        )
    )

    print()

    print(
        "FROZEN CONFIGURATION REMAINS:"
    )

    print(
        f"entry = "
        f"{config.entry_threshold}"
    )

    print(
        f"exit  = "
        f"{config.exit_threshold}"
    )


def main():

    config = (
        get_frozen_config()
    )

    print(
        "Loading market data..."
    )

    raw = load_data()

    print(
        f"Loaded {len(raw)} rows."
    )

    print(
        "Preparing development period..."
    )

    df_dev, kalman_diag = (
        prepare_development_data(
            raw,
            config,
        )
    )

    result = (
        run_local_sensitivity(
            df_dev,
            config,
        )
    )

    summarize(
        result,
        config,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
            RESULTS_DIR
            / "parameter_sensitivity.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print()

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()