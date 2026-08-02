"""
Diagnostic parameter sensitivity analysis.

IMPORTANT:
This is a diagnostic only - the frozen strategy configuration is NOT
changed. It tests whether the frozen thresholds (entry=1.8, exit=0.3)
sit inside a reasonably stable local region of nearby parameters.

The sweep is performed only on the development period (2019-01-01 ->
2021-12-31). The 2022-2023 validation period and 2024-2026 final test
period are NOT used for parameter selection.

Reuses `load_data` from oos_evaluation.py rather than keeping a second
copy - both need the same WTI/Brent series, and a second copy risks
silently drifting from the first if one is ever edited alone.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import spread_model as sm
from config import OOS_PERIODS, get_frozen_config
from oos_evaluation import load_data

RESULTS_DIR = Path("results")

ENTRY_GRID = (1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2)
EXIT_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def prepare_development_data(raw: pd.DataFrame, config) -> tuple[pd.DataFrame, object]:
    start, end = OOS_PERIODS["development_2019_2021"]

    # Compute on everything up to 'end' first, then slice to [start, end] -
    # matching oos_evaluation.py's expanding-history pattern, so this stays
    # correct if development ever stops being the first slice of raw (e.g.
    # if earlier history is added later). Today start == raw's own true
    # beginning, so this produces identical output to slicing first - this
    # guards against future drift, it does not change today's numbers.
    history = raw.loc[:end].copy()
    if len(history) < 150:
        raise ValueError("Development period is too short.")

    history["x"] = np.log(history["WTI"])
    history["y"] = np.log(history["Brent"])

    history, kalman_diag = sm.compute_beta_and_spread(history, config)
    history = sm.compute_zscore(history, config)
    history = sm.compute_regime_filter(history, config)
    history = sm.compute_risk_scale(history, config)

    df = history.loc[start:end].copy()
    return df, kalman_diag


def run_local_sensitivity(df_dev: pd.DataFrame, config) -> pd.DataFrame:
    rows = []
    for entry in ENTRY_GRID:
        for exit_ in EXIT_GRID:
            if exit_ >= entry:
                continue

            positions = sm.compute_positions(df_dev, entry, exit_)
            position = positions * df_dev["risk_scale"].values
            perf = sm.compute_performance(df_dev, position, config.cost_per_turnover)

            rows.append({
                "entry": entry, "exit": exit_,
                "sharpe": perf["sharpe"], "sortino": perf["sortino"],
                "max_dd": perf["max_dd"], "n_trades": perf["n_trades"],
            })

    return pd.DataFrame(rows).sort_values(["entry", "exit"]).reset_index(drop=True)


def summarize(result: pd.DataFrame, config) -> None:
    valid = result.dropna(subset=["sharpe"])
    configured = valid[
        np.isclose(valid["entry"], config.entry_threshold) & np.isclose(valid["exit"], config.exit_threshold)
        ]

    if configured.empty:
        print("Configured pair not found.")
        return

    configured_sharpe = configured["sharpe"].iloc[0]
    positive_fraction = (valid["sharpe"] > 0).mean()

    print("\n" + "=" * 70)
    print("LOCAL PARAMETER SENSITIVITY")
    print("=" * 70)
    print(f"Frozen entry: {config.entry_threshold}")
    print(f"Frozen exit:  {config.exit_threshold}\n")
    print(f"Frozen pair Sharpe: {configured_sharpe:.3f}")
    print(f"Sharpe range: {valid['sharpe'].min():.3f} -> {valid['sharpe'].max():.3f}")
    print(f"Positive Sharpe combinations: {positive_fraction:.1%}\n")

    best = valid.sort_values("sharpe", ascending=False).head(10)
    print("TOP 10 COMBINATIONS")
    print(best.to_string(index=False))

    print("\nFROZEN CONFIGURATION REMAINS:")
    print(f"entry = {config.entry_threshold}")
    print(f"exit  = {config.exit_threshold}")


def main():
    config = get_frozen_config()

    print("Loading market data...")
    raw = load_data()
    print(f"Loaded {len(raw)} rows.")

    print("Preparing development period...")
    df_dev, kalman_diag = prepare_development_data(raw, config)

    result = run_local_sensitivity(df_dev, config)
    summarize(result, config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "parameter_sensitivity.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()