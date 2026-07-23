"""
Single source of truth for the frozen strategy configuration.

STEP 1:
The strategy parameters are frozen before the honest OOS evaluation.

OOS periods:
    2019-2021 -> development
    2022-2023 -> validation
    2024-2026 -> final test

IMPORTANT:
After the final test is run, these parameters must NOT be retuned based
on the final-test results.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple


CONFIG_VERSION = "frozen_step_1_v1"
FROZEN_DATE = "2026-07-22"


OOS_PERIODS: Dict[str, Tuple[str, str]] = {
    "development_2019_2021": ("2019-01-01", "2021-12-31"),
    "validation_2022_2023": ("2022-01-01", "2023-12-31"),
    "final_test_2024_2026": ("2024-01-01", "2026-12-31"),
}


@dataclass(frozen=True)
class StrategyConfig:

    # -------------------------------------------------
    # Kalman filter
    # -------------------------------------------------
    kalman_q: float = 1e-6
    kalman_r: float = 0.01

    kalman_q_vol_multiplier: float = 10.0
    kalman_r_vol_multiplier: float = 5.0

    beta_init: float = 1.0
    p_init: float = 1.0

    # Diagnostic only.
    # These values do NOT clip or modify beta.
    beta_warn_min: float = 0.5
    beta_warn_max: float = 2.0

    # -------------------------------------------------
    # Signal generation
    # -------------------------------------------------
    z_window: int = 30

    entry_threshold: float = 1.8
    exit_threshold: float = 0.3

    # -------------------------------------------------
    # Regime filter
    # -------------------------------------------------
    regime_vol_window: int = 20
    regime_lookback: int = 100
    regime_quantile: float = 0.7

    # -------------------------------------------------
    # Risk scaling
    # -------------------------------------------------
    target_annual_vol: float = 0.10

    risk_scale_min: float = 0.1
    risk_scale_max: float = 2.0

    ewma_span: int = 30

    # -------------------------------------------------
    # Transaction costs
    # -------------------------------------------------
    cost_per_turnover: float = 0.0005

    def to_dict(self) -> dict:
        return asdict(self)


def get_frozen_config() -> StrategyConfig:
    """
    Returns the one frozen configuration used by the OOS evaluation.
    """
    return StrategyConfig()


def print_frozen_config(config: StrategyConfig) -> None:
    print("=" * 70)
    print("FROZEN STRATEGY CONFIGURATION")
    print("=" * 70)
    print(f"Config version: {CONFIG_VERSION}")
    print(f"Frozen date:    {FROZEN_DATE}")
    print()

    for key, value in config.to_dict().items():
        print(f"{key}: {value}")

    print()
    print("OOS PERIODS")
    print("-" * 70)

    for name, (start, end) in OOS_PERIODS.items():
        print(f"{name}: {start} -> {end}")

    print("=" * 70)