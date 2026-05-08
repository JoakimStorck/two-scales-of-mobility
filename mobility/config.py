"""Configuration for the mobility analysis.

All tunable parameters are collected here so that the notebook stays clean
and so that sensitivity analyses can vary parameters without touching code.
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MobilityConfig:
    """Parameters for the two-scales mobility analysis.

    Attributes
    ----------
    seed : int
        Random seed for reproducibility.
    p_threshold : float
        Minimum row-normalised transition probability to keep an edge.
    top_k : int | None
        If set, keep at most this many strongest destinations per source.
    task_policy : {"all", "core"}
        Whether to use all O*NET tasks or only those flagged as core
        when computing per-occupation task radii.
    centroid_method : {"median_xy"}
        How to aggregate O*NET-detailed occupations into SOC2018 centres.
    """

    # --- Reproducibility ---
    seed: int = 42

    # --- Edge construction ---
    p_threshold: float = 0.0
    top_k: int | None = None

    # --- Task radius construction ---
    task_policy: Literal["all", "core"] = "all"

    # --- SOC aggregation ---
    centroid_method: Literal["median_xy"] = "median_xy"

    # --- Sensitivity grid (P_THRESHOLD, TOP_K) ---
    sensitivity_grid: tuple[tuple[float, int], ...] = field(
        default_factory=lambda: (
            (1e-4, 3),
            (5e-4, 3),
            (1e-3, 3),
            (2e-3, 3),
            (1e-3, 1),
            (1e-3, 5),
            (1e-3, 10),
        )
    )