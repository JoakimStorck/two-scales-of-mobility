"""Geometric primitives and SOC2018 aggregation.

Two coordinate systems are used:

* Polar (xi, chi): xi in radians, chi as scaled radial coordinate.
* Cartesian (x, y): standard Euclidean, derived from polar.

The unit of analysis for mobility is SOC2018, since IPUMS-CPS reports
occupations at this granularity. O*NET-detailed codes (e.g. 15-1252.00)
are aggregated to SOC2018 (e.g. 15-1252) by taking the robust median of
their (x, y) positions.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Code normalisation
# ---------------------------------------------------------------------------

def norm_onet_full(code: str | float) -> str | float:
    """Strip whitespace and uppercase an O*NET-SOC code; keep decimal suffix.

    Examples
    --------
    >>> norm_onet_full(" 15-1252.00 ")
    '15-1252.00'
    >>> norm_onet_full(np.nan)
    nan
    """
    if pd.isna(code):
        return np.nan
    return str(code).strip().upper().split()[0]


def norm_soc2018(code: str | float) -> str:
    """Coerce a SOC code to canonical 'XX-XXXX' form.

    Accepts inputs like '35-9031', '359031', or '35-9031.00'.
    """
    digits = re.sub(r"\D", "", str(code))[:6].rjust(6, "0")
    return f"{digits[:2]}-{digits[2:]}"


def onet_full_to_soc2018(onet_full: str | float) -> str | float:
    """Convert an O*NET-detailed code to its SOC2018 parent.

    Examples
    --------
    >>> onet_full_to_soc2018('35-9031.00')
    '35-9031'
    """
    if onet_full is None or pd.isna(onet_full):
        return np.nan
    head = str(onet_full).strip().split()[0].split(".")[0]
    return norm_soc2018(head)


# ---------------------------------------------------------------------------
# Polar / Cartesian conversion
# ---------------------------------------------------------------------------

def polar_to_xy(xi: np.ndarray, chi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert polar (xi in radians, chi) to Cartesian (x, y)."""
    xi = np.asarray(xi, dtype=float)
    chi = np.asarray(chi, dtype=float)
    return chi * np.cos(xi), chi * np.sin(xi)


def xy_to_polar(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian (x, y) to polar (xi in radians, chi).

    xi is wrapped to [0, 2*pi).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    chi = np.hypot(x, y)
    xi = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
    return xi, chi


# ---------------------------------------------------------------------------
# Loading polar exports from the first paper
# ---------------------------------------------------------------------------

def load_occupations(path) -> pd.DataFrame:
    """Load the occupation polar export and add Cartesian coordinates.

    The input file is expected to have columns ``onet_code``, ``xi``, ``chi``.

    Returns a DataFrame with columns
    ``[onet_full, xi, chi, x, y, soc2018]`` plus any optional descriptive
    columns (``Title``, ``Job Family``, ``sector_zone``) that were present.
    """
    df = pd.read_csv(path)
    required = {"onet_code", "xi", "chi"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"occupations file missing columns: {missing}")

    keep = ["onet_code", "xi", "chi"]
    for col in ("Title", "Job Family", "sector_zone"):
        if col in df.columns:
            keep.append(col)
    df = df[keep].copy()

    df["onet_code"] = df["onet_code"].astype("string").str.strip()
    df["onet_full"] = df["onet_code"].map(norm_onet_full)
    df = df.dropna(subset=["onet_full", "xi", "chi"]).drop_duplicates("onet_full")

    x, y = polar_to_xy(df["xi"].to_numpy(), df["chi"].to_numpy())
    df["x"] = x
    df["y"] = y
    df["soc2018"] = df["onet_full"].map(onet_full_to_soc2018)
    df = df.dropna(subset=["soc2018"]).reset_index(drop=True)
    return df


def load_tasks(path) -> pd.DataFrame:
    """Load the task polar export and add Cartesian coordinates.

    Returns a DataFrame with columns ``[onet_full, xi, chi, x, y, soc2018]``
    plus optional ``Task ID``, ``Task``, ``is_core``.
    """
    df = pd.read_csv(path)
    required = {"onet_code", "xi", "chi"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"tasks file missing columns: {missing}")

    keep = ["onet_code", "xi", "chi"]
    for col in ("Task ID", "Task", "is_core"):
        if col in df.columns:
            keep.append(col)
    df = df[keep].copy()

    df["onet_code"] = df["onet_code"].astype("string").str.strip()
    df["onet_full"] = df["onet_code"].map(norm_onet_full)

    if "Task ID" in df.columns:
        df = df.drop_duplicates(subset=["onet_full", "Task ID"])
    else:
        df = df.drop_duplicates(subset=["onet_full", "xi", "chi"])

    df = df.dropna(subset=["onet_full", "xi", "chi"]).reset_index(drop=True)

    x, y = polar_to_xy(df["xi"].to_numpy(), df["chi"].to_numpy())
    df["x"] = x
    df["y"] = y
    df["soc2018"] = df["onet_full"].map(onet_full_to_soc2018)
    df = df.dropna(subset=["soc2018"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# SOC2018 aggregation
# ---------------------------------------------------------------------------

def build_soc_centres(df_occ: pd.DataFrame) -> pd.DataFrame:
    """Aggregate O*NET-detailed occupations into SOC2018 centres.

    The centre is the median of (x, y) over all O*NET-detailed codes mapping
    to the same SOC2018 parent. The polar form is computed from the median.

    Parameters
    ----------
    df_occ : DataFrame
        Output of :func:`load_occupations`.

    Returns
    -------
    DataFrame with columns
    ``[soc2018, x_soc, y_soc, xi_soc, chi_soc, n_onet_detailed]``.
    """
    agg = (
        df_occ.dropna(subset=["soc2018", "x", "y"])
              .groupby("soc2018", as_index=False)
              .agg(
                  x_soc=("x", "median"),
                  y_soc=("y", "median"),
                  n_onet_detailed=("onet_full", "nunique"),
              )
    )
    xi_soc, chi_soc = xy_to_polar(agg["x_soc"].to_numpy(), agg["y_soc"].to_numpy())
    agg["xi_soc"] = xi_soc
    agg["chi_soc"] = chi_soc
    return agg


# ---------------------------------------------------------------------------
# Task radii per SOC2018
# ---------------------------------------------------------------------------

# Threshold for the tau-quantile diagnostic: 1 - 1/e ≈ 0.632.
TAU = 1.0 - np.exp(-1.0)


def _tau_quantile(values) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, TAU)) if arr.size else np.nan


def build_task_radii(
    df_occ: pd.DataFrame,
    df_tasks: pd.DataFrame,
    task_policy: str = "all",
) -> pd.DataFrame:
    """Compute per-SOC task statistics including the RMS task radius.

    For each task we compute the Euclidean distance to its parent O*NET-detailed
    occupation centre. We then aggregate first within each O*NET-detailed
    occupation, then across O*NET-detailed occupations within a SOC2018 parent
    (robust median). The RMS radius is the square root of the mean squared
    task distance.

    Parameters
    ----------
    df_occ : DataFrame
        Output of :func:`load_occupations`.
    df_tasks : DataFrame
        Output of :func:`load_tasks`.
    task_policy : {"all", "core"}
        If ``"core"``, restrict to tasks with ``is_core == True`` (when
        available).

    Returns
    -------
    DataFrame with one row per SOC2018 and columns
    ``[soc2018, med_task_dist, tau_task_dist, task_disp, R_task_rms,
    n_tasks, n_onet_detailed_used, task_policy]``.
    """
    if task_policy not in ("all", "core"):
        raise ValueError("task_policy must be 'all' or 'core'")

    occ_centres = (
        df_occ[["onet_full", "x", "y"]]
        .rename(columns={"x": "x_occ", "y": "y_occ"})
    )

    cols = ["soc2018", "onet_full", "x", "y"]
    if "Task ID" in df_tasks.columns:
        cols.append("Task ID")
    if "is_core" in df_tasks.columns:
        cols.append("is_core")
    tasks = df_tasks[cols].copy()

    if task_policy == "core" and "is_core" in tasks.columns:
        tasks = tasks[tasks["is_core"] == True].copy()

    if "Task ID" in tasks.columns:
        n_tasks_soc = (
            tasks.groupby("soc2018")["Task ID"].nunique()
                 .rename("n_tasks").reset_index()
        )
    else:
        n_tasks_soc = (
            tasks.groupby("soc2018").size()
                 .rename("n_tasks").reset_index()
        )

    tasks = tasks.merge(occ_centres, on="onet_full", how="inner")
    dx = tasks["x"].to_numpy(dtype=float) - tasks["x_occ"].to_numpy(dtype=float)
    dy = tasks["y"].to_numpy(dtype=float) - tasks["y_occ"].to_numpy(dtype=float)
    tasks["task_dist"] = np.hypot(dx, dy)
    tasks["task_dist2"] = tasks["task_dist"].to_numpy(dtype=float) ** 2

    onet_stats = (
        tasks.groupby(["soc2018", "onet_full"], as_index=False)
             .agg(
                 med_task_dist=("task_dist", "median"),
                 tau_task_dist=("task_dist", _tau_quantile),
                 task_disp=("task_dist2", "mean"),
                 n_tasks_onet=("task_dist", "size"),
             )
    )

    soc_stats = (
        onet_stats.groupby("soc2018", as_index=False)
                  .agg(
                      med_task_dist=("med_task_dist", "median"),
                      tau_task_dist=("tau_task_dist", "median"),
                      task_disp=("task_disp", "median"),
                      n_onet_detailed_used=("onet_full", "nunique"),
                  )
                  .merge(n_tasks_soc, on="soc2018", how="inner")
    )
    soc_stats["R_task_rms"] = np.sqrt(soc_stats["task_disp"].to_numpy(dtype=float))
    soc_stats["task_policy"] = task_policy
    return soc_stats