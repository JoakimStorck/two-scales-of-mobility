"""Subgroup analysis of crossing transitions in the residual system.

The residual system (transitions whose source and target belong to different
mobility systems) is heterogeneous. Two structural axes break it down:

* **Managerial direction.** Whether the source and target occupations are
  managerial roles (titles containing manager/supervisor/director/chief/
  superintendent or similar). The combinations give: upward (non-mgr to
  mgr), downward (mgr to non-mgr), lateral (mgr to mgr), or non-mgr-to-
  non-mgr.

* **Geographic segment of the source.** Four arc segments are defined on
  the polar disk based on the source angle xi. They mirror how the
  separatrix divides the disk and let us track flow direction along the
  boundary.

The combination yields seven subgroups (three managerial + four
non-managerial geographic), each with its own typical signature in
task-space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Manager classification
# ---------------------------------------------------------------------------

DEFAULT_MANAGER_KEYWORDS = (
    "supervisor", "manager", "director", "chief", "superintendent",
)


def is_manager_title(
    title,
    keywords=DEFAULT_MANAGER_KEYWORDS,
) -> bool:
    """True if any keyword (case-insensitive) appears in the title."""
    if title is None or (isinstance(title, float) and np.isnan(title)):
        return False
    s = str(title).lower()
    return any(kw in s for kw in keywords)


def classify_manager_direction(
    df_edges: pd.DataFrame,
    title_map: pd.Series,
    keywords=DEFAULT_MANAGER_KEYWORDS,
) -> pd.DataFrame:
    """Add manager-direction columns to a crossing-edge table.

    Parameters
    ----------
    df_edges : DataFrame
        Crossing edges (or any subset). Must have ``src_soc2018`` and
        ``tgt_soc2018``.
    title_map : Series
        Indexed by SOC code, values are occupation titles.
    keywords : tuple of str
        Substrings (lower-cased) that mark a title as managerial.

    Returns
    -------
    DataFrame
        Copy of ``df_edges`` with five new columns:
        ``src_title``, ``tgt_title``, ``is_mgr_src``, ``is_mgr_tgt``,
        ``mgr_dir`` in {"non-mgr", "upward", "downward", "lateral"}.
    """
    df = df_edges.copy()
    df["src_title"] = df["src_soc2018"].map(title_map)
    df["tgt_title"] = df["tgt_soc2018"].map(title_map)
    df["is_mgr_src"] = df["src_title"].apply(
        lambda t: is_manager_title(t, keywords)
    )
    df["is_mgr_tgt"] = df["tgt_title"].apply(
        lambda t: is_manager_title(t, keywords)
    )

    src_m = df["is_mgr_src"].to_numpy()
    tgt_m = df["is_mgr_tgt"].to_numpy()
    mgr_dir = np.where(
        ~src_m & ~tgt_m, "non-mgr",
        np.where(~src_m & tgt_m, "upward",
                 np.where(src_m & ~tgt_m, "downward", "lateral")),
    )
    df["mgr_dir"] = mgr_dir
    return df


# ---------------------------------------------------------------------------
# Geographic segments
# ---------------------------------------------------------------------------

# Default angular boundaries for the four geographic segments along the
# separatrix. Chosen so that the boundary roughly passes through the centres.
DEFAULT_SEGMENTS = (
    ("W->S (CCW)",   178, 266),
    ("S->NE (CW)",   266, 34),
    ("NE->NW (CCW)", 34, 135),
    ("NW->W (CCW)",  135, 178),
)


def assign_segment(xi_deg: float, segments=DEFAULT_SEGMENTS) -> str:
    """Map a source angle in degrees to a segment label."""
    if not np.isfinite(xi_deg):
        return "unknown"
    for label, lo, hi in segments:
        if lo < hi:
            if lo <= xi_deg < hi:
                return label
        else:
            if xi_deg >= lo or xi_deg < hi:
                return label
    return "unknown"


def add_source_angle(
    df_edges: pd.DataFrame,
    df_soc_centres: pd.DataFrame,
) -> pd.DataFrame:
    """Add the source angle (in degrees) as ``xi_deg_src``."""
    xi_map = df_soc_centres.set_index("soc2018")["xi_soc"]
    df = df_edges.copy()
    df["xi_deg_src"] = df["src_soc2018"].map(xi_map).apply(
        lambda r: float(np.degrees(r) % 360) if np.isfinite(r) else np.nan
    )
    return df


# ---------------------------------------------------------------------------
# Subgroup labelling (manager direction crossed with segment)
# ---------------------------------------------------------------------------

def classify_subgroups(
    df_edges_crossing: pd.DataFrame,
    df_soc_centres: pd.DataFrame,
    title_map: pd.Series,
    keywords=DEFAULT_MANAGER_KEYWORDS,
    segments=DEFAULT_SEGMENTS,
) -> pd.DataFrame:
    """Add ``mgr_dir``, ``xi_deg_src``, ``segment``, and ``subgroup`` to edges.

    The subgroup is:

    * Managerial directions (``upward``, ``downward``, ``lateral``) carry
      their own subgroup label.
    * Non-managerial edges are split by source segment, giving four
      ``Non-mgr <segment>`` subgroups plus an ``Non-mgr unknown`` catch-all.
    """
    df = classify_manager_direction(df_edges_crossing, title_map, keywords)
    df = add_source_angle(df, df_soc_centres)
    df["segment"] = df["xi_deg_src"].apply(
        lambda x: assign_segment(x, segments)
    )

    mgr_dir = df["mgr_dir"].to_numpy()
    segment = df["segment"].to_numpy()
    subgroup = np.where(
        mgr_dir == "upward", "Upward (-> mgr)",
        np.where(mgr_dir == "downward", "Downward (mgr ->)",
                 np.where(mgr_dir == "lateral", "Lateral (mgr <-> mgr)",
                          np.array(["Non-mgr " + s for s in segment]))),
    )
    df["subgroup"] = subgroup
    return df


# Default subgroup ordering and styling. The plot functions accept overrides.
SUBGROUP_ORDER = (
    "Upward (-> mgr)",
    "Downward (mgr ->)",
    "Lateral (mgr <-> mgr)",
    "Non-mgr W->S (CCW)",
    "Non-mgr S->NE (CW)",
    "Non-mgr NE->NW (CCW)",
    "Non-mgr NW->W (CCW)",
    "Non-mgr unknown",
)

SUBGROUP_STYLES = {
    "Upward (-> mgr)":       dict(color="#4477AA", marker="^"),
    "Downward (mgr ->)":     dict(color="#EE7733", marker="v"),
    "Lateral (mgr <-> mgr)": dict(color="#AA3377", marker="D"),
    "Non-mgr W->S (CCW)":    dict(color="#66CCEE", marker="o"),
    "Non-mgr S->NE (CW)":    dict(color="#DDCC77", marker="o"),
    "Non-mgr NE->NW (CCW)":  dict(color="#44AA99", marker="o"),
    "Non-mgr NW->W (CCW)":   dict(color="#BB5566", marker="o"),
    "Non-mgr unknown":       dict(color="0.7",     marker="o"),
}


# ---------------------------------------------------------------------------
# Source positions per subgroup (for scatter plotting)
# ---------------------------------------------------------------------------

def source_positions_by_subgroup(
    df_edges_classified: pd.DataFrame,
    df_soc_centres: pd.DataFrame,
    title_map: pd.Series = None,
) -> pd.DataFrame:
    """Aggregate per-source outflow weight by subgroup, with positions.

    Returns one row per (source SOC, subgroup) pair, with columns
    ``src_soc2018, subgroup, w_out, x, y, title``.
    """
    src_pos = (
        df_edges_classified.groupby(["src_soc2018", "subgroup"])["wP"]
        .sum().reset_index()
        .rename(columns={"wP": "w_out"})
    )
    centres = df_soc_centres.set_index("soc2018")[["x_soc", "y_soc"]]
    src_pos = src_pos.merge(
        centres.reset_index().rename(columns={
            "soc2018": "src_soc2018",
            "x_soc": "x", "y_soc": "y",
        }),
        on="src_soc2018", how="left",
    )
    if title_map is not None:
        src_pos["title"] = src_pos["src_soc2018"].map(title_map)
    return src_pos


# ---------------------------------------------------------------------------
# Subgroup-level flow synthesis (one weighted arrow per subgroup)
# ---------------------------------------------------------------------------

def flow_synthesis(df_edges_classified: pd.DataFrame) -> pd.DataFrame:
    """Compute one weighted source/target centroid per subgroup.

    For each subgroup, returns the weighted-mean source position
    ``(x0, y0)``, the weighted-mean target position ``(x1, y1)``, the
    arrow length, and a perpendicular spread measure that captures how
    spread out the source occupations are around the centroid line.

    Parameters
    ----------
    df_edges_classified : DataFrame
        Output of :func:`classify_subgroups`. Must include
        ``x_src, y_src, x_tgt, y_tgt, wP, subgroup``.

    Returns
    -------
    DataFrame
        One row per subgroup with columns
        ``[subgroup, n_edges, w_sum, x0, y0, x1, y1, arrow_len, spread]``.
    """
    rows = []
    for sg, sub in df_edges_classified.groupby("subgroup"):
        sub = sub.dropna(subset=["x_src", "y_src", "x_tgt", "y_tgt", "wP"])
        if len(sub) < 5:
            continue
        w = sub["wP"].to_numpy(dtype=float)
        if w.sum() <= 0:
            continue
        w_norm = w / w.sum()

        x0 = float(np.sum(w_norm * sub["x_src"].to_numpy(dtype=float)))
        y0 = float(np.sum(w_norm * sub["y_src"].to_numpy(dtype=float)))
        x1 = float(np.sum(w_norm * sub["x_tgt"].to_numpy(dtype=float)))
        y1 = float(np.sum(w_norm * sub["y_tgt"].to_numpy(dtype=float)))

        dx = x1 - x0
        dy = y1 - y0
        L = float(np.hypot(dx, dy))
        if L < 1e-12:
            spread = float("nan")
        else:
            ux = dx / L
            uy = dy / L
            xs = sub["x_src"].to_numpy(dtype=float)
            ys = sub["y_src"].to_numpy(dtype=float)
            perp = np.abs((xs - x0) * (-uy) + (ys - y0) * ux)
            spread = float(np.sum(w_norm * perp)) / L

        rows.append({
            "subgroup": sg,
            "n_edges": int(len(sub)),
            "w_sum": float(sub["wP"].sum()),
            "x0": x0, "y0": y0,
            "x1": x1, "y1": y1,
            "arrow_len": L,
            "spread": spread,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Subgroup-level summary table (for paper appendix)
# ---------------------------------------------------------------------------

def _xi_to_cardinal(xi_deg: float) -> str:
    xi = xi_deg % 360
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int((xi + 22.5) / 45) % 8
    return f"{dirs[idx]} ({xi:.0f} deg)"


def subgroup_summary(
    df_edges_classified: pd.DataFrame,
    *,
    weighted_ecdf_fn=None,
) -> pd.DataFrame:
    """Per-subgroup summary table with arrow geometry and median u_R.

    Parameters
    ----------
    df_edges_classified : DataFrame
        Output of :func:`classify_subgroups`. Must also contain ``u_R``.
    weighted_ecdf_fn : callable, optional
        Function with the signature of :func:`mobility.stats.weighted_ecdf`.
        If not given, we import it.

    Returns
    -------
    DataFrame
        One row per subgroup, ordered by :data:`SUBGROUP_ORDER`.
    """
    if weighted_ecdf_fn is None:
        from mobility.stats import weighted_ecdf as weighted_ecdf_fn

    synth = flow_synthesis(df_edges_classified).set_index("subgroup")

    rows = []
    for sg in SUBGROUP_ORDER:
        if sg not in synth.index:
            continue
        sub = df_edges_classified[
            df_edges_classified["subgroup"] == sg
        ].dropna(subset=["x_src", "y_src", "x_tgt", "y_tgt"])
        if len(sub) < 5:
            continue

        synth_row = synth.loc[sg]
        u = sub["u_R"].to_numpy(dtype=float)
        w = sub["wP"].to_numpy(dtype=float)
        valid = np.isfinite(u) & np.isfinite(w) & (w > 0)
        if valid.any():
            xs_cdf, cdf = weighted_ecdf_fn(u[valid], w[valid])
            med_uR = float(np.interp(0.5, cdf, xs_cdf))
        else:
            med_uR = float("nan")

        xi_src = float(np.degrees(np.arctan2(synth_row.y0, synth_row.x0)) % 360)
        xi_tgt = float(np.degrees(np.arctan2(synth_row.y1, synth_row.x1)) % 360)

        rows.append({
            "Subgroup": sg,
            "n_edges": int(synth_row.n_edges),
            "w_sum": round(float(synth_row.w_sum), 1),
            "n_src_occ": int(sub["src_soc2018"].nunique()),
            "n_tgt_occ": int(sub["tgt_soc2018"].nunique()),
            "src_centroid": _xi_to_cardinal(xi_src),
            "tgt_centroid": _xi_to_cardinal(xi_tgt),
            "arrow_len": round(float(synth_row.arrow_len), 3),
            "spread": round(float(synth_row.spread), 3),
            "median_uR": round(med_uR, 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Source-sector x target-sector flow matrix (non-managers)
# ---------------------------------------------------------------------------

def source_segment_to_sector(segment_label: str) -> str:
    """Map a (source) segment label to its full sector label."""
    mapping = {
        "W->S (CCW)":   "Sector 1 (xi 178-266 deg)",
        "S->NE (CW)":   "Sector 2 (xi 266-34 deg)",
        "NE->NW (CCW)": "Sector 3 (xi 34-135 deg)",
        "NW->W (CCW)":  "Sector 4 (xi 135-178 deg)",
    }
    return mapping.get(segment_label, "unknown")


def non_manager_flow_matrix(
    df_edges_classified: pd.DataFrame,
    df_soc_centres: pd.DataFrame,
    *,
    column_normalise: bool = True,
) -> pd.DataFrame:
    """Source-sector by target-sector flow share for non-managerial edges.

    Returns a 4x4 DataFrame where each entry is the share of non-mgr flow
    from a given source sector to a given target sector.

    Parameters
    ----------
    df_edges_classified : DataFrame
        Must have ``mgr_dir == 'non-mgr'`` rows along with ``segment``,
        ``tgt_soc2018``, ``wP``.
    df_soc_centres : DataFrame
        For looking up target angles.
    column_normalise : bool
        If True, columns sum to 100 (each target sector partitioned by
        source sector). If False, returns raw weight totals.
    """
    sector_order = [
        "Sector 1 (xi 178-266 deg)",
        "Sector 2 (xi 266-34 deg)",
        "Sector 3 (xi 34-135 deg)",
        "Sector 4 (xi 135-178 deg)",
    ]

    xi_map = df_soc_centres.set_index("soc2018")["xi_soc"]
    df = df_edges_classified.copy()
    df["xi_deg_tgt"] = df["tgt_soc2018"].map(xi_map).apply(
        lambda r: float(np.degrees(r) % 360) if np.isfinite(r) else np.nan
    )
    df["target_sector"] = df["xi_deg_tgt"].apply(
        lambda x: source_segment_to_sector(assign_segment(x))
    )
    df["source_sector"] = df["segment"].apply(source_segment_to_sector)

    non_mgr = df[df["mgr_dir"] == "non-mgr"]
    matrix = (
        non_mgr.groupby(["source_sector", "target_sector"])["wP"]
        .sum().unstack(fill_value=0.0)
        .reindex(index=sector_order, columns=sector_order, fill_value=0.0)
    )
    if column_normalise:
        col_sum = matrix.sum(axis=0).replace(0, np.nan)
        matrix = (100 * matrix / col_sum).round(1)
    return matrix