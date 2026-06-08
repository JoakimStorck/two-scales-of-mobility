"""IPUMS-CPS transition matrix construction.

The pipeline:

1. Obtain the source-destination-weight cell table (:func:`load_transitions`):
   read the raw IPUMS microdata extract if present and aggregate it, otherwise
   read the redistributable aggregated cell table.
2. Build a square row-normalised matrix P over the SOC2018 universe.
3. Build an edge table where each row is a source-destination pair with
   probability above threshold, plus geometric distance and normalised
   hop length.

Two on-disk representations of the mobility data
------------------------------------------------
Raw extract (e.g. ``transitions_20_24.csv``)
    Individual-level IPUMS-CPS microdata: one row per matched person, with
    columns ``CPSIDP, YEAR, MONTH, soc2018, WTFINL, source, destination``.
    This is the IPUMS-repackaged microdata and **must not** be redistributed
    under the IPUMS-CPS terms of use; it is git-ignored.

Aggregated cell table (e.g. ``transitions_20_24_aggregated.csv``)
    A derived tabulation with all person-level identifiers
    (``CPSIDP``, ``YEAR``, ``MONTH``) removed: one row per
    source-destination SOC2018 pair, with the summed person-weight ``w`` and
    the unweighted observation count ``n_obs``. This is a derived statistic
    rather than the microdata and is the file committed to the repository.

The cell table is a sufficient statistic for everything downstream (F, P, the
edge table), so the two representations yield numerically identical results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mobility.geometry import norm_soc2018, polar_to_xy


# Default file names within the data directory.
RAW_TRANSITIONS_NAME = "transitions_20_24.csv"
AGG_TRANSITIONS_NAME = "transitions_20_24_aggregated.csv"

# Canonical schema of the aggregated cell table on disk.
AGG_COLUMNS = ["source", "destination", "w", "n_obs"]


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def load_raw_transitions(path) -> pd.DataFrame:
    """Load and clean the raw IPUMS-CPS microdata extract.

    Required columns: ``source``, ``destination``, ``WTFINL``. Any further
    columns (``CPSIDP``, ``YEAR``, ``MONTH``, ``soc2018``, ...) are ignored.

    Returns a per-record DataFrame with normalised columns
    ``[source_soc, dest_soc, w]`` and only positive finite weights.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    required = {"source", "destination", "WTFINL"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Raw IPUMS file missing columns: {missing}")

    df["source_soc"] = df["source"].map(norm_soc2018)
    df["dest_soc"] = df["destination"].map(norm_soc2018)
    df["w"] = pd.to_numeric(df["WTFINL"], errors="coerce")
    df = df.dropna(subset=["source_soc", "dest_soc", "w"])
    df = df[np.isfinite(df["w"]) & (df["w"] > 0)].copy()
    return df[["source_soc", "dest_soc", "w"]].reset_index(drop=True)


# Backward-compatible alias.
load_ipums_transitions = load_raw_transitions


def aggregate_transitions(
    df_records: pd.DataFrame,
    min_cell_n: int | None = None,
) -> pd.DataFrame:
    """Collapse cleaned per-record transitions to a source-destination cell table.

    Parameters
    ----------
    df_records : DataFrame
        Output of :func:`load_raw_transitions`, columns ``[source_soc,
        dest_soc, w]``.
    min_cell_n : int | None
        If set, drop cells supported by fewer than ``min_cell_n`` unweighted
        observations. Leave as ``None`` (the default) to preserve every cell
        and keep results exactly reproducible from the published file. Any
        suppression here is baked into the aggregated table itself, so the raw
        and aggregated paths stay numerically identical.

    Returns
    -------
    DataFrame with columns ``[source_soc, dest_soc, w, n_obs]``, sorted for a
    deterministic on-disk ordering. ``w`` is the summed person-weight and
    ``n_obs`` the unweighted record count for the cell.
    """
    agg = (
        df_records.groupby(["source_soc", "dest_soc"], as_index=False)
        .agg(w=("w", "sum"), n_obs=("w", "size"))
    )
    if min_cell_n is not None and int(min_cell_n) > 1:
        agg = agg[agg["n_obs"] >= int(min_cell_n)]
    agg = agg.sort_values(["source_soc", "dest_soc"], kind="stable")
    return agg.reset_index(drop=True)


def save_aggregated_transitions(df_agg: pd.DataFrame, path) -> None:
    """Write the aggregated cell table to ``path`` with the canonical schema.

    Columns are renamed to the on-disk schema ``[source, destination, w,
    n_obs]`` so the published file reads cleanly without exposing internal
    column names.
    """
    out = df_agg.rename(columns={"source_soc": "source", "dest_soc": "destination"})
    out = out[AGG_COLUMNS]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_aggregated_transitions(path) -> pd.DataFrame:
    """Load the redistributable aggregated cell table.

    Required columns: ``source``, ``destination``, ``w``. ``n_obs`` is read
    when present. Returns columns ``[source_soc, dest_soc, w, n_obs]``,
    drop-in compatible with :func:`build_transition_matrix`.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    required = {"source", "destination", "w"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Aggregated transitions file missing columns: {missing}")

    df["source_soc"] = df["source"].map(norm_soc2018)
    df["dest_soc"] = df["destination"].map(norm_soc2018)
    df["w"] = pd.to_numeric(df["w"], errors="coerce")
    if "n_obs" in df.columns:
        df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce")
    else:
        df["n_obs"] = np.nan
    df = df.dropna(subset=["source_soc", "dest_soc", "w"])
    df = df[np.isfinite(df["w"]) & (df["w"] > 0)].copy()
    return df[["source_soc", "dest_soc", "w", "n_obs"]].reset_index(drop=True)


def load_transitions(
    data_dir,
    raw_name: str = RAW_TRANSITIONS_NAME,
    agg_name: str = AGG_TRANSITIONS_NAME,
    min_cell_n: int | None = None,
    write_aggregated: bool = True,
) -> pd.DataFrame:
    """Obtain the transition cell table, preferring raw microdata if present.

    Resolution order:

    1. If ``data_dir/raw_name`` exists, load and aggregate it. When
       ``write_aggregated`` is true, (re)write ``data_dir/agg_name`` as the
       redistributable derived table.
    2. Otherwise load the already-aggregated ``data_dir/agg_name``.
    3. If neither exists, raise ``FileNotFoundError``.

    Both branches return columns ``[source_soc, dest_soc, w, n_obs]`` and feed
    :func:`build_transition_matrix` unchanged. Because the aggregated table is
    a sufficient statistic, the numbers are identical whether a collaborator
    has the raw extract or only the published aggregate.
    """
    data_dir = Path(data_dir)
    raw_path = data_dir / raw_name
    agg_path = data_dir / agg_name

    if raw_path.exists():
        df_records = load_raw_transitions(raw_path)
        df_agg = aggregate_transitions(df_records, min_cell_n=min_cell_n)
        if write_aggregated:
            save_aggregated_transitions(df_agg, agg_path)
        return df_agg

    if agg_path.exists():
        return load_aggregated_transitions(agg_path)

    raise FileNotFoundError(
        f"Neither raw extract ({raw_path.name}) nor aggregated table "
        f"({agg_path.name}) found in {data_dir}. See data/README.md."
    )


def build_transition_matrix(
    df_ipums: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the row-normalised SOC2018 transition matrix.

    Parameters
    ----------
    df_ipums : DataFrame
        Output of :func:`load_ipums_transitions`.

    Returns
    -------
    F : ndarray, shape (N, N)
        Raw flow matrix (sum of weights). Diagonal includes self-flows
        if any were present in the input.
    P : ndarray, shape (N, N)
        Row-normalised transition matrix. Rows that summed to zero in F
        remain zero in P.
    soc_index : DataFrame
        Two columns ``[id, soc2018]`` mapping matrix indices to SOC codes.
    """
    soc_list = sorted(set(df_ipums["source_soc"]) | set(df_ipums["dest_soc"]))
    soc_to_id = {s: i for i, s in enumerate(soc_list)}
    n = len(soc_list)

    grouped = (
        df_ipums.groupby(["source_soc", "dest_soc"], as_index=False)["w"].sum()
    )
    src_idx = grouped["source_soc"].map(soc_to_id).to_numpy(dtype=int)
    dst_idx = grouped["dest_soc"].map(soc_to_id).to_numpy(dtype=int)
    weights = grouped["w"].to_numpy(dtype=float)

    F = np.zeros((n, n), dtype=np.float64)
    np.add.at(F, (src_idx, dst_idx), weights)

    row_sum = F.sum(axis=1, keepdims=True)
    P = np.divide(
        F, row_sum,
        out=np.zeros_like(F, dtype=np.float64),
        where=(row_sum > 0),
    )

    soc_index = pd.DataFrame({"id": range(n), "soc2018": soc_list})
    return F, P, soc_index


# ---------------------------------------------------------------------------
# Edge table construction
# ---------------------------------------------------------------------------

def build_edges(
    P: np.ndarray,
    soc_index: pd.DataFrame,
    df_soc_centres: pd.DataFrame,
    df_task_radii: pd.DataFrame,
    p_threshold: float = 0.0,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Build the SOC2018 → SOC2018 edge table with geometric distances.

    Self-loops are dropped. For each source, edges below ``p_threshold``
    are excluded; if ``top_k`` is given, only the strongest ``top_k``
    destinations are kept.

    Parameters
    ----------
    P : ndarray
        Row-normalised transition matrix from :func:`build_transition_matrix`.
    soc_index : DataFrame
        SOC index from :func:`build_transition_matrix`.
    df_soc_centres : DataFrame
        Output of :func:`mobility.geometry.build_soc_centres`. Must contain
        ``soc2018``, ``x_soc``, ``y_soc``.
    df_task_radii : DataFrame
        Output of :func:`mobility.geometry.build_task_radii`. Must contain
        ``soc2018``, ``R_task_rms``.
    p_threshold : float
        Minimum probability to retain an edge.
    top_k : int | None
        If set, keep at most this many destinations per source.

    Returns
    -------
    DataFrame with columns
    ``[src_soc2018, tgt_soc2018, wP, d_xy, R_src_task_rms, u_R,
    x_src, y_src, x_tgt, y_tgt]``.

    The normalised hop length is ``u_R = d_xy / R_src_task_rms``, where
    ``R_src_task_rms`` is the RMS task radius of the source occupation.
    """
    soc_pool = sorted(
        set(soc_index["soc2018"])
        & set(df_soc_centres["soc2018"])
        & set(df_task_radii["soc2018"])
    )

    soc_to_id = dict(zip(soc_index["soc2018"], soc_index["id"]))
    id_to_soc = soc_index.set_index("id")["soc2018"]

    centres = df_soc_centres.set_index("soc2018")[["x_soc", "y_soc"]]
    radii = (
        df_task_radii.set_index("soc2018")[["R_task_rms"]]
        .rename(columns={"R_task_rms": "R_src_task_rms"})
    )

    rows = []
    for src in soc_pool:
        i = soc_to_id.get(src)
        if i is None:
            continue
        probs = P[i, :].copy()
        probs[i] = 0.0  # drop self-loop

        idx = np.where(probs > float(p_threshold))[0]
        if idx.size == 0:
            continue
        if top_k is not None and int(top_k) > 0 and idx.size > int(top_k):
            idx = idx[np.argsort(probs[idx])[-int(top_k):]]

        x_src = float(centres.loc[src, "x_soc"])
        y_src = float(centres.loc[src, "y_soc"])
        R_src = float(radii.loc[src, "R_src_task_rms"])

        for j in idx:
            tgt = id_to_soc.iloc[int(j)]
            if tgt not in centres.index or tgt not in radii.index:
                continue
            x_tgt = float(centres.loc[tgt, "x_soc"])
            y_tgt = float(centres.loc[tgt, "y_soc"])
            d_xy = float(np.hypot(x_src - x_tgt, y_src - y_tgt))
            u_R = d_xy / R_src if R_src > 0 else np.nan
            rows.append((
                src, tgt, float(probs[j]), d_xy,
                R_src, u_R,
                x_src, y_src, x_tgt, y_tgt,
            ))
    
    return pd.DataFrame(rows, columns=[
        "src_soc2018", "tgt_soc2018", "wP", "d_xy",
        "R_src_task_rms", "u_R",
        "x_src", "y_src", "x_tgt", "y_tgt",
    ])