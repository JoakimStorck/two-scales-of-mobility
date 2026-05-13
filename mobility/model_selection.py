"""Model selection for the number of mobility subsystems.

The main analysis fits a Gaussian mixture in 4D edge space with
K = 2 components. This module evaluates alternative K via BIC and AIC,
with seed variation to characterise stability. Two BIC/AIC variants are
reported:

* **Replicated.** Computed via :meth:`GaussianMixture.bic` /
  :meth:`~GaussianMixture.aic` on the weight-replicated design matrix,
  matching how :func:`mobility.systems.detect_systems` fits the model.
* **Weighted.** A true weighted log-likelihood evaluated on the
  unreplicated points, using each edge's wP as its weight, with N taken
  as Kish's effective sample size.

Both should agree on the preferred K if the choice is robust.

For each (K, seed) we also re-use :func:`mobility.systems._flow_stats`
to compute ``flow_between`` and ``min_sys_frac`` for the same fit, so
the BIC-preferred K can be compared directly with the criterion used in
:func:`mobility.systems.detect_systems`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from mobility.systems import _flow_stats


# ---------------------------------------------------------------------------
# Likelihood and information criteria
# ---------------------------------------------------------------------------

def _n_free_params(n_components: int, n_features: int,
                   covariance_type: str = "full") -> int:
    """Number of free parameters in a Gaussian mixture model.

    Matches sklearn's accounting in GaussianMixture._n_parameters().
    """
    k = n_components
    d = n_features
    if covariance_type == "full":
        cov_params = k * d * (d + 1) // 2
    elif covariance_type == "tied":
        cov_params = d * (d + 1) // 2
    elif covariance_type == "diag":
        cov_params = k * d
    elif covariance_type == "spherical":
        cov_params = k
    else:
        raise ValueError(f"unknown covariance_type: {covariance_type}")
    mean_params = k * d
    weight_params = k - 1
    return int(cov_params + mean_params + weight_params)


def weighted_log_likelihood(
    model: GaussianMixture,
    X: np.ndarray,
    w: np.ndarray,
) -> float:
    """Sum of w_i * log p(x_i | model) over all edges."""
    log_prob = model.score_samples(X)  # log p(x_i) per point
    return float(np.sum(w * log_prob))


def kish_effective_n(w: np.ndarray) -> float:
    """Kish's effective sample size: (sum w)^2 / sum(w^2)."""
    s1 = float(np.sum(w))
    s2 = float(np.sum(w ** 2))
    return s1 * s1 / s2 if s2 > 0 else 0.0


def bic_aic_weighted(
    model: GaussianMixture,
    X: np.ndarray,
    w: np.ndarray,
    covariance_type: str = "full",
) -> tuple[float, float, float, float]:
    """BIC and AIC computed from a true weighted log-likelihood.

    Returns
    -------
    bic, aic, log_lik_weighted, n_eff
    """
    ll = weighted_log_likelihood(model, X, w)
    n_eff = kish_effective_n(w)
    p = _n_free_params(model.n_components, X.shape[1], covariance_type)
    bic = -2.0 * ll + p * np.log(max(n_eff, 1.0))
    aic = -2.0 * ll + 2.0 * p
    return bic, aic, ll, n_eff


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@dataclass
class SelectionRun:
    """One (K, seed) GMM fit with all derived metrics."""

    K: int
    seed: int
    # Replicated-design criteria (consistent with sklearn's accounting)
    bic_rep: float
    aic_rep: float
    log_lik_rep: float
    # Weighted criteria (true weighted log-likelihood)
    bic_w: float
    aic_w: float
    log_lik_w: float
    n_eff: float
    # Selection-relevant metrics from the SOC aggregation
    flow_within: float        # % of total weight
    flow_between: float       # % of total weight
    min_sys_frac: float       # % of smallest system
    silhouette: float
    n_edges: int
    n_params: int
    converged: bool


def evaluate_n_components(
    df_edges: pd.DataFrame,
    *,
    K_values: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    n_seeds: int = 20,
    n_init: int = 3,
    covariance_type: str = "full",
    weight_replication_max: int = 100,
    silhouette_sample: int = 2000,
    base_seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit GMMs across K and seeds and return all per-run metrics.

    Parameters
    ----------
    df_edges : DataFrame
        Edge table with ``x_src, y_src, x_tgt, y_tgt, wP`` plus the SOC
        columns needed by ``_flow_stats``.
    K_values : tuple[int]
        Numbers of mixture components to evaluate.
    n_seeds : int
        Number of random seeds per K.
    n_init : int
        ``n_init`` for each GaussianMixture fit. Lower than the production
        value (5) to keep the loop manageable; seed variation provides
        the robustness we need.
    covariance_type : str
        Passed to GaussianMixture. Default ``"full"`` matches
        :func:`mobility.systems.detect_systems`.
    weight_replication_max : int
        Same convention as ``detect_systems``: cap on integer multiplicity
        for weight-replicated fitting.
    silhouette_sample : int
        Subsample size for the silhouette computation.
    base_seed : int
        Seeds used are ``base_seed + i`` for i in range(n_seeds).
    verbose : bool
        Print progress per K.

    Returns
    -------
    DataFrame
        One row per (K, seed). Columns include the BIC/AIC variants,
        flow statistics, and convergence flag.
    """
    X = df_edges[["x_src", "y_src", "x_tgt", "y_tgt"]].to_numpy(dtype=float)
    w = df_edges["wP"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    w_int = np.clip(
        np.round(weight_replication_max * w / w.max()),
        1, weight_replication_max,
    ).astype(int)
    X_rep = np.repeat(X_sc, w_int, axis=0)

    rows: list[SelectionRun] = []
    for K in K_values:
        if verbose:
            print(f"K={K}: ", end="", flush=True)
        for i in range(n_seeds):
            seed = base_seed + i
            model = GaussianMixture(
                n_components=K,
                covariance_type=covariance_type,
                random_state=seed,
                n_init=n_init,
            )
            model.fit(X_rep)

            # Replicated-design criteria (sklearn's accounting)
            bic_rep = float(model.bic(X_rep))
            aic_rep = float(model.aic(X_rep))
            ll_rep = float(model.score(X_rep) * len(X_rep))

            # Weighted criteria (true weighted likelihood)
            bic_w, aic_w, ll_w, n_eff = bic_aic_weighted(
                model, X_sc, w, covariance_type=covariance_type,
            )

            # SOC-aggregated flow statistics
            labels = model.predict(X_sc)
            within, between, _, _, min_frac = _flow_stats(
                df_edges, labels, K,
            )

            # Silhouette on subsample
            rng = np.random.default_rng(seed)
            sub_n = min(silhouette_sample, len(X_sc))
            sub = rng.choice(len(X_sc), size=sub_n, replace=False)
            if K >= 2 and len(np.unique(labels[sub])) >= 2:
                try:
                    sil = float(silhouette_score(
                        X_sc[sub], labels[sub], sample_weight=w[sub],
                    ))
                except Exception:
                    sil = float("nan")
            else:
                sil = float("nan")

            rows.append(SelectionRun(
                K=K, seed=seed,
                bic_rep=bic_rep, aic_rep=aic_rep, log_lik_rep=ll_rep,
                bic_w=bic_w, aic_w=aic_w, log_lik_w=ll_w, n_eff=n_eff,
                flow_within=within * 100, flow_between=between * 100,
                min_sys_frac=min_frac * 100,
                silhouette=sil,
                n_edges=len(df_edges),
                n_params=_n_free_params(K, X_sc.shape[1], covariance_type),
                converged=bool(model.converged_),
            ))
            if verbose:
                print(".", end="", flush=True)
        if verbose:
            print()

    return pd.DataFrame([r.__dict__ for r in rows])


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def summarize_selection(
    df_runs: pd.DataFrame,
    metrics: tuple[str, ...] = (
        "bic_rep", "aic_rep", "bic_w", "aic_w",
        "flow_between", "min_sys_frac", "silhouette",
    ),
) -> pd.DataFrame:
    """Aggregate per-run metrics to mean and std per K.

    Returns a DataFrame indexed by K with columns
    ``<metric>_mean`` and ``<metric>_std`` for each requested metric, plus
    ``best_seed_<metric>`` for criteria where lower is better (BIC/AIC) or
    higher is better (silhouette).
    """
    out = {}
    for K, sub in df_runs.groupby("K"):
        row = {"n_seeds": int(len(sub))}
        for m in metrics:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_std"] = float(sub[m].std(ddof=1)) if len(sub) > 1 else 0.0
            row[f"{m}_min"] = float(sub[m].min())
            row[f"{m}_max"] = float(sub[m].max())
        out[K] = row
    df = pd.DataFrame(out).T
    df.index.name = "K"
    return df.reset_index()


def winning_K(
    df_runs: pd.DataFrame,
    criterion: str = "bic_w",
    exclude_K: tuple[int, ...] = (),
) -> dict:
    """Identify the K preferred by a given criterion.

    For BIC/AIC, lower is better. For silhouette, higher is better.
    For flow_between, lower is better (more separable subsystems) but
    K=1 is trivially 0 and should be excluded.

    Parameters
    ----------
    df_runs : DataFrame
        Output of :func:`evaluate_n_components`.
    criterion : str
        Column to optimise. NaN values within a group are skipped; K
        values where the criterion is NaN for all seeds are dropped.
    exclude_K : tuple of int
        K values to exclude before selecting. Use ``(1,)`` for
        ``flow_between`` and ``silhouette`` (both are degenerate at K=1).

    Returns the winning K per seed (a vote count), and the K with the
    best seed-averaged value.
    """
    df = df_runs.copy()
    if exclude_K:
        df = df[~df["K"].isin(exclude_K)]
    # Drop rows where the criterion is NaN
    df = df.dropna(subset=[criterion])
    if df.empty:
        return {
            "criterion": criterion,
            "best_by_mean": None,
            "vote_counts_per_seed": {},
            "n_seeds": 0,
            "note": "all values NaN after exclusions",
        }

    higher_is_better = {"silhouette"}
    if criterion in higher_is_better:
        winners = df.loc[df.groupby("seed")[criterion].idxmax(), "K"]
        best_by_mean = int(df.groupby("K")[criterion].mean().idxmax())
    else:
        winners = df.loc[df.groupby("seed")[criterion].idxmin(), "K"]
        best_by_mean = int(df.groupby("K")[criterion].mean().idxmin())

    vote_counts = winners.value_counts().sort_index().to_dict()
    vote_counts = {int(k): int(v) for k, v in vote_counts.items()}

    return {
        "criterion": criterion,
        "best_by_mean": best_by_mean,
        "vote_counts_per_seed": vote_counts,
        "n_seeds": int(df["seed"].nunique()),
    }


def relative_improvement(
    df_runs: pd.DataFrame,
    criterion: str = "bic_w",
) -> pd.DataFrame:
    """Marginal improvement when going from K-1 to K.

    For BIC/AIC (lower is better) the improvement is the *drop*.
    Returns a DataFrame with one row per K (skipping the smallest K),
    showing the mean criterion value, the absolute drop from K-1, and
    the drop as a fraction of the K=1 -> K=2 drop (so the magnitudes
    are comparable across orders of magnitude).
    """
    summary = summarize_selection(df_runs, metrics=(criterion,))
    K_vals = summary["K"].to_numpy()
    means = summary[f"{criterion}_mean"].to_numpy()

    rows = []
    base_drop = None
    for i, K in enumerate(K_vals):
        if i == 0:
            rows.append({
                "K": int(K),
                f"{criterion}_mean": float(means[i]),
                "drop_from_prev": float("nan"),
                "drop_share_of_first": float("nan"),
            })
            continue
        drop = float(means[i - 1] - means[i])  # positive if improving
        if i == 1:
            base_drop = drop if abs(drop) > 1e-12 else 1.0
        rows.append({
            "K": int(K),
            f"{criterion}_mean": float(means[i]),
            "drop_from_prev": drop,
            "drop_share_of_first": drop / base_drop if base_drop else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model_selection(
    df_runs: pd.DataFrame,
    out_path: str | None = None,
    *,
    show_flow_between: bool = True,
) -> "matplotlib.figure.Figure":
    """Elbow plot: BIC (both variants), AIC (both variants), flow_between.

    Each metric is shown as mean over seeds plus +/- 1 SD as a shaded band.
    BIC/AIC are plotted on twin y-axes appropriate to their scale.
    """
    import matplotlib.pyplot as plt

    summary = summarize_selection(df_runs)
    K_arr = summary["K"].to_numpy()

    if show_flow_between:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        ax_ic, ax_fb = axes
    else:
        fig, ax_ic = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        ax_fb = None

    # --- Information criteria panel ---
    metrics_ic = [
        ("bic_rep", "BIC (replicated)",  "tab:blue",   "-"),
        ("aic_rep", "AIC (replicated)",  "tab:blue",   "--"),
        ("bic_w",   "BIC (weighted)",    "tab:orange", "-"),
        ("aic_w",   "AIC (weighted)",    "tab:orange", "--"),
    ]
    # Plot replicated and weighted on separate y-axes (different magnitudes).
    ax_ic_rep = ax_ic
    ax_ic_w = ax_ic.twinx()

    for col, label, color, ls in metrics_ic:
        mean = summary[f"{col}_mean"].to_numpy()
        std = summary[f"{col}_std"].to_numpy()
        ax = ax_ic_rep if col.endswith("_rep") else ax_ic_w
        ax.plot(K_arr, mean, color=color, linestyle=ls, marker="o",
                label=label, linewidth=1.6)
        ax.fill_between(K_arr, mean - std, mean + std, color=color, alpha=0.12)

    ax_ic_rep.set_xlabel("Number of components K")
    ax_ic_rep.set_ylabel("Information criterion (replicated design)",
                         color="tab:blue")
    ax_ic_w.set_ylabel("Information criterion (weighted)", color="tab:orange")
    ax_ic_rep.tick_params(axis="y", labelcolor="tab:blue")
    ax_ic_w.tick_params(axis="y", labelcolor="tab:orange")
    ax_ic_rep.set_xticks(K_arr)
    ax_ic_rep.grid(alpha=0.3)

    # Combined legend
    handles_rep, labels_rep = ax_ic_rep.get_legend_handles_labels()
    handles_w, labels_w = ax_ic_w.get_legend_handles_labels()
    ax_ic_rep.legend(handles_rep + handles_w, labels_rep + labels_w,
                     loc="best", frameon=True, fontsize=8)

    # --- Flow_between panel ---
    if ax_fb is not None:
        mean = summary["flow_between_mean"].to_numpy()
        std = summary["flow_between_std"].to_numpy()
        ax_fb.plot(K_arr, mean, color="tab:green", marker="o", linewidth=1.6)
        ax_fb.fill_between(K_arr, mean - std, mean + std,
                           color="tab:green", alpha=0.18)
        ax_fb.set_xlabel("Number of components K")
        ax_fb.set_ylabel("Crossing flow (% of total weight)")
        ax_fb.set_xticks(K_arr)
        ax_fb.grid(alpha=0.3)

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig