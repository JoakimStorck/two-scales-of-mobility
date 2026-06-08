# Data

This directory must contain three CSV files for the analysis to run.

## 1. `occupation_embeddings_polar_scaled.csv`

Source: *The Polar Geometry of Work* (Storck & Andersson, 2026).
GitHub: http://www.github.com/JoakimStorck/geometry-of-work/

Required columns:
- `onet_code` — O*NET-SOC code (e.g. `15-1252.00`)
- `xi` — angular polar coordinate, radians
- `chi` — radial polar coordinate, scaled to unit disk

Optional columns used by the analysis if present:
- `Title`, `Job Family`, `sector_zone`

## 2. `task_embeddings_polar_scaled.csv`

Source: *The Polar Geometry of Work* (Storck & Andersson, 2026).
GitHub: http://www.github.com/JoakimStorck/geometry-of-work/

Required columns:
- `onet_code` — O*NET-SOC code of the parent occupation
- `xi`, `chi` — task position in polar task-space

Optional columns:
- `Task ID`, `Task`, `is_core`

## 3. Occupational transitions 2020–2024 (IPUMS-CPS)

The mobility data is held in two on-disk representations. The loader
`mobility.transitions.load_transitions(DATA_DIR)` prefers the raw extract when
present and otherwise falls back to the aggregated table; both yield
numerically identical results downstream.

### 3a. `transitions_20_24.csv` — raw microdata (not redistributed)

Individual-level IPUMS-CPS extract: **one row per matched person**.

Columns: `CPSIDP, YEAR, MONTH, soc2018, WTFINL, source, destination`, where
- `source` — origin SOC2018 code (e.g. `35-2021`)
- `destination` — destination SOC2018 code
- `WTFINL` — that person's CPS final weight
- `CPSIDP, YEAR, MONTH, soc2018` — person identifier and panel info

This is the IPUMS-repackaged microdata. Under the IPUMS-CPS terms of use it
**must not** be redistributed, so it is git-ignored and never committed.

### 3b. `transitions_20_24_aggregated.csv` — derived cell table (redistributed)

A derived tabulation with all person-level identifiers (`CPSIDP`, `YEAR`,
`MONTH`) removed: **one row per source–destination SOC2018 pair**.

Columns:
- `source` — origin SOC2018 code
- `destination` — destination SOC2018 code
- `w` — summed person-weight (`WTFINL`) over all records in the cell
- `n_obs` — unweighted count of records (matched persons) in the cell

This is a descriptive aggregate rather than the microdata. IPUMS permits
sharing "descriptive statistics, aggregate data, or analytical results
derived from the microdata", so this file **is** committed to the repository
and is what reproduces the published analysis. When the raw extract is
present, `load_transitions` regenerates this file automatically.

Optional disclosure control: `aggregate_transitions(..., min_cell_n=K)` can
suppress cells with `n_obs < K`. It is **off by default** so that the
committed file reproduces the paper exactly; any suppression is applied
identically on both load paths.

### Reproducing the IPUMS extract

*To be filled in.* The raw extract was prepared by J. Andersson. Documentation
of the variable selection, panel matching rules (CPS 4-8-4 rotation, `CPSIDP`
linkage), and aggregation procedure will be added.