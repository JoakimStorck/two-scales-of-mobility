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

## 3. `transitions_20_24.csv`

Aggregated occupational transitions 2020–2024 from IPUMS-CPS.

Required columns:
- `source` — source SOC2018 code (e.g. `35-9031`)
- `destination` — destination SOC2018 code
- `WTFINL` — sum of person-weights for this source–destination pair

This file is **not** redistributed in this repository due to IPUMS-CPS
licensing restrictions.

### Reproducing the IPUMS extract

*To be filled in.* The extract was prepared by J. Andersson. Documentation
of the variable selection, panel matching rules, and aggregation procedure
will be added.