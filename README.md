# Two Scales of Occupational Mobility

Replication code for *Two Scales of Occupational Mobility* (Storck & Andersson, 2026).

This repository contains the code required to reproduce the figures and tables
in the paper. It builds on the polar task-space geometry developed in
*The Polar Geometry of Work* (Storck & Andersson, 2026).

## Requirements

- Python 3.10 or newer
- See `requirements.txt` for package dependencies

## Setup

```bash
git clone https://github.com/JoakimStorck/two-scales-of-mobility.git
cd two-scales-of-mobility
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The `data/` directory must contain three CSV files before the analysis can run:

1. `occupation_embeddings_polar_scaled.csv` — occupation positions in polar
   task-space, from the first paper.
2. `task_embeddings_polar_scaled.csv` — task positions in polar task-space,
   from the first paper.
3. `transitions_20_24.csv` — aggregated occupational transitions 2020–2024
   from IPUMS-CPS.

See `data/README.md` for details on each file and how to generate them.

The IPUMS-CPS extract is not redistributed in this repository due to licensing
restrictions. Instructions for reproducing the extract are in `data/README.md`.

## Reproducing the analysis

Run the notebook:

```bash
jupyter notebook notebooks/two_scales.ipynb
```

All figures and tables are written to `outputs/figures/` and `outputs/tables/`.

## Repository structure

```
mobility/      Python package with analysis modules
notebooks/     Reproduction notebook
data/          Input data (CSV)
outputs/       Generated figures and tables
paper/         LaTeX source
```

## Citation

If you use this code, please cite:

> Storck, J. & Andersson, J. (2026). Two scales of occupational mobility.

## License

MIT — see `LICENSE`.
