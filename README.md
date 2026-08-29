# Team 9 — Silicon Sample Benchmark Tier 3 Submission

This repository contains Team 9's `secondary-1` Tier 3 submission. Each ATE is the organizer-specified difference between the poststratified calibrated intervention mean and the corresponding poststratified pooled-control mean from Team 9's Tier 2 pipeline.

## Submission file

- `predictions/team_9_T3_secondary-1_v1.csv` — 208 intervention × outcome ATEs; SHA-256 `c66a2a23fa2c9159322d1feecff0afd38ba059ab802593a466fbde9118210021`.
- `registration.md` and `metadata.json` — method and submission metadata.
- `code/calib/` and `code/syn-digits/` — calibration, aggregation, and ATE code.
- `artifacts/` — aligned Wave-4 anchors, raw and calibrated Silicon target matrices, the 40-cell target grid, respondent weights, fit diagnostics, and audit reports.

Run the organizers' validator with `make check`. To regenerate the elastic-net calibration and verify the submitted ATEs from the public matrices, run `PYTHONPATH=code python -m calib.reproduce_submission` in an environment with the listed scientific dependencies. The simulation request and response archive is available under restricted access at [Zenodo](https://doi.org/10.5281/zenodo.22150315).

Authors: Olivier Toubia, Tianyi Peng, George Gui, Yuchen Qiu, and Naveen Venkat. Corresponding contact: Olivier Toubia (`ot2107@gsb.columbia.edu`).
