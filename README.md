# Team 9 — Silicon Sample Benchmark Tier 3 Submission

This repository contains Team 9's `secondary-1` Tier 3 submission. Each ATE is the organizer-specified difference between the demographically reweighted intervention mean and the corresponding reweighted pooled-control mean from Team 9's Tier 2 pipeline. Calibration uses the SYN-DIGITS framework of Fan et al. (2026) and its elastic-net specification with the published adaptive-transfer gate (`tau = 0.15`), which keeps the calibrated prediction for 68 of the 221 condition-outcome targets and the uncalibrated digital-twin response for the remaining 153.

## Submission file

- `predictions/team_9_T3_secondary-1_v1.csv` — 208 intervention × outcome ATEs; SHA-256 `518948872a8c759914078980fe26bc1b9765a6c59690ae46dbf625de4f25693e`.
- `registration.md` and `metadata.json` — method and submission metadata.
- `code/calib/` and `code/syn-digits/` — calibration, aggregation, and ATE code.
- `artifacts/` — aligned Wave-4 anchors, raw and calibrated Silicon target matrices, the 40-cell target grid, respondent weights, fit diagnostics, and audit reports.

Run the organizers' validator with `make check`. To regenerate the elastic-net calibration and verify the submitted ATEs from the public matrices, run `PYTHONPATH=code python -m calib.reproduce_submission` in an environment with the listed scientific dependencies. The individual prediction matrices and supporting reproduction materials are available under restricted access at [Zenodo](https://doi.org/10.5281/zenodo.22168937).

Authors: Olivier Toubia, Tianyi Peng, George Gui, Yuchen Qiu, and Naveen Venkat.
