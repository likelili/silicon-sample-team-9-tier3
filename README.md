# Team 9 — Silicon Sample Benchmark Tier 3 Submission

This repository contains Team 9's `secondary-1` Tier 3 submission. Each ATE is the organizer-specified difference between the calibrated intervention mean and calibrated pooled-control mean from Team 9's Tier 2 pipeline.

## Submission file

- `predictions/team_9_T3_secondary-1_v1.csv` — 208 intervention × outcome ATEs; SHA-256 `68a8dba73c29a4c0d9071dc4b3a6c09b3e48bb270f9d156f596b20c6bb96f65f`.
- `registration.md` and `metadata.json` — method and submission metadata.
- `code/calib/` and `code/syn-digits/` — calibration, aggregation, and ATE code.
- `artifacts/` — aligned Wave-4 anchors, raw and calibrated Silicon target matrices, fit diagnostics, and audit reports.

Run the organizers' validator with `make check`. The simulation request and response archive is available under restricted access at [Zenodo](https://doi.org/10.5281/zenodo.22150315).

Authors: Olivier Toubia, Tianyi Peng, George Gui, Yuchen Qiu, and Naveen Venkat. Corresponding contact: Olivier Toubia (`ot2107@gsb.columbia.edu`).
