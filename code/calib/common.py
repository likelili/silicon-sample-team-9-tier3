"""Shared plumbing for the SYN-DIGITS calibration of the Silicon Sample runs.

This package deliberately owns almost no numerics.  Everything statistical is
the paper authors' own code, imported from the cloned repository at
``silicon_bench/syn-digits`` (github.com/yw3453/syn-digits, MIT).  What lives
here is the wiring: which twins form the rows, which items form the anchor
columns, and where results land.

Settled design (2026-08-28):
  * fit on the FULL cleaned pool — the 1,921 twins that are structurally
    complete in every one of their 17 silicon conditions;
  * aggregate Tier 2 and Tier 3 over the full clean pool using one fixed set of
    40-cell gender x age x race poststratification weights;
  * calibration feeds Tier 2 / Tier 3 only — Tier 1 ships raw twin responses.

Environment: ``silicon_bench/.venv-calib`` carries the authors' dependency set
(numpy/scipy/sklearn/matplotlib/causaltensor/joblib/pandas); torch is omitted,
so the NN method is out of scope here — it was not among the winners on the
Summary persona construction anyway.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[2]           # .../silicon_bench
SYN_REPO = (BENCH_ROOT / "code" / "syn-digits"
            if (BENCH_ROOT / "code" / "syn-digits").exists()
            else BENCH_ROOT / "syn-digits")
SYN_SRC = SYN_REPO / "src"
W4_DATA = BENCH_ROOT / "wave4" / "data"
ANCHOR_RUN = BENCH_ROOT / "wave4" / "runs" / "20260827-213507_full-anchor"
ANSWER_BLOCKS = BENCH_ROOT / "surveytwin" / "personas" / "mega_persona_json" / "answer_blocks"
STUDY_RUN = BENCH_ROOT / "runs" / "20260827-134528_full-study"
CALIB_ROOT = BENCH_ROOT / "calib"
CALIB_DATA = CALIB_ROOT / "data"

# Published 2K500 settings from the authors' own notebook
# (notebooks/synthetic_control/2K500_experiments.ipynb) — the configuration
# their Table-4 numbers were produced under.  Deviations from these are
# explicit config entries, never silent edits.
PUBLISHED = {
    "imputation_rank": 5,
    "min_col_std": 1.0,
    # New-question column, Twin-2K-500, from the paper's Table 8 (Appendix A.4).
    # NOTE: the shipped notebook is NOT at these settings for three methods —
    # it runs lasso at 0.01 (paper 0.001) and both mc_soft_svd and mc_als at
    # rank=5, lambda=1 (paper rank=20, lambda=20). Table 8 is the record of what
    # produced Table 4, so it is the authority here; the notebook cells look like
    # a left-over scratch state.
    "ridge": {"regularization_multiplier": 100},
    "lasso": {"regularization_multiplier": 0.001},
    "elastic_net": {"regularization_multiplier": 0.01, "en_l1_ratio": 0.3},
    "synthetic_control": {"regularization_multiplier": 1e-6},
    "mc_hard_svd": {"mc_rank": 5, "mc_max_iter": 1000, "mc_tol": 1e-4},
    "mc_soft_svd": {"mc_rank": 20, "mc_lambda": 20, "mc_max_iter": 1000, "mc_tol": 1e-4},
    "mc_als": {"mc_rank": 20, "mc_lambda": 20},
    "synthetic_intervention": {"si_rank": 20, "regularization_multiplier": 100},
    "neural_net": {"nn_hidden_dims": [8], "nn_weight_decay": 0.05, "nn_epochs": 200},
}


def wire_authors() -> None:
    """Put the authors' package on sys.path, headless-safe.

    Their module imports matplotlib at module scope; without a display that
    needs the Agg backend set BEFORE the first import.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    s = str(SYN_SRC)
    if s not in sys.path:
        sys.path.insert(0, s)


def read_csv_dicts(path: Path) -> list[dict]:
    import csv

    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=1, ensure_ascii=False)


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def clean_pool() -> list[str]:
    """The 1,921 twins: eligible minus the 86 structurally incomplete."""
    eligible = [r["base_pid"] for r in read_csv_dicts(STUDY_RUN / "data" / "eligible_twins.csv")]
    excluded = {r["base_pid"] for r in
                read_csv_dicts(STUDY_RUN / "data" / "tier1" / "tier1_exclusions.csv")}
    pool = [p for p in eligible if p not in excluded]
    return sorted(set(pool), key=lambda p: int(p.split("_")[1]))


def tier1_ids() -> tuple[set[str], set[str]]:
    """(panel_500, control_1000) from the frozen Tier-1 selection."""
    portable = BENCH_ROOT / "artifacts" / "tier1_selection.csv"
    source = portable if portable.exists() else STUDY_RUN / "data" / "tier1" / "tier1_selection.csv"
    selection = read_csv_dicts(source)
    panel = {r["base_pid"] for r in selection if r.get("in_panel500") == "yes"}
    control = {r["base_pid"] for r in selection if r.get("condition") == "control"}
    return panel, control


def demographics(pool: list[str]) -> dict[str, dict]:
    """base_pid -> the six moderators, recoded to the OFFICIAL submission labels.

    The frozen pre-study answers carry the survey's *on-screen* wordings, which
    differ from the canonical submission strings for four levels:

        "Black / African-American"  -> "Black / African American"
        "Latino / Hispanic"         -> "Hispanic / Latino"
        "Asian / Asian-American"    -> "Asian / Asian American"
        "Other (please specify)"    -> "Other"          (party)

    ``clean_lib.R`` recodes these via ``.race_map`` / ``.party_map``; skipping it
    silently empties three of the five race levels and one party level — 589 of
    the 1,921 twins — which would drop them from the subgroup guardrail without
    raising anything.  The maps are reused from the exporter rather than
    written a second time.
    """
    import sys as _sys

    pipeline_dir = str(Path(__file__).resolve().parents[1])
    if pipeline_dir not in _sys.path:
        _sys.path.insert(0, pipeline_dir)
    from sbench.tier1_export import (EDU_MAP, GENDER_MAP, INCOME_MAP, PARTY_MAP,
                                     RACE_MAP)

    maps = {"gender": GENDER_MAP, "race": RACE_MAP, "education": EDU_MAP,
            "income": INCOME_MAP, "party": PARTY_MAP}
    portable = BENCH_ROOT / "artifacts" / "prestudy_frozen_wide.csv"
    source = portable if portable.exists() else STUDY_RUN / "data" / "prestudy_frozen_wide.csv"
    wide = {r["persona_id"]: r for r in read_csv_dicts(source)}
    keep = ("gender", "age_band", "race", "education", "income", "party")

    out: dict[str, dict] = {}
    for pid in pool:
        row = wide.get(pid)
        if row is None:
            continue
        record = {}
        for key in keep:
            value = (row.get(key) or "").strip()
            mapping = maps.get(key)
            # a value already canonical passes through; only survey wordings map
            record[key] = mapping.get(value, value) if mapping else value
        out[pid] = record
    return out


def anchor_items() -> list[str]:
    """The 123 anchor item ids, in the verified reference order.

    The order is the column_map's key order — the same order `syndigits-export`
    writes — so every matrix built here shares columns with the exported
    ``real.csv``/``LLM.csv`` pair.
    """
    verified = load_json(W4_DATA / "reference_column_map.json")
    if not verified.get("ok"):
        raise SystemExit("reference column map is not fully verified — refusing to build matrices")
    return list(verified["column_map"].keys())


def anchor_matrices(pool: list[str]):
    """(Y, Ytilde) as float arrays over ``pool`` rows x 123 anchor columns.

    Y      = the humans' waves 1-3 answers to the Wave-4 item set — the same
             matrix the published results used as ``real.csv``.
    Ytilde = our Luna twins' Wave-4 answers.
    NaN    = the variant the person was not assigned (structural), or the rare
             unmatched answer (0.12% on the twin side).
    """
    import numpy as np

    items = anchor_items()
    human = {r["base_pid"]: r for r in
             read_csv_dicts(W4_DATA / "human_answers_wave1_3_wide.csv")}
    twin = {r["base_pid"]: r for r in
            read_csv_dicts(ANCHOR_RUN / "data" / "twin_wave4_wide.csv")}

    def cell(row: dict | None, item: str) -> float:
        value = ((row or {}).get(item) or "").strip()
        return float(value) if value else float("nan")

    Y = np.array([[cell(human.get(p), i) for i in items] for p in pool], dtype=float)
    T = np.array([[cell(twin.get(p), i) for i in items] for p in pool], dtype=float)
    return Y, T, items
