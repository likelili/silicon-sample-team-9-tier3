"""Out-of-domain test of the transfer gate, on the 19-megastudy results.

The Wave-4 evaluation cannot answer the question that matters for production.
There, donors and target are both Wave-4 items — same instrument, same scales,
same session. The Silicon targets are a different experiment entirely, and a
transfer gate is supposed to earn its keep precisely when the target is unlike
the donors. So a gate that looks useless within Wave-4 might still help
out-of-domain, and the reverse.

The megastudies give that test. They are experiments on the **same Twin-2K
panel**, with human answers known, so the full chain can be scored:

    DT Wave-4 anchors  ->  DT megastudy outcome     (fit)
    human Wave-4 anchors ->  predicted human outcome (transfer)
    versus the ACTUAL human megastudy outcome        (score)

which is structurally what production does, with a target whose human values we
can check.

Three caveats, recorded because they bound what this can conclude:

1. **Different simulator.** These runs are ``persona_type=full`` with
   gpt-5-mini-medium; production is ``summary`` with gpt-5.6-luna. The mechanism
   question — does ``train_mse`` mark where out-of-domain transfer works —
   carries over; a tau *value* tuned here would not.
2. **Smaller N.** Only 300 twins were simulated, so after intersecting with
   participating humans and the clean pool, roughly 130-195 rows per study
   against 123 donors. Noisier than production's 1,921.
3. **Read-only.** Nothing in the megastudy or validation folders is written.

Targets are **aggregated constructs**, not raw items: a multi-item numeric scale
enters as the mean of its items, which is the measurement the study reports and
matches how the Silicon outcomes are built (composites before calibration).
Non-numeric items are excluded — the outcome objects here are numeric scales.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(os.environ.get("MEGASTUDY_HUMAN_DIR", "external/19_DT_experiments"))
VAL = Path(os.environ.get("MEGASTUDY_TWIN_DIR", "external/validation"))

# validation folder name -> megastudy folder name, where they differ
FOLDER_ALIAS = {
    "preference_redistribution": "preferences_for_redistribution",
    "affective_priming": "affective_primes",
    "digital_certification": "digital_certifications_for_luxury_consumption",
    "infotainment": "infotainment_news_sharing",
    "default_eric": "defaults",
}

MIN_SCALE_ROWS = 60          # a construct needs enough joined respondents to score
MIN_DISTINCT = 3             # and enough distinct values to be a scale, not a flag


def _qualtrics(path: Path) -> tuple[list[str], dict[str, str], list[list[str]]]:
    """Read a Qualtrics-style export: names / question text / ImportId / data.

    Both the megastudy human export and the twin export carry a three-row
    header. ``ImportId`` is the join key rather than the column NAME: the two
    exports agree on 63 ImportIds but only 12 names for quantitative_intuition,
    because the twin export re-labels columns.
    """
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    names = rows[0]
    import_of: dict[str, str] = {}
    for name, cell in zip(names, rows[2] if len(rows) > 2 else []):
        match = re.search(r'"ImportId"\s*:\s*"([^"]+)"', cell or "")
        if match:
            import_of[match.group(1)] = name
    return names, import_of, rows[3:]


def _numeric(value) -> float:
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def load_study(val_name: str) -> dict | None:
    """Join one study's human and twin outcomes onto shared Twin-2K ids."""
    exp_dir = EXP / FOLDER_ALIAS.get(val_name, val_name)
    val_dir = VAL / val_name
    human_path = exp_dir / "human_values.csv"
    twin_path = val_dir / "responses_values.csv"      # NUMERIC codes, not labels
    if not human_path.is_file() or not twin_path.is_file():
        return None

    h_names, h_imp, h_rows = _qualtrics(human_path)
    t_names, t_imp, t_rows = _qualtrics(twin_path)

    # TWIN_ID is an embedded field: the human export carries it as a column NAME
    # with no ImportId entry, while the twin export has both. Look up by name
    # first and fall back to the ImportId map.
    def id_index(names, imp):
        if "TWIN_ID" in names:
            return names.index("TWIN_ID")
        return names.index(imp["TWIN_ID"]) if "TWIN_ID" in imp else None

    h_id, t_id = id_index(h_names, h_imp), id_index(t_names, t_imp)
    if h_id is None or t_id is None:
        return None

    def pid(value: str) -> str | None:
        value = value.strip()
        if value.startswith("pid_"):
            return value
        return f"pid_{int(float(value))}" if value.replace(".", "").isdigit() else None

    human_by_pid = {p: dict(zip(h_names, r)) for r in h_rows
                    if (p := pid(r[h_id]))}
    twin_by_pid = {p: dict(zip(t_names, r)) for r in t_rows
                   if (p := pid(r[t_id]))}
    shared = sorted(set(human_by_pid) & set(twin_by_pid),
                    key=lambda p: int(p.split("_")[1]))
    if len(shared) < MIN_SCALE_ROWS:
        return None

    # Join on ImportId, excluding metadata and demographics.
    skip = re.compile(r"^(startDate|endDate|progress|duration|finished|recordedDate|"
                      r"_recordId|responseId|TWIN_ID|demo_|Status|IPAddress|"
                      r"RecipientEmail|RecipientFirstName|RecipientLastName|"
                      r"ExternalReference|LocationLatitude|LocationLongitude|"
                      r"DistributionChannel|UserLanguage|Consent)", re.I)
    pairs = [(t_imp[k], h_imp[k]) for k in (set(t_imp) & set(h_imp)) if not skip.match(k)]

    # Group items into constructs by their parent question, so a multi-item
    # numeric scale enters as ONE target at its mean — the reported measurement.
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for twin_col, human_col in pairs:
        groups[re.sub(r"[_#]\d+$", "", twin_col)].append((twin_col, human_col))

    constructs = []
    for parent, members in sorted(groups.items()):
        tv, hv = [], []
        for p in shared:
            t = [v for v in (_numeric(twin_by_pid[p][c]) for c, _ in members)
                 if not math.isnan(v)]
            h = [v for v in (_numeric(human_by_pid[p][c]) for _, c in members)
                 if not math.isnan(v)]
            tv.append(sum(t) / len(t) if t else math.nan)
            hv.append(sum(h) / len(h) if h else math.nan)
        tv, hv = np.array(tv), np.array(hv)
        both = np.isfinite(tv) & np.isfinite(hv)
        if both.sum() < MIN_SCALE_ROWS:
            continue
        if len(set(np.round(hv[both], 6))) < MIN_DISTINCT:
            continue                                   # a flag, not a numeric scale
        span = float(np.nanmax(hv[both]) - np.nanmin(hv[both]))
        if span <= 0:
            continue
        constructs.append({
            "study": val_name, "construct": f"{val_name}:{parent}",
            "parent": parent, "n_items": len(members),
            "twin": tv, "human": hv, "span": span, "n_joined": int(both.sum()),
        })
    return {"study": val_name, "pids": shared, "constructs": constructs} if constructs else None


def run(studies: tuple[str, ...], tau_grid=None) -> dict:
    """Fit the frozen elastic-net spec on DT Wave-4 -> DT construct, transfer to
    human Wave-4, and score against the ACTUAL human construct.

    This is the production chain with a checkable answer, so it can test both
    whether calibration helps out-of-domain and whether ``train_mse`` marks
    where it helps.
    """
    from . import authors
    from .common import anchor_matrices, clean_pool
    from .published import FIT_PARAMS, SPEC

    pool = clean_pool()
    Yw, Tw, _ = anchor_matrices(pool)          # human / DT Wave-4 anchors
    index_of = {p: i for i, p in enumerate(pool)}

    rows = []
    for study in studies:
        loaded = load_study(study)
        if not loaded:
            continue
        keep = [(k, index_of[p]) for k, p in enumerate(loaded["pids"])
                if p in index_of]
        if len(keep) < MIN_SCALE_ROWS:
            continue
        local = [k for k, _ in keep]
        anchor_rows = [i for _, i in keep]

        Y_anchor = Yw[anchor_rows, :]          # human anchors, aligned
        T_anchor = Tw[anchor_rows, :]          # DT anchors, same people

        for c in loaded["constructs"]:
            twin_t = c["twin"][local]
            human_t = c["human"][local]
            # append the target as one extra column, exactly as production does
            real = np.column_stack([Y_anchor, np.full(len(anchor_rows), np.nan)])
            synth = np.column_stack([T_anchor, twin_t])
            target = real.shape[1] - 1
            sc = authors.build(real, synth, name="mega",
                               imputation_rank=SPEC["imputation_rank"],
                               min_col_std=SPEC["min_col_std"])
            try:
                pred, train_mse, info = authors.predict_column(
                    sc, target, method=SPEC["method"], fit_finite_only=True,
                    **FIT_PARAMS)
                err = ""
            except Exception as exc:
                pred, train_mse, info = np.full(len(anchor_rows), np.nan), math.nan, {}
                err = f"{type(exc).__name__}: {exc}"[:120]

            valid = np.isfinite(human_t) & np.isfinite(twin_t) & np.isfinite(pred)
            if valid.sum() < MIN_SCALE_ROWS:
                continue
            span = c["span"]
            raw_ae = abs(float(twin_t[valid].mean() - human_t[valid].mean())) / span * 100
            cal_ae = abs(float(pred[valid].mean() - human_t[valid].mean())) / span * 100
            rows.append({
                "construct": c["construct"], "study": study,
                "n_items": c["n_items"], "n": int(valid.sum()),
                "train_mse": train_mse, "n_fit_rows": info.get("n_fit_rows"),
                "raw_panel_ae": raw_ae, "cal_panel_ae": cal_ae,
                "benefit": cal_ae - raw_ae, "error": err,
            })
    return {"rows": rows, "n_constructs": len(rows)}
