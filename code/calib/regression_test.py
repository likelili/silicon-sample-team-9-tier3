"""Regression test: our 13 outcomes must equal the official R cleaning output.

``clean_lib.R`` is the source of record for the scored outcomes, and this pipeline
reimplements it in Python.  A reimplementation that drifts is worse than no
reimplementation, because the drift is silent — a composite built in the wrong
order, a reversal applied twice, an ``na.rm`` rule changed — none of it raises.

So the Python construction is checked cell by cell against the R-derived
reference (``tier1_predictions.csv``) over **every** clean session the reference
covers, not a sample.  The comparison keys on ``(base_pid, canonical condition)``
after stripping the reference's per-condition ``profile_id`` prefix.

Run it after any change to ``outcomes.py``, and before any production fit:

    python -m calib regression-test
"""

from __future__ import annotations

import math
import re

from .common import CALIB_DATA, STUDY_RUN, read_csv_dicts, save_json
from .dryrun import _label_map, session_items
from .outcomes import CODENAMES, OUTCOMES

TOLERANCE = 1e-6


def run(limit: int | None = None) -> int:
    labels = _label_map()

    reference: dict[tuple[str, str], dict] = {}
    for row in read_csv_dicts(STUDY_RUN / "data" / "tier1" / "tier1_predictions.csv"):
        match = re.search(r"(pid_\d+)", row["profile_id"])
        if match:
            reference[(match.group(1), row["condition"])] = row

    completeness = read_csv_dicts(
        STUDY_RUN / "audit" / "batch" / "completeness_by_session.csv")
    by_key = {(r["base_pid"],
               "control" if r["condition"] == "control" else r["raw_condition"]): r
              for r in completeness}

    selection = read_csv_dicts(STUDY_RUN / "data" / "tier1" / "tier1_selection.csv")
    if limit:
        selection = selection[:limit]

    compared = mismatched = skipped = 0
    worst: list[dict] = []
    sessions = 0
    for pick in selection:
        session = by_key.get((pick["base_pid"], pick["condition"]))
        if session is None or session["status"] != "exact":
            skipped += 1
            continue
        canonical = CODENAMES.get(session["raw_condition"].strip(),
                                  session["raw_condition"])
        expected = reference.get((pick["base_pid"], canonical))
        if expected is None:
            skipped += 1
            continue
        built = __import__("calib.outcomes", fromlist=["build_outcomes"]).build_outcomes(
            session_items(session["run_id"], session["s1_source"],
                          session["s2_source"], labels))
        sessions += 1
        for outcome in OUTCOMES:
            raw = (expected.get(outcome) or "").strip()
            if not raw:
                continue
            compared += 1
            theirs, ours = float(raw), built[outcome]
            if math.isnan(ours) or abs(theirs - ours) > TOLERANCE:
                mismatched += 1
                worst.append({"base_pid": pick["base_pid"], "condition": canonical,
                              "outcome": outcome, "official_R": theirs,
                              "ours": None if math.isnan(ours) else ours,
                              "abs_diff": (math.inf if math.isnan(ours)
                                           else abs(theirs - ours))})
        if sessions % 1000 == 0:
            print(f"  {sessions} sessions, {compared:,} cells, "
                  f"{mismatched} mismatch(es)", flush=True)

    worst.sort(key=lambda d: -d["abs_diff"])
    report = {
        "reference": "tier1_predictions.csv (derived via the official clean_lib.R schema)",
        "tolerance": TOLERANCE,
        "sessions_compared": sessions,
        "cells_compared": compared,
        "mismatches": mismatched,
        "sessions_skipped": skipped,
        "worst": worst[:20],
        "passed": mismatched == 0,
    }
    CALIB_DATA.mkdir(parents=True, exist_ok=True)
    save_json(report, CALIB_DATA / "regression_official_r.json")

    print(f"\nregression test: {sessions:,} sessions, {compared:,} cells, "
          f"{mismatched} mismatch(es), {skipped} skipped")
    if mismatched:
        for entry in worst[:5]:
            print(f"  {entry['base_pid']} {entry['condition']} {entry['outcome']}: "
                  f"R={entry['official_R']} ours={entry['ours']} "
                  f"diff={entry['abs_diff']}")
        print("FAIL — the Python construction has drifted from clean_lib.R")
        return 1
    print("PASS — every cell matches the official R output")
    return 0
