"""Joined respondent-level export.

``simulation_raw_wide.csv`` deliberately holds only what each twin-condition
session produced; the frozen pre-study answers live in their own file.  This
exporter joins the two on ``source_twin_id`` — plus eligibility and repair
metadata — into one respondent-level file, which is the shape a submission
formatter will start from.  Still raw: no calibration, no weighting, no
submission renaming.
"""

from __future__ import annotations

from .common import DATA, read_csv, write_csv


def run_export_joined() -> int:
    simulation = read_csv(DATA / "simulation_raw_wide.csv")
    frozen = {r["persona_id"]: r for r in read_csv(DATA / "prestudy_frozen_wide.csv")}
    eligibility = {r["source_twin_id"]: r for r in read_csv(DATA / "eligibility_audit.csv")}
    repairs: dict[str, list[str]] = {}
    for row in read_csv(DATA / "demographic_repairs.csv"):
        repairs.setdefault(row["source_twin_id"], []).append(
            f"{row['field']}:{row['retry_result']}"
        )

    joined: list[dict] = []
    missing_frozen: list[str] = []
    for row in simulation:
        twin = row["source_twin_id"]
        base = frozen.get(twin)
        if base is None:
            missing_frozen.append(row["profile_id"])
            continue
        record = dict(row)
        for column, value in base.items():
            if column == "persona_id":
                continue
            record[f"pre_{column}"] = value
        meta = eligibility.get(twin, {})
        record["eligibility_status"] = meta.get("eligible", "")
        record["repair_summary"] = ";".join(repairs.get(twin, []))
        joined.append(record)

    if missing_frozen:
        raise SystemExit(
            f"export-joined: {len(missing_frozen)} simulation record(s) have no frozen "
            f"pre-study row (first: {missing_frozen[:3]}) — refusing to write a partial join"
        )

    columns: list[str] = []
    seen: set[str] = set()
    for row in joined:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    write_csv(joined, DATA / "simulation_joined_wide.csv", columns)
    print(f"export-joined: {len(joined)} respondent-condition rows, {len(columns)} columns")
    return 0
