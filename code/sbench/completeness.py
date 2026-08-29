"""Per-session completeness verdicts, decided at the question-ID SET level.

"Exactly complete" here means the adopted response's set of question ids
EQUALS the set asked — not merely that the counts match. Count equality can
hide a swap (one asked id missing, one invented id present), and the Tier-1
stratification excludes on exactness, so the verdict has to be set-based.

Asked ids are recovered from the submitted request bodies themselves (the
questions JSON inside each user message) and validated line-by-line against
the render plans' ``n_questions``; any mismatch aborts the build rather than
guessing.

Adoption follows the repair phase's "more complete wins" rule, made precise:
among all responses for a session (original, real-time retry, repair re-run),
adopt the one maximizing answered-of-asked, breaking ties by fewest invented
ids. The adopted source file is recorded so downstream assembly can adopt the
identical response.

Practical planarian is exact only if BOTH stages are exact. A planarian
session whose round-2 response has not arrived yet is ``pending`` — a distinct
state from incomplete, because it may still become exact; Tier-1 refuses to
stratify a condition containing pending sessions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .common import AUDIT, DATA, read_csv, save_json, sha256_file, write_csv

PLANARIAN = "practical planarian"
_QID_PATTERN = re.compile(r'\{"id": "([^"]+)"')


def _asked_ids_by_stage() -> dict[tuple[str, int], list[str]]:
    """(run_id, stage) -> asked question ids, validated against the render plans."""
    plans = {
        1: {r["run_id"]: int(r["n_questions"])
            for r in read_csv(AUDIT / "batch" / "render_plan.csv")},
    }
    sources = {1: sorted((DATA / "batch" / "round1").glob("requests_*.jsonl"))}
    round2_plan = AUDIT / "batch" / "render_plan_round2.csv"
    if round2_plan.exists():
        plans[2] = {r["run_id"]: int(r["n_questions"]) for r in read_csv(round2_plan)}
        sources[2] = sorted((DATA / "batch" / "round2").glob("requests*.jsonl"))

    asked: dict[tuple[str, int], list[str]] = {}
    for stage, files in sources.items():
        for path in files:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    run_id = obj["custom_id"]
                    ids = _QID_PATTERN.findall(obj["body"]["messages"][1]["content"])
                    want = plans[stage].get(run_id)
                    if want is None or len(ids) != want or len(set(ids)) != want:
                        raise SystemExit(
                            f"completeness: asked-id extraction failed for {run_id} "
                            f"stage {stage}: {len(ids)} ids ({len(set(ids))} unique) "
                            f"vs plan {want}")
                    asked[(run_id, stage)] = ids
    return asked


def _delivered(obj: dict) -> set[str] | None:
    """Question-id set delivered by one batch response line; None if unusable."""
    response = obj.get("response") or {}
    if response.get("status_code") != 200:
        return None
    choice = ((response.get("body") or {}).get("choices") or [{}])[0]
    if choice.get("finish_reason") != "stop":
        return None
    try:
        parsed = json.loads(choice["message"]["content"])
        answers = parsed.get("answers") if isinstance(parsed, dict) else parsed
        return {str(a.get("question_id")) for a in answers if isinstance(a, dict)}
    except Exception:
        return None


def _best_by_run(results_dir: Path, asked: dict, stage: int) -> dict[str, dict]:
    """run_id -> adopted response summary for one stage's results directory."""
    best: dict[str, dict] = {}
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = obj.get("custom_id")
            if not run_id or (run_id, stage) not in asked:
                continue
            delivered = _delivered(obj)
            asked_set = set(asked[(run_id, stage)])
            if delivered is None:
                score = (-1, -1)
                answered = extra = 0
            else:
                answered = len(delivered & asked_set)
                extra = len(delivered - asked_set)
                score = (answered, -extra)
            current = best.get(run_id)
            if current is None or score > current["score"]:
                best[run_id] = {"score": score, "answered": answered, "extra": extra,
                                "asked": len(asked_set), "source": path.name}
    return best


def build_completeness() -> int:
    plan_rows = read_csv(AUDIT / "batch" / "render_plan.csv")
    asked = _asked_ids_by_stage()
    best1 = _best_by_run(DATA / "batch" / "round1_results", asked, stage=1)
    round2_dir = DATA / "batch" / "round2_results"
    best2 = _best_by_run(round2_dir, asked, stage=2) if round2_dir.is_dir() else {}

    rows = []
    tally = {"exact": 0, "incomplete": 0, "pending": 0}
    for plan in plan_rows:
        run_id = plan["run_id"]
        stage1 = best1.get(run_id)
        if stage1 is None:
            raise SystemExit(f"completeness: no round-1 response at all for {run_id}")
        exact1 = stage1["answered"] == stage1["asked"] and stage1["extra"] == 0
        record = {
            "run_id": run_id,
            "base_pid": plan["base_pid"],
            "condition": plan["condition"],
            "raw_condition": plan["raw_condition"],
            "control_variant": plan.get("control_variant", ""),
            "s1_asked": stage1["asked"], "s1_answered": stage1["answered"],
            "s1_extra": stage1["extra"], "s1_source": stage1["source"],
            "s2_asked": "", "s2_answered": "", "s2_extra": "", "s2_source": "",
        }
        if plan["raw_condition"] == PLANARIAN:
            stage2 = best2.get(run_id)
            if (run_id, 2) not in asked or stage2 is None:
                record["status"] = "pending"
            else:
                exact2 = stage2["answered"] == stage2["asked"] and stage2["extra"] == 0
                record.update({"s2_asked": stage2["asked"],
                               "s2_answered": stage2["answered"],
                               "s2_extra": stage2["extra"],
                               "s2_source": stage2["source"]})
                record["status"] = "exact" if (exact1 and exact2) else "incomplete"
        else:
            record["status"] = "exact" if exact1 else "incomplete"
        tally[record["status"]] += 1
        rows.append(record)

    rows.sort(key=lambda r: (r["raw_condition"], r["run_id"]))
    out = AUDIT / "batch" / "completeness_by_session.csv"
    write_csv(rows, out, list(rows[0].keys()))
    summary = {
        "sessions": len(rows),
        "tally": tally,
        "definition": "exact = adopted response's question-id SET equals the asked set, "
                      "every stage; planarian needs both stages; pending = round-2 "
                      "response not yet collected",
        "adoption": "max answered-of-asked, tie-break fewest invented ids; source "
                    "file recorded per stage",
        "completeness_csv_sha256": sha256_file(out),
    }
    save_json(summary, AUDIT / "batch" / "completeness_summary.json")
    print(f"completeness: {len(rows):,} sessions -> {tally}")
    by_cond: dict[str, int] = {}
    for r in rows:
        if r["status"] != "exact":
            by_cond[r["raw_condition"]] = by_cond.get(r["raw_condition"], 0) + 1
    for cond, n in sorted(by_cond.items()):
        print(f"  not-exact in {cond}: {n}")
    return 0
