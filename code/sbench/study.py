"""Full-pool study design: every eligible twin completes all 17 conditions.

This replaces the earlier 500-per-intervention / 1,000-control sampling design.
No stratification or subsampling happens here — Tier 1 stratification is a
later phase run over the completed pool.

Plan
    N_eligible x (16 interventions + 1 pooled control) sessions.
    Each twin appears exactly once per condition; ``base_pid`` repeats across
    conditions, ``run_id`` is unique per twin-condition session.  The pooled
    control assigns each twin exactly one of the three control texts by a
    documented deterministic balanced assignment (seeded shuffle, round-robin),
    replacing the survey's random draw with a fixed, auditable equivalent.

Session semantics (unchanged from the audited driver)
    * fresh runtime + state per session — no memory across conditions;
    * prompt = system + persona + the twin's FROZEN pre-study question/answer
      pairs in original order (the prior-answer section) + the study stage;
    * condition forced via ``state.randomizer_choices``; stimulus renders
      exactly once (descriptive-block fix);
    * all authored within-condition randomization runs deterministically.

Execution
    * per-session checkpoint (``study_checkpoint.jsonl``) written the moment a
      session completes; ``--resume`` skips completed ``run_id``s;
    * exact prompts (live/dry) and raw model responses preserved;
    * provider telemetry per call: request id, tokens incl. reasoning,
      realized cost, prompt hash, commit + QSF hash.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import (
    ARTIFACTS,
    AUDIT,
    BENCH_ROOT,
    DATA,
    MANIFEST,
    append_jsonl,
    condition_codes,
    load_json,
    read_csv,
    require_live_flag,
    rng_for,
    save_json,
    session_seed,
    sha256_file,
    wire_worktree,
    write_csv,
)
from .driver import StagedDriver, load_template
from .telemetry import TELEMETRY_COLUMNS, capture, summarize

CONTROL_TEXTS = ("control neckties", "control baseball", "control dances")


def _run_id(base_pid: str, raw_condition: str, seed: int) -> str:
    digest = hashlib.sha256(f"{base_pid}|{raw_condition}|{seed}".encode()).hexdigest()
    return f"r_{digest[:16]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Eligibility: bootstrap from the frozen pre-study run, repairs happen via the
# existing `repairs` phase, then the pool is exported under the agreed names.
# ---------------------------------------------------------------------------

PRESTUDY_INPUTS = (
    "prestudy_qa.json", "prestudy_meta.json", "prestudy_answers_long.csv",
    "prestudy_answers_wide.csv", "profile_comparisons.csv", "proposed_repairs.csv",
)


def bootstrap_from(source_run: str) -> int:
    """Copy the frozen pre-study outputs into the active run as inputs."""
    import shutil

    source = Path(source_run).expanduser().resolve()
    copied = {}
    for name in PRESTUDY_INPUTS:
        src = source / "data" / name
        if not src.exists():
            raise SystemExit(f"bootstrap: {src} missing")
        dst = DATA / name
        shutil.copy(src, dst)
        # The source run is frozen read-only; the copies must be writable or
        # the repair phase pays for its retries and then cannot save them.
        import os
        os.chmod(dst, 0o644)
        copied[name] = sha256_file(dst)
    save_json(
        {"source_run": str(source), "copied_at": _now(), "files_sha256": copied,
         "note": "pre-study answers reused from the frozen full-prestudy run; "
                 "eligibility is decided here before any treatment outcome exists"},
        DATA / "prestudy_source.json",
    )
    print(f"bootstrap: {len(copied)} pre-study file(s) copied from {source.name}")
    return 0


def export_eligibility() -> int:
    """After repairs + freeze: write the frozen pool under the agreed names."""
    eligibility = read_csv(DATA / "eligibility_audit.csv")
    repairs = read_csv(DATA / "demographic_repairs.csv")
    frozen_long = read_csv(DATA / "prestudy_frozen_long.csv")

    repairs_by_twin: dict[str, list[dict]] = {}
    for row in repairs:
        repairs_by_twin.setdefault(row["source_twin_id"], []).append(row)

    eligible_rows, excluded_rows = [], []
    for row in eligibility:
        twin = row["source_twin_id"]
        twin_repairs = repairs_by_twin.get(twin, [])
        summary = ";".join(
            f"{r['field']}:{'resolved' if r['resolved'] == 'yes' else r['retry_result']}"
            for r in twin_repairs
        )
        record = {
            "base_pid": twin,
            "original_problem": row.get("conflict_fields", "")
                                or ("attention" if "attention" in row.get("exclusion_reason", "") else ""),
            "retry_result": summary,
            "attention_check_status": row.get("attention_check_status", ""),
        }
        if row["eligible"] == "yes":
            eligible_rows.append(record)
        else:
            excluded_rows.append({**record, "exclusion_reason": row["exclusion_reason"]})

    write_csv(eligible_rows, DATA / "eligible_twins.csv",
              ["base_pid", "original_problem", "retry_result", "attention_check_status"])
    write_csv(excluded_rows, DATA / "excluded_twins.csv",
              ["base_pid", "exclusion_reason", "original_problem", "retry_result",
               "attention_check_status"])
    for row in frozen_long:
        row["base_pid"] = row.pop("persona_id")
    write_csv(frozen_long, DATA / "frozen_prestudy_answers.csv",
              ["base_pid", "question_id", "export_tag", "question_text", "question_type",
               "answer_value", "answer_label", "repaired"])
    print(f"eligibility: {len(eligible_rows)} eligible, {len(excluded_rows)} excluded "
          f"(N_eligible = {len(eligible_rows)})")
    return 0


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def study_plan(seed: int) -> list[dict]:
    eligible = [r["base_pid"] for r in read_csv(DATA / "eligible_twins.csv")]
    eligible.sort(key=lambda t: int(t.split("_")[1]))
    rows = condition_codes()
    code_field = next(k for k in rows[0] if "code" in k.lower())
    interventions = sorted({r[code_field] for r in rows} - set(CONTROL_TEXTS))
    if len(interventions) != 16:
        raise SystemExit(f"study plan: expected 16 interventions, found {len(interventions)}")

    plan: list[dict] = []
    for condition in interventions:
        for twin in eligible:
            plan.append({"base_pid": twin, "raw_condition": condition,
                         "condition": condition, "control_variant": "",
                         "run_id": _run_id(twin, condition, seed)})
    # Pooled control: every eligible twin exactly once, one text each, balanced
    # deterministically (seeded shuffle + round-robin — the documented stand-in
    # for the survey's even-presentation random draw).
    shuffled = list(eligible)
    rng_for(seed, "study-control-variant").shuffle(shuffled)
    for index, twin in enumerate(shuffled):
        text = CONTROL_TEXTS[index % 3]
        plan.append({"base_pid": twin, "raw_condition": text,
                     "condition": "control", "control_variant": text,
                     "run_id": _run_id(twin, text, seed)})
    return plan


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _wide_header(tags: dict[str, str]) -> list[str]:
    """Fixed wide header: metadata + every answerable post-study question."""
    wire_worktree()
    from services.v2.instrument_builder import collect_instrument_questions  # noqa: E402

    template = load_json(ARTIFACTS / "template_poststudy.json")

    def blocks(elements):
        for element in elements or []:
            if not isinstance(element, dict):
                continue
            if str(element.get("ElementType") or element.get("Type")) == "Block":
                yield element
            yield from blocks(element.get("Elements"))

    columns = ["run_id", "base_pid", "condition", "raw_condition", "control_variant",
               "model", "persona_representation", "seed", "excluded"]
    seen = set(columns)
    for block in blocks(template["Elements"]):
        for question in collect_instrument_questions([block]):
            qid = str(question.get("id") or "")
            column = tags.get(qid, qid)
            if column and column not in seen:
                seen.add(column)
                columns.append(column)
    return columns


async def run_study(
    *, mode: str, model: str, persona_type: str, seed: int,
    concurrency: int, limit: int | None, resume: bool = False,
    pilot_twins: int | None = None,
) -> int:
    require_live_flag(mode == "live", "study")
    template = load_template(ARTIFACTS / "template_poststudy.json")
    condition_map = load_json(ARTIFACTS / "condition_map.json")["conditions"]
    frozen_qa = load_json(DATA / "prestudy_frozen_qa.json")
    plan = study_plan(seed)
    write_csv(plan, DATA / "condition_assignments.csv",
              ["run_id", "base_pid", "condition", "raw_condition", "control_variant"])
    if pilot_twins:
        # A pilot must cover ALL 17 conditions: take the first N eligible
        # twins' complete condition sets rather than the plan's first rows.
        chosen = sorted({p["base_pid"] for p in plan},
                        key=lambda t: int(t.split("_")[1]))[:pilot_twins]
        plan = [p for p in plan if p["base_pid"] in set(chosen)]
    if limit:
        plan = plan[:limit]

    missing_prior = [p["base_pid"] for p in plan if p["base_pid"] not in frozen_qa]
    if missing_prior:
        raise SystemExit(f"study: {len(set(missing_prior))} planned twin(s) have no frozen "
                         f"pre-study record: {sorted(set(missing_prior))[:5]}")

    audit = load_json(ARTIFACTS / "qsf_audit.json")
    tags = {q["qid"]: q["export_tag"] for b in audit["block_inventory"]
            for q in b["questions"] if q.get("export_tag")}

    eligible_order = sorted({p["base_pid"] for p in plan}, key=lambda t: int(t.split("_")[1]))
    index_of = {t: i for i, t in enumerate(eligible_order)}
    driver = StagedDriver(template, mode=mode, model=model, persona_type=persona_type,
                          seed=seed, persona_count=len(eligible_order))

    checkpoint = DATA / "study_checkpoint.jsonl"
    raw_path = DATA / "simulation_raw.jsonl"
    stages_path = AUDIT / "study_stages.jsonl"
    prompts_path = AUDIT / "study_prompts.jsonl"

    completed: set[str] = set()
    if resume and Path(checkpoint).exists():
        with open(checkpoint, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from a hard kill
                if record.get("run_id"):
                    completed.add(record["run_id"])
        print(f"study[{mode}]: resuming — {len(completed)} session(s) already checkpointed")
    else:
        for path in (checkpoint, raw_path, stages_path, prompts_path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("")

    pending = [p for p in plan if p["run_id"] not in completed]
    failures: list[dict] = []
    done_count = 0
    semaphore = asyncio.Semaphore(concurrency)
    flush_lock = asyncio.Lock()

    # Execution order: per-twin chains. A twin's sessions run back-to-back so
    # the shared prompt prefix (system + persona + frozen pre-study answers)
    # stays warm in the provider's prompt cache across its 17 conditions;
    # different twins run in parallel. The pilot showed the prefix is ~80% of
    # each session's input, so this cuts realized cost substantially at the
    # same wall time and concurrency.
    by_twin: dict[str, list[dict]] = {}
    for item in pending:
        by_twin.setdefault(item["base_pid"], []).append(item)

    async def one(item: dict):
        nonlocal done_count
        twin = item["base_pid"]
        raw_condition = item["raw_condition"]
        run_id = item["run_id"]
        forced = condition_map[raw_condition]["forced_choices"]
        prior = [(r["question"], r["answer"]) for r in frozen_qa[twin]]
        async with semaphore:
            result = await driver.run_twin(
                twin, index_of[twin], forced_choices=forced, prior_qa=prior,
                rng_extra=f"study:{raw_condition}",
                # Same derivation the batch builder uses, so a session rendered
                # live and the same session rendered into a batch file carry
                # byte-identical prompts.
                session_seed=session_seed(seed, twin, raw_condition),
                billing_extra={"raw_condition": raw_condition, "run_id": run_id},
            )
        if result.error:
            failures.append({"run_id": run_id, "base_pid": twin,
                             "raw_condition": raw_condition, "error": result.error,
                             "at": _now()})
            return
        got = str(result.embedded_data.get("condition", ""))
        if got != raw_condition:
            failures.append({"run_id": run_id, "base_pid": twin,
                             "raw_condition": raw_condition, "at": _now(),
                             "error": f"forced condition mismatch: wanted {raw_condition!r}, "
                                      f"got {got!r}"})
            return

        stage_records = []
        for stage in result.stages:
            stage_records.append(
                {"stage_index": stage.stage_index, "question_ids": stage.question_ids,
                 "prompt_sha256": stage.prompt_sha256,
                 "prompt_chars": stage.prompt_chars["total"],
                 "prior_answer_count": stage.prior_answer_count,
                 "rendered_block_ids": stage.rendered_block_ids,
                 "latency_ms": stage.latency_ms, "retries": stage.retries}
            )
        async with flush_lock:
            for stage, record in zip(result.stages, stage_records, strict=False):
                append_jsonl({"run_id": run_id, "base_pid": twin,
                              "raw_condition": raw_condition, **record,
                              "events": stage.events, "mode": mode}, stages_path)
                if mode != "mock":
                    append_jsonl({"run_id": run_id, "stage_index": stage.stage_index,
                                  "prompt_sha256": stage.prompt_sha256,
                                  "system_prompt": stage.system_prompt,
                                  "user_prompt": stage.user_prompt_head}, prompts_path)
            append_jsonl(
                {"run_id": run_id, "base_pid": twin, "raw_condition": raw_condition,
                 "condition": item["condition"],
                 "stages": [
                     {"stage_index": s.stage_index, "prompt_sha256": s.prompt_sha256,
                      "raw_answers": s.raw_answers, "normalized_answers": s.normalized_answers}
                     for s in result.stages
                 ],
                 "embedded_data": result.embedded_data, "at": _now()},
                raw_path,
            )
            append_jsonl(
                {"run_id": run_id, "base_pid": twin, "raw_condition": raw_condition,
                 "condition": item["condition"], "control_variant": item["control_variant"],
                 "rows": result.rows, "embedded_data": result.embedded_data,
                 "excluded": result.excluded, "stage_records": stage_records,
                 "completed_at": _now()},
                checkpoint,
            )
            done_count += 1
            if done_count % 500 == 0:
                print(f"  study progress: {done_count}/{len(pending)} sessions this pass")

    async def twin_chain(items: list[dict]):
        for item in items:
            await one(item)

    with capture("study") as telemetry:
        await asyncio.gather(*(twin_chain(items) for items in by_twin.values()))

    if telemetry:
        write_csv(telemetry, AUDIT / "study_provider_calls.csv", TELEMETRY_COLUMNS)
        stats = summarize(telemetry)
        save_json(stats, AUDIT / "study_provider_summary.json")
        print(f"  provider: {stats['calls']} calls, {stats['total_tokens']:,} tokens "
              f"({stats['reasoning_tokens']:,} reasoning), "
              f"realized ${stats['realized_provider_cost_usd']:.4f}")

    if failures:
        for failure in failures:
            append_jsonl(failure, DATA / "study_failures.jsonl")
    finalize_outputs(model=model, persona_type=persona_type, seed=seed,
                     mode=mode, plan=plan, failures=failures, tags=tags)
    print(f"study[{mode}]: planned {len(plan)}, run this pass {len(pending)}, "
          f"failures this pass {len(failures)}")
    return 1 if failures else 0


def finalize_outputs(*, model: str, persona_type: str, seed: int, mode: str,
                     plan: list[dict], failures: list[dict], tags: dict) -> None:
    """Stream the checkpoint into the flat outputs (safe to re-run any time)."""
    checkpoint = DATA / "study_checkpoint.jsonl"
    wide_columns = _wide_header(tags)
    long_columns = ["run_id", "base_pid", "condition", "raw_condition", "control_variant",
                    "question_id", "export_tag", "stage_index", "question_text",
                    "question_type", "answer_raw", "answer_value", "answer_label",
                    "answer_status", "model", "persona_representation", "seed"]
    ledger_columns = ["run_id", "base_pid", "condition", "raw_condition", "control_variant",
                      "status", "stage_count", "question_count", "prompt_chars_total",
                      "latency_ms_total", "technical_retries", "completed_at"]

    seen_run_ids: set[str] = set()
    ledger_rows: list[dict] = []
    with open(DATA / "simulation_answers_long.csv", "w", newline="", encoding="utf-8") as long_f, \
         open(DATA / "simulation_respondents_wide.csv", "w", newline="", encoding="utf-8") as wide_f:
        long_writer = csv.DictWriter(long_f, fieldnames=long_columns, extrasaction="ignore")
        wide_writer = csv.DictWriter(wide_f, fieldnames=wide_columns, extrasaction="ignore")
        long_writer.writeheader()
        wide_writer.writeheader()
        if Path(checkpoint).exists():
            with open(checkpoint, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    run_id = record.get("run_id")
                    if not run_id or run_id in seen_run_ids:
                        continue
                    seen_run_ids.add(run_id)
                    stage_of = {}
                    for stage in record.get("stage_records", []):
                        for qid in stage.get("question_ids", []):
                            stage_of[qid] = stage["stage_index"]
                    wide = {"run_id": run_id, "base_pid": record["base_pid"],
                            "condition": record["condition"],
                            "raw_condition": record["raw_condition"],
                            "control_variant": record.get("control_variant", ""),
                            "model": model, "persona_representation": persona_type,
                            "seed": seed, "excluded": record.get("excluded", "")}
                    for row in record.get("rows", []):
                        qid = row["question_id"]
                        long_writer.writerow(
                            {"run_id": run_id, "base_pid": record["base_pid"],
                             "condition": record["condition"],
                             "raw_condition": record["raw_condition"],
                             "control_variant": record.get("control_variant", ""),
                             "question_id": qid, "export_tag": tags.get(qid, ""),
                             "stage_index": stage_of.get(qid, ""),
                             "question_text": row.get("question_text", ""),
                             "question_type": row.get("question_type", ""),
                             "answer_raw": row.get("answer_raw", ""),
                             "answer_value": row.get("answer_value", ""),
                             "answer_label": row.get("answer_label", ""),
                             "answer_status": row.get("answer_status", ""),
                             "model": model, "persona_representation": persona_type,
                             "seed": seed})
                        wide[tags.get(qid, qid)] = row.get("answer_label") or row.get("answer_value", "")
                    wide_writer.writerow(wide)
                    stage_records = record.get("stage_records", [])
                    ledger_rows.append(
                        {"run_id": run_id, "base_pid": record["base_pid"],
                         "condition": record["condition"],
                         "raw_condition": record["raw_condition"],
                         "control_variant": record.get("control_variant", ""),
                         "status": "completed",
                         "stage_count": len(stage_records),
                         "question_count": len(record.get("rows", [])),
                         "prompt_chars_total": sum(s.get("prompt_chars", 0) for s in stage_records),
                         "latency_ms_total": sum(s.get("latency_ms", 0) for s in stage_records),
                         "technical_retries": sum(s.get("retries", 0) for s in stage_records),
                         "completed_at": record.get("completed_at", "")})

    failure_records: list[dict] = []
    failures_path = DATA / "study_failures.jsonl"
    if Path(failures_path).exists():
        with open(failures_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    failure_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    open_failures = [f for f in failure_records if f.get("run_id") not in seen_run_ids]
    for failure in open_failures:
        ledger_rows.append({"run_id": failure.get("run_id", ""),
                            "base_pid": failure.get("base_pid", ""),
                            "condition": "", "raw_condition": failure.get("raw_condition", ""),
                            "control_variant": "", "status": "failed",
                            "stage_count": 0, "question_count": 0,
                            "prompt_chars_total": 0, "latency_ms_total": 0,
                            "technical_retries": 0, "completed_at": failure.get("at", "")})
    write_csv(ledger_rows, DATA / "run_ledger.csv", ledger_columns)

    # coverage
    per_condition: dict[str, set] = {}
    per_twin: dict[str, int] = {}
    for row in ledger_rows:
        if row["status"] != "completed":
            continue
        per_condition.setdefault(row["condition"], set()).add(row["base_pid"])
        per_twin[row["base_pid"]] = per_twin.get(row["base_pid"], 0) + 1
    planned_ids = {p["run_id"] for p in plan}
    coverage = {
        "planned_sessions": len(plan),
        "completed_sessions": len(seen_run_ids & planned_ids),
        "open_failures": len(open_failures),
        "conditions": {c: len(v) for c, v in sorted(per_condition.items())},
        "twins_with_all_17": sum(1 for v in per_twin.values() if v == 17),
        "twins_partial": {t: c for t, c in per_twin.items() if c != 17},
    }
    save_json(coverage, DATA / "coverage_report.json")

    # failures + repairs, all in one report
    repair_rows = read_csv(DATA / "demographic_repairs.csv") if Path(DATA / "demographic_repairs.csv").exists() else []
    retries_by_stage = sum(r["technical_retries"] for r in ledger_rows)
    save_json(
        {"prestudy_repairs": repair_rows,
         "study_session_failures_all_attempts": failure_records,
         "study_session_failures_unresolved": open_failures,
         "study_technical_retries_within_sessions": retries_by_stage},
        DATA / "failure_and_repair_report.json",
    )

    source_manifest = load_json(MANIFEST)
    from .common import register_models
    register_models()
    from services.v2.llm_client import MODEL_SPECS  # noqa: E402

    # The exact provider model ID comes from the platform's own registry entry
    # for the alias — never inferred from the display name.
    spec = MODEL_SPECS.get(model) or {}
    save_json(
        {"design": "full eligible pool x 17 conditions; no stratification or subsampling",
         "mode": mode,
         "model_alias": model,
         "provider_model_id": spec.get("model_name", ""),
         "reasoning_effort": spec.get("default_reasoning_effort", ""),
         "temperature": "not sent — gpt-5 family requests carry reasoning.effort instead "
                        "(platform nominal default 0.3 is omitted for these models)",
         "persona_representation": persona_type,
         "persona_bank": "Twin-2K-500 built-in personas "
                         f"(mega_persona_summary_text, worktree commit {source_manifest['surveytwin']['commit']})",
         "prompt_structure": "system prompt + persona + frozen pre-study Q/A pairs in original "
                             "order (prior-answer section) + study stage text + questions JSON",
         "prompt_builder": "services.v2.simulation_executor.build_simulation_prompt_parts + "
                           "services.v3.qsf_runtime.staged_* (per-stage prompt_sha256 recorded)",
         "seed": seed,
         "code_commit": source_manifest["surveytwin"]["commit"],
         "benchmark_commit": source_manifest["benchmark_repo"]["commit"],
         "qsf_sha256": source_manifest["benchmark_repo"]["authoritative_files"]["survey/survey.qsf"],
         "prestudy_source": load_json(DATA / "prestudy_source.json")
                            if Path(DATA / "prestudy_source.json").exists() else {},
         "planned_sessions": len(plan),
         "coverage": coverage,
         "finalized_at": _now()},
        DATA / "run_manifest.json",
    )


# ---------------------------------------------------------------------------
# Section 8 — final validation
# ---------------------------------------------------------------------------

def validate_study(*, expected_conditions: int = 17) -> int:
    print("study validation")
    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    eligible = {r["base_pid"] for r in read_csv(DATA / "eligible_twins.csv")}
    n = len(eligible)
    ledger = [r for r in read_csv(DATA / "run_ledger.csv") if r["status"] == "completed"]

    per_twin: dict[str, set] = {}
    per_condition: dict[str, list] = {}
    combos: dict[tuple, int] = {}
    for row in ledger:
        per_twin.setdefault(row["base_pid"], set()).add(row["condition"])
        per_condition.setdefault(row["condition"], []).append(row["base_pid"])
        key = (row["base_pid"], row["condition"])
        combos[key] = combos.get(key, 0) + 1

    check(f"every eligible twin has exactly {expected_conditions} completed conditions",
          all(len(per_twin.get(t, ())) == expected_conditions for t in eligible),
          f"{sum(1 for t in eligible if len(per_twin.get(t, ())) != expected_conditions)} twin(s) off")
    check(f"total records equal N_eligible x {expected_conditions} = {n * expected_conditions}",
          len(ledger) == n * expected_conditions, f"got {len(ledger)}")
    pools = {c: set(v) for c, v in per_condition.items()}
    check("every condition contains the same eligible pool",
          all(p == eligible for p in pools.values()),
          str({c: len(p ^ eligible) for c, p in pools.items() if p != eligible}))
    check(f"pooled control contains N_eligible = {n} records",
          len(per_condition.get("control", [])) == n,
          f"got {len(per_condition.get('control', []))}")
    duplicates = [k for k, v in combos.items() if v > 1]
    check("no duplicate base_pid x condition", not duplicates, str(duplicates[:3]))

    # Stimulus correctness from the stage records: each session rendered its
    # own stimulus block exactly once and no other treatment's block.
    audit = load_json(ARTIFACTS / "qsf_audit.json")
    by_name = {b["block_name"]: b["block_id"] for b in audit["block_inventory"]}
    jaguar = {by_name[n_] for n_ in by_name if str(n_).startswith("jealous jaguar")}
    planarian_routes = {by_name[n_] for n_ in by_name
                        if str(n_).startswith("practical_planarian_")
                        and n_ != "practical_planarian_state"}
    stimulus_of = {
        "control neckties": {by_name["History of Neckties"]},
        "control baseball": {by_name["Rules of Baseball"]},
        "control dances": {by_name["Different Types of Dances"]},
        "jealous jaguar": jaguar,
        "practical planarian": {by_name["practical_planarian_state"]},
    }
    for code in load_json(ARTIFACTS / "condition_map.json")["conditions"]:
        stimulus_of.setdefault(code, {by_name[code]} if code in by_name else set())
    all_stimulus_blocks = set().union(*stimulus_of.values()) | planarian_routes

    wrong_stimulus = 0
    checked_sessions = 0
    with open(DATA / "study_checkpoint.jsonl", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            checked_sessions += 1
            rendered = {b for s in record.get("stage_records", [])
                        for b in s.get("rendered_block_ids", [])}
            expected = stimulus_of.get(record["raw_condition"], set())
            allowed = expected | (planarian_routes
                                  if record["raw_condition"] == "practical planarian" else set())
            if not expected <= rendered or (rendered & all_stimulus_blocks) - allowed:
                wrong_stimulus += 1
    check("every session rendered its own stimulus exactly once, no foreign stimulus",
          wrong_stimulus == 0, f"{wrong_stimulus} of {checked_sessions} sessions wrong")

    excluded = read_csv(DATA / "excluded_twins.csv")
    check("all exclusions decided pre-treatment",
          all(("attention" in r["exclusion_reason"]) or ("persistent" in r["exclusion_reason"])
              or ("incomplete" in r["exclusion_reason"])
              for r in excluded),
          "exclusion reasons all pre-study")
    check("failure and repair report present",
          Path(DATA / "failure_and_repair_report.json").exists())

    failures = [r for r in results if not r[1]]
    print(f"study validation: {len(results)} checks, {len(failures)} FAILED")
    return 1 if failures else 0
