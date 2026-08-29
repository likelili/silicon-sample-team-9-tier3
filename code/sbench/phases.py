"""Sections 7-9 — pre-study run, conflict diagnosis/repair, and the freeze.

Nothing here calls the API unless invoked with ``--live`` AND the approval
gate (``SBENCH_APPROVED=1``) is set.  Mock and dry modes exercise the exact
same code paths.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from .common import (
    ARTIFACTS,
    AUDIT,
    BENCH_ROOT,
    DATA,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    QID,
    append_jsonl,
    load_json,
    read_csv,
    require_live_flag,
    rng_for,
    save_json,
    wire_worktree,
    write_csv,
)
from .conflicts import FIELDS, STRATIFICATION_FIELDS, compare_field, compare_twin
from .driver import StagedDriver, load_template
from .telemetry import TELEMETRY_COLUMNS, capture, summarize

ATTENTION_OVERRIDES = {
    "QID1721185793": "Somewhat disagree",   # attention1: instructed choice
    "QID1721185922": "attention",           # attention2: free text containing "atten"
}


def _export_tags() -> dict[str, str]:
    audit = load_json(ARTIFACTS / "qsf_audit.json")
    tags: dict[str, str] = {}
    for block in audit["block_inventory"]:
        for question in block["questions"]:
            if question.get("export_tag"):
                tags[question["qid"]] = question["export_tag"]
    return tags


def _profiles() -> dict[str, dict]:
    return {row["persona_id"]: row for row in read_csv(BENCH_ROOT / "data" / "twin_profiles.csv")}


def _answers_by_qid(rows: list[dict]) -> dict[str, str]:
    return {r["question_id"]: (r.get("answer_label") or r.get("answer_value") or "") for r in rows}


# ---------------------------------------------------------------------------
# Section 7 — pre-study run over the panel
# ---------------------------------------------------------------------------

async def run_prestudy(
    *, mode: str, model: str, persona_type: str, seed: int,
    limit: int | None, concurrency: int, mock_profile_aware: bool,
    inject_overrides: dict[str, dict[str, str]] | None = None,
    twins_subset: list[str] | None = None,
    resume: bool = False,
) -> int:
    require_live_flag(mode == "live", "prestudy")
    template = load_template(ARTIFACTS / "template_prestudy.json")
    profiles = _profiles()
    twins = sorted(profiles, key=lambda p: int(p.split("_")[1]))
    if twins_subset is not None:
        twins = [t for t in twins if t in set(twins_subset)]
    if limit:
        twins = twins[:limit]

    driver = StagedDriver(
        template, mode=mode, model=model, persona_type=persona_type,
        seed=seed, persona_count=len(profiles),
    )
    tags = _export_tags()

    stages_path = AUDIT / "prestudy_stages.jsonl"
    prompts_path = AUDIT / "prestudy_prompts.jsonl"
    for path in (stages_path, prompts_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    # Checkpointing: every twin is flushed to a JSONL line the moment it
    # finishes, so an interruption loses at most the in-flight twins.  A resume
    # reads the checkpoint and skips completed twins instead of re-running (and
    # re-paying for) them.
    checkpoint = DATA / "prestudy_checkpoint.jsonl"
    completed: dict[str, dict] = {}
    if resume and Path(checkpoint).exists():
        with open(checkpoint, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue          # torn final line from a hard kill
                if record.get("persona_id"):
                    completed[record["persona_id"]] = record
        print(f"prestudy[{mode}]: resuming — {len(completed)} twin(s) already checkpointed")
    else:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        Path(checkpoint).write_text("")

    pending = [t for t in twins if t not in completed]
    long_rows: list[dict] = []
    call_rows: list[dict] = []
    qa_records: dict[str, list] = {}
    run_meta: dict[str, dict] = {}
    semaphore = asyncio.Semaphore(concurrency)
    flush_lock = asyncio.Lock()

    async def one(index: int, twin: str):
        overrides = dict(ATTENTION_OVERRIDES) if mode in ("mock", "dry") else None
        if overrides is not None and mock_profile_aware:
            overrides.update(_profile_consistent_overrides(profiles[twin]))
        if overrides is not None and inject_overrides and twin in inject_overrides:
            overrides.update(inject_overrides[twin])
        async with semaphore:
            result = await driver.run_twin(twin, index, overrides=overrides, rng_extra="prestudy")
        for stage in result.stages:
            append_jsonl(
                {
                    "persona_id": twin, "stage_index": stage.stage_index,
                    "question_ids": stage.question_ids, "prompt_chars": stage.prompt_chars,
                    "prompt_sha256": stage.prompt_sha256,
                    "prior_answer_count": stage.prior_answer_count,
                    "embedded_data": stage.embedded_data, "events": stage.events,
                    "rendered_block_ids": stage.rendered_block_ids,
                    "retries": stage.retries, "latency_ms": stage.latency_ms, "mode": stage.mode,
                    **({"raw_answers": stage.raw_answers,
                        "normalized_answers": stage.normalized_answers}
                       if stage.mode != "mock" else {}),
                },
                stages_path,
            )
            if mode != "mock":
                append_jsonl(
                    {"persona_id": twin, "stage_index": stage.stage_index,
                     "prompt_sha256": stage.prompt_sha256,
                     "system_prompt": stage.system_prompt, "user_prompt": stage.user_prompt_head},
                    prompts_path,
                )
            call_rows.append(
                {"persona_id": twin, "stage_index": stage.stage_index,
                 "question_count": len(stage.question_ids),
                 "prompt_chars_total": stage.prompt_chars["total"],
                 "prompt_sha256": stage.prompt_sha256,
                 "latency_ms": stage.latency_ms, "retries": stage.retries, "mode": mode}
            )
        for row in result.rows:
            row = dict(row)
            row["export_tag"] = tags.get(row["question_id"], "")
            long_rows.append(row)
        qa_records[twin] = [
            {"question": {"id": r["question_id"], "text": r["question_text"], "type": r["question_type"]},
             "answer": {"question_id": r["question_id"], "answer_value": r["answer_value"],
                        "answer_label": r["answer_label"], "answer_raw": r["answer_raw"]}}
            for r in result.rows
        ]
        run_meta[twin] = {
            "excluded": result.excluded, "done": result.done, "error": result.error,
            "embedded_data": result.embedded_data, "stage_count": len(result.stages),
        }
        async with flush_lock:
            append_jsonl(
                {"persona_id": twin, "qa": qa_records[twin], "meta": run_meta[twin],
                 "rows": [r for r in result.rows]},
                checkpoint,
            )

    index_of = {twin: i for i, twin in enumerate(twins)}
    with capture("prestudy") as telemetry:
        await asyncio.gather(*(one(index_of[t], t) for t in pending))

    # Fold checkpointed twins back in so the outputs cover every twin.
    for twin, record in completed.items():
        qa_records.setdefault(twin, record.get("qa") or [])
        run_meta.setdefault(twin, record.get("meta") or {})
        for row in record.get("rows") or []:
            row = dict(row)
            row["export_tag"] = tags.get(row.get("question_id", ""), "")
            long_rows.append(row)

    write_csv(
        long_rows, DATA / "prestudy_answers_long.csv",
        ["persona_id", "question_id", "export_tag", "question_text", "question_type",
         "answer_raw", "answer_value", "answer_label", "answer_status"],
    )
    wide_columns = ["persona_id", "excluded", "age_band"]
    seen = set(wide_columns)
    wide_rows = []
    for twin in twins:
        row = {"persona_id": twin,
               "excluded": run_meta[twin]["excluded"],
               "age_band": str(run_meta[twin]["embedded_data"].get("age_band", ""))}
        for record in qa_records.get(twin, []):
            column = tags.get(record["question"]["id"], record["question"]["id"])
            row[column] = record["answer"]["answer_label"] or record["answer"]["answer_value"]
            if column not in seen:
                seen.add(column)
                wide_columns.append(column)
        wide_rows.append(row)
    write_csv(wide_rows, DATA / "prestudy_answers_wide.csv", wide_columns)
    write_csv(call_rows, AUDIT / "prestudy_calls.csv",
              ["persona_id", "stage_index", "question_count", "prompt_chars_total",
               "prompt_sha256", "latency_ms", "retries", "mode"])
    save_json({t: qa_records[t] for t in twins}, DATA / "prestudy_qa.json")
    save_json({t: run_meta[t] for t in twins}, DATA / "prestudy_meta.json")

    if telemetry:
        write_csv(telemetry, AUDIT / "prestudy_provider_calls.csv", TELEMETRY_COLUMNS)
        stats = summarize(telemetry)
        save_json(stats, AUDIT / "prestudy_provider_summary.json")
        print(f"  provider: {stats['calls']} calls, "
              f"{stats['total_tokens']:,} tokens "
              f"({stats['reasoning_tokens']:,} reasoning), "
              f"realized ${stats['realized_provider_cost_usd']:.4f}")

    attempted = len(twins)
    errors = [t for t in twins if run_meta.get(t, {}).get("error")]
    excluded = [t for t in twins if run_meta.get(t, {}).get("excluded")]
    print(f"prestudy[{mode}]: attempted {attempted} "
          f"({len(pending)} run now, {len(completed)} resumed), errors {len(errors)}, "
          f"attention-excluded {len(excluded)}, long rows {len(long_rows)}")
    if errors:
        print("  errors on:", errors[:10])
    return 1 if errors else 0


def _profile_consistent_overrides(profile: dict) -> dict[str, str]:
    """Mock answers consistent with the profile, so mock preflights model the
    healthy case.  The panel band is mapped to a birth year INSIDE the band."""
    band_year = {"18-29": "2000", "30-49": "1990", "50-64": "1970", "65+": "1950"}
    education = {
        "Less than high school": "Less than high school",
        "High school graduate": "High school diploma / GED",
        "Some college, no degree": "Some college or Associate's degree",
        "Associate's degree": "Some college or Associate's degree",
        "College graduate/some postgrad": "Bachelor's degree",
        "Postgraduate": "Master's degree / Professional degree",
    }
    income = {
        "Less than $30,000": "Less than $30,000",
        "$30,000-$50,000": "$30,000 to $55,999",
        "$50,000-$75,000": "$56,000 to $99,999",
        "$75,000-$100,000": "$56,000 to $99,999",
        "$100,000 or more": "$100,000 to $167,999",
    }
    household = {"More than 4": "6 or more"}
    party = {"Democrat": "Democrat", "Independent": "Independent",
             "Republican": "Republican", "Something else": "Other (please specify)"}
    religion = {
        "Protestant": "Protestant", "Roman Catholic": "Catholic", "Jewish": "Jewish",
        "Mormon": "Mormon", "Buddhist": "Buddhist", "Muslim": "Muslim",
        "Orthodox": "Orthodox Christian", "Hindu": "Hindu",
        "Atheist": "I am not religious", "Agnostic": "I am not religious",
        "Nothing in particular": "I am not religious", "Other": "Other religion (please specify)",
    }
    race = {"White": "White / Caucasian", "Black": "Black / African-American",
            "Hispanic": "Latino / Hispanic", "Asian": "Asian / Asian-American", "Other": "Other"}
    out = {
        QID["year_birth"]: band_year.get(profile.get("age_band", ""), "1985"),
        QID["gender"]: profile.get("sex_at_birth", "Female"),
        QID["race"]: race.get(profile.get("race_origin", ""), "Other"),
        QID["education"]: education.get(profile.get("education", ""), "Bachelor's degree"),
        QID["income"]: income.get(profile.get("income_bracket", ""), "$56,000 to $99,999"),
        QID["party"]: party.get(profile.get("political_affiliation", ""), "Independent"),
        QID["religion"]: religion.get(profile.get("religion", ""), "I am not religious"),
    }
    size = profile.get("household_size", "")
    out[QID["household"]] = household.get(size, size if size.isdigit() else "2")
    return out


# ---------------------------------------------------------------------------
# Section 8 — conflict diagnosis + targeted repair
# ---------------------------------------------------------------------------

ROUTING_PARENTS = {QID["year_birth"], QID["party"], QID["religion"]}
# Questions whose ROUTING depends on a parent answer.  When a parent repair
# changes the route, these are invalidated or regenerated to match.
DEPENDENT_QIDS = {
    QID["party"]: ("QID281",),                 # partisan importance
    QID["religion"]: ("QID287", "QID285"),     # born again / religiosity
}


def _required_prestudy_qids(template: dict) -> set[str]:
    """Answerable questions in UNCONDITIONAL pre-study blocks (top level, not
    under any Branch).  Conditional questions (partisan importance, born again,
    religiosity) cannot be universally required; optional _TEXT companions are
    excluded."""
    from services.v2.instrument_builder import collect_instrument_questions  # noqa: E402

    required: set[str] = set()
    for element in template["Elements"]:
        kind = str(element.get("ElementType") or element.get("Type") or "")
        if kind != "Block":
            continue
        for question in collect_instrument_questions([element]):
            qid = str(question.get("id") or "")
            if qid and not qid.endswith("_TEXT"):
                required.add(qid)
    return required


def run_diagnose() -> int:
    """Section 8a — diagnosis only.  Makes NO API call under any flag.

    Produces the profile comparison table, the incomplete-run findings, and the
    proposed repair list, so the conflict picture can be reviewed and signed off
    before any paid retry is issued.  ``run_repairs`` performs the same
    diagnosis internally; this phase exists so the two can be separated in time.
    """
    wire_worktree()   # diagnose runs without a driver, so nothing else wires sys.path
    qa = load_json(DATA / "prestudy_qa.json")
    meta = load_json(DATA / "prestudy_meta.json")
    profiles = _profiles()
    template = load_template(ARTIFACTS / "template_prestudy.json")
    required_qids = _required_prestudy_qids(template)

    comparison_rows: list[dict] = []
    proposed: list[dict] = []
    findings: list[dict] = []

    for twin in sorted(qa, key=lambda p: int(p.split("_")[1])):
        answers = _answers_by_qid([
            {"question_id": r["question"]["id"],
             "answer_label": r["answer"]["answer_label"],
             "answer_value": r["answer"]["answer_value"]}
            for r in qa[twin]
        ])
        comparisons = compare_twin(profiles.get(twin, {}), answers)
        for row in comparisons:
            comparison_rows.append({"source_twin_id": twin, **row})

        run_error = str(meta[twin].get("error") or "")
        missing_required = sorted(required_qids - {r["question"]["id"] for r in qa[twin]})
        attention = meta[twin].get("excluded") or ""
        missing_strat = [f for f in STRATIFICATION_FIELDS
                         if not answers.get(FIELDS[f][1], "").strip()]
        if run_error or missing_required or attention or missing_strat:
            findings.append({
                "source_twin_id": twin,
                "execution_error": run_error,
                "missing_required_questions": ";".join(missing_required[:8]),
                "attention_check": attention,
                "missing_stratification_fields": ";".join(missing_strat),
                "would_be_excluded_before_repair": "yes",
            })

        for conflict in [c for c in comparisons if c["result"] == "conflict"]:
            qid = FIELDS[conflict["field"]][1]
            proposed.append({
                "source_twin_id": twin, "field": conflict["field"], "qid": qid,
                "profile_value": conflict["profile_value"],
                "survey_value": conflict["survey_value"],
                "routing_parent": "yes" if qid in ROUTING_PARENTS else "no",
                "dependents_if_route_changes": ";".join(DEPENDENT_QIDS.get(qid, ())),
                "planned_action": "one targeted retry of this question; "
                                  "keep twin if it resolves, exclude if it persists",
            })

    write_csv(comparison_rows, DATA / "profile_comparisons.csv",
              ["source_twin_id", "field", "profile_value", "survey_value",
               "free_text", "result", "note"])
    write_csv(proposed, DATA / "proposed_repairs.csv",
              ["source_twin_id", "field", "qid", "profile_value", "survey_value",
               "routing_parent", "dependents_if_route_changes", "planned_action"])
    write_csv(findings, DATA / "incomplete_run_findings.csv",
              ["source_twin_id", "execution_error", "missing_required_questions",
               "attention_check", "missing_stratification_fields",
               "would_be_excluded_before_repair"])

    twins_with_conflicts = len({r["source_twin_id"] for r in proposed})
    routing_parents = sum(1 for r in proposed if r["routing_parent"] == "yes")
    save_json(
        {"twins_examined": len(qa), "field_comparisons": len(comparison_rows),
         "conflicts": len(proposed), "twins_with_conflicts": twins_with_conflicts,
         "routing_parent_conflicts": routing_parents,
         "twins_flagged_incomplete_or_excluded": len(findings),
         "api_calls_made": 0},
        DATA / "diagnosis_summary.json",
    )
    print(f"diagnose: {len(comparison_rows)} comparisons, {len(proposed)} conflict(s) "
          f"on {twins_with_conflicts} twin(s) ({routing_parents} routing-parent), "
          f"{len(findings)} twin(s) flagged incomplete/excluded — NO API calls")
    print(f"  review data/proposed_repairs.csv before running `repairs --mode live`")
    return 0


async def run_repairs(
    *, mode: str, model: str, persona_type: str, seed: int, concurrency: int,
    mock_retry_values: dict[str, dict[str, str]] | None = None,
) -> int:
    require_live_flag(mode == "live", "repairs")
    qa = load_json(DATA / "prestudy_qa.json")
    meta = load_json(DATA / "prestudy_meta.json")
    profiles = _profiles()
    template = load_template(ARTIFACTS / "template_prestudy.json")
    driver = StagedDriver(template, mode=mode, model=model, persona_type=persona_type,
                          seed=seed, persona_count=len(profiles))

    comparison_rows: list[dict] = []
    repair_rows: list[dict] = []
    eligibility: dict[str, dict] = {}
    field_by_qid = {qid: name for name, (_, qid) in FIELDS.items()}
    required_qids = _required_prestudy_qids(template)
    _capture = capture("repairs")
    repair_telemetry = _capture.__enter__()

    semaphore = asyncio.Semaphore(concurrency)

    async def retry_question(twin: str, index: int, qid: str) -> str:
        """Ask exactly one demographic question again, same context up to it."""
        records = qa[twin]
        position = next(i for i, r in enumerate(records) if r["question"]["id"] == qid)
        prior = [(r["question"], r["answer"]) for r in records[:position]]
        question_payload = _question_payload_from_template(template, qid)
        from services.v3.qsf_runtime import (  # noqa: E402
            staged_prior_answer_context, staged_survey_sequence_with_prior_context,
        )
        from services.v2.simulation_executor import build_simulation_prompt_parts  # noqa: E402

        if mode in ("mock", "dry"):
            # Mock repair answers consistently with the profile (the success
            # path); mock_retry_values pins specific outcomes for preflight
            # fixtures, including deliberately persistent conflicts.
            if mock_retry_values and qid in (mock_retry_values.get(twin) or {}):
                return mock_retry_values[twin][qid]
            fixed = _profile_consistent_overrides(profiles[twin]).get(qid, "")
            return fixed

        from services.v2.simulation_executor import simulate_persona  # noqa: E402
        from services.v3.billing import BillingContext  # noqa: E402
        # Rebuild the exact prior-answer context through a scratch state.
        scratch_runtime = driver._QsfRuntime(template, seed=seed, persona_count=len(profiles))
        scratch = scratch_runtime.new_state(persona_id=twin, persona_index=index)
        scratch_runtime.apply_answers(
            scratch, questions=[q for q, _ in prior], answers=[a for _, a in prior]
        )
        sequence = staged_survey_sequence_with_prior_context(
            prior_answer_context=staged_prior_answer_context(scratch),
            survey_sequence_text="",
        )
        async with semaphore:
            raw = await simulate_persona(
                model=model, persona_type=persona_type, persona_id=twin,
                questions=[question_payload], survey_sequence_text=sequence,
                billing_context=BillingContext(
                    user_id="silicon-bench", feature="benchmark.repair",
                    metadata={"persona_id": twin, "qid": qid},
                ),
            )
        answer = (raw or [{}])[0]
        return str(answer.get("answer_label") or answer.get("answer_value") or "")

    for index, twin in enumerate(sorted(qa, key=lambda p: int(p.split("_")[1]))):
        answers = _answers_by_qid([
            {"question_id": r["question"]["id"],
             "answer_label": r["answer"]["answer_label"], "answer_value": r["answer"]["answer_value"]}
            for r in qa[twin]
        ])
        profile = profiles.get(twin, {})
        comparisons = compare_twin(profile, answers)
        for row in comparisons:
            comparison_rows.append({"source_twin_id": twin, **row})

        conflicts = [c for c in comparisons if c["result"] == "conflict"]
        attention = meta[twin]["excluded"]
        record = {
            "source_twin_id": twin,
            "attention_check_status": "failed: " + attention if attention else "passed",
            "eligible": "yes", "exclusion_reason": "",
            "conflict_fields": ";".join(c["field"] for c in conflicts),
        }

        run_error = str(meta[twin].get("error") or "")
        if run_error:
            record.update(eligible="no",
                          exclusion_reason=f"incomplete pre-study run (execution error): {run_error}")
        elif attention:
            record.update(eligible="no", exclusion_reason=f"attention check: {attention}")
        elif required_qids - {r["question"]["id"] for r in qa[twin]}:
            missing_required = sorted(required_qids - {r["question"]["id"] for r in qa[twin]})
            record.update(eligible="no",
                          exclusion_reason="incomplete pre-study run (missing required questions): "
                                           + ";".join(missing_required[:6]))
        else:
            unresolved = []
            for conflict in conflicts:
                qid = FIELDS[conflict["field"]][1]
                retry_value = await retry_question(twin, index, qid)
                verdict = compare_field(
                    conflict["field"], conflict["profile_value"], retry_value,
                    free_text=conflict.get("free_text", ""),
                )
                resolved = verdict == "no_conflict"
                repair_rows.append(
                    {"source_twin_id": twin, "field": conflict["field"], "qid": qid,
                     "profile_value": conflict["profile_value"],
                     "original_value": conflict["survey_value"],
                     "retry_value": retry_value,
                     "retry_result": verdict or "skipped",
                     "resolved": "yes" if resolved else "no",
                     "routing_parent": "yes" if qid in ROUTING_PARENTS else "no"}
                )
                if resolved:
                    _apply_repair(qa[twin], qid, retry_value)
                else:
                    unresolved.append(conflict["field"])
            if unresolved:
                record.update(
                    eligible="no",
                    exclusion_reason="persistent logical conflict: " + ";".join(unresolved),
                )
            missing = [
                f for f in STRATIFICATION_FIELDS
                if not answers.get(FIELDS[f][1], "").strip()
            ]
            if missing and record["eligible"] == "yes":
                record.update(eligible="no",
                              exclusion_reason="missing stratification field: " + ";".join(missing))

            # Reconcile routing-dependent answers after any resolved repair of
            # a routing parent: replay the routing with the repaired answers,
            # drop dependents that are no longer routed, generate dependents
            # that are newly required, and reorder the record to the replayed
            # survey order.
            repaired_parents = [
                r["qid"] for r in repair_rows
                if r["source_twin_id"] == twin and r["resolved"] == "yes"
                and r["qid"] in ROUTING_PARENTS
            ]
            if repaired_parents and record["eligible"] == "yes":
                actions = await _reconcile_dependents(
                    twin=twin, index=index, records=qa[twin],
                    repaired_parents=repaired_parents, profiles=profiles,
                    template=template, driver=driver, mode=mode, model=model,
                    persona_type=persona_type, seed=seed, semaphore=semaphore,
                )
                for action in actions:
                    repair_rows.append(
                        {"source_twin_id": twin, "field": action["field"],
                         "qid": action["qid"], "profile_value": "",
                         "original_value": action.get("original_value", ""),
                         "retry_value": action.get("new_value", ""),
                         "retry_result": action["action"],
                         "resolved": "n/a", "routing_parent": "dependent"}
                    )
        eligibility[twin] = record

    _capture.__exit__(None, None, None)
    write_csv(comparison_rows, DATA / "profile_comparisons.csv",
              ["source_twin_id", "field", "profile_value", "survey_value",
               "free_text", "result", "note"])
    write_csv(repair_rows, DATA / "demographic_repairs.csv",
              ["source_twin_id", "field", "qid", "profile_value", "original_value",
               "retry_value", "retry_result", "resolved", "routing_parent"])
    eligibility_rows = []
    for twin, record in eligibility.items():
        repair = next((r for r in repair_rows if r["source_twin_id"] == twin), {})
        eligibility_rows.append(
            {**record,
             "original_profile_value": repair.get("profile_value", ""),
             "original_survey_value": repair.get("original_value", ""),
             "retry_value": repair.get("retry_value", ""),
             "final_frozen_value": repair.get("retry_value") if repair.get("resolved") == "yes"
                                   else repair.get("original_value", ""),
             "logical_compatibility_result": repair.get("retry_result", "")}
        )
    write_csv(eligibility_rows, DATA / "eligibility_audit.csv",
              ["source_twin_id", "eligible", "exclusion_reason", "conflict_fields",
               "original_profile_value", "original_survey_value", "retry_value",
               "final_frozen_value", "logical_compatibility_result", "attention_check_status"])
    save_json(qa, DATA / "prestudy_qa_repaired.json")

    if repair_telemetry:
        write_csv(repair_telemetry, AUDIT / "repairs_provider_calls.csv", TELEMETRY_COLUMNS)
        stats = summarize(repair_telemetry)
        save_json(stats, AUDIT / "repairs_provider_summary.json")
        print(f"  provider: {stats['calls']} calls, {stats['total_tokens']:,} tokens, "
              f"realized ${stats['realized_provider_cost_usd']:.4f}")

    n_conflict_twins = len({r["source_twin_id"] for r in repair_rows})
    n_excluded = sum(1 for r in eligibility.values() if r["eligible"] == "no")
    print(f"repairs[{mode}]: {len(comparison_rows)} comparisons, "
          f"{len(repair_rows)} repairs attempted on {n_conflict_twins} twin(s), "
          f"{n_excluded} twin(s) ineligible")
    return 0


async def _reconcile_dependents(
    *, twin: str, index: int, records: list, repaired_parents: list[str],
    profiles: dict, template: dict, driver, mode: str, model: str,
    persona_type: str, seed: int, semaphore,
) -> list[dict]:
    """After a resolved repair of a routing parent, make the record consistent
    with the routes the repaired answers imply.

      * birth year  -> recompute age_band (no dependent questions);
      * party       -> partisan importance appears/disappears;
      * religion    -> born again / religiosity appear/disappear.

    The routing itself is decided by replaying the pre-study template through
    the real runtime in mock mode with every current answer pinned as an
    override — no reimplemented logic, no API calls.  Dependents that are no
    longer routed are removed; newly required dependents are generated (mock:
    deterministic; live: a targeted single-question call with the same prior
    context).  Finally the record is reordered to the replayed survey order.
    """
    actions: list[dict] = []
    field_of = {"QID281": "partisan_importance", "QID287": "born_again", "QID285": "religiosity"}

    # Recompute age_band whenever birth year was repaired.
    if QID["year_birth"] in repaired_parents:
        year = next((r["answer"]["answer_value"] for r in records
                     if r["question"]["id"] == QID["year_birth"]), "")
        actions.append({"field": "age_band", "qid": "age_band",
                        "action": "recomputed", "new_value": _age_band_from_year(year)})

    affected: list[str] = []
    for parent in repaired_parents:
        affected.extend(DEPENDENT_QIDS.get(parent, ()))
    if not affected:
        return actions

    overrides = dict(ATTENTION_OVERRIDES)
    for record in records:
        value = record["answer"]["answer_label"] or record["answer"]["answer_value"]
        if value:
            overrides[record["question"]["id"]] = value

    replay_driver = StagedDriver(
        driver.template, mode="mock", model=model, persona_type=persona_type,
        seed=seed, persona_count=driver.persona_count,
    )
    replay = await replay_driver.run_twin(twin, index, overrides=overrides,
                                          rng_extra="repair-replay")
    rendered_order = [row["question_id"] for row in replay.rows]
    rendered_set = set(rendered_order)
    have = {record["question"]["id"] for record in records}
    replay_questions = {row["question_id"]: row for row in replay.rows}

    for dependent in affected:
        if dependent in have and dependent not in rendered_set:
            removed = next(r for r in records if r["question"]["id"] == dependent)
            records.remove(removed)
            actions.append({"field": field_of.get(dependent, dependent), "qid": dependent,
                            "action": "invalidated (no longer routed)",
                            "original_value": removed["answer"]["answer_label"]})
        elif dependent in rendered_set and dependent not in have:
            question_payload = _question_payload_from_template(template, dependent)
            if mode in ("mock", "dry"):
                value = str((question_payload.get("options") or ["3"])[0])
            else:
                value = await _targeted_question_call(
                    twin=twin, index=index, records=records, qid=dependent,
                    position=rendered_order.index(dependent), template=template,
                    driver=driver, model=model, persona_type=persona_type,
                    seed=seed, semaphore=semaphore,
                )
            replay_row = replay_questions[dependent]
            records.append(
                {"question": {"id": dependent, "text": replay_row["question_text"],
                              "type": replay_row["question_type"]},
                 "answer": {"question_id": dependent, "answer_value": value,
                            "answer_label": value, "answer_raw": value,
                            "generated_after_repair": True}}
            )
            actions.append({"field": field_of.get(dependent, dependent), "qid": dependent,
                            "action": "generated (newly routed)", "new_value": value})

    # Reorder to the replayed survey order; anything the replay no longer
    # renders (outside the affected set this should not happen) keeps its
    # relative position at the end.
    order = {qid: i for i, qid in enumerate(rendered_order)}
    records.sort(key=lambda r: order.get(r["question"]["id"], len(order)))
    return actions


async def _targeted_question_call(
    *, twin: str, index: int, records: list, qid: str, position: int,
    template: dict, driver, model: str, persona_type: str, seed: int, semaphore,
) -> str:
    """One live call for one question, with the prior context the survey would
    have shown at that position."""
    from services.v3.qsf_runtime import (  # noqa: E402
        staged_prior_answer_context, staged_survey_sequence_with_prior_context,
    )
    from services.v2.simulation_executor import simulate_persona  # noqa: E402
    from services.v3.billing import BillingContext  # noqa: E402

    question_payload = _question_payload_from_template(template, qid)
    prior = [(r["question"], r["answer"]) for r in records][:position]
    scratch_runtime = driver._QsfRuntime(template, seed=seed, persona_count=driver.persona_count)
    scratch = scratch_runtime.new_state(persona_id=twin, persona_index=index)
    scratch_runtime.apply_answers(
        scratch, questions=[q for q, _ in prior], answers=[a for _, a in prior]
    )
    sequence = staged_survey_sequence_with_prior_context(
        prior_answer_context=staged_prior_answer_context(scratch),
        survey_sequence_text="",
    )
    async with semaphore:
        raw = await simulate_persona(
            model=model, persona_type=persona_type, persona_id=twin,
            questions=[question_payload], survey_sequence_text=sequence,
            billing_context=BillingContext(
                user_id="silicon-bench", feature="benchmark.repair.dependent",
                metadata={"persona_id": twin, "qid": qid},
            ),
        )
    answer = (raw or [{}])[0]
    return str(answer.get("answer_label") or answer.get("answer_value") or "")


def _question_payload_from_template(template: dict, qid: str) -> dict:
    """The instrument-builder question payload for one QID (same wording and
    options).  Blocks nested under Branches/Groups are searched too — the
    routing-dependent questions (partisan importance, born again, religiosity)
    live exactly there."""
    from services.v2.instrument_builder import collect_instrument_questions  # noqa: E402

    def blocks(elements):
        for element in elements or []:
            if not isinstance(element, dict):
                continue
            kind = str(element.get("ElementType") or element.get("Type") or "")
            if kind == "Block":
                yield element
            yield from blocks(element.get("Elements"))

    for block in blocks(template["Elements"]):
        for question in collect_instrument_questions([block]):
            if str(question.get("id")) == qid:
                return question
    raise KeyError(f"question {qid} not found in template")


def _apply_repair(records: list, qid: str, value: str) -> None:
    for record in records:
        if record["question"]["id"] == qid:
            record["answer"]["answer_value"] = value
            record["answer"]["answer_label"] = value
            record["answer"]["answer_raw"] = value
            record["answer"]["repaired"] = True
            return


# ---------------------------------------------------------------------------
# Section 9 — freeze
# ---------------------------------------------------------------------------

def run_freeze() -> int:
    qa = load_json(DATA / "prestudy_qa_repaired.json")
    meta = load_json(DATA / "prestudy_meta.json")
    eligibility = {r["source_twin_id"]: r for r in read_csv(DATA / "eligibility_audit.csv")}
    tags = _export_tags()

    long_rows, wide_rows = [], []
    wide_columns = ["persona_id", "age_band"]
    seen = set(wide_columns)
    frozen_qa = {}
    for twin in sorted(qa, key=lambda p: int(p.split("_")[1])):
        if eligibility.get(twin, {}).get("eligible") != "yes":
            continue
        frozen_qa[twin] = qa[twin]
        year = next((r["answer"]["answer_value"] for r in qa[twin]
                     if r["question"]["id"] == QID["year_birth"]), "")
        age_band = _age_band_from_year(year) or str(meta[twin]["embedded_data"].get("age_band", ""))
        row = {"persona_id": twin, "age_band": age_band}
        for record in qa[twin]:
            qid = record["question"]["id"]
            long_rows.append(
                {"persona_id": twin, "question_id": qid, "export_tag": tags.get(qid, ""),
                 "question_text": record["question"]["text"],
                 "question_type": record["question"]["type"],
                 "answer_value": record["answer"]["answer_value"],
                 "answer_label": record["answer"]["answer_label"],
                 "repaired": "yes" if record["answer"].get("repaired") else "no"}
            )
            column = tags.get(qid, qid)
            row[column] = record["answer"]["answer_label"] or record["answer"]["answer_value"]
            if column not in seen:
                seen.add(column)
                wide_columns.append(column)
        wide_rows.append(row)

    long_path = DATA / "prestudy_frozen_long.csv"
    wide_path = DATA / "prestudy_frozen_wide.csv"
    write_csv(long_rows, long_path,
              ["persona_id", "question_id", "export_tag", "question_text", "question_type",
               "answer_value", "answer_label", "repaired"])
    write_csv(wide_rows, wide_path, wide_columns)
    save_json(frozen_qa, DATA / "prestudy_frozen_qa.json")

    hashes = {
        "prestudy_frozen_long.csv": hashlib.sha256(long_path.read_bytes()).hexdigest(),
        "prestudy_frozen_wide.csv": hashlib.sha256(wide_path.read_bytes()).hexdigest(),
        "eligible_twins": len(wide_rows),
    }
    save_json(hashes, DATA / "prestudy_frozen_hashes.json")
    print(f"freeze: {len(wide_rows)} eligible twins frozen "
          f"({len(long_rows)} answers); hashes recorded")
    return 0


def _age_band_from_year(year: str) -> str:
    if not str(year).strip().isdigit():
        return ""
    y = int(year)
    if y >= 1997:
        return "18-29"
    if y >= 1982:
        return "30-44"
    if y >= 1967:
        return "45-59"
    return "60+"
