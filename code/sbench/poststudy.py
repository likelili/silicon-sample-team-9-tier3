"""Sections 11-12 — post-study simulations and raw outputs.

Design enforced here:
  * the same core 500 twins in every one of the 16 interventions;
  * the nested 1,000-twin sample in pooled control, one control text per twin,
    spread deterministically near-equally across the three texts;
  * every twin x condition combination is a fresh session — a new runtime and
    state, no memory of any other condition;
  * the twin's frozen pre-study questions and answers are seeded into the
    state before the first render, so they appear as prior survey context in
    their original order and satisfy any reference to a pre-study answer;
  * condition assignment is forced by pre-seeding ``state.randomizer_choices``
    from the extracted condition map — the treatment randomizer never draws;
  * every other randomizer (pre-treatment order, jealous-jaguar sub-blocks,
    outcome block order, choice order) runs exactly as authored, seeded
    deterministically by the runtime;
  * the practical-planarian condition stays staged: the state question is
    asked, its answer applied through the runtime, and exactly one of the four
    state blocks renders.

Outputs are raw only — no calibration, weighting, effect estimation, or
submission formatting happens here.
"""

from __future__ import annotations

import asyncio
import json

from .common import (
    ARTIFACTS,
    AUDIT,
    DATA,
    DEFAULT_SEED,
    append_jsonl,
    condition_codes,
    load_json,
    read_csv,
    require_live_flag,
    rng_for,
    save_json,
    write_csv,
)
from .driver import StagedDriver, load_template
from .telemetry import TELEMETRY_COLUMNS, capture, summarize

CONTROL_TEXTS = ("control neckties", "control baseball", "control dances")


def condition_plan(seed: int) -> list[dict]:
    """The full 9,000-record plan: 16 interventions x core 500 + control 1,000."""
    assignment = read_csv(DATA / "sample_assignment.csv")
    core = [r["source_twin_id"] for r in assignment if r["in_core_500"] == "yes"]
    control = [r["source_twin_id"] for r in assignment if r["in_control_1000"] == "yes"]
    rows = condition_codes()
    code_field = next(k for k in rows[0] if "code" in k.lower())
    interventions = sorted({r[code_field] for r in rows} - set(CONTROL_TEXTS))

    plan: list[dict] = []
    for condition in interventions:
        for twin in core:
            plan.append({"source_twin_id": twin, "raw_condition": condition,
                         "condition": condition, "control_variant": ""})
    # Deterministic near-equal split of the 1,000 control twins across texts.
    ordered = sorted(control, key=lambda t: int(t.split("_")[1]))
    rng = rng_for(seed, "control-variant")
    rng.shuffle(ordered)
    for index, twin in enumerate(ordered):
        text = CONTROL_TEXTS[index % 3]
        plan.append({"source_twin_id": twin, "raw_condition": text,
                     "condition": "control", "control_variant": text})
    return plan


async def run_poststudy(
    *, mode: str, model: str, persona_type: str, seed: int,
    concurrency: int, limit: int | None,
) -> int:
    require_live_flag(mode == "live", "poststudy")
    template = load_template(ARTIFACTS / "template_poststudy.json")
    condition_map = load_json(ARTIFACTS / "condition_map.json")["conditions"]
    frozen_qa = load_json(DATA / "prestudy_frozen_qa.json")
    plan = condition_plan(seed)
    if limit:
        plan = plan[:limit]

    audit = load_json(ARTIFACTS / "qsf_audit.json")
    tags = {}
    for block in audit["block_inventory"]:
        for question in block["questions"]:
            if question.get("export_tag"):
                tags[question["qid"]] = question["export_tag"]

    driver = StagedDriver(template, mode=mode, model=model, persona_type=persona_type,
                          seed=seed, persona_count=len(frozen_qa) or 2058)

    stages_path = AUDIT / "poststudy_stages.jsonl"
    prompts_path = AUDIT / "poststudy_prompts.jsonl"
    for path in (stages_path, prompts_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    long_rows: list[dict] = []
    call_rows: list[dict] = []
    wide_rows: list[dict] = []
    failures: list[dict] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, item: dict):
        twin = item["source_twin_id"]
        raw_condition = item["raw_condition"]
        profile_id = f"{twin}::{raw_condition}"
        forced = condition_map[raw_condition]["forced_choices"]
        prior = [(r["question"], r["answer"]) for r in frozen_qa[twin]]
        async with semaphore:
            result = await driver.run_twin(
                twin, index, forced_choices=forced, prior_qa=prior,
                rng_extra=f"post:{raw_condition}",
                billing_extra={"raw_condition": raw_condition, "profile_id": profile_id},
            )
        if result.error:
            failures.append({"profile_id": profile_id, "error": result.error})
            return
        got_condition = str(result.embedded_data.get("condition", ""))
        if got_condition != raw_condition:
            failures.append({"profile_id": profile_id,
                             "error": f"forced condition mismatch: wanted {raw_condition!r}, "
                                      f"runtime recorded {got_condition!r}"})
            return
        for stage in result.stages:
            append_jsonl(
                {"profile_id": profile_id, "source_twin_id": twin,
                 "raw_condition": raw_condition, "stage_index": stage.stage_index,
                 "question_ids": stage.question_ids, "prompt_chars": stage.prompt_chars,
                 "prior_answer_count": stage.prior_answer_count,
                 "embedded_data": stage.embedded_data, "events": stage.events,
                 "retries": stage.retries, "latency_ms": stage.latency_ms, "mode": mode},
                stages_path,
            )
            if mode != "mock":
                append_jsonl({"profile_id": profile_id, "stage_index": stage.stage_index,
                              "system_prompt": stage.system_prompt,
                              "user_prompt": stage.user_prompt_head}, prompts_path)
            call_rows.append({"profile_id": profile_id, "source_twin_id": twin,
                              "raw_condition": raw_condition, "stage_index": stage.stage_index,
                              "question_count": len(stage.question_ids),
                              "prompt_chars_total": stage.prompt_chars["total"],
                              "latency_ms": stage.latency_ms, "retries": stage.retries,
                              "mode": mode})
        stage_of_qid = {}
        for stage in result.stages:
            for qid in stage.question_ids:
                stage_of_qid[qid] = stage.stage_index
        wide = {"profile_id": profile_id, "source_twin_id": twin,
                "raw_condition": raw_condition, "condition": item["condition"],
                "control_variant": item["control_variant"],
                "model": model, "persona_representation": persona_type,
                "seed": seed, "excluded": result.excluded}
        for row in result.rows:
            qid = row["question_id"]
            long_rows.append(
                {"source_twin_id": twin, "profile_id": profile_id,
                 "raw_condition": raw_condition, "condition": item["condition"],
                 "control_variant": item["control_variant"],
                 "question_id": qid, "export_tag": tags.get(qid, ""),
                 "stage_index": stage_of_qid.get(qid, ""),
                 "question_text": row["question_text"],
                 "question_type": row["question_type"],
                 "answer_raw": row["answer_raw"], "answer_value": row["answer_value"],
                 "answer_label": row["answer_label"],
                 "answer_status": row.get("answer_status", ""),
                 "embedded_data": json.dumps(result.embedded_data, ensure_ascii=False),
                 "model": model, "persona_representation": persona_type, "seed": seed,
                 "retry_status": max((s.retries for s in result.stages), default=0)}
            )
            wide[tags.get(qid, qid)] = row["answer_label"] or row["answer_value"]
        wide_rows.append(wide)

    with capture("poststudy") as telemetry:
        await asyncio.gather(*(one(i, item) for i, item in enumerate(plan)))

    write_csv(long_rows, DATA / "simulation_answers_long.csv",
              ["source_twin_id", "profile_id", "raw_condition", "condition", "control_variant",
               "question_id", "export_tag", "stage_index", "question_text", "question_type",
               "answer_raw", "answer_value", "answer_label", "answer_status", "embedded_data",
               "model", "persona_representation", "seed", "retry_status"])
    wide_columns: list[str] = []
    seen: set[str] = set()
    for row in wide_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                wide_columns.append(key)
    write_csv(wide_rows, DATA / "simulation_raw_wide.csv", wide_columns)
    write_csv(call_rows, AUDIT / "poststudy_calls.csv",
              ["profile_id", "source_twin_id", "raw_condition", "stage_index",
               "question_count", "prompt_chars_total", "latency_ms", "retries", "mode"])
    provider_stats = {}
    if telemetry:
        write_csv(telemetry, AUDIT / "poststudy_provider_calls.csv", TELEMETRY_COLUMNS)
        provider_stats = summarize(telemetry)
        save_json(provider_stats, AUDIT / "poststudy_provider_summary.json")
        print(f"  provider: {provider_stats['calls']} calls, "
              f"{provider_stats['total_tokens']:,} tokens "
              f"({provider_stats['reasoning_tokens']:,} reasoning), "
              f"realized ${provider_stats['realized_provider_cost_usd']:.4f}")
    save_json(
        {"mode": mode, "model": model, "persona_representation": persona_type, "seed": seed,
         "planned_records": len(plan), "completed_records": len(wide_rows),
         "failures": failures, "provider": provider_stats},
        DATA / "run_manifest.json",
    )
    print(f"poststudy[{mode}]: planned {len(plan)}, completed {len(wide_rows)}, "
          f"failures {len(failures)}")
    for failure in failures[:8]:
        print("  FAIL:", failure)
    return 1 if failures else 0
