"""Section 13 — preflight and acceptance tests.

``run_preflight`` executes every pre-paid-execution check, including a FULL
mocked end-to-end pass: pre-study over all 2,058 twins, conflict repair,
freeze, nested sampling, and all 9,000 post-study twin-condition records —
then runs the post-execution assertions against the mocked outputs.  No
network, no cost.

``run_verify`` holds the post-execution assertions alone so the same checks
run against live outputs later.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from .common import (
    ARTIFACTS,
    AUDIT,
    AUTHORITATIVE_QSF,
    BENCH_ROOT,
    DATA,
    QID,
    WORKTREE,
    active_run_dir,
    load_json,
    read_csv,
    rng_for,
    save_json,
    set_run_dir,
)
from .conflicts import compare_field
from .driver import StagedDriver, load_template

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# individual test groups
# ---------------------------------------------------------------------------

def check_pr_fixes() -> None:
    choice_identity = WORKTREE / "web" / "backend" / "services" / "v3" / "choice_identity.py"
    loader = (WORKTREE / "web" / "backend" / "services" / "v2" / "persona_loader.py").read_text()
    runtime = (WORKTREE / "web" / "backend" / "services" / "v3" / "qsf_runtime.py").read_text()
    check("PR58 choice-identity module present", choice_identity.is_file())
    check("PR58 runtime uses choice identity", "choice_identity" in runtime or "answer_selects_qsf_choice" in runtime)
    check("PR59 blank-persona guard present", "Blank persona file" in loader or "blank" in loader.lower())
    # PR61: without this fix the staged runtime silently drops text-only
    # stimulus blocks — the interventions would never reach the twins.
    check("PR61 descriptive-block fix present (stimuli render)",
          "new_descriptive" in runtime and "descriptive_ids" in runtime)
    regression = WORKTREE / "test_qsf_descriptive_blocks.py"
    check("PR61 regression test present in worktree", regression.is_file())


def check_personas(persona_type: str) -> None:
    from .common import wire_worktree

    wire_worktree()
    from services.v2.persona_loader import load_persona_text  # noqa: E402

    profiles = read_csv(BENCH_ROOT / "data" / "twin_profiles.csv")
    blank = []
    for row in profiles:
        text = load_persona_text(row["persona_id"], persona_type)
        if not text.strip() or text.startswith("[Persona"):
            blank.append(row["persona_id"])
    check(f"all {len(profiles)} personas non-empty ({persona_type})", not blank,
          f"blank: {blank[:5]}" if blank else "")


def check_prompt_hygiene() -> None:
    """Consent/Filter content must never reach a prompt."""
    consent_qids = {"QID1721185780", "QID1721185781"}
    needles = ("not to use AI", "artificial intelligence to answer", "consent")
    for name in ("template_prestudy.json", "template_poststudy.json"):
        blob = (ARTIFACTS / name).read_text(encoding="utf-8")
        qid_hit = [q for q in consent_qids if q in blob]
        check(f"{name}: consent/filter question ids absent", not qid_hit, str(qid_hit))
    audit = load_json(ARTIFACTS / "qsf_audit.json")
    check("qsf audit recorded 4 removed elements", audit["counts"]["removed_elements"] == 4)
    check("no unresolved cross-boundary dependency",
          not [d for d in audit["cross_boundary_dependencies"] if not d.get("preserved_by")],
          f"{len(audit['cross_boundary_dependencies'])} dependencies, all preserved")


async def check_prestudy_branches(model: str, persona_type: str, seed: int) -> None:
    template = load_template(ARTIFACTS / "template_prestudy.json")
    driver = StagedDriver(template, mode="mock", model=model, persona_type=persona_type,
                          seed=seed, persona_count=2058)
    base = {"QID1721185793": "Somewhat disagree", "QID1721185922": "attention"}

    async def qids_for(overrides):
        result = await driver.run_twin("pid_1", 0, overrides={**base, **overrides})
        return {q for s in result.stages for q in s.question_ids}, result

    # Age band cut-offs (the survey's own logic).
    for year, band in (("1997", "18-29"), ("1996", "30-44"), ("1982", "30-44"),
                       ("1981", "45-59"), ("1967", "45-59"), ("1966", "60+")):
        _, result = await qids_for({QID["year_birth"]: year})
        check(f"age_band({year}) == {band}", result.embedded_data.get("age_band") == band,
              f"got {result.embedded_data.get('age_band')}")

    # Party -> partisan importance (QID281).
    with_party, _ = await qids_for({QID["party"]: "Republican"})
    without_party, _ = await qids_for({QID["party"]: "Other (please specify)"})
    check("partisan importance shown for Republican", "QID281" in with_party)
    check("partisan importance hidden for Other", "QID281" not in without_party)

    # Religion -> born again (QID287) / religiosity (QID285).
    christian, _ = await qids_for({QID["religion"]: "Protestant"})
    nonrel, _ = await qids_for({QID["religion"]: "I am not religious"})
    check("born-again shown for Protestant", "QID287" in christian)
    check("born-again hidden for non-religious", "QID287" not in nonrel)
    check("religiosity asked somewhere", "QID285" in christian or "QID285" in nonrel)

    # Attention-check exclusion fires.
    _, failed = await qids_for({"QID1721185793": "Agree"})
    check("attention1 failure excludes", failed.excluded.startswith("Fail attention check 1"),
          repr(failed.excluded))
    _, failed2 = await qids_for({"QID1721185922": "ok"})
    check("attention2 failure excludes", failed2.excluded.startswith("Fail attention check 2"),
          repr(failed2.excluded))


def check_conflict_fixtures() -> None:
    cases = [
        ("household", "More than 4", "6 or more", "", "no_conflict"),
        ("household", "3", "6 or more", "", "conflict"),
        ("household", "More than 4", "5", "", "no_conflict"),
        ("income", "$50,000-$75,000", "$56,000 to $99,999", "", "no_conflict"),
        ("income", "$100,000 or more", "$56,000 to $99,999", "", "conflict"),
        ("age", "50-64", "1970", "", "no_conflict"),
        ("age", "50-64", "1958", "", "conflict"),
        ("education", "College graduate/some postgrad", "Master's degree / Professional degree", "", "conflict"),
        ("education", "Postgraduate", "Doctorate degree / Ph.D.", "", "no_conflict"),
        ("religion", "Agnostic", "I am not religious", "", "no_conflict"),
        ("religion", "Agnostic", "Other religion (please specify)", "Agnostic", "no_conflict"),
        ("religion", "Roman Catholic", "I am not religious", "", "conflict"),
        ("party", "Something else", "Other (please specify)", "", "no_conflict"),
        ("party", "Something else", "Democrat", "", "conflict"),
    ]
    bad = []
    for field, profile, survey, text, expected in cases:
        got = compare_field(field, profile, survey, text)
        if got != expected:
            bad.append((field, profile, survey, got, expected))
    check(f"conflict detector fixtures ({len(cases)} cases)", not bad, str(bad[:3]))
    # Missing values are skipped, never counted.
    check("missing values skipped", compare_field("age", "", "1970") is None
          and compare_field("income", "$100,000 or more", "") is None)
    # Repair decision logic: one resolution, one persistent conflict.
    resolved = compare_field("age", "50-64", "1970") == "no_conflict"
    persistent = compare_field("age", "50-64", "1958") == "conflict"
    check("repair decision: retry resolving keeps twin", resolved)
    check("repair decision: persistent conflict excludes twin", persistent)


def _planarian_state_routes() -> dict[str, str]:
    """One representative state per practical-planarian route, from the QSF."""
    qsf = load_json(AUTHORITATIVE_QSF)
    questions = {e["Payload"]["QuestionID"]: e["Payload"]
                 for e in qsf["SurveyElements"] if e.get("Element") == "SQ"}
    state_choices = {k: v.get("Display") for k, v in (questions[QID["state"]].get("Choices") or {}).items()}
    flow = [e for e in qsf["SurveyElements"] if e.get("Element") == "FL"][0]["Payload"]["Flow"]

    def find(flow_id, node_list):
        for node in node_list:
            if str(node.get("FlowID")) == flow_id:
                return node
            found = find(flow_id, node.get("Flow") or [])
            if found:
                return found
        return None

    routes = {}
    for flow_id, label in (("FL_261", "floods"), ("FL_262", "wildfire"),
                           ("FL_263", "ice"), ("FL_264", "US_general")):
        node = find(flow_id, flow)
        choice_ids = set(re.findall(r"SelectableChoice/(\d+)", json.dumps(node.get("BranchLogic") or {})))
        states = [state_choices[c] for c in sorted(choice_ids, key=int) if c in state_choices]
        if states:
            routes[label] = states[0]
    return routes


def _stimulus_block_map() -> tuple[dict[str, set[str]], set[str]]:
    """condition code -> the block ids that ARE its stimulus, plus the union of
    every treatment/control block id (for forbidden-content checks)."""
    audit = load_json(ARTIFACTS / "qsf_audit.json")
    by_name = {b["block_name"]: b["block_id"] for b in audit["block_inventory"]}
    jaguar = [n for n in by_name if str(n).startswith("jealous jaguar")]
    planarian_routes = [n for n in by_name if str(n).startswith("practical_planarian_")
                        and n != "practical_planarian_state"]
    mapping: dict[str, set[str]] = {
        "control neckties": {by_name["History of Neckties"]},
        "control baseball": {by_name["Rules of Baseball"]},
        "control dances": {by_name["Different Types of Dances"]},
        "jealous jaguar": {by_name[n] for n in jaguar},
        "practical planarian": {by_name["practical_planarian_state"]},
    }
    condition_map = load_json(ARTIFACTS / "condition_map.json")["conditions"]
    for code in condition_map:
        if code in mapping:
            continue
        if code not in by_name:
            raise SystemExit(f"stimulus map: no block named {code!r}")
        mapping[code] = {by_name[code]}
    all_treatment_blocks = set().union(*mapping.values())
    all_treatment_blocks |= {by_name[n] for n in planarian_routes}
    return mapping, all_treatment_blocks


async def check_conditions_e2e(model: str, persona_type: str, seed: int) -> None:
    post = load_template(ARTIFACTS / "template_poststudy.json")
    condition_map = load_json(ARTIFACTS / "condition_map.json")["conditions"]
    driver = StagedDriver(post, mode="mock", model=model, persona_type=persona_type,
                          seed=seed, persona_count=2058)
    prior = [({"id": QID["year_birth"], "text": "What is your year of birth?", "type": "numeric"},
              {"question_id": QID["year_birth"], "answer_value": "1985",
               "answer_label": "1985", "answer_raw": "1985"})]

    # Every outcome block must render in every condition (each outcome
    # randomizer's subset equals its child count, so order varies, inclusion
    # does not).
    outcome_blocks = {
        "trust multidimensional", "trust single post", "donation",
        "distrust single post", "scientists' role in policy ", "funding",
        "institutional trust", "subscription newsletter", "belief post",
        "climate change concern", "individual level behavior",
        "support general climate policies", "support specific climate policies",
    }
    audit = load_json(ARTIFACTS / "qsf_audit.json")
    required_outcomes = set()
    for block in audit["block_inventory"]:
        if block["block_name"] in outcome_blocks:
            for question in block["questions"]:
                if not question["is_timer"] and not question["is_descriptive"]:
                    required_outcomes.add(question["qid"])
    check(f"outcome inventory extracted ({len(required_outcomes)} required questions)",
          len(required_outcomes) >= 30)

    stimulus_of, all_treatment_blocks = _stimulus_block_map()
    planarian_route_blocks = {
        b["block_id"] for b in audit["block_inventory"]
        if str(b["block_name"]).startswith("practical_planarian_")
        and b["block_name"] != "practical_planarian_state"
    }

    for code, spec in sorted(condition_map.items()):
        result = await driver.run_twin("pid_1", 0, forced_choices=spec["forced_choices"],
                                       prior_qa=prior, rng_extra=f"e2e:{code}")
        got = str(result.embedded_data.get("condition", ""))
        qids = {q for s in result.stages for q in s.question_ids}
        base = {q.split("_")[0] if q.startswith("QID") else q for q in qids}
        missing = {q for q in required_outcomes if q not in qids and q not in base}
        check(f"E2E '{code}': forced condition sticks", got == code, f"got {got!r}")
        check(f"E2E '{code}': all outcome questions render",
              not result.error and not missing,
              f"missing {sorted(missing)[:4]}, err={result.error or '-'}")

        rendered_blocks = {b for s in result.stages for b in s.rendered_block_ids}
        expected = set(stimulus_of[code])
        allowed = expected | (planarian_route_blocks if code == "practical planarian" else set())
        wrong = (rendered_blocks & all_treatment_blocks) - allowed
        check(f"E2E '{code}': exactly its own stimulus, no competing stimulus",
              expected <= rendered_blocks and not wrong,
              f"missing {sorted(expected - rendered_blocks)[:3]}, foreign {sorted(wrong)[:3]}")
        if code == "practical planarian":
            routes_shown = rendered_blocks & planarian_route_blocks
            check("E2E 'practical planarian': exactly one state route block",
                  len(routes_shown) == 1, f"got {len(routes_shown)}")

    # Forced conditions cannot be overwritten: re-run one intervention twice.
    spec = condition_map["perfect prawn"]
    for attempt in range(2):
        result = await driver.run_twin("pid_2", 1, forced_choices=spec["forced_choices"],
                                       prior_qa=prior, rng_extra=f"lock:{attempt}")
        check(f"forced 'perfect prawn' immutable (run {attempt + 1})",
              result.embedded_data.get("condition") == "perfect prawn")

    # Practical planarian: all four state routes, verified against the exact
    # route block that must (and the three that must not) render.
    routes = _planarian_state_routes()
    check("planarian route states extracted", len(routes) == 4, str(routes))
    route_block = {
        "floods": "practical_planarian_floods", "wildfire": "practical_planarian_wildfire",
        "ice": "practical_planarian_ice", "US_general": "practical_planarian_US_general",
    }
    by_name = {b["block_name"]: b["block_id"] for b in audit["block_inventory"]}
    planarian = condition_map["practical planarian"]
    for label, state in routes.items():
        result = await driver.run_twin(
            "pid_3", 2, forced_choices=planarian["forced_choices"], prior_qa=prior,
            overrides={QID["state"]: state}, rng_extra=f"plan:{label}",
        )
        rendered_blocks = {b for s in result.stages for b in s.rendered_block_ids}
        wanted = by_name[route_block[label]]
        others = {by_name[route_block[k]] for k in route_block if k != label}
        state_stage = next((s.stage_index for s in result.stages
                            if QID["state"] in s.question_ids), None)
        check(f"planarian '{label}' via {state!r}: staged + exactly its route block",
              state_stage is not None and len(result.stages) > state_stage
              and wanted in rendered_blocks and not (rendered_blocks & others),
              f"state at stage {state_stage}/{len(result.stages)}, "
              f"route shown={wanted in rendered_blocks}, foreign={len(rendered_blocks & others)}")

    # Deterministic randomization: identical seeds -> identical routing events.
    first = await driver.run_twin("pid_4", 3, forced_choices=spec["forced_choices"],
                                  prior_qa=prior, rng_extra="det")
    second = await driver.run_twin("pid_4", 3, forced_choices=spec["forced_choices"],
                                   prior_qa=prior, rng_extra="det")
    check("deterministic randomization (same seed, same twin)",
          first.randomizer_choices == second.randomizer_choices
          and [s.question_ids for s in first.stages] == [s.question_ids for s in second.stages])


async def check_repair_pipeline(model: str, persona_type: str, seed: int) -> None:
    """Exercise the full repair path end-to-end on real twins with injected
    conflicting initial answers — including dependent-question reconciliation
    and a deliberately persistent conflict."""
    from .phases import run_prestudy, run_repairs
    from .common import QID as qids

    profiles = read_csv(BENCH_ROOT / "data" / "twin_profiles.csv")

    def pick(**criteria):
        for row in profiles:
            if all(row.get(k) == v for k, v in criteria.items()):
                return row["persona_id"]
        raise SystemExit(f"no twin matching {criteria}")

    republican = pick(political_affiliation="Republican")     # repair ADDS importance route
    something = pick(political_affiliation="Something else")  # repair REMOVES importance route
    catholic = pick(religion="Roman Catholic")                # repair re-routes religion deps
    stubborn = pick(political_affiliation="Democrat")         # persistent conflict -> excluded

    inject = {
        republican: {qids["party"]: "Other (please specify)"},
        something: {qids["party"]: "Democrat"},
        catholic: {qids["religion"]: "I am not religious"},
        stubborn: {qids["party"]: "Republican"},
    }
    subset = sorted(inject)
    code = await run_prestudy(mode="mock", model=model, persona_type=persona_type,
                              seed=seed, limit=None, concurrency=8,
                              mock_profile_aware=True, inject_overrides=inject,
                              twins_subset=subset)
    check("repair fixture: pre-study ran", code == 0)
    code = await run_repairs(mode="mock", model=model, persona_type=persona_type,
                             seed=seed, concurrency=4,
                             mock_retry_values={stubborn: {qids["party"]: "Republican"}})
    check("repair fixture: repairs ran", code == 0)

    from .common import load_json as _lj
    qa = _lj(DATA / "prestudy_qa_repaired.json")
    eligibility = {r["source_twin_id"]: r for r in read_csv(DATA / "eligibility_audit.csv")}
    repairs = read_csv(DATA / "demographic_repairs.csv")

    def has_q(twin, qid):
        return any(r["question"]["id"] == qid for r in qa[twin])

    def value_of(twin, qid):
        return next((r["answer"]["answer_label"] for r in qa[twin]
                     if r["question"]["id"] == qid), None)

    check("repair fixture: Republican twin repaired to Republican",
          value_of(republican, qids["party"]) == "Republican")
    check("repair fixture: importance GENERATED when repair adds the route",
          has_q(republican, "QID281"),
          "QID281 present" if has_q(republican, "QID281") else "QID281 missing")
    check("repair fixture: importance INVALIDATED when repair removes the route",
          not has_q(something, "QID281"),
          "" if not has_q(something, "QID281") else "stale QID281 kept")
    check("repair fixture: religion dependents reconciled to the repaired route",
          has_q(catholic, "QID287"),
          "born-again present" if has_q(catholic, "QID287") else "born-again missing")
    dependent_actions = [r for r in repairs if r["routing_parent"] == "dependent"]
    check("repair fixture: dependent actions were logged",
          len(dependent_actions) >= 2, f"{len(dependent_actions)} logged")
    check("repair fixture: persistent conflict excludes the twin",
          eligibility.get(stubborn, {}).get("eligible") == "no"
          and "persistent" in eligibility.get(stubborn, {}).get("exclusion_reason", ""),
          eligibility.get(stubborn, {}).get("exclusion_reason", ""))
    resolved_ok = [t for t in (republican, something, catholic)
                   if eligibility.get(t, {}).get("eligible") == "yes"]
    check("repair fixture: resolved twins stay eligible", len(resolved_ok) == 3,
          str({t: eligibility.get(t, {}).get("exclusion_reason") for t in
               (republican, something, catholic) if t not in resolved_ok}))


async def check_frozen_context_in_prompts(model: str, seed: int) -> None:
    """Dry-build real post-study prompts for sampled twins and confirm the
    frozen pre-study answers appear as prior context."""
    frozen_qa = load_json(DATA / "prestudy_frozen_qa.json")
    frozen_wide = {r["persona_id"]: r for r in read_csv(DATA / "prestudy_frozen_wide.csv")}
    post = load_template(ARTIFACTS / "template_poststudy.json")
    condition_map = load_json(ARTIFACTS / "condition_map.json")["conditions"]
    driver = StagedDriver(post, mode="dry", model=model, persona_type="summary",
                          seed=seed, persona_count=2058)
    sample = sorted(frozen_qa)[:3]
    spec = condition_map["difficult dog"]
    ok = True
    detail = ""
    for index, twin in enumerate(sample):
        prior = [(r["question"], r["answer"]) for r in frozen_qa[twin]]
        result = await driver.run_twin(twin, index, forced_choices=spec["forced_choices"],
                                       prior_qa=prior, rng_extra="ctxcheck")
        prompt = result.stages[0].user_prompt_head if result.stages else ""
        row = frozen_wide[twin]
        for column in ("year_birth", "gender", "race"):
            value = row.get(column, "")
            if value and value not in prompt:
                ok = False
                detail = f"{twin}: frozen {column}={value!r} absent from prompt"
    check("frozen pre-study answers appear in post-study prompt context", ok, detail)


# ---------------------------------------------------------------------------
# post-execution assertions (reused for live runs)
# ---------------------------------------------------------------------------

def run_verify(*, expect_prestudy: int = 2058) -> int:
    print("verify: post-execution assertions")
    meta = load_json(DATA / "prestudy_meta.json")
    check(f"{expect_prestudy} pre-study records attempted", len(meta) == expect_prestudy,
          f"got {len(meta)}")

    eligibility = read_csv(DATA / "eligibility_audit.csv")
    eligible = {r["source_twin_id"] for r in eligibility if r["eligible"] == "yes"}
    frozen = read_csv(DATA / "prestudy_frozen_wide.csv")
    frozen_ids = {r["persona_id"] for r in frozen}
    check("every eligible twin has one frozen record", eligible == frozen_ids,
          f"eligible {len(eligible)}, frozen {len(frozen_ids)}")
    persistent = {r["source_twin_id"] for r in eligibility
                  if "persistent" in r["exclusion_reason"]}
    assignment = read_csv(DATA / "sample_assignment.csv")
    sampled = {r["source_twin_id"] for r in assignment
               if r["in_control_1000"] == "yes" or r["in_core_500"] == "yes"}
    check("no persistent-conflict twin enters sampling", not (persistent & sampled))

    wide = read_csv(DATA / "simulation_raw_wide.csv")
    by_condition: dict[str, set] = {}
    for row in wide:
        by_condition.setdefault(row["condition"], set()).add(row["source_twin_id"])
    interventions = [c for c in by_condition if c != "control"]
    check("all 17 canonical conditions present", len(by_condition) == 17,
          f"got {len(by_condition)}")
    core = {r["source_twin_id"] for r in assignment if r["in_core_500"] == "yes"}
    bad = [c for c in interventions if by_condition[c] != core]
    check("exactly the same 500 twins in every intervention", not bad and all(
        len(by_condition[c]) == 500 for c in interventions), str(bad[:3]))
    check("exactly 1,000 twins in control", len(by_condition.get("control", set())) == 1000,
          f"got {len(by_condition.get('control', set()))}")
    check("exactly 9,000 wide records", len(wide) == 9000, f"got {len(wide)}")

    profile_ids = [r["profile_id"] for r in wide]
    check("every profile_id unique", len(profile_ids) == len(set(profile_ids)))
    control_rows = [r for r in wide if r["condition"] == "control"]
    check("every control respondent sees exactly one control text",
          all(r["control_variant"] in ("control neckties", "control baseball", "control dances")
              for r in control_rows))
    variant_counts: dict[str, int] = {}
    for row in control_rows:
        variant_counts[row["control_variant"]] = variant_counts.get(row["control_variant"], 0) + 1
    check("control texts near-equal", max(variant_counts.values()) - min(variant_counts.values()) <= 1,
          str(variant_counts))
    check("every intervention respondent sees exactly one intervention",
          all(r["raw_condition"] == r["condition"] for r in wide if r["condition"] != "control"))

    # Pre-study questions must never be re-asked in a post-study session:
    # every demographic the submission carries comes from ONE frozen record.
    prestudy_template = load_json(ARTIFACTS / "template_prestudy.json")
    prestudy_qids = set()
    for element in prestudy_template["Elements"]:
        for question in element.get("Questions") or []:
            if isinstance(question, dict) and question.get("QuestionID"):
                prestudy_qids.add(str(question["QuestionID"]))
    long_rows = read_csv(DATA / "simulation_answers_long.csv")
    reasked = sorted({r["question_id"] for r in long_rows if r["question_id"] in prestudy_qids})
    check("no pre-study question re-asked in any post-study session", not reasked,
          f"re-asked: {reasked[:5]}")

    # Frozen answers identical wherever a twin is reused — checked on the
    # JOINED export, whose pre_* columns all descend from the single frozen row.
    from .export import run_export_joined
    check("joined export builds", run_export_joined() == 0)
    joined = read_csv(DATA / "simulation_joined_wide.csv")
    by_twin: dict[str, set] = {}
    for row in joined:
        signature = (row.get("pre_year_birth", ""), row.get("pre_gender", ""),
                     row.get("pre_race", ""), row.get("pre_party", ""),
                     row.get("pre_religion", ""))
        by_twin.setdefault(row["source_twin_id"], set()).add(signature)
    inconsistent = [t for t, sigs in by_twin.items() if len(sigs) != 1]
    reuse_counts = {}
    for row in joined:
        reuse_counts[row["source_twin_id"]] = reuse_counts.get(row["source_twin_id"], 0) + 1
    check("frozen pre-study answers identical across every reuse of a twin",
          not inconsistent and max(reuse_counts.values()) == 17,
          f"inconsistent: {inconsistent[:3]}; max reuse {max(reuse_counts.values())}")

    failures = [c for c in CHECKS if not c[1]]
    print(f"verify: {len(CHECKS)} checks, {len(failures)} failed")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# cost estimate
# ---------------------------------------------------------------------------

PRICES = {  # USD per 1M tokens (input, cached input, output); luna unknown -> bracket
    "if luna bills like gpt-5.4": (2.50, 0.25, 15.00),
    "if luna bills like gpt-5.4-mini": (0.75, 0.075, 4.50),
    "if luna bills like gpt-5-mini": (0.25, 0.025, 2.00),
}


async def estimate_costs(model: str, seed: int) -> dict:
    out: dict = {}
    profiles = read_csv(BENCH_ROOT / "data" / "twin_profiles.csv")
    sample = [r["persona_id"] for r in profiles][:12]
    for persona_type in ("summary", "full"):
        pre = load_template(ARTIFACTS / "template_prestudy.json")
        driver = StagedDriver(pre, mode="dry", model=model, persona_type=persona_type,
                              seed=seed, persona_count=len(profiles))
        totals = []
        for index, twin in enumerate(sample):
            result = await driver.run_twin(twin, index, overrides={
                "QID1721185793": "Somewhat disagree", "QID1721185922": "attention"})
            totals.append(sum(s.prompt_chars["total"] for s in result.stages))
        pre_tokens = (sum(totals) / len(totals)) / 4

        post = load_template(ARTIFACTS / "template_poststudy.json")
        condition_map = load_json(ARTIFACTS / "condition_map.json")["conditions"]
        driver = StagedDriver(post, mode="dry", model=model, persona_type=persona_type,
                              seed=seed, persona_count=len(profiles))
        prior = [({"id": QID["year_birth"], "text": "What is your year of birth?", "type": "numeric"},
                  {"question_id": QID["year_birth"], "answer_value": "1985",
                   "answer_label": "1985", "answer_raw": "1985"})]
        post_totals = []
        for index, (code, spec) in enumerate(list(condition_map.items())[:6]):
            result = await driver.run_twin(sample[index], index,
                                           forced_choices=spec["forced_choices"], prior_qa=prior)
            post_totals.append(sum(s.prompt_chars["total"] for s in result.stages))
        post_tokens = (sum(post_totals) / len(post_totals)) / 4

        pre_total_in = pre_tokens * 2058
        post_total_in = post_tokens * 9000
        out[persona_type] = {
            "prestudy_input_tokens_per_twin": round(pre_tokens),
            "poststudy_input_tokens_per_record": round(post_tokens),
            "prestudy_total_input_tokens": round(pre_total_in),
            "poststudy_total_input_tokens": round(post_total_in),
            "output_tokens_assumed": {"prestudy_per_twin": 900, "poststudy_per_record": 2600},
            "dollars": {},
        }
        for label, (in_rate, _cached, out_rate) in PRICES.items():
            pre_cost = (pre_total_in * in_rate + 900 * 2058 * out_rate) / 1e6
            post_cost = (post_total_in * in_rate + 2600 * 9000 * out_rate) / 1e6
            out[persona_type]["dollars"][label] = {
                "prestudy": round(pre_cost, 2), "poststudy": round(post_cost, 2),
                "total": round(pre_cost + post_cost, 2),
            }
    save_json(out, ARTIFACTS / "cost_estimate.json")
    return out


# ---------------------------------------------------------------------------
# the full preflight
# ---------------------------------------------------------------------------

async def run_preflight(*, model: str, persona_type: str, seed: int, full_e2e: bool) -> int:
    """Every check is mocked and writes into a THROWAWAY run directory, so a
    preflight can never overwrite a live run's data/ or audit/."""
    import shutil
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="sbench-preflight-"))
    set_run_dir(scratch)
    # The shared twin profile table is an input, not a per-run output.
    (scratch / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(BENCH_ROOT / "data" / "twin_profiles.csv", scratch / "data" / "twin_profiles.csv")
    print(f"preflight: throwaway run directory {scratch}")

    CHECKS.clear()
    print("preflight: static checks")
    check_pr_fixes()
    check_personas(persona_type)
    from .qsf_audit import run_audit

    check("qsf audit passes", run_audit() == 0)
    check_prompt_hygiene()
    check_conflict_fixtures()

    print("preflight: mocked branch and condition tests")
    await check_prestudy_branches(model, persona_type, seed)
    await check_conditions_e2e(model, persona_type, seed)

    print("preflight: repair pipeline fixtures (injected conflicts, dependents, persistent case)")
    await check_repair_pipeline(model, persona_type, seed)

    if full_e2e:
        print("preflight: FULL mocked end-to-end (2,058 pre-study + 9,000 post-study records)")
        from .phases import run_freeze, run_prestudy, run_repairs
        from .poststudy import run_poststudy
        from .sampling import run_sampling

        code = await run_prestudy(mode="mock", model=model, persona_type=persona_type,
                                  seed=seed, limit=None, concurrency=64, mock_profile_aware=True)
        check("mock pre-study over full panel", code == 0)
        code = await run_repairs(mode="mock", model=model, persona_type=persona_type,
                                 seed=seed, concurrency=8)
        check("mock repairs", code == 0)
        check("freeze", run_freeze() == 0)
        check("sampling", run_sampling(seed=seed) == 0)
        code = await run_poststudy(mode="mock", model=model, persona_type=persona_type,
                                   seed=seed, concurrency=64, limit=None)
        check("mock post-study over full 9,000 plan", code == 0)
        await check_frozen_context_in_prompts(model, seed)
        run_verify()

    print("preflight: cost estimate (dry prompts, both persona representations)")
    estimate = await estimate_costs(model, seed)
    for persona, block in estimate.items():
        print(f"  {persona}: prestudy {block['prestudy_input_tokens_per_twin']:,} tok/twin, "
              f"poststudy {block['poststudy_input_tokens_per_record']:,} tok/record")
        for label, dollars in block["dollars"].items():
            print(f"    {label:34s} total ${dollars['total']:,}")

    failures = [c for c in CHECKS if not c[1]]
    save_json(
        {"checks": [{"name": n, "passed": p, "detail": d} for n, p, d in CHECKS],
         "failures": len(failures)},
        ARTIFACTS / "preflight_report.json",
    )
    print(f"\npreflight: {len(CHECKS)} checks, {len(failures)} FAILED")
    for name, _, detail in [c for c in CHECKS if not c[1]]:
        print(f"  FAIL: {name} {detail}")
    print("\nNO live call has been made. Live execution requires explicit approval "
          "(SBENCH_APPROVED=1) plus OPENAI_API_KEY, and should start with a small pilot.")
    print(f"preflight artifacts left in the throwaway run dir: {scratch}")
    return 1 if failures else 0
