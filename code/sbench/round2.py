"""Practical planarian round 2 — resuming one session's randomization state.

Round 1 stops at ``FL_261``-``FL_264``, which branch on the state question
``QID1721185837``. Round 2 has to continue *that exact session*, not start a
fresh one, and it runs in a different process after the batch returns. Two
mechanisms carry the state across, and they are deliberately redundant:

1. **The same session seed.** ``session_seed(global_seed, base_pid,
   raw_condition)`` carries no stage component, so round 2 derives the identical
   integer from the identical inputs. Because ``QsfRuntime`` interpolates its
   seed into every randomization path, that alone reproduces FL_16's ordering
   and determines FL_49/FL_55 the same way round 1 would have.
2. **Explicitly restored state.** Round 1's resolved ``randomizer_choices`` are
   replayed into ``state.randomizer_choices``, where ``_randomizer_indexes``
   short-circuits on them before consulting the seed at all; round 1's answers
   are replayed through ``apply_answers`` so the runtime advances past stage 1
   and resolves the geographic branch on the real state answer; and round 1's
   ``displayed_qids`` are restored, without which every text-only block it
   already showed — the transitions and the treatment stimulus itself — would
   be emitted a second time, since blocks are gated on first display rather
   than on being answered.

Either mechanism would suffice. Keeping both means a divergence between them is
a test failure rather than a silent difference in a shipped prompt —
``verify_continuity`` asserts the seed-only path and the restored-state path
both reproduce the continuous run byte for byte.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .common import (
    ARTIFACTS,
    AUDIT,
    DATA,
    load_json,
    read_csv,
    session_seed,
    wire_worktree,
)
from .driver import StagedDriver, load_template
from .study import study_plan

PLANARIAN = "practical planarian"
STATE_QID = "QID1721185837"


def resume_inputs(
    *,
    frozen_pairs: list[tuple[dict, dict]],
    stage1_questions: list[dict],
    stage1_answers: list[dict],
    forced_choices: dict,
    stage1_randomizer_choices: dict | None = None,
) -> tuple[list[tuple[dict, dict]], dict]:
    """(prior_qa, forced_choices) that resume a session in a fresh runtime.

    ``prior_qa`` is the frozen pre-study record followed by round 1's own
    question/answer pairs, in order, so the runtime replays the session up to
    the branch and then resolves it on the real state answer.
    """
    prior = list(frozen_pairs) + list(zip(stage1_questions, stage1_answers, strict=False))
    choices = dict(forced_choices)
    if stage1_randomizer_choices:
        choices.update({k: list(v) for k, v in stage1_randomizer_choices.items()})
    return prior, choices


async def resume_session(
    driver: StagedDriver,
    *,
    base_pid: str,
    persona_index: int,
    global_seed: int,
    frozen_pairs: list[tuple[dict, dict]],
    stage1_questions: list[dict],
    stage1_answers: list[dict],
    forced_choices: dict,
    stage1_randomizer_choices: dict | None = None,
    stage1_displayed_qids: list[str] | None = None,
):
    """Render round 2 for one planarian session."""
    prior, choices = resume_inputs(
        frozen_pairs=frozen_pairs,
        stage1_questions=stage1_questions,
        stage1_answers=stage1_answers,
        forced_choices=forced_choices,
        stage1_randomizer_choices=stage1_randomizer_choices,
    )
    return await driver.run_twin(
        base_pid, persona_index,
        forced_choices=choices,
        prior_qa=prior,
        rng_extra=PLANARIAN,
        # No stage component: round 1 and round 2 derive the same integer.
        session_seed=session_seed(global_seed, base_pid, PLANARIAN),
        restore_displayed_qids=stage1_displayed_qids,
    )


def _route_blocks() -> dict[str, str]:
    audit = load_json(ARTIFACTS / "qsf_audit.json")
    return {
        b["block_id"]: b["block_name"]
        for b in audit["block_inventory"]
        if str(b["block_name"]).startswith("practical_planarian_")
        and b["block_name"] != "practical_planarian_state"
    }


async def verify_continuity(
    *, model: str, persona_type: str, seed: int, twins: int = 25,
) -> int:
    """Regression test: a split round 1 / round 2 must equal one continuous run.

    For each sampled twin this renders the planarian session three ways and
    requires the round-2 prompt to be byte-identical in all of them:

      reference   one run_twin, both stages in a single runtime
      restored    fresh runtime, same seed, round 1's answers AND randomizer
                  choices replayed
      seed-only   fresh runtime, same seed, round 1's answers replayed but
                  randomizer choices left to re-derive from the seed

    It also forces each of the four geographic routes through the state answer
    and asserts exactly one treatment block renders, followed by that session's
    own outcome-block ordering.
    """
    wire_worktree()
    from .batch import _install_persona_cache, _ensure_offline_provider

    _install_persona_cache()
    _ensure_offline_provider()

    template = load_template(ARTIFACTS / "template_poststudy.json")
    condition_map = load_json(ARTIFACTS / "condition_map.json")["conditions"]
    frozen_qa = load_json(DATA / "prestudy_frozen_qa.json")
    forced = condition_map[PLANARIAN]["forced_choices"]
    routes = _route_blocks()

    plan = study_plan(seed)
    eligible = sorted({p["base_pid"] for p in plan}, key=lambda t: int(t.split("_")[1]))
    index_of = {t: i for i, t in enumerate(eligible)}
    sample = eligible[:twins]

    driver = StagedDriver(template, mode="dry", model=model,
                          persona_type=persona_type, seed=seed,
                          persona_count=len(eligible))

    # Round 1's recorded hashes, when a round-1 build exists to compare against.
    recorded: dict[str, str] = {}
    plan_path = AUDIT / "batch" / "render_plan.csv"
    if Path(plan_path).exists():
        recorded = {
            r["base_pid"]: r["prompt_sha256"]
            for r in read_csv(plan_path) if r["raw_condition"] == PLANARIAN
        }

    results = {"restored": 0, "seed_only": 0, "order": 0, "round1": 0,
               "seed_match": 0, "n": 0}
    failures: list[dict] = []

    for twin in sample:
        results["n"] += 1
        pairs = [(r["question"], r["answer"]) for r in frozen_qa[twin]]
        sess = session_seed(seed, twin, PLANARIAN)

        reference = await driver.run_twin(
            twin, index_of[twin], forced_choices=forced, prior_qa=pairs,
            rng_extra=PLANARIAN, session_seed=sess,
        )
        if len(reference.stages) != 2:
            failures.append({"twin": twin, "why": f"{len(reference.stages)} stages, expected 2"})
            continue
        s1, s2 = reference.stages

        if reference.randomization_seed == sess:
            results["seed_match"] += 1
        if recorded.get(twin) in (None, s1.prompt_sha256):
            results["round1"] += 1
        else:
            failures.append({"twin": twin, "why": "round-1 build hash != continuous stage 1"})

        restored = await resume_session(
            driver, base_pid=twin, persona_index=index_of[twin], global_seed=seed,
            frozen_pairs=pairs, stage1_questions=s1.questions,
            stage1_answers=s1.normalized_answers, forced_choices=forced,
            stage1_randomizer_choices=reference.randomizer_choices,
            stage1_displayed_qids=s1.displayed_qids_after,
        )
        seed_only = await resume_session(
            driver, base_pid=twin, persona_index=index_of[twin], global_seed=seed,
            frozen_pairs=pairs, stage1_questions=s1.questions,
            stage1_answers=s1.normalized_answers, forced_choices=forced,
            stage1_randomizer_choices=None,
            stage1_displayed_qids=s1.displayed_qids_after,
        )
        for label, run in (("restored", restored), ("seed_only", seed_only)):
            if run.stages and run.stages[0].prompt_sha256 == s2.prompt_sha256:
                results[label] += 1
            else:
                got = run.stages[0].prompt_sha256[:12] if run.stages else "no stage"
                failures.append({"twin": twin, "why": f"{label} round-2 prompt differs",
                                 "want": s2.prompt_sha256[:12], "got": got})
        if restored.stages and restored.stages[0].rendered_block_ids == s2.rendered_block_ids:
            results["order"] += 1

    # Every geographic route must be reachable, one block at a time.
    route_hits: dict[str, str] = {}
    probe_twin = sample[0]
    pairs = [(r["question"], r["answer"]) for r in frozen_qa[probe_twin]]
    for state_label in ("Texas", "California", "New York", "Prefer not to say"):
        run = await driver.run_twin(
            probe_twin, index_of[probe_twin], forced_choices=forced, prior_qa=pairs,
            rng_extra=PLANARIAN, session_seed=session_seed(seed, probe_twin, PLANARIAN),
            overrides={STATE_QID: state_label},
        )
        rendered = [b for s in run.stages for b in s.rendered_block_ids]
        hit = [routes[b] for b in rendered if b in routes]
        route_hits[state_label] = "/".join(hit) if hit else "(none)"

    n = results["n"]
    print(f"=== planarian round-1 / round-2 continuity over {n} twins ===")
    print(f"  round 2 identical, state + choices restored : {results['restored']}/{n}")
    print(f"  round 2 identical, seed only (no choices)   : {results['seed_only']}/{n}")
    print(f"  round 2 block order identical               : {results['order']}/{n}")
    print(f"  round 1 matches the built batch prompt      : {results['round1']}/{n}"
          + ("" if recorded else "   (no round-1 build to compare)"))
    print(f"  session seed carries no stage component     : {results['seed_match']}/{n}")
    print()
    print("=== one geographic treatment per state answer ===")
    for state_label, hit in route_hits.items():
        print(f"  {state_label:22s} -> {hit}")

    exactly_one = all(h and "/" not in h for h in route_hits.values())
    distinct = len(set(route_hits.values())) == 4
    ok = (results["restored"] == n and results["seed_only"] == n
          and results["order"] == n and results["round1"] == n
          and results["seed_match"] == n and exactly_one and distinct)
    print()
    if failures:
        print(f"  {len(failures)} failure(s):")
        for f in failures[:6]:
            print(f"    {f}")
    print("CONTINUITY: PASS" if ok else "CONTINUITY: FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Round 2 builder — runs after the round-1 batch returns.
# ---------------------------------------------------------------------------

def _parse_batch_line(line: str) -> tuple[str, list[dict]] | None:
    """(custom_id, raw answers) from one OpenAI batch output line."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    custom_id = str(obj.get("custom_id") or "")
    body = ((obj.get("response") or {}).get("body")) or {}
    choices = body.get("choices") or []
    if not custom_id or not choices:
        return None
    content = ((choices[0] or {}).get("message") or {}).get("content") or ""
    try:
        from services.v2.llm_client import llm_client

        parsed = json.loads(llm_client._extract_json_string(content))
    except Exception:
        return None
    if isinstance(parsed, dict):
        parsed = parsed.get("answers") or parsed.get("responses") or []
    return (custom_id, parsed) if isinstance(parsed, list) else None


async def build_round2(
    *, model: str, persona_type: str, seed: int, concurrency: int,
    results_path: str,
) -> int:
    """Build planarian round-2 requests from the round-1 batch output.

    Stage 1 is re-rendered offline rather than read back from a file: its
    content sits before the branch, so it is a pure function of the session
    seed and the frozen pre-study record. Only the ANSWERS come from the batch.
    """
    wire_worktree()
    from .batch import (_ChunkWriter, _install_persona_cache, _ensure_offline_provider,
                        _newsletter_order)

    _install_persona_cache()
    _ensure_offline_provider()
    from services.v2.llm_client import llm_client
    from services.v2.simulation_executor import _answer_generation_max_tokens
    import services.v3.tool_registry  # noqa: F401  (breaks the registry<->tool cycle)
    from services.v3.tools.simulation_tool import _answers_with_fallbacks

    template = load_template(ARTIFACTS / "template_poststudy.json")
    condition_map = load_json(ARTIFACTS / "condition_map.json")["conditions"]
    frozen_qa = load_json(DATA / "prestudy_frozen_qa.json")
    forced = condition_map[PLANARIAN]["forced_choices"]
    routes = _route_blocks()

    plan = study_plan(seed)
    eligible = sorted({p["base_pid"] for p in plan}, key=lambda t: int(t.split("_")[1]))
    index_of = {t: i for i, t in enumerate(eligible)}
    planarian_plan = {p["run_id"]: p for p in plan if p["raw_condition"] == PLANARIAN}

    # A directory (or glob) is accepted so the 15 round-1 output files plus any
    # repair file can be read as one set; a single file still works.
    target = Path(results_path)
    if target.is_dir():
        sources = sorted(target.glob("*.jsonl"))
    else:
        sources = sorted(Path().glob(results_path)) or [target]
    if not sources:
        raise SystemExit(f"round2: no result files found at {results_path}")

    answers_by_run: dict[str, list[dict]] = {}
    unparsed = 0
    for src in sources:
        with open(src, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parsed = _parse_batch_line(line)
                if parsed is None:
                    unparsed += 1
                elif parsed[0] in planarian_plan:
                    answers_by_run[parsed[0]] = parsed[1]
    print(f"round2: read {len(sources)} result file(s); {unparsed} line(s) unparseable")
    print(f"round2: {len(answers_by_run)} of {len(planarian_plan)} planarian results parsed")

    # Hard gate. A partial result file must never yield a partial round 2: the
    # missing twins would silently drop out of the planarian arm and leave the
    # condition with fewer respondents than every other one, which is invisible
    # downstream. Fix the round-1 gap and re-run rather than proceeding.
    missing = sorted(set(planarian_plan) - set(answers_by_run))
    if missing:
        sample = [planarian_plan[r]["base_pid"] for r in missing[:5]]
        raise SystemExit(
            f"round2: REFUSING to build — {len(missing)} of {len(planarian_plan)} planarian "
            f"round-1 results are missing (e.g. {sample}). Every eligible twin must have a "
            f"round-1 answer before round 2 is generated; re-run the missing round-1 requests "
            f"and try again."
        )
    if len(answers_by_run) != 2007:
        print(f"round2: NOTE — planarian pool is {len(answers_by_run)}, not the expected 2,007 "
              f"(this is only correct if the eligible pool itself changed)")

    driver = StagedDriver(template, mode="dry", model=model, persona_type=persona_type,
                          seed=seed, persona_count=len(eligible))
    out_dir = DATA / "batch" / "round2"
    audit_dir = AUDIT / "batch"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    writer = _ChunkWriter(out_dir, "requests_round2")
    rows: list[dict] = []
    problems: list[dict] = []
    semaphore = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()

    async def one(run_id: str, raw_answers: list[dict]) -> None:
        item = planarian_plan[run_id]
        twin = item["base_pid"]
        pairs = [(r["question"], r["answer"]) for r in frozen_qa[twin]]
        sess = session_seed(seed, twin, PLANARIAN)
        async with semaphore:
            # Re-render stage 1 offline to recover its questions and displayed
            # set; it precedes the branch, so it cannot depend on the answers.
            stage1_run = await driver.run_twin(
                twin, index_of[twin], forced_choices=forced, prior_qa=pairs,
                rng_extra=PLANARIAN, session_seed=sess,
            )
            s1 = stage1_run.stages[0]
            normalized = _answers_with_fallbacks(
                answers=raw_answers, questions=s1.questions,
                embedded_data=s1.embedded_data,
                displayed_qids=set(s1.displayed_qids_after),
            )
            run = await resume_session(
                driver, base_pid=twin, persona_index=index_of[twin], global_seed=seed,
                frozen_pairs=pairs, stage1_questions=s1.questions,
                stage1_answers=normalized, forced_choices=forced,
                stage1_randomizer_choices=stage1_run.randomizer_choices,
                stage1_displayed_qids=s1.displayed_qids_after,
            )
        async with lock:
            if run.error or not run.stages:
                problems.append({"run_id": run_id, "base_pid": twin,
                                 "error": run.error or "no stage rendered"})
                return
            stage = run.stages[0]
            body = llm_client.build_chat_request_body(
                model=model,
                messages=[{"role": "system", "content": stage.system_prompt},
                          {"role": "user", "content": stage.user_prompt_head}],
                temperature=0.3,
                max_tokens=_answer_generation_max_tokens(len(stage.question_ids)),
                response_format={"type": "json_object"},
            )
            chunk = writer.write({"custom_id": run_id, "method": "POST",
                                  "url": "/v1/chat/completions", "body": body})
            hit = [routes[b] for b in stage.rendered_block_ids if b in routes]
            state_answer = next((str(a.get("answer_label", ""))
                                 for a in normalized
                                 if str(a.get("question_id")) == STATE_QID), "")
            rows.append({
                "run_id": run_id, "base_pid": twin, "condition": item["condition"],
                "raw_condition": PLANARIAN, "chunk_file": chunk, "stage_index": 2,
                "state_answer": state_answer,
                "route_block": "/".join(hit),
                "n_questions": len(stage.question_ids),
                "prompt_sha256": stage.prompt_sha256,
                "randomization_seed": run.randomization_seed,
                "block_order": "|".join(stage.rendered_block_ids),
                "newsletter_options": _newsletter_order(stage.user_prompt_head),
            })

    await asyncio.gather(*(one(rid, ans) for rid, ans in answers_by_run.items()))
    writer.close()

    from .common import write_csv
    if rows:
        rows.sort(key=lambda r: int(r["base_pid"].split("_")[1]))
        write_csv(rows, audit_dir / "render_plan_round2.csv", list(rows[0].keys()))

    multi = [r["run_id"] for r in rows if "/" in r["route_block"]]
    none_ = [r["run_id"] for r in rows if not r["route_block"]]
    seeds_ok = all(int(r["randomization_seed"]) == session_seed(seed, r["base_pid"], PLANARIAN)
                   for r in rows)
    print(f"round2: {len(rows)} requests in {len(writer.chunks)} chunk file(s)")
    print(f"  exactly one geographic route each : {not multi and not none_}"
          f"  ({len(multi)} multi, {len(none_)} none)")
    print(f"  session seed matches round 1      : {seeds_ok}")
    if problems:
        print(f"  PROBLEMS: {len(problems)}")
    (Path(out_dir) / "manifest.json").write_text(json.dumps({
        "round": 2, "model_alias": model, "persona_representation": persona_type,
        "seed": seed, "requests": len(rows), "chunks": writer.chunks,
        "endpoint": "/v1/chat/completions", "completion_window": "24h",
        "source_results": str(results_path),
        "continuity": "same session_seed as round 1; round 1 answers, randomizer "
                      "choices and displayed_qids restored before rendering",
        "problems": len(problems),
    }, indent=2), encoding="utf-8")
    return 0 if (rows and not problems and not multi and not none_ and seeds_ok) else 1
