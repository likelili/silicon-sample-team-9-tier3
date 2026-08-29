"""Round-1 Batch API request builder — every prompt, zero network calls.

Fidelity is the whole point of this module, so it reuses the production path
rather than reconstructing it:

  * ``StagedDriver`` in ``dry`` mode renders each session's first stage through
    the real ``QsfRuntime`` — same flow evaluation, same seeded randomizers,
    same forced condition, same frozen pre-study answers applied first.
  * The recorded stage carries exactly the ``(system, user)`` pair that
    ``build_simulation_prompt_messages`` produces for the live call
    (``user`` is ``persona_prompt + "\\n\\n" + questions_prompt``).
  * The request body comes from the platform's own
    ``LLMClient.build_chat_request_body`` with the same ``temperature=0.3``,
    ``response_format={"type": "json_object"}`` and
    ``_answer_generation_max_tokens(n)`` sizing that ``chat_json`` uses live.

Round 1 covers every session exactly once: the 16 single-stage interventions,
the pooled control, and practical planarian's FIRST stage. Planarian's second
stage is deliberately absent — it branches on the state answer
(``QID1721185837``), so its prompt cannot exist until round 1 comes back.

``dry`` mode advances the runtime on mock answers after a stage is recorded.
That is harmless for every single-stage session (stage 1 is the whole session)
but it means a planarian *second* stage rendered here would sit on a mock state
answer. It is never emitted, and the builder asserts that.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path

from .common import (
    ARTIFACTS,
    AUDIT,
    DATA,
    load_json,
    session_seed,
    write_csv,
    wire_worktree,
)
from .driver import StagedDriver, load_template
from .study import study_plan

# OpenAI caps a batch input file at 200 MB and 50,000 requests. Size binds
# first here (~71 KB per line).
#
# Decimal bytes deliberately: 190 MiB is 199.2 MB, which sat 0.4% under a cap
# that is quoted in decimal — close enough that one unusually long prompt could
# push a file over. 150 MB leaves a quarter of the cap as headroom for the cost
# of a few more files.
MAX_CHUNK_BYTES = 150_000_000
MAX_CHUNK_REQUESTS = 45_000

PLANARIAN = "practical planarian"


def _install_persona_cache() -> None:
    """Cache persona text across a twin's 17 sessions.

    The driver re-imports ``load_persona_text`` on every call, so replacing the
    module attribute is enough. Persona text is a pure file read, so caching
    cannot change a prompt — it only avoids re-reading the same file 17 times.
    """
    import services.v2.persona_loader as loader

    if getattr(loader, "_sbench_persona_cache_installed", False):
        return
    original = loader.load_persona_text
    cache: dict[tuple[str, str], str] = {}

    def cached(persona_id: str, persona_type: str) -> str:
        key = (str(persona_id), str(persona_type))
        if key not in cache:
            cache[key] = original(persona_id, persona_type)
        return cache[key]

    loader.load_persona_text = cached
    loader._sbench_persona_cache_installed = True


def _ensure_offline_provider() -> str:
    """Give the client a placeholder key so the build needs no real secret.

    ``build_chat_request_body`` is pure — it only reads the key's *presence* to
    pick a provider, and the OpenAI branch passes ``model_name`` through
    untouched. A placeholder therefore yields a byte-identical request body
    while guaranteeing this step cannot authenticate against anything.
    """
    import os

    real = os.environ.get("OPENAI_API_KEY", "")
    os.environ["OPENAI_API_KEY"] = "offline-prompt-build-no-network"
    import services.v2.llm_client as module  # noqa: E402

    module.llm_client.openai_api_key = os.environ["OPENAI_API_KEY"]
    module.llm_client.openrouter_api_key = ""
    return "placeholder (real key was present but is not used)" if real else "placeholder"


# Randomizers that run BEFORE the condition branch. Everything else sits after
# the treatment, so planarian's round 1 legitimately stops short of it.
_PRE_TREATMENT_FLOWS = {"FL_16"}

# The newsletter question is the only one in the study half carrying choice
# randomization, so its rendered option order is the direct read on whether
# choice order is session-specific.
_NEWSLETTER_QID = "QID1721186006"


def _randomizer_groups(template: dict) -> dict[str, list[str]]:
    """{FlowID: [child block ids]} for every randomizer whose children are blocks.

    Derived from the template rather than hardcoded so a QSF change shows up as
    a coverage gap instead of quietly shrinking what gets validated. Randomizers
    over embedded-data children (the forced condition assignment) are skipped —
    they carry no block ordering to audit.
    """
    groups: dict[str, list[str]] = {}

    def walk(elements) -> None:
        for element in elements or []:
            if not isinstance(element, dict):
                continue
            if str(element.get("ElementType") or element.get("Type")) == "Randomizer":
                children = [
                    str(child.get("BlockID"))
                    for child in (element.get("Elements") or [])
                    if isinstance(child, dict)
                    and str(child.get("ElementType") or child.get("Type")) == "Block"
                    and child.get("BlockID")
                ]
                if len(children) > 1:
                    groups[str(element.get("FlowID"))] = children
            walk(element.get("Elements"))

    walk(template.get("Elements"))
    return groups


def _newsletter_order(user_prompt: str) -> str:
    """The rendered option order for the one choice-randomized question."""
    marker = f'"id": "{_NEWSLETTER_QID}"'
    at = user_prompt.find(marker)
    if at < 0:
        return ""
    opts = user_prompt.find('"options":', at)
    if opts < 0 or opts - at > 1200:
        return ""
    start = user_prompt.find("[", opts)
    end = user_prompt.find("]", start)
    if start < 0 or end < 0:
        return ""
    try:
        return "|".join(json.loads(user_prompt[start:end + 1]))
    except (ValueError, TypeError):
        return ""


def _git_state(path) -> dict:
    """HEAD and cleanliness of a repo, so a manifest can name exact code."""
    import subprocess

    def run(*args) -> str:
        try:
            return subprocess.run(["git", "-C", str(path), *args],
                                  capture_output=True, text=True, timeout=15).stdout.strip()
        except Exception:
            return ""

    head = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {
        "commit": head or "unknown",
        "clean": head != "" and dirty == "",
        "dirty_paths": [line[2:].strip() for line in dirty.splitlines()[:20]] if dirty else [],
    }


def _pipeline_source_sha256() -> str:
    """Content hash of the sbench package.

    Recorded alongside the git commit so provenance still pins the exact code
    if the tree is dirty — a commit id alone would be misleading then.
    """
    here = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(here.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _stimulus_map() -> tuple[dict[str, set[str]], set[str], set[str]]:
    """(raw_condition -> its stimulus block ids, planarian routes, all stimuli).

    Same derivation the study validator uses, so the pre-run audit and the
    post-run audit judge stimulus correctness by one definition.
    """
    audit = load_json(ARTIFACTS / "qsf_audit.json")
    by_name = {b["block_name"]: b["block_id"] for b in audit["block_inventory"]}
    jaguar = {by_name[n] for n in by_name if str(n).startswith("jealous jaguar")}
    routes = {by_name[n] for n in by_name
              if str(n).startswith("practical_planarian_") and n != "practical_planarian_state"}
    stimulus_of: dict[str, set[str]] = {
        "control neckties": {by_name["History of Neckties"]},
        "control baseball": {by_name["Rules of Baseball"]},
        "control dances": {by_name["Different Types of Dances"]},
        "jealous jaguar": jaguar,
        PLANARIAN: {by_name["practical_planarian_state"]},
    }
    for code in load_json(ARTIFACTS / "condition_map.json")["conditions"]:
        stimulus_of.setdefault(code, {by_name[code]} if code in by_name else set())
    return stimulus_of, routes, set().union(*stimulus_of.values()) | routes


class _ChunkWriter:
    """Streams JSONL request lines into size- and count-bounded chunk files."""

    def __init__(self, directory: Path, stem: str) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stem = stem
        self.index = 0
        self.handle = None
        self.bytes = 0
        self.count = 0
        self.chunks: list[dict] = []

    def _open(self) -> None:
        path = self.dir / f"{self.stem}_{self.index:03d}.jsonl"
        self.handle = open(path, "w", encoding="utf-8")
        self.bytes = 0
        self.count = 0

    def write(self, obj: dict) -> str:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        payload = len(line.encode("utf-8"))
        if self.handle is None:
            self._open()
        elif self.bytes + payload > MAX_CHUNK_BYTES or self.count + 1 > MAX_CHUNK_REQUESTS:
            self.close()
            self.index += 1
            self._open()
        self.handle.write(line)
        self.bytes += payload
        self.count += 1
        return f"{self.stem}_{self.index:03d}.jsonl"

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.chunks.append({
                "file": f"{self.stem}_{self.index:03d}.jsonl",
                "requests": self.count,
                "bytes": self.bytes,
                "mb": round(self.bytes / 1e6, 2),
            })
            self.handle = None


async def build_round1(
    *, model: str, persona_type: str, seed: int, concurrency: int,
    limit: int | None = None, pilot_twins: int | None = None,
) -> int:
    """Render every round-1 prompt and write submittable batch files."""
    wire_worktree()
    _install_persona_cache()
    key_mode = _ensure_offline_provider()
    from services.v2.llm_client import llm_client  # noqa: E402
    from services.v2.simulation_executor import _answer_generation_max_tokens  # noqa: E402

    print(f"batch-build: OFFLINE prompt build — provider key is a {key_mode}; "
          f"no request is sent", flush=True)

    template = load_template(ARTIFACTS / "template_poststudy.json")
    condition_map = load_json(ARTIFACTS / "condition_map.json")["conditions"]
    frozen_qa = load_json(DATA / "prestudy_frozen_qa.json")
    plan = study_plan(seed)

    if pilot_twins:
        chosen = set(sorted({p["base_pid"] for p in plan},
                            key=lambda t: int(t.split("_")[1]))[:pilot_twins])
        plan = [p for p in plan if p["base_pid"] in chosen]
    if limit:
        plan = plan[:limit]

    eligible_order = sorted({p["base_pid"] for p in plan}, key=lambda t: int(t.split("_")[1]))
    index_of = {t: i for i, t in enumerate(eligible_order)}
    driver = StagedDriver(template, mode="dry", model=model, persona_type=persona_type,
                          seed=seed, persona_count=len(eligible_order))

    stimulus_of, planarian_routes, all_stimuli = _stimulus_map()

    out_dir = DATA / "batch" / "round1"
    audit_dir = AUDIT / "batch"
    examples_dir = audit_dir / "examples"
    for d in (out_dir, audit_dir, examples_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    writer = _ChunkWriter(out_dir, "requests")
    render_rows: list[dict] = []
    example_written: set[str] = set()
    problems: list[dict] = []
    semaphore = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    done = 0
    total = len(plan)

    async def one(item: dict) -> None:
        nonlocal done
        twin = item["base_pid"]
        raw_condition = item["raw_condition"]
        forced = condition_map[raw_condition]["forced_choices"]
        prior = [(r["question"], r["answer"]) for r in frozen_qa[twin]]

        sess_seed = session_seed(seed, twin, raw_condition)
        async with semaphore:
            result = await driver.run_twin(
                twin, index_of[twin], forced_choices=forced, prior_qa=prior,
                rng_extra=raw_condition, session_seed=sess_seed,
                billing_extra={"run_id": item["run_id"], "raw_condition": raw_condition},
            )

        async with lock:
            nonlocal done
            done += 1
            if result.error or not result.stages:
                problems.append({"run_id": item["run_id"], "base_pid": twin,
                                 "raw_condition": raw_condition,
                                 "error": result.error or "no stage rendered"})
                return

            stage = result.stages[0]
            n_questions = len(stage.question_ids)
            max_tokens = _answer_generation_max_tokens(n_questions)
            body = llm_client.build_chat_request_body(
                model=model,
                messages=[
                    {"role": "system", "content": stage.system_prompt},
                    {"role": "user", "content": stage.user_prompt_head},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            chunk = writer.write({
                "custom_id": item["run_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })

            rendered = list(stage.rendered_block_ids)
            expected = stimulus_of.get(raw_condition, set())
            foreign = (set(rendered) & all_stimuli) - expected - (
                planarian_routes if raw_condition == PLANARIAN else set())
            render_rows.append({
                "run_id": item["run_id"],
                "base_pid": twin,
                "condition": item["condition"],
                "raw_condition": raw_condition,
                "control_variant": item.get("control_variant", ""),
                "chunk_file": chunk,
                "stage_index": 1,
                "n_questions": n_questions,
                "max_tokens": max_tokens,
                "prompt_sha256": stage.prompt_sha256,
                "system_chars": stage.prompt_chars["system"],
                "user_chars": stage.prompt_chars["persona"] + stage.prompt_chars["questions"],
                "prior_answer_count": stage.prior_answer_count,
                "block_order": "|".join(rendered),
                "stimulus_present": ";".join(sorted(expected & set(rendered))),
                "foreign_stimulus": ";".join(sorted(foreign)),
                "randomization_seed": result.randomization_seed,
                "newsletter_options": _newsletter_order(stage.user_prompt_head),
                "randomizer_choices": json.dumps(result.randomizer_choices, sort_keys=True),
                "needs_round2": "yes" if raw_condition == PLANARIAN else "no",
                "stages_rendered_in_dry": len(result.stages),
            })

            if raw_condition not in example_written:
                example_written.add(raw_condition)
                safe = raw_condition.replace("/", "_").replace(";", "").replace(" ", "_")
                (examples_dir / f"{safe}.txt").write_text(
                    f"run_id: {item['run_id']}\nbase_pid: {twin}\n"
                    f"raw_condition: {raw_condition}\ncondition: {item['condition']}\n"
                    f"control_variant: {item.get('control_variant','')}\n"
                    f"questions: {n_questions}   max_tokens: {max_tokens}\n"
                    f"block order: {' -> '.join(rendered)}\n"
                    f"prompt_sha256: {stage.prompt_sha256}\n"
                    f"{'=' * 78}\nSYSTEM MESSAGE\n{'=' * 78}\n{stage.system_prompt}\n\n"
                    f"{'=' * 78}\nUSER MESSAGE\n{'=' * 78}\n{stage.user_prompt_head}\n",
                    encoding="utf-8")

            if done % 500 == 0 or done == total:
                print(f"  built {done}/{total} requests "
                      f"({writer.index + 1} chunk file(s))", flush=True)

    print(f"batch-build: rendering {total} round-1 prompts (dry, no network)", flush=True)
    await asyncio.gather(*(one(item) for item in plan))
    writer.close()

    render_rows.sort(key=lambda r: (int(r["base_pid"].split("_")[1]), r["raw_condition"]))
    write_csv(render_rows, audit_dir / "render_plan.csv", list(render_rows[0].keys()))

    full_pool = not (limit or pilot_twins)
    reports = _write_reports(render_rows, problems, writer, stimulus_of,
                             planarian_routes, plan, model, persona_type, seed, audit_dir,
                             full_pool, template)
    from .common import BENCH_ROOT, MANIFEST, sha256_file
    source_manifest = load_json(MANIFEST)
    manifest = {
        "round": 1,
        "provenance": {
            "qsf_sha256": (source_manifest.get("benchmark_repo", {})
                           .get("authoritative_files", {}).get("survey/survey.qsf")),
            "template_poststudy_sha256": sha256_file(ARTIFACTS / "template_poststudy.json"),
            "condition_map_sha256": sha256_file(ARTIFACTS / "condition_map.json"),
            "qsf_audit_sha256": sha256_file(ARTIFACTS / "qsf_audit.json"),
            "frozen_prestudy_qa_sha256": sha256_file(DATA / "prestudy_frozen_qa.json"),
            "eligible_twins_sha256": sha256_file(DATA / "eligible_twins.csv"),
            "silicon_bench_repo": _git_state(BENCH_ROOT),
            "surveytwin_worktree": _git_state(BENCH_ROOT / "surveytwin"),
            "pipeline_source_sha256": _pipeline_source_sha256(),
        },
        "scope": {
            "eligible_twins": len({r["base_pid"] for r in render_rows}),
            "conditions": 17,
            "round1_requests": len(render_rows),
            "round2_requests_pending": sum(1 for r in render_rows
                                           if r["needs_round2"] == "yes"),
            "total_sessions": len(render_rows),
            "total_calls": len(render_rows) + sum(1 for r in render_rows
                                                  if r["needs_round2"] == "yes"),
        },
        "randomization": {
            "session_seed": "sha256(global_seed|base_pid|raw_condition) >> 1",
            "global_seed": seed,
            "note": "one runtime seed per twin x condition; forced condition "
                    "assignment bypasses it via pre-seeded randomizer_choices",
        },
        "model_alias": model,
        "persona_representation": persona_type,
        "seed": seed,
        "requests": len(render_rows),
        "sessions_planned": total,
        "chunks": writer.chunks,
        "total_bytes": sum(c["bytes"] for c in writer.chunks),
        "total_mb": round(sum(c["bytes"] for c in writer.chunks) / 1e6, 2),
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "temperature_sent": "not sent for gpt-5 family (reasoning_effort instead)",
        "response_format": {"type": "json_object"},
        "custom_id": "run_id (unique per session)",
        "round2_note": ("practical planarian stage 2 is NOT in this round: it branches on "
                        "QID1721185837, so its prompt does not exist until round 1 returns"),
        "problems": len(problems),
        "checks": reports,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nbatch-build: {len(render_rows)} requests in {len(writer.chunks)} chunk file(s), "
          f"{manifest['total_mb']} MB")
    print(f"  requests : {out_dir}")
    print(f"  audit    : {audit_dir}")
    failed = [k for k, v in reports.items() if v.get("ok") is False]
    skipped = [k for k, v in reports.items() if v.get("ok") is None]
    if problems:
        print(f"  PROBLEMS : {len(problems)} session(s) failed to render")
    if skipped:
        print(f"  not asserted on a subset: {skipped}")
    if failed:
        print(f"  FAILED CHECKS: {failed}")
        return 1
    print(f"  all {len([v for v in reports.values() if v.get('ok') is True])} "
          f"pre-submission checks passed")
    return 0


def _write_reports(rows, problems, writer, stimulus_of, planarian_routes,
                   plan, model, persona_type, seed, audit_dir, full_pool,
                   template) -> dict:
    """Randomization and branching audits, written for independent review."""
    import collections

    checks: dict[str, dict] = {}

    def check(key, ok, detail):
        checks[key] = {"ok": bool(ok), "detail": detail}

    # -- coverage -----------------------------------------------------------
    per_condition = collections.Counter(r["condition"] for r in rows)
    per_twin = collections.Counter(r["base_pid"] for r in rows)
    combos = collections.Counter((r["base_pid"], r["condition"]) for r in rows)
    n_twins = len(per_twin)
    check("every session rendered", len(rows) == len(plan) and not problems,
          f"{len(rows)} rendered of {len(plan)} planned, {len(problems)} problem(s)")
    check("17 conditions present", len(per_condition) == 17,
          f"{len(per_condition)} conditions: {sorted(per_condition)}")
    check("every twin has exactly 17 sessions",
          all(v == 17 for v in per_twin.values()),
          f"{sum(1 for v in per_twin.values() if v != 17)} twin(s) off 17")
    check("no duplicate base_pid x condition", all(v == 1 for v in combos.values()),
          f"{sum(1 for v in combos.values() if v > 1)} duplicate(s)")
    check("control has one session per twin", per_condition.get("control", 0) == n_twins,
          f"control={per_condition.get('control', 0)} twins={n_twins}")

    # -- stimulus / branching ----------------------------------------------
    missing = [r["run_id"] for r in rows if not r["stimulus_present"]]
    foreign = [r["run_id"] for r in rows if r["foreign_stimulus"]]
    check("every session carries its own stimulus", not missing,
          f"{len(missing)} missing, e.g. {missing[:3]}")
    check("no session carries a foreign stimulus", not foreign,
          f"{len(foreign)} with foreign stimulus, e.g. {foreign[:3]}")

    planarian = [r for r in rows if r["raw_condition"] == PLANARIAN]
    leaked = [r["run_id"] for r in planarian
              if set(r["block_order"].split("|")) & planarian_routes]
    check("planarian round 1 shows the state block and NO route block yet",
          not leaked and all(r["needs_round2"] == "yes" for r in planarian),
          f"{len(planarian)} planarian sessions, {len(leaked)} leaked a route block")
    check("only planarian needs a second round",
          {r["raw_condition"] for r in rows if r["needs_round2"] == "yes"} <= {PLANARIAN},
          "round 2 is planarian-only")

    # -- randomization ------------------------------------------------------
    orders = collections.Counter(r["block_order"] for r in rows)
    control_rows = [r for r in rows if r["condition"] == "control"]
    variants = collections.Counter(r["control_variant"] for r in control_rows)
    spread = (max(variants.values()) - min(variants.values())) if variants else 0
    if full_pool:
        check("control texts balanced across the pool",
              len(variants) == 3 and spread <= 1, f"{dict(variants)} (max-min={spread})")
    else:
        # A subset takes the first N twins out of a plan whose control texts
        # were round-robined over the SHUFFLED full pool, so balance is only
        # an invariant of the whole pool.
        checks["control texts balanced across the pool"] = {
            "ok": None, "detail": f"not asserted on a {n_twins}-twin subset: {dict(variants)}"}

    # Per-randomizer permutation coverage. Groups are derived from the
    # template, never hardcoded, so a QSF change surfaces here instead of
    # silently narrowing what is checked.
    groups = _randomizer_groups(template)
    coverage: dict[str, dict] = {}
    for flow_id, children in sorted(groups.items()):
        child_set = set(children)
        k = len(children)
        seqs = collections.Counter()
        for r in rows:
            seq = tuple(b for b in r["block_order"].split("|") if b in child_set)
            if len(seq) == k:                    # group fully rendered in this session
                seqs[seq] += 1
        total = math.factorial(k)
        n = sum(seqs.values())
        # Coupon-collector expectation: how many of the `total` orderings should
        # appear in `n` independent draws. Falling far short means the draws are
        # not independent.
        expected = total * (1 - (1 - 1 / total) ** n) if n and total else 0
        coverage[flow_id] = {
            "children": k, "possible_orders": total, "sessions": n,
            "observed_orders": len(seqs),
            "expected_orders": round(expected, 1),
        }
        if not n:
            continue
        # Require the full set only when this many draws should have produced
        # it; a small subset legitimately cannot show all 6 of a 3-block group.
        if total <= 24 and expected >= total - 0.5:
            check(f"{flow_id}: all {total} orderings appear ({k} blocks)",
                  len(seqs) == total,
                  f"{len(seqs)} of {total} over {n} sessions")
        elif total <= 24:
            checks[f"{flow_id}: all {total} orderings appear ({k} blocks)"] = {
                "ok": None,
                "detail": f"not asserted: {n} sessions expect only {expected:.1f} of "
                          f"{total} orderings; observed {len(seqs)}"}
        else:
            check(f"{flow_id}: broad ordering coverage ({k} blocks, {total} possible)",
                  len(seqs) >= 0.95 * expected,
                  f"{len(seqs)} of {total} observed, {expected:.0f} expected over {n} sessions")

    check("outcome block order varies across twins", len(orders) > 1 or len(rows) <= 1,
          f"{len(orders)} distinct block orders over {len(rows)} sessions")

    # Randomized CHOICE order, independent of block order.
    news = collections.Counter(r["newsletter_options"] for r in rows if r["newsletter_options"])
    check("both newsletter option orders appear", len(news) == 2,
          f"{dict(news)} over {sum(news.values())} sessions that rendered it")

    # The point of the session seed: a twin's orders must now be
    # condition-specific. Accidental coincidences are fine; what would signal a
    # shared seed is a twin whose conditions ALL collapse to one ordering.
    by_twin_orders = collections.defaultdict(list)
    for r in rows:
        if r["raw_condition"] == PLANARIAN:
            continue                              # truncated at the branch
        stripped = [b for b in r["block_order"].split("|")
                    if b not in stimulus_of.get(r["raw_condition"], set())
                    and b not in planarian_routes]
        by_twin_orders[r["base_pid"]].append("|".join(stripped))
    distinct_per_twin = {t: len(set(v)) for t, v in by_twin_orders.items()}
    collapsed = [t for t, v in distinct_per_twin.items()
                 if v == 1 and len(by_twin_orders[t]) > 1]
    mean_distinct = (sum(distinct_per_twin.values()) / len(distinct_per_twin)
                     if distinct_per_twin else 0)
    check("no twin reuses one block order across all its conditions",
          not collapsed, f"{len(collapsed)} twin(s) collapsed, e.g. {collapsed[:3]}")
    check("per-twin orders are condition-specific",
          mean_distinct > 0.9 * (len(by_twin_orders[next(iter(by_twin_orders))])
                                 if by_twin_orders else 1),
          f"mean {mean_distinct:.2f} distinct orders per twin across its 16 "
          f"single-stage conditions")

    # Session seeds must be distinct per twin x condition and reproducible.
    seeds = {(r["base_pid"], r["raw_condition"]): r["randomization_seed"] for r in rows}
    check("every twin x condition has its own randomization seed",
          len(set(seeds.values())) == len(seeds),
          f"{len(set(seeds.values()))} distinct seeds over {len(seeds)} sessions")

    # Planarian round 1 must still render the pre-treatment group and the state
    # block, and nothing from the outcome groups (those come in round 2).
    outcome_children = set()
    for flow_id, children in groups.items():
        if flow_id not in _PRE_TREATMENT_FLOWS:
            outcome_children |= set(children)
    early_outcome = [r["run_id"] for r in rows if r["raw_condition"] == PLANARIAN
                     and set(r["block_order"].split("|")) & outcome_children]
    check("planarian round 1 defers every outcome block to round 2",
          not early_outcome, f"{len(early_outcome)} leaked, e.g. {early_outcome[:3]}")

    report = {
        "generated_for": {"model_alias": model, "persona_representation": persona_type,
                          "seed": seed},
        "counts": {"sessions": len(rows), "twins": n_twins,
                   "per_condition": dict(sorted(per_condition.items()))},
        "randomization": {
            "control_variant_distribution": dict(variants),
            "distinct_block_orders": len(orders),
            "per_randomizer_coverage": coverage,
            "newsletter_option_orders": dict(news),
            "mean_distinct_orders_per_twin": round(mean_distinct, 3),
            "twins_collapsing_to_one_order": len(collapsed),
        },
        "branching": {
            "planarian_sessions": len(planarian),
            "planarian_route_blocks_deferred_to_round2": sorted(planarian_routes),
            "sessions_missing_stimulus": len(missing),
            "sessions_with_foreign_stimulus": len(foreign),
        },
        "checks": checks,
        "problems": problems[:50],
    }
    (Path(audit_dir) / "randomization_and_branching_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return checks
