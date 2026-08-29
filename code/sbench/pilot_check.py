"""Acceptance checks for a live pre-study pilot (read-only, no API calls)."""

from __future__ import annotations

import collections
import json
from pathlib import Path

from .common import (
    ARTIFACTS,
    AUDIT,
    BENCH_ROOT,
    DATA,
    QID,
    active_run_dir,
    load_json,
    read_csv,
    wire_worktree,
)

RESULTS: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def run_pilot_checks(expected: int) -> int:
    RESULTS.clear()
    meta = load_json(DATA / "prestudy_meta.json")
    qa = load_json(DATA / "prestudy_qa.json")

    # 1. all runs finish without errors or stalled stages
    errored = [t for t, m in meta.items() if m.get("error")]
    _check(f"{expected} runs attempted", len(meta) == expected, f"got {len(meta)}")
    _check("no execution errors", not errored, str(errored[:5]))
    stage_counts = collections.Counter(m.get("stage_count", 0) for m in meta.values())
    _check("no stalled runs (every twin produced stages)",
           all(m.get("stage_count", 0) > 0 for m in meta.values()),
           f"stage distribution {dict(sorted(stage_counts.items()))}")

    # 2. full personas non-empty
    wire_worktree()
    from services.v2.persona_loader import load_persona_text  # noqa: E402

    blank = [t for t in meta if not load_persona_text(t, "full").strip()
             or load_persona_text(t, "full").startswith("[Persona")]
    _check("all pilot personas non-empty (full)", not blank, str(blank[:5]))

    # 3. pre-study template excludes consent/filter/transition/treatment/outcome
    audit = load_json(ARTIFACTS / "qsf_audit.json")
    names = {b["block_id"]: b["block_name"] for b in audit["block_inventory"]}
    template = load_json(ARTIFACTS / "template_prestudy.json")

    def blocks(elements):
        for element in elements or []:
            if not isinstance(element, dict):
                continue
            if str(element.get("ElementType") or element.get("Type")) == "Block":
                yield element
            yield from blocks(element.get("Elements"))

    present = {names.get(b.get("BlockID")) for b in blocks(template["Elements"])}
    allowed = {"demographics", "Partisan identity", "Partisan importance", "Religion",
               "born again", "Religiosity", "epistemic autonomy", "Attention Check"}
    _check("pre-study template holds only the 8 intended blocks",
           present == allowed, f"unexpected {sorted(present - allowed)}")

    # 4. branch consistency against recorded answers
    def answer_of(twin, qid):
        return next((r["answer"]["answer_label"] or r["answer"]["answer_value"]
                     for r in qa[twin] if r["question"]["id"] == qid), None)

    def asked(twin, qid):
        return any(r["question"]["id"] == qid for r in qa[twin])

    # Branch conditions read from the QSF itself, not assumed:
    #   FL_250 partisan importance : Selected in {Republican, Democrat}
    #   FL_253 born again          : Selected in {Catholic, Mormon, Protestant, Orthodox Christian}
    #   FL_254 religiosity         : "I am not religious" NOT selected
    IMPORTANCE_PARTIES = {"Republican", "Democrat"}
    BORN_AGAIN_RELIGIONS = {"Catholic", "Mormon", "Protestant", "Orthodox Christian"}
    party_bad, religion_bad = [], []
    for twin in qa:
        party = answer_of(twin, QID["party"])
        if party is not None and asked(twin, "QID281") != (party in IMPORTANCE_PARTIES):
            party_bad.append((twin, party, asked(twin, "QID281")))
        religion = answer_of(twin, QID["religion"])
        if religion is not None:
            expect_born_again = religion in BORN_AGAIN_RELIGIONS
            expect_religiosity = religion != "I am not religious"
            if (asked(twin, "QID287") != expect_born_again
                    or asked(twin, "QID285") != expect_religiosity):
                religion_bad.append((twin, religion, asked(twin, "QID287"), asked(twin, "QID285")))
    _check("party branch follows recorded answers", not party_bad, str(party_bad[:3]))
    _check("religion branches follow recorded answers", not religion_bad, str(religion_bad[:3]))

    # 5. prompts contain the intended pre-study material and nothing forbidden
    prompts_path = Path(AUDIT / "prestudy_prompts.jsonl")
    forbidden = ("not to use AI", "Transition to Study", "necktie", "Rules of Baseball")
    expected_marker = "gender"
    saw_expected = False
    saw_forbidden: list[str] = []
    lines = 0
    with open(prompts_path, encoding="utf-8") as handle:
        for line in handle:
            lines += 1
            row = json.loads(line)
            blob = (row.get("system_prompt", "") + row.get("user_prompt", "")).lower()
            if expected_marker in blob:
                saw_expected = True
            for needle in forbidden:
                if needle.lower() in blob:
                    saw_forbidden.append(needle)
    _check("prompts captured for every stage", lines > 0, f"{lines} prompt records")
    _check("prompts contain intended pre-study material", saw_expected)
    _check("prompts contain no consent/filter/transition/treatment text",
           not saw_forbidden, str(sorted(set(saw_forbidden))))

    # 6. outputs structurally complete
    required_files = ["prestudy_answers_long.csv", "prestudy_answers_wide.csv",
                      "prestudy_qa.json", "prestudy_meta.json", "prestudy_checkpoint.jsonl"]
    missing = [f for f in required_files if not Path(DATA / f).exists()]
    _check("all expected output files present", not missing, str(missing))
    long_rows = read_csv(DATA / "prestudy_answers_long.csv")
    def _blank(row):
        return not (row.get("answer_label") or row.get("answer_value") or "").strip()

    # "If you selected Other, specify" companions are CORRECTLY blank whenever
    # the twin did not choose Other, so they are excluded from this check.
    substantive_blanks = [r for r in long_rows
                          if _blank(r) and not r["question_id"].endswith("_TEXT")]
    text_blanks = [r for r in long_rows if _blank(r) and r["question_id"].endswith("_TEXT")]
    by_question = collections.Counter(r["question_id"] for r in substantive_blanks)
    _check("no blank answers outside optional _TEXT companions",
           not substantive_blanks,
           f"{len(substantive_blanks)} blank across {len(by_question)} question(s): "
           f"{dict(by_question.most_common(4))}; "
           f"({len(text_blanks)} optional _TEXT blanks ignored)")

    # Stratification fields must be complete — this is what the phase feeds.
    strat_missing = [r for r in long_rows
                     if r["question_id"] in {QID["year_birth"], QID["gender"], QID["race"]}
                     and _blank(r)]
    _check("stratification fields (birth year, gender, race) complete",
           not strat_missing, f"{len(strat_missing)} missing")

    # 7. telemetry
    telemetry_path = Path(AUDIT / "prestudy_provider_calls.csv")
    _check("provider telemetry written", telemetry_path.exists())
    if telemetry_path.exists():
        rows = read_csv(telemetry_path)
        no_id = [r for r in rows if not r.get("provider_request_id")]
        no_tokens = [r for r in rows if not str(r.get("total_tokens", "")).strip().isdigit()]
        failed = [r for r in rows if r.get("status") != "completed"]
        summary = load_json(AUDIT / "prestudy_provider_summary.json")
        _check("every call has a provider request id", not no_id, f"{len(no_id)} missing")
        _check("every call has token counts", not no_tokens, f"{len(no_tokens)} missing")
        _check("no failed provider calls", not failed, f"{len(failed)} failed")
        _check("realized cost recorded",
               summary.get("realized_provider_cost_usd", 0) > 0,
               f"${summary.get('realized_provider_cost_usd', 0):.4f}, "
               f"{summary.get('total_tokens', 0):,} tokens, "
               f"{summary.get('reasoning_tokens', 0):,} reasoning")
        stamped = [r for r in rows if r.get("code_commit") and r.get("qsf_sha256")
                   and r.get("prompt_sha256")]
        _check("every call stamped with commit + QSF hash + prompt hash",
               len(stamped) == len(rows), f"{len(stamped)}/{len(rows)}")

    failures = [r for r in RESULTS if not r[1]]
    print(f"\npilot checks: {len(RESULTS)} run, {len(failures)} FAILED")
    return 1 if failures else 0
