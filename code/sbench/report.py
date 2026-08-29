"""Pre-study checkpoint report — everything a team review needs, no API calls.

Reads a completed run directory and emits ``artifacts/prestudy_checkpoint_report.md``
plus a machine-readable ``data/prestudy_checkpoint_report.json``.  It only reads;
it never repairs, freezes, samples or simulates.
"""

from __future__ import annotations

import collections
import csv
import hashlib
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
    save_json,
    wire_worktree,
)
from .conflicts import FIELDS


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pct(part: int, whole: int) -> str:
    return f"{(part / whole * 100):.1f}%" if whole else "n/a"


def build_report() -> int:
    wire_worktree()
    run = active_run_dir()
    meta = load_json(DATA / "prestudy_meta.json")
    qa = load_json(DATA / "prestudy_qa.json")
    manifest = load_json(BENCH_ROOT / "manifest" / "source_manifest.json")
    template = load_json(ARTIFACTS / "template_prestudy.json")

    # ---- counts -----------------------------------------------------------
    attempted = len(meta)
    errored = [t for t, m in meta.items() if m.get("error")]
    attention = [t for t, m in meta.items() if m.get("excluded")]

    # Required = answerable questions in UNCONDITIONAL pre-study blocks, taken
    # from the instrument builder rather than the raw template: it expands a
    # Matrix into its row-level sub-questions, which is what actually gets
    # answered (the parent Matrix id never appears in a response).  Optional
    # "_TEXT" companions and conditional blocks are excluded.
    from services.v2.instrument_builder import collect_instrument_questions  # noqa: E402

    required = set()
    for element in template["Elements"]:
        if str(element.get("ElementType") or element.get("Type") or "") != "Block":
            continue
        for question in collect_instrument_questions([element]):
            qid = str(question.get("id") or "")
            if qid and not qid.endswith("_TEXT"):
                required.add(qid)
    incomplete = {}
    for twin, records in qa.items():
        answered = {r["question"]["id"] for r in records}
        missing = sorted(
            qid for qid in required
            if qid not in answered and not any(a.startswith(f"{qid}_") for a in answered)
        )
        if missing:
            incomplete[twin] = missing
    completed = [t for t in meta if t not in set(errored) | set(incomplete)]

    # ---- stage counts and warnings ----------------------------------------
    stage_dist = collections.Counter(m.get("stage_count", 0) for m in meta.values())
    warnings = collections.Counter()
    staging = collections.Counter()
    stages_path = Path(AUDIT / "prestudy_stages.jsonl")
    retries_total = 0
    if stages_path.exists():
        with open(stages_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                retries_total += int(row.get("retries") or 0)
                for event in row.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    reason = str(event.get("reason") or event.get("message") or event.get("type"))
                    # A "blocked" event on a branch/display dependency is not a
                    # fault: it is the mechanism that ENDS a stage so the next
                    # one can be rendered with the answer in hand. Expect
                    # roughly one per stage. Counted separately from warnings
                    # that indicate something actually went wrong.
                    if event.get("type") == "blocked" and reason in {
                        "branch_question_dependency",
                        "question_display_logic_dependency",
                        "skip_logic_question_dependency",
                        "block_q_piping",
                        "embedded_data_q_piping",
                    }:
                        staging[reason] += 1
                    elif event.get("type") in {"blocked", "warning"}:
                        warnings[reason] += 1

    # ---- missing / malformed answers --------------------------------------
    blank_substantive = 0
    blank_optional_text = 0
    blank_by_question = collections.Counter()
    status_counts = collections.Counter()
    for row in read_csv(DATA / "prestudy_answers_long.csv"):
        if not (row.get("answer_label") or row.get("answer_value") or "").strip():
            # "If you selected Other, specify" companions are CORRECTLY blank
            # whenever the twin did not choose Other.
            if row["question_id"].endswith("_TEXT"):
                blank_optional_text += 1
            else:
                blank_substantive += 1
                blank_by_question[row["question_id"]] += 1
        if row.get("answer_status"):
            status_counts[row["answer_status"]] += 1

    # ---- branch consistency ----------------------------------------------
    def answer_of(twin, qid):
        return next((r["answer"]["answer_label"] or r["answer"]["answer_value"]
                     for r in qa[twin] if r["question"]["id"] == qid), None)

    def asked(twin, qid):
        return any(r["question"]["id"] == qid for r in qa[twin])

    # Conditions read from the QSF itself, not assumed:
    #   FL_250 partisan importance : Selected in {Republican, Democrat}
    #   FL_253 born again          : Selected in {Catholic, Mormon, Protestant, Orthodox Christian}
    #   FL_254 religiosity         : "I am not religious" NOT selected
    IMPORTANCE_PARTIES = {"Republican", "Democrat"}
    BORN_AGAIN_RELIGIONS = {"Catholic", "Mormon", "Protestant", "Orthodox Christian"}
    branch = {"party_ok": 0, "party_bad": [], "religion_ok": 0, "religion_bad": []}
    for twin in qa:
        party = answer_of(twin, QID["party"])
        if party is not None:
            if asked(twin, "QID281") == (party in IMPORTANCE_PARTIES):
                branch["party_ok"] += 1
            else:
                branch["party_bad"].append((twin, party, asked(twin, "QID281")))
        religion = answer_of(twin, QID["religion"])
        if religion is not None:
            expect_born_again = religion in BORN_AGAIN_RELIGIONS
            expect_religiosity = religion != "I am not religious"
            if (asked(twin, "QID287") == expect_born_again
                    and asked(twin, "QID285") == expect_religiosity):
                branch["religion_ok"] += 1
            else:
                branch["religion_bad"].append(
                    (twin, religion, asked(twin, "QID287"), asked(twin, "QID285")))

    # ---- conflicts and proposed repairs ------------------------------------
    conflicts_by_field = collections.Counter()
    comparisons = 0
    comparison_path = Path(DATA / "profile_comparisons.csv")
    if comparison_path.exists():
        for row in read_csv(comparison_path):
            comparisons += 1
            if row["result"] == "conflict":
                conflicts_by_field[row["field"]] += 1
    proposed = read_csv(DATA / "proposed_repairs.csv") if Path(DATA / "proposed_repairs.csv").exists() else []

    # ---- demographic distributions available for stratification -----------
    strata = {}
    wide_path = Path(DATA / "prestudy_answers_wide.csv")
    if wide_path.exists():
        wide = read_csv(wide_path)
        for column in ("age_band", "gender", "race", "education", "income", "party", "religion"):
            counts = collections.Counter((r.get(column) or "(missing)").strip() for r in wide)
            strata[column] = dict(counts.most_common())

    # ---- provider totals ---------------------------------------------------
    provider = {}
    summary_path = Path(AUDIT / "prestudy_provider_summary.json")
    if summary_path.exists():
        provider = load_json(summary_path)

    # ---- hashes of every saved output -------------------------------------
    hashes = {}
    for folder in ("data", "audit"):
        base = run / folder
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    hashes[str(path.relative_to(run))] = {
                        "sha256": _sha(path), "bytes": path.stat().st_size,
                    }

    report = {
        "run_dir": str(run),
        "provenance": {
            "surveytwin_commit": manifest["surveytwin"]["commit"],
            "benchmark_commit": manifest["benchmark_repo"]["commit"],
            "qsf_sha256": manifest["benchmark_repo"]["authoritative_files"]["survey/survey.qsf"],
            "model": manifest["run_config"]["model"],
            "persona_representation": manifest["run_config"]["persona_representation"],
            "seed": manifest["run_config"]["seed"],
        },
        "counts": {
            "attempted": attempted, "completed": len(completed), "errored": len(errored),
            "attention_excluded": len(attention), "incomplete": len(incomplete),
        },
        "stage_count_distribution": dict(sorted(stage_dist.items())),
        "runtime_warnings": dict(warnings),
        "staging_boundaries_normal": dict(staging),
        "technical_retries": retries_total,
        "answers": {
            "blank_substantive": blank_substantive,
            "blank_by_question": dict(blank_by_question),
            "blank_optional_text_companions": blank_optional_text,
            "answer_status_counts": dict(status_counts),
        },
        "branch_consistency": {
            "party_consistent": branch["party_ok"],
            "party_inconsistent": len(branch["party_bad"]),
            "party_examples": branch["party_bad"][:5],
            "religion_consistent": branch["religion_ok"],
            "religion_inconsistent": len(branch["religion_bad"]),
            "religion_examples": branch["religion_bad"][:5],
        },
        "profile_conflicts": {
            "field_comparisons": comparisons,
            "by_field": dict(conflicts_by_field),
            "total": sum(conflicts_by_field.values()),
            "twins_with_conflicts": len({r["source_twin_id"] for r in proposed}),
        },
        "proposed_repairs_not_executed": proposed,
        "stratification_distributions": strata,
        "provider_totals": provider,
        "output_hashes": hashes,
    }
    save_json(report, DATA / "prestudy_checkpoint_report.json")

    counts = report["counts"]
    lines = [
        "# Pre-study checkpoint report",
        "",
        f"Run directory: `{run}`",
        "",
        "## Provenance",
        "",
        f"- SurveyTwin commit `{report['provenance']['surveytwin_commit'][:8]}` "
        f"(branch `bench/silicon-integration`)",
        f"- Benchmark repo `{report['provenance']['benchmark_commit'][:8]}`, "
        f"QSF `{report['provenance']['qsf_sha256'][:16]}…`",
        f"- Model `{report['provenance']['model']}`, persona "
        f"`{report['provenance']['persona_representation']}`, seed {report['provenance']['seed']}",
        "",
        "## Counts",
        "",
        "| | n | share |",
        "|---|---|---|",
        f"| Attempted | {counts['attempted']} | |",
        f"| Completed | {counts['completed']} | {_pct(counts['completed'], counts['attempted'])} |",
        f"| Errored | {counts['errored']} | {_pct(counts['errored'], counts['attempted'])} |",
        f"| Attention-excluded | {counts['attention_excluded']} | "
        f"{_pct(counts['attention_excluded'], counts['attempted'])} |",
        f"| Incomplete (missing required questions) | {counts['incomplete']} | "
        f"{_pct(counts['incomplete'], counts['attempted'])} |",
        "",
        "## Stages and warnings",
        "",
        "| Stages | Twins |",
        "|---|---|",
    ]
    for stages, n in report["stage_count_distribution"].items():
        lines.append(f"| {stages} | {n} |")
    lines += [
        "",
        f"Technical retries across all calls: **{retries_total}**",
        "",
        ("**Runtime warnings: " + ", ".join(f"`{k}` × {v}" for k, v in warnings.items()) + "**")
        if warnings else "No runtime warnings.",
        "",
        ("Staging boundaries (normal control flow, ~1 per stage — the runtime pausing "
         "to get an answer a branch depends on): "
         + ", ".join(f"`{k}` × {v}" for k, v in staging.items()))
        if staging else "",
        "",
        "## Answer completeness",
        "",
        f"- Blank answers outside optional `_TEXT` companions: **{blank_substantive}**"
        + (f" across {len(blank_by_question)} question(s)" if blank_substantive else ""),
        f"- Optional `_TEXT` \"if Other, specify\" companions left blank (expected): "
        f"{blank_optional_text}",
        ("- Answer statuses: " + ", ".join(f"`{k}` × {v}" for k, v in status_counts.items()))
        if status_counts else "- No flagged answer statuses.",
        "",
        "## Branch consistency",
        "",
        f"- Party → partisan importance: **{branch['party_ok']} consistent**, "
        f"{len(branch['party_bad'])} inconsistent",
        f"- Religion → born-again / religiosity: **{branch['religion_ok']} consistent**, "
        f"{len(branch['religion_bad'])} inconsistent",
        "",
        "## Logical profile conflicts",
        "",
        f"{comparisons} field comparisons, **{sum(conflicts_by_field.values())} conflicts** "
        f"on {report['profile_conflicts']['twins_with_conflicts']} twin(s).",
        "",
        "| Field | Conflicts |",
        "|---|---|",
    ]
    for field in FIELDS:
        lines.append(f"| {field} | {conflicts_by_field.get(field, 0)} |")
    lines += [
        "",
        f"**{len(proposed)} proposed repair(s) — NOT executed.** See `data/proposed_repairs.csv`.",
        "",
        "## Demographic distributions available for stratification",
        "",
    ]
    for column, counts_map in strata.items():
        top = ", ".join(f"{k} {v}" for k, v in list(counts_map.items())[:6])
        lines.append(f"- **{column}**: {top}")
    if provider:
        lines += [
            "",
            "## Provider totals",
            "",
            f"- Calls: **{provider.get('calls', 0)}** "
            f"({provider.get('failed_calls', 0)} failed)",
            f"- Input tokens: {provider.get('uncached_input_tokens', 0):,} uncached + "
            f"{provider.get('cached_input_tokens', 0):,} cached",
            f"- Output tokens: {provider.get('output_tokens', 0):,} "
            f"(of which {provider.get('reasoning_tokens', 0):,} reasoning)",
            f"- **Realized provider cost: ${provider.get('realized_provider_cost_usd', 0):.4f}**",
            f"- Calls missing a request id: {provider.get('calls_missing_request_id', 0)}",
        ]
    lines += [
        "",
        "## Output hashes",
        "",
        f"{len(hashes)} file(s) hashed; full list in `data/prestudy_checkpoint_report.json`.",
        "",
        "| File | SHA-256 (first 16) | Bytes |",
        "|---|---|---|",
    ]
    for name, info in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{info['sha256'][:16]}` | {info['bytes']:,} |")
    lines += [
        "",
        "---",
        "",
        "**Stopped for team review.** No repair, freeze, sampling or post-study phase has run.",
    ]
    (ARTIFACTS / "prestudy_checkpoint_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checkpoint report: {counts['completed']}/{counts['attempted']} completed, "
          f"{counts['errored']} errored, {counts['incomplete']} incomplete, "
          f"{sum(conflicts_by_field.values())} conflict(s), {len(hashes)} file(s) hashed")
    return 0
