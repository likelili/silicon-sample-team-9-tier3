"""Section 3 — automated audit of the authoritative QSF.

Inventories every structural feature the split depends on, classifies each
top-level flow element as pre-study / boundary / post-study / removed, traces
every reference a flow or question makes to another question or embedded-data
field, and fails hard when the assumptions behind the template split stop
holding.

Failure conditions (exit non-zero, per spec):
  * the "Transition to Study" boundary block is absent;
  * a post-study element references a pre-study answer in a way the separated
    runtime cannot preserve (references ARE preserved when the post-study
    runtime is seeded with the frozen pre-study answers; a reference to a
    REMOVED question can never be preserved and always fails);
  * an expected condition branch or randomizer child is missing;
  * the QSF hash differs from the recorded manifest value.
"""

from __future__ import annotations

import re
from collections import OrderedDict

from .common import (
    ARTIFACTS,
    AUTHORITATIVE_QSF,
    BOUNDARY_BLOCK_ID,
    REMOVED_ELEMENTS,
    TREATMENT_RANDOMIZER_FLOW_ID,
    condition_codes,
    load_json,
    manifest,
    save_json,
    sha256_file,
)

PIPE_RE = re.compile(r"\$\{([qe])://([^}/]+)[^}]*\}")


def _blocks(qsf: dict) -> "OrderedDict[str, dict]":
    out: "OrderedDict[str, dict]" = OrderedDict()
    for element in qsf["SurveyElements"]:
        if element.get("Element") != "BL":
            continue
        payload = element["Payload"]
        items = payload.values() if isinstance(payload, dict) else payload
        for block in items:
            if isinstance(block, dict):
                out[str(block.get("ID"))] = block
    return out


def _questions(qsf: dict) -> dict[str, dict]:
    return {
        e["Payload"]["QuestionID"]: e["Payload"]
        for e in qsf["SurveyElements"]
        if e.get("Element") == "SQ"
    }


def _flow(qsf: dict) -> list:
    payload = [e for e in qsf["SurveyElements"] if e.get("Element") == "FL"][0]["Payload"]
    return payload.get("Flow") or []


def _logic_refs(node) -> set[str]:
    """Every QID or embedded field referenced inside a BranchLogic/DisplayLogic tree."""
    refs: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            qid = value.get("QuestionID")
            if qid:
                refs.add(str(qid))
            for key in ("LeftOperand", "ChoiceLocator"):
                locator = str(value.get(key) or "")
                match = re.match(r"^([qe])://([^/]+)", locator)
                if match:
                    refs.add(match.group(2))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return refs


def _pipe_refs(text) -> set[str]:
    return {m.group(2) for m in PIPE_RE.finditer(str(text or ""))}


def _question_refs(payload: dict) -> set[str]:
    refs = set()
    refs |= _logic_refs(payload.get("DisplayLogic"))
    refs |= _pipe_refs(payload.get("QuestionText"))
    for key in ("Choices", "Answers"):
        choices = payload.get(key)
        if isinstance(choices, dict):
            for choice in choices.values():
                if isinstance(choice, dict):
                    refs |= _pipe_refs(choice.get("Display"))
    return refs


def _flow_node_summary(node, blocks) -> dict:
    kind = node.get("Type")
    out = {"flow_id": node.get("FlowID"), "type": kind}
    if kind in ("Block", "Standard"):
        out["block_id"] = node.get("ID")
        out["block_name"] = (blocks.get(node.get("ID")) or {}).get("Description")
    if kind == "BlockRandomizer":
        out["subset"] = node.get("SubSet")
        out["even_presentation"] = node.get("EvenPresentation")
    if kind == "EmbeddedData":
        out["assignments"] = [
            {"field": item.get("Field"), "value": item.get("Value")}
            for item in (node.get("EmbeddedData") or [])
        ]
    if kind == "Branch":
        out["references"] = sorted(_logic_refs(node.get("BranchLogic")))
    return out


def run_audit() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    qsf_hash = sha256_file(AUTHORITATIVE_QSF)
    recorded = manifest()["benchmark_repo"]["authoritative_files"]["survey/survey.qsf"]
    if qsf_hash != recorded:
        failures.append(
            f"survey.qsf hash {qsf_hash} differs from the recorded manifest value "
            f"{recorded} — a new audit of the changed file is required before anything runs."
        )

    qsf = load_json(AUTHORITATIVE_QSF)
    blocks = _blocks(qsf)
    questions = _questions(qsf)
    flow = _flow(qsf)

    # ---- map QID -> owning block, and per-block inventory -------------------
    qid_to_block: dict[str, str] = {}
    block_inventory = []
    for block_id, block in blocks.items():
        entries = []
        for element in block.get("BlockElements") or []:
            if element.get("Type") != "Question":
                continue
            qid = element.get("QuestionID")
            payload = questions.get(qid) or {}
            qid_to_block[qid] = block_id
            entries.append(
                {
                    "qid": qid,
                    "export_tag": payload.get("DataExportTag"),
                    "question_type": payload.get("QuestionType"),
                    "selector": payload.get("Selector"),
                    "forced_response": (payload.get("Validation") or {}).get("Settings", {}).get("ForceResponse"),
                    "has_display_logic": bool(payload.get("DisplayLogic")),
                    "references": sorted(_question_refs(payload)),
                    "is_timer": payload.get("QuestionType") == "Timing",
                    "is_descriptive": payload.get("QuestionType") == "DB",
                    "choice_randomization": bool(payload.get("Randomization")),
                }
            )
        block_inventory.append(
            {
                "block_id": block_id,
                "block_name": block.get("Description"),
                "block_type": block.get("Type"),
                "block_randomization": bool((block.get("Options") or {}).get("RandomizeQuestions")),
                "questions": entries,
            }
        )

    # ---- classify top-level flow into removed / pre / boundary / post -------
    removed_block_ids = set(REMOVED_ELEMENTS["blocks"])
    removed_flow_ids = set(REMOVED_ELEMENTS["flows"])
    boundary_index = None
    for index, node in enumerate(flow):
        if node.get("Type") in ("Block", "Standard") and node.get("ID") == BOUNDARY_BLOCK_ID:
            boundary_index = index
            break
    if boundary_index is None:
        failures.append(f"boundary block {BOUNDARY_BLOCK_ID} ('Transition to Study') is absent from the flow")

    sections: dict[str, list[dict]] = {"removed": [], "pre": [], "post": []}
    section_of_flow_index: list[str] = []
    for index, node in enumerate(flow):
        summary = _flow_node_summary(node, {k: v for k, v in blocks.items()})
        if (node.get("Type") in ("Block", "Standard") and node.get("ID") in removed_block_ids) or (
            str(node.get("FlowID")) in removed_flow_ids
        ):
            section = "removed"
        elif boundary_index is not None and index < boundary_index:
            section = "pre"
        else:
            section = "post"
        sections[section].append(summary)
        section_of_flow_index.append(section)

    # Which blocks belong to which section (via flow reachability, flat + nested).
    def blocks_under(node) -> set[str]:
        found = set()

        def walk(item):
            if isinstance(item, dict):
                if item.get("Type") in ("Block", "Standard") and item.get("ID"):
                    found.add(item["ID"])
                for child in item.get("Flow") or []:
                    walk(child)
            elif isinstance(item, list):
                for child in item:
                    walk(child)

        walk(node)
        return found

    block_section: dict[str, str] = {}
    for index, node in enumerate(flow):
        for block_id in blocks_under(node):
            block_section[block_id] = section_of_flow_index[index]

    qid_section = {qid: block_section.get(block_id, "unknown") for qid, block_id in qid_to_block.items()}

    # ---- cross-boundary dependency analysis ---------------------------------
    def deep_refs(node) -> set[str]:
        refs = set()

        def walk(item):
            if isinstance(item, dict):
                refs.update(_logic_refs(item.get("BranchLogic")))
                for child in item.get("Flow") or []:
                    walk(child)
            elif isinstance(item, list):
                for child in item:
                    walk(child)

        walk(node)
        return refs

    cross_boundary: list[dict] = []
    for index, node in enumerate(flow):
        if section_of_flow_index[index] != "post":
            continue
        for ref in sorted(deep_refs(node)):
            ref_base = ref.split("_")[0] if ref.startswith("QID") else ref
            owner = qid_section.get(ref) or qid_section.get(ref_base)
            if owner == "pre":
                cross_boundary.append(
                    {
                        "flow_id": node.get("FlowID"),
                        "references": ref,
                        "kind": "flow-branch",
                        "preserved_by": "frozen pre-study answers seeded into the post-study runtime",
                    }
                )
            elif owner == "removed":
                failures.append(
                    f"post-study flow {node.get('FlowID')} references {ref}, which belongs to a REMOVED "
                    "element — this dependency cannot be preserved"
                )
    # Question-level references (display logic + piping) crossing the boundary.
    for qid, block_id in qid_to_block.items():
        if block_section.get(block_id) != "post":
            continue
        for ref in sorted(_question_refs(questions[qid])):
            owner = qid_section.get(ref)
            if owner == "pre":
                cross_boundary.append(
                    {
                        "question": qid,
                        "references": ref,
                        "kind": "question display-logic or piping",
                        "preserved_by": "frozen pre-study answers seeded into the post-study runtime",
                    }
                )
            elif owner == "removed":
                failures.append(
                    f"post-study question {qid} references {ref} from a REMOVED element — cannot be preserved"
                )
    # And nothing anywhere may reference the removed consent/filter questions.
    removed_qids = {
        qid for qid, block_id in qid_to_block.items() if block_id in removed_block_ids
    }
    for qid, payload in questions.items():
        if qid in removed_qids or qid_to_block.get(qid) in removed_block_ids:
            continue
        hit = _question_refs(payload) & removed_qids
        if hit:
            failures.append(f"question {qid} references removed question(s) {sorted(hit)}")

    # ---- expected condition branches and randomizer children ----------------
    codes = condition_codes()
    code_field = next((k for k in (codes[0].keys()) if "code" in k.lower()), None) if codes else None
    expected_codes = sorted({row[code_field] for row in codes}) if code_field else []
    randomizer = next(
        (n for n in flow if str(n.get("FlowID")) == TREATMENT_RANDOMIZER_FLOW_ID), None
    )
    randomizer_conditions: list[str] = []
    control_groups = 0
    if randomizer is None:
        failures.append(f"treatment randomizer {TREATMENT_RANDOMIZER_FLOW_ID} not found")
    else:
        for child in randomizer.get("Flow") or []:
            if child.get("Type") == "EmbeddedData":
                for item in child.get("EmbeddedData") or []:
                    if item.get("Field") == "condition":
                        randomizer_conditions.append(str(item.get("Value")))
            elif child.get("Type") == "Group":
                control_groups += 1
        if control_groups != 2:
            failures.append(f"expected 2 control groups under {TREATMENT_RANDOMIZER_FLOW_ID}, found {control_groups}")

    branch_conditions: list[str] = []
    for node in flow:
        if node.get("Type") != "Branch":
            continue
        logic = str(node.get("BranchLogic") or "")
        match = re.search(r"e://Field/condition[^']*'?.*?RightOperand': '([^']+)'", logic)
        # Fallback: walk the structure for the RightOperand where LeftOperand is the condition field.
        def right_operands(value):
            found = []
            if isinstance(value, dict):
                if "condition" in str(value.get("LeftOperand") or ""):
                    found.append(str(value.get("RightOperand")))
                for child in value.values():
                    found += right_operands(child)
            elif isinstance(value, list):
                for child in value:
                    found += right_operands(child)
            return found

        branch_conditions += [c for c in right_operands(node.get("BranchLogic")) if c and c != "None"]

    interventions = [c for c in expected_codes if not c.startswith("control")]
    for code in interventions:
        if code not in randomizer_conditions:
            failures.append(f"intervention '{code}' missing from {TREATMENT_RANDOMIZER_FLOW_ID} children")
        if code not in branch_conditions:
            failures.append(f"intervention '{code}' has no condition branch in the flow")
    for control_text in ("control neckties", "control baseball", "control dances"):
        if control_text not in branch_conditions:
            failures.append(f"'{control_text}' has no condition branch in the flow")

    # ---- embedded-data assignments across the survey ------------------------
    embedded_assignments = []

    def collect_embedded(node, where):
        if isinstance(node, dict):
            if node.get("Type") == "EmbeddedData":
                for item in node.get("EmbeddedData") or []:
                    embedded_assignments.append(
                        {"flow_id": node.get("FlowID"), "section": where,
                         "field": item.get("Field"), "value": item.get("Value")}
                    )
            for child in node.get("Flow") or []:
                collect_embedded(child, where)
        elif isinstance(node, list):
            for child in node:
                collect_embedded(child, where)

    for index, node in enumerate(flow):
        collect_embedded(node, section_of_flow_index[index])

    timers = [
        {"qid": qid, "block": qid_to_block.get(qid)}
        for qid, payload in questions.items()
        if payload.get("QuestionType") == "Timing"
    ]

    report = {
        "qsf": str(AUTHORITATIVE_QSF),
        "qsf_sha256": qsf_hash,
        "hash_matches_manifest": qsf_hash == recorded,
        "boundary": {"block_id": BOUNDARY_BLOCK_ID, "flow_index": boundary_index},
        "counts": {
            "blocks": len(blocks),
            "questions": len(questions),
            "timing_questions": len(timers),
            "top_level_flow_elements": len(flow),
            "pre_study_elements": len(sections["pre"]),
            "post_study_elements": len(sections["post"]),
            "removed_elements": len(sections["removed"]),
        },
        "sections": sections,
        "block_inventory": block_inventory,
        "embedded_data_assignments": embedded_assignments,
        "cross_boundary_dependencies": cross_boundary,
        "treatment_randomizer": {
            "flow_id": TREATMENT_RANDOMIZER_FLOW_ID,
            "condition_children": randomizer_conditions,
            "control_groups": control_groups,
        },
        "condition_branches_found": sorted(set(branch_conditions)),
        "expected_condition_codes": expected_codes,
        "timing_questions": timers,
        "failures": failures,
        "warnings": warnings,
    }
    save_json(report, ARTIFACTS / "qsf_audit.json")

    lines = [
        "# QSF audit — Silicon Sample Benchmark",
        "",
        f"- QSF: `{AUTHORITATIVE_QSF.name}` sha256 `{qsf_hash[:16]}…` — "
        + ("matches the recorded manifest value" if qsf_hash == recorded else "**HASH MISMATCH**"),
        f"- Boundary: `{BOUNDARY_BLOCK_ID}` (Transition to Study) at top-level flow index {boundary_index}",
        f"- {len(blocks)} blocks, {len(questions)} questions ({len(timers)} Timing), "
        f"{len(flow)} top-level flow elements: {len(sections['pre'])} pre-study, "
        f"{len(sections['post'])} post-study, {len(sections['removed'])} removed",
        f"- Treatment randomizer `{TREATMENT_RANDOMIZER_FLOW_ID}`: {len(randomizer_conditions)} condition children "
        f"+ {control_groups} control groups (each drawing one of three control texts)",
        f"- Condition branches found: {len(set(branch_conditions))}",
        f"- Cross-boundary dependencies: {len(cross_boundary)}"
        + (" — every one preserved by seeding frozen pre-study answers" if cross_boundary else ""),
        "",
        "## Failures" if failures else "## No failures",
    ]
    lines += [f"- {f}" for f in failures]
    if warnings:
        lines += ["", "## Warnings"] + [f"- {w}" for w in warnings]
    (ARTIFACTS / "qsf_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"qsf_audit: {'FAIL' if failures else 'PASS'}"
          f" — {len(failures)} failure(s), {len(cross_boundary)} cross-boundary dependenc(ies)")
    for failure in failures:
        print("  FAIL:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_audit())
