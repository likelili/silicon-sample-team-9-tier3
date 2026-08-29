"""Sections 4-6 — build the pre-study and post-study templates and the
fixed-assignment condition map.

The QSF is parsed once through SurveyTwin's own parser (the same
``parse_qsf_file`` + ``generate_template_format`` pair ``PipelineService``
wraps), producing the internal template whose flat, flow-ordered ``Elements``
list the staged runtime consumes.  The split is pure element-list surgery on
that template:

  * remove the Consent Form and Filter blocks and the two screenout branches
    (FL_6, FL_7) — they gate human entry and do not apply to twins;
  * pre-study  = everything before the "Transition to Study" block;
  * post-study = the "Transition to Study" block onward.

No question payload is touched: wording, IDs, export tags, choice IDs, recode
values and validation settings pass through byte-identical.  Timing questions
stay in the template — the instrument builder already excludes them from the
questions sent to the model, which the build verifies rather than assumes.

Fixed condition assignment (section 6) does not edit the template at all.  The
runtime honours pre-seeded ``state.randomizer_choices``, so the build extracts
a condition map from the post-study template's FL_18 children:

    condition code -> which FL_18 child index to force
    control texts  -> FL_18 child index of the control group
                      + the inner randomizer's flow id and child index

The driver seeds those choices per twin-condition; the randomizer event in the
audit trail records exactly what was forced.
"""

from __future__ import annotations

import copy

from .common import (
    ARTIFACTS,
    AUTHORITATIVE_QSF,
    BOUNDARY_BLOCK_ID,
    REMOVED_ELEMENTS,
    TREATMENT_RANDOMIZER_FLOW_ID,
    condition_codes,
    load_json,
    save_json,
    sha256_file,
    wire_worktree,
)


def _element_identity(element: dict) -> tuple[str, str, str]:
    return (
        str(element.get("ElementType") or element.get("Type") or ""),
        str(element.get("BlockID") or ""),
        str(element.get("FlowID") or ""),
    )


def build_full_template() -> dict:
    wire_worktree()
    from parse_qsf import generate_template_format, parse_qsf_file  # noqa: E402

    parsed = parse_qsf_file(str(AUTHORITATIVE_QSF))
    return generate_template_format(parsed)


def _describe(element: dict) -> dict:
    kind, block_id, flow_id = _element_identity(element)
    out = {"type": kind}
    if flow_id:
        out["flow_id"] = flow_id
    if block_id:
        out["block_id"] = block_id
        out["block_name"] = element.get("BlockName")
    return out


def extract_condition_map(post_template: dict) -> dict:
    """condition code -> forced randomizer choices, read from the template itself."""
    randomizer = None
    for element in post_template["Elements"]:
        if str(element.get("FlowID")) == TREATMENT_RANDOMIZER_FLOW_ID:
            randomizer = element
            break
    if randomizer is None:
        raise SystemExit(f"condition map: {TREATMENT_RANDOMIZER_FLOW_ID} not in post-study template")

    children = randomizer.get("Elements") or []
    mapping: dict[str, dict] = {}
    control_groups: list[dict] = []
    for index, child in enumerate(children):
        kind = str(child.get("ElementType") or child.get("Type") or "")
        if kind == "EmbeddedData":
            for item in child.get("EmbeddedData") or []:
                if item.get("Field") == "condition":
                    code = str(item.get("Value"))
                    mapping[code] = {
                        "kind": "intervention",
                        "forced_choices": {TREATMENT_RANDOMIZER_FLOW_ID: [index]},
                    }
        elif kind == "Group":
            # A control group wraps one inner randomizer over the three texts.
            inner = None
            for grand in child.get("Elements") or []:
                grand_kind = str(grand.get("ElementType") or grand.get("Type") or "")
                if grand_kind in ("Randomizer", "BlockRandomizer"):
                    inner = grand
                    break
            if inner is None:
                raise SystemExit(f"condition map: control group at index {index} has no inner randomizer")
            variants = {}
            for inner_index, node in enumerate(inner.get("Elements") or []):
                for item in node.get("EmbeddedData") or []:
                    if item.get("Field") == "condition":
                        variants[str(item.get("Value"))] = inner_index
            control_groups.append(
                {
                    "outer_index": index,
                    "inner_flow_id": str(inner.get("FlowID")),
                    "variants": variants,
                }
            )

    if len(control_groups) != 2:
        raise SystemExit(f"condition map: expected 2 control groups, found {len(control_groups)}")

    # Control assignment always routes through the FIRST control group; both
    # groups are identical duplicates that exist only to double the random
    # draw, which fixed assignment replaces.
    group = control_groups[0]
    for text, inner_index in group["variants"].items():
        mapping[text] = {
            "kind": "control",
            "forced_choices": {
                TREATMENT_RANDOMIZER_FLOW_ID: [group["outer_index"]],
                group["inner_flow_id"]: [inner_index],
            },
        }

    # Verify against the authoritative codename list — complete codes, never split.
    rows = condition_codes()
    code_field = next(k for k in rows[0] if "code" in k.lower())
    expected = {str(row[code_field]) for row in rows}
    missing = expected - set(mapping)
    extra = set(mapping) - expected
    if missing:
        raise SystemExit(f"condition map: authoritative codes missing from template: {sorted(missing)}")
    if extra:
        raise SystemExit(f"condition map: template codes not in condition_codenames.csv: {sorted(extra)}")

    return {
        "source": "extracted from the post-study template's FL_18 children; "
                  "verified complete against survey/condition_codenames.csv",
        "control_groups": control_groups,
        "conditions": mapping,
    }


def build() -> int:
    qsf_hash = sha256_file(AUTHORITATIVE_QSF)
    audit = load_json(ARTIFACTS / "qsf_audit.json")
    if audit.get("failures"):
        raise SystemExit("templates: refusing to build — qsf_audit.json records failures")
    if audit.get("qsf_sha256") != qsf_hash:
        raise SystemExit("templates: QSF changed since the audit ran — re-run the audit first")

    template = build_full_template()
    elements = template["Elements"]

    removed_blocks = set(REMOVED_ELEMENTS["blocks"])
    removed_flows = set(REMOVED_ELEMENTS["flows"])

    kept: list[dict] = []
    removed: list[dict] = []
    for element in elements:
        _, block_id, flow_id = _element_identity(element)
        if block_id in removed_blocks or flow_id in removed_flows:
            removed.append(_describe(element))
        else:
            kept.append(element)

    if len(removed) != len(removed_blocks) + len(removed_flows):
        raise SystemExit(
            f"templates: expected to remove {len(removed_blocks) + len(removed_flows)} elements, "
            f"matched {len(removed)} — identity drift, refusing to continue"
        )

    boundary_index = None
    for index, element in enumerate(kept):
        if str(element.get("BlockID")) == BOUNDARY_BLOCK_ID:
            boundary_index = index
            break
    if boundary_index is None:
        raise SystemExit("templates: boundary block missing after removals")

    pre_elements = copy.deepcopy(kept[:boundary_index])
    post_elements = copy.deepcopy(kept[boundary_index:])

    pre_template = {"Metadata": copy.deepcopy(template["Metadata"]), "Elements": pre_elements}
    post_template = {"Metadata": copy.deepcopy(template["Metadata"]), "Elements": post_elements}

    # Verify no Timing question is ever presented to the model: collect the
    # instrument questions for every block element and assert no Timing ids.
    wire_worktree()
    from services.v2.instrument_builder import collect_instrument_questions  # noqa: E402

    def timing_check(tpl: dict) -> list[str]:
        offenders = []
        for element in tpl["Elements"]:
            kind = str(element.get("ElementType") or element.get("Type") or "")
            if kind != "Block":
                continue
            for question in collect_instrument_questions([element]):
                if str(question.get("type") or "").lower() == "timing":
                    offenders.append(str(question.get("id")))
        return offenders

    timing_offenders = timing_check(pre_template) + timing_check(post_template)

    condition_map = extract_condition_map(post_template)

    save_json(template, ARTIFACTS / "template_full.json")
    save_json(pre_template, ARTIFACTS / "template_prestudy.json")
    save_json(post_template, ARTIFACTS / "template_poststudy.json")
    save_json(condition_map, ARTIFACTS / "condition_map.json")
    record = {
        "source_qsf": str(AUTHORITATIVE_QSF),
        "source_qsf_sha256": qsf_hash,
        "removed_elements": removed,
        "removed_reasons": REMOVED_ELEMENTS,
        "boundary_block": BOUNDARY_BLOCK_ID,
        "prestudy_elements": [_describe(e) for e in pre_elements],
        "poststudy_elements": [_describe(e) for e in post_elements],
        "timing_questions_reaching_model": timing_offenders,
        "notes": [
            "question payloads pass through unmodified (wording, IDs, export tags, "
            "raw choice IDs, recode values, validation)",
            "attention checks and their exclusion branches are retained in the pre-study template",
            "fixed condition assignment seeds state.randomizer_choices; the template is not edited",
        ],
    }
    save_json(record, ARTIFACTS / "template_manifest.json")

    ok = not timing_offenders
    print(f"templates: {'PASS' if ok else 'FAIL'} — removed {len(removed)}, "
          f"pre {len(pre_elements)} elements, post {len(post_elements)} elements, "
          f"{len(condition_map['conditions'])} condition codes mapped, "
          f"timing-to-model offenders: {len(timing_offenders)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(build())
