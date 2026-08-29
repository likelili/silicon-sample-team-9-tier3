"""Dry-run: build the 13 scored outcomes for a handful of real sessions.

Shows one composite (``trust_multidimensional``), one binary
(``newsletter_signup``) and one single-item (``belief_post``) outcome end to
end — raw item values, the official formula applied, the constructed value, and
the declared-range check — so the scoring can be inspected before any
production fit runs.
"""

from __future__ import annotations

import json
import math
import re
import sys

from .common import STUDY_RUN, read_csv_dicts
from .outcomes import CODENAMES, OUTCOME_SPEC, RENAME, build_outcomes, to_binary

sys.path.insert(0, str(STUDY_RUN.parents[2] / "pipeline"))


def _label_map() -> dict[str, str]:
    """qid -> Qualtrics export column label.

    Reuses the exporter's own builder rather than reimplementing it.  The QSF's
    ``export_tag`` is the BASE name (``trust_competent``); Qualtrics appends the
    slider statement / matrix row index on export (``trust_competent_1``), which
    is what the codebook and ``clean_lib.R`` key on.  Rebuilding that suffix by
    hand is how a reader silently resolves 2 of 44 items.
    """
    from sbench.tier1_export import _qualtrics_label_map

    return _qualtrics_label_map()


def session_items(run_id: str, s1_source: str, s2_source: str,
                  labels: dict[str, str]) -> dict[str, float | None]:
    """Raw item values for one session, keyed by TARGET name (post-rename)."""
    merged: dict[str, dict] = {}
    for stage, source in (("1", s1_source), ("2", s2_source)):
        if not source:
            continue
        directory = STUDY_RUN / "data" / "batch" / (
            "round1_results" if stage == "1" else "round2_results")
        with open(directory / source, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("custom_id") != run_id:
                    continue
                parsed = json.loads(obj["response"]["body"]["choices"][0]["message"]["content"])
                answers = parsed.get("answers") if isinstance(parsed, dict) else parsed
                for answer in answers or []:
                    qid = str(answer.get("question_id"))
                    merged.setdefault(qid, answer)

    items: dict[str, float | None] = {}
    for qid, answer in merged.items():
        label = labels.get(qid)
        target = RENAME.get(label)
        if target is None:
            continue
        value = str(answer.get("answer_value", "")).strip() or \
            str(answer.get("answer_label", "")).strip()
        if target == "newsletter_signup":
            items[target] = to_binary(str(answer.get("answer_label", "")).strip() or value)
        elif target == "donation_ams":
            match = re.search(r"\d+", value)
            items[target] = float(match.group(0)) if match else None
        else:
            try:
                items[target] = float(value)
            except ValueError:
                # Twins sometimes write a number as a word ("fifty-five",
                # "thirty") despite the prompt asking for digits. The official
                # exporter converts these before cleaning, so dropping them here
                # would silently shrink a composite's item count and shift its
                # mean. Reuse the exporter's own table rather than a second one.
                from sbench.tier1_export import _word_to_number

                as_word = _word_to_number(value)
                items[target] = float(as_word) if as_word is not None else None
    return items


def run(n_sessions: int = 3) -> int:
    labels = _label_map()
    completeness = read_csv_dicts(
        STUDY_RUN / "audit" / "batch" / "completeness_by_session.csv")
    excluded = {r["base_pid"] for r in
                read_csv_dicts(STUDY_RUN / "data" / "tier1" / "tier1_exclusions.csv")}

    # one control session and two intervention sessions, from clean-pool twins
    picks, seen = [], set()
    for row in completeness:
        if row["base_pid"] in excluded or row["status"] != "exact":
            continue
        key = row["condition"]
        if key in seen or len(picks) >= n_sessions:
            continue
        seen.add(key)
        picks.append(row)

    demo = {"composite": "trust_multidimensional",
            "binary": "newsletter_signup",
            "single_item": "belief_post"}

    for row in picks:
        canonical = CODENAMES.get(row["raw_condition"].strip(), row["raw_condition"])
        items = session_items(row["run_id"], row["s1_source"], row["s2_source"], labels)
        built = build_outcomes(items)

        print("=" * 78)
        print(f"{row['base_pid']}   raw_condition={row['raw_condition']!r}")
        print(f"  -> canonical condition: {canonical!r}"
              f"{'   [POOLED CONTROL]' if canonical == 'control' else ''}")
        print(f"  raw items present: {sum(1 for v in items.values() if v is not None)}/44")

        for kind, name in demo.items():
            spec = OUTCOME_SPEC[name]
            value = built[name]
            lo, hi = spec["range"]
            inside = (not math.isnan(value)) and lo <= value <= hi
            print(f"\n  [{kind}] {name}")
            print(f"    formula : {spec['formula']}")
            print(f"    source  : {spec['source']}")
            sources = {k: items.get(k) for k in spec["items"]}
            shown = list(sources.items())[:6]
            print(f"    inputs  : {shown}{' ...' if len(sources) > 6 else ''}")
            if name == "trust_multidimensional":
                from .outcomes import _mean

                subs = {f: round(_mean([items.get(f"trust_{f}_{i}") for i in (1, 2, 3)]), 4)
                        for f in ("competence", "integrity", "benevolence", "openness")}
                print(f"    subscales: {subs}")
                flat = _mean([items.get(k) for k in spec["items"]])
                print(f"    mean-of-subscales = {value:.4f}   "
                      f"(flat mean of 12 items would be {flat:.4f})")
            print(f"    value   : {value:.4f}   range [{lo:g}, {hi:g}]   "
                  f"{'OK' if inside else 'OUT OF RANGE'}")

        out_of_range = [n for n, v in built.items()
                        if math.isnan(v) or not (OUTCOME_SPEC[n]["range"][0] <= v
                                                 <= OUTCOME_SPEC[n]["range"][1])]
        print(f"\n  all 13 outcomes built; out-of-range or NaN: "
              f"{out_of_range if out_of_range else 'none'}")
    return 0
