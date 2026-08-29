"""Tier-1 submission exporter — raw Qualtrics-style export + cleaned prediction file.

The benchmark's intended Tier-1 path is: drop a raw export into
``raw_data_deposit/`` and run their ``make clean`` (R). This machine has no R,
so this module does both halves in Python:

1. ``build_raw_export`` writes the plain one-header raw CSV the benchmark
   accepts (README: "so does a plain one-header CSV"), with Qualtrics variable
   names driven by the QSF's own DataExportTags plus the slider statement-id
   suffix, asserted equal to the codebook's label set — never transcribed by
   hand.
2. ``python_clean`` is a line-by-line port of ``scripts/lib/clean_lib.R``
   (rename map, demographic recodes, birth-year parser, 100−funding reverse
   code, na.rm row means, the age-band cut, the exact 33-column schema).
3. ``parity_check`` proves the port: cleaning the benchmark's shipped
   ``example_raw_export.csv`` must reproduce their shipped
   ``example_T1_primary_v1.csv`` value-for-value. If that fails, nothing else
   runs.
4. ``check_t1`` ports the structural checks of ``scripts/lib/check_lib.R``.

Deliberate value handling in the raw export (kept verbatim so the deposited
raw file is an honest record, normalized only where R's ``as.numeric`` would
otherwise silently destroy data):

* donation: "$3" / "8$" / "3" -> integer 3/8/3. R coerces "$3" to NA, which
  would blank 8,995 of 9,000 donations — normalization is required, and the
  audit called for it.
* number words ("fifty-five") -> digits, logged per occurrence.
* "I never fly" / "I never drive by myself" stay as text: they coerce to NA
  and drop out of ``behavior_mean``'s na.rm mean — the benchmark's own
  cleaning semantics for not-applicable items.
* duplicate question ids within a response: first occurrence wins (the audit
  verified duplicates are identical).
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from .common import ARTIFACTS, AUDIT, BENCH_ROOT, DATA, load_json, read_csv, save_json

SUBMISSION_SRC = BENCH_ROOT / "silicon-sample-submission"

# Two-letter experiment codes for profile ids (user-specified format:
# <code>_<base_pid>). Unique across all 17 conditions; control distinct.
CONDITION_CODE = {
    "control": "CT",
    "apple aardvark": "AA",
    "complicated cockroach": "CC",
    "crushing chicken; gross grasshopper; homely halibut": "CH",
    "difficult dog": "DD",
    "flimsy fish": "FF",
    "giant gibbon; brick bobcat": "GB",
    "heartfelt hummingbird": "HH",
    "honored haddock": "HK",
    "jealous jaguar": "JJ",
    "limping llama; friendly frog": "LL",
    "orchid orangutan; defiant dragonfly": "OD",
    "perfect prawn": "PP",
    "periwinkle partridge": "PW",
    "phony parrotfish": "PF",
    "practical planarian": "PL",
    "worse wildfowl": "WW",
}

# --- the benchmark's canonical spec (ported from submission_spec.R) ---------

CODENAMES = {
    "control neckties": "control", "control baseball": "control",
    "control dances": "control",
    "practical planarian": "Extreme weather predictions",
    "complicated cockroach": "Portrait Prof. Cherry",
    "flimsy fish": "Interview Prof. Maraun",
    "honored haddock": "Peer-review",
    "jealous jaguar": "Consensus",
    "phony parrotfish": "Funding",
    "crushing chicken; gross grasshopper; homely halibut": "High public trust",
    "worse wildfowl": "Oil industry misinformation",
    "periwinkle partridge": "Scientist community helpers",
    "difficult dog": "Social justice",
    "giant gibbon; brick bobcat": "Corporate reliance",
    "limping llama; friendly frog": "Former skeptics",
    "perfect prawn": "Measurement & modeling (1)",
    "orchid orangutan; defiant dragonfly": "Measurement & modeling (2)",
    "apple aardvark": "Model accuracy",
    "heartfelt hummingbird": "Interview Prof. Sebille",
}
CONDITIONS = ["control"] + sorted(set(CODENAMES.values()) - {"control"})
TRUST_ITEMS = [f"trust_{s}_{i}" for s in ("competence", "integrity", "benevolence", "openness")
               for i in (1, 2, 3)]
MODERATORS = {
    "gender": ["Male", "Female", "Other"],
    "age_band": ["18-29", "30-44", "45-59", "60+"],
    "race": ["White / Caucasian", "Black / African American", "Hispanic / Latino",
             "Asian / Asian American", "Other"],
    "education": ["Less than high school", "High school diploma / GED",
                  "Some college or Associate's degree", "Bachelor's degree",
                  "Master's degree / Professional degree", "Doctorate degree / Ph.D."],
    "income": ["Less than $30,000", "$30,000 to $55,999", "$56,000 to $99,999",
               "$100,000 to $167,999", "$168,000 or more"],
    "party": ["Republican", "Democrat", "Independent", "Other"],
}
TIER1_REQUIRED = (["profile_id", "condition"] + list(MODERATORS) +
                  ["trust_multidimensional"] + TRUST_ITEMS +
                  ["trust_post", "distrust_post", "funding_perceptions",
                   "policy_role_mean", "inst_trust_mean", "belief_post",
                   "concern_mean", "policy_general", "policy_specific_mean",
                   "behavior_mean", "donation_ams", "newsletter_signup"])
SCALE_0_100 = ["trust_multidimensional", "trust_post", "distrust_post",
               "funding_perceptions", "policy_role_mean", "inst_trust_mean",
               "belief_post", "concern_mean", "policy_general",
               "policy_specific_mean", "behavior_mean"]

RENAME = {  # raw Qualtrics label -> target name (clean_lib.R .rename_map, inverted orientation)
    "trust_competent_1": "trust_competence_1", "trust_intelligent_1": "trust_competence_2",
    "trust_qualified_1": "trust_competence_3", "trust_honest_1": "trust_integrity_1",
    "trust_ethical_1": "trust_integrity_2", "trust_sincere_1": "trust_integrity_3",
    "trust_concerned_1": "trust_benevolence_1", "trust_improve_1": "trust_benevolence_2",
    "trust_considerate_1": "trust_benevolence_3", "trust_feedback_1": "trust_openness_1",
    "trust_transparent_1": "trust_openness_2", "trust_attention_1": "trust_openness_3",
    "trust_post_1": "trust_post", "distrust_1": "distrust_post",
    "donation": "donation_ams", "newsletter": "newsletter_signup",
    "funding_5": "funding_perceptions",
    "policy_1_1": "policy_role_1", "policy_2_1": "policy_role_2",
    "policy_3_1": "policy_role_3", "policy_4_1": "policy_role_4",
    "inst_trust_epa_1": "inst_trust_epa", "inst_trust_nasa_1": "inst_trust_nasa",
    "inst_trust_noaa_1": "inst_trust_noaa", "inst_trust_uni_1": "inst_trust_universities",
    "inst_trust_gov_1": "inst_trust_federal_gov", "belief_post_1": "belief_post",
    "concern_1_1": "concern_1", "concern_2_1": "concern_2", "concern_3_1": "concern_3",
    "policy_general_1": "policy_general",
    **{f"policy_specific_{i}_1": f"policy_specific_{i}" for i in range(1, 8)},
    "individual_meat_1": "behavior_meat", "individual_transport_1": "behavior_transport",
    "individual_solar_1": "behavior_solar", "individual_fly_1": "behavior_fly",
    "individual_talk_1": "behavior_talk", "individual_donate_1": "behavior_donate",
}
RAW_ITEM_COLUMNS = list(RENAME)          # the 44 study item columns, export order
GENDER_MAP = {"1": "Male", "2": "Female", "3": "Other"}
RACE_MAP = {"1": "White / Caucasian", "2": "Black / African American",
            "3": "Hispanic / Latino", "4": "Asian / Asian American", "5": "Other",
            "Black / African-American": "Black / African American",
            "Latino / Hispanic": "Hispanic / Latino",
            "Asian / Asian-American": "Asian / Asian American"}
EDU_MAP = {"1": "Less than high school", "2": "High school diploma / GED",
           "3": "Some college or Associate's degree", "4": "Bachelor's degree",
           "5": "Master's degree / Professional degree", "6": "Doctorate degree / Ph.D."}
INCOME_MAP = {"1": "Less than $30,000", "2": "$30,000 to $55,999",
              "3": "$56,000 to $99,999", "4": "$100,000 to $167,999",
              "5": "$168,000 or more"}
PARTY_MAP = {"1": "Republican", "2": "Democrat", "3": "Independent", "4": "Other",
             "Other (please specify)": "Other"}

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100,
    "fifty-five": 55, "sixty-five": 65, "seventy-five": 75, "eighty-five": 85,
    "twenty-five": 25, "thirty-five": 35, "forty-five": 45, "ninety-five": 95,
}


# ---------------------------------------------------------------------------
# Raw export
# ---------------------------------------------------------------------------

def _qualtrics_label_map() -> dict[str, str]:
    """qid -> Qualtrics export column label, from the QSF itself."""
    audit = load_json(ARTIFACTS / "qsf_audit.json")
    tags = {q["qid"]: q["export_tag"] for b in audit["block_inventory"]
            for q in b["questions"] if q.get("export_tag")}
    qsf = load_json(SUBMISSION_SRC / "survey" / "survey.qsf")
    labels: dict[str, str] = {}
    for element in qsf["SurveyElements"]:
        if element.get("Element") != "SQ":
            continue
        payload = element.get("Payload") or {}
        qid = str(payload.get("QuestionID"))
        tag = tags.get(qid)
        if not tag:
            continue
        if str(payload.get("QuestionType")) == "Slider":
            statements = list((payload.get("Choices") or {}).keys())
            if len(statements) == 1:
                labels[qid] = f"{tag}_{statements[0]}"
                continue
        labels[qid] = tag
    produced = {v for v in labels.values() if v in RAW_ITEM_COLUMNS}
    missing = [c for c in RAW_ITEM_COLUMNS if c not in produced]
    if missing:
        raise SystemExit(f"tier1-export: QSF-derived labels missing {missing}")
    return labels


def _word_to_number(text: str) -> str | None:
    key = text.strip().lower().rstrip(".")
    if key in _WORD_NUMBERS:
        return str(_WORD_NUMBERS[key])
    return None


def _adopted_answers(session_rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """run_id -> stage answers from each session's RECORDED adopted source file."""
    wanted: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for row in session_rows:
        for stage, src in (("1", row["s1_source"]), ("2", row["s2_source"])):
            if src:
                wanted.setdefault((stage, src), []).append((row["run_id"], stage))
    out: dict[str, dict[str, list[dict]]] = {}
    for (stage, src), members in wanted.items():
        ids = {run_id for run_id, _ in members}
        directory = DATA / "batch" / ("round1_results" if stage == "1" else "round2_results")
        path = directory / src
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                run_id = obj.get("custom_id")
                if run_id not in ids:
                    continue
                choice = obj["response"]["body"]["choices"][0]
                parsed = json.loads(choice["message"]["content"])
                answers = parsed.get("answers") if isinstance(parsed, dict) else parsed
                out.setdefault(run_id, {})[stage] = answers
    return out


def build_raw_export(out_path: Path) -> dict:
    labels = _qualtrics_label_map()
    selection = read_csv(DATA / "tier1" / "tier1_selection.csv")
    completeness = {r["run_id"]: r
                    for r in read_csv(AUDIT / "batch" / "completeness_by_session.csv")}
    # (base_pid, tier1 condition) -> session record
    by_key: dict[tuple[str, str], dict] = {}
    for row in completeness.values():
        key = (row["base_pid"],
               "control" if row["condition"] == "control" else row["raw_condition"])
        by_key[key] = row
    wide = {r["persona_id"]: r for r in read_csv(DATA / "prestudy_frozen_wide.csv")}

    sessions = []
    for pick in selection:
        session = by_key.get((pick["base_pid"], pick["condition"]))
        if session is None or session["status"] != "exact":
            raise SystemExit(f"tier1-export: selected session not exact: {pick}")
        sessions.append((pick, session))
    answers_by_run = _adopted_answers([s for _, s in sessions])

    header = (["profile_id", "condition", "gender", "year_birth", "race",
               "education", "income", "party"] + RAW_ITEM_COLUMNS)
    stats = {"rows": 0, "dedup_dropped": 0, "word_numbers": [], "donation_forms": {},
             "never_items": 0}
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for pick, session in sessions:
            run_id = session["run_id"]
            stages = answers_by_run.get(run_id)
            if not stages:
                raise SystemExit(f"tier1-export: no adopted answers for {run_id}")
            merged: dict[str, dict] = {}
            for stage in ("1", "2"):
                for answer in stages.get(stage, []):
                    qid = str(answer.get("question_id"))
                    if qid in merged:
                        stats["dedup_dropped"] += 1
                        continue
                    merged[qid] = answer

            values: dict[str, str] = {}
            for qid, answer in merged.items():
                label = labels.get(qid)
                if label not in RENAME:
                    continue
                raw_value = str(answer.get("answer_value", "")).strip()
                raw_label = str(answer.get("answer_label", "")).strip()
                value = raw_value or raw_label
                if label == "donation":
                    match = re.search(r"\d+", value)
                    if not match:
                        raise SystemExit(f"tier1-export: unparseable donation {value!r} "
                                         f"in {run_id}")
                    stats["donation_forms"][value] = stats["donation_forms"].get(value, 0) + 1
                    value = match.group(0)
                elif label == "newsletter":
                    value = raw_label or raw_value
                else:
                    numeric = re.fullmatch(r"-?\d+(\.\d+)?", value)
                    if not numeric:
                        as_word = _word_to_number(value)
                        if as_word is not None:
                            stats["word_numbers"].append((run_id, label, value))
                            value = as_word
                        else:
                            stats["never_items"] += 1   # stays text -> NA in cleaning
                values[label] = value

            missing_items = [c for c in RAW_ITEM_COLUMNS if c not in values]
            if missing_items:
                raise SystemExit(f"tier1-export: {run_id} missing items {missing_items}")

            demo = wide[pick["base_pid"]]
            raw_condition = (session["raw_condition"]
                             if pick["condition"] == "control" else pick["condition"])
            writer.writerow(
                [f"{CONDITION_CODE[pick['condition']]}_{pick['base_pid']}",
                 raw_condition, demo["gender"], demo["year_birth"], demo["race"],
                 demo["education"], demo["income"], demo["party"]]
                + [values[c] for c in RAW_ITEM_COLUMNS])
            stats["rows"] += 1
    return stats


# ---------------------------------------------------------------------------
# Python port of clean_lib.R
# ---------------------------------------------------------------------------

def _extract_birth_year(text: str) -> float:
    x = str(text or "")
    years = re.findall(r"\b(19\d{2}|200[0-9])\b", x)
    if len(years) == 1:
        return float(years[0])
    if len(years) > 1:
        born = re.search(r"born in (19\d{2}|200[0-9])", x)
        return float(born.group(1)) if born else float(years[-1])
    digits = re.search(r"\d+", x)
    if digits:
        age = float(digits.group(0))
        if 18 <= age <= 100:
            return 2026 - age
    return math.nan


def _to_binary(value: str):
    s = str(value or "").strip().lower()
    try:
        n = float(s)
    except ValueError:
        n = None
    if n == 1 or s in ("yes", "true"):
        return 1
    if n in (0, 2) or s in ("no", "false"):
        return 0
    return math.nan


def _recode(value: str, mapping: dict, what: str) -> str | float:
    s = str(value or "").strip()
    if not s or s.upper() == "NA":
        return math.nan
    if s in mapping:
        return mapping[s]
    if s in mapping.values():
        return s
    raise SystemExit(f"python_clean: unrecognized {what} value {s!r}")


def _num(value) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return math.nan


def _row_mean(values: list[float]) -> float:
    usable = [v for v in values if not math.isnan(v)]
    return sum(usable) / len(usable) if usable else math.nan


def python_clean(raw_path: Path) -> list[dict]:
    with open(raw_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # strip a genuine Qualtrics double header if present
    drop = 0
    for i in range(min(2, len(rows))):
        if any("ImportId" in str(v) for v in rows[i].values()):
            drop = i + 1
    rows = rows[drop:]

    out = []
    for raw in rows:
        d = {RENAME.get(k, k): v for k, v in raw.items()}
        condition = re.sub(r"\s+", " ", str(d.get("condition", ""))).strip()
        condition = CODENAMES.get(condition, condition)
        if condition not in CONDITIONS:
            raise SystemExit(f"python_clean: unrecognized condition {condition!r}")

        age = 2026 - _extract_birth_year(d.get("year_birth", ""))
        numerics = {k: _num(d.get(k)) for k in
                    TRUST_ITEMS + ["trust_post", "distrust_post"]
                    + [f"policy_role_{i}" for i in range(1, 5)]
                    + ["inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa",
                       "inst_trust_universities", "inst_trust_federal_gov",
                       "belief_post", "concern_1", "concern_2", "concern_3",
                       "policy_general"]
                    + [f"policy_specific_{i}" for i in range(1, 8)]
                    + ["behavior_meat", "behavior_transport", "behavior_solar",
                       "behavior_fly", "behavior_talk", "behavior_donate"]}
        record = {
            "profile_id": d.get("profile_id", ""),
            "condition": condition,
            "gender": _recode(d.get("gender"), GENDER_MAP, "gender"),
            "race": _recode(d.get("race"), RACE_MAP, "race"),
            "education": _recode(d.get("education"), EDU_MAP, "education"),
            "income": _recode(d.get("income"), INCOME_MAP, "income"),
            "party": _recode(d.get("party"), PARTY_MAP, "party"),
            **numerics,
            "donation_ams": _num(d.get("donation_ams")),
            "funding_perceptions": 100 - _num(d.get("funding_perceptions")),
            "newsletter_signup": _to_binary(d.get("newsletter_signup")),
        }
        subs = {s: _row_mean([record[f"trust_{s}_{i}"] for i in (1, 2, 3)])
                for s in ("competence", "integrity", "benevolence", "openness")}
        record["trust_multidimensional"] = _row_mean(list(subs.values()))
        record["policy_role_mean"] = _row_mean([record[f"policy_role_{i}"] for i in range(1, 5)])
        record["inst_trust_mean"] = _row_mean(
            [record[k] for k in ("inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa",
                                 "inst_trust_universities", "inst_trust_federal_gov")])
        record["concern_mean"] = _row_mean([record[f"concern_{i}"] for i in (1, 2, 3)])
        record["policy_specific_mean"] = _row_mean(
            [record[f"policy_specific_{i}"] for i in range(1, 8)])
        record["behavior_mean"] = _row_mean(
            [record[k] for k in ("behavior_meat", "behavior_transport", "behavior_solar",
                                 "behavior_fly", "behavior_talk", "behavior_donate")])
        if isinstance(age, float) and math.isnan(age):
            record["age_band"] = math.nan
        elif age <= 29:
            record["age_band"] = "18-29"
        elif age <= 44:
            record["age_band"] = "30-44"
        elif age <= 59:
            record["age_band"] = "45-59"
        else:
            record["age_band"] = "60+"
        # R's cut() with breaks starting at 17 leaves age <= 17 out of every band
        if isinstance(age, float) and not math.isnan(age) and age <= 17:
            record["age_band"] = math.nan
        out.append({k: record.get(k, math.nan) for k in TIER1_REQUIRED})
    return out


def _format_value(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        if value == int(value):
            return str(int(value))
        return repr(value)
    return str(value)


def write_clean_csv(records: list[dict], path: Path) -> None:
    # lineterminator="\n": the csv module defaults to CRLF, while the
    # benchmark's own R cleaner (readr::write_csv) emits LF. The values are
    # identical either way, but matching the official tool byte-for-byte means
    # the SHA-256 recorded in metadata.json is the same whichever cleaner
    # produced the file.
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TIER1_REQUIRED)
        for record in records:
            writer.writerow([_format_value(record[k]) for k in TIER1_REQUIRED])


# ---------------------------------------------------------------------------
# Parity with the benchmark's own cleaner, and the ported checks
# ---------------------------------------------------------------------------

def parity_check() -> None:
    """python_clean(example raw) must reproduce the shipped example predictions."""
    mine = python_clean(SUBMISSION_SRC / "raw_data_deposit" / "example_raw_export.csv")
    with open(SUBMISSION_SRC / "predictions" / "example_T1_primary_v1.csv",
              newline="", encoding="utf-8") as handle:
        theirs = list(csv.DictReader(handle))
    if len(mine) != len(theirs):
        raise SystemExit(f"parity: row count {len(mine)} vs {len(theirs)}")
    mismatch = 0
    for i, (a, b) in enumerate(zip(mine, theirs, strict=True)):
        for column in TIER1_REQUIRED:
            va, vb = a[column], str(b[column])
            if isinstance(va, float):
                if math.isnan(va):
                    same = vb in ("NA", "NaN", "")
                else:
                    try:
                        same = abs(va - float(vb)) < 1e-9
                    except ValueError:
                        same = False
            else:
                same = str(va) == vb
            if not same:
                mismatch += 1
                if mismatch <= 5:
                    print(f"  parity mismatch row {i} {column}: mine={va!r} theirs={vb!r}")
    if mismatch:
        raise SystemExit(f"parity: {mismatch} value mismatch(es) vs the shipped example — "
                         f"the Python port does not match clean_lib.R")
    print(f"parity: python_clean reproduces the shipped example exactly "
          f"({len(mine)} rows x {len(TIER1_REQUIRED)} columns)")


def check_t1(records: list[dict]) -> list[str]:
    """Port of check_lib.R .check_t1 — returns failures (hard) plus warnings."""
    problems: list[str] = []
    counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    for record in records:
        counts[record["condition"]] = counts.get(record["condition"], 0) + 1
        pid = record["profile_id"]
        if pid in seen_ids:
            problems.append(f"duplicate profile_id {pid}")
        seen_ids.add(pid)
        for moderator, levels in MODERATORS.items():
            value = record[moderator]
            if isinstance(value, float) and math.isnan(value):
                problems.append(f"{moderator} NA for {pid}")
            elif value not in levels:
                problems.append(f"{moderator} invalid {value!r} for {pid}")
        for outcome in SCALE_0_100 + TRUST_ITEMS:
            value = record[outcome]
            if not isinstance(value, float) or math.isnan(value):
                problems.append(f"{outcome} missing for {pid}")
            elif not 0 <= value <= 100:
                problems.append(f"{outcome} out of range ({value}) for {pid}")
        donation = record["donation_ams"]
        if math.isnan(donation) or not 0 <= donation <= 10:
            problems.append(f"donation_ams bad ({donation}) for {pid}")
        if record["newsletter_signup"] not in (0, 1):
            problems.append(f"newsletter_signup bad for {pid}")
        subscales = [_row_mean([record[f"trust_{s}_{i}"] for i in (1, 2, 3)])
                     for s in ("competence", "integrity", "benevolence", "openness")]
        if abs(record["trust_multidimensional"] - _row_mean(subscales)) > 0.51:
            problems.append(f"trust_multidimensional inconsistent for {pid}")
    for condition in CONDITIONS:
        n = counts.get(condition, 0)
        floor = 1000 if condition == "control" else 500
        if n < floor:
            problems.append(f"{condition}: {n} rows, below the {floor} floor")
    return problems


def run_export(out_dir: Path | str | None = None) -> int:
    out_dir = Path(out_dir) if out_dir else (DATA / "tier1")
    parity_check()

    raw_path = out_dir / "tier1_raw_export.csv"
    stats = build_raw_export(raw_path)
    print(f"raw export: {stats['rows']:,} rows -> {raw_path}")
    print(f"  duplicate answers dropped : {stats['dedup_dropped']}")
    print(f"  number-words normalized   : {len(stats['word_numbers'])} "
          f"{stats['word_numbers'][:4]}")
    odd = {k: v for k, v in stats["donation_forms"].items()
           if not re.fullmatch(r"\$\d+", k)}
    print(f"  donation formats normalized: {sum(stats['donation_forms'].values()):,} total, "
          f"non-'$N' forms: {odd}")
    print(f"  not-applicable text answers kept verbatim: {stats['never_items']}")

    records = python_clean(raw_path)
    clean_path = out_dir / "tier1_predictions.csv"
    write_clean_csv(records, clean_path)
    print(f"cleaned: {len(records):,} rows x {len(TIER1_REQUIRED)} columns -> {clean_path}")

    problems = check_t1(records)
    n_conditions = len({r['condition'] for r in records})
    print(f"check_t1: {n_conditions}/17 conditions, "
          f"{len(problems)} problem(s)")
    for problem in problems[:10]:
        print("  !!", problem)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["condition"]] = counts.get(record["condition"], 0) + 1
    print("  per-condition N:", {k: counts[k] for k in sorted(counts)[:4]}, "...")
    save_json({"raw_rows": stats["rows"], "clean_rows": len(records),
               "dedup_dropped": stats["dedup_dropped"],
               "word_numbers": [list(w) for w in stats["word_numbers"]],
               "not_applicable_text": stats["never_items"],
               "check_problems": problems,
               "condition_codes": CONDITION_CODE},
              out_dir / "tier1_export_report.json")
    return 0 if not problems else 1
