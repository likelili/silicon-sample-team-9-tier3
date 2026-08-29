"""The 13 scored benchmark outcomes, constructed per participant-session.

Calibration targets are the **13 scored outcomes**, not the 44 raw survey
items.  Composites are therefore built *before* calibration, and the production
stage fits 17 conditions x 13 outcomes = **221** target relationships.

Every formula here is transcribed from the benchmark's own cleaning code, not
inferred from variable names.  Source of record:

  ``silicon-sample-submission/scripts/lib/clean_lib.R``      (recodes, composites)
  ``silicon-sample-submission/scripts/lib/submission_spec.R`` (outcome list, scales,
                                                               conditions, moderators)
  ``silicon-sample-submission/scripts/lib/check_lib.R``       (declared value ranges)

which the repository states reproduce ``data/cleaning.qmd :: clean_common`` from
the human study's own pipeline.

Three details are easy to get wrong and are called out because getting them
wrong is silent:

1. ``trust_multidimensional`` is the mean of the **four subscale means**, not
   the mean of the twelve items.  With complete data the two agree; with any
   item missing they do not, because the subscales would then carry unequal
   weight.  clean_lib.R builds the subscales first and averages those.
2. Composites use ``rowMeans(..., na.rm = TRUE)`` — the mean of whatever items
   are present, not listwise deletion.  A respondent missing one item still
   gets a composite.
3. ``funding_perceptions`` is **reverse-coded**: ``100 - funding_5``.

Ranges are the benchmark's *declared* scales (0-100, donation 0-10, newsletter
0/1), never observed sample minima and maxima.
"""

from __future__ import annotations

import math
from typing import Callable

# --- raw label -> target name, clean_lib.R .rename_map ----------------------
RENAME: dict[str, str] = {
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
    "inst_trust_gov_1": "inst_trust_federal_gov",
    "belief_post_1": "belief_post",
    "concern_1_1": "concern_1", "concern_2_1": "concern_2", "concern_3_1": "concern_3",
    "policy_general_1": "policy_general",
    "policy_specific_1_1": "policy_specific_1", "policy_specific_2_1": "policy_specific_2",
    "policy_specific_3_1": "policy_specific_3", "policy_specific_4_1": "policy_specific_4",
    "policy_specific_5_1": "policy_specific_5", "policy_specific_6_1": "policy_specific_6",
    "policy_specific_7_1": "policy_specific_7",
    "individual_meat_1": "behavior_meat", "individual_transport_1": "behavior_transport",
    "individual_solar_1": "behavior_solar", "individual_fly_1": "behavior_fly",
    "individual_talk_1": "behavior_talk", "individual_donate_1": "behavior_donate",
}

# --- the 17 conditions, submission_spec.R ----------------------------------
INTERVENTIONS = [
    "Corporate reliance", "Social justice", "Interview Prof. Maraun", "Funding",
    "Oil industry misinformation", "Measurement & modeling (1)", "Former skeptics",
    "High public trust", "Measurement & modeling (2)", "Peer-review",
    "Scientist community helpers", "Consensus", "Portrait Prof. Cherry",
    "Model accuracy", "Interview Prof. Sebille", "Extreme weather predictions",
]
CONDITIONS = ["control"] + INTERVENTIONS

# Raw survey code name -> canonical title.  The three control texts all map to
# "control": the pooled control is formed here, not treated as three arms.
CODENAMES: dict[str, str] = {
    "control neckties": "control", "control baseball": "control",
    "control dances": "control",
    "practical planarian": "Extreme weather predictions",
    "complicated cockroach": "Portrait Prof. Cherry",
    "flimsy fish": "Interview Prof. Maraun", "honored haddock": "Peer-review",
    "jealous jaguar": "Consensus", "phony parrotfish": "Funding",
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

MODERATORS: dict[str, list[str]] = {
    "gender": ["Male", "Female", "Other"],
    "age_band": ["18-29", "30-44", "45-59", "60+"],
    "race": ["White / Caucasian", "Black / African American",
             "Hispanic / Latino", "Asian / Asian American", "Other"],
    "education": ["Less than high school", "High school diploma / GED",
                  "Some college or Associate's degree", "Bachelor's degree",
                  "Master's degree / Professional degree", "Doctorate degree / Ph.D."],
    "income": ["Less than $30,000", "$30,000 to $55,999", "$56,000 to $99,999",
               "$100,000 to $167,999", "$168,000 or more"],
    "party": ["Republican", "Democrat", "Independent", "Other"],
}
MODERATOR_LEVELS = [(m, lv) for m, levels in MODERATORS.items() for lv in levels]  # 27


def _mean(values: list[float | None]) -> float:
    """rowMeans(..., na.rm = TRUE): mean of the present values, NaN if none."""
    present = [v for v in values if v is not None and not math.isnan(v)]
    return sum(present) / len(present) if present else math.nan


def _trust_multidimensional(get: Callable[[str], float | None]) -> float:
    """Mean of the FOUR subscale means — not the mean of the twelve items."""
    subscales = [
        _mean([get(f"trust_{facet}_{i}") for i in (1, 2, 3)])
        for facet in ("competence", "integrity", "benevolence", "openness")
    ]
    return _mean(subscales)


# --- the 13 scored outcomes -------------------------------------------------
# Each entry: builder, declared range, source items, and the formula as written
# in the official code, so the manifest can be generated rather than narrated.
OUTCOME_SPEC: dict[str, dict] = {
    "trust_multidimensional": {
        "kind": "composite",
        "range": (0.0, 100.0),
        "items": [f"trust_{f}_{i}" for f in
                  ("competence", "integrity", "benevolence", "openness") for i in (1, 2, 3)],
        "formula": "mean(mean(trust_competence_1..3), mean(trust_integrity_1..3), "
                   "mean(trust_benevolence_1..3), mean(trust_openness_1..3))",
        "source": "clean_lib.R:222-228",
        "build": _trust_multidimensional,
    },
    "trust_post": {
        "kind": "single_item", "range": (0.0, 100.0), "items": ["trust_post"],
        "formula": "trust_post_1 (renamed)", "source": "clean_lib.R:.rename_map",
        "build": lambda get: _mean([get("trust_post")]),
    },
    "distrust_post": {
        "kind": "single_item", "range": (0.0, 100.0), "items": ["distrust_post"],
        "formula": "distrust_1 (renamed)", "source": "clean_lib.R:.rename_map",
        "build": lambda get: _mean([get("distrust_post")]),
    },
    "funding_perceptions": {
        "kind": "single_item_reversed", "range": (0.0, 100.0),
        "items": ["funding_perceptions"],
        "formula": "100 - funding_5", "source": "clean_lib.R:218",
        "build": lambda get: (math.nan if get("funding_perceptions") is None
                              or math.isnan(get("funding_perceptions"))
                              else 100.0 - get("funding_perceptions")),
    },
    "policy_role_mean": {
        "kind": "composite", "range": (0.0, 100.0),
        "items": [f"policy_role_{i}" for i in range(1, 5)],
        "formula": "mean(policy_role_1..4)", "source": "clean_lib.R:229",
        "build": lambda get: _mean([get(f"policy_role_{i}") for i in range(1, 5)]),
    },
    "inst_trust_mean": {
        "kind": "composite", "range": (0.0, 100.0),
        "items": ["inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa",
                  "inst_trust_universities", "inst_trust_federal_gov"],
        "formula": "mean(inst_trust_epa, nasa, noaa, universities, federal_gov)",
        "source": "clean_lib.R:230-231",
        "build": lambda get: _mean([get(k) for k in
                                    ("inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa",
                                     "inst_trust_universities", "inst_trust_federal_gov")]),
    },
    "belief_post": {
        "kind": "single_item", "range": (0.0, 100.0), "items": ["belief_post"],
        "formula": "belief_post_1 (renamed)", "source": "clean_lib.R:.rename_map",
        "build": lambda get: _mean([get("belief_post")]),
    },
    "concern_mean": {
        "kind": "composite", "range": (0.0, 100.0),
        "items": [f"concern_{i}" for i in (1, 2, 3)],
        "formula": "mean(concern_1..3)", "source": "clean_lib.R:232",
        "build": lambda get: _mean([get(f"concern_{i}") for i in (1, 2, 3)]),
    },
    "policy_general": {
        "kind": "single_item", "range": (0.0, 100.0), "items": ["policy_general"],
        "formula": "policy_general_1 (renamed)", "source": "clean_lib.R:.rename_map",
        "build": lambda get: _mean([get("policy_general")]),
    },
    "policy_specific_mean": {
        "kind": "composite", "range": (0.0, 100.0),
        "items": [f"policy_specific_{i}" for i in range(1, 8)],
        "formula": "mean(policy_specific_1..7)", "source": "clean_lib.R:233",
        "build": lambda get: _mean([get(f"policy_specific_{i}") for i in range(1, 8)]),
    },
    "behavior_mean": {
        "kind": "composite", "range": (0.0, 100.0),
        "items": ["behavior_meat", "behavior_transport", "behavior_solar",
                  "behavior_fly", "behavior_talk", "behavior_donate"],
        "formula": "mean(behavior_meat, transport, solar, fly, talk, donate)",
        "source": "clean_lib.R:234-235",
        "build": lambda get: _mean([get(k) for k in
                                    ("behavior_meat", "behavior_transport", "behavior_solar",
                                     "behavior_fly", "behavior_talk", "behavior_donate")]),
    },
    "donation_ams": {
        "kind": "single_item", "range": (0.0, 10.0), "items": ["donation_ams"],
        "formula": "as.numeric(donation)", "source": "clean_lib.R:217",
        "build": lambda get: _mean([get("donation_ams")]),
    },
    "newsletter_signup": {
        "kind": "binary", "range": (0.0, 1.0), "items": ["newsletter_signup"],
        "formula": "1 if value in {1,'yes','true'}; 0 if in {0,2,'no','false'}; else NA",
        "source": "clean_lib.R:.to_binary (131-140)",
        "build": lambda get: _mean([get("newsletter_signup")]),
    },
}
OUTCOMES = list(OUTCOME_SPEC)                      # 13, submission_spec.R order
TARGETS = [(c, o) for c in CONDITIONS for o in OUTCOMES]      # 221


def to_binary(value: str | None) -> float:
    """clean_lib.R .to_binary — 1/'yes'/'true' -> 1; 0/2/'no'/'false' -> 0."""
    if value is None:
        return math.nan
    text = str(value).strip().lower()
    if not text:
        return math.nan
    try:
        number = float(text)
    except ValueError:
        number = None
    if number == 1 or text in ("yes", "true"):
        return 1.0
    if number in (0, 2) or text in ("no", "false"):
        return 0.0
    return math.nan


def build_outcomes(items: dict[str, float | None]) -> dict[str, float]:
    """The 13 scored outcomes for one participant-session.

    ``items`` is keyed by TARGET name (post-rename), values already numeric with
    None/NaN for missing.  ``newsletter_signup`` must already be 0/1 via
    ``to_binary``; ``funding_perceptions`` must be the RAW funding_5 value — the
    reversal happens here so it cannot be applied twice.
    """
    def get(key: str) -> float | None:
        value = items.get(key)
        if value is None:
            return None
        return float(value)

    return {name: spec["build"](get) for name, spec in OUTCOME_SPEC.items()}


def target_columns() -> list[str]:
    """The 221 production column names, ``<condition>||<outcome>``."""
    return [f"{c}||{o}" for c, o in TARGETS]


def definition_manifest() -> dict:
    """The outcome-definition manifest: formula, range and source per outcome."""
    return {
        "source_of_record": {
            "recodes_and_composites": "silicon-sample-submission/scripts/lib/clean_lib.R",
            "outcome_list_scales_moderators": "silicon-sample-submission/scripts/lib/submission_spec.R",
            "declared_value_ranges": "silicon-sample-submission/scripts/lib/check_lib.R",
            "upstream": "the repo states these reproduce data/cleaning.qmd :: clean_common",
        },
        "n_conditions": len(CONDITIONS),
        "n_outcomes": len(OUTCOMES),
        "n_targets": len(TARGETS),
        "pooled_control": "the three control texts (neckties / baseball / dances) map to "
                          "the single condition 'control' via submission_spec.R codenames; "
                          "each twin contributes exactly one control session",
        "missing_value_rule": "composites use rowMeans(na.rm = TRUE) — the mean of the "
                              "items present; NaN only when every item is missing",
        "outcomes": {
            name: {"kind": spec["kind"], "range": list(spec["range"]),
                   "n_source_items": len(spec["items"]), "source_items": spec["items"],
                   "formula": spec["formula"], "source_file_line": spec["source"]}
            for name, spec in OUTCOME_SPEC.items()
        },
        "raw_item_count": len(RENAME),
        "rename_map": RENAME,
    }
