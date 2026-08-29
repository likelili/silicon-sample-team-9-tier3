"""Section 8 — logical-compatibility rules between survey answers and the
Twin-2K profile.

The rule is binary: a field is a ``conflict`` when the two non-missing values
cannot both be true of one person, ``no_conflict`` when they agree or their
categories/intervals overlap.  Comparisons are skipped when either side is
missing or has no valid profile counterpart.  There is no multi-conflict
threshold and no "detached" classification.

Where the two sides use different category schemes, the mapping is explicit
and deliberately generous — a panel level maps to every survey level it could
describe, and open-ended levels are compared as intervals:

  * age          — birth year -> age, tested against the profile band interval;
  * education    — strict: both questions ask the highest level COMPLETED and
                   the panel has a separate "Postgraduate" level, so
                   "College graduate/some postgrad" is incompatible with a
                   completed master's or doctorate;
  * income /     — interval overlap; disjoint intervals conflict
    household        ("More than 4" vs "6 or more" overlap; "3" vs "6 or more"
                   conflict);
  * religion     — the survey has no Agnostic/Atheist option, so the three
                   non-affiliated panel levels accept "I am not religious",
                   and "Other religion" with a non-affiliation free text
                   (e.g. "Agnostic") is compatible — encoded in the mapping
                   itself, not as an afterthought;
  * sex          — CAVEAT: the panel records sex assigned at birth, the survey
                   asks for gender; a mismatch is a strong signal rather than
                   a logical impossibility, and is reported with its own flag.
"""

from __future__ import annotations

from .common import QID, SURVEY_YEAR

PANEL_AGE_BANDS = {"18-29": (18, 29), "30-49": (30, 49), "50-64": (50, 64), "65+": (65, 120)}

RACE = {
    "White": {"White / Caucasian"},
    "Black": {"Black / African-American"},
    "Hispanic": {"Latino / Hispanic"},
    "Asian": {"Asian / Asian-American"},
    "Other": {"Other"},
}

EDUCATION = {
    "Less than high school": {"Less than high school"},
    "High school graduate": {"High school diploma / GED"},
    "Some college, no degree": {"Some college or Associate's degree"},
    "Associate's degree": {"Some college or Associate's degree"},
    "College graduate/some postgrad": {"Bachelor's degree"},
    "Postgraduate": {"Master's degree / Professional degree", "Doctorate degree / Ph.D."},
}

PANEL_INCOME = {
    "Less than $30,000": (0, 29_999),
    "$30,000-$50,000": (30_000, 50_000),
    "$50,000-$75,000": (50_000, 75_000),
    "$75,000-$100,000": (75_000, 100_000),
    "$100,000 or more": (100_000, 10**9),
}
SURVEY_INCOME = {
    "Less than $30,000": (0, 29_999),
    "$30,000 to $55,999": (30_000, 55_999),
    "$56,000 to $99,999": (56_000, 99_999),
    "$100,000 to $167,999": (100_000, 167_999),
    "$168,000 or more": (168_000, 10**9),
}

PARTY = {
    "Democrat": {"Democrat"},
    "Independent": {"Independent"},
    "Republican": {"Republican"},
    "Something else": {"Other (please specify)"},
}

NONRELIGIOUS_PANEL = {"Atheist", "Agnostic", "Nothing in particular"}
NONRELIGIOUS_FREE_TEXT = {
    "agnostic", "atheist", "none", "no religion", "nothing", "nothing in particular",
    "not religious", "non-religious", "nonreligious", "spiritual but not religious",
}
RELIGION = {
    "Protestant": {"Protestant"},
    "Roman Catholic": {"Catholic"},
    "Jewish": {"Jewish"},
    "Mormon": {"Mormon"},
    "Buddhist": {"Buddhist"},
    "Muslim": {"Muslim"},
    "Orthodox": {"Orthodox Christian"},
    "Hindu": {"Hindu"},
    "Atheist": {"I am not religious"},
    "Agnostic": {"I am not religious"},
    "Nothing in particular": {"I am not religious"},
    "Other": {"Other religion (please specify)"},
}

# field name -> (profile column, survey QID)
FIELDS = {
    "age": ("age_band", QID["year_birth"]),
    "sex": ("sex_at_birth", QID["gender"]),
    "race": ("race_origin", QID["race"]),
    "education": ("education", QID["education"]),
    "income": ("income_bracket", QID["income"]),
    "household": ("household_size", QID["household"]),
    "party": ("political_affiliation", QID["party"]),
    "religion": ("religion", QID["religion"]),
}
STRATIFICATION_FIELDS = ("age", "sex", "race")   # required for sampling eligibility


def _household_range(value: str):
    value = value.strip()
    if value.isdigit():
        return (int(value), int(value))
    if value.startswith("More than 4"):
        return (5, 99)
    if value.startswith("6 or more"):
        return (6, 99)
    if value.startswith("5"):
        return (5, 5)
    return None


def _overlaps(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def compare_field(field: str, profile_value: str, survey_value: str, free_text: str = "") -> str | None:
    """Return 'conflict' | 'no_conflict' | None (skipped: missing / unmappable)."""
    profile_value = (profile_value or "").strip()
    survey_value = (survey_value or "").strip()
    if not profile_value or not survey_value:
        return None

    if field == "age":
        if profile_value not in PANEL_AGE_BANDS or not survey_value.isdigit():
            return None
        age = SURVEY_YEAR - int(survey_value)
        low, high = PANEL_AGE_BANDS[profile_value]
        return "no_conflict" if low <= age <= high else "conflict"

    if field == "sex":
        return "no_conflict" if profile_value == survey_value else "conflict"

    if field == "race":
        if profile_value not in RACE:
            return None
        return "no_conflict" if survey_value in RACE[profile_value] else "conflict"

    if field == "education":
        if profile_value not in EDUCATION:
            return None
        return "no_conflict" if survey_value in EDUCATION[profile_value] else "conflict"

    if field == "income":
        if profile_value not in PANEL_INCOME or survey_value not in SURVEY_INCOME:
            return None
        return (
            "no_conflict"
            if _overlaps(PANEL_INCOME[profile_value], SURVEY_INCOME[survey_value])
            else "conflict"
        )

    if field == "household":
        a, b = _household_range(profile_value), _household_range(survey_value)
        if not a or not b:
            return None
        return "no_conflict" if _overlaps(a, b) else "conflict"

    if field == "party":
        if profile_value not in PARTY:
            return None
        return "no_conflict" if survey_value in PARTY[profile_value] else "conflict"

    if field == "religion":
        if profile_value not in RELIGION:
            return None
        if survey_value in RELIGION[profile_value]:
            return "no_conflict"
        if (
            profile_value in NONRELIGIOUS_PANEL
            and survey_value == "Other religion (please specify)"
            and free_text.strip().lower() in NONRELIGIOUS_FREE_TEXT
        ):
            return "no_conflict"
        return "conflict"

    raise ValueError(f"unknown field {field}")


def compare_twin(profile: dict, answers_by_qid: dict[str, str]) -> list[dict]:
    """One comparison row per evaluated field for one twin."""
    rows = []
    for field, (profile_column, qid) in FIELDS.items():
        free_text = answers_by_qid.get(QID["religion_text"], "") if field == "religion" else ""
        verdict = compare_field(
            field, profile.get(profile_column, ""), answers_by_qid.get(qid, ""), free_text
        )
        if verdict is None:
            continue
        rows.append(
            {
                "field": field,
                "profile_value": profile.get(profile_column, ""),
                "survey_value": answers_by_qid.get(qid, ""),
                "free_text": free_text,
                "result": verdict,
                "note": "sex-vs-gender: strong signal, not a logical impossibility" if field == "sex" else "",
            }
        )
    return rows
