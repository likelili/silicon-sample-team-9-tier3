"""Declared scale ranges for the 123 Wave-4 anchor items.

Errors are normalised by each item's **declared** scale, read from the survey
definition — never by the observed sample minimum and maximum.  An observed
range is a property of the respondents who happened to answer; normalising by it
makes the same absolute error score differently depending on how much spread the
sample happened to show, and lets a degenerate item (everyone picked the same
option) blow up or vanish.

Sources, per question type, from the Twin-2K-500 Wave-4 block JSON:

  choice (MC / matrix row)  ``len(Options)`` or ``len(Columns)`` -> span is
                            ``n_options - 1``, because the encoding is the
                            1-based option position;
  slider                    ``Range.Max - Range.Min`` as declared in the QSF;
  numeric text entry        no declared bound exists in the instrument.

That last category is the honest gap.  Three anchoring/sunk-cost items are free
numeric entry with no bound in the survey definition; two of them state a bound
in the question text ("Enter a number between 0 and 20") and one does not.
Rather than invent a scale, items with no declared range are **excluded from the
normalised error aggregate** and reported as such — they still serve as donor
columns and still get an ``item_r`` diagnostic.
"""

from __future__ import annotations

import json
import re

from .common import ANSWER_BLOCKS, CALIB_DATA, anchor_items, clean_pool, save_json

# Bounds stated in the question text where the instrument declares none.
# Transcribed from the item wording, and recorded so the choice is auditable.
_TEXT_STATED_BOUNDS: dict[str, tuple[float, float]] = {
    # "how many of your next 20 coffee purchases would be from Java Coffee?"
    "QID181": (0.0, 20.0),
    "QID182": (0.0, 20.0),
}


def _raw_questions(pid: str) -> dict[str, dict]:
    with open(ANSWER_BLOCKS / f"{pid}_wave4_Q_wave4_A.json", encoding="utf-8") as handle:
        blocks = json.load(handle)
    return {q["QuestionID"]: q for b in blocks for q in b.get("Questions", [])}


def _sub_ids(question: dict) -> list[str]:
    """Item ids the runtime expands this question into, by its own convention."""
    qid = question["QuestionID"]
    qtype = str(question.get("QuestionType", "")).upper()
    if qtype == "MATRIX":
        rows = question.get("RowsID") or [str(i + 1) for i in range(len(question.get("Rows", [])))]
        return [f"{qid}_{r}" for r in rows]
    if qtype == "SLIDER":
        statements = question.get("StatementsID") or [
            str(i + 1) for i in range(len(question.get("Statements", [])))]
        if len(statements) == 1 and not question.get("StatementsID"):
            return [qid]
        return [f"{qid}_{s}" for s in statements]
    return [qid]


def declared_ranges() -> dict[str, dict]:
    """item_id -> {span, lo, hi, basis} for every anchor item that has one."""
    wanted = set(anchor_items())
    found: dict[str, dict] = {}
    for pid in clean_pool():
        if len(found) >= len(wanted):
            break
        for qid, question in _raw_questions(pid).items():
            qtype = str(question.get("QuestionType", "")).upper()
            options = question.get("Options") or question.get("Columns") or []
            rng = question.get("Range") or {}
            for item_id in _sub_ids(question):
                if item_id not in wanted or item_id in found:
                    continue
                if options:
                    span = len(options) - 1
                    found[item_id] = {"lo": 1.0, "hi": float(len(options)),
                                      "span": float(span) if span > 0 else None,
                                      "basis": f"{qtype.lower()} with {len(options)} ordered options"}
                elif rng.get("Max") is not None and rng.get("Min") is not None:
                    span = float(rng["Max"]) - float(rng["Min"])
                    found[item_id] = {"lo": float(rng["Min"]), "hi": float(rng["Max"]),
                                      "span": span if span > 0 else None,
                                      "basis": f"slider Range {rng['Min']}-{rng['Max']}"}
                elif qid in _TEXT_STATED_BOUNDS:
                    lo, hi = _TEXT_STATED_BOUNDS[qid]
                    found[item_id] = {"lo": lo, "hi": hi, "span": hi - lo,
                                      "basis": "bound stated in the question text"}
                else:
                    found[item_id] = {"lo": None, "hi": None, "span": None,
                                      "basis": "free numeric entry — no declared bound"}
    return found


def write_manifest() -> dict:
    ranges = declared_ranges()
    scored = {k: v for k, v in ranges.items() if v["span"]}
    unscored = sorted(k for k, v in ranges.items() if not v["span"])
    manifest = {
        "n_anchor_items": len(ranges),
        "n_with_declared_range": len(scored),
        "n_without": len(unscored),
        "excluded_from_normalised_aggregate": unscored,
        "rule": "errors are divided by the item's declared span and expressed in "
                "percentage points; items with no declared span are excluded from "
                "the aggregate rather than assigned an invented scale",
        "ranges": ranges,
    }
    CALIB_DATA.mkdir(parents=True, exist_ok=True)
    save_json(manifest, CALIB_DATA / "anchor_declared_ranges.json")
    return manifest
