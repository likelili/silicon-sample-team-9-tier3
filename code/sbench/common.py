"""Shared plumbing for the Silicon Sample Benchmark pipeline.

Everything here is deliberately boring: fixed paths, hashing, deterministic
randomness, sys.path wiring into the SurveyTwin integration worktree, and the
one model-registry shim the worktree needs.  No module in this package may make
a paid API call unless the phase was invoked with ``--live``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[2]          # .../silicon_bench
WORKTREE = BENCH_ROOT / "surveytwin"                       # integration worktree
BENCH_REPO = BENCH_ROOT / "silicon-sample-submission"      # frozen benchmark inputs
# --- run isolation ---------------------------------------------------------
# ARTIFACTS holds build products shared by every run (QSF audit, templates,
# condition map) — they are pure functions of the frozen inputs.  DATA and
# AUDIT are PER-RUN and resolve through an active run directory so a later
# mocked preflight can never overwrite live results.  ``set_run_dir`` is the
# only way to point them somewhere; ``RUNS`` is the parent of all run dirs.
ARTIFACTS = BENCH_ROOT / "artifacts"
RUNS = BENCH_ROOT / "runs"
MANIFEST = BENCH_ROOT / "manifest" / "source_manifest.json"

_ACTIVE_RUN: Path | None = None


def set_run_dir(run_dir: "str | Path | None") -> Path:
    """Point the per-run outputs at ``run_dir`` (created if needed)."""
    global _ACTIVE_RUN
    if run_dir is None:
        _ACTIVE_RUN = None
        return BENCH_ROOT
    path = Path(run_dir).expanduser().resolve()
    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "audit").mkdir(parents=True, exist_ok=True)
    _ACTIVE_RUN = path
    return path


def new_run_dir(label: str, stamp: str | None = None) -> Path:
    """Create and activate a fresh timestamped run directory.

    ``stamp`` must be supplied by the caller (the CLI passes the wall clock);
    it is recorded so a run directory is self-describing.
    """
    stamp = stamp or "unstamped"
    return set_run_dir(RUNS / f"{stamp}_{label}")


def active_run_dir() -> Path:
    if _ACTIVE_RUN is None:
        raise SystemExit(
            "no run directory is active — pass --run-dir <path> (or --new-run) so "
            "outputs cannot overwrite another run's results"
        )
    return _ACTIVE_RUN


class _RunPath:
    """Lazily resolves to ``<active run>/<name>`` at attribute-access time."""

    def __init__(self, name: str) -> None:
        self._name = name

    def _resolve(self) -> Path:
        return active_run_dir() / self._name

    def __truediv__(self, other):
        return self._resolve() / other

    def __fspath__(self) -> str:
        return str(self._resolve())

    def __str__(self) -> str:
        return str(self._resolve())

    def mkdir(self, **kwargs):
        return self._resolve().mkdir(**kwargs)

    def exists(self) -> bool:
        return self._resolve().exists()

    def glob(self, pattern):
        return self._resolve().glob(pattern)


DATA = _RunPath("data")
AUDIT = _RunPath("audit")

AUTHORITATIVE_QSF = BENCH_REPO / "survey" / "survey.qsf"
CONDITION_CODENAMES = BENCH_REPO / "survey" / "condition_codenames.csv"
CODEBOOK = BENCH_REPO / "codebook.csv"

# --- fixed structural facts of the authoritative survey ---------------------
# Verified by the QSF audit; any drift fails the audit, not silently.
REMOVED_ELEMENTS = {
    "blocks": {
        "BL_b4nxolDPBjQHMTY": "Consent Form",
        "BL_0My4IaAYJZG5ipE": "Filter (agree not to use AI — inapplicable to twins)",
    },
    "flows": {
        "FL_6": "screenout branch on QID1721185780 (consent to pay attention)",
        "FL_7": "screenout branch on QID1721185781 (consent not to use AI)",
    },
}
BOUNDARY_BLOCK_ID = "BL_86roDEL7lnFM5n0"   # "Transition to Study" (FL_15)
TREATMENT_RANDOMIZER_FLOW_ID = "FL_18"
ATTENTION_QIDS = {"QID1721185793": "attention1", "QID1721185922": "attention2"}

# Pre-study QIDs used by the conflict rules and stratification.
QID = {
    "gender": "QID1721185783",
    "gender_text": "QID1721185783_3_TEXT",
    "year_birth": "QID1721185784",
    "race": "QID1721185785",
    "race_text": "QID1721185785_5_TEXT",
    "education": "QID1721185786",
    "income": "QID1721185788",
    "household": "QID1721185789",
    "party": "QID1721185795",
    "party_text": "QID1721185795_4_TEXT",
    "religion": "QID1721185824",
    "religion_text": "QID1721185824_22_TEXT",
    "state": "QID1721185837",
}
SURVEY_YEAR = 2026  # from the QSF's own age-band cut-offs (>=1997 -> 18-29)

# --- quota targets ----------------------------------------------------------
# Source: benchmark preregistration quota tables (2024 Census Bureau Population
# Estimates cross quotas), mirrored in the human study preregistration
# (Zenodo record 20160212). NOT derived from any example prediction file.
QUOTA_SOURCE = (
    "Benchmark preregistration quota tables "
    "(janpfander.github.io/llm_predictions_megastudy/preregistration_benchmark.html); "
    "human study preregistration Zenodo 20160212; 2024 Census PEP cross quotas."
)
AGE_GENDER_TARGETS = {  # band -> (share, male_share_within, female_share_within)
    "18-29": (0.202, 0.509, 0.491),
    "30-44": (0.260, 0.505, 0.495),
    "45-59": (0.229, 0.497, 0.503),
    "60+": (0.309, 0.461, 0.539),
}
RACE_GENDER_TARGETS = {  # survey race label -> (share, male, female)
    "White / Caucasian": (0.602, 0.492, 0.508),
    "Latino / Hispanic": (0.181, 0.505, 0.495),
    "Black / African-American": (0.123, 0.471, 0.529),
    "Asian / Asian-American": (0.067, 0.473, 0.527),
    "Other": (0.027, 0.488, 0.512),
}

DEFAULT_MODEL = "gpt-5.6-luna-low"
DEFAULT_PERSONA_TYPE = "full"   # spec default; changing it requires team approval
DEFAULT_SEED = 42


def wire_worktree() -> None:
    """Put the integration worktree's backend + QSF parser on sys.path."""
    logging.disable(logging.INFO)
    for p in (WORKTREE / "web" / "backend", WORKTREE / "processing_qualtrics_qsf"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def register_models() -> None:
    """The worktree's MODEL_SPECS predates the gpt-5.6 family; add the entries
    exactly as the deployed platform defines them (alias -> API model + effort)."""
    wire_worktree()
    from services.v2.llm_client import MODEL_SPECS  # noqa: E402

    for family in ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"):
        for suffix, effort in (("-low", "low"), ("-medium", "medium"), ("-high", "high"), ("", None)):
            MODEL_SPECS.setdefault(
                f"{family}{suffix}",
                {
                    "model_name": family,
                    "openrouter_model_name": f"openai/{family}",
                    "default_reasoning_effort": effort,
                },
            )


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path | str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=1, ensure_ascii=False)


def append_jsonl(obj, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict], path: Path | str, columns: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path | str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rng_for(*parts) -> random.Random:
    seed = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(seed[:16], 16))


def session_seed(global_seed: int, base_pid: str, raw_condition: str) -> int:
    """A randomization seed unique to one twin x condition session.

    QsfRuntime funnels every randomizer through ``self.seed`` — flow randomizer
    ordering, even presentation, block question order and choice order all
    interpolate it — so handing each session its own seed makes every authored
    randomization independent across a twin's 17 conditions. Without this a twin
    draws one block order and reuses it in every arm, which would tie each
    intervention to its control by a shared ordering.

    ``raw_condition`` rather than ``condition`` is deliberate: the three pooled
    control texts are distinct raw labels, so a control session randomizes off
    the text it actually received.

    SHA-256, not ``hash()`` — the built-in is salted per process and would break
    reproducibility across runs. Truncated to 63 bits so the value stays a
    positive signed 64-bit integer.
    """
    digest = hashlib.sha256(
        f"{int(global_seed)}|{base_pid}|{raw_condition}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) >> 1


def manifest() -> dict:
    return load_json(MANIFEST)


def condition_codes() -> dict:
    """Read the authoritative condition list. Codes with semicolons stay whole."""
    rows = read_csv(CONDITION_CODENAMES)
    return rows


def require_live_flag(live: bool, phase: str) -> None:
    if not live:
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(f"{phase}: --live requires OPENAI_API_KEY in the environment.")
    if os.getenv("SBENCH_APPROVED") != "1":
        raise SystemExit(
            f"{phase}: --live additionally requires SBENCH_APPROVED to be EXACTLY '1' "
            "— set it only after the cost estimate has been explicitly approved."
        )
