"""One repair pass over round-1 sessions that came back incomplete.

A session is "incomplete" when the number of distinct ``question_id`` values it
returned differs from the number asked — short (the model skipped questions) or
over (it invented ids). Each such session is re-sent **once**, with the
byte-identical original request body: same prompt, same arm, same parameters,
so a repair can never alter the experimental condition. Only the sampling
nondeterminism differs.

Practical-planarian round-1 sessions are deliberately excluded. Round 2 is built
from round 1's answers and embeds them as prior context, so replacing a round-1
answer after round 2 has been generated would leave the two rounds describing
different histories for the same twin. Those sessions keep their flagged skip.

Adoption is decided after the fact by ``merge_repairs``: a repair replaces the
original only if it is strictly more complete.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .common import AUDIT, DATA, read_csv, wire_worktree, write_csv

PLANARIAN = "practical planarian"


def _best_responses(results_dir: Path) -> dict[str, dict]:
    """custom_id -> response, with any repair/retry file taking precedence."""
    best: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        late = ("retries" in path.name) or ("repair" in path.name)
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("custom_id")
            if not cid or (cid in best and not late):
                continue
            best[cid] = obj
    return best


def _delivered(obj: dict) -> tuple[str, int]:
    """(state, distinct question ids delivered) for one batch response line."""
    response = obj.get("response") or {}
    if response.get("status_code") != 200:
        return ("http_error", 0)
    choice = ((response.get("body") or {}).get("choices") or [{}])[0]
    if choice.get("finish_reason") != "stop":
        return ("truncated", 0)
    try:
        parsed = json.loads(choice["message"]["content"])
        answers = parsed.get("answers") if isinstance(parsed, dict) else parsed
        return ("ok", len({str(a.get("question_id")) for a in answers if isinstance(a, dict)}))
    except Exception:
        return ("unparseable", 0)


def find_incomplete(*, round_no: int = 1) -> list[dict]:
    plan = {r["run_id"]: r for r in read_csv(AUDIT / "batch" / "render_plan.csv")}
    best = _best_responses(DATA / "batch" / f"round{round_no}_results")
    rows: list[dict] = []
    for cid, obj in best.items():
        if cid not in plan:
            continue
        want = int(plan[cid]["n_questions"])
        state, got = _delivered(obj)
        if state == "ok" and got == want:
            continue
        rows.append({
            "run_id": cid,
            "base_pid": plan[cid]["base_pid"],
            "raw_condition": plan[cid]["raw_condition"],
            "asked": want,
            "delivered": got,
            "state": state,
            "excluded_from_repair": "planarian round 2 already built from these answers"
                                    if plan[cid]["raw_condition"] == PLANARIAN else "",
        })
    rows.sort(key=lambda r: (r["raw_condition"], r["run_id"]))
    return rows


async def build_repair_batch(*, round_no: int = 1) -> int:
    """Write a repair chunk holding the original bodies of incomplete sessions."""
    wire_worktree()
    rows = find_incomplete(round_no=round_no)
    write_csv(rows, AUDIT / "batch" / "repair_candidates.csv", list(rows[0].keys()) if rows else
              ["run_id", "base_pid", "raw_condition", "asked", "delivered", "state",
               "excluded_from_repair"])
    targets = {r["run_id"] for r in rows if not r["excluded_from_repair"]}
    excluded = [r for r in rows if r["excluded_from_repair"]]
    print(f"repair: {len(rows)} incomplete session(s); "
          f"{len(targets)} to re-run, {len(excluded)} excluded")
    for row in excluded:
        print(f"  excluded {row['run_id']} ({row['base_pid']}, {row['raw_condition']}): "
              f"{row['delivered']}/{row['asked']} — {row['excluded_from_repair']}")
    if not targets:
        print("repair: nothing to do")
        return 0

    out_dir = DATA / "batch" / "repair1"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(out_dir) / "requests_repair_000.jsonl"

    # Copy the ORIGINAL request lines verbatim out of the round-1 chunks.
    written = 0
    seen: set[str] = set()
    with open(dest, "w", encoding="utf-8") as out:
        for chunk in sorted((DATA / "batch" / f"round{round_no}").glob("requests_*.jsonl")):
            with open(chunk, encoding="utf-8") as handle:
                for line in handle:
                    # cheap prefilter before parsing a 70 KB line
                    if not any(t in line[:120] for t in targets):
                        continue
                    obj = json.loads(line)
                    cid = obj.get("custom_id")
                    if cid in targets and cid not in seen:
                        seen.add(cid)
                        out.write(line if line.endswith("\n") else line + "\n")
                        written += 1
    missing = targets - seen
    if missing:
        raise SystemExit(f"repair: could not recover {len(missing)} original body(ies): "
                         f"{sorted(missing)[:5]}")

    manifest = {
        "round": "repair1",
        "source_round": round_no,
        "requests": written,
        "policy": "one re-run per incomplete session, byte-identical original body",
        "excluded": [{k: r[k] for k in ("run_id", "base_pid", "raw_condition",
                                        "excluded_from_repair")} for r in excluded],
        "chunks": [{"file": dest.name, "requests": written, "bytes": dest.stat().st_size}],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"repair: wrote {written:,} request(s) -> {dest} "
          f"({dest.stat().st_size/1e6:.1f} MB)")
    return 0


def merge_repairs(*, round_no: int = 1) -> int:
    """Report what the repair pass recovered; adoption is 'more complete wins'."""
    plan = {r["run_id"]: r for r in read_csv(AUDIT / "batch" / "render_plan.csv")}
    results = DATA / "batch" / f"round{round_no}_results"
    originals: dict[str, dict] = {}
    repairs: dict[str, dict] = {}
    for path in sorted(Path(results).glob("*.jsonl")):
        bucket = repairs if "repair" in path.name else originals
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("custom_id"):
                bucket[obj["custom_id"]] = obj

    rows = []
    improved = same = worse = 0
    for cid, rep in repairs.items():
        want = int(plan[cid]["n_questions"])
        o_state, o_got = _delivered(originals.get(cid, {}))
        r_state, r_got = _delivered(rep)
        adopt = r_got > o_got
        if r_got > o_got:
            improved += 1
        elif r_got == o_got:
            same += 1
        else:
            worse += 1
        rows.append({"run_id": cid, "base_pid": plan[cid]["base_pid"],
                     "raw_condition": plan[cid]["raw_condition"], "asked": want,
                     "original": o_got, "repaired": r_got,
                     "adopted": "repair" if adopt else "original"})
    if rows:
        rows.sort(key=lambda r: r["run_id"])
        write_csv(rows, AUDIT / "batch" / "repair_outcomes.csv", list(rows[0].keys()))
    exact = sum(1 for r in rows if r["repaired"] == r["asked"])
    print(f"repair outcomes over {len(rows):,} re-run session(s):")
    print(f"  now exactly complete : {exact:,}")
    print(f"  improved             : {improved:,}")
    print(f"  unchanged            : {same:,}")
    print(f"  worse (keep original): {worse:,}")
    return 0
