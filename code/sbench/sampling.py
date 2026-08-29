"""Section 10 — nested stratified samples from the frozen pre-study answers.

Strata come from the twins' FROZEN SURVEY ANSWERS, never from the Twin-2K
profile bands: age band from the reported birth year (the survey's own
cut-offs), gender and race from the survey answers.

Targets are the benchmark's released quota tables (recorded source in
``common.QUOTA_SOURCE``) — never the example prediction files.

Design: core intervention sample of exactly 500; control sample of exactly
1,000 that contains the core 500 plus 500 more.

Selection is EXACT, not greedy: because each twin contributes to exactly one
age x gender cell and one race x gender cell, choosing per-type counts is a
minimum-cost-flow problem (source -> age cell -> type arc -> race cell ->
sink, with convex |achieved - target| costs on the cell arcs), solved to
optimality by successive shortest-path augmentation — no external solver.
The nested control keeps the core counts as a fixed base and augments the
remaining 500 optimally.  A fixed seed picks concrete twins within each
demographic type.  No outcome calibration, no post-simulation weighting.
"""

from __future__ import annotations

from .common import (
    AGE_GENDER_TARGETS,
    ARTIFACTS,
    DATA,
    DEFAULT_SEED,
    QUOTA_SOURCE,
    RACE_GENDER_TARGETS,
    load_json,
    read_csv,
    rng_for,
    save_json,
    write_csv,
)

GENDERS = ("Male", "Female")


def _integer_targets(table: dict, n: int) -> dict[tuple[str, str], int]:
    """Largest-remainder rounding of share x gender-split to exactly n."""
    raw = {}
    for level, (share, male, female) in table.items():
        raw[(level, "Male")] = n * share * male
        raw[(level, "Female")] = n * share * female
    floor = {k: int(v) for k, v in raw.items()}
    remainder = n - sum(floor.values())
    order = sorted(raw, key=lambda k: raw[k] - floor[k], reverse=True)
    for key in order[:remainder]:
        floor[key] += 1
    return floor


def _strata(row: dict) -> tuple[str, str, str] | None:
    age = (row.get("age_band") or "").strip()
    gender = (row.get("gender") or "").strip()
    race = (row.get("race") or "").strip()
    if age not in AGE_GENDER_TARGETS:
        return None
    if gender not in GENDERS:      # "Other" gender: keep eligible, matched on race only
        gender = ""
    if race not in RACE_GENDER_TARGETS:
        return None
    return age, gender, race


def _min_cost_counts(
    supply: dict[tuple, int], n: int, age_targets: dict, race_targets: dict,
    base: dict[tuple, int] | None = None,
) -> dict[tuple, int]:
    """Exact per-type counts minimizing total |deviation| across both tables.

    Types are (age_cell | None, race_cell | None); a None cell contributes no
    cost (gender outside Male/Female).  Successive shortest paths on the
    residual graph with Bellman-Ford handles the convex cell costs exactly.
    """
    age_cells = sorted({t[0] for t in supply if t[0]})
    race_cells = sorted({t[1] for t in supply if t[1]})
    S, T = 0, 1
    age_id = {c: 2 + i for i, c in enumerate(age_cells)}
    race_id = {c: 2 + len(age_cells) + i for i, c in enumerate(race_cells)}
    n_nodes = 2 + len(age_cells) + len(race_cells)

    arcs: list[dict] = []   # {u, v, cap, flow, kind, cell/type}

    def add(u, v, cap, kind, key=None):
        arcs.append({"u": u, "v": v, "cap": cap, "flow": 0, "floor": 0,
                     "kind": kind, "key": key})

    for cell in age_cells:
        add(S, age_id[cell], 10**9, "age", cell)
    for cell in race_cells:
        add(race_id[cell], T, 10**9, "race", cell)
    for t, cap in sorted(supply.items(), key=lambda kv: str(kv[0])):
        u = age_id[t[0]] if t[0] else S
        v = race_id[t[1]] if t[1] else T
        add(u, v, cap, "type", t)

    def marginal(arc, forward: bool) -> int:
        if arc["kind"] == "age":
            target = age_targets.get(arc["key"], 0)
        elif arc["kind"] == "race":
            target = race_targets.get(arc["key"], 0)
        else:
            return 0
        if forward:                       # cost of unit flow+1
            return -1 if arc["flow"] < target else 1
        return 1 if arc["flow"] <= target else -1   # undo unit flow

    # Pre-load the fixed base (nested lower bounds): the flow may never drop
    # below a type's core count, so the reverse residual is floored.
    if base:
        for arc in arcs:
            if arc["kind"] == "type" and arc["key"] in base:
                preload = base[arc["key"]]
                arc["flow"] = preload
                arc["floor"] = preload
        for arc in arcs:
            if arc["kind"] == "age":
                arc["flow"] = sum(
                    a["flow"] for a in arcs
                    if a["kind"] == "type" and a["key"][0] == arc["key"]
                )
            if arc["kind"] == "race":
                arc["flow"] = sum(
                    a["flow"] for a in arcs
                    if a["kind"] == "type" and a["key"][1] == arc["key"]
                )
    preloaded = sum(a["flow"] for a in arcs if a["kind"] == "type")

    for _ in range(n - preloaded):
        # Bellman-Ford on marginal costs.
        INF = 10**9
        dist = [INF] * n_nodes
        parent: list[tuple | None] = [None] * n_nodes
        dist[S] = 0
        for _round in range(n_nodes + 2):
            changed = False
            for ai, arc in enumerate(arcs):
                if arc["flow"] < arc["cap"] and dist[arc["u"]] < INF:
                    cost = dist[arc["u"]] + marginal(arc, True)
                    if cost < dist[arc["v"]]:
                        dist[arc["v"]] = cost
                        parent[arc["v"]] = (ai, True)
                        changed = True
                if arc["flow"] > arc["floor"] and dist[arc["v"]] < INF:
                    cost = dist[arc["v"]] + marginal(arc, False)
                    if cost < dist[arc["u"]]:
                        dist[arc["u"]] = cost
                        parent[arc["u"]] = (ai, False)
                        changed = True
            if not changed:
                break
        if dist[T] >= INF:
            raise SystemExit("sampling: supply exhausted before reaching the target n")
        node = T
        while node != S:
            ai, forward = parent[node]
            arc = arcs[ai]
            if forward:
                arc["flow"] += 1
                node = arc["u"]
            else:
                arc["flow"] -= 1
                node = arc["v"]

    return {arc["key"]: arc["flow"] for arc in arcs if arc["kind"] == "type"}


def _type_of(row: dict) -> tuple:
    age, gender, race = row["_strata"]
    age_cell = (age, gender) if gender else None
    race_cell = (race, gender) if gender else None
    return (age_cell, race_cell)


def _select(candidates: list[dict], n: int, fixed: list[dict], seed: int) -> list[dict]:
    """Exact selection to n via min-cost flow over demographic types."""
    age_targets = _integer_targets(AGE_GENDER_TARGETS, n)
    race_targets = _integer_targets(RACE_GENDER_TARGETS, n)

    by_type: dict[tuple, list[dict]] = {}
    for row in candidates:
        by_type.setdefault(_type_of(row), []).append(row)
    supply = {t: len(rows) for t, rows in by_type.items()}

    base_counts: dict[tuple, int] = {}
    fixed_ids = {row["persona_id"] for row in fixed}
    for row in fixed:
        base_counts[_type_of(row)] = base_counts.get(_type_of(row), 0) + 1

    counts = _min_cost_counts(supply, n, age_targets, race_targets,
                              base=base_counts or None)

    chosen = list(fixed)
    for t, count in sorted(counts.items(), key=lambda kv: str(kv[0])):
        need = count - base_counts.get(t, 0)
        if need <= 0:
            continue
        pool = [row for row in by_type.get(t, []) if row["persona_id"] not in fixed_ids]
        pool.sort(key=lambda row: int(row["persona_id"].split("_")[1]))
        rng = rng_for(seed, "pick", str(t))
        rng.shuffle(pool)
        chosen.extend(pool[:need])
    if len(chosen) != n:
        raise SystemExit(f"sampling: selected {len(chosen)} of target {n}")
    return chosen


def _achieved(rows: list[dict], n: int):
    age_targets = _integer_targets(AGE_GENDER_TARGETS, n)
    race_targets = _integer_targets(RACE_GENDER_TARGETS, n)
    age_count: dict = {}
    race_count: dict = {}
    for row in rows:
        age, gender, race = row["_strata"]
        if gender:
            age_count[(age, gender)] = age_count.get((age, gender), 0) + 1
            race_count[(race, gender)] = race_count.get((race, gender), 0) + 1
    out = []
    deviation = 0
    for table, targets, counts in (
        ("age_x_gender", age_targets, age_count),
        ("race_x_gender", race_targets, race_count),
    ):
        for (level, gender), target in sorted(targets.items()):
            got = counts.get((level, gender), 0)
            deviation += abs(got - target)
            out.append({"sample_n": n, "table": table, "level": level, "gender": gender,
                        "target": target, "achieved": got, "deviation": got - target})
    return out, deviation


def run_sampling(*, seed: int = DEFAULT_SEED) -> int:
    from .common import QID as qids

    wide = read_csv(DATA / "prestudy_frozen_wide.csv")
    eligibility = {r["source_twin_id"]: r for r in read_csv(DATA / "eligibility_audit.csv")}

    candidates = []
    for row in wide:
        twin = row["persona_id"]
        if eligibility.get(twin, {}).get("eligible") != "yes":
            continue
        entry = {
            "persona_id": twin,
            "age_band": row.get("age_band", ""),
            "gender": row.get("gender", ""),
            "race": row.get("race", ""),
        }
        strata = _strata(entry)
        if strata is None:
            continue
        entry["_strata"] = strata
        candidates.append(entry)

    core = _select(candidates, 500, [], seed)
    control = _select(candidates, 1000, core, seed)
    core_ids = {row["persona_id"] for row in core}
    control_ids = {row["persona_id"] for row in control}
    assert core_ids <= control_ids and len(core_ids) == 500 and len(control_ids) == 1000

    assignment = []
    for row in sorted(candidates, key=lambda r: int(r["persona_id"].split("_")[1])):
        twin = row["persona_id"]
        assignment.append(
            {"source_twin_id": twin,
             "in_core_500": "yes" if twin in core_ids else "no",
             "in_control_1000": "yes" if twin in control_ids else "no",
             "age_band": row["age_band"], "gender": row["gender"], "race": row["race"]}
        )
    write_csv(assignment, DATA / "sample_assignment.csv",
              ["source_twin_id", "in_core_500", "in_control_1000", "age_band", "gender", "race"])

    target_rows = []
    for n in (500, 1000):
        for table, targets in (("age_x_gender", _integer_targets(AGE_GENDER_TARGETS, n)),
                               ("race_x_gender", _integer_targets(RACE_GENDER_TARGETS, n))):
            for (level, gender), value in sorted(targets.items()):
                target_rows.append({"sample_n": n, "table": table, "level": level,
                                    "gender": gender, "target": value, "source": QUOTA_SOURCE})
    write_csv(target_rows, DATA / "quota_targets.csv",
              ["sample_n", "table", "level", "gender", "target", "source"])

    achieved_rows, dev500 = _achieved(core, 500)
    more_rows, dev1000 = _achieved(control, 1000)
    write_csv(achieved_rows + more_rows, DATA / "quota_achieved.csv",
              ["sample_n", "table", "level", "gender", "target", "achieved", "deviation"])

    report = [
        "# Sampling report",
        "",
        f"- Quota target source: {QUOTA_SOURCE}",
        f"- Eligible candidates: {len(candidates)}",
        f"- Core intervention sample: 500 (total |deviation| across both cross-tabs: {dev500})",
        f"- Control sample: 1,000 nested over the core 500 (total |deviation|: {dev1000})",
        f"- Selection: greedy joint deficit fill, seed {seed}; strata from frozen survey answers.",
        "- No outcome calibration or weighting applied at this phase.",
    ]
    (ARTIFACTS / "sampling_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"sampling: 500 core + 1,000 control selected from {len(candidates)} eligible; "
          f"|deviation| core={dev500}, control={dev1000}")
    return 0
