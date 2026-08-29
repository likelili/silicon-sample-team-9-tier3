"""Tier-1 stratified samples from the 40-cell joint quota table (panel design).

Supersedes the marginal-table sampler in ``sampling.py``: targets now come from
the single raked 2x4x5 gender x age x race joint table
(``artifacts/quota_joint_gender_age_race.csv``), whose own margins ARE the
benchmark's two published two-way tables. Every margin used here is derived
from that one file — nothing is mixed in from a second source.

Design (revised 2026-08-28) — a WITHIN-SUBJECTS panel:

* **Universal clean pool.** A twin qualifies only if every one of its 17
  sessions is exactly complete. This is stricter than excluding per condition,
  and it is what makes one panel usable everywhere: a twin admitted to the
  panel has usable data in all 16 interventions and control.
* **One frozen panel of 500** answers all 16 interventions. Every intervention
  sample is the same 500 people, so an intervention-vs-control contrast is
  within-subject and free of between-sample composition noise.
* **Control = that panel plus 500 further twins** (1,000 total), disjoint from
  each other, drawn from the same universal pool. The panel is nested inside
  control by construction, which the benchmark's design calls for.
* Targets: 500 for the panel, 1,000 for control. Quotas by constrained integer
  optimization, not greedy redistribution:

      minimize   W * [sum over age x gender cells (achieved - target)^2
                      + sum over race x gender cells (achieved - target)^2]
                 + sum over joint cells (n_c - N p_c)^2 / (N p_c + eps)
      subject to sum n_c = N,  panel_c <= n_c <= available_c

  Gender totals are exact by construction (each gender is solved as its own
  4x5 transportation problem with exactly its integer total). W is large
  enough that the optimizer is lexicographic: first as close as possible to
  the published two-way margins (exact whenever supply permits), then as
  close as possible to the 40-cell joint target. The quadratic margin
  penalty is what spreads an unavoidable shortfall across several nearby
  cells instead of dumping it into one. The ``panel_c <=`` lower bound is the
  nesting constraint, enforced exactly rather than by post-hoc repair.
* Solved exactly by successive shortest-path augmentation on a min-cost flow
  with convex per-unit costs (source -> age cell -> joint cell arc -> race
  cell -> sink), lower bounds preloaded and the reverse residual floored.
  ``selftest`` verifies optimality against brute-force enumeration on random
  small instances, half of them carrying lower bounds.
* Twins are drawn within each cell by seeded shuffle; quotas are
  seed-independent, the people chosen are not.

The audit keeps the three quantities the design calls for, per cell: the
original target (real and integer), the available supply, and the final
adjusted quota.
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import (
    ARTIFACTS,
    AUDIT,
    DATA,
    DEFAULT_SEED,
    read_csv,
    rng_for,
    save_json,
    sha256_file,
    write_csv,
)

PLANARIAN = "practical planarian"
GENDERS = ("Male", "Female")
AGES = ("18-29", "30-44", "45-59", "60+")
# quota-table race label -> frozen survey race label
RACE_MAP = {
    "White (non-Hispanic)": "White / Caucasian",
    "Black / African American": "Black / African-American",
    "Hispanic / Latino": "Latino / Hispanic",
    "Asian / Asian American": "Asian / Asian-American",
    "Other": "Other",
}
RACES = tuple(RACE_MAP.values())
EPS = 1e-9
W_MARGIN = 1e9   # lexicographic weight; margin-cost integers x W stay exact in floats


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

def load_joint_cells() -> dict[tuple[str, str, str], float]:
    """(gender, age, survey-race) -> proportion, renormalized to exactly 1."""
    rows = read_csv(ARTIFACTS / "quota_joint_gender_age_race.csv")
    cells: dict[tuple[str, str, str], float] = {}
    for row in rows:
        race = RACE_MAP.get(row["race"].strip())
        gender, age = row["gender"].strip(), row["age_band"].strip()
        if race is None or gender not in GENDERS or age not in AGES:
            raise SystemExit(f"tier1: unmapped quota row {row}")
        cells[(gender, age, race)] = float(row["proportion"])
    if len(cells) != 40:
        raise SystemExit(f"tier1: expected 40 cells, got {len(cells)}")
    total = sum(cells.values())
    if abs(total - 1.0) > 1e-3:
        raise SystemExit(f"tier1: proportions sum to {total}")
    return {k: v / total for k, v in cells.items()}


def _largest_remainder(shares: dict, n: int) -> dict:
    """Integer allocation of n by largest remainder; deterministic tie-break."""
    raw = {k: n * v for k, v in shares.items()}
    floor = {k: int(v) for k, v in raw.items()}
    leftover = n - sum(floor.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - floor[k]), str(k)))
    for key in order[:leftover]:
        floor[key] += 1
    return floor


def integer_targets(cells: dict, n: int):
    """Nested rounding: gender totals to n, then each margin to its gender total.

    Guarantees sum(age margins) == sum(race margins) == gender total for each
    gender, so the per-gender transportation problems are internally
    consistent.  Also returns the integer 40-cell reference quota (the
    'original target' column of the audit) and the real-valued ideals.
    """
    gender_share = {g: sum(p for (gg, _, _), p in cells.items() if gg == g)
                    for g in GENDERS}
    gender_total = _largest_remainder(gender_share, n)
    age_targets, race_targets, cell_int = {}, {}, {}
    for g in GENDERS:
        share = gender_share[g]
        age_targets[g] = _largest_remainder(
            {a: sum(p for (gg, aa, _), p in cells.items() if gg == g and aa == a) / share
             for a in AGES}, gender_total[g])
        race_targets[g] = _largest_remainder(
            {r: sum(p for (gg, _, rr), p in cells.items() if gg == g and rr == r) / share
             for r in RACES}, gender_total[g])
        cell_int.update(_largest_remainder(
            {(g, a, r): cells[(g, a, r)] / share for a in AGES for r in RACES},
            gender_total[g]))
    ideal = {k: n * p for k, p in cells.items()}
    return gender_total, age_targets, race_targets, cell_int, ideal


# --------------------------------------------------------------------------
# Optimizer: per-gender 4x5 transportation with convex costs
# --------------------------------------------------------------------------

def solve_gender(
    n_units: int,
    age_targets: dict[str, int],
    race_targets: dict[str, int],
    ideal: dict[tuple[str, str], float],   # (age, race) -> N*p
    caps: dict[tuple[str, str], int],
    floors: dict[tuple[str, str], int] | None = None,
) -> dict[tuple[str, str], int]:
    """Exact integer minimizer via successive shortest paths.

    Convex separable costs make every augmentation's greedy shortest path
    globally optimal; Bellman-Ford tolerates the negative below-target
    marginals.

    ``floors`` are per-cell lower bounds: the nested control sample must
    CONTAIN the frozen panel, so each cell's count may never fall below the
    panel's. Preloading the flow and blocking the reverse residual below it
    keeps the optimum exact under that constraint rather than approximating it.
    """
    if sum(caps.values()) < n_units:
        raise SystemExit(f"tier1: supply {sum(caps.values())} < required {n_units}")
    floors = floors or {}
    if any(floors.get(k, 0) > caps.get(k, 0) for k in caps):
        bad = [k for k in caps if floors.get(k, 0) > caps.get(k, 0)]
        raise SystemExit(f"tier1: floor exceeds supply in {bad[:3]}")
    if sum(floors.values()) > n_units:
        raise SystemExit(f"tier1: floors sum to {sum(floors.values())} > {n_units}")
    ages = [a for a in AGES]
    races = [r for r in RACES]
    S, T = 0, 1
    age_id = {a: 2 + i for i, a in enumerate(ages)}
    race_id = {r: 2 + len(ages) + i for i, r in enumerate(races)}
    n_nodes = 2 + len(ages) + len(races)

    arcs: list[dict] = []

    def add(u, v, cap, kind, key):
        arcs.append({"u": u, "v": v, "cap": cap, "flow": 0, "floor": 0,
                     "kind": kind, "key": key})

    for a in ages:
        add(S, age_id[a], 10 ** 9, "age", a)
    for r in races:
        add(race_id[r], T, 10 ** 9, "race", r)
    for a in ages:
        for r in races:
            cap = caps.get((a, r), 0)
            if cap > 0:
                add(age_id[a], race_id[r], cap, "cell", (a, r))

    def marginal(arc, forward: bool) -> float:
        f = arc["flow"]
        if arc["kind"] == "age":
            t, w = float(age_targets[arc["key"]]), W_MARGIN
        elif arc["kind"] == "race":
            t, w = float(race_targets[arc["key"]]), W_MARGIN
        else:
            t = ideal[arc["key"]]
            w = 1.0 / (t + EPS)
        # quadratic (f - t)^2: cost delta of one more / one fewer unit
        return w * ((2 * f + 1 - 2 * t) if forward else (-2 * f + 1 + 2 * t))

    # Preload the floors so the search starts from a feasible nested solution.
    for arc in arcs:
        if arc["kind"] == "cell" and floors.get(arc["key"], 0):
            arc["flow"] = arc["floor"] = floors[arc["key"]]
    for arc in arcs:
        if arc["kind"] == "age":
            arc["flow"] = sum(a["flow"] for a in arcs
                              if a["kind"] == "cell" and a["key"][0] == arc["key"])
        elif arc["kind"] == "race":
            arc["flow"] = sum(a["flow"] for a in arcs
                              if a["kind"] == "cell" and a["key"][1] == arc["key"])
    preloaded = sum(a["flow"] for a in arcs if a["kind"] == "cell")

    INF = float("inf")
    for _ in range(n_units - preloaded):
        dist = [INF] * n_nodes
        parent: list[tuple[int, bool] | None] = [None] * n_nodes
        dist[S] = 0.0
        for _round in range(n_nodes + 2):
            changed = False
            for ai, arc in enumerate(arcs):
                if arc["flow"] < arc["cap"] and dist[arc["u"]] < INF:
                    cost = dist[arc["u"]] + marginal(arc, True)
                    if cost < dist[arc["v"]] - 1e-12:
                        dist[arc["v"]], parent[arc["v"]] = cost, (ai, True)
                        changed = True
                if arc["flow"] > arc["floor"] and dist[arc["v"]] < INF:
                    cost = dist[arc["v"]] + marginal(arc, False)
                    if cost < dist[arc["u"]] - 1e-12:
                        dist[arc["u"]], parent[arc["u"]] = cost, (ai, False)
                        changed = True
            if not changed:
                break
        if dist[T] == INF:
            raise SystemExit("tier1: no augmenting path — supply exhausted")
        node = T
        while node != S:
            ai, forward = parent[node]
            if forward:
                arcs[ai]["flow"] += 1
                node = arcs[ai]["u"]
            else:
                arcs[ai]["flow"] -= 1
                node = arcs[ai]["v"]

    counts = {arc["key"]: arc["flow"] for arc in arcs if arc["kind"] == "cell"}
    for key, cap in caps.items():
        counts.setdefault(key, 0)
        assert floors.get(key, 0) <= counts[key] <= cap
    assert sum(counts.values()) == n_units
    return counts


def _objective(counts, age_targets, race_targets, ideal) -> float:
    age_dev = {a: sum(v for (aa, _), v in counts.items() if aa == a) - age_targets[a]
               for a in age_targets}
    race_dev = {r: sum(v for (_, rr), v in counts.items() if rr == r) - race_targets[r]
                for r in race_targets}
    return (W_MARGIN * (sum(d * d for d in age_dev.values())
                        + sum(d * d for d in race_dev.values()))
            + sum((counts.get(k, 0) - ideal[k]) ** 2 / (ideal[k] + EPS) for k in ideal))


def selftest() -> int:
    """Flow optimum == brute-force optimum on random small instances."""
    import itertools
    import random as _random

    rng = _random.Random(20260828)
    global AGES, RACES
    saved = (AGES, RACES)
    failures = 0
    try:
        for trial in range(300):
            AGES = ("A1", "A2")
            RACES = ("R1", "R2", "R3")[: rng.choice((2, 3))]
            caps = {(a, r): rng.randint(0, 4) for a in AGES for r in RACES}
            n = rng.randint(1, max(1, min(8, sum(caps.values()))))
            if sum(caps.values()) < n:
                continue
            # margins deliberately allowed to be infeasible against caps
            age_t = _largest_remainder({a: rng.random() + 0.05 for a in AGES}, n)
            race_t = _largest_remainder({r: rng.random() + 0.05 for r in RACES}, n)
            share = {(a, r): rng.random() + 0.05 for a in AGES for r in RACES}
            total = sum(share.values())
            ideal = {k: n * v / total for k, v in share.items()}

            # half the trials also carry random lower bounds (the nested case)
            floors = {}
            if trial % 2:
                for k in caps:
                    if caps[k] and rng.random() < 0.4:
                        floors[k] = rng.randint(1, caps[k])
                if sum(floors.values()) > n:
                    floors = {}
            got = solve_gender(n, age_t, race_t, ideal, caps, floors=floors or None)
            best = None
            keys = list(caps)
            for combo in itertools.product(*(range(caps[k] + 1) for k in keys)):
                if sum(combo) != n:
                    continue
                cand = dict(zip(keys, combo))
                if any(cand[k] < floors.get(k, 0) for k in keys):
                    continue
                val = _objective(cand, age_t, race_t, ideal)
                if best is None or val < best:
                    best = val
            flow_val = _objective(got, age_t, race_t, ideal)
            if best is None or abs(flow_val - best) > 1e-6 * max(1.0, abs(best)):
                failures += 1
                print(f"  selftest FAIL trial {trial}: flow {flow_val} vs brute {best}")
    finally:
        AGES, RACES = saved
    print(f"tier1 selftest: 300 random instances, {failures} failure(s)")
    return 0 if failures == 0 else 1


# --------------------------------------------------------------------------
# Per-condition stratification
# --------------------------------------------------------------------------

def _demographics() -> dict[str, tuple[str, str, str]]:
    """base_pid -> (gender, age, race) from the frozen survey answers."""
    out: dict[str, tuple[str, str, str]] = {}
    for row in read_csv(DATA / "prestudy_frozen_wide.csv"):
        gender = (row.get("gender") or "").strip()
        age = (row.get("age_band") or "").strip()
        race = (row.get("race") or "").strip()
        if gender in GENDERS and age in AGES and race in RACES:
            out[row["persona_id"]] = (gender, age, race)
    return out


def _solve_pool(pool: dict[str, tuple], cells: dict, n: int,
                floors: dict | None = None):
    """Quotas for one sample of size n drawn from `pool`. Returns (counts, bundle)."""
    gender_total, age_t, race_t, cell_int, ideal = integer_targets(cells, n)
    supply: dict[tuple[str, str, str], int] = {}
    for demo in pool.values():
        supply[demo] = supply.get(demo, 0) + 1
    counts: dict[tuple[str, str, str], int] = {}
    for g in GENDERS:
        caps = {(a, r): supply.get((g, a, r), 0) for a in AGES for r in RACES}
        gender_floor = ({(a, r): floors.get((g, a, r), 0) for a in AGES for r in RACES}
                        if floors else None)
        if gender_floor and sum(gender_floor.values()) > gender_total[g]:
            raise SystemExit(
                f"tier1: the frozen panel already holds {sum(gender_floor.values())} "
                f"{g} twins, more than the {gender_total[g]} this sample allows — "
                f"the nested design is infeasible at n={n}")
        got = solve_gender(
            gender_total[g], age_t[g], race_t[g],
            {(a, r): ideal[(g, a, r)] for a in AGES for r in RACES}, caps,
            floors=gender_floor)
        counts.update({(g, a, r): v for (a, r), v in got.items()})
    return counts, (gender_total, age_t, race_t, cell_int, ideal, supply)


def _margin_rows(condition, label, counts, bundle):
    gender_total, age_t, race_t, _cell_int, _ideal, _supply = bundle
    rows = []
    for g in GENDERS:
        for a in AGES:
            got = sum(v for (gg, aa, _), v in counts.items() if gg == g and aa == a)
            rows.append({"condition": condition, "sample": label, "table": "age_x_gender",
                         "level": a, "gender": g, "target": age_t[g][a], "achieved": got,
                         "deviation": got - age_t[g][a],
                         "exact": "yes" if got == age_t[g][a] else "no"})
        for r in RACES:
            got = sum(v for (gg, _, rr), v in counts.items() if gg == g and rr == r)
            rows.append({"condition": condition, "sample": label, "table": "race_x_gender",
                         "level": r, "gender": g, "target": race_t[g][r], "achieved": got,
                         "deviation": got - race_t[g][r],
                         "exact": "yes" if got == race_t[g][r] else "no"})
    return rows


def _cell_rows(condition, label, counts, bundle, cells, n):
    _gt, _at, _rt, cell_int, ideal, supply = bundle
    rows = []
    for key in sorted(cells):
        g, a, r = key
        rows.append({
            "condition": condition, "sample": label, "gender": g, "age_band": a,
            "race": r, "target_proportion": round(cells[key], 6),
            "ideal_real": round(ideal[key], 3), "target_integer": cell_int[key],
            "available": supply.get(key, 0), "final_quota": counts.get(key, 0),
            "deviation_vs_integer": counts.get(key, 0) - cell_int[key],
            "supply_shortfall": max(0, cell_int[key] - supply.get(key, 0)),
        })
    return rows


def universal_pool(sessions: list[dict], demographics: dict) -> tuple[dict, list[dict]]:
    """Twins exactly complete in EVERY condition, plus the exclusion ledger.

    The panel design needs one set of twins usable in all 17 conditions, so a
    twin is dropped from the study entirely if any single one of its sessions
    is not exactly complete — a stricter rule than the per-condition exclusion
    it replaces, and the reason the same 500 people can answer every arm.
    """
    by_twin: dict[str, list[dict]] = {}
    for row in sessions:
        by_twin.setdefault(row["base_pid"], []).append(row)
    pool: dict[str, tuple] = {}
    excluded: list[dict] = []
    for twin, rows in sorted(by_twin.items(), key=lambda kv: int(kv[0].split("_")[1])):
        bad = [r for r in rows if r["status"] != "exact"]
        if len(rows) != 17:
            excluded.append({"base_pid": twin, "reason": "incomplete_condition_set",
                             "detail": f"{len(rows)} sessions, expected 17"})
        elif bad:
            excluded.append({
                "base_pid": twin, "reason": "not_exact_in_some_condition",
                "detail": "; ".join(f"{b['raw_condition']}: s1 "
                                    f"{b['s1_answered']}/{b['s1_asked']}" for b in bad[:3])})
        elif twin not in demographics:
            excluded.append({"base_pid": twin, "reason": "unmapped_demographics",
                             "detail": ""})
        else:
            pool[twin] = demographics[twin]
    return pool, excluded


def _pick(pool: dict[str, tuple], counts: dict, seed: int, tag: str,
          exclude: set[str] | None = None) -> dict[str, tuple]:
    """Seeded within-cell selection of `counts` twins from `pool`."""
    exclude = exclude or set()
    by_cell: dict[tuple, list[str]] = {}
    for twin, demo in pool.items():
        if twin not in exclude:
            by_cell.setdefault(demo, []).append(twin)
    chosen: dict[str, tuple] = {}
    for cell in sorted(by_cell):
        members = sorted(by_cell[cell], key=lambda t: int(t.split("_")[1]))
        rng_for(seed, "tier1", tag, *cell).shuffle(members)
        take = counts.get(cell, 0)
        if take > len(members):
            raise SystemExit(f"tier1: cell {cell} needs {take}, has {len(members)}")
        for twin in members[:take]:
            chosen[twin] = cell
    return chosen


def run_tier1(*, seed: int = DEFAULT_SEED, only: set[str] | None = None,
              preview: bool = False) -> int:
    """Panel design: one frozen 500-twin panel across all 16 interventions,
    control = that panel plus 500 further twins."""
    cells = load_joint_cells()
    comp_path = AUDIT / "batch" / "completeness_by_session.csv"
    if not comp_path.exists():
        raise SystemExit("tier1: run `sbench completeness` first")
    sessions = read_csv(comp_path)
    demographics = _demographics()
    pool, excluded = universal_pool(sessions, demographics)
    n_twins = len({r["base_pid"] for r in sessions})
    print(f"tier1: universal clean pool {len(pool):,} of {n_twins:,} eligible twins "
          f"({len(excluded)} excluded: not exact in every condition)")

    out_dir = (AUDIT / "tier1_preview") if preview else (DATA / "tier1")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1. the frozen panel -------------------------------------------------
    panel_counts, panel_bundle = _solve_pool(pool, cells, 500)
    panel = _pick(pool, panel_counts, seed, "panel")
    if len(panel) != 500:
        raise SystemExit(f"tier1: panel selected {len(panel)}, expected 500")

    # 2. control: 1,000 nested over the panel ------------------------------
    control_counts, control_bundle = _solve_pool(pool, cells, 1000, floors=panel_counts)
    extra_counts = {k: control_counts.get(k, 0) - panel_counts.get(k, 0) for k in cells}
    if any(v < 0 for v in extra_counts.values()):
        raise SystemExit("tier1: control quota falls below the panel in some cell")
    extra = _pick(pool, extra_counts, seed, "control-extra", exclude=set(panel))
    if len(extra) != 500:
        raise SystemExit(f"tier1: control top-up selected {len(extra)}, expected 500")
    assert not (set(panel) & set(extra))

    # 3. emit --------------------------------------------------------------
    interventions = sorted({("control" if r["condition"] == "control" else r["raw_condition"])
                            for r in sessions} - {"control"})
    if len(interventions) != 16:
        raise SystemExit(f"tier1: found {len(interventions)} interventions, expected 16")

    selection_rows = []
    for condition in interventions + ["control"]:
        members = dict(panel) if condition != "control" else {**panel, **extra}
        for twin in sorted(members, key=lambda t: int(t.split("_")[1])):
            g, a, r = members[twin]
            selection_rows.append({
                "condition": condition, "base_pid": twin, "gender": g, "age_band": a,
                "race": r, "selected_n": len(members),
                "in_panel500": "yes" if twin in panel else "no",
            })

    cell_rows = _cell_rows("panel (all 16 interventions)", "500", panel_counts,
                           panel_bundle, cells, 500)
    cell_rows += _cell_rows("control", "1000", control_counts, control_bundle, cells, 1000)
    margin_rows = _margin_rows("panel (all 16 interventions)", "500", panel_counts, panel_bundle)
    margin_rows += _margin_rows("control", "1000", control_counts, control_bundle)

    write_csv(cell_rows, Path(out_dir) / "tier1_cells.csv", list(cell_rows[0].keys()))
    write_csv(margin_rows, Path(out_dir) / "tier1_margins.csv", list(margin_rows[0].keys()))
    write_csv(selection_rows, Path(out_dir) / "tier1_selection.csv",
              list(selection_rows[0].keys()))
    if excluded:
        write_csv(excluded, Path(out_dir) / "tier1_exclusions.csv", list(excluded[0].keys()))

    def exact_of(label):
        rows = [m for m in margin_rows if m["sample"] == label]
        return f"{sum(1 for m in rows if m['exact'] == 'yes')}/{len(rows)}"

    save_json({
        "seed": seed, "preview": preview,
        "design": "frozen panel: the same 500 twins answer all 16 interventions; "
                  "control = those 500 plus 500 further twins (1,000 total). Every "
                  "twin is exactly complete in all 17 conditions.",
        "universal_pool": len(pool), "excluded_twins": len(excluded),
        "panel": 500, "control_total": 1000, "control_extra": 500,
        "unique_twins_used": len(panel) + len(extra),
        "rows": len(selection_rows),
        "margins_exact": {"panel_500": exact_of("500"), "control_1000": exact_of("1000")},
        "quota_table_sha256": sha256_file(ARTIFACTS / "quota_joint_gender_age_race.csv"),
        "completeness_sha256": sha256_file(comp_path),
        "method": "per-gender 4x5 capacitated transportation, convex costs, "
                  "successive shortest paths with per-cell lower bounds for the "
                  "nested control; W=1e9 lexicographic margins-first; objective "
                  "sum (n-Np)^2/(Np+eps) on joint cells",
    }, Path(out_dir) / "tier1_manifest.json")

    print(f"tier1: panel 500 (margins exact {exact_of('500')}), "
          f"control 1,000 = panel + 500 (margins exact {exact_of('1000')})")
    print(f"tier1: {len(selection_rows):,} rows over "
          f"{len(panel) + len(extra):,} unique twins -> {out_dir}"
          + ("  [PREVIEW]" if preview else ""))
    return 0
