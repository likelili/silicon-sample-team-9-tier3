"""CLI for the Silicon Sample Benchmark pipeline.

    python -m sbench audit                          # section 3
    python -m sbench templates                      # sections 4-6
    python -m sbench preflight [--quick]            # section 13 (mocked, free)
    python -m sbench prestudy  --mode mock|dry|live # section 7
    python -m sbench repairs   --mode mock|live     # section 8
    python -m sbench freeze                         # section 9
    python -m sbench sampling                       # section 10
    python -m sbench poststudy --mode mock|dry|live # sections 11-12
    python -m sbench verify                         # post-execution assertions

Live modes refuse to run without OPENAI_API_KEY *and* SBENCH_APPROVED=1.
"""

from __future__ import annotations

import argparse
import asyncio

from datetime import datetime

from .common import (
    BENCH_ROOT,
    DEFAULT_MODEL,
    DEFAULT_PERSONA_TYPE,
    DEFAULT_SEED,
    RUNS,
    new_run_dir,
    set_run_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="sbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # Run isolation: every phase that writes per-run data needs a run dir.
    def runopts(p):
        p.add_argument("--run-dir", help="existing run directory to read/write")
        p.add_argument("--new-run", metavar="LABEL",
                       help="create a fresh timestamped run directory with this label")

    def common(p):
        runopts(p)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--persona-type", default=DEFAULT_PERSONA_TYPE,
                       choices=["full", "summary", "demographics"])
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--concurrency", type=int, default=8)
        p.add_argument("--limit", type=int)
        p.add_argument("--mode", default="mock", choices=["mock", "dry", "live"])

    sub.add_parser("audit")
    sub.add_parser("templates")
    runopts(sub.add_parser("diagnose"))
    p = sub.add_parser("preflight")
    common(p)
    p.add_argument("--quick", action="store_true", help="skip the full mocked end-to-end")
    for name in ("prestudy", "repairs", "poststudy"):
        p_ = sub.add_parser(name)
        common(p_)
        if name == "prestudy":
            p_.add_argument("--resume", action="store_true",
                            help="skip twins already in the run's checkpoint")
    runopts(sub.add_parser("freeze"))
    p = sub.add_parser("sampling")
    runopts(p)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    runopts(sub.add_parser("verify"))
    runopts(sub.add_parser("export-joined"))
    runopts(sub.add_parser("runs"))
    runopts(sub.add_parser("report"))
    runopts(sub.add_parser("archive"))
    pc = sub.add_parser("pilot-check")
    runopts(pc)
    pc.add_argument("--expected", type=int, default=25)
    bp = sub.add_parser("study-bootstrap")
    runopts(bp)
    bp.add_argument("--source-run", required=True,
                    help="frozen pre-study run directory to reuse")
    runopts(sub.add_parser("study-eligibility"))
    st = sub.add_parser("study")
    common(st)
    st.add_argument("--resume", action="store_true",
                    help="skip sessions already in the run's study checkpoint")
    st.add_argument("--pilot-twins", type=int,
                    help="run only the first N eligible twins across ALL 17 conditions")
    runopts(sub.add_parser("study-validate"))
    bs = sub.add_parser("batch-submit")
    runopts(bs)
    bs.add_argument("--dry-run", action="store_true", help="list what would be submitted")
    bs.add_argument("--round", default="1")
    bst = sub.add_parser("batch-status")
    runopts(bst)
    bst.add_argument("--collect", action="store_true", help="download finished output files")
    bst.add_argument("--round", default="1")
    runopts(sub.add_parser("completeness"))
    t1 = sub.add_parser("tier1")
    runopts(t1)
    t1.add_argument("--seed", type=int, default=DEFAULT_SEED)
    t1.add_argument("--selftest", action="store_true",
                    help="verify the optimizer against brute force; no data needed")
    t1.add_argument("--preview", action="store_true",
                    help="write into audit/tier1_preview instead of data/tier1")
    t1.add_argument("--conditions", help="comma-separated subset of conditions")
    runopts(sub.add_parser("tier1-export"))
    rp = sub.add_parser("batch-repair-build")
    runopts(rp)
    rm = sub.add_parser("batch-repair-merge")
    runopts(rm)
    bc = sub.add_parser("batch-continuity")
    common(bc)
    bc.add_argument("--twins", type=int, default=25,
                    help="how many twins to exercise through the split round 1 / round 2")
    b2 = sub.add_parser("batch-build-round2")
    common(b2)
    b2.add_argument("--results", required=True,
                    help="round-1 batch output JSONL (OpenAI batch result format)")
    bb = sub.add_parser("batch-build")
    common(bb)
    bb.add_argument("--pilot-twins", type=int,
                    help="build only the first N eligible twins' complete 17-condition sets")

    args = parser.parse_args()

    # Resolve the run directory before any phase touches DATA/AUDIT.
    if getattr(args, "new_run", None):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = new_run_dir(args.new_run, stamp)
        print(f"run directory: {path}")
    elif getattr(args, "run_dir", None):
        path = set_run_dir(args.run_dir)
        print(f"run directory: {path}")

    if args.command == "runs":
        if RUNS.is_dir():
            for entry in sorted(RUNS.iterdir()):
                if entry.is_dir():
                    marker = entry / "data" / "run_manifest.json"
                    print(f"  {entry.name}" + ("   [has run_manifest]" if marker.exists() else ""))
        else:
            print("  no runs yet")
        return 0

    if args.command == "audit":
        from .qsf_audit import run_audit
        return run_audit()
    if args.command == "templates":
        from .templates import build
        return build()
    if args.command == "preflight":
        from .preflight import run_preflight
        return asyncio.run(run_preflight(
            model=args.model, persona_type=args.persona_type, seed=args.seed,
            full_e2e=not args.quick))
    if args.command == "prestudy":
        from .phases import run_prestudy
        return asyncio.run(run_prestudy(
            mode=args.mode, model=args.model, persona_type=args.persona_type,
            seed=args.seed, limit=args.limit, concurrency=args.concurrency,
            mock_profile_aware=True, resume=args.resume))
    if args.command == "repairs":
        from .phases import run_repairs
        return asyncio.run(run_repairs(
            mode=args.mode, model=args.model, persona_type=args.persona_type,
            seed=args.seed, concurrency=args.concurrency))
    if args.command == "study-bootstrap":
        from .study import bootstrap_from
        return bootstrap_from(args.source_run)
    if args.command == "study-eligibility":
        from .study import export_eligibility
        return export_eligibility()
    if args.command == "study":
        from .study import run_study
        return asyncio.run(run_study(
            mode=args.mode, model=args.model, persona_type=args.persona_type,
            seed=args.seed, concurrency=args.concurrency, limit=args.limit,
            resume=args.resume, pilot_twins=args.pilot_twins))
    if args.command == "batch-submit":
        from .submit import submit_round1
        return asyncio.run(submit_round1(dry_run=args.dry_run, round_no=args.round))
    if args.command == "batch-status":
        from .submit import batch_status
        return asyncio.run(batch_status(collect=args.collect, round_no=args.round))
    if args.command == "completeness":
        from .completeness import build_completeness
        return build_completeness()
    if args.command == "tier1":
        from .tier1 import run_tier1, selftest
        if args.selftest:
            return selftest()
        only = set(args.conditions.split(",")) if args.conditions else None
        return run_tier1(seed=args.seed, only=only, preview=args.preview)
    if args.command == "tier1-export":
        from .tier1_export import run_export
        return run_export()
    if args.command == "batch-repair-build":
        from .repair import build_repair_batch
        return asyncio.run(build_repair_batch())
    if args.command == "batch-repair-merge":
        from .repair import merge_repairs
        return merge_repairs()
    if args.command == "batch-continuity":
        from .round2 import verify_continuity
        return asyncio.run(verify_continuity(
            model=args.model, persona_type=args.persona_type, seed=args.seed,
            twins=args.twins))
    if args.command == "batch-build-round2":
        from .round2 import build_round2
        return asyncio.run(build_round2(
            model=args.model, persona_type=args.persona_type, seed=args.seed,
            concurrency=args.concurrency, results_path=args.results))
    if args.command == "batch-build":
        from .batch import build_round1
        return asyncio.run(build_round1(
            model=args.model, persona_type=args.persona_type, seed=args.seed,
            concurrency=args.concurrency, limit=args.limit,
            pilot_twins=args.pilot_twins))
    if args.command == "study-validate":
        from .study import validate_study
        return validate_study()
    if args.command == "archive":
        from .archive import run_archive
        return run_archive()
    if args.command == "pilot-check":
        from .pilot_check import run_pilot_checks
        return run_pilot_checks(expected=args.expected)
    if args.command == "report":
        from .report import build_report
        return build_report()
    if args.command == "diagnose":
        from .phases import run_diagnose
        return run_diagnose()
    if args.command == "freeze":
        from .phases import run_freeze
        return run_freeze()
    if args.command == "sampling":
        from .sampling import run_sampling
        return run_sampling(seed=args.seed)
    if args.command == "poststudy":
        from .poststudy import run_poststudy
        return asyncio.run(run_poststudy(
            mode=args.mode, model=args.model, persona_type=args.persona_type,
            seed=args.seed, concurrency=args.concurrency, limit=args.limit))
    if args.command == "verify":
        from .preflight import run_verify
        return run_verify()
    if args.command == "export-joined":
        from .export import run_export_joined
        return run_export_joined()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
