"""CLI for the SYN-DIGITS calibration stage.

    python -m calib tune [--jobs N] [--no-nn]   hyperparameter search, honest split
    python -m calib loo  [--jobs N]             single-config LOO (legacy check)
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="calib")
    sub = parser.add_subparsers(dest="command", required=True)
    p_tune = sub.add_parser("tune", help="grid search on the tune half, report on the holdout half")
    p_tune.add_argument("--jobs", type=int, default=10)
    p_tune.add_argument("--refine-top", type=int, default=3,
                        help="how many screened methods get the full grid")
    p_tune.add_argument("--no-nn", action="store_true", help="skip the neural-net grid")
    p_dry = sub.add_parser("dryrun", help="build the 13 scored outcomes for a few real sessions")
    p_dry.add_argument("--sessions", type=int, default=3)
    sub.add_parser("outcome-manifest", help="print the outcome-definition manifest")
    sub.add_parser("declared-ranges", help="write the anchor declared-range manifest")
    p_reg = sub.add_parser("regression-test",
                           help="check the 13 outcomes against the official R output")
    p_reg.add_argument("--limit", type=int, help="only the first N selection rows")
    sub.add_parser("production", help="fit 221 frozen calibrations and write Tier 2/3 tables")
    args = parser.parse_args(argv)

    if args.command == "tune":
        from .tune import run

        out = run(n_jobs=args.jobs, include_nn=not args.no_nn,
                  refine_top=args.refine_top)
        f, h = out["frozen"], out["holdout"]
        print("\nFROZEN (chosen on tune anchors only)")
        for key in ("method", "config", "min_col_std", "tau", "donor_policy"):
            print(f"  {key:16s} {f[key]}")
        print("\nHOLDOUT (evaluates only the frozen choice)")
        print(f"  {'panel_ae':16s} {h['panel_ae']:.4f} pp   "
              f"baseline {h['baseline_panel_ae']:.4f}   "
              f"({h['improvement_pct']:+.1f}%)")
        print(f"  {'subgroup_rmse':16s} {h['subgroup_rmse']:.4f} pp   "
              f"baseline {h['baseline_subgroup_rmse']:.4f}")
        print(f"  {'item_r':16s} {h['item_r']:.4f}       "
              f"baseline {h['baseline_item_r']:.4f}")
        sel = out["selection"]
        print(f"\n  selection band: {sel['n_within_one_se']} config(s) within "
              f"1 SE ({sel['panel_ae_se']:.3f} pp) of {sel['best_panel_ae']:.4f}")
        return 0

    if args.command == "dryrun":
        from .dryrun import run as dry

        return dry(n_sessions=args.sessions)

    if args.command == "outcome-manifest":
        import json as _json

        from .common import CALIB_DATA, save_json
        from .outcomes import definition_manifest

        manifest = definition_manifest()
        save_json(manifest, CALIB_DATA / "outcome_definitions.json")
        print(_json.dumps({k: v for k, v in manifest.items()
                           if k not in ("rename_map", "outcomes")}, indent=1))
        return 0

    if args.command == "declared-ranges":
        from .ranges import write_manifest

        m = write_manifest()
        print(f"{m['n_anchor_items']} anchor items | "
              f"{m['n_with_declared_range']} with a declared range | "
              f"{m['n_without']} without ({', '.join(m['excluded_from_normalised_aggregate'])})")
        return 0

    if args.command == "regression-test":
        from .regression_test import run as reg

        return reg(limit=args.limit)

    if args.command == "production":
        from .production import run

        result = run()
        print("production:", result["status"])
        print("rows:", result["row_counts"])
        print("Tier-1 parity mismatches:", result["tier1_parity"]["mismatches"])
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
