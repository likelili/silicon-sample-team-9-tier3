"""Freeze a completed run directory into a read-only snapshot.

Produces, next to the run directory:
  * ``<run>.MANIFEST.json`` — SHA-256 and size of every file, plus provenance
  * ``<run>.tar.gz``        — the archive itself
  * ``<run>.tar.gz.sha256`` — hash of the archive

and then marks every file inside the run directory read-only (0o444) and the
directories 0o555, so a later phase cannot mutate a reviewed run in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

from .common import BENCH_ROOT, active_run_dir, load_json, save_json


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_archive(*, read_only: bool = True) -> int:
    run = active_run_dir()
    manifest = load_json(BENCH_ROOT / "manifest" / "source_manifest.json")

    files = sorted(p for p in run.rglob("*") if p.is_file())
    inventory = {
        str(p.relative_to(run)): {"sha256": _sha(p), "bytes": p.stat().st_size}
        for p in files
    }
    record = {
        "run_dir": run.name,
        "files": len(inventory),
        "total_bytes": sum(v["bytes"] for v in inventory.values()),
        "provenance": {
            "surveytwin_commit": manifest["surveytwin"]["commit"],
            "benchmark_commit": manifest["benchmark_repo"]["commit"],
            "qsf_sha256": manifest["benchmark_repo"]["authoritative_files"]["survey/survey.qsf"],
            "model": manifest["run_config"]["model"],
            "persona_representation": manifest["run_config"]["persona_representation"],
            "seed": manifest["run_config"]["seed"],
        },
        "phases_run": ["prestudy", "diagnose"],
        "phases_deliberately_not_run": ["repairs", "freeze", "sampling", "poststudy"],
        "inventory": inventory,
    }
    manifest_path = run.parent / f"{run.name}.MANIFEST.json"
    save_json(record, manifest_path)

    archive_path = run.parent / f"{run.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(run, arcname=run.name)
        tar.add(manifest_path, arcname=manifest_path.name)
    archive_hash = _sha(archive_path)
    (run.parent / f"{run.name}.tar.gz.sha256").write_text(
        f"{archive_hash}  {archive_path.name}\n", encoding="utf-8"
    )

    frozen = 0
    if read_only:
        for path in files:
            os.chmod(path, 0o444)
            frozen += 1
        for directory in sorted((p for p in run.rglob("*") if p.is_dir()), reverse=True):
            os.chmod(directory, 0o555)
        os.chmod(run, 0o555)

    print(f"archive: {len(inventory)} file(s), {record['total_bytes']:,} bytes")
    print(f"  {archive_path.name}  sha256 {archive_hash[:16]}…")
    print(f"  {manifest_path.name}")
    if read_only:
        print(f"  run directory frozen read-only ({frozen} files)")
    return 0
