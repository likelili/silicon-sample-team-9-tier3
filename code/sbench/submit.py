"""Submit the built round-1 chunk files to the OpenAI Batch API.

Uploads the chunk files **byte for byte** rather than round-tripping the
requests through Python. Those exact bytes are what the pre-submission checks
validated and what was reviewed; re-serializing could change key order or
unicode escaping and would leave the shipped payload subtly different from the
approved one.

Batch ids are appended to ``data/batch/round1/submitted_batches.jsonl`` the
instant each job is created, before anything else happens. Losing an id would
mean paying for work that can never be collected, so that write is the first
thing after the API returns.

Nothing here cancels a job. A submitted batch is left alone.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from .common import DATA, load_json, wire_worktree

ROUND1 = "batch/round1"
LEDGER = "submitted_batches.jsonl"


def _record(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def submit_round1(*, dry_run: bool = False, round_no = 1) -> int:
    wire_worktree()
    import httpx

    from .common import require_live_flag

    require_live_flag(not dry_run, "batch-submit")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-offline"):
        raise SystemExit("batch-submit: a real OPENAI_API_KEY is required")

    label = str(round_no)
    sub = label if label.startswith("repair") else f"round{label}"
    out_dir = DATA / "batch" / sub
    manifest = load_json(out_dir / "manifest.json")
    chunks = sorted(Path(out_dir).glob("requests*.jsonl"))
    if not chunks:
        raise SystemExit(f"batch-submit: no chunk files in {out_dir}")

    # Refuse to submit a build whose checks did not pass.
    failed = [k for k, v in manifest.get("checks", {}).items() if v.get("ok") is False]
    if failed:
        raise SystemExit(f"batch-submit: REFUSING — build has failing checks: {failed}")

    ledger = Path(out_dir) / LEDGER
    already = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                already.add(json.loads(line)["file"])
            except (json.JSONDecodeError, KeyError):
                continue
        if already:
            print(f"batch-submit: {len(already)} chunk(s) already submitted, skipping those")

    expected = sum(c["requests"] for c in manifest.get("chunks", []))
    if not expected:
        expected = int(manifest.get("requests") or 0)
    total_lines = 0
    for path in chunks:
        with open(path, "rb") as handle:
            total_lines += sum(1 for _ in handle)
    print(f"batch-submit: {len(chunks)} chunk file(s), {total_lines:,} requests, "
          f"{sum(p.stat().st_size for p in chunks)/1e9:.2f} GB")
    if expected and total_lines != expected:
        raise SystemExit(f"batch-submit: REFUSING — {total_lines} lines on disk but the "
                         f"manifest expects {expected}; rebuild before submitting")
    if dry_run:
        for path in chunks:
            print(f"  would submit {path.name}  {path.stat().st_size/1e6:.1f} MB")
        return 0

    auth = {"Authorization": f"Bearer {key}"}
    submitted: list[dict] = []
    async with httpx.AsyncClient(timeout=None) as client:
        for index, path in enumerate(chunks, start=1):
            if path.name in already:
                continue
            size_mb = path.stat().st_size / 1e6
            print(f"  [{index}/{len(chunks)}] uploading {path.name} ({size_mb:.1f} MB)...",
                  flush=True)
            with open(path, "rb") as handle:
                upload = await client.post(
                    "https://api.openai.com/v1/files", headers=auth,
                    files={"file": (path.name, handle, "application/jsonl")},
                    data={"purpose": "batch"},
                )
            if upload.status_code != 200:
                print(f"      UPLOAD FAILED {upload.status_code}: {upload.text[:400]}")
                return 1
            file_id = upload.json()["id"]

            created = await client.post(
                "https://api.openai.com/v1/batches",
                headers={**auth, "Content-Type": "application/json"},
                json={"input_file_id": file_id, "endpoint": "/v1/chat/completions",
                      "completion_window": "24h",
                      "metadata": {"purpose": f"sbench-{sub}", "chunk": path.name,
                                   "run": Path(DATA).parent.name}},
            )
            if created.status_code != 200:
                print(f"      CREATE FAILED {created.status_code}: {created.text[:400]}")
                # The uploaded file is recorded so it is not orphaned silently.
                _record(ledger, {"file": path.name, "input_file_id": file_id,
                                 "batch_id": None, "error": created.text[:300]})
                return 1
            batch = created.json()
            row = {"file": path.name, "input_file_id": file_id,
                   "batch_id": batch["id"], "status": batch.get("status")}
            _record(ledger, row)          # durable BEFORE anything else
            submitted.append(row)
            print(f"      batch_id={batch['id']}  status={batch.get('status')}", flush=True)

    print(f"\nbatch-submit: {len(submitted)} job(s) created")
    print(f"  ledger: {ledger}")
    print("  poll with: python -m sbench batch-status --run-dir <dir>")
    return 0


async def _get_with_retry(client, url, headers, attempts: int = 4):
    """GET with backoff.

    Transient DNS and connection errors are routine on a poller that runs for
    hours. Without this one blip aborts the whole pass, and a job that finished
    during it goes uncollected until some later pass happens to catch it.
    """
    import asyncio as _asyncio

    last = None
    for attempt in range(attempts):
        try:
            return await client.get(url, headers=headers)
        except Exception as exc:          # network-level only; HTTP errors return normally
            last = exc
            await _asyncio.sleep(2 * (attempt + 1))
    raise last


async def batch_status(*, collect: bool = False, round_no = 1) -> int:
    """Status of every submitted round-1 job; optionally download finished output."""
    wire_worktree()
    import httpx

    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-offline"):
        raise SystemExit("batch-status: a real OPENAI_API_KEY is required")
    label = str(round_no)
    sub = label if label.startswith("repair") else f"round{label}"
    out_dir = DATA / "batch" / sub
    ledger = Path(out_dir) / LEDGER
    if not ledger.exists():
        raise SystemExit(f"batch-status: nothing submitted yet ({ledger} missing)")

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("batch_id")]
    auth = {"Authorization": f"Bearer {key}"}
    totals = {"total": 0, "completed": 0, "failed": 0}
    states: dict[str, int] = {}
    results_dir = Path(out_dir).parent / f"{sub}_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    unreachable: list[str] = []
    async with httpx.AsyncClient(timeout=None) as client:
        for row in rows:
            try:
                resp = await _get_with_retry(
                    client, f"https://api.openai.com/v1/batches/{row['batch_id']}", auth)
                batch = resp.json()
            except Exception as exc:
                unreachable.append(row["file"])
                print(f"  {row['file']:22s} UNREACHABLE  {type(exc).__name__}")
                continue
            status = batch.get("status", "?")
            counts = batch.get("request_counts") or {}
            states[status] = states.get(status, 0) + 1
            for k in totals:
                totals[k] += int(counts.get(k) or 0)
            print(f"  {row['file']:22s} {status:12s} "
                  f"{counts.get('completed',0):>6}/{counts.get('total',0):<6} "
                  f"failed={counts.get('failed',0):<5} {row['batch_id']}")
            if collect and status == "completed" and batch.get("output_file_id"):
                dest = results_dir / f"{row['file'].replace('.jsonl','')}_output.jsonl"
                if not dest.exists():
                    try:
                        content = await _get_with_retry(
                            client,
                            f"https://api.openai.com/v1/files/{batch['output_file_id']}/content",
                            auth)
                    except Exception as exc:
                        unreachable.append(row["file"])
                        print(f"      download failed ({type(exc).__name__}); will retry")
                        continue
                    # Write to a temp name and rename, so an interrupted download
                    # can never leave a half-file that later looks collected.
                    tmp = dest.with_suffix(".partial")
                    tmp.write_bytes(content.content)
                    got = sum(1 for _ in open(tmp, "rb"))
                    want = int(counts.get("completed") or 0)
                    if got != want:
                        print(f"      TRUNCATED: {got} lines, expected {want}; discarding")
                        tmp.unlink()
                        unreachable.append(row["file"])
                        continue
                    tmp.rename(dest)
                    print(f"      -> saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB, "
                          f"{got:,} lines)")
            if batch.get("error_file_id"):
                print(f"      ERROR FILE present: {batch['error_file_id']}")

    print(f"\n  jobs by state: {states}")
    print(f"  requests: {totals['completed']:,} completed / {totals['total']:,} "
          f"({totals['failed']:,} failed)")
    if collect:
        print(f"  outputs in: {results_dir}")
    if unreachable:
        print(f"  could not reach/collect {len(set(unreachable))} job(s) this pass; will retry")
    done = (states
            and all(s in ("completed", "failed", "expired", "cancelled") for s in states)
            and not unreachable
            and len(states) > 0
            and sum(states.values()) == len(rows))
    return 0 if done else 2
