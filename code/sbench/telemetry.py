"""Per-call provider telemetry capture.

``LLMClient._record_usage_event`` already assembles everything worth keeping —
provider request id, the raw provider ``usage`` block, normalized token counts
(including cached input and reasoning), the pricing row actually applied, and
the realized provider cost in nanos — and posts it to the usage ledger.  The
benchmark has no ledger, so this module taps the same call and mirrors the
event into the run's own audit files.

Nothing is reimplemented and no extra request is made: this is a wrapper around
``repository.record_llm_usage_event`` installed only while a live phase runs.
Events are correlated to the survey stage through the ``BillingContext``
metadata the driver already attaches (persona id, condition, stage index).
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager

from .common import AUDIT, append_jsonl, manifest, wire_worktree

_LOCK = threading.Lock()
_EVENTS: list[dict] = []


def _provenance() -> dict:
    """Code commit and QSF hash stamped on every call, so a row is self-describing."""
    try:
        record = manifest()
        return {
            "code_commit": record["surveytwin"]["commit"],
            "qsf_sha256": record["benchmark_repo"]["authoritative_files"]["survey/survey.qsf"],
            "benchmark_commit": record["benchmark_repo"]["commit"],
        }
    except Exception:
        return {"code_commit": "", "qsf_sha256": "", "benchmark_commit": ""}


_PROVENANCE = None


def _flatten(event: dict) -> dict:
    """One CSV row per provider call, with the fields a cost audit needs."""
    usage = event.get("usage") or {}
    context = event.get("context") or {}
    metadata = context.get("metadata") or {}
    nanos = event.get("provider_cost_nanos_usd") or 0
    pricing = event.get("pricing") or {}
    priced = any(
        int(pricing.get(key) or 0) > 0
        for key in ("input_nanos_per_token", "output_nanos_per_token")
    )
    global _PROVENANCE
    if _PROVENANCE is None:
        _PROVENANCE = _provenance()
    return {
        **_PROVENANCE,
        "prompt_sha256": metadata.get("prompt_sha256", ""),
        "app_event_id": event.get("app_event_id", ""),
        "provider_request_id": event.get("provider_request_id", ""),
        "provider_response_id": event.get("provider_response_id", ""),
        "status": event.get("status", ""),
        "model_alias": event.get("model_alias", ""),
        "resolved_model": event.get("resolved_model", ""),
        "persona_id": metadata.get("persona_id", ""),
        "raw_condition": metadata.get("raw_condition", ""),
        "stage_index": metadata.get("stage_index", ""),
        "feature": context.get("feature", ""),
        "uncached_input_tokens": usage.get("uncached_input_tokens", ""),
        "cached_input_tokens": usage.get("cached_input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""),
        "reasoning_tokens": usage.get("reasoning_tokens", ""),
        "total_tokens": usage.get("total_tokens", ""),
        "provider_cost_nanos_usd": nanos,
        "provider_cost_usd": round(int(nanos) / 1e9, 8) if nanos else 0.0,
        "cost_priced": "yes" if priced else "no",
        "pricing_source": pricing.get("source", ""),
        "latency_ms": event.get("latency_ms", ""),
        "error_text": (event.get("error_text") or "")[:300],
    }


@contextmanager
def capture(phase: str):
    """Mirror every provider call made inside the block into the run audit.

    Writes ``audit/<phase>_provider_calls.jsonl`` (the full event, raw provider
    usage block included, verbatim) and returns the flattened rows so the caller
    can also write a CSV and report realized cost.
    """
    wire_worktree()
    from services.v3 import repository as repo_module  # noqa: E402
    from services.v2 import llm_client as llm_module  # noqa: E402
    from services.v3.billing import default_pricing_for_model  # noqa: E402

    # The benchmark runs with no platform database, so two calls inside
    # ``_record_usage_event`` would otherwise abort it before any event is
    # emitted: the ``usage_tracking_enabled()`` feature flag (read from the
    # app_settings table) and the pricing lookup.  Both are replaced with
    # database-free equivalents for the duration of the capture — the seeded
    # provider rate card is the same table the platform falls back to.
    original_flag = llm_module.usage_tracking_enabled
    original_pricing = repo_module.repository.get_current_pricing
    llm_module.usage_tracking_enabled = lambda: True

    def _pricing(*, provider="openai", model="", **_kwargs):
        return default_pricing_for_model(provider, model)

    repo_module.repository.get_current_pricing = _pricing

    original = repo_module.repository.record_llm_usage_event
    jsonl_path = AUDIT / f"{phase}_provider_calls.jsonl"
    collected: list[dict] = []

    def tapped(event, *args, **kwargs):
        try:
            with _LOCK:
                collected.append(_flatten(event))
                append_jsonl(json.loads(json.dumps(event, default=str)), jsonl_path)
        except Exception:  # telemetry must never break a run
            pass
        try:
            return original(event, *args, **kwargs)
        except Exception:
            # No usage ledger/DB in the benchmark environment — the mirrored
            # copy above is the record of truth here.
            return None

    repo_module.repository.record_llm_usage_event = tapped
    try:
        yield collected
    finally:
        repo_module.repository.record_llm_usage_event = original
        repo_module.repository.get_current_pricing = original_pricing
        llm_module.usage_tracking_enabled = original_flag


def summarize(rows: list[dict]) -> dict:
    def total(key):
        return sum(int(r[key]) for r in rows if str(r.get(key, "")).strip().isdigit())

    nanos = sum(int(r["provider_cost_nanos_usd"]) for r in rows
                if str(r.get("provider_cost_nanos_usd", "")).strip().lstrip("-").isdigit())
    return {
        "calls": len(rows),
        "failed_calls": sum(1 for r in rows if r.get("status") != "completed"),
        "uncached_input_tokens": total("uncached_input_tokens"),
        "cached_input_tokens": total("cached_input_tokens"),
        "output_tokens": total("output_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "total_tokens": total("total_tokens"),
        "realized_provider_cost_usd": round(nanos / 1e9, 6),
        "calls_missing_request_id": sum(1 for r in rows if not r.get("provider_request_id")),
        "unpriced_calls": sum(1 for r in rows if r.get("cost_priced") == "no"),
        "cost_note": (
            "gpt-5.6-luna has no entry in the platform's seeded rate card, so realized "
            "cost is reported as 0 and token counts are the authoritative record; "
            "multiply tokens by the true rate card to price the run"
            if any(r.get("cost_priced") == "no" for r in rows) else ""
        ),
    }


TELEMETRY_COLUMNS = [
    # provenance — every row is self-describing
    "code_commit", "benchmark_commit", "qsf_sha256", "prompt_sha256",
    # identity of the call
    "app_event_id", "provider_request_id", "provider_response_id", "status",
    "model_alias", "resolved_model",
    # where it sits in the survey
    "persona_id", "raw_condition", "stage_index", "feature",
    # usage and cost.  A technical retry is a SEPARATE provider call with its
    # own row and its own status, so there is no per-call retry counter; the
    # per-stage retry count lives in <phase>_calls.csv and the stage JSONL.
    "uncached_input_tokens", "cached_input_tokens", "output_tokens",
    "reasoning_tokens", "total_tokens",
    "provider_cost_nanos_usd", "provider_cost_usd", "cost_priced", "pricing_source",
    "latency_ms", "error_text",
]
