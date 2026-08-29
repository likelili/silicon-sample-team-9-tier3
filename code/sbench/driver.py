"""Staged execution driver — one twin through one template, with full logging.

This is a thin harness around SurveyTwin's own machinery; it reimplements no
Qualtrics logic:

  * ``QsfRuntime``                     — staged flow evaluation, branching,
                                         randomization, embedded data;
  * ``staged_prior_answer_context`` /
    ``staged_survey_sequence_with_prior_context``
                                       — the production prior-answer prompt;
  * ``build_simulation_prompt_parts``  — the production prompt construction;
  * ``simulate_persona``               — the production LLM call
                                         (persona loading included);
  * ``_answers_with_fallbacks``        — the production response normalization
                                         and answer-status flagging.

Three modes:
  mock — deterministic seeded answers, no network.  Used by every preflight
         test.  Supports scripted per-question overrides so branch routes can
         be forced.
  dry  — builds every real prompt and records sizes, then advances the runtime
         on mock answers.  No network.  Used for cost estimation.
  live — real API calls.  Refused unless the phase passed the approval gate.

Two levers the phases use:
  forced_choices — pre-seeded ``state.randomizer_choices`` (fixed condition
                   assignment).  The runtime honours seeds and records them in
                   its own randomizer audit events.
  prior_qa       — frozen pre-study (question, answer) pairs applied to the
                   state before the first render.  They enter the prior-answer
                   context in their original order and satisfy any latent
                   reference to a pre-study answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .common import register_models, wire_worktree


@dataclass
class StageLog:
    stage_index: int
    question_ids: list[str]
    # The full question dicts, kept so a later round can replay this stage's
    # (question, answer) pairs into a fresh runtime — apply_answers needs the
    # dicts, not just the ids.
    questions: list[dict]
    prompt_sha256: str
    prompt_chars: dict[str, int]
    system_prompt: str
    user_prompt_head: str
    raw_answers: list[dict]
    normalized_answers: list[dict]
    embedded_data: dict
    events: list[dict]
    rendered_block_ids: list[str]
    # The runtime's displayed-question set as it stands AFTER this stage. A
    # later round must restore it or every text-only block it already showed
    # (transitions, the treatment stimulus) is re-emitted into the next prompt:
    # blocks are gated on first display, not on being answered.
    displayed_qids_after: list[str]
    prior_answer_count: int
    latency_ms: int
    retries: int
    mode: str


@dataclass
class TwinRunResult:
    persona_id: str
    persona_index: int
    rows: list[dict] = field(default_factory=list)          # one per rendered question
    stages: list[StageLog] = field(default_factory=list)
    embedded_data: dict = field(default_factory=dict)
    randomizer_choices: dict = field(default_factory=dict)
    randomization_seed: int = 0
    displayed_qids: list[str] = field(default_factory=list)
    done: bool = False
    excluded: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def _mock_answer(question: dict, rng, overrides: dict[str, str] | None) -> dict:
    qid = str(question.get("id", ""))
    overrides = overrides or {}
    if qid in overrides:
        value = overrides[qid]
    else:
        options = question.get("options") or []
        if options:
            value = str(options[rng.randrange(len(options))])
        else:
            lo = question.get("numeric_min")
            hi = question.get("numeric_max")
            if lo is not None and hi is not None:
                try:
                    value = str(int((float(lo) + float(hi)) // 2))
                except (TypeError, ValueError):
                    value = "50"
            elif str(question.get("type")) == "numeric":
                value = "50"
            else:
                value = f"mock answer for {qid}"
    return {"question_id": qid, "answer_value": value, "answer_label": value, "answer_raw": value}


class StagedDriver:
    def __init__(
        self,
        template: dict,
        *,
        mode: str,
        model: str,
        persona_type: str,
        seed: int,
        persona_count: int,
        max_stages: int = 25,
        max_retries: int = 2,
    ) -> None:
        assert mode in ("mock", "dry", "live")
        wire_worktree()
        register_models()
        from services.v3.qsf_runtime import QsfRuntime  # noqa: E402

        self._QsfRuntime = QsfRuntime
        self.template = template
        self.mode = mode
        self.model = model
        self.persona_type = persona_type
        self.seed = seed
        self.persona_count = persona_count
        self.max_stages = max_stages
        self.max_retries = max_retries

    # -- internals ----------------------------------------------------------

    def _persona_text(self, persona_id: str) -> str:
        from services.v2.persona_loader import load_persona_text  # noqa: E402

        text = load_persona_text(persona_id, self.persona_type)
        if not text.strip() or text.startswith("[Persona"):
            raise RuntimeError(f"persona {persona_id} resolves to blank/placeholder text")
        return text

    async def _answers_for_stage(
        self,
        *,
        persona_id: str,
        questions: list[dict],
        sequence_text: str,
        rng,
        overrides: dict[str, str] | None,
        billing_meta: dict,
    ) -> tuple[list[dict], int]:
        """Returns (raw answers, technical retry count)."""
        if self.mode in ("mock", "dry"):
            return [_mock_answer(q, rng, overrides) for q in questions], 0

        from services.v2.simulation_executor import simulate_persona  # noqa: E402
        from services.v3.billing import BillingContext  # noqa: E402

        context = BillingContext(
            user_id="silicon-bench",
            feature="benchmark.simulation",
            metadata=billing_meta,
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = await simulate_persona(
                    model=self.model,
                    persona_type=self.persona_type,
                    persona_id=persona_id,
                    questions=questions,
                    survey_sequence_text=sequence_text,
                    billing_context=context,
                )
                return raw or [], attempt
            except Exception as exc:  # technical failure -> bounded retry
                last_error = exc
                await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts: {last_error}")

    # -- public -------------------------------------------------------------

    async def run_twin(
        self,
        persona_id: str,
        persona_index: int,
        *,
        forced_choices: dict[str, list[int]] | None = None,
        prior_qa: list[tuple[dict, dict]] | None = None,
        overrides: dict[str, str] | None = None,
        rng_extra: str = "",
        session_seed: int | None = None,
        restore_displayed_qids: list[str] | None = None,
        billing_extra: dict | None = None,
        on_stage: Callable[[StageLog], None] | None = None,
    ) -> TwinRunResult:
        from services.v3.qsf_runtime import (  # noqa: E402
            staged_prior_answer_context,
            staged_survey_sequence_with_prior_context,
        )
        from services.v2.simulation_executor import build_simulation_prompt_parts  # noqa: E402
        import services.v3.tool_registry  # noqa: F401,E402  (breaks the registry<->tool import cycle)
        from services.v3.tools.simulation_tool import _answers_with_fallbacks  # noqa: E402
        from .common import rng_for

        result = TwinRunResult(persona_id=persona_id, persona_index=persona_index)
        # Every QsfRuntime randomization path interpolates ``seed`` — flow
        # randomizer ordering, even presentation, block question order and
        # choice order alike — so a session-specific seed makes all of them
        # independent per twin x condition. Falls back to the global seed when
        # a caller has no session identity (preflight fixtures, pre-study).
        runtime_seed = self.seed if session_seed is None else int(session_seed)
        runtime = self._QsfRuntime(
            self.template, seed=runtime_seed, max_stages=self.max_stages,
            persona_count=self.persona_count,
        )
        state = runtime.new_state(persona_id=persona_id, persona_index=persona_index)

        if prior_qa:
            questions = [q for q, _ in prior_qa]
            answers = [a for _, a in prior_qa]
            runtime.apply_answers(state, questions=questions, answers=answers)

        if forced_choices:
            for flow_id, indexes in forced_choices.items():
                state.randomizer_choices[str(flow_id)] = list(indexes)

        # Applying answers records what was ANSWERED, never what was SHOWN, so a
        # resumed session must be told which descriptive content the earlier
        # round already displayed. Without this every text-only block reappears.
        if restore_displayed_qids:
            state.displayed_qids.update(str(q) for q in restore_displayed_qids)

        seeded_count = len(state.rendered_questions)
        persona_text = self._persona_text(persona_id)
        rng = rng_for(self.seed, persona_id, rng_extra)

        try:
            while True:
                stage = runtime.render_next_stage(state)
                if not stage.questions:
                    break
                stage_index = len(result.stages) + 1
                prior_context = staged_prior_answer_context(state)
                sequence = staged_survey_sequence_with_prior_context(
                    prior_answer_context=prior_context,
                    survey_sequence_text=stage.survey_sequence_text,
                )
                system_p, persona_p, questions_p = build_simulation_prompt_parts(
                    persona_text, stage.questions, survey_sequence_text=sequence,
                )
                prompt_sha = hashlib.sha256(
                    (system_p + "\n" + persona_p + "\n" + questions_p).encode("utf-8")
                ).hexdigest()
                started = time.monotonic()
                raw, retries = await self._answers_for_stage(
                    persona_id=persona_id,
                    questions=stage.questions,
                    sequence_text=sequence,
                    rng=rng,
                    overrides=overrides,
                    billing_meta={
                        "persona_id": persona_id,
                        "persona_index": persona_index,
                        "stage_index": stage_index,
                        "rng_extra": rng_extra,
                        "prompt_sha256": prompt_sha,
                        **(billing_extra or {}),
                    },
                )
                latency_ms = int((time.monotonic() - started) * 1000)
                normalized = _answers_with_fallbacks(
                    answers=raw,
                    questions=stage.questions,
                    embedded_data=stage.embedded_data,
                    displayed_qids=state.displayed_qids,
                )
                runtime.apply_answers(state, questions=stage.questions, answers=normalized)
                log = StageLog(
                    stage_index=stage_index,
                    question_ids=[str(q.get("id")) for q in stage.questions],
                    questions=[dict(q) for q in stage.questions],
                    prompt_sha256=prompt_sha,
                    prompt_chars={
                        "system": len(system_p),
                        "persona": len(persona_p),
                        "questions": len(questions_p),
                        "total": len(system_p) + len(persona_p) + len(questions_p),
                    },
                    system_prompt=system_p if self.mode != "mock" else "",
                    user_prompt_head=(persona_p + "\n\n" + questions_p) if self.mode != "mock" else "",
                    raw_answers=raw,
                    normalized_answers=normalized,
                    embedded_data=dict(state.embedded_data),
                    events=list(stage.audit_event.get("events") or []),
                    rendered_block_ids=[
                        str(e.get("BlockID"))
                        for e in (stage.elements or [])
                        if isinstance(e, dict) and e.get("BlockID")
                    ],
                    displayed_qids_after=sorted(state.displayed_qids),
                    prior_answer_count=len(state.rendered_answers) - len(stage.questions),
                    latency_ms=latency_ms,
                    retries=retries,
                    mode=self.mode,
                )
                result.stages.append(log)
                if on_stage:
                    on_stage(log)
                if state.done:
                    break
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"

        # Long rows for everything answered in THIS run (exclude seeded prior).
        for question, answer in zip(
            state.rendered_questions[seeded_count:], state.rendered_answers[seeded_count:], strict=False,
        ):
            row = {
                "persona_id": persona_id,
                "question_id": str(question.get("id", "")),
                "question_text": str(question.get("text", "")),
                "question_type": str(question.get("type", "")),
                "answer_raw": str(answer.get("answer_raw", "")),
                "answer_value": str(answer.get("answer_value", "")),
                "answer_label": str(answer.get("answer_label", "")),
            }
            if answer.get("answer_status"):
                row["answer_status"] = str(answer.get("answer_status"))
            result.rows.append(row)

        result.embedded_data = dict(state.embedded_data)
        result.randomizer_choices = {k: list(v) for k, v in state.randomizer_choices.items()}
        result.randomization_seed = runtime_seed
        result.displayed_qids = sorted(state.displayed_qids)
        result.done = bool(state.done)
        # The QSF initializes the embedded field to the sentinel "NA" (visible
        # in real exports); only a screenout branch overwrites it with a reason.
        raw_excluded = str(state.embedded_data.get("excluded", "") or "").strip()
        result.excluded = "" if raw_excluded in ("", "NA") else raw_excluded
        result.warnings = list(state.warnings)
        return result


def load_template(path) -> dict:
    from .common import load_json

    return load_json(path)
