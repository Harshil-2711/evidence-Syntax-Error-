"""
api/routes/evidence.py

Evidence endpoint — GET /workflow/{id}/evidence

The evidence response is the centrepiece of the zero-trust story.

Every action produces an EvidenceTriple that explicitly separates:
  ┌─────────────────────────────────────────────────────────────┐
  │  LAYER 1 — EXECUTOR_CLAIM                                   │
  │  The tool's self-reported outcome. MAY BE FALSE.            │
  │  Example: "Member added successfully" (FALSE_SUCCESS)       │
  ├─────────────────────────────────────────────────────────────┤
  │  LAYER 2 — MACHINE_EVIDENCE                                 │
  │  What the sandbox actually observed.  GROUND TRUTH.         │
  │  Source: always "sandbox.state_store" — never fabricated.   │
  │  Example: member_exists=False  (contradiction!)             │
  ├─────────────────────────────────────────────────────────────┤
  │  LAYER 3 — VERIFICATION_RESULT                              │
  │  Deterministic comparison: expected vs observed.            │
  │  Verdict: PASS | FAIL | INCONCLUSIVE                        │
  │  INVARIANT: INCONCLUSIVE ≠ PASS — never silently promoted.  │
  └─────────────────────────────────────────────────────────────┘

A contradiction (EXECUTOR_CLAIM=success + VERIFICATION_RESULT=FAIL) is
the clearest demonstration that the system cannot be fooled by a lying
executor. The verifier always checks independently.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import _app_state
from core.models import VerificationStatus

router = APIRouter(prefix="/workflow", tags=["Evidence"])


# ===========================================================================
# Schema — the 3-layer EvidenceTriple
# ===========================================================================


class ExecutorClaim(BaseModel):
    """
    Layer 1: The tool's self-reported result.

    WARNING: This is an UNVERIFIED claim by the executor.
    It can be false — see FALSE_SUCCESS injection in scenarios.
    Do NOT trust this alone. Always check machine_evidence + verification_result.
    """

    self_reported_success: bool | None = Field(
        description="What the tool reported. May be false."
    )
    reported_data: dict[str, Any] | None = Field(
        description="Data the tool returned. Unverified."
    )
    failure_mode: str | None = Field(
        description="Detected failure mode (e.g. FALSE_SUCCESS). None if clean execution."
    )
    error_message: str | None
    LAYER: str = Field(
        default="EXECUTOR_CLAIM",
        description="This field identifies this as the EXECUTOR_CLAIM layer.",
    )
    CAUTION: str = Field(
        default=(
            "UNVERIFIED self-report from the executor. "
            "The executor can lie (see FALSE_SUCCESS). "
            "Always check MACHINE_EVIDENCE and VERIFICATION_RESULT."
        )
    )


class MachineEvidence(BaseModel):
    """
    Layer 2: Ground-truth state observed from the sandbox.

    SOURCE GUARANTEE: Evidence always originates from sandbox.state_store.
    It is NEVER generated, inferred, or fabricated by any LLM.
    """

    source: str = Field(
        description="Always 'sandbox.state_store'. Any other value is a bug."
    )
    observed_state: dict[str, Any] = Field(
        description="Actual sandbox state after the action ran."
    )
    expected_state: dict[str, Any] = Field(
        description="Postconditions the action was expected to satisfy."
    )
    collected_at: str = Field(description="ISO 8601 timestamp of evidence collection.")
    LAYER: str = Field(
        default="MACHINE_EVIDENCE",
        description="This field identifies this as the MACHINE_EVIDENCE layer.",
    )
    SOURCE_GUARANTEE: str = Field(
        default=(
            "Evidence originates exclusively from sandbox.state_store. "
            "It is NEVER generated or fabricated by any LLM or agent."
        )
    )


class VerificationDetail(BaseModel):
    """
    Layer 3: Deterministic PASS/FAIL/INCONCLUSIVE verdict.

    The verifier compares MACHINE_EVIDENCE against expected postconditions.
    It is the sole arbiter of whether an action succeeded.

    ZERO-TRUST INVARIANTS (enforced in code, not just policy):
    - INCONCLUSIVE ≠ PASS. Completion is never claimed on inconclusive evidence.
    - Evidence with an 'error' key → INCONCLUSIVE (never PASS).
    - Empty expected_state → INCONCLUSIVE (not PASS by default).
    """

    verdict: Literal["PASS", "FAIL", "INCONCLUSIVE", "PENDING"]
    reasons: list[str] = Field(description="Per-condition check results.")
    LAYER: str = Field(
        default="VERIFICATION_RESULT",
        description="This field identifies this as the VERIFICATION_RESULT layer.",
    )
    ZERO_TRUST_INVARIANT: str = Field(
        default=(
            "INCONCLUSIVE ≠ PASS. "
            "Completion is ONLY claimed when verdict=PASS. "
            "This invariant is enforced in code — not just policy."
        )
    )


class EvidenceTriple(BaseModel):
    """
    The complete 3-layer evidence record for one action attempt.

    This is the core data structure of the Evidence-Gated Self-Healing Agent.
    It makes the zero-trust verification pipeline tangible and auditable.

    contradiction_detected: True when executor_claim says success but
    machine_evidence proves the action did not complete (the verifier
    caught a FALSE_SUCCESS).
    """

    action_id: str
    action_type: str
    action_index: int
    attempt_number: int = Field(description="0=first attempt, 1=first retry, etc.")

    executor_claim: ExecutorClaim
    machine_evidence: MachineEvidence | None
    verification_result: VerificationDetail

    contradiction_detected: bool = Field(
        description=(
            "True when executor_claim.self_reported_success=True "
            "but verification_result.verdict=FAIL. "
            "Demonstrates that the verifier caught the executor lying."
        )
    )


class EvidenceResponse(BaseModel):
    workflow_id: str
    goal: str
    final_status: str
    total_evidence_records: int
    contradictions_detected: int = Field(
        description="How many times an executor lied and was caught."
    )
    evidence: list[EvidenceTriple]
    zero_trust_guarantee: str = Field(
        default=(
            "All evidence in this response originates from sandbox.state_store. "
            "No LLM was consulted during evidence collection or verification. "
            "INCONCLUSIVE is never promoted to PASS."
        )
    )


# ===========================================================================
# Endpoint
# ===========================================================================


@router.get(
    "/{workflow_id}/evidence",
    summary="Get machine-checkable evidence (3-layer zero-trust view)",
    description=(
        "Returns the full evidence record for every action attempt.\n\n"
        "Each record exposes the three layers explicitly:\n"
        "1. **EXECUTOR_CLAIM** — tool self-report (may be false)\n"
        "2. **MACHINE_EVIDENCE** — sandbox ground truth (cannot be fabricated)\n"
        "3. **VERIFICATION_RESULT** — deterministic PASS/FAIL/INCONCLUSIVE\n\n"
        "**contradiction_detected=true** means the verifier caught the executor lying — "
        "the executor claimed success but the sandbox said otherwise.\n\n"
        "**Source guarantee:** `machine_evidence.source` is always `sandbox.state_store`. "
        "No LLM generates or influences evidence."
    ),
    response_model=EvidenceResponse,
)
def get_evidence(workflow_id: str) -> EvidenceResponse:
    record = _app_state.get_record(workflow_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    run = record.result.run
    result = record.result
    action_to_record = {r.action.action_id: r for r in result.records}

    triples: list[EvidenceTriple] = []
    contradictions = 0

    for i, action in enumerate(run.actions):
        rec = action_to_record.get(action.action_id)
        if rec is None:
            continue

        exec_res = action.execution_result or {}
        self_reported = exec_res.get("tool_success")
        failure_mode_raw = exec_res.get("failure_mode")

        # Collect all attempts (historical + final)
        all_snaps = list(rec.attempt_history)
        already_snapshotted = any(s.vresult is rec.vresult for s in all_snaps)
        if not already_snapshotted:
            from agents.coordinator import AttemptSnapshot
            all_snaps.append(AttemptSnapshot(
                attempt_number=rec.attempts,
                outcome=rec.outcome,
                evidence=rec.evidence,
                vresult=rec.vresult,
                recovery=None,
            ))

        for snap in all_snaps:
            vr = snap.vresult
            ev = snap.evidence

            verdict = vr.status.value if vr else "PENDING"
            reasons = vr.reasons if vr else []

            # Use this snapshot's own executor outcome (not the shared final one).
            # action.execution_result is overwritten each retry attempt, so we
            # must read per-attempt data from snap.outcome to correctly display
            # what the executor reported on THIS specific attempt.
            snap_success = snap.outcome.success if snap.outcome else None
            snap_failure_mode = None
            if snap.outcome:
                # The failure_mode is stored on action.execution_result which is
                # the last write — use it only for the LAST snapshot (final attempt).
                # For historical (retried/failed) snapshots, infer from snap.outcome.
                if snap.vresult is rec.vresult:
                    # This is the final attempt
                    snap_failure_mode = str(failure_mode_raw) if failure_mode_raw else None
                else:
                    # Historical attempt — failure_mode came from that attempt's metadata
                    snap_fm = snap.outcome.tool_result.metadata.get("failure_mode") if snap.outcome else None
                    snap_failure_mode = str(snap_fm) if snap_fm else None

            # Contradiction: executor said success, verifier said FAIL (per-attempt)
            is_contradiction = (
                snap_success is True
                and vr is not None
                and vr.status == VerificationStatus.FAIL
            )
            if is_contradiction:
                contradictions += 1

            machine_ev = None
            if ev is not None:
                machine_ev = MachineEvidence(
                    source=ev.source,
                    observed_state=ev.observed_state,
                    expected_state=ev.expected_state,
                    collected_at=ev.timestamp.isoformat(),
                )

            triples.append(EvidenceTriple(
                action_id=action.action_id,
                action_type=action.action_type.value,
                action_index=i,
                attempt_number=snap.attempt_number,
                executor_claim=ExecutorClaim(
                    self_reported_success=snap_success,
                    reported_data=snap.outcome.tool_result.data if snap.outcome else None,
                    failure_mode=snap_failure_mode,
                    error_message=snap.outcome.error if snap.outcome else None,
                ),
                machine_evidence=machine_ev,
                verification_result=VerificationDetail(
                    verdict=verdict,
                    reasons=reasons,
                ),
                contradiction_detected=is_contradiction,
            ))

    return EvidenceResponse(
        workflow_id=workflow_id,
        goal=record.goal,
        final_status=run.status.value,
        total_evidence_records=len(triples),
        contradictions_detected=contradictions,
        evidence=triples,
    )
