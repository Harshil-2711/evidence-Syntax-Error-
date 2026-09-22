"""
core/verifier.py

Deterministic verification engine.

Compares expected postconditions (from the Action model) against observed
sandbox state (from Evidence) and produces a VerificationStatus:

  PASS        — All required conditions are satisfied.
  FAIL        — One or more required conditions are contradicted.
  INCONCLUSIVE — Required evidence or state is unavailable.

INVARIANT:
  INCONCLUSIVE must NEVER become PASS automatically.
  Only explicit, machine-observable satisfaction earns a PASS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.models import Evidence, VerificationStatus


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    """
    Immutable result returned by the verification engine.

    Attributes:
        status:   Final verdict.
        reasons:  Human-readable list of individual condition outcomes.
        evidence: The Evidence object that was evaluated.
    """

    status: VerificationStatus
    reasons: list[str]
    evidence: Evidence


# ---------------------------------------------------------------------------
# Condition evaluators
# ---------------------------------------------------------------------------


def _check_condition(key: str, expected_value: Any, observed: dict) -> tuple[VerificationStatus, str]:
    """
    Evaluate a single postcondition key/value pair against the observed state.

    Returns (status, reason_string).

    Rules
    -----
    * If the key is absent from observed → INCONCLUSIVE (can't determine truth).
    * If the observed value matches expected → PASS for this condition.
    * If the observed value exists but doesn't match → FAIL for this condition.
    """
    # Flatten nested lookup: "member_exists" → observed["member_exists"]
    # Supports one level of nesting using dot notation ("data.name")
    parts = key.split(".", 1)
    if len(parts) == 2:
        outer, inner = parts
        outer_val = observed.get(outer)
        if outer_val is None or not isinstance(outer_val, dict):
            return (
                VerificationStatus.INCONCLUSIVE,
                f"Key '{outer}' absent or not a dict in observed state — INCONCLUSIVE",
            )
        actual = outer_val.get(inner)
        present = inner in outer_val
    else:
        actual = observed.get(key)
        present = key in observed

    if not present:
        return (
            VerificationStatus.INCONCLUSIVE,
            f"Key '{key}' absent from observed state — INCONCLUSIVE",
        )

    if actual == expected_value:
        return (
            VerificationStatus.PASS,
            f"'{key}': expected={expected_value!r}, observed={actual!r} → PASS",
        )

    return (
        VerificationStatus.FAIL,
        f"'{key}': expected={expected_value!r}, observed={actual!r} → FAIL",
    )


# ---------------------------------------------------------------------------
# Verification engine
# ---------------------------------------------------------------------------


class VerificationEngine:
    """
    Deterministic, stateless verification engine.

    Usage::

        engine = VerificationEngine()
        result = engine.verify(evidence)
        if result.status == VerificationStatus.PASS:
            ...
    """

    def verify(self, evidence: Evidence) -> VerificationResult:
        """
        Compare evidence.expected_state against evidence.observed_state.

        Algorithm
        ---------
        1. If observed_state contains an "error" key → INCONCLUSIVE immediately.
        2. For each (key, value) in expected_state, evaluate the condition.
        3. Aggregate:
             - Any FAIL → overall FAIL (short-circuits fairness: one contradiction
               is enough to reject the action).
             - Any INCONCLUSIVE (and no FAIL) → overall INCONCLUSIVE.
             - All PASS → overall PASS.
        4. Update the evidence object's verification_status in place and return
           a VerificationResult.

        INVARIANT enforced: INCONCLUSIVE is never promoted to PASS.
        """
        observed = evidence.observed_state
        expected = evidence.expected_state

        # Fast-path: if the collector couldn't observe state, mark INCONCLUSIVE
        if "error" in observed:
            reasons = [f"Evidence collection error: {observed['error']} — INCONCLUSIVE"]
            status = VerificationStatus.INCONCLUSIVE
            evidence.verification_status = status
            return VerificationResult(status=status, reasons=reasons, evidence=evidence)

        # No expected postconditions → trivially INCONCLUSIVE (nothing to check)
        if not expected:
            reasons = ["No expected postconditions defined — INCONCLUSIVE"]
            status = VerificationStatus.INCONCLUSIVE
            evidence.verification_status = status
            return VerificationResult(status=status, reasons=reasons, evidence=evidence)

        reasons: list[str] = []
        has_inconclusive = False
        has_fail = False

        for key, expected_value in expected.items():
            cond_status, reason = _check_condition(key, expected_value, observed)
            reasons.append(reason)
            if cond_status == VerificationStatus.FAIL:
                has_fail = True
            elif cond_status == VerificationStatus.INCONCLUSIVE:
                has_inconclusive = True

        # Aggregate — FAIL beats INCONCLUSIVE beats PASS
        if has_fail:
            status = VerificationStatus.FAIL
        elif has_inconclusive:
            # INVARIANT: INCONCLUSIVE must NEVER become PASS automatically
            status = VerificationStatus.INCONCLUSIVE
        else:
            status = VerificationStatus.PASS

        # Persist verdict back into the evidence object for audit trail
        evidence.verification_status = status

        return VerificationResult(status=status, reasons=reasons, evidence=evidence)
