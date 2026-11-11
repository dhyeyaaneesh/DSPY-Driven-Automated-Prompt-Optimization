"""Triage programs: the DSPy signature/module that gets optimised, and the hand-written
manual-prompt baseline that bypasses DSPy entirely."""

from __future__ import annotations

import time

import dspy
import litellm

from .labels import LEVEL_DEFINITIONS, UrgencyLevel

# Provider-side failures (rate limits, outages, network). These say nothing about the
# prompt, so they must never be scored as a malformed model answer.
# DSPy >= 3.4 wraps provider errors in its own LM* classes, so both families are listed.
from dspy.utils import exceptions as _dx  # noqa: E402

API_ERRORS = (litellm.exceptions.RateLimitError, litellm.exceptions.APIConnectionError,
              litellm.exceptions.Timeout, litellm.exceptions.ServiceUnavailableError,
              litellm.exceptions.InternalServerError, litellm.exceptions.AuthenticationError,
              _dx.LMRateLimitError, _dx.LMTransportError, _dx.LMTimeoutError,
              _dx.LMServerError, _dx.LMAuthError, _dx.LMBillingError)
from .schema import parse_json_strict, validate_fields

_LEVEL_TEXT = "\n".join(f"- {k}: {v}" for k, v in LEVEL_DEFINITIONS.items())

# --------------------------------------------------------------------------------------
# The *manual* instruction. Deliberately written the way a careful student would write a
# first prompt: clear, but not tuned. It is reused (a) verbatim in the raw baseline and
# (b) as the starting instruction of the DSPy signature, so the optimiser starts from the
# same place as the baseline.
# --------------------------------------------------------------------------------------
MANUAL_INSTRUCTION = f"""You are a medical triage assistant. Read the symptom description and assign an urgency level.

Urgency levels:
{_LEVEL_TEXT}

Identify any red-flag features, state the most likely primary concern, and give a short recommended action.
Also give a confidence between 0 and 1."""


class TriageSignature(dspy.Signature):
    __doc__ = MANUAL_INSTRUCTION

    symptoms: str = dspy.InputField(desc="Free-text description of the patient's symptoms and context")
    urgency_level: UrgencyLevel = dspy.OutputField(desc="Exactly one urgency label")
    primary_concern: str = dspy.OutputField(desc="Most likely clinical concern in a few words")
    red_flags: list[str] = dspy.OutputField(desc="Red-flag features present in the description (empty list if none)")
    recommended_action: str = dspy.OutputField(desc="One-sentence next step for the patient")
    confidence: float = dspy.OutputField(desc="Confidence in the urgency label, between 0 and 1")


class DSPyTriage(dspy.Module):
    """The program that optimisers compile. `reasoning=True` uses ChainOfThought."""

    def __init__(self, reasoning: bool = False):
        super().__init__()
        self.reasoning = reasoning
        self.predict = dspy.ChainOfThought(TriageSignature) if reasoning else dspy.Predict(TriageSignature)

    def forward(self, symptoms: str):
        return self.predict(symptoms=symptoms)


# --------------------------------------------------------------------------------------
# Manual baseline: one plain chat call, JSON parsed by us. This is "prompt engineering
# without DSPy" and gives the most honest measure of structured-output validity.
# --------------------------------------------------------------------------------------
MANUAL_JSON_SUFFIX = """

Respond with ONLY a JSON object with these keys:
{"urgency_level": "EMERGENCY|URGENT|NON_URGENT|SELF_CARE", "primary_concern": "...", "red_flags": ["..."], "recommended_action": "...", "confidence": 0.0}"""


class ManualPromptTriage:
    def __init__(self, lm: dspy.LM, instruction: str = MANUAL_INSTRUCTION):
        self.lm = lm
        self.system = instruction + MANUAL_JSON_SUFFIX

    def __call__(self, symptoms: str) -> dict:
        t0 = time.time()
        out = self.lm(messages=[{"role": "system", "content": self.system},
                                {"role": "user", "content": f"Symptoms: {symptoms}"}])
        raw = out[0] if isinstance(out[0], str) else out[0].get("text", "")
        fields = parse_json_strict(raw)
        return {"raw": raw, "fields": fields, "check": validate_fields(fields), "latency": time.time() - t0}


def run_dspy_program(program: dspy.Module, symptoms: str) -> dict:
    """Run a DSPy program and normalise success/failure into the same record shape."""
    t0 = time.time()
    try:
        pred = program(symptoms=symptoms)
        fields = {k: pred.get(k) for k in ["urgency_level", "primary_concern", "red_flags",
                                            "recommended_action", "confidence"]}
        raw = None
        err = None
    except API_ERRORS:
        raise  # not the model's fault; the caller retries
    except Exception as e:  # AdapterParseError and friends => structurally invalid output
        fields, raw, err = None, getattr(e, "lm_response", None), f"{type(e).__name__}: {str(e)[:300]}"
    check = validate_fields(fields)
    if err:
        check["errors"] = [err] + check["errors"]
    return {"raw": raw, "fields": fields, "check": check, "latency": time.time() - t0}
