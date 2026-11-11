"""The structured output contract every condition must satisfy, and its validator."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

from .labels import URGENCY_LEVELS, normalize_label, INVALID

REQUIRED_FIELDS = ["urgency_level", "primary_concern", "red_flags", "recommended_action", "confidence"]


class TriageOutput(BaseModel):
    urgency_level: str = Field(description="One of " + ", ".join(URGENCY_LEVELS))
    primary_concern: str = Field(min_length=3)
    red_flags: list[str]
    recommended_action: str = Field(min_length=3)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("urgency_level")
    @classmethod
    def _label_ok(cls, v):
        if v not in URGENCY_LEVELS:
            raise ValueError(f"urgency_level must be one of {URGENCY_LEVELS}, got {v!r}")
        return v


def validate_fields(fields: dict | None) -> dict:
    """Check a parsed output dict against the contract.

    Returns: {valid, completeness (0-1), label (canonical or INVALID), errors}
    `label` uses lenient normalisation so we can separately report
    "label extractable" vs "strictly valid structure".
    """
    if not isinstance(fields, dict):
        return {"valid": False, "completeness": 0.0, "label": INVALID, "errors": ["no parsable output"]}

    present = 0
    for f in REQUIRED_FIELDS:
        v = fields.get(f)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if f == "red_flags" and not isinstance(v, list):
            continue
        present += 1
    completeness = present / len(REQUIRED_FIELDS)

    label = normalize_label(fields.get("urgency_level"))
    try:
        TriageOutput(**{k: fields.get(k) for k in REQUIRED_FIELDS})
        return {"valid": True, "completeness": completeness, "label": label, "errors": []}
    except (ValidationError, TypeError) as e:
        errs = [err.get("msg", str(err)) for err in e.errors()] if isinstance(e, ValidationError) else [str(e)]
        return {"valid": False, "completeness": completeness, "label": label, "errors": errs}


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_strict(text: str) -> dict | None:
    """Parse a raw completion as a JSON object. Accepts a single fenced block, nothing more."""
    if text is None:
        return None
    t = text.strip()
    m = _FENCE.fullmatch(t) or (_FENCE.search(t) if t.startswith("```") else None)
    if m:
        t = m.group(1).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
