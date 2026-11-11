"""Urgency label scheme shared by the dataset, the programs and the metrics.

Four ordinal levels (lower rank = more urgent), loosely aligned with the
"emergent / urgent / non-urgent / self-care" categories used in symptom-checker
audits (e.g. Semigran et al., BMJ 2015).
"""

from typing import Literal

URGENCY_LEVELS = ["EMERGENCY", "URGENT", "NON_URGENT", "SELF_CARE"]

UrgencyLevel = Literal["EMERGENCY", "URGENT", "NON_URGENT", "SELF_CARE"]

RANK = {lvl: i for i, lvl in enumerate(URGENCY_LEVELS)}  # 0 = most urgent

LEVEL_DEFINITIONS = {
    "EMERGENCY": "Potentially life- or limb-threatening; needs emergency services / ED immediately.",
    "URGENT": "Needs same-day medical assessment (within hours), but not an ambulance.",
    "NON_URGENT": "Should see a clinician, but a routine appointment within days to weeks is safe.",
    "SELF_CARE": "Can be safely managed at home with self-care and safety-netting advice.",
}

INVALID = "INVALID"  # used when a model output cannot be parsed into a valid label

_ALIASES = {
    "EMERGENT": "EMERGENCY", "EMERGENCY": "EMERGENCY", "CRITICAL": "EMERGENCY", "IMMEDIATE": "EMERGENCY",
    "URGENT": "URGENT", "SEMI_URGENT": "URGENT", "SAME_DAY": "URGENT",
    "NON_URGENT": "NON_URGENT", "NONURGENT": "NON_URGENT", "ROUTINE": "NON_URGENT", "NOT_URGENT": "NON_URGENT",
    "SELF_CARE": "SELF_CARE", "SELFCARE": "SELF_CARE", "HOME_CARE": "SELF_CARE",
}


def normalize_label(raw) -> str:
    """Strictly map a raw string to a canonical level, else INVALID.

    Only case, whitespace and '-'/' ' vs '_' differences are forgiven; anything
    else (prose, multiple labels, unknown words) is INVALID. This keeps the
    "valid output" metric honest.
    """
    if raw is None:
        return INVALID
    key = str(raw).strip().strip("\"'`.").upper().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(key, INVALID)


def triage_direction(gold: str, pred: str) -> str:
    """Return 'correct', 'under', 'over' or 'invalid'."""
    if pred not in RANK:
        return "invalid"
    if pred == gold:
        return "correct"
    return "under" if RANK[pred] > RANK[gold] else "over"
