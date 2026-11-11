"""Optimisation metrics (what DSPy maximises) and evaluation metrics (what the paper reports)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from .labels import INVALID, RANK, URGENCY_LEVELS, normalize_label
from .schema import validate_fields

# --------------------------------------------------------------------------------------
# Optimisation metrics
# --------------------------------------------------------------------------------------
# Asymmetric credit: under-triage is the dangerous error, so it earns nothing; mild
# over-triage earns partial credit. This makes the optimiser *safety-aware*.
SAFETY_CREDIT = {0: 1.0, -1: 0.5, -2: 0.25, -3: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
# key = RANK[pred] - RANK[gold]; negative = predicted more urgent (over-triage)


def _pred_fields(pred) -> dict:
    return {k: getattr(pred, k, None) if not isinstance(pred, dict) else pred.get(k)
            for k in ["urgency_level", "primary_concern", "red_flags", "recommended_action", "confidence"]}


def accuracy_metric(example, pred, trace=None):
    return float(normalize_label(getattr(pred, "urgency_level", None)) == example.urgency_level)


def safety_metric(example, pred, trace=None):
    """Safety-weighted score in [0, 1] combining label credit and structural completeness.

    During bootstrapping (`trace is not None`) DSPy wants a pass/fail signal for whether a
    trace is good enough to become a demo; we only accept exactly-correct, fully valid ones.
    """
    check = validate_fields(_pred_fields(pred))
    label = check["label"]
    if trace is not None:
        return check["valid"] and label == example.urgency_level
    if label == INVALID:
        return 0.0
    credit = SAFETY_CREDIT[RANK[label] - RANK[example.urgency_level]]
    return 0.85 * credit + 0.15 * check["completeness"]


METRICS = {"safety": safety_metric, "accuracy": accuracy_metric}


def gepa_feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
    """Score + natural-language feedback for GEPA's reflective prompt evolution."""
    import dspy
    score = safety_metric(gold, pred)
    check = validate_fields(_pred_fields(pred))
    label = check["label"]
    if label == INVALID or not check["valid"]:
        fb = f"Output was not valid structured output ({'; '.join(check['errors'][:2])}). All five fields are required."
    elif label == gold.urgency_level:
        fb = f"Correct: {label}."
    else:
        kind = "UNDER-TRIAGE (dangerous)" if RANK[label] > RANK[gold.urgency_level] else "over-triage"
        flags = ", ".join(gold.red_flags) if getattr(gold, "red_flags", None) else "none (low-acuity case)"
        fb = (f"Predicted {label} but the reference is {gold.urgency_level}: {kind}. "
              f"Decisive reference red flags: {flags}.")
    return dspy.Prediction(score=score, feedback=fb)

# --------------------------------------------------------------------------------------
# Evaluation metrics
# --------------------------------------------------------------------------------------


def classification_report(gold: list[str], pred: list[str]) -> dict:
    """All headline numbers for one condition. `pred` may contain INVALID."""
    gold, pred = list(gold), list(pred)
    n = len(gold)
    labels_plus = URGENCY_LEVELS + [INVALID]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=URGENCY_LEVELS, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=labels_plus)[: len(URGENCY_LEVELS)]

    under = sum(1 for g, q in zip(gold, pred) if q in RANK and RANK[q] > RANK[g])
    over = sum(1 for g, q in zip(gold, pred) if q in RANK and RANK[q] < RANK[g])
    invalid = sum(1 for q in pred if q == INVALID)
    emerg_idx = [i for i, g in enumerate(gold) if g == "EMERGENCY"]
    emerg_missed = sum(1 for i in emerg_idx if pred[i] != "EMERGENCY")  # includes INVALID
    severe_under = sum(1 for g, q in zip(gold, pred) if q in RANK and RANK[q] - RANK[g] >= 2)

    # Linear-weighted kappa on valid predictions (ordinal agreement)
    valid_idx = [i for i, q in enumerate(pred) if q in RANK]
    kappa = (cohen_kappa_score([RANK[gold[i]] for i in valid_idx], [RANK[pred[i]] for i in valid_idx],
                               weights="linear") if len(valid_idx) > 1 else float("nan"))

    return {
        "n": n,
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=URGENCY_LEVELS, average="macro", zero_division=0),
        "weighted_kappa": kappa,
        "under_triage_rate": under / n,
        "over_triage_rate": over / n,
        "severe_under_triage_rate": severe_under / n,       # >=2 levels too low
        "emergency_sensitivity": 1 - emerg_missed / max(1, len(emerg_idx)),
        "invalid_label_rate": invalid / n,
        "per_class": {lvl: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, lvl in enumerate(URGENCY_LEVELS)},
        "confusion_matrix": cm.tolist(),
        "confusion_labels_rows": URGENCY_LEVELS,
        "confusion_labels_cols": labels_plus,
    }


def structure_report(checks: list[dict]) -> dict:
    n = len(checks)
    return {
        "valid_output_rate": sum(c["valid"] for c in checks) / n,
        "mean_field_completeness": float(np.mean([c["completeness"] for c in checks])),
        "label_extractable_rate": sum(c["label"] != INVALID for c in checks) / n,
    }
