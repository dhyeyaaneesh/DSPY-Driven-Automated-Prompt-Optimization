"""Run a condition over a test set and turn the per-case records into reports."""

from __future__ import annotations

import random
import re
import time

import dspy
import pandas as pd
from dspy.utils.parallelizer import ParallelExecutor

from .labels import INVALID, triage_direction
from .metrics import classification_report, structure_report
from .program import API_ERRORS, ManualPromptTriage, run_dspy_program

MAX_API_ATTEMPTS = 30


def _wait_seconds(err: Exception, attempt: int) -> float:
    """Use the provider's 'try again in Xs' hint when present, else exponential backoff."""
    m = re.search(r"try again in ([\d.]+)(ms|s)", str(err))
    if m:
        hint = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
    else:
        hint = min(2 ** attempt, 60)
    return hint + 1 + random.random()


def predict_all(runner, examples: list[dspy.Example], num_threads: int = 4) -> list[dict]:
    """`runner` is a ManualPromptTriage or a dspy.Module. Returns one record per example."""
    def call(ex):
        for attempt in range(MAX_API_ATTEMPTS):
            try:
                if isinstance(runner, ManualPromptTriage):
                    return runner(ex.symptoms), None
                return run_dspy_program(runner, ex.symptoms), None
            except API_ERRORS as e:
                last = e
                wait = _wait_seconds(e, attempt)
                if wait > 300:  # a daily quota, not a per-minute one: don't sleep for hours
                    break
                time.sleep(wait)
        return None, last

    def one(ex):
        rec, api_err = call(ex)
        if rec is None:  # provider kept failing: record it, flagged, instead of crashing
            return {
                "case_id": ex.case_id, "scenario_id": ex.scenario_id, "symptoms": ex.symptoms,
                "gold": ex.urgency_level, "pred": INVALID, "direction": None,
                "valid": False, "completeness": 0.0,
                "errors": f"api_error: {type(api_err).__name__}: {str(api_err)[:300]}",
                "primary_concern": None, "red_flags": None, "recommended_action": None,
                "confidence": None, "latency_s": None, "raw": None, "api_error": True,
            }
        c = rec["check"]
        f = rec["fields"] or {}
        return {
            "case_id": ex.case_id, "scenario_id": ex.scenario_id, "symptoms": ex.symptoms,
            "gold": ex.urgency_level, "pred": c["label"],
            "direction": triage_direction(ex.urgency_level, c["label"]),
            "valid": c["valid"], "completeness": c["completeness"], "errors": " | ".join(c["errors"]),
            "primary_concern": f.get("primary_concern"), "red_flags": f.get("red_flags"),
            "recommended_action": f.get("recommended_action"), "confidence": f.get("confidence"),
            "latency_s": round(rec["latency"], 3), "raw": rec["raw"], "api_error": False,
        }

    executor = ParallelExecutor(num_threads=num_threads, max_errors=10**6, disable_progress_bar=False)
    return executor.execute(one, examples)


def summarise(records: list[dict]) -> dict:
    df = pd.DataFrame(records)
    out = classification_report(df["gold"].tolist(), df["pred"].tolist())
    out.update(structure_report([{"valid": v, "completeness": c, "label": p}
                                 for v, c, p in zip(df["valid"], df["completeness"], df["pred"])]))
    out["mean_latency_s"] = float(df["latency_s"].mean())
    out["n_api_errors"] = int(df["api_error"].sum()) if "api_error" in df else 0
    if out["n_api_errors"]:
        print(f"WARNING: {out['n_api_errors']} case(s) failed at the API, not the model; "
              "they are counted as invalid. Re-run with fewer threads or a higher rate limit.")
    return out
