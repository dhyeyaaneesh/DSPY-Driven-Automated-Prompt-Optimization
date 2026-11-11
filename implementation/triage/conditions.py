"""Experimental conditions: how each program is built / optimised.

All conditions use the same task LM, adapter, signature and starting instruction, so
any difference on the test set is attributable to the prompt (instructions + demos).
"""

from __future__ import annotations

import dspy

from .metrics import METRICS, gepa_feedback_metric
from .program import DSPyTriage, ManualPromptTriage

CONDITION_DESCRIPTIONS = {
    "manual_raw": "Hand-written prompt, single chat call, JSON parsed manually (no DSPy)",
    "dspy_zeroshot": "DSPy Predict with the same manual instruction, no optimisation",
    "dspy_cot_zeroshot": "DSPy ChainOfThought with the manual instruction, no optimisation",
    "bootstrap_fewshot": "BootstrapFewShot: adds up to k bootstrapped + labelled demos",
    "mipro_instructions": "MIPROv2, 0-shot: optimises the instruction only",
    "mipro_full": "MIPROv2: jointly optimises instruction + few-shot demos",
    "gepa": "GEPA: reflective instruction evolution with textual feedback",
}


def build_condition(name: str, cfg: dict, trainset, valset, task_lm, prompt_lm=None, seed: int = 0):
    """Return a runnable (ManualPromptTriage or compiled dspy.Module)."""
    ocfg = cfg.get("optimizers", {})
    metric = METRICS[cfg.get("metric", "safety")]
    reasoning = cfg.get("reasoning", False)
    threads = cfg.get("num_threads", 4)

    if name == "manual_raw":
        return ManualPromptTriage(task_lm)
    if name == "dspy_zeroshot":
        return DSPyTriage(reasoning=False)
    if name == "dspy_cot_zeroshot":
        return DSPyTriage(reasoning=True)

    student = DSPyTriage(reasoning=reasoning)

    if name == "bootstrap_fewshot":
        o = ocfg.get("bootstrap_fewshot", {})
        tp = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=o.get("max_bootstrapped_demos", 4),
                                   max_labeled_demos=o.get("max_labeled_demos", 4))
        return tp.compile(student, trainset=trainset)

    if name in ("mipro_instructions", "mipro_full"):
        o = ocfg.get("mipro", {})
        zero = name == "mipro_instructions"
        # auto: light|medium|heavy uses DSPy's presets. auto: null lets a small budget
        # set num_candidates / num_trials directly (e.g. for free-tier API limits).
        auto = o.get("auto", "light")
        tp = dspy.MIPROv2(metric=metric, prompt_model=prompt_lm or task_lm, task_model=task_lm,
                          auto=auto, num_candidates=None if auto else o.get("num_candidates", 3),
                          num_threads=threads, seed=seed,
                          max_bootstrapped_demos=0 if zero else o.get("max_bootstrapped_demos", 4),
                          max_labeled_demos=0 if zero else o.get("max_labeled_demos", 4), verbose=False)
        return tp.compile(student, trainset=trainset, valset=valset,
                          num_trials=None if auto else o.get("num_trials", 6),
                          minibatch_size=min(o.get("minibatch_size", 35), len(valset)),
                          minibatch_full_eval_steps=o.get("minibatch_full_eval_steps", 5),
                          requires_permission_to_run=False)

    if name == "gepa":
        o = ocfg.get("gepa", {})
        tp = dspy.GEPA(metric=gepa_feedback_metric, auto=o.get("auto", "light"),
                       reflection_lm=prompt_lm or task_lm, num_threads=threads, seed=seed,
                       track_stats=False)
        return tp.compile(student, trainset=trainset, valset=valset)

    raise ValueError(f"Unknown condition {name}")


def export_prompt(runner, example_input: str) -> str:
    """Human-readable dump of the final prompt (instruction, demos, and a rendered call)."""
    if isinstance(runner, ManualPromptTriage):
        return "=== SYSTEM ===\n" + runner.system + f"\n\n=== USER ===\nSymptoms: {example_input}\n"
    parts = []
    adapter = dspy.settings.adapter or dspy.ChatAdapter()
    for pname, pred in runner.named_predictors():
        parts.append(f"##### Predictor: {pname}\n\n=== INSTRUCTION ===\n{pred.signature.instructions}\n")
        parts.append(f"=== {len(pred.demos)} DEMOS ===")
        for i, d in enumerate(pred.demos):
            dd = dict(d) if not isinstance(d, dict) else d
            parts.append(f"--- demo {i + 1} ---\n" + "\n".join(f"{k}: {v}" for k, v in dd.items()
                                                              if k not in ("case_id", "scenario_id", "augmented")))
        msgs = adapter.format(pred.signature, pred.demos, {"symptoms": example_input})
        parts.append("\n=== FULL RENDERED MESSAGES (as sent to the LM) ===")
        for m in msgs:
            parts.append(f"[{m['role'].upper()}]\n{m['content']}\n")
    return "\n".join(parts)
