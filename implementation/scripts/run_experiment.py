"""Main experiment: build/optimise every condition, evaluate on the held-out test set,
and save per-case predictions, optimised prompts, metrics and cost.

    python scripts/run_experiment.py --config config.yaml
    python scripts/run_experiment.py --config config.yaml --model mock --max-test 60   # offline smoke test
    python scripts/run_experiment.py --conditions manual_raw mipro_full                 # subset
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import dspy
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from triage.conditions import CONDITION_DESCRIPTIONS, build_condition, export_prompt  # noqa: E402
from triage.dataset import load_jsonl, to_examples  # noqa: E402
from triage.evaluate import predict_all, summarise  # noqa: E402
from triage.lm_setup import build_lm, track  # noqa: E402
from triage.program import ManualPromptTriage  # noqa: E402

NEEDS_OPT = {"bootstrap_fewshot", "mipro_instructions", "mipro_full", "gepa"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", help="override task_lm.model (e.g. mock)")
    ap.add_argument("--conditions", nargs="+")
    ap.add_argument("--max-test", type=int)
    ap.add_argument("--out", help="override results dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.model:
        cfg["task_lm"]["model"] = args.model
    conditions = args.conditions or cfg["conditions"]
    max_test = args.max_test or cfg["data"].get("max_test")

    run_id = time.strftime("%Y%m%d-%H%M%S") + "_" + cfg["task_lm"]["model"].replace("/", "-").replace(":", "-")
    out = Path(args.out or cfg["results_dir"]) / run_id
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(exist_ok=True)
    (out / "programs").mkdir(exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config_used.yaml", "w", encoding="utf-8"))

    # ---------------- data ----------------
    ddir = ROOT / cfg["data"]["dir"]
    if not (ddir / "train.jsonl").exists():
        sys.exit(f"No data in {ddir}. Run: python scripts/generate_data.py")
    train = to_examples(load_jsonl(ddir / "train.jsonl"))
    val = to_examples(load_jsonl(ddir / "val.jsonl"))
    test_rows = load_jsonl(ddir / "test.jsonl")
    if max_test and max_test < len(test_rows):
        rng = random.Random(cfg["data"]["seed"])
        test_rows = rng.sample(test_rows, max_test)
    test = to_examples(test_rows)
    # Optional caps on the optimisation splits, for small API budgets.
    rng = random.Random(cfg["data"]["seed"])
    if cfg["data"].get("max_train") and cfg["data"]["max_train"] < len(train):
        train = rng.sample(train, cfg["data"]["max_train"])
    if cfg["data"].get("max_val") and cfg["data"]["max_val"] < len(val):
        val = rng.sample(val, cfg["data"]["max_val"])
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    # ---------------- models ----------------
    task_lm = build_lm(cfg["task_lm"])
    prompt_lm = build_lm({**cfg["task_lm"], **cfg["prompt_lm"]}) if cfg.get("prompt_lm") else None
    dspy.configure(lm=task_lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=cfg.get("json_adapter_fallback", False)))
    prices = cfg.get("prices", {})

    summary, costs = {}, {}
    for cond in conditions:
        seeds = cfg.get("optimizer_seeds", [0]) if cond in NEEDS_OPT else [0]
        for seed in seeds:
            tag = cond if len(seeds) == 1 else f"{cond}_s{seed}"
            print(f"\n================ {tag}: {CONDITION_DESCRIPTIONS[cond]} ================")
            random.seed(seed); np.random.seed(seed)
            cost = {}
            with track("optimize", prices, cost):
                runner = build_condition(cond, cfg, train, val, task_lm, prompt_lm, seed=seed)
            with track("evaluate", prices, cost):
                records = predict_all(runner, test, num_threads=cfg.get("num_threads", 4))

            pd.DataFrame(records).to_csv(out / "predictions" / f"{tag}.csv", index=False)
            (out / "prompts" / f"{tag}.txt").write_text(export_prompt(runner, test[0].symptoms), encoding="utf-8")
            if not isinstance(runner, ManualPromptTriage):
                runner.save(str(out / "programs" / f"{tag}.json"))

            s = summarise(records)
            s["condition"], s["seed"] = cond, seed
            summary[tag], costs[tag] = s, cost
            print(f"acc={s['accuracy']:.3f}  macroF1={s['macro_f1']:.3f}  under={s['under_triage_rate']:.3f}  "
                  f"over={s['over_triage_rate']:.3f}  valid={s['valid_output_rate']:.3f}  "
                  f"opt_calls={cost['optimize']['lm_calls']}  opt_time={cost['optimize']['wall_time_s']}s")
            # save after every condition so a crash / rate-limit never loses finished work
            json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), indent=2)
            json.dump(costs, open(out / "costs.json", "w", encoding="utf-8"), indent=2)
    print(f"\nSaved run to {out}\nNext: python scripts/make_report.py --run {out}")


if __name__ == "__main__":
    main()
