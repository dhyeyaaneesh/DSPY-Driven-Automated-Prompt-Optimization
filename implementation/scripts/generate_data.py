"""Generate the synthetic triage dataset and the train/val/test splits.

    python scripts/generate_data.py --config config.yaml
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triage.dataset import SCENARIOS, generate_cases, save_jsonl, split_cases  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))["data"]

    cases = generate_cases(cfg["cases_per_scenario"], seed=cfg["seed"])
    train, val, test = split_cases(cases, tuple(cfg["fractions"]), grouped=cfg["grouped_split"], seed=cfg["seed"])
    out = Path(cfg["dir"])
    save_jsonl(cases, out / "all.jsonl")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        save_jsonl(split, out / f"{name}.jsonl")
        print(f"{name:5s} n={len(split):4d}  scenarios={len({c.scenario_id for c in split}):3d}  "
              f"{dict(sorted(Counter(c.urgency for c in split).items()))}")
    print(f"{len(SCENARIOS)} scenarios, {len(cases)} cases -> {out}/")


if __name__ == "__main__":
    main()
