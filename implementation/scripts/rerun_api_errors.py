"""Re-predict only the test cases that failed because of the API or network, not the model.

    python scripts/rerun_api_errors.py --run results/<run_id>

Uses the run's saved config and optimised programs, so nothing is re-optimised. Rewrites
predictions/<cond>.csv and the metrics in summary.json. Test-time token counts in
costs.json are not updated (the extra calls are few).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dspy
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triage.evaluate import predict_all, summarise  # noqa: E402
from triage.lm_setup import build_lm  # noqa: E402
from triage.program import DSPyTriage, ManualPromptTriage  # noqa: E402

# Error prefixes written by an API/network failure rather than a model answer.
API_PREFIXES = ("api_error", "LMTransportError", "LMRateLimitError", "LMTimeoutError",
                "LMServerError", "LMAuthError", "LMBillingError")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    cfg = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
    lm = build_lm(cfg["task_lm"])
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=cfg.get("json_adapter_fallback", False)))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))

    for tag in summary:
        path = run / "predictions" / f"{tag}.csv"
        df = pd.read_csv(path)
        bad = df["errors"].fillna("").astype(str).str.startswith(API_PREFIXES)
        if not bad.any():
            continue
        print(f"{tag}: re-running {int(bad.sum())} case(s): {', '.join(df.loc[bad, 'case_id'])}")
        if tag == "manual_raw":
            runner = ManualPromptTriage(lm)
        else:
            runner = DSPyTriage(reasoning=cfg.get("reasoning", False))
            runner.load(str(run / "programs" / f"{tag}.json"))
        exs = [dspy.Example(symptoms=r.symptoms, urgency_level=r.gold, case_id=r.case_id,
                            scenario_id=r.scenario_id).with_inputs("symptoms") for r in df[bad].itertuples()]
        new = pd.DataFrame(predict_all(runner, exs, num_threads=1))
        df = df.set_index("case_id")
        new = new.set_index("case_id")
        for col in new.columns:
            if col not in df.columns:
                df[col] = None
        df = df.astype(object)
        df.loc[new.index, new.columns] = new.astype(object)
        df = df.reset_index()
        df.to_csv(path, index=False)
        s = summarise(df.to_dict("records"))
        s["condition"], s["seed"] = summary[tag].get("condition", tag), summary[tag].get("seed", 0)
        summary[tag] = s
        print(f"  now acc={s['accuracy']:.3f}  valid={s['valid_output_rate']:.3f}  api_errors={s['n_api_errors']}")

    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done. Rebuild the report: python scripts/make_report.py --run", run)


if __name__ == "__main__":
    main()
