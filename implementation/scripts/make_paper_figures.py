"""Build the four paper figures for one run into report/paper_figures/.

    python scripts/make_paper_figures.py --run results/<run_id>

Fig. 1  experimental workflow (drawn here)
Fig. 2  accuracy and macro-F1 per condition       (from make_report.py: accuracy_f1)
Fig. 3  confusion matrices per condition          (from make_report.py: confusion_matrices)
Fig. 4  under-/over-triage and invalid output     (from make_report.py: triage_errors)

Run make_report.py first. Writes PDF (for LaTeX) and PNG, plus captions.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

COPIED = {  # paper name -> figure made by make_report.py
    "fig2_accuracy": "accuracy_f1",
    "fig3_confusion_matrices": "confusion_matrices",
    "fig4_triage_errors": "triage_errors",
}

INK, QUIET, EDGE, ACCENT = "#1f1f1f", "#5f5f5f", "#7a7a7a", "#2a6fdb"


def workflow_figure(out: Path, model: str, n_test: int, n_train: int, n_val: int):
    """Fig. 1: one row of six stages, with the train/val vs test routing marked."""
    plt.rcParams["font.family"] = "serif"
    stages = [
        ("Synthetic\ndataset", ["78 scenarios x 8", "renderings = 624 cases", "4 urgency levels"]),
        ("Scenario-grouped\nsplit", [f"train {n_train}, val {n_val}", f"held-out test {n_test}", "no scenario shared"]),
        ("Five prompting\nconditions", ["Manual prompt", "DSPy zero-shot", "BootstrapFewShot", "MIPROv2 (instr.)", "MIPROv2 (instr.+demos)"]),
        (f"LLM\n{model}", ["same model for all", "temperature 0", "same output schema"]),
        ("Structured\npredictions", ["JSON: urgency, concern,", "red flags, action,", "confidence", "strict schema check"]),
        ("Evaluation &\nstatistics", ["accuracy, macro-F1, κ", "under-/over-triage", "paired bootstrap CI", "McNemar test"]),
    ]
    W, H, gap, x0 = 2.0, 2.1, 0.42, 0.05
    ytop, ybot = 3.0, 3.0 - H          # box top and bottom
    fig, ax = plt.subplots(figsize=(13.0, 4.1))
    xmax = x0 + 6 * W + 5 * gap + 0.05
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.05, 3.55)
    ax.axis("off")
    bx = [x0 + i * (W + gap) for i in range(6)]

    for i, (title, lines) in enumerate(stages):
        x = bx[i]
        key = i == 2  # the only thing that varies between conditions: the prompt
        ax.add_patch(FancyBboxPatch((x, ybot), W, H, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=ACCENT if key else "white", alpha=0.10 if key else 1, ec="none"))
        ax.add_patch(FancyBboxPatch((x, ybot), W, H, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="none", ec=ACCENT if key else EDGE, lw=1.8 if key else 1.1))
        ax.text(x + W / 2, ytop - 0.32, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=INK, linespacing=1.15)
        for j, line in enumerate(lines):
            ax.text(x + W / 2, ytop - 0.82 - j * 0.24, line, ha="center", va="center", fontsize=9, color=QUIET)
        if i < 5:
            yc = ybot + H / 2
            ax.add_patch(FancyArrowPatch((x + W + 0.04, yc), (x + W + gap - 0.04, yc),
                                         arrowstyle="-|>", mutation_scale=13, lw=1.2, color=EDGE))

    # routing drawn under the boxes: train/val reach only the optimisers, test only the final evaluation
    def route(x_from, x_to, y, color, ls, label, label_x, ha):
        ax.plot([x_from, x_from, x_to], [ybot - 0.02, y, y], color=color, lw=1.1, ls=ls)
        ax.add_patch(FancyArrowPatch((x_to, y), (x_to, ybot - 0.02), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.1, color=color, ls=ls))
        ax.text(label_x, y - 0.17, label, ha=ha, va="center", fontsize=9, color=color)

    # test leaves further left and runs lower, so the two routes never cross
    route(bx[1] + W * 0.65, bx[2] + W / 2, ybot - 0.3, ACCENT, "-",
          "train/val: used only by the optimisers", bx[1] + W * 0.65, "left")
    route(bx[1] + W * 0.35, bx[3] + W / 2, ybot - 0.72, EDGE, "--",
          "test: unseen until the final evaluation", bx[1] + W * 0.35, "left")

    ax.text(x0, 3.38, "Only the prompt changes between conditions; model, decoding, test cases and metric are fixed.",
            fontsize=10.5, color=INK, va="center")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig1_workflow.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    src = run / "report" / "figures"
    if not src.exists():
        sys.exit(f"No figures in {src}. Run: python scripts/make_report.py --run {run}")
    out = run / "report" / "paper_figures"
    out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    n_test = next(iter(summary.values()))["n"]
    model = cfg["task_lm"]["model"].split("/")[-1]
    n_train = cfg["data"].get("max_train") or 120
    n_val = cfg["data"].get("max_val") or 120

    workflow_figure(out, model, n_test, n_train, n_val)
    for new, old in COPIED.items():
        for ext in ("pdf", "png"):
            shutil.copyfile(src / f"{old}.{ext}", out / f"{new}.{ext}")

    acc = {s.get("condition", k): s["accuracy"] for k, s in summary.items()}
    captions = f"""# Paper figure captions ({run.name})

**Fig. 1. Experimental workflow.** Synthetic cases are split by scenario; training and
validation cases are used only by the optimisers, and the held-out test set ({n_test} cases)
only for the final evaluation. All five prompting conditions use the same model
({model}, temperature 0), output schema and metric; only the prompt differs.

**Fig. 2. Accuracy and macro-F1 per prompting condition.** Evaluated on {n_test} held-out
cases, with 95% bootstrap confidence intervals on accuracy. Accuracy: manual prompt {acc.get('manual_raw', 0):.3f},
DSPy zero-shot {acc.get('dspy_zeroshot', 0):.3f}, BootstrapFewShot {acc.get('bootstrap_fewshot', 0):.3f},
MIPROv2 (instr.) {acc.get('mipro_instructions', 0):.3f}, MIPROv2 (instr.+demos) {acc.get('mipro_full', 0):.3f}.

**Fig. 3. Confusion matrices per prompting condition.** Rows are reference labels, columns
predicted labels, both ordered from most to least urgent; INV = output that failed strict
schema validation. Cells right of the diagonal are under-triage (predicted less urgent than
the reference), cells left of it are over-triage.

**Fig. 4. Triage error profile.** Share of test cases that were under-triaged, over-triaged
or invalid, per condition. Invalid outputs are counted as wrong but not as under- or
over-triage.
"""
    (out / "captions.md").write_text(captions, encoding="utf-8")
    print(f"Paper figures written to {out}")


if __name__ == "__main__":
    main()
