"""Turn a finished run into paper-ready tables, statistics and figures.

    python scripts/make_report.py --run results/<run_id>

Writes into <run>/report/:
  results.md                 all tables in Markdown (quick look)
  tables/*.tex               IEEE-ready LaTeX tables (booktabs)
  stats.json                 bootstrap CIs + paired tests vs. the baselines
  figures/*.pdf|png          confusion matrices, metric bars, triage-error bars, cost vs. accuracy
  error_analysis.csv         per-scenario accuracy for every condition
  missed_emergencies.csv     every EMERGENCY case a condition failed to flag
"""
import argparse
import json
import sys
from pathlib import Path

# The report contains κ and Δ; Windows consoles default to cp1252 and can't print them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triage.labels import INVALID, URGENCY_LEVELS  # noqa: E402
from triage.stats import bootstrap_ci, paired_comparison  # noqa: E402

PRETTY = {"manual_raw": "Manual prompt", "dspy_zeroshot": "DSPy zero-shot", "dspy_cot_zeroshot": "DSPy CoT zero-shot",
          "bootstrap_fewshot": "BootstrapFewShot", "mipro_instructions": "MIPROv2 (instr.)",
          "mipro_full": "MIPROv2 (instr.+demos)", "gepa": "GEPA"}
# fixed categorical order (validated palette), assigned by condition, never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7d5fd6"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})


def pretty(tag):
    base, _, seed = tag.partition("_s") if tag.rsplit("_s", 1)[-1].isdigit() else (tag, "", "")
    return PRETTY.get(base, base) + (f" (seed {seed})" if seed else "")


def latex_table(df: pd.DataFrame, caption: str, label: str, bold_best: dict | None = None) -> str:
    cols = "l" + "c" * (len(df.columns))
    body = []
    for idx, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            s = v if isinstance(v, str) else f"{v:.3f}"
            if bold_best and c in bold_best:
                best = df[c].max() if bold_best[c] == "max" else df[c].min()
                if not isinstance(v, str) and np.isclose(v, best):
                    s = f"\\textbf{{{s}}}"
            cells.append(s)
        body.append(f"{idx} & " + " & ".join(cells) + r" \\")
    hdr = " & ".join(["Condition"] + [c.replace("_", r"\_").replace("%", r"\%") for c in df.columns]) + r" \\"
    return "\n".join([r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                      r"\resizebox{\columnwidth}{!}{%", f"\\begin{{tabular}}{{{cols}}}", r"\toprule", hdr,
                      r"\midrule", *body, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baselines", nargs="+", default=["manual_raw", "dspy_zeroshot"])
    args = ap.parse_args()
    run = Path(args.run)
    rep = run / "report"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    (rep / "figures").mkdir(exist_ok=True)

    summary = json.load(open(run / "summary.json", encoding="utf-8"))
    costs = json.load(open(run / "costs.json", encoding="utf-8"))
    preds = {t: pd.read_csv(run / "predictions" / f"{t}.csv") for t in summary}
    tags = list(summary)
    gold = preds[tags[0]]["gold"].tolist()
    for t in tags:  # paired design sanity check
        assert preds[t]["case_id"].tolist() == preds[tags[0]]["case_id"].tolist(), "conditions saw different cases"

    # ---------------- tables ----------------
    main_t = pd.DataFrame({pretty(t): {
        "Accuracy": s["accuracy"], "Macro-F1": s["macro_f1"], "Weighted κ": s["weighted_kappa"],
        "Emerg. sens.": s["emergency_sensitivity"]} for t, s in summary.items()}).T
    safety_t = pd.DataFrame({pretty(t): {
        "Under-triage": s["under_triage_rate"], "Severe under": s["severe_under_triage_rate"],
        "Over-triage": s["over_triage_rate"], "Invalid": s["invalid_label_rate"]} for t, s in summary.items()}).T
    struct_t = pd.DataFrame({pretty(t): {
        "Valid output": s["valid_output_rate"], "Field compl.": s["mean_field_completeness"],
        "Label extractable": s["label_extractable_rate"]} for t, s in summary.items()}).T
    cost_t = pd.DataFrame({pretty(t): {
        "Opt. time (s)": costs[t]["optimize"]["wall_time_s"], "Opt. LM calls": costs[t]["optimize"]["lm_calls"],
        "Opt. tokens (k)": (costs[t]["optimize"]["input_tokens"] + costs[t]["optimize"]["output_tokens"]) / 1e3,
        "Opt. cost ($)": costs[t]["optimize"]["est_cost_usd"],
        "Test tokens/case": (costs[t]["evaluate"]["input_tokens"] + costs[t]["evaluate"]["output_tokens"]) / summary[t]["n"],
        "Latency/case (s)": summary[t]["mean_latency_s"]} for t in tags}).T
    per_class = pd.DataFrame({pretty(t): {f"{lvl} recall": s["per_class"][lvl]["recall"] for lvl in URGENCY_LEVELS}
                              for t, s in summary.items()}).T

    texs = [
        latex_table(main_t, "Triage classification performance on the held-out test set.", "tab:main",
                    {c: "max" for c in main_t.columns}),
        latex_table(safety_t, "Safety-critical error rates (fraction of test cases; lower is better).", "tab:safety",
                    {c: "min" for c in safety_t.columns}),
        latex_table(struct_t, "Structured-output reliability.", "tab:structure", {c: "max" for c in struct_t.columns}),
        latex_table(cost_t, "Optimisation and inference cost.", "tab:cost"),
        latex_table(per_class, "Per-class recall.", "tab:perclass", {c: "max" for c in per_class.columns}),
    ]
    for name, tex in zip(["main", "safety", "structure", "cost", "per_class"], texs):
        (rep / "tables" / f"{name}.tex").write_text(tex, encoding="utf-8")

    # ---------------- statistics ----------------
    stats = {"bootstrap_ci": {}, "paired_vs_baseline": {}}
    for t in tags:
        stats["bootstrap_ci"][t] = bootstrap_ci(gold, preds[t]["pred"].tolist())
    for b in [b for b in args.baselines if b in tags]:
        stats["paired_vs_baseline"][b] = {t: paired_comparison(gold, preds[b]["pred"].tolist(), preds[t]["pred"].tolist())
                                          for t in tags if t != b}
    json.dump(stats, open(rep / "stats.json", "w", encoding="utf-8"), indent=2)

    sig_rows = []
    for b, comps in stats["paired_vs_baseline"].items():
        for t, c in comps.items():
            d = c["diff_ci"]
            sig_rows.append({"Baseline": pretty(b), "Condition": pretty(t),
                             "ΔAcc [95% CI]": f"{d['accuracy'][0]:+.3f} [{d['accuracy'][1]:+.3f}, {d['accuracy'][2]:+.3f}]",
                             "ΔUnder [95% CI]": f"{d['under_triage_rate'][0]:+.3f} [{d['under_triage_rate'][1]:+.3f}, {d['under_triage_rate'][2]:+.3f}]",
                             "McNemar p": f"{c['mcnemar_p']:.4f}"})
    sig_t = pd.DataFrame(sig_rows)

    md = ["# Results\n", f"Run: `{run.name}`  —  n_test = {summary[tags[0]]['n']}\n",
          "## Classification\n", main_t.round(3).to_markdown(), "\n## Safety\n", safety_t.round(3).to_markdown(),
          "\n## Structured output\n", struct_t.round(3).to_markdown(), "\n## Cost\n", cost_t.round(3).to_markdown(),
          "\n## Per-class recall\n", per_class.round(3).to_markdown(),
          "\n## Paired comparisons (bootstrap CI on difference; exact McNemar on correctness)\n",
          sig_t.to_markdown(index=False) if len(sig_t) else "_n/a_"]
    (rep / "results.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------- error analysis ----------------
    ea = pd.DataFrame({pretty(t): preds[t].assign(ok=lambda d: d.gold == d.pred).groupby("scenario_id")["ok"].mean()
                       for t in tags})
    ea.insert(0, "gold", preds[tags[0]].groupby("scenario_id")["gold"].first())
    ea.to_csv(rep / "error_analysis.csv")
    missed = pd.concat([preds[t].query("gold == 'EMERGENCY' and pred != 'EMERGENCY'")
                        .assign(condition=t)[["condition", "case_id", "pred", "symptoms", "errors"]] for t in tags])
    missed.to_csv(rep / "missed_emergencies.csv", index=False)

    # ---------------- figures ----------------
    colors = {t: SERIES[i % len(SERIES)] for i, t in enumerate(tags)}  # >7 conditions: use fewer conditions
    short = ["EMERG", "URG", "NON-URG", "SELF"]

    # 1. confusion matrices (sequential single hue)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
    k = len(tags); ncol = min(3, k); nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.3 * nrow), squeeze=False)
    for ax, t in zip(axes.flat, tags):
        cm = np.array(summary[t]["confusion_matrix"], dtype=float)
        rown = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(rown, cmap=cmap, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if rown[i, j] > 0.55 else INK)
        ax.set_xticks(range(5), short + ["INV"], rotation=45, fontsize=6)
        ax.set_yticks(range(4), short, fontsize=6)
        ax.set_title(pretty(t), fontsize=8, color=INK)
        ax.set_xlabel("Predicted", fontsize=7); ax.set_ylabel("Reference", fontsize=7)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in list(axes.flat)[k:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(rep / "figures" / "confusion_matrices.pdf"); fig.savefig(rep / "figures" / "confusion_matrices.png")
    plt.close(fig)

    # 2. accuracy with 95% bootstrap CI, + macro-F1 (two panels, one axis each)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    y = np.arange(len(tags))
    for ax, key, title in [(axs[0], "accuracy", "Accuracy (95% bootstrap CI)"), (axs[1], "macro_f1", "Macro-F1")]:
        vals = [summary[t][key] for t in tags]
        ax.barh(y, vals, color=[colors[t] for t in tags], height=0.6, edgecolor="white", linewidth=2)
        if key == "accuracy":
            lo = [vals[i] - stats["bootstrap_ci"][t]["accuracy"][1] for i, t in enumerate(tags)]
            hi = [stats["bootstrap_ci"][t]["accuracy"][2] - vals[i] for i, t in enumerate(tags)]
            ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for i, (t, v) in enumerate(zip(tags, vals)):
            x = stats["bootstrap_ci"][t]["accuracy"][2] if key == "accuracy" else v
            ax.text(x + 0.02, i, f"{v:.2f}", va="center", fontsize=7, color=INK)
        ax.set_xlim(0, 1.12); ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    axs[0].set_yticks(y, [pretty(t) for t in tags]); axs[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(rep / "figures" / "accuracy_f1.pdf"); fig.savefig(rep / "figures" / "accuracy_f1.png")
    plt.close(fig)

    # 3. triage error profile: under vs over vs invalid (grouped, not stacked, so each is readable)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    kinds = [("under_triage_rate", "Under-triage", "#eb6834"), ("over_triage_rate", "Over-triage", "#2a78d6"),
             ("invalid_label_rate", "Invalid output", "#6b6a63")]
    w = 0.26
    for j, (key, lab, col) in enumerate(kinds):
        ax.bar(np.arange(len(tags)) + (j - 1) * w, [summary[t][key] for t in tags], width=w - 0.03,
               color=col, label=lab)
    ax.set_xticks(np.arange(len(tags)), [pretty(t) for t in tags], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Fraction of test cases")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(rep / "figures" / "triage_errors.pdf"); fig.savefig(rep / "figures" / "triage_errors.png")
    plt.close(fig)

    # 4. optimisation cost vs accuracy
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for t in tags:
        c = costs[t]["optimize"]["lm_calls"]
        ax.scatter(max(c, 1), summary[t]["accuracy"], s=40, color=colors[t], edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(pretty(t), (max(c, 1), summary[t]["accuracy"]), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK)
    ax.set_xscale("log"); ax.set_xlim(0.6, max(max(costs[t]["optimize"]["lm_calls"] for t in tags), 10) * 12)
    ax.set_xlabel("Optimisation LM calls (log; 1 = none)"); ax.set_ylabel("Test accuracy")
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(rep / "figures" / "cost_vs_accuracy.pdf"); fig.savefig(rep / "figures" / "cost_vs_accuracy.png")
    plt.close(fig)

    print((rep / "results.md").read_text(encoding="utf-8"))
    print(f"\nReport written to {rep}/")


if __name__ == "__main__":
    main()
