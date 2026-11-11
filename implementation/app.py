"""PEGAI triage demo UI.

    streamlit run app.py

Page 1 "Try it": type symptoms, pick a prompt condition from a finished run, see the triage.
Page 2 "Results": metrics, costs, figures, per-case predictions and the final prompts of a run.

Research prototype on synthetic cases. Not a medical device and not clinical advice.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import dspy
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from triage.evaluate import _wait_seconds  # noqa: E402
from triage.labels import URGENCY_LEVELS  # noqa: E402
from triage.lm_setup import build_lm  # noqa: E402
from triage.program import API_ERRORS, DSPyTriage, ManualPromptTriage, run_dspy_program  # noqa: E402

RESULTS = ROOT / "results"
DATA = ROOT / "data"

NAMES = {
    "manual_raw": "Manual prompt",
    "dspy_zeroshot": "DSPy zero-shot",
    "dspy_cot_zeroshot": "DSPy CoT zero-shot",
    "bootstrap_fewshot": "BootstrapFewShot",
    "mipro_instructions": "MIPROv2 (instr.)",
    "mipro_full": "MIPROv2 (instr.+demos)",
    "gepa": "GEPA",
}
LEVEL_STYLE = {  # colour, label shown to the user
    "EMERGENCY": ("#c62828", "EMERGENCY: call emergency services now"),
    "URGENT": ("#ef6c00", "URGENT: be seen today"),
    "NON_URGENT": ("#f9a825", "NON-URGENT: book a routine appointment"),
    "SELF_CARE": ("#2e7d32", "SELF-CARE: manage at home"),
}

st.set_page_config(page_title="PEGAI Triage", page_icon="🩺", layout="wide")


# ----------------------------------------------------------------------------- helpers
def show_table(df: pd.DataFrame, max_height: int | None = None):
    """Plain HTML table. st.dataframe/st.bar_chart need pyarrow, whose DLL Windows
    Smart App Control blocks on some machines, so the app avoids them."""
    html = df.to_html(index=False, escape=True, na_rep="-", border=0,
                      float_format=lambda x: f"{x:.3f}", classes="pegai")
    style = f"max-height:{max_height}px;overflow:auto;" if max_height else "overflow-x:auto;"
    st.markdown(
        "<style>table.pegai{border-collapse:collapse;width:100%;font-size:0.85rem}"
        "table.pegai th,table.pegai td{padding:6px 10px;border-bottom:1px solid rgba(128,128,128,.3);"
        "text-align:left;vertical-align:top}table.pegai th{position:sticky;top:0;"
        "background:var(--background-color,#0e1117)}</style>"
        f"<div style='{style}'>{html}</div>", unsafe_allow_html=True)


def bar_chart(df: pd.DataFrame, cols: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    d = df.set_index("Condition")[cols].astype(float)
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.55 * len(d)))
    y = np.arange(len(d))
    h = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.barh(y + i * h, d[c], height=h, label=c)
    ax.set_yticks(y + h * (len(cols) - 1) / 2, d.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def list_runs() -> list[Path]:
    if not RESULTS.exists():
        return []
    runs = [p for p in RESULTS.iterdir() if (p / "config_used.yaml").exists() and (p / "summary.json").exists()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def load_cfg(run: Path) -> dict:
    return yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def conditions_in(run: Path) -> list[str]:
    """Conditions that finished in this run and can be re-run from the UI."""
    done = load_json(run / "summary.json").keys()
    return [c for c in done if c == "manual_raw" or (run / "programs" / f"{c}.json").exists()]


@st.cache_resource(show_spinner=False)
def get_lm(lm_cfg_json: str):
    return build_lm(json.loads(lm_cfg_json))


@st.cache_resource(show_spinner=False)
def get_program(path: str, reasoning: bool):
    prog = DSPyTriage(reasoning=reasoning)
    prog.load(path)
    return prog


@st.cache_data(show_spinner=False)
def example_cases() -> pd.DataFrame:
    p = DATA / "test.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["case_id", "text", "urgency"])
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def triage(run: Path, cond: str, symptoms: str) -> dict:
    cfg = load_cfg(run)
    lm = get_lm(json.dumps(cfg["task_lm"], sort_keys=True))
    adapter = dspy.ChatAdapter(use_json_adapter_fallback=cfg.get("json_adapter_fallback", False))
    # dspy.context (not dspy.configure): Streamlit runs each rerun on a different thread.
    with dspy.context(lm=lm, adapter=adapter):
        for attempt in range(6):
            try:
                if cond == "manual_raw":
                    return ManualPromptTriage(lm)(symptoms)
                prog = get_program(str(run / "programs" / f"{cond}.json"), cfg.get("reasoning", False))
                return run_dspy_program(prog, symptoms)
            except API_ERRORS as e:
                last = e
                wait = _wait_seconds(e, attempt)
                if wait > 60:
                    break
                import time
                time.sleep(wait)
    raise last


def show_result(rec: dict, gold: str | None = None):
    c, f = rec["check"], rec["fields"] or {}
    label = c["label"]
    colour, text = LEVEL_STYLE.get(label, ("#616161", f"{label}: output could not be parsed"))
    st.markdown(
        f"<div style='background:{colour};color:white;padding:14px 18px;border-radius:8px;"
        f"font-size:1.25rem;font-weight:600'>{text}</div>", unsafe_allow_html=True)
    if gold:
        if gold == label:
            st.success(f"Matches the reference label ({gold}).")
        else:
            st.warning(f"Reference label is **{gold}**.")
    a, b = st.columns(2)
    a.markdown(f"**Primary concern:** {f.get('primary_concern') or '-'}")
    a.markdown(f"**Recommended action:** {f.get('recommended_action') or '-'}")
    flags = f.get("red_flags") or []
    a.markdown("**Red flags:** " + (", ".join(map(str, flags)) if flags else "none"))
    conf = f.get("confidence")
    b.metric("Confidence", f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "-")
    b.metric("Valid structured output", "yes" if c["valid"] else "no")
    b.caption(f"Latency {rec['latency']:.2f}s")
    if c["errors"]:
        with st.expander("Validation errors"):
            st.write(c["errors"])
    if rec.get("raw"):
        with st.expander("Raw model output"):
            st.code(rec["raw"])


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🩺 PEGAI Triage")
runs = list_runs()
if not runs:
    st.sidebar.error("No finished runs in results/. Run scripts/run_experiment.py first.")
    st.stop()

run = st.sidebar.selectbox("Experiment run", runs, format_func=lambda p: p.name)
cfg = load_cfg(run)
model = cfg["task_lm"]["model"]
st.sidebar.caption(f"Model: `{model}`  \nMetric: `{cfg.get('metric')}`")

needs_key = model.startswith("groq/") and not os.environ.get("GROQ_API_KEY")
if needs_key:
    key = st.sidebar.text_input("GROQ_API_KEY", type="password",
                                help="Only kept in this app's memory; not written to disk.")
    if key:
        os.environ["GROQ_API_KEY"] = key
        needs_key = False
if model == "mock":
    st.sidebar.info("Mock run: answers come from a fake offline model and mean nothing.")

page = st.sidebar.radio("Page", ["Try it", "Results", "Figures"])
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype on synthetic cases. **Not** a medical device and **not** clinical advice.")


# ----------------------------------------------------------------------------- page: try it
if page == "Try it":
    st.header("Try a triage")
    conds = conditions_in(run)
    if not conds:
        st.warning("This run has no reloadable conditions.")
        st.stop()

    ex = example_cases()
    left, right = st.columns([2, 1])
    with right:
        pick = st.selectbox("Load a test case (optional)", ["-"] + ex["case_id"].tolist()) if len(ex) else "-"
    gold = None
    default = ""
    if pick != "-":
        row = ex[ex["case_id"] == pick].iloc[0]
        default, gold = row["text"], row["urgency"]
    with left:
        symptoms = st.text_area("Symptoms", value=default, height=130,
                                placeholder="e.g. 58-year-old man, crushing chest pain for 20 minutes, sweating")

    mode = st.radio("Prompt", ["One condition", "Compare all conditions"], horizontal=True)
    if mode == "One condition":
        cond = st.selectbox("Condition", conds, format_func=lambda c: NAMES.get(c, c),
                            index=len(conds) - 1)
    else:
        st.caption(f"Makes {len(conds)} model calls.")

    if st.button("Run triage", type="primary", disabled=not symptoms.strip() or needs_key):
        try:
            if mode == "One condition":
                with st.spinner("Asking the model..."):
                    show_result(triage(run, cond, symptoms), gold)
            else:
                rows = []
                for c in conds:
                    with st.spinner(f"{NAMES.get(c, c)}..."):
                        r = triage(run, c, symptoms)
                    f = r["fields"] or {}
                    rows.append({"Condition": NAMES.get(c, c), "Urgency": r["check"]["label"],
                                 "Valid": r["check"]["valid"], "Primary concern": f.get("primary_concern"),
                                 "Recommended action": f.get("recommended_action"),
                                 "Latency (s)": round(r["latency"], 2)})
                df = pd.DataFrame(rows)
                if gold:
                    df["Correct"] = df["Urgency"] == gold
                    st.caption(f"Reference label: **{gold}**")
                show_table(df)
        except API_ERRORS as e:
            st.error(f"The model API refused the call ({type(e).__name__}). "
                     f"If this is a rate limit, wait a minute and try again.\n\n{str(e)[:300]}")
    if needs_key:
        st.info("Enter a GROQ_API_KEY in the sidebar to run this model.")


# ----------------------------------------------------------------------------- page: results
elif page == "Results":
    st.header(f"Results: {run.name}")
    summary = load_json(run / "summary.json")
    costs = load_json(run / "costs.json")
    if model == "mock":
        st.warning("These numbers come from the mock model. They only show that the pipeline works.")

    rows = []
    for tag, s in summary.items():
        cst = costs.get(tag, {})
        rows.append({
            "Condition": NAMES.get(s.get("condition", tag), tag),
            "Accuracy": s.get("accuracy"), "Macro-F1": s.get("macro_f1"),
            "Under-triage": s.get("under_triage_rate"), "Over-triage": s.get("over_triage_rate"),
            "Emergency sens.": s.get("emergency_sensitivity"), "Valid output": s.get("valid_output_rate"),
            "API errors": s.get("n_api_errors", 0),
            "Opt. calls": cst.get("optimize", {}).get("lm_calls"),
            "Opt. cost ($)": cst.get("optimize", {}).get("est_cost_usd"),
        })
    df = pd.DataFrame(rows)
    st.subheader("Main metrics")
    n_test = next(iter(summary.values()), {}).get("n", "?")
    st.caption(f"{n_test} test cases. Lower is better for under/over-triage.")
    show_table(df)

    if len(df):
        st.subheader("Accuracy vs. under-triage")
        bar_chart(df, ["Accuracy", "Under-triage"])

    st.caption("The workflow diagram, accuracy, confusion-matrix and error figures are on the **Figures** page.")

    st.subheader("Per-case predictions")
    preds = sorted((run / "predictions").glob("*.csv"))
    if preds:
        p = st.selectbox("Condition", preds, format_func=lambda x: NAMES.get(x.stem, x.stem))
        pdf = pd.read_csv(p)
        only = st.radio("Show", ["All", "Wrong only", "Under-triage only"], horizontal=True)
        if only == "Wrong only":
            pdf = pdf[pdf["gold"] != pdf["pred"]]
        elif only == "Under-triage only":
            pdf = pdf[pdf["direction"] == "under"]
        cols = [c for c in ["case_id", "gold", "pred", "direction", "valid", "symptoms",
                            "primary_concern", "recommended_action", "errors"] if c in pdf]
        show_table(pdf[cols], max_height=450)
        st.caption(f"{len(pdf)} rows")

    st.subheader("Final prompts")
    prompts = sorted((run / "prompts").glob("*.txt"))
    if prompts:
        p = st.selectbox("Prompt of", prompts, format_func=lambda x: NAMES.get(x.stem, x.stem), key="prompt")
        st.code(p.read_text(encoding="utf-8"), language=None)


# ----------------------------------------------------------------------------- page: figures
else:
    import re
    import subprocess

    st.header("Figures")
    if model == "mock":
        st.warning("Mock run: these figures only show that the pipeline works.")
    pf = run / "report" / "paper_figures"
    figures = [  # file stem, title
        ("fig1_workflow", "Overall experimental workflow"),
        ("fig2_accuracy", "Accuracy comparison"),
        ("fig3_confusion_matrices", "Confusion matrices"),
        ("fig4_triage_errors", "Under-triage and over-triage comparison"),
    ]

    if not all((pf / f"{stem}.png").exists() for stem, _ in figures):
        st.info("The figures for this run have not been built yet.")
        if st.button("Build figures", type="primary"):
            with st.spinner("Building the report and figures (no API calls)..."):
                for script in ("make_report.py", "make_paper_figures.py"):
                    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--run", str(run)],
                                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       env={**os.environ, "PYTHONUTF8": "1"}, cwd=str(ROOT))
                    if r.returncode:
                        st.error(f"{script} failed:\n\n{r.stderr[-1500:]}")
                        st.stop()
            st.rerun()
        st.stop()

    # captions.md holds one bold "Fig. N. Title." paragraph per figure
    caps = {}
    cap_file = pf / "captions.md"
    if cap_file.exists():
        for m in re.finditer(r"\*\*Fig\. (\d)\.[^*]*\*\*(.*?)(?=\n\n\*\*Fig\.|\Z)", cap_file.read_text(encoding="utf-8"), re.S):
            caps[int(m.group(1))] = " ".join(m.group(2).split())

    for i, (stem, title) in enumerate(figures, start=1):
        st.subheader(f"Figure {i}. {title}")
        st.image(str(pf / f"{stem}.png"), use_container_width=True)
        if caps.get(i):
            st.caption(caps[i])
        pdf = pf / f"{stem}.pdf"
        if pdf.exists():
            st.download_button(f"Download Figure {i} (PDF)", pdf.read_bytes(), file_name=f"{stem}.pdf",
                               mime="application/pdf", key=f"dl-{stem}")
        st.markdown("---")
