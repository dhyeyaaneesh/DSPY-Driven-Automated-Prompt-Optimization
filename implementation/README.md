# DSPy-Driven Automated Prompt Optimisation for Structured Medical Triage

Codebase for the PEGAI mini-project (Kamani, Aneesh, Agarwal, Gagana — PES University).
It implements the controlled experiment from the Rev-1 presentation: **a fixed LLM, a
fixed held-out test set, and different prompts** (one hand-written, the rest produced by
DSPy optimisers). It measures classification accuracy, safety-critical errors
(under- and over-triage), how reliably the model returns valid structured output, and
what the optimisation costs.

> Research prototype on **synthetic** cases. Not a medical device and not clinical advice.

---

## 1. What maps to what (PPT → code)

| PPT item | Implementation |
|---|---|
| Dataset of symptom descriptions with reference triage labels | `triage/dataset.py`: 78 hand-written clinical scenarios × 8 renderings = 624 synthetic cases, 4 ordinal urgency levels |
| Data preparation: optimisation / validation / held-out test | Scenario-grouped stratified split 20/20/60 (`split_cases`). No scenario appears in two splits, so demos can't be near-copies of test cases |
| Baseline prompt | `MANUAL_INSTRUCTION` in `triage/program.py`, used two ways: `manual_raw` (plain chat call, we parse the JSON ourselves) and `dspy_zeroshot` (the same text as an unoptimised DSPy signature) |
| DSPy program: input, output fields, signature | `TriageSignature`: `symptoms → urgency_level (Literal), primary_concern, red_flags[list], recommended_action, confidence[0-1]` |
| Optimiser, e.g. MIPROv2 | `bootstrap_fewshot`, `mipro_instructions` (instruction only, 0-shot), `mipro_full` (instruction + demos), optional `gepa` |
| Scoring metric | `safety_metric`: asymmetric credit. Correct = 1, over-triage by one level = 0.5, any under-triage = 0, plus 15% weight on field completeness. `accuracy_metric` is available as an ablation |
| Same model and inference settings | One `task_lm` (temperature 0) shared by all conditions; `config_used.yaml` is saved with every run |
| Classification metrics | Accuracy, per-class P/R/F1, macro-F1, linear-weighted κ, confusion matrix |
| Under/over-triage | Under-, severe-under- (≥2 levels too low), over-triage rate, emergency sensitivity |
| Structured-output metrics | Valid-output rate (strict pydantic validation), field completeness, label-extractable rate |
| Efficiency metrics | Optimisation wall time, number of LM calls, tokens, estimated USD; per-case inference tokens and latency |
| Comparative analysis | Paired bootstrap 95% CIs on differences plus an exact McNemar test against each baseline (`triage/stats.py`) |

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Choose a task model in `config.yaml` (`task_lm.model`). DSPy uses LiteLLM ids:

| Option | `model` | Key |
|---|---|---|
| Groq (free tier, fast) | `groq/llama-3.1-8b-instant` | `export GROQ_API_KEY=...` |
| Google Gemini (free tier) | `gemini/gemini-2.0-flash` | `export GEMINI_API_KEY=...` |
| Local, no cost | `ollama_chat/llama3.1:8b` + `api_base: http://localhost:11434` | none |
| OpenAI | `openai/gpt-4o-mini` | `export OPENAI_API_KEY=...` |

A **small model** (e.g. Llama-3.1-8B) is the more interesting test subject, because there
is more headroom for prompt optimisation to help. Update `prices` in the config to match
your provider's pricing so the cost columns are correct.

## 3. Run it

```bash
# 1) dataset + splits  (deterministic, seed 42)
python scripts/generate_data.py

# 2) offline smoke test — no API key, fake model, checks every code path (~1 min)
python scripts/run_experiment.py --model mock --max-test 80

# 3) real experiment (start with --max-test 100 while developing, then the full test set)
python scripts/run_experiment.py --config config.yaml

# 4) tables, stats and figures for the paper
python scripts/make_report.py --run results/<run_id>
```

Each run writes the following to `results/<run_id>/`:

```
config_used.yaml          exact settings (put these in the paper's Experimental Setup)
summary.json, costs.json  all metrics and costs per condition
predictions/<cond>.csv    every test case: gold, pred, direction (under/over), validity, raw output
prompts/<cond>.txt        the final prompt: instruction, demos and the full rendered messages (for the appendix)
programs/<cond>.json      the saved, reloadable optimised DSPy program
report/results.md         all tables in Markdown
report/tables/*.tex       IEEE-ready booktabs tables (main, safety, structure, cost, per_class)
report/figures/*.pdf      confusion matrices, accuracy/F1 with CIs, triage-error profile, cost vs. accuracy
report/stats.json         bootstrap CIs plus paired tests vs. manual_raw and dspy_zeroshot
report/error_analysis.csv per-scenario accuracy per condition (find which presentations fail)
report/missed_emergencies.csv  every EMERGENCY case a condition failed to flag
```

**Ignore the mock model's numbers.** They only show that the pipeline works.

### Demo UI

```bash
streamlit run app.py
```

Opens at http://localhost:8501. **Try it** runs any finished condition of a run on your own
symptom text or a test case (set `GROQ_API_KEY` first, or paste it in the sidebar).
**Results** shows the metrics, report figures, per-case predictions and final prompts.
**Figures** shows the four paper figures (workflow, accuracy, confusion matrices, triage
errors) with captions and PDF downloads; a **Build figures** button makes them for a run that
has none yet. The same figures come from the command line:

```bash
python scripts/make_paper_figures.py --run results/<run_id>   # after make_report.py
```

They are written to `results/<run_id>/report/paper_figures/` (PDF + PNG + `captions.md`).

### Free-tier budget run

`config_free.yaml` fits the Groq free tier (1,000 requests/day): 40 test cases, capped
optimisation splits and a small manual MIPROv2 search (~420 calls in total).

```bash
python scripts/run_experiment.py --config config_free.yaml
```

### Rate limits / budget tips
- Set `num_threads: 1` or `2` on free tiers.
- `mipro.auto: light` is the cheapest MIPROv2 setting. Each optimiser costs a few hundred
  to a few thousand calls; check `costs.json` after a small run before going bigger.
- You can run conditions one at a time with `--conditions manual_raw dspy_zeroshot`. Note
  that the report needs all conditions in the same run folder, because the comparison is
  paired.

## 4. Experimental design notes (worth stating in the paper)

- **Controlled variable:** only the prompt changes. The model, temperature (0), adapter,
  signature, test cases and metric are fixed.
- **Two baselines.** `manual_raw` is how people usually prompt an LLM: one message, with
  the JSON parsed by hand. `dspy_zeroshot` uses the same instruction through DSPy's
  adapter. Comparing the two separates the effect of *DSPy's structured formatting* from
  the effect of *optimisation*.
- **No silent repairs.** `json_adapter_fallback: false` means a malformed answer counts as
  invalid instead of being retried, so the validity numbers are honest.
- **Invalid outputs count as wrong** in accuracy and F1. They are reported separately and
  are not counted as under- or over-triage.
- **Grouped split.** Test scenarios are unseen during optimisation, so the test measures
  generalisation, not memorisation of templates. Setting `grouped_split: false` is an
  ablation; expect higher scores from leakage.
- **Paired statistics.** Every condition sees the identical test cases, so use paired tests
  (McNemar) and paired bootstrap CIs, not unpaired ones.
- **Optimiser variance.** Set `optimizer_seeds: [0, 1, 2]` and report mean ± sd. A single
  optimiser run is a common reviewer criticism.

## 5. Suggested ablations (each one is a paragraph or table in the paper)

1. **Metric:** `metric: safety` vs `metric: accuracy`. Does a safety-aware objective trade
   accuracy for fewer under-triage errors?
2. **Instructions vs demos:** `mipro_instructions` vs `mipro_full` vs `bootstrap_fewshot`.
3. **Reasoning:** `reasoning: true` (ChainOfThought) together with `dspy_cot_zeroshot`.
4. **Model size:** repeat with a small and a large model (e.g. 8B vs 70B).
5. **Proposer strength:** `prompt_lm: {model: <stronger model>}` for MIPROv2/GEPA.
6. **Optimiser:** add `gepa`, a reflective optimiser that uses our textual feedback
   (`gepa_feedback_metric`).

## 6. Using a different dataset

`to_examples()` accepts any rows with `text` and `urgency` (one of `EMERGENCY, URGENT,
NON_URGENT, SELF_CARE`), plus optional `case_id`, `scenario_id` and `red_flags`. Write
`train/val/test.jsonl` in that format into `data/`. Options to consider: published
symptom-checker vignettes (e.g. Semigran et al., BMJ 2015, 45 vignettes with
emergent/non-emergent/self-care labels), or MIMIC-IV-ED, which needs PhysioNet
credentialing and mapping of ESI 1–5 to these four levels. Never use identifiable patient
data.

## 7. Limitations (be upfront about these in the paper)

- The cases are synthetic and template-based. The reference labels follow common
  safety-netting rules, written by the authors and **not clinician-validated**. Ideally a
  clinician reviews a sample (for example 50 cases) and you report the agreement.
- Vignettes contain one decisive feature. Real patient messages are noisier and more
  ambiguous.
- Results are specific to the model, optimiser budget and metric that were tested.

## 8. Code map

```
triage/labels.py      urgency scheme, strict label normalisation, under/over logic
triage/dataset.py     scenario bank, renderer, grouped split, I/O
triage/schema.py      output contract (pydantic) + strict JSON parser
triage/program.py     manual instruction, TriageSignature, DSPyTriage, ManualPromptTriage
triage/metrics.py     safety/accuracy optimisation metrics, GEPA feedback, evaluation metrics
triage/conditions.py  builds/optimises each condition; prompt export
triage/evaluate.py    parallel prediction + summaries
triage/stats.py       bootstrap CIs, McNemar, paired differences
triage/lm_setup.py    LM factory, call/token/cost tracking
triage/mock_lm.py     offline fake LM for smoke tests
scripts/              generate_data.py, run_experiment.py, make_report.py
```
