# Paper figure captions (20261104-111155_groq-openai-gpt-oss-20b)

**Fig. 1. Experimental workflow.** Synthetic cases are split by scenario; training and
validation cases are used only by the optimisers, and the held-out test set (40 cases)
only for the final evaluation. All five prompting conditions use the same model
(gpt-oss-20b, temperature 0), output schema and metric; only the prompt differs.

**Fig. 2. Accuracy and macro-F1 per prompting condition.** Evaluated on 40 held-out
cases, with 95% bootstrap confidence intervals on accuracy. Accuracy: manual prompt 0.875,
DSPy zero-shot 0.700, BootstrapFewShot 0.750,
MIPROv2 (instr.) 0.625, MIPROv2 (instr.+demos) 0.800.

**Fig. 3. Confusion matrices per prompting condition.** Rows are reference labels, columns
predicted labels, both ordered from most to least urgent; INV = output that failed strict
schema validation. Cells right of the diagonal are under-triage (predicted less urgent than
the reference), cells left of it are over-triage.

**Fig. 4. Triage error profile.** Share of test cases that were under-triaged, over-triaged
or invalid, per condition. Invalid outputs are counted as wrong but not as under- or
over-triage.
