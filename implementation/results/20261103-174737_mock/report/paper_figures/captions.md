# Paper figure captions (20261103-174737_mock)

**Fig. 1. Experimental workflow.** Synthetic cases are split by scenario; training and
validation cases are used only by the optimisers, and the held-out test set (80 cases)
only for the final evaluation. All five prompting conditions use the same model
(mock, temperature 0), output schema and metric; only the prompt differs.

**Fig. 2. Accuracy and macro-F1 per prompting condition.** Evaluated on 80 held-out
cases, with 95% bootstrap confidence intervals on accuracy. Accuracy: manual prompt 0.562,
DSPy zero-shot 0.525, BootstrapFewShot 0.688,
MIPROv2 (instr.) 0.525, MIPROv2 (instr.+demos) 0.762.

**Fig. 3. Confusion matrices per prompting condition.** Rows are reference labels, columns
predicted labels, both ordered from most to least urgent; INV = output that failed strict
schema validation. Cells right of the diagonal are under-triage (predicted less urgent than
the reference), cells left of it are over-triage.

**Fig. 4. Triage error profile.** Share of test cases that were under-triaged, over-triaged
or invalid, per condition. Invalid outputs are counted as wrong but not as under- or
over-triage.
