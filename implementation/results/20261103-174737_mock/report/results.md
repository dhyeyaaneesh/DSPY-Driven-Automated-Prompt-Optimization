# Results

Run: `20261103-174737_mock`  —  n_test = 80

## Classification

|                        |   Accuracy |   Macro-F1 |   Weighted κ |   Emerg. sens. |
|:-----------------------|-----------:|-----------:|-------------:|---------------:|
| Manual prompt          |      0.562 |      0.576 |        0.65  |          0.667 |
| DSPy zero-shot         |      0.525 |      0.558 |        0.63  |          0.556 |
| BootstrapFewShot       |      0.688 |      0.693 |        0.731 |          0.704 |
| MIPROv2 (instr.)       |      0.525 |      0.558 |        0.63  |          0.556 |
| MIPROv2 (instr.+demos) |      0.762 |      0.753 |        0.799 |          0.926 |

## Safety

|                        |   Under-triage |   Severe under |   Over-triage |   Invalid |
|:-----------------------|---------------:|---------------:|--------------:|----------:|
| Manual prompt          |          0.15  |              0 |         0.162 |     0.125 |
| DSPy zero-shot         |          0.175 |              0 |         0.188 |     0.112 |
| BootstrapFewShot       |          0.125 |              0 |         0.15  |     0.038 |
| MIPROv2 (instr.)       |          0.175 |              0 |         0.188 |     0.112 |
| MIPROv2 (instr.+demos) |          0.062 |              0 |         0.162 |     0.012 |

## Structured output

|                        |   Valid output |   Field compl. |   Label extractable |
|:-----------------------|---------------:|---------------:|--------------------:|
| Manual prompt          |          0.875 |          0.875 |               0.875 |
| DSPy zero-shot         |          0.888 |          0.888 |               0.888 |
| BootstrapFewShot       |          0.962 |          0.962 |               0.962 |
| MIPROv2 (instr.)       |          0.888 |          0.888 |               0.888 |
| MIPROv2 (instr.+demos) |          0.988 |          0.988 |               0.988 |

## Cost

|                        |   Opt. time (s) |   Opt. LM calls |   Opt. tokens (k) |   Opt. cost ($) |   Test tokens/case |   Latency/case (s) |
|:-----------------------|----------------:|----------------:|------------------:|----------------:|-------------------:|-------------------:|
| Manual prompt          |            0    |               0 |             0     |           0     |            304.725 |              0.005 |
| DSPy zero-shot         |            0    |               0 |             0     |           0     |            727.75  |              0.011 |
| BootstrapFewShot       |            0.03 |               5 |             6.829 |           0     |           1186.97  |              0.009 |
| MIPROv2 (instr.)       |            3.22 |             658 |           434.254 |           0.023 |            727.75  |              0.017 |
| MIPROv2 (instr.+demos) |            3.17 |             682 |           724.736 |           0.038 |           1073.47  |              0.008 |

## Per-class recall

|                        |   EMERGENCY recall |   URGENT recall |   NON_URGENT recall |   SELF_CARE recall |
|:-----------------------|-------------------:|----------------:|--------------------:|-------------------:|
| Manual prompt          |              0.667 |            0.65 |               0.533 |              0.333 |
| DSPy zero-shot         |              0.556 |            0.45 |               0.733 |              0.389 |
| BootstrapFewShot       |              0.704 |            0.8  |               0.667 |              0.556 |
| MIPROv2 (instr.)       |              0.556 |            0.45 |               0.733 |              0.389 |
| MIPROv2 (instr.+demos) |              0.926 |            0.65 |               0.733 |              0.667 |

## Paired comparisons (bootstrap CI on difference; exact McNemar on correctness)

| Baseline       | Condition              | ΔAcc [95% CI]           | ΔUnder [95% CI]         |   McNemar p |
|:---------------|:-----------------------|:------------------------|:------------------------|------------:|
| Manual prompt  | DSPy zero-shot         | -0.037 [-0.200, +0.113] | +0.025 [-0.087, +0.138] |      0.7493 |
| Manual prompt  | BootstrapFewShot       | +0.125 [-0.025, +0.275] | -0.025 [-0.138, +0.087] |      0.1325 |
| Manual prompt  | MIPROv2 (instr.)       | -0.037 [-0.200, +0.113] | +0.025 [-0.087, +0.138] |      0.7493 |
| Manual prompt  | MIPROv2 (instr.+demos) | +0.200 [+0.062, +0.338] | -0.087 [-0.188, +0.013] |      0.0113 |
| DSPy zero-shot | Manual prompt          | +0.037 [-0.113, +0.200] | -0.025 [-0.138, +0.087] |      0.7493 |
| DSPy zero-shot | BootstrapFewShot       | +0.163 [+0.062, +0.263] | -0.050 [-0.125, +0.013] |      0.0023 |
| DSPy zero-shot | MIPROv2 (instr.)       | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |      1      |
| DSPy zero-shot | MIPROv2 (instr.+demos) | +0.237 [+0.113, +0.375] | -0.113 [-0.225, -0.013] |      0.0013 |