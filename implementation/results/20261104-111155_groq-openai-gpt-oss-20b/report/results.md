# Results

Run: `20261104-111155_groq-openai-gpt-oss-20b`  —  n_test = 40

## Classification

|                        |   Accuracy |   Macro-F1 |   Weighted κ |   Emerg. sens. |
|:-----------------------|-----------:|-----------:|-------------:|---------------:|
| Manual prompt          |      0.875 |      0.863 |        0.88  |          0.917 |
| DSPy zero-shot         |      0.7   |      0.675 |        0.747 |          0.917 |
| BootstrapFewShot       |      0.75  |      0.804 |        0.89  |          0.833 |
| MIPROv2 (instr.)       |      0.625 |      0.623 |        0.676 |          0.75  |
| MIPROv2 (instr.+demos) |      0.8   |      0.78  |        0.835 |          0.917 |

## Safety

|                        |   Under-triage |   Severe under |   Over-triage |   Invalid |
|:-----------------------|---------------:|---------------:|--------------:|----------:|
| Manual prompt          |          0.1   |          0.025 |         0.025 |      0    |
| DSPy zero-shot         |          0.175 |          0     |         0.125 |      0    |
| BootstrapFewShot       |          0.1   |          0.025 |         0     |      0.15 |
| MIPROv2 (instr.)       |          0.3   |          0.025 |         0.075 |      0    |
| MIPROv2 (instr.+demos) |          0.1   |          0     |         0.1   |      0    |

## Structured output

|                        |   Valid output |   Field compl. |   Label extractable |
|:-----------------------|---------------:|---------------:|--------------------:|
| Manual prompt          |           1    |           1    |                1    |
| DSPy zero-shot         |           1    |           1    |                1    |
| BootstrapFewShot       |           0.85 |           0.85 |                0.85 |
| MIPROv2 (instr.)       |           1    |           1    |                1    |
| MIPROv2 (instr.+demos) |           1    |           1    |                1    |

## Cost

|                        |   Opt. time (s) |   Opt. LM calls |   Opt. tokens (k) |   Opt. cost ($) |   Test tokens/case |   Latency/case (s) |
|:-----------------------|----------------:|----------------:|------------------:|----------------:|-------------------:|-------------------:|
| Manual prompt          |            0    |               0 |             0     |           0     |            471.625 |              1.4   |
| DSPy zero-shot         |            0    |               0 |             0     |           0     |            844.45  |            201.033 |
| BootstrapFewShot       |            2.44 |               3 |             3.645 |           0     |           1178.05  |            473.163 |
| MIPROv2 (instr.)       |        54437.3  |             110 |           129.678 |           0.014 |           1265.97  |            357.382 |
| MIPROv2 (instr.+demos) |        69692.9  |             107 |           139.244 |           0.014 |            844.55  |              5.769 |

## Per-class recall

|                        |   EMERGENCY recall |   URGENT recall |   NON_URGENT recall |   SELF_CARE recall |
|:-----------------------|-------------------:|----------------:|--------------------:|-------------------:|
| Manual prompt          |              0.917 |           0.917 |               0.75  |              0.875 |
| DSPy zero-shot         |              0.917 |           0.667 |               0.75  |              0.375 |
| BootstrapFewShot       |              0.833 |           0.583 |               0.625 |              1     |
| MIPROv2 (instr.)       |              0.75  |           0.5   |               0.625 |              0.625 |
| MIPROv2 (instr.+demos) |              0.917 |           0.833 |               0.75  |              0.625 |

## Paired comparisons (bootstrap CI on difference; exact McNemar on correctness)

| Baseline       | Condition              | ΔAcc [95% CI]           | ΔUnder [95% CI]         |   McNemar p |
|:---------------|:-----------------------|:------------------------|:------------------------|------------:|
| Manual prompt  | DSPy zero-shot         | -0.175 [-0.350, +0.000] | +0.075 [-0.075, +0.225] |      0.0923 |
| Manual prompt  | BootstrapFewShot       | -0.125 [-0.275, +0.000] | +0.000 [-0.100, +0.100] |      0.1797 |
| Manual prompt  | MIPROv2 (instr.)       | -0.250 [-0.425, -0.075] | +0.200 [+0.050, +0.350] |      0.0129 |
| Manual prompt  | MIPROv2 (instr.+demos) | -0.075 [-0.200, +0.050] | +0.000 [-0.100, +0.100] |      0.4531 |
| DSPy zero-shot | Manual prompt          | +0.175 [+0.000, +0.350] | -0.075 [-0.225, +0.075] |      0.0923 |
| DSPy zero-shot | BootstrapFewShot       | +0.050 [-0.150, +0.250] | -0.075 [-0.200, +0.050] |      0.8036 |
| DSPy zero-shot | MIPROv2 (instr.)       | -0.075 [-0.225, +0.075] | +0.125 [-0.025, +0.275] |      0.5488 |
| DSPy zero-shot | MIPROv2 (instr.+demos) | +0.100 [-0.025, +0.225] | -0.075 [-0.175, +0.025] |      0.2891 |