"""Statistical comparison between conditions on the same held-out cases (paired design)."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from .labels import RANK


def _per_case(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    correct = (gold == pred).astype(float)
    under = np.array([1.0 if (p in RANK and RANK[p] > RANK[g]) else 0.0 for g, p in zip(gold, pred)])
    over = np.array([1.0 if (p in RANK and RANK[p] < RANK[g]) else 0.0 for g, p in zip(gold, pred)])
    return {"accuracy": correct, "under_triage_rate": under, "over_triage_rate": over}


def bootstrap_ci(gold, pred, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CIs for per-case-decomposable metrics."""
    rng = np.random.default_rng(seed)
    pc = _per_case(gold, pred)
    n = len(gold)
    idx = rng.integers(0, n, size=(n_boot, n))
    return {k: (float(v.mean()),
                float(np.quantile(v[idx].mean(1), alpha / 2)),
                float(np.quantile(v[idx].mean(1), 1 - alpha / 2))) for k, v in pc.items()}


def paired_comparison(gold, pred_a, pred_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Compare condition B against A on identical cases.

    - McNemar exact test on correctness (discordant pairs, two-sided binomial).
    - Paired bootstrap CI for the difference (B - A) in accuracy / under- / over-triage.
    """
    a, b = _per_case(gold, pred_a), _per_case(gold, pred_b)
    ca, cb = a["accuracy"].astype(bool), b["accuracy"].astype(bool)
    b01 = int((~ca & cb).sum())   # A wrong, B right
    b10 = int((ca & ~cb).sum())   # A right, B wrong
    p = binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) > 0 else 1.0

    rng = np.random.default_rng(seed)
    n = len(gold)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = {}
    for k in a:
        d = b[k] - a[k]
        boots = d[idx].mean(1)
        diffs[k] = (float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    return {"mcnemar_b01": b01, "mcnemar_b10": b10, "mcnemar_p": float(p), "diff_ci": diffs}
