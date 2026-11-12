"""Statistical reliability & efficiency metrics shared across all tasks.

Implements:
- 95% bootstrap confidence intervals
- Paired McNemar test between two variants on the same questions
- Accuracy per 1k tokens and evidence-survival rate summaries
"""

from typing import Dict, Sequence, Tuple
import numpy as np
from scipy.stats import binomtest

from src.config import SEED


def bootstrap_accuracy_ci(
    scores: Sequence[int],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    """Return (mean_accuracy, ci_low, ci_high) via non-parametric bootstrap."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot_means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return float(arr.mean()), float(low), float(high)


def paired_mcnemar_test(
    scores_a: Sequence[int],
    scores_b: Sequence[int],
) -> Dict[str, float]:
    """Compute exact two-sided paired McNemar test between Variant A and Variant B."""
    a = np.asarray(scores_a, dtype=int)
    b = np.asarray(scores_b, dtype=int)
    if a.shape != b.shape:
        raise ValueError("Paired McNemar test requires score arrays of identical length.")

    # Discordant pairs: n10 = A right & B wrong; n01 = A wrong & B right
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n_discordant = n10 + n01

    if n_discordant == 0:
        p_val = 1.0
    else:
        p_val = float(binomtest(min(n10, n01), n=n_discordant, p=0.5).pvalue)

    return {
        "a_only_correct": n10,
        "b_only_correct": n01,
        "discordant_total": n_discordant,
        "p_value": p_val,
    }
