import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    values = np.asarray(values, dtype=float)
    point = float(values.mean())
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    if len(a) != len(b):
        raise ValueError("a and b must have the same length for paired bootstrap")

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    point = float(diffs.mean())

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = diffs[idx].mean()

    lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return (point, lo, hi)
