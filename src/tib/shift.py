"""Covariate shift splits and few-shot adaptation (Dr. Zhou's suggestion).

Setup: a model is trained on a *source* region of feature space and tested on a
*target* region. We then add k labelled target examples to the training data
and refit, to measure how quickly each model adapts.

For TabPFN, "refitting" means putting the extra rows in its context, which is
exactly in-context learning. For trees and the MLP it means retraining on
source + k target rows. Optionally the target rows can be up-weighted.
"""

from __future__ import annotations

import numpy as np


def shift_split(
    X: np.ndarray,
    rng: np.random.Generator,
    kind: str = "pc1",
    quantile: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (source_idx, target_idx).

    kind="pc1": project on the first principal component; source = below the
        ``quantile`` cut, target = above. Shifts along the main direction of
        variation, an oblique direction.
    kind="feature": same, but along one randomly chosen original feature, an
        axis-aligned shift. Comparing the two connects shift to orientation.
    """
    Xc = X - X.mean(0)
    if kind == "pc1":
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        s = Xc @ Vt[0]
    elif kind == "feature":
        s = Xc[:, rng.integers(X.shape[1])]
    else:
        raise ValueError(kind)
    cut = np.quantile(s, quantile)
    return np.where(s <= cut)[0], np.where(s > cut)[0]


def target_pool_and_test(
    target_idx: np.ndarray, n_test: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a fixed target test set; the rest is the pool for k-shot rows."""
    perm = rng.permutation(target_idx)
    n_test = min(n_test, len(perm) // 2)
    return perm[n_test:], perm[:n_test]
