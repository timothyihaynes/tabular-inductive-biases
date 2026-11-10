"""The three diagnostic perturbations from Grinsztajn et al. (2022), Sec. 5.

1. ``smooth_labels``: replace training labels by a Gaussian-kernel-smoothed
   version, removing irregular, sharp parts of the target function.
2. ``add_noise_features`` / ``drop_least_important``: change the share of
   uninformative features.
3. ``random_rotation``: rotate the (standardized) feature space, destroying the
   original feature orientation.

Each function only transforms arrays; fitting and scoring happen in ``run.py``.
Inputs are assumed to be already standardized (see ``models.Preprocessor``).
"""

from __future__ import annotations

import numpy as np


# --- 1. Irregularity: target smoothing --------------------------------------

def smooth_labels(
    X_train: np.ndarray, y_train: np.ndarray, lengthscale: float
) -> np.ndarray:
    """Gaussian-kernel (Nadaraya-Watson) smoothing of binary training labels.

    The smoothed value p_i = sum_j K(x_i, x_j) y_j / sum_j K(x_i, x_j) estimates
    P(y=1 | x) with the sharp parts removed; we threshold at 1/2 to get labels.
    Larger ``lengthscale`` = smoother target. ``lengthscale=0`` returns y.

    ``lengthscale`` is relative: the kernel bandwidth is lengthscale times the
    median pairwise distance between training points. Without this, a fixed
    bandwidth means very different amounts of smoothing for different d (in
    high d every other point is far away and each label is just smoothed with
    itself).

    Note: Grinsztajn et al. smooth with a Gaussian kernel on the training set;
    thresholding at 1/2 is our choice for classification and is stated in the
    write-up.
    """
    if lengthscale <= 0:
        return y_train.copy()
    sq = (X_train**2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * X_train @ X_train.T
    d2 = np.maximum(d2, 0)
    med = np.sqrt(np.median(d2[np.triu_indices_from(d2, k=1)]))
    h = lengthscale * med
    K = np.exp(-d2 / (2 * h**2))
    p = K @ y_train / K.sum(1)
    out = (p > 0.5).astype(y_train.dtype)
    # Degenerate case: smoothing wiped out a class. Fall back to original labels.
    return out if len(np.unique(out)) == 2 else y_train.copy()


# --- 2. Uninformative features ----------------------------------------------

def add_noise_features(
    X_train: np.ndarray, X_test: np.ndarray, n_new: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Append ``n_new`` i.i.d. N(0, 1) features (independent of y)."""
    if n_new <= 0:
        return X_train, X_test
    Z_tr = rng.standard_normal((X_train.shape[0], n_new))
    Z_te = rng.standard_normal((X_test.shape[0], n_new))
    return np.hstack([X_train, Z_tr]), np.hstack([X_test, Z_te])


def feature_importance_order(
    X: np.ndarray, y: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Feature indices sorted from least to most important (RF impurity importance)."""
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=250, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    return np.argsort(rf.feature_importances_)


def drop_least_important(
    X_train: np.ndarray, X_test: np.ndarray, order: np.ndarray, frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the ``frac`` least-important features (``order`` from above)."""
    k = int(round(frac * len(order)))
    k = min(k, len(order) - 1)  # always keep at least one feature
    keep = np.sort(order[k:])
    return X_train[:, keep], X_test[:, keep]


# --- 3. Orientation: random rotation ----------------------------------------

def random_rotation_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed random orthogonal matrix via QR with sign correction."""
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))


def random_rotation(
    X_train: np.ndarray, X_test: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    Q = random_rotation_matrix(X_train.shape[1], rng)
    return X_train @ Q, X_test @ Q
