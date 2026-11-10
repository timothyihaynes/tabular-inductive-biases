"""Synthetic 2x2 design: {smooth, jumpy} x {axis-aligned, oblique}.

P(y = 1 | x) = sigmoid(g(x) / temperature), x ~ N(0, I_d).

* axis-aligned: g depends on x through a sum of functions of single coordinates
  x_1, ..., x_k (the first k features are informative, the rest are noise).
* oblique: the same g applied to z = Q x for a random rotation Q, so every
  feature carries signal and no single coordinate is aligned with it.
* smooth: each component is sin(c * u).
* jumpy: each component is a step function with a few random thresholds.

Because "oblique" is literally "axis-aligned after rotation", the two cells in
a row differ only in orientation, and the two cells in a column differ only in
smoothness. That is what lets us separate the two inductive biases.
"""

from __future__ import annotations

import numpy as np

from .data import Dataset
from .perturb import random_rotation_matrix


def _smooth(u: np.ndarray, freq: float) -> np.ndarray:
    return np.sin(freq * u)


def _jumpy(u: np.ndarray, thresholds: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return levels[np.searchsorted(thresholds, u)]


def make_synthetic(
    n: int,
    d: int = 10,
    k_informative: int = 3,
    smooth: bool = True,
    oblique: bool = False,
    temperature: float = 0.25,
    seed: int = 0,
    n_steps: int = 3,
    freq: float = 1.5,
) -> Dataset:
    """Draw a dataset from one cell of the 2x2 design.

    The function g is determined by ``seed`` alone (not n), so datasets of
    different sizes with the same seed share the same ground truth.
    """
    g_rng = np.random.default_rng(seed)  # ground-truth function
    Q = random_rotation_matrix(d, g_rng)
    comps = []
    for _ in range(k_informative):
        th = np.sort(g_rng.normal(0, 0.8, size=n_steps))
        lv = g_rng.choice([-1.0, 1.0], size=n_steps + 1)
        comps.append((th, lv))

    x_rng = np.random.default_rng([seed, n])  # sample, independent of g
    X = x_rng.standard_normal((n, d))
    Z = X @ Q if oblique else X
    g = np.zeros(n)
    for j, (th, lv) in enumerate(comps):
        g += _smooth(Z[:, j], freq) if smooth else _jumpy(Z[:, j], th, lv)
    g /= np.sqrt(k_informative)
    p = 1.0 / (1.0 + np.exp(-g / temperature))
    y = (x_rng.random(n) < p).astype(np.int64)
    tag = f"syn_{'smooth' if smooth else 'jumpy'}_{'oblique' if oblique else 'axis'}"
    return Dataset(tag, X, y)


CELLS = [
    dict(smooth=True, oblique=False),
    dict(smooth=True, oblique=True),
    dict(smooth=False, oblique=False),
    dict(smooth=False, oblique=True),
]
