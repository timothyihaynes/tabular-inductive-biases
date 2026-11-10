"""Dataset loading and train/test subsampling.

Real data comes from the Grinsztajn et al. (2022) tabular benchmark, which is
hosted on OpenML as benchmark suites. We use the "classification on numerical
features" suite. Synthetic data lives in ``synthetic.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# OpenML suite for the Grinsztajn benchmark, classification with numerical
# features only. Verify against https://github.com/LeoGrin/tabular-benchmark
# if loading fails (suite ids: 337 clf-num, 334 clf-cat, 336 reg-num, 335 reg-cat).
GRINSZTAJN_CLF_NUM_SUITE = 337

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


@dataclass
class Dataset:
    name: str
    X: np.ndarray  # (n, d) float
    y: np.ndarray  # (n,) int labels in {0, 1}

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]


def list_benchmark_tasks(suite_id: int = GRINSZTAJN_CLF_NUM_SUITE) -> pd.DataFrame:
    """Return a table of (task_id, dataset_id, name, n, d) for the suite."""
    import openml

    suite = openml.study.get_suite(suite_id)
    rows = []
    for tid in suite.tasks:
        task = openml.tasks.get_task(tid, download_data=False)
        ds = openml.datasets.get_dataset(
            task.dataset_id, download_data=False, download_qualities=True
        )
        q = ds.qualities or {}
        rows.append(
            dict(
                task_id=tid,
                dataset_id=task.dataset_id,
                name=ds.name,
                n=int(q.get("NumberOfInstances", -1)),
                d=int(q.get("NumberOfFeatures", -1)) - 1,
            )
        )
    return pd.DataFrame(rows).sort_values("n").reset_index(drop=True)


def load_openml(dataset_id: int) -> Dataset:
    """Load an OpenML dataset as a numeric binary classification problem.

    Results are cached as .npz under ``data_cache/`` so experiments can run
    offline after the first download.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"openml_{dataset_id}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return Dataset(str(z["name"]), z["X"], z["y"])

    import openml

    ds = openml.datasets.get_dataset(
        dataset_id, download_data=True, download_qualities=False
    )
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
    X = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    keep = ~np.isnan(X).any(axis=1)
    X, y = X[keep], np.asarray(y)[keep]
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"{ds.name}: expected binary target, got {len(classes)} classes")
    y = (y == classes[1]).astype(np.int64)
    np.savez_compressed(cache, name=ds.name, X=X, y=y)
    return Dataset(ds.name, X, y)


def split_pool_test(
    data: Dataset, n_test: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into a training pool and a fixed, stratified test set."""
    from sklearn.model_selection import train_test_split

    n_test = min(n_test, data.n // 3)
    idx = np.arange(data.n)
    pool, test = train_test_split(
        idx, test_size=n_test, stratify=data.y, random_state=int(rng.integers(2**31))
    )
    return pool, test


def subsample(
    pool: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Stratified subsample of size ``n`` from ``pool``.

    Nested sizes are not enforced; each (n, repeat) draws its own subsample, and
    the same subsample is shared by every model so comparisons are paired.
    """
    from sklearn.model_selection import train_test_split

    if n >= len(pool):
        return pool.copy()
    sub, _ = train_test_split(
        pool, train_size=n, stratify=y[pool], random_state=int(rng.integers(2**31))
    )
    return sub
