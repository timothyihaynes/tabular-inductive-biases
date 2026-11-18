"""Find a single feature that acts like a lookup key for the label.

Whole-row nearest neighbours can miss this: if one column is an ID (e.g. the
line or trial a row belongs to) and the label is constant within each ID, a
random split puts rows of the same ID in both train and test, and any model
that can match on that one column gets the label almost for free. In the full
20-dimensional distance that column is diluted, so 1-NN on all features misses it.

For each feature alone, this reports the number of distinct values and the
1-NN accuracy using only that feature, on the runner's repeat-0 split at the
largest n. A feature far above the others is the suspect.

Usage:  python scripts/check_single_feature.py [dataset_id] [n]
Default: eye_movements (44130), n = 3000.
"""
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from tib.data import load_openml, split_pool_test, subsample

did = int(sys.argv[1]) if len(sys.argv) > 1 else 44130
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

data = load_openml(did)
names = list(data.features) if data.features else [f"x{j}" for j in range(data.d)]
rng = np.random.default_rng([0, 0])
pool, test = split_pool_test(data, 2000, rng)
tr = subsample(pool, data.y, n, rng)
base = max(data.y[test].mean(), 1 - data.y[test].mean())
print(f"== {data.name}, n={n}; majority-class accuracy = {base:.3f}")
rows = []
for j in range(data.d):
    xtr, xte = data.X[tr, j : j + 1], data.X[test, j : j + 1]
    acc = accuracy_score(
        data.y[test], KNeighborsClassifier(n_neighbors=1).fit(xtr, data.y[tr]).predict(xte)
    )
    share = np.isin(xte[:, 0], xtr[:, 0]).mean()  # test values seen exactly in train
    rows.append((acc, names[j], len(np.unique(data.X[:, j])), share))
for acc, name, k, share in sorted(rows, reverse=True):
    print(f"   {name:>24s}  distinct={k:6d}  test value seen in train={share:6.1%}  1-NN acc={acc:.3f}")
