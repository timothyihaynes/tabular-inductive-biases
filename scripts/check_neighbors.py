"""Diagnose a jump in TabPFN accuracy at large n: is it near-duplicate rows?

If test rows have near-identical "siblings" in the training set (e.g. several
rows recorded from the same trial), a model that can look up its training rows,
like TabPFN's attention or a 1-nearest-neighbour rule, gains a lot as n grows,
because more test rows find their sibling. This script reports, for the same
splits the runner uses (seed 0, repeat 0):

  - 1-NN accuracy and AUC (a pure lookup model);
  - the share of test rows whose nearest training row is much closer than usual
    (distance under 10% of the median nearest-neighbour distance).

Usage:  python scripts/check_neighbors.py [dataset_id ...]
Default: eye_movements (44130), electricity (44120), house_16H (44123).
"""
import sys

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from tib.data import load_openml, split_pool_test, subsample
from tib.models import Preprocessor

ids = [int(a) for a in sys.argv[1:]] or [44130, 44120, 44123]
for did in ids:
    data = load_openml(did)
    n_dup = data.n - len(np.unique(data.X, axis=0))
    print(f"== {data.name}: n={data.n}, d={data.d}, exact duplicate rows={n_dup}")
    rng = np.random.default_rng([0, 0])  # same as repeat 0 in the runner
    pool, test = split_pool_test(data, 2000, rng)
    for n in [100, 300, 1000, 3000]:
        if n > len(pool):
            continue
        tr = subsample(pool, data.y, n, rng)
        pre = Preprocessor(seed=0).fit(data.X[tr])
        Xtr, Xte = pre.transform(data.X[tr]), pre.transform(data.X[test])
        ytr, yte = data.y[tr], data.y[test]
        knn = KNeighborsClassifier(n_neighbors=1).fit(Xtr, ytr)
        # 1-NN gives hard labels; use a 15-NN vote for AUC so it has a ranking.
        p = KNeighborsClassifier(n_neighbors=15).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        dist = NearestNeighbors(n_neighbors=1).fit(Xtr).kneighbors(Xte)[0][:, 0]
        close = float((dist < 0.1 * np.median(dist)).mean())
        print(f"   n={n:5d}  1-NN acc={accuracy_score(yte, knn.predict(Xte)):.3f}"
              f"  15-NN AUC={roc_auc_score(yte, p):.3f}"
              f"  test rows with a near-twin in train={close:.1%}")
