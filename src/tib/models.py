"""Model zoo: random forest, XGBoost, MLP, TabPFN.

All models share one interface: ``fit(X, y)`` and ``predict_proba(X)``, with
inputs already passed through ``Preprocessor``. Hyperparameters are fixed,
reasonable defaults (no per-dataset tuning). This is a deliberate choice for a
small-n study and is discussed as a limitation: Grinsztajn et al. tune heavily.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class Preprocessor:
    """Quantile-transform to Gaussian marginals, then standardize.

    Grinsztajn et al. use a Gaussian quantile transform for neural nets. We apply
    the same transform for every model so that perturbations (rotation, smoothing)
    act on one common representation. Trees are invariant to monotone per-feature
    maps, so this does not change tree results on unperturbed data.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: np.ndarray) -> "Preprocessor":
        nq = max(10, min(1000, X.shape[0]))
        self.qt = QuantileTransformer(
            n_quantiles=nq, output_distribution="normal", random_state=self.seed
        ).fit(X)
        self.sc = StandardScaler().fit(self.qt.transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.sc.transform(self.qt.transform(X))


def make_model(name: str, seed: int, device: str = "cpu"):
    name = name.lower()
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return _QuietMLP(
            MLPClassifier(
                hidden_layer_sizes=(256, 256, 256),
                alpha=1e-4,
                learning_rate_init=1e-3,
                batch_size=128,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=seed,
            )
        )
    if name in ("tabpfn", "tabpfn_v2"):
        # TabPFN v2 (Hollmann et al., Nature 2025), current `tabpfn` package.
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device=device, random_state=seed)
    if name == "tabpfn_v1":
        # Original ICLR 2023 model. Needs the old package: pip install "tabpfn<1".
        # Kept separate because v1 vs v2 tokenization matters for the rotation test.
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device=device, seed=seed)
    raise ValueError(f"unknown model {name!r}")


class _QuietMLP:
    """Wrap sklearn's MLP to silence convergence warnings and handle tiny n."""

    def __init__(self, mlp):
        self.mlp = mlp

    def fit(self, X, y):
        # Early stopping holds out a stratified 10% validation split, which needs
        # at least 2 rows of each class, and a handful to be meaningful. Small n,
        # or a rare class (e.g. after heavy label smoothing), turns it off.
        minority = int(np.bincount(np.asarray(y, dtype=np.int64)).min()) if len(np.unique(y)) > 1 else 0
        if X.shape[0] < 60 or minority < 20:
            self.mlp.set_params(early_stopping=False, max_iter=500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mlp.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.mlp.predict_proba(X)


def model_available(name: str) -> bool:
    """False if the model's package is missing or fails to load.

    Catches more than ImportError on purpose: e.g. XGBoost on macOS raises
    XGBoostError when the OpenMP runtime (libomp) is not installed.
    """
    try:
        make_model(name, seed=0)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[skip] {name}: {type(e).__name__}: {str(e).strip().splitlines()[0]}")
        return False
