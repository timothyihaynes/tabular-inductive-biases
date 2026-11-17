"""Experiment runner.

    python -m tib.run --config configs/nsweep.yaml
    python -m tib.run --config configs/smoke.yaml      # fast, synthetic only

Every (dataset, repeat, n, condition) draws one training subsample that all
models share, so model comparisons are paired. One row per fitted model is
appended to ``results/<experiment>.csv``; reruns skip rows already present, so a
long run can be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from . import perturb, shift, synthetic
from .data import Dataset, list_benchmark_tasks, load_openml, split_pool_test, subsample
from .models import Preprocessor, make_model, model_available

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- utils

def _sid(s: str) -> int:
    """Stable integer id for a string (Python's hash() is randomized per process)."""
    return zlib.crc32(s.encode()) % 100_000


def score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    p1 = np.clip(proba[:, 1], 1e-7, 1 - 1e-7)
    return dict(
        acc=accuracy_score(y_true, (p1 > 0.5).astype(int)),
        logloss=log_loss(y_true, p1, labels=[0, 1]),
        auc=roc_auc_score(y_true, p1) if len(np.unique(y_true)) == 2 else np.nan,
        brier=float(np.mean((p1 - y_true) ** 2)),
    )


def fit_score(model_name, Xtr, ytr, Xte, yte, seed, device):
    t0 = time.perf_counter()
    m = make_model(model_name, seed=seed, device=device)
    m.fit(Xtr, ytr)
    out = score(yte, m.predict_proba(Xte))
    out["seconds"] = time.perf_counter() - t0
    return out


class ResultWriter:
    def __init__(self, path: Path, key_cols: list[str]):
        self.path, self.key_cols = path, key_cols
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if path.exists():
            old = pd.read_csv(path)
            self.done = set(map(tuple, old[key_cols].astype(str).to_numpy()))

    def seen(self, row: dict) -> bool:
        return tuple(str(row[c]) for c in self.key_cols) in self.done

    def write(self, row: dict) -> None:
        pd.DataFrame([row]).to_csv(
            self.path, mode="a", header=not self.path.exists(), index=False
        )
        self.done.add(tuple(str(row[c]) for c in self.key_cols))


def resolve_datasets(cfg: dict) -> list:
    """Yield callables returning Dataset, from config entries.

    Entries:
      - {openml: 44089}
      - {suite: 337, names: [...]}        # filter benchmark by dataset name
      - {suite: 337, max_datasets: 6}     # smallest datasets first
      - {synthetic: {smooth: true, oblique: false, n: 5000, d: 10, seed: 0}}
    """
    out = []
    for entry in cfg["datasets"]:
        if "openml" in entry:
            out.append(lambda i=entry["openml"]: load_openml(i))
        elif "suite" in entry:
            tab = list_benchmark_tasks(entry["suite"])
            if "names" in entry:
                missing = sorted(set(entry["names"]) - set(tab.name))
                if missing:
                    print(f"[warn] not in suite {entry['suite']}: {missing}")
                tab = tab[tab.name.isin(entry["names"])]
            tab = tab.head(entry.get("max_datasets", len(tab)))
            out += [lambda i=i: load_openml(i) for i in tab.dataset_id]
        elif "synthetic" in entry:
            out.append(lambda kw=entry["synthetic"]: synthetic.make_synthetic(**kw))
        else:
            raise ValueError(entry)
    return out


def conditions(cfg: dict) -> list[tuple[str, float]]:
    """(perturbation, level) pairs. ('none', 0) is always included."""
    p = cfg.get("perturbations", {})
    cs = [("none", 0.0)]
    cs += [("smooth", float(v)) for v in p.get("smooth_lengthscales", [])]
    cs += [("add_noise", float(v)) for v in p.get("add_noise_fracs", [])]
    cs += [("drop_uninformative", float(v)) for v in p.get("drop_fracs", [])]
    if p.get("rotate", False):
        cs.append(("rotate", 1.0))
    return cs


def apply_condition(kind, level, Xtr, ytr, Xte, rng, imp_order):
    if kind == "none":
        return Xtr, ytr, Xte
    if kind == "smooth":
        return Xtr, perturb.smooth_labels(Xtr, ytr, level), Xte
    if kind == "add_noise":
        n_new = int(round(level * Xtr.shape[1]))
        a, b = perturb.add_noise_features(Xtr, Xte, n_new, rng)
        return a, ytr, b
    if kind == "drop_uninformative":
        a, b = perturb.drop_least_important(Xtr, Xte, imp_order, level)
        return a, ytr, b
    if kind == "rotate":
        a, b = perturb.random_rotation(Xtr, Xte, rng)
        return a, ytr, b
    raise ValueError(kind)


def active_models(cfg: dict) -> list[str]:
    ms = []
    for m in cfg["models"]:
        if model_available(m):
            ms.append(m)
    return ms


# --------------------------------------------------------------------------- experiments

def run_nsweep(cfg: dict, out: Path) -> None:
    """Experiment 1: perturbation x training size x model."""
    models = active_models(cfg)
    max_n_model = cfg.get("max_n_per_model", {})
    w = ResultWriter(out, ["dataset", "repeat", "n", "perturbation", "level", "model"])
    for load in resolve_datasets(cfg):
        data: Dataset = load()
        print(f"== {data.name}: n={data.n}, d={data.d}")
        for r in range(cfg["repeats"]):
            rng = np.random.default_rng([cfg["seed"], r])
            pool, test = split_pool_test(data, cfg["n_test"], rng)
            # Importance order computed once per repeat on the whole pool, so the
            # "drop" perturbation removes the same features at every n.
            imp = None
            for n in cfg["train_sizes"]:
                if n > len(pool):
                    continue
                tr = subsample(pool, data.y, n, rng)
                pre = Preprocessor(seed=r).fit(data.X[tr])
                Xtr, Xte = pre.transform(data.X[tr]), pre.transform(data.X[test])
                ytr, yte = data.y[tr], data.y[test]
                for kind, level in conditions(cfg):
                    if kind == "drop_uninformative" and imp is None:
                        Xp = Preprocessor(seed=r).fit(data.X[pool]).transform(data.X[pool])
                        imp = perturb.feature_importance_order(Xp, data.y[pool], seed=r)
                    c_rng = np.random.default_rng([cfg["seed"], r, n, _sid(kind)])
                    Xa, ya, Xb = apply_condition(kind, level, Xtr, ytr, Xte, c_rng, imp)
                    for m in models:
                        row = dict(dataset=data.name, repeat=r, n=n, perturbation=kind,
                                   level=level, model=m, d=Xa.shape[1],
                                   labels_changed=float((ya != ytr).mean()))
                        if w.seen(row) or n > max_n_model.get(m, np.inf):
                            continue
                        row.update(fit_score(m, Xa, ya, Xb, yte, seed=r,
                                             device=cfg.get("device", "cpu")))
                        w.write(row)
                print(f"   repeat {r} n={n} done")


def run_shift(cfg: dict, out: Path) -> None:
    """Experiment 2: covariate shift + k-shot adaptation."""
    models = active_models(cfg)
    w = ResultWriter(out, ["dataset", "repeat", "shift", "n_source", "k", "model"])
    for load in resolve_datasets(cfg):
        data = load()
        print(f"== {data.name}: n={data.n}, d={data.d}")
        for r in range(cfg["repeats"]):
            for kind in cfg["shift_kinds"]:
                rng = np.random.default_rng([cfg["seed"], r, _sid(kind)])
                Xq = Preprocessor(seed=r).fit(data.X).transform(data.X)
                src, tgt = shift.shift_split(Xq, rng, kind, cfg.get("quantile", 0.6))
                tgt_pool, tgt_test = shift.target_pool_and_test(tgt, cfg["n_test"], rng)
                for n_src in cfg["source_sizes"]:
                    src_tr = subsample(src, data.y, n_src, rng)
                    extra_all = rng.permutation(tgt_pool)
                    for k in cfg["k_shots"]:
                        if k > len(extra_all):
                            continue
                        tr = np.concatenate([src_tr, extra_all[:k]])  # nested in k
                        pre = Preprocessor(seed=r).fit(data.X[tr])
                        Xtr, Xte = pre.transform(data.X[tr]), pre.transform(data.X[tgt_test])
                        for m in models:
                            row = dict(dataset=data.name, repeat=r, shift=kind,
                                       n_source=n_src, k=k, model=m)
                            if w.seen(row):
                                continue
                            row.update(fit_score(m, Xtr, data.y[tr], Xte, data.y[tgt_test],
                                                 seed=r, device=cfg.get("device", "cpu")))
                            w.write(row)
                print(f"   repeat {r} shift={kind} done")


def run_synthetic(cfg: dict, out: Path) -> None:
    """Experiment 3: 2x2 smooth/jumpy x axis/oblique, across training sizes."""
    models = active_models(cfg)
    w = ResultWriter(out, ["cell", "fn_seed", "repeat", "n", "model"])
    base = cfg.get("synthetic", {})
    for fn_seed in range(cfg.get("n_functions", 3)):
        for cell in synthetic.CELLS:
            # Large draw from one ground-truth g; split into pool and test.
            big = synthetic.make_synthetic(n=cfg["pool_size"] + cfg["n_test"],
                                           seed=fn_seed, **cell, **base)
            for r in range(cfg["repeats"]):
                rng = np.random.default_rng([cfg["seed"], fn_seed, r])
                pool, test = split_pool_test(big, cfg["n_test"], rng)
                for n in cfg["train_sizes"]:
                    tr = subsample(pool, big.y, n, rng)
                    pre = Preprocessor(seed=r).fit(big.X[tr])
                    Xtr, Xte = pre.transform(big.X[tr]), pre.transform(big.X[test])
                    for m in models:
                        row = dict(cell=big.name, fn_seed=fn_seed, repeat=r, n=n, model=m,
                                   **cell)
                        if w.seen(row):
                            continue
                        row.update(fit_score(m, Xtr, big.y[tr], Xte, big.y[test],
                                             seed=r, device=cfg.get("device", "cpu")))
                        w.write(row)
            print(f"== fn_seed {fn_seed} {big.name} done")


EXPERIMENTS = dict(nsweep=run_nsweep, shift=run_shift, synthetic=run_synthetic)


def main(argv=None) -> None:
    import faulthandler

    faulthandler.enable()  # on a crash (e.g. segfault), print where it happened
    # TabPFN repeats this on every CPU fit with n > 1000; once is enough to know.
    warnings.filterwarnings("once", message="Running on CPU with more than")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="CSV path (default results/<name>.csv)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out) if args.out else ROOT / "results" / f"{cfg['name']}.csv"
    EXPERIMENTS[cfg["experiment"]](cfg, out)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
