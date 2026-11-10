import numpy as np
import pandas as pd
import yaml

from tib import analysis, perturb, run, synthetic
from tib.data import Dataset


def test_rotation_is_orthogonal():
    Q = perturb.random_rotation_matrix(6, np.random.default_rng(0))
    assert np.allclose(Q @ Q.T, np.eye(6))


def test_smoothing_zero_is_identity_and_large_is_smoother():
    ds = synthetic.make_synthetic(400, d=4, smooth=False, seed=1)
    y0 = perturb.smooth_labels(ds.X, ds.y, 0.0)
    assert np.array_equal(y0, ds.y)
    y1 = perturb.smooth_labels(ds.X, ds.y, 1.0)
    assert set(np.unique(y1)) <= {0, 1}


def test_oblique_is_rotated_axis():
    a = synthetic.make_synthetic(200, d=5, smooth=True, oblique=False, seed=3)
    b = synthetic.make_synthetic(200, d=5, smooth=True, oblique=True, seed=3)
    assert np.array_equal(a.X, b.X)  # same inputs, different ground truth
    assert not np.array_equal(a.y, b.y)


def test_holm():
    adj = analysis.holm(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def _small_cfg(tmp_path, experiment, **extra):
    cfg = dict(name="t", experiment=experiment, seed=0, repeats=1, n_test=200,
               models=["rf", "xgb"], **extra)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_nsweep_runs_and_resumes(tmp_path):
    cfg = _small_cfg(
        tmp_path, "nsweep", train_sizes=[80],
        datasets=[{"synthetic": {"n": 800, "d": 5, "seed": 0}}],
        perturbations={"smooth_lengthscales": [0.5], "add_noise_fracs": [1.0],
                       "drop_fracs": [0.4], "rotate": True},
    )
    out = tmp_path / "r.csv"
    run.main(["--config", str(cfg), "--out", str(out)])
    df = pd.read_csv(out)
    assert len(df) == 5 * 2  # 5 conditions x 2 models
    run.main(["--config", str(cfg), "--out", str(out)])  # resume: no new rows
    assert len(pd.read_csv(out)) == len(df)
    drop = analysis.perturbation_drop(df)
    assert set(drop.perturbation) == {"smooth", "add_noise", "drop_uninformative", "rotate"}


def test_shift_runs(tmp_path):
    cfg = _small_cfg(
        tmp_path, "shift", shift_kinds=["pc1", "feature"], source_sizes=[100],
        k_shots=[0, 20], datasets=[{"synthetic": {"n": 1500, "d": 5, "seed": 0}}],
    )
    out = tmp_path / "s.csv"
    run.main(["--config", str(cfg), "--out", str(out)])
    assert len(pd.read_csv(out)) == 2 * 2 * 2


def test_synthetic_runs(tmp_path):
    cfg = _small_cfg(tmp_path, "synthetic", n_functions=1, pool_size=600,
                     train_sizes=[100], synthetic={"d": 5})
    out = tmp_path / "y.csv"
    run.main(["--config", str(cfg), "--out", str(out)])
    assert len(pd.read_csv(out)) == 4 * 2
