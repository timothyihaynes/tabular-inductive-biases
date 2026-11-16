# Which inductive biases matter at small n?

STATS 295 final project (UCI, Fall 2026), Timothy Haynes.

Grinsztajn et al. (NeurIPS 2022) argue that tree ensembles beat neural nets on
tabular data because of three inductive biases: they fit **irregular** targets,
they are **robust to uninformative features**, and they **depend on the
feature orientation** (they are not rotation-invariant). Their benchmark has
medium-sized data. TabPFN (Hollmann et al., ICLR 2023) lacks two of those
biases, yet it wins below about 1,000 rows. This project asks:

1. **How does training-set size change which inductive biases matter?**
   We rerun the three diagnostics at n = 100 … 3,000 and track how much each
   perturbation hurts each model (`configs/nsweep.yaml`).
2. **What happens under distribution shift, and how fast do models adapt when a
   few labelled target examples arrive?** TabPFN adapts by putting new rows in its
   context. Trees and the MLP must be retrained (`configs/shift.yaml`).
3. **Is the tree advantage about sharp jumps or axis alignment?** A synthetic
   2×2 design separates the two (`configs/synthetic.yaml`).

## Setup

```bash
git clone <this repo> && cd tabular-inductive-biases
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tabpfn,dev]"
pytest -q                       # ~15 s, no downloads
```

**macOS notes.** Use Python 3.10 or newer (the built-in 3.9 is too old;
`uv venv --python 3.12` is an easy way to get one). XGBoost needs the OpenMP
runtime: `brew install libomp`. If a model fails to load, the runner prints
`[skip]` and continues without it. If `python -m tib.run` says
`No module named 'tib'` (this happens with editable installs inside an
iCloud-synced folder such as ~/Documents), run
`echo 'export PYTHONPATH="'$PWD'/src"' >> .venv/bin/activate` and re-activate,
or keep the project outside iCloud.

TabPFN downloads its weights from Hugging Face on first use. Recent `tabpfn`
versions may ask you to accept the model license or log in once. It runs on CPU
for small n, and `device: mps` (Apple silicon) or `cuda` is faster.

## Running

```bash
python -m tib.run --config configs/smoke.yaml       # ~1-2 min, synthetic only
python scripts/list_datasets.py                     # benchmark datasets (OpenML)
python -m tib.run --config configs/nsweep.yaml      # experiment 1
python -m tib.run --config configs/shift.yaml       # experiment 2
python -m tib.run --config configs/synthetic.yaml   # experiment 3
python -m tib.analysis results/nsweep.csv           # table and figure
```

Results are appended row by row to `results/<name>.csv`. A rerun skips rows
that are already there, so you can stop a long run and resume it later. Start
with a small `repeats` and `max_datasets` to check timings, then scale up.

## Design choices to state in the paper

- **Paired comparisons.** Every model sees the same training subsample and the
  same test set, for each (dataset, repeat, n, perturbation).
- **Replication unit = dataset.** Repeats are averaged within a dataset first,
  then CIs and Wilcoxon tests are computed across datasets. Use Holm-Bonferroni
  for multiple comparisons, as the TabPFN paper does.
- **No per-dataset hyperparameter tuning.** Fixed defaults for every model.
  This favors TabPFN, which needs no tuning, and is a limitation to discuss.
  Grinsztajn et al. tune extensively.
- **Common preprocessing.** A Gaussian quantile transform plus standardization
  for every model. Trees are invariant to it on unperturbed data.
- **Target smoothing.** Nadaraya-Watson smoothing of the training labels,
  thresholded at 1/2. The bandwidth is `level` × the median pairwise distance.
  At a fixed bandwidth, more training points means more neighbors are averaged,
  so we log `labels_changed`, the fraction of labels the smoothing flipped. Use
  it to check that the "same" smoothing level is comparable across n.
- **Uninformative features.** Both of Grinsztajn's versions: adding Gaussian
  noise features, and removing the least important features by random forest
  importance (computed once per repeat on the full training pool).
- **Rotation.** A Haar-random orthogonal matrix applied after preprocessing.
- **Shift.** The source region is the bottom 60% along either the first
  principal component (an oblique shift) or one random feature (an
  axis-aligned shift). The k target rows are nested: the k = 30 set contains the
  k = 10 set.

## Layout

```
src/tib/data.py       OpenML loading, cached splits, stratified subsampling
src/tib/perturb.py    smoothing, noise features, feature removal, rotation
src/tib/models.py     rf, xgb, mlp, tabpfn (v2), tabpfn_v1; shared preprocessing
src/tib/synthetic.py  2x2 smooth/jumpy x axis/oblique generator
src/tib/shift.py      covariate shift splits for the adaptation experiment
src/tib/run.py        experiment runner (resumable CSV output)
src/tib/analysis.py   paired deltas, t-intervals across datasets, Wilcoxon, Holm
configs/              one YAML per experiment
paper/                ICLR-style write-up (to come)
```

## Plan (about four weeks)

| Week | Work |
|------|------|
| 1 | Environment + TabPFN working; pick 6–10 benchmark datasets; smoke and timing runs; baseline n-sweep (no perturbations) |
| 2 | Experiment 1: three diagnostics × n; first figures; check `labels_changed` |
| 3 | Experiment 2: shift and k-shot adaptation; Experiment 3 (synthetic 2×2) if time allows |
| 4 | Analysis, figures, ICLR-template write-up |

Optional extension: compare TabPFN v1 with v2 under rotation. v1 puts a whole
row into one token, while v2 gives each feature (cell) its own token, so this
tests the orientation bias inside one model family. v1 needs the old package
(`pip install "tabpfn<1"`) in a separate environment.
