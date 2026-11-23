"""Summaries and plots with honest uncertainty.

Unit of replication: the *dataset*. Within a dataset we average over repeats
first, then form a t-interval across datasets. Repeats on the same dataset are
not independent evidence about "tabular data in general", so pooling them
would overstate confidence.

    python -m tib.analysis results/nsweep.csv --metric acc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def perturbation_drop(df: pd.DataFrame, metric: str = "acc") -> pd.DataFrame:
    """metric(perturbed) - metric(unperturbed), paired on (dataset, repeat, n, model).

    Negative = the perturbation hurts. A large negative value under "rotate"
    means the model relied on the original feature orientation, etc.
    """
    keys = ["dataset", "repeat", "n", "model"]
    base = df[df.perturbation == "none"][keys + [metric]].rename(columns={metric: "base"})
    pert = df[df.perturbation != "none"].merge(base, on=keys)
    pert["delta"] = pert[metric] - pert["base"]
    return pert


def mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def summarize(df: pd.DataFrame, value: str, group: list[str]) -> pd.DataFrame:
    """Average over repeats within dataset, then mean and CI across datasets."""
    per_ds = df.groupby(group + ["dataset"], as_index=False)[value].mean()
    rows = []
    for g, sub in per_ds.groupby(group):
        m, lo, hi = mean_ci(sub[value].to_numpy())
        rows.append(dict(zip(group, g if isinstance(g, tuple) else (g,)),
                         mean=m, lo=lo, hi=hi, n_datasets=len(sub)))
    return pd.DataFrame(rows)


def paired_model_test(df: pd.DataFrame, a: str, b: str, metric: str = "acc",
                      by: list[str] = ("n",)) -> pd.DataFrame:
    """Paired comparison of model a vs b across datasets, per group.

    Returns the mean difference a - b with a 95% t-interval across datasets,
    the number of datasets where a beats b (``wins``), and a two-sided Wilcoxon
    signed-rank p-value. Only the pre-specified comparisons in
    docs/analysis_plan.md get a p-value in the write-up (Holm within each
    experiment); report the rest as effect sizes with intervals.

    With N datasets the smallest possible Wilcoxon p is 2 / 2**N (0.0078 for
    N = 8, 0.0625 for N = 5), which bounds how many tests a family can hold.
    """
    by = list(by)
    keys = ["dataset"] + by
    per_ds = df.groupby(keys + ["model"], as_index=False)[metric].mean()
    wide = per_ds.pivot_table(index=keys, columns="model", values=metric).reset_index()
    rows = []
    for g, sub in wide.groupby(by):
        d = (sub[a] - sub[b]).dropna().to_numpy()
        p = stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
        m, lo, hi = mean_ci(d) if len(d) else (np.nan, np.nan, np.nan)
        rows.append(dict(zip(by, g if isinstance(g, tuple) else (g,)),
                         mean_diff=m, lo=lo, hi=hi, wins=int((d > 0).sum()),
                         n_datasets=len(d), p=p))
    return pd.DataFrame(rows)


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def plot_drop_vs_n(drop: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    s = summarize(drop, "delta", ["perturbation", "level", "model", "n"])
    panels = s[["perturbation", "level"]].drop_duplicates().to_numpy()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3), sharey=True,
                             squeeze=False)
    for ax, (kind, level) in zip(axes[0], panels):
        sub = s[(s.perturbation == kind) & (s.level == level)]
        for m, g in sub.groupby("model"):
            g = g.sort_values("n")
            ax.plot(g.n, g["mean"], marker="o", label=m)
            ax.fill_between(g.n, g.lo, g.hi, alpha=0.15)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{kind} ({level:g})", fontsize=9)
        ax.set_xlabel("training size n")
    axes[0, 0].set_ylabel("change in accuracy vs. unperturbed")
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"figure -> {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--metric", default="acc")
    args = ap.parse_args(argv)
    df = pd.read_csv(args.csv)
    if "perturbation" in df:
        drop = perturbation_drop(df, args.metric)
        print(summarize(drop, "delta", ["perturbation", "level", "model", "n"]).to_string())
        plot_drop_vs_n(drop, Path(args.csv).with_suffix(".drop_vs_n.png"))
    else:
        group = [c for c in ("cell", "shift", "k", "n", "n_source") if c in df] + ["model"]
        if "cell" in df:  # synthetic: treat function seed as the replication unit
            df = df.assign(dataset=df.cell + "_" + df.fn_seed.astype(str))
        print(summarize(df, args.metric, group).to_string())


if __name__ == "__main__":
    main()
