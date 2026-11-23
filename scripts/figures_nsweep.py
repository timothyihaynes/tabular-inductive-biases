"""First figures and tables for Experiment 1 (n-sweep).

    python scripts/figures_nsweep.py results/nsweep.csv [out_dir]

Writes to out_dir (default: results/figures/):
  fig1_baseline.png      accuracy vs n per model, unperturbed (mean and 95% CI
                         across datasets)
  fig2_drop_vs_n.png     change in accuracy from each perturbation vs n
  fig3_smooth_flips.png  smoothing: accuracy drop vs share of training labels
                         flipped, which separates "more smoothing" from
                         "bigger n" (at a fixed bandwidth, larger n flips more)
  fig4_rotate_by_dataset.png  rotation drop per dataset (one dataset can
                         dominate the average)
  tables.md              the numbers behind the figures, the 4 primary
                         TabPFN-vs-RF tests (Holm), and effect sizes with
                         intervals for everything else
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tib.analysis import holm, paired_model_test, perturbation_drop, summarize  # noqa: E402

COLORS = {"rf": "#2a7f62", "xgb": "#c9792b", "mlp": "#7a5ea8", "tabpfn": "#2b6cb0"}
ORDER = ["rf", "xgb", "mlp", "tabpfn"]

csv = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else csv.parent / "figures"
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(csv)
if "degenerate" in df:
    df = df[df.degenerate != True]  # noqa: E712  (NaN in older rows = not degenerate)
base = df[df.perturbation == "none"]
drop = perturbation_drop(df, "acc")
md = [f"# Experiment 1 tables\n\nSource: `{csv.name}`, {df.dataset.nunique()} datasets, "
      f"{df.repeat.nunique()} repeats. Mean across datasets (repeats averaged within "
      "dataset first), 95% t-interval in brackets.\n"]

# --- Figure 1: baseline accuracy vs n
s = summarize(base, "acc", ["model", "n"])
fig, ax = plt.subplots(figsize=(4.2, 3.2))
for m in ORDER:
    g = s[s.model == m].sort_values("n")
    ax.plot(g.n, g["mean"], marker="o", color=COLORS[m], label=m)
    ax.fill_between(g.n, g.lo, g.hi, color=COLORS[m], alpha=0.12)
ax.set_xscale("log")
ax.set_xlabel("training size n")
ax.set_ylabel("test accuracy (unperturbed)")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(out / "fig1_baseline.png", dpi=200)
plt.close(fig)
for metric in ["acc", "auc"]:
    t = summarize(base, metric, ["model", "n"])
    t["cell"] = t.apply(lambda r: f"{r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}]", axis=1)
    md.append(f"\n## Baseline {metric}\n\n"
              + t.pivot(index="model", columns="n", values="cell").loc[ORDER].to_markdown())

# Primary tests, fixed in docs/analysis_plan.md: TabPFN vs RF at each n, Holm
# over these 4 only. Everything else is reported as an effect size with a
# 95% t-interval across datasets and a win count, without p-values.
primary = paired_model_test(base, "tabpfn", "rf", "acc", by=["n"])
primary["p_holm"] = holm(primary.p.to_numpy())
primary["ci"] = primary.apply(
    lambda r: f"{100*r.mean_diff:+.1f} [{100*r.lo:+.1f}, {100*r.hi:+.1f}]", axis=1)
primary["wins"] = primary.wins.astype(str) + "/" + primary.n_datasets.astype(str)
md.append("\n\n## Primary tests: TabPFN - RF, accuracy\n\n"
          "Mean difference in points with 95% t-interval across datasets; wins = datasets "
          "where TabPFN is ahead. Two-sided Wilcoxon signed-rank across datasets, Holm over "
          f"these {len(primary)} tests only. Note: this family was fixed after Experiment 1 "
          "had been run (see docs/analysis_plan.md).\n\n"
          + primary[["n", "ci", "wins", "p", "p_holm"]]
          .rename(columns={"ci": "TabPFN - RF (points)"})
          .round(4).to_markdown(index=False))
eff = []
for b in ["xgb", "mlp"]:
    t = paired_model_test(base, "tabpfn", b, "acc", by=["n"])
    t["cell"] = t.apply(lambda r: f"{100*r.mean_diff:+.1f} [{100*r.lo:+.1f}, "
                        f"{100*r.hi:+.1f}], {int(r.wins)}/{int(r.n_datasets)}", axis=1)
    t["comparison"] = f"tabpfn - {b}"
    eff.append(t)
eff = pd.concat(eff, ignore_index=True)
md.append("\n\n## Secondary comparisons, accuracy (effect sizes, no tests)\n\n"
          "Mean difference in points [95% t-interval across datasets], then wins.\n\n"
          + eff.pivot(index="comparison", columns="n", values="cell").to_markdown())

# --- Figure 2: perturbation drop vs n
s = summarize(drop, "delta", ["perturbation", "level", "model", "n"])
panels = s[["perturbation", "level"]].drop_duplicates().sort_values(["perturbation", "level"])
panels = panels.to_numpy()
ncol = 4
nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow), sharex=True,
                         squeeze=False)
for ax, (kind, level) in zip(axes.flat, panels):
    sub = s[(s.perturbation == kind) & (s.level == level)]
    for m in ORDER:
        g = sub[sub.model == m].sort_values("n")
        ax.plot(g.n, 100 * g["mean"], marker="o", ms=3, color=COLORS[m], label=m)
        ax.fill_between(g.n, 100 * g.lo, 100 * g.hi, color=COLORS[m], alpha=0.10)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_title(f"{kind} ({level:g})", fontsize=9)
for ax in axes.flat[len(panels):]:
    ax.axis("off")
for ax in axes[:, 0]:
    ax.set_ylabel("Δ accuracy (points)")
for ax in axes[-1, :]:
    ax.set_xlabel("training size n")
axes.flat[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(out / "fig2_drop_vs_n.png", dpi=200)
plt.close(fig)
s["cell"] = s.apply(lambda r: f"{100*r['mean']:+.1f} [{100*r.lo:+.1f}, {100*r.hi:+.1f}]",
                    axis=1)
md.append("\n\n## Change in accuracy from each perturbation (points)\n\n"
          + s.pivot_table(index=["perturbation", "level", "model"], columns="n",
                          values="cell", aggfunc="first").to_markdown())

# --- Figure 3: smoothing drop vs labels flipped
sm = drop[drop.perturbation == "smooth"]
per = sm.groupby(["dataset", "level", "n", "model"], as_index=False)[
    ["delta", "labels_changed"]].mean()
fig, ax = plt.subplots(figsize=(4.2, 3.2))
for m in ORDER:
    g = per[per.model == m]
    ax.scatter(100 * g.labels_changed, 100 * g.delta, s=8, alpha=0.5, color=COLORS[m],
               label=m)
    x, y = g.labels_changed.to_numpy(), g.delta.to_numpy()
    slope = (x @ y) / (x @ x)  # least squares through the origin
    xs = np.linspace(0, x.max(), 10)
    ax.plot(100 * xs, 100 * slope * xs, color=COLORS[m], lw=1.2)
ax.set_xlabel("training labels flipped by smoothing (%)")
ax.set_ylabel("Δ accuracy (points)")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(out / "fig3_smooth_flips.png", dpi=200)
plt.close(fig)
flips = sm.groupby(["level", "n"]).labels_changed.mean().unstack().round(3)
md.append("\n\n## Share of training labels flipped by smoothing\n\n" + flips.to_markdown())
slopes = {m: float((g.labels_changed @ g.delta) / (g.labels_changed @ g.labels_changed))
          for m, g in per.groupby("model")}
md.append("\n\nAccuracy lost per unit of labels flipped (slope through origin, "
          "dataset x level x n means): "
          + ", ".join(f"{m} {slopes[m]:.2f}" for m in ORDER))

# --- Figure 4: rotation by dataset
rot = drop[drop.perturbation == "rotate"].groupby(["dataset", "model"]).delta.mean()
rot = 100 * rot.unstack()[ORDER]
fig, ax = plt.subplots(figsize=(6, 3.2))
xs = np.arange(len(rot))
w = 0.2
for i, m in enumerate(ORDER):
    ax.bar(xs + (i - 1.5) * w, rot[m], w, color=COLORS[m], label=m)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(rot.index, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Δ accuracy from rotation (points)")
ax.legend(fontsize=8, frameon=False, ncol=4)
fig.tight_layout()
fig.savefig(out / "fig4_rotate_by_dataset.png", dpi=200)
plt.close(fig)
md.append("\n\n## Rotation, change in accuracy by dataset (points, all n)\n\n"
          + rot.round(1).to_markdown())

(out / "tables.md").write_text("\n".join(md) + "\n")
print(f"figures and tables -> {out}")
