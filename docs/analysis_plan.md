# Analysis plan

Written 2026-11-23, after Experiment 1 and **before** Experiments 2 and 3 were
run. Each experiment gets one small family of hypothesis tests, fixed here, with
Holm-Bonferroni inside the family. Every other comparison is reported as an
effect size: the mean difference with a 95% t-interval across replication units
(datasets, or ground-truth functions for Experiment 3), plus the number of
units where the difference has the expected sign. Those have no p-values.

## Why the families are small

Tests are paired across replication units (two-sided Wilcoxon signed-rank). With
N units the smallest possible p is 2 / 2^N, which is 0.0078 for N = 8 and 0.0625
for N = 5. Holm multiplies the smallest p by the family size. So with 8 datasets
a family of m tests can reject at 0.05 only if m ≤ 6, and only when the effect has
the same sign on every dataset. Experiment 1 originally ran 12 tests (TabPFN vs
RF, XGBoost and MLP at four n). After Holm, nothing could reach significance,
even with TabPFN ahead of RF on all 8 datasets at n ≥ 300.

## Common rules

- Metric: test accuracy. AUC and log loss go in an appendix as effect sizes.
- Average repeats within each unit first. Each test sees one number per unit.
- α = 0.05, two-sided, Holm within each experiment's family. No correction
  across experiments, because each answers a separate question.
- Rows flagged `degenerate` are excluded, as in Experiment 1.
- If a run is incomplete, analyze the units that have every cell of the family
  and report how many were dropped.

## Experiment 1 (n-sweep): 4 tests

TabPFN − RF accuracy on unperturbed data at n = 100, 300, 1000 and 3000.

**Caveat for the write-up:** this family was chosen *after* the Experiment 1
results were known (12 tests had been planned). Say so in the paper. RF was the
natural choice before the results anyway: it is the reference model in
Grinsztajn et al., and the project question is about tree biases.

Result (from `scripts/figures_nsweep.py`): TabPFN − RF is +0.7 points at n = 100
(6/8 datasets, Holm p = 0.078), and +1.6, +2.1 and +3.6 points at n = 300, 1000
and 3000 (8/8 each, Holm p = 0.031). The perturbation results are effect sizes
only.

## Experiment 2 (shift and k-shot): 4 tests

Replication unit: the dataset (8). For each model, *adaptation gain* =
accuracy on the target test set at k = 100 minus at k = 0, in the same
(dataset, repeat, shift, n_source) cell.

Test: TabPFN gain − RF gain, in each of the 4 cells shift ∈ {pc1, feature} ×
n_source ∈ {300, 1000}. Hypothesis: TabPFN adapts faster, because new rows enter
its context directly.

k = 100 was chosen in advance as the middle of the k grid. k = 10 is mostly
noise for 1,000 target test rows, and k = 300 can be as large as the source set.
Report as effect sizes: k = 0 accuracy (robustness to shift), the full gain-vs-k
curves for all four models, and the pc1 − feature contrast (oblique vs
axis-aligned shift).

## Experiment 3 (synthetic 2 × 2): 2 tests

Replication unit: the ground-truth function (`fn_seed`). All four cells share
each seed, so contrasts are paired. With the original `n_functions: 5`, the
smallest Wilcoxon p is 0.0625, so no test could pass. The config now uses
`n_functions: 10` with `repeats: 3` (30 fits per cell per n, up from 25), which
puts the smallest p at 0.002.

Let G(cell) = RF accuracy − TabPFN accuracy, averaged over repeats and all four n.

1. **Orientation:** [G(axis, smooth) + G(axis, jumpy)] / 2 − [G(oblique, smooth) +
   G(oblique, jumpy)] / 2. Hypothesis: positive, because trees gain from axis
   alignment.
2. **Smoothness:** [G(jumpy, axis) + G(jumpy, oblique)] / 2 − [G(smooth, axis) +
   G(smooth, oblique)] / 2. Hypothesis: positive, because trees gain from
   irregular targets.

The larger of the two effect sizes answers the project question: is the tree
advantage about sharp jumps or about axis alignment? Report as effect sizes:
the same contrasts for XGBoost and the MLP, the interaction, and how each
contrast changes with n.
