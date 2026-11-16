"""Tabular inductive biases at small n: STATS 295 final project."""

# On macOS, PyTorch (used by TabPFN) and XGBoost each ship or link their own
# OpenMP runtime. Loading XGBoost's first and then torch's can crash the process
# with a segmentation fault. Importing torch first avoids that, so do it here,
# before any model code runs. Harmless if torch is not installed.
try:  # pragma: no cover
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    pass
