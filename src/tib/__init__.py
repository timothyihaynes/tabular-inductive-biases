"""Tabular inductive biases at small n: STATS 295 final project."""

import os
import sys

# On macOS, torch, XGBoost and LightGBM can each load their own copy of the
# OpenMP runtime, and two copies in one process segfault. These two settings
# let them coexist (tested on Apple Silicon). They must be set before any of
# those libraries load, which is why they live here. Override them by setting
# the variables yourself before running.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# On macOS, PyTorch (used by TabPFN) and XGBoost each ship or link their own
# OpenMP runtime. Loading XGBoost's first and then torch's can crash the process
# with a segmentation fault. Importing torch first avoids that, so do it here,
# before any model code runs. Harmless if torch is not installed.
try:  # pragma: no cover
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    pass
