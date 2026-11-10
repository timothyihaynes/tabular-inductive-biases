"""Print the datasets in the Grinsztajn benchmark suite (needs internet once)."""
import sys

from tib.data import GRINSZTAJN_CLF_NUM_SUITE, list_benchmark_tasks

suite = int(sys.argv[1]) if len(sys.argv) > 1 else GRINSZTAJN_CLF_NUM_SUITE
print(list_benchmark_tasks(suite).to_string(index=False))
