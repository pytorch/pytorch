# Owner(s): ["module: cuda"]
# run time cuda tests, but with the allocator using expandable segments

import os


# Must precede the test_cuda import: EXPANDABLE_SEGMENTS in
# torch.testing._internal.common_utils reads PYTORCH_CUDA_ALLOC_CONF once at
# import time, and the C10 caching allocator picks up the same env var on
# first use. Setting it here keeps the @skipIf(not EXPANDABLE_SEGMENTS, ...)
# guards on expandable-only tests in test_cuda.py from skipping in this runner.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pathlib
import sys

from test_cuda import (  # noqa: F401
    TestBlockStateAbsorption,
    TestCuda,
    TestCudaAllocator,
    TestMemPool,
)

import torch
from torch.testing._internal.common_cuda import IS_JETSON, IS_WINDOWS
from torch.testing._internal.common_utils import run_tests


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.import_test_stats import get_disabled_tests


# Make sure to remove REPO_ROOT after import is done
sys.path.remove(str(REPO_ROOT))

if __name__ == "__main__":
    if torch.cuda.is_available() and not IS_JETSON and not IS_WINDOWS:
        get_disabled_tests(".")
        run_tests()
