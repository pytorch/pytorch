# Owner(s): ["module: cuda"]
# run time cuda tests, but with the allocator using expandable segments

import os


# Must precede the test_cuda import: EXPANDABLE_SEGMENTS in
# torch.testing._internal.common_utils reads PYTORCH_CUDA_ALLOC_CONF once at
# import time. Setting it here lets the @skipIf(not EXPANDABLE_SEGMENTS, ...)
# guards on expandable-only tests in test_cuda.py run in this runner. We
# restore the prior env state below (after the import) so subprocesses
# spawned by tests don't inherit expandable_segments:True, which would
# conflict with backend:cudaMallocAsync and similar settings those
# subprocesses set themselves.
_PRIOR_ALLOC_CONF = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
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


# Restore prior env state. EXPANDABLE_SEGMENTS has already been resolved at
# import time above; the runtime allocator is forced into expandable mode
# via _set_allocator_settings in __main__ below.
if _PRIOR_ALLOC_CONF is None:
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _PRIOR_ALLOC_CONF

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.import_test_stats import get_disabled_tests


# Make sure to remove REPO_ROOT after import is done
sys.path.remove(str(REPO_ROOT))

if __name__ == "__main__":
    if torch.cuda.is_available() and not IS_JETSON and not IS_WINDOWS:
        get_disabled_tests(".")
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        run_tests()
