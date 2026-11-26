# Owner(s): ["module: cuda"]
# run time accelerator tests, but with the allocator using expandable segments

import os
import pathlib
import sys

import torch
from torch.testing._internal.common_cuda import IS_JETSON, IS_WINDOWS
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import (
    HAS_CUDA_AND_TRITON,
    HAS_XPU_AND_TRITON,
)


pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if HAS_CUDA_AND_TRITON or HAS_XPU_AND_TRITON:
    try:
        from . import test_cudagraph_trees
    except ImportError:
        import test_cudagraph_trees  # @manual=fbcode//caffe2/test/inductor:cudagraph_trees-library

    if HAS_CUDA_AND_TRITON:
        CudaGraphTreeTestsCUDA = test_cudagraph_trees.CudaGraphTreeTestsCUDA
    if HAS_XPU_AND_TRITON:
        CudaGraphTreeTestsXPU = test_cudagraph_trees.CudaGraphTreeTestsXPU

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from tools.stats.import_test_stats import get_disabled_tests  # @manual


# Make sure to remove REPO_ROOT after import is done
sys.path.remove(str(REPO_ROOT))

if __name__ == "__main__":
    if not IS_WINDOWS and (
        (HAS_CUDA_AND_TRITON and not IS_JETSON) or HAS_XPU_AND_TRITON
    ):
        get_disabled_tests(".")

        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")

        run_tests()
