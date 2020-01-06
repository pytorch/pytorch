r"""This file is allowed to initialize CUDA context when imported."""

import torch
import torch.cuda
import unittest
import functools
from common_utils import TEST_WITH_ROCM, TEST_NUMBA


TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
CUDA_DEVICE = TEST_CUDA and torch.device("cuda:0")
# note: if ROCm is targeted, TEST_CUDNN is code for TEST_MIOPEN
TEST_CUDNN = TEST_CUDA and (TEST_WITH_ROCM or torch.backends.cudnn.is_acceptable(torch.tensor(1., device=CUDA_DEVICE)))
TEST_CUDNN_VERSION = torch.backends.cudnn.version() if TEST_CUDNN else 0

if TEST_NUMBA:
    import numba.cuda
    TEST_NUMBA_CUDA = numba.cuda.is_available()
else:
    TEST_NUMBA_CUDA = False

# Used below in `initialize_cuda_context_rng` to ensure that CUDA context and
# RNG have been initialized.
__cuda_ctx_rng_initialized = False


# after this call, CUDA context and RNG must have been initialized on each GPU
def initialize_cuda_context_rng():
    global __cuda_ctx_rng_initialized
    assert TEST_CUDA, 'CUDA must be available when calling initialize_cuda_context_rng'
    if not __cuda_ctx_rng_initialized:
        # initialize cuda context and rng for memory tests
        for i in range(torch.cuda.device_count()):
            torch.randn(1, device="cuda:{}".format(i))
        __cuda_ctx_rng_initialized = True


def allowCUDAOOM(reason):
    def wrapper(f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            try:
                f(*args, **kwargs)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    raise unittest.SkipTest(reason)
                raise e
        return wrapped_f
    return wrapper
