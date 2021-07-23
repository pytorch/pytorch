# -*- coding: utf-8 -*-

from torch.testing._internal.common_utils import run_tests

# Kernels
from ao.sparsity.test_kernels import TestQuantizedSparseKernels  # noqa: F401
from ao.sparsity.test_kernels import TestQuantizedSparseLayers  # noqa: F401

# Parametrizations
from ao.sparsity.test_parametrization import TestFakeSparsity  # noqa: F401

# Sparsifier
from ao.sparsity.test_sparsifier import TestBaseSparsifier  # noqa: F401
from ao.sparsity.test_sparsifier import TestWeightNormSparsifier  # noqa: F401

# Scheduler
from ao.sparsity.test_scheduler import TestScheduler  # noqa: F401

if __name__ == '__main__':
    run_tests()
