# -*- coding: utf-8 -*-

from torch.testing._internal.common_utils import run_tests

# Kernels
from ao.sparsity.test_kernels import TestQuantizedSparseKernels  # noqa: F401
from ao.sparsity.test_kernels import TestQuantizedSparseLayers  # noqa: F401

if __name__ == '__main__':
    run_tests()
