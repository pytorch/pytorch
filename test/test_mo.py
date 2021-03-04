# -*- coding: utf-8 -*-

r"""Tests related to the model optimization"""

from torch.testing._internal.common_utils import run_tests

# Sparsity tests
from mo.sparsity.test_quantized import TestQuantizedSparseKernels  # noqa: F401
from mo.sparsity.test_quantized import TestQuantizedSparseLayers  # noqa: F401

if __name__ == '__main__':
    run_tests()
