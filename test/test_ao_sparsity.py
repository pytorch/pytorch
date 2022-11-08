# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

from torch.testing._internal.common_utils import run_tests, IS_ARM64

# Kernels
from ao.sparsity.test_kernels import TestQuantizedSparseKernels  # noqa: F401
from ao.sparsity.test_kernels import TestQuantizedSparseLayers  # noqa: F401

# Parametrizations
from ao.sparsity.test_parametrization import TestFakeSparsity  # noqa: F401

# Sparsifier
from ao.sparsity.test_sparsifier import TestBaseSparsifier  # noqa: F401
from ao.sparsity.test_sparsifier import TestWeightNormSparsifier  # noqa: F401
from ao.sparsity.test_sparsifier import TestNearlyDiagonalSparsifier  # noqa: F401
<<<<<<< HEAD
from ao.sparsity.test_structured_sparsifier import TestBaseStructuredSparsifier  # noqa: F401
=======

# Pruner
from ao.sparsity.test_pruner import TestBaseStructuredPruner, TestBaseStructuredPrunerConvert  # noqa: F401
>>>>>>> 83eb036eb8 (Add fx mode structured pruning)

# Scheduler
from ao.sparsity.test_scheduler import TestScheduler  # noqa: F401
from ao.sparsity.test_scheduler import TestCubicScheduler  # noqa: F401

# Composability
if not IS_ARM64:
    from ao.sparsity.test_composability import TestComposability  # noqa: F401
    from ao.sparsity.test_composability import TestFxComposability  # noqa: F401

# Utilities
from ao.sparsity.test_sparsity_utils import TestSparsityUtilFunctions  # noqa: F401

# Data Sparsifier
from ao.sparsity.test_data_sparsifier import TestBaseDataSparsifier  # noqa: F401
from ao.sparsity.test_data_sparsifier import TestNormDataSparsifiers  # noqa: F401
from ao.sparsity.test_data_sparsifier import TestQuantizationUtils  # noqa: F401

# Data Scheduler
from ao.sparsity.test_data_scheduler import TestBaseDataScheduler  # noqa: F401

# Activation Sparsifier
from ao.sparsity.test_activation_sparsifier import TestActivationSparsifier  # noqa: F401

if __name__ == '__main__':
    run_tests()
