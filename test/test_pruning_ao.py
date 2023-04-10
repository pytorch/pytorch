# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

from torch.testing._internal.common_utils import run_tests, IS_ARM64

from ao.pruning.test_kernels import TestQuantizedSparseKernels  # noqa: F401
# Kernels
from ao.pruning.test_kernels import TestQuantizedSparseLayers  # noqa: F401

# Parametrizations
from ao.pruning.test_parametrization import TestFakeSparsity  # noqa: F401

# Pruner
from ao.pruning.test_pruner import TestBasePruner  # noqa: F401
from ao.pruning.test_pruner import TestNearlyDiagonalPruner  # noqa: F401
from ao.pruning.test_pruner import TestWeightNormPruner  # noqa: F401

# Structured Pruning
from ao.pruning.test_structured_sparsifier import TestBaseStructuredSparsifier  # noqa: F401
from ao.pruning.test_structured_sparsifier import TestSaliencyPruner  # noqa: F401

# Scheduler
from ao.pruning.test_scheduler import TestScheduler  # noqa: F401
from ao.pruning.test_scheduler import TestCubicScheduler  # noqa: F401

# Composability
if not IS_ARM64:
    from ao.pruning.test_composability import TestComposability  # noqa: F401
    from ao.pruning.test_composability import TestFxComposability  # noqa: F401

# Utilities
from ao.pruning.test_pruning_utils import TestPruningUtilFunctions  # noqa: F401

# Data Sparsifier
from ao.pruning.test_data_sparsifier import TestBaseDataSparsifier  # noqa: F401
from ao.pruning.test_data_sparsifier import TestNormDataSparsifiers  # noqa: F401
from ao.pruning.test_data_sparsifier import TestQuantizationUtils  # noqa: F401

# Data Scheduler
from ao.pruning.test_data_scheduler import TestBaseDataScheduler  # noqa: F401

# Activation Sparsifier
from ao.pruning.test_activation_sparsifier import TestActivationSparsifier  # noqa: F401

if __name__ == "__main__":
    run_tests()
