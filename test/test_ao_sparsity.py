# Owner(s): ["module: unknown"]

# Kernels
from ao.sparsity.test_kernels import (  # noqa: F401  # noqa: F401
    TestQuantizedSparseKernels,
    TestQuantizedSparseLayers,
)

# Parametrizations
from ao.sparsity.test_parametrization import TestFakeSparsity  # noqa: F401

# Scheduler
from ao.sparsity.test_scheduler import (  # noqa: F401  # noqa: F401
    TestCubicScheduler,
    TestScheduler,
)

# Sparsifier
from ao.sparsity.test_sparsifier import (  # noqa: F401  # noqa: F401  # noqa: F401
    TestBaseSparsifier,
    TestNearlyDiagonalSparsifier,
    TestWeightNormSparsifier,
)

# Structured Pruning
from ao.sparsity.test_structured_sparsifier import (  # noqa: F401  # noqa: F401  # noqa: F401
    TestBaseStructuredSparsifier,
    TestFPGMPruner,
    TestSaliencyPruner,
)

from torch.testing._internal.common_utils import IS_ARM64, run_tests

# Composability
if not IS_ARM64:
    from ao.sparsity.test_composability import (  # noqa: F401  # noqa: F401
        TestComposability,
        TestFxComposability,
    )

# Activation Sparsifier
from ao.sparsity.test_activation_sparsifier import (  # noqa: F401
    TestActivationSparsifier,
)

# Data Scheduler
from ao.sparsity.test_data_scheduler import TestBaseDataScheduler  # noqa: F401

# Data Sparsifier
from ao.sparsity.test_data_sparsifier import (  # noqa: F401  # noqa: F401  # noqa: F401
    TestBaseDataSparsifier,
    TestNormDataSparsifiers,
    TestQuantizationUtils,
)

# Utilities
from ao.sparsity.test_sparsity_utils import TestSparsityUtilFunctions  # noqa: F401

if __name__ == "__main__":
    run_tests()
