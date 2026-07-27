# CuTeDSL native reduction kernels + dispatcher.
#
# Kernels (all built on .._cutedsl traits + launch glue):
#   kernel_general - K0: TensorIterator-driven general kernel, any geometry (row,
#                    column, n-D, transposed, reduce-all); the correctness floor +
#                    the _reduce/_try_fast_row dispatcher that routes to the fast paths.
#   kernel_row     - K1: vectorized one-shot row reduction (contiguous last dim,
#                    smem-fits), dynamic-M.
#   kernel_col     - K2: vectorized column reduction (reduce dim 0) with an M-split.
#   kernel_xcta    - fused two-stage cross-CTA row reduction for few-row / huge-N
#                    and reduce-all.
#
# Importing this package registers the aten reduction overrides (sum / mean / ...)
# with the _native registry via register_reduction_overrides(); the registry
# installs them at its _register_all_overrides() step. Each override is gated by a
# capability `cond` and falls back to aten when it does not apply.

# The kernel modules import `cutlass`, so they are NOT imported here -- that
# would pull the DSL runtime into `import torch` (the lazy-DSL-import contract;
# see test_no_dsl_imports_after_import_torch). overrides.py binds them lazily.
from .overrides import register_reduction_overrides


register_reduction_overrides()
