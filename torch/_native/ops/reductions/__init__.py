# CuTeDSL native reduction kernels + dispatcher.
#
# Two independent override families share this package:
#   cutedsl_impl / inner_tree_* - the PYTORCH_SUM_INNER_TREE inner-tree sum/prod
#                    overrides, which carry a BITWISE-equivalence contract.
#   kernel_general - K0: TensorIterator-driven general kernel, any geometry (row,
#                    column, n-D, transposed, reduce-all); the correctness floor +
#                    the _reduce/_try_fast_row dispatcher that routes to the fast paths.
#
# The fast paths that K0's dispatcher routes to, and the aten op overrides that put
# torch.sum / mean / var / ... on this package at all, arrive in later stages.
#
# The kernel modules import `cutlass`, so they are NOT imported here -- that
# would pull the DSL runtime into `import torch` (the lazy-DSL-import contract;
# see test_no_dsl_imports_after_import_torch). overrides.py binds them lazily.

from .cutedsl_impl import register_to_dispatch


# Registration order IS evaluation order (the router's first-match-wins loop walks
# the graph in registration order), so the inner-tree family must register FIRST:
# both families claim sum.dim_IntList / prod.dim_int on CUDA, and the inner-tree one
# asserts an exact bit pattern from its own kernel. Its cond is gated off unless
# PYTORCH_SUM_INNER_TREE is set, so this costs one env lookup in the default config.
register_to_dispatch()
