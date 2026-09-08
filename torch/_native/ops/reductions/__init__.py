# CuTeDSL native reduction kernels + dispatcher. Two independent override families share this
# package: cutedsl_impl / inner_tree_*, the PYTORCH_SUM_INNER_TREE overrides, which carry a
# BITWISE-equivalence contract; and the general dispatcher with the shared kernel behind it.
# `tile` is that kernel -- one @cute.kernel parameterized by which axis is reduced -- and the
# kernel_* modules are its drivers, each owning one axis's launch policy and plan cache.
#
# They import `cutlass`, so they are NOT imported here: that would pull the DSL runtime into
# `import torch` (see test_no_dsl_imports_after_import_torch). overrides.py binds them lazily.

from .cutedsl_impl import register_to_dispatch


# Registration order IS evaluation order, so the inner-tree family must register FIRST: both
# families claim sum.dim_IntList / prod.dim_int on CUDA, and that one asserts an exact bit
# pattern. Its cond is gated off unless its env var is set.
register_to_dispatch()
