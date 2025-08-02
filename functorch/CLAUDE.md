# Functorch Development Guide

JAX-like composable function transforms for PyTorch - functional transforms that enable vectorization, automatic differentiation, and compilation.
This implementation was mostly moved to `torch/func`, `torch._functorch` and the core library.

functorch.dim and functorch.einops are the remaining features not migrated to core but they have issues.
Moreover no package is built anymore for functorch so we will delete it soon.