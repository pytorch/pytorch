# CuteDSL reduction kernels and dispatchers. The inner-tree overrides require bitwise
# equivalence; the general dispatcher supplies the shared fallback. Kernel modules import
# cutlass, so overrides.py binds them lazily to keep `import torch` DSL-free.

from .cutedsl_impl import register_to_dispatch


# Registration order is evaluation order. Register the opt-in inner-tree family first because
# both families claim CUDA sum/prod and only it promises an exact bit pattern.
register_to_dispatch()
