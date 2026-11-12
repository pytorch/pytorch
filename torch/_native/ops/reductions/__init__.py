# Keep reduction-kernel imports lazy so `import torch` does not load cutlass.

from .cutedsl_impl import register_to_dispatch
from .overrides import register_reduction_overrides


# The router is first-match-wins. The bitwise inner-tree family registers first, but only
# when PYTORCH_SUM_INNER_TREE enables it; its declined sum.dim_IntList and prod.dim_int
# calls then fall through to the CuTeDSL reduction overrides, and its declined out= calls
# reach ATen. Otherwise the reduction overrides are the only registrations.
register_to_dispatch()
register_reduction_overrides()
