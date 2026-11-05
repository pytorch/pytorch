# Keep reduction-kernel imports lazy so `import torch` does not load cutlass.

from .cutedsl_impl import register_to_dispatch
from .overrides import register_reduction_overrides


# The router is first-match-wins, so register the bitwise inner-tree family first.
register_to_dispatch()
register_reduction_overrides()
