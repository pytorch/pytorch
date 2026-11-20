# Keep reduction-kernel imports lazy so `import torch` does not load cutlass.

from .cutedsl_impl import register_to_dispatch


register_to_dispatch()
