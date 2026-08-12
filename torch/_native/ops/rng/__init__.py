# CuTeDSL native RNG (distribution) op overrides. Importing this package registers the
# aten uniform_ / normal_ overrides with the _native registry.

from .impl import register_rng_overrides


register_rng_overrides()
