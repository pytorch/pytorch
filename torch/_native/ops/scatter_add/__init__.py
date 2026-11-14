from .cutedsl_impl import register_to_dispatch as _register_cutedsl
from .flydsl_impl import register_to_dispatch as _register_flydsl


# The FlyDSL overrides register only on ROCm builds and the CuTeDSL overrides
# decline on ROCm, so at most one family is active for a given call.
_register_flydsl()
_register_cutedsl()
