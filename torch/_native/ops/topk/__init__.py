from .cutedsl_impl import register_to_dispatch as _register_cutedsl
from .flydsl_impl import register_to_dispatch as _register_flydsl


_register_flydsl()
_register_cutedsl()
