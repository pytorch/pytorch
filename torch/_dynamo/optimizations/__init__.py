from .backends import BACKENDS
from .training import create_aot_backends

create_aot_backends()

__all__ = ["BACKENDS"]
