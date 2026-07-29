"""Backward-compatibility shim.

The ``Source`` subclasses moved to :mod:`torch._guards.source` as part of
extracting the guard machinery into an independent package. This module
re-exports them so existing ``torch._dynamo.source`` imports keep working.
"""

# ``import *`` does not re-export underscore-prefixed names; re-export the private
# helpers that other modules import by name and preserve the historical surface.
from torch._guards.source import *  # noqa: F403
from torch._guards.source import (  # noqa: F401
    _esc_str,
    _get_source_debug_name,
    _GUARD_SOURCE_FSDP_MODULE,
    _GUARD_SOURCE_SPECIALIZED_NN_MODULE,
    _GUARD_SOURCE_UNSPECIALIZED_BUILTIN_NN_MODULE,
    _GUARD_SOURCE_UNSPECIALIZED_NN_MODULE,
)
