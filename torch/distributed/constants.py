from datetime import timedelta
from typing import Optional

from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT


__all__ = ["default_pg_timeout", "default_pg_nccl_timeout"]

# Default process group wide timeout, if applicable.
# This only applies to the non-nccl backends
# To make an attempt at backwards compatibility with THD, we use an
# extraordinarily high default timeout, given that THD did not have timeouts.
default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT
# Separate timeout for PGNCCL mainly becuase it's always been that way in the C++ layer, but until recently
# there was one default that applied across all backends in the python layer.
# Later, we could consider merging them back together at the c++ layer if we can align on a same value.
# (only if TORCH_NCCL_BLOCKING_WAIT or TORCH_NCCL_ASYNC_ERROR_HANDLING is set to 1).

try:
    from torch._C._distributed_c10d import _DEFAULT_PG_NCCL_TIMEOUT

    default_pg_nccl_timeout: Optional[timedelta] = _DEFAULT_PG_NCCL_TIMEOUT
except ImportError:
    # if C++ NCCL support is not compiled, we don't have access to the default nccl value.
    # if anyone is actually trying to use nccl in this state, it should error.
    default_pg_nccl_timeout = None
