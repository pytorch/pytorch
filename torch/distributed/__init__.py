
import torch
import sys
import os
from enum import Enum

class DistributedDebugMode(Enum):
    """
    An enum whose values correspond to different debug settings of the
    torch.distributed package.
    """
    OFF = "OFF"
    INFO = "INFO"
    DETAIL = "DETAIL"


def is_available():
    """
    Returns ``True`` if the distributed package is available. Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return hasattr(torch._C, "_c10d_init")


if is_available() and not torch._C._c10d_init():
    raise RuntimeError("Failed to initialize torch.distributed")

def get_debug_mode():
    """
    Reads the environment variable ``TORCH_DISTRIBUTED_DEBUG`` and returns a
    ``DistributedDebugMode`` corresponding to the appropriate debug level. The
    env var ``TORCH_DISTRIBUTED_DEBUG`` must be set to one of "OFF", "INFO", or
    "DETAIL". Default setting is INFO.
    """
    debug_mode_str = os.environ.get("TORCH_DISTRIBUTED_DEBUG", None)
    if debug_mode_str == DistributedDebugMode.OFF.value:
        return DistributedDebugMode.OFF
    elif debug_mode_str == DistributedDebugMode.DETAIL.value:
        return DistributedDebugMode.DETAIL
    else:
        # None, unknown or INFO, which is the default
        return DistributedDebugMode.INFO


if is_available():
    from torch._C._distributed_c10d import (
        Store,
        FileStore,
        TCPStore,
        ProcessGroup,
        Reducer,
        Logger,
        BuiltinCommHookType,
        _DEFAULT_FIRST_BUCKET_BYTES,
        _GradBucket,
        _register_comm_hook,
        _register_builtin_comm_hook,
        _broadcast_coalesced,
        _compute_bucket_assignment_by_size,
        _test_python_store,
    )
    if sys.platform != 'win32':
        from torch._C._distributed_c10d import (
            HashStore,
            _round_robin_process_groups,
        )

    from .distributed_c10d import *
    # Variables prefixed with underscore are not auto imported
    # See the comment in `distributed_c10d.py` above `_backend` on why we expose
    # this.

    from .distributed_c10d import _backend
