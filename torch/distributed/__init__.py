
import os
import sys
from enum import Enum

import torch


def is_available() -> bool:
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


if is_available():
    from torch._C._distributed_c10d import (
        _DEFAULT_FIRST_BUCKET_BYTES,
        BuiltinCommHookType,
        FileStore,
        GradBucket,
        Logger,
        ProcessGroup,
        Reducer,
        Store,
        TCPStore,
        _broadcast_coalesced,
        _compute_bucket_assignment_by_size,
        _DistributedDebugLevel,
        _get_debug_mode,
        _register_builtin_comm_hook,
        _register_comm_hook,
        _test_python_store,
        _verify_model_across_ranks
    )
    if sys.platform != 'win32':
        from torch._C._distributed_c10d import (
            HashStore,
            _round_robin_process_groups,
        )

    # Variables prefixed with underscore are not auto imported
    # See the comment in `distributed_c10d.py` above `_backend` on why we expose
    # this.
    from .distributed_c10d import *  # noqa: F403
    from .distributed_c10d import _backend
