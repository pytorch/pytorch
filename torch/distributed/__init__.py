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
        Store,
        FileStore,
        TCPStore,
        ProcessGroup,
        PrefixStore,
        Reducer,
        Logger,
        BuiltinCommHookType,
        GradBucket,
        Work as _Work,
        _DEFAULT_FIRST_BUCKET_BYTES,
        _register_comm_hook,
        _register_builtin_comm_hook,
        _broadcast_coalesced,
        _compute_bucket_assignment_by_size,
        _verify_params_across_processes,
        _test_python_store,
        DebugLevel,
        get_debug_level,
        set_debug_level,
        set_debug_level_from_env,
    )

    if sys.platform != "win32":
        from torch._C._distributed_c10d import (
            HashStore,
            _round_robin_process_groups,
        )

    from .distributed_c10d import *  # noqa: F403

    # Variables prefixed with underscore are not auto imported
    # See the comment in `distributed_c10d.py` above `_backend` on why we expose
    # this.

    from .distributed_c10d import (
        _backend,
        _all_gather_base,
        _reduce_scatter_base,
        _create_process_group_wrapper,
        _rank_not_in_group,
    )

    from .remote_device import _remote_device

    set_debug_level_from_env()
