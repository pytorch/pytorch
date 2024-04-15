import os
import sys
from enum import Enum
import pdb
import io

import torch

def is_available() -> bool:
    """
    Return ``True`` if the distributed package is available.

    Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return hasattr(torch._C, "_c10d_init")


if is_available() and not torch._C._c10d_init():
    raise RuntimeError("Failed to initialize torch.distributed")

# Custom Runtime Errors thrown from the distributed package
DistError = torch._C._DistError
DistBackendError = torch._C._DistBackendError
DistNetworkError = torch._C._DistNetworkError
DistStoreError = torch._C._DistStoreError

if is_available():
    from torch._C._distributed_c10d import (
        Store,
        FileStore,
        TCPStore,
        ProcessGroup as ProcessGroup,
        Backend as _Backend,
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
        _make_nccl_premul_sum,
    )

    class _DistributedPdb(pdb.Pdb):
        """
        Supports using PDB from inside a multiprocessing child process.

        Usage:
        _DistributedPdb().set_trace()
        """
        def interaction(self, *args, **kwargs):
            _stdin = sys.stdin
            try:
                sys.stdin = open('/dev/stdin')
                pdb.Pdb.interaction(self, *args, **kwargs)
            finally:
                sys.stdin = _stdin

    def breakpoint(rank: int = 0):
        """
        Set a breakpoint, but only on a single rank.  All other ranks will wait for you to be
        done with the breakpoint before continuing.

        Args:
            rank (int): Which rank to break on.  Default: ``0``
        """
        if get_rank() == rank:
            pdb = _DistributedPdb()
            pdb.message(
                "\n!!! ATTENTION !!!\n\n"
                f"Type 'up' to get to the frame that called dist.breakpoint(rank={rank})\n"
            )
            pdb.set_trace()
        # If Meta/Python keys are in the TLS, we want to make sure that we ignore them
        # and hit the (default) CPU/CUDA implementation of barrier.
        meta_in_tls = torch._C._meta_in_tls_dispatch_include()
        guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
        torch._C._set_meta_in_tls_dispatch_include(False)
        try:
            barrier()
        finally:
            torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)
            del guard

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
        _all_gather_base,
        _reduce_scatter_base,
        _create_process_group_wrapper,
        _rank_not_in_group,
        _coalescing_manager,
        _CoalescingManager,
        _get_process_group_name,
        get_node_local_rank,
    )

    from .rendezvous import (
        rendezvous,
        _create_store_from_options,
        register_rendezvous_handler,
    )

    from .remote_device import _remote_device

    set_debug_level_from_env()

else:
    # This stub is sufficient to get
    #   python test/test_public_bindings.py -k test_correct_module_names
    # working even when USE_DISTRIBUTED=0.  Feel free to add more
    # stubs as necessary.
    # We cannot define stubs directly because they confuse pyre

    class _ProcessGroupStub:
        pass
    sys.modules["torch.distributed"].ProcessGroup = _ProcessGroupStub  # type: ignore[attr-defined]
