from typing import cast, Union

import torch

import torch.distributed.distributed_c10d as c10d

_CUDA_P2P_AVAILABLE = True

try:
    from torch._C._distributed_c10d import (
        _DistributedBackendOptions,
        Backend,
        ProcessGroupCudaP2P,
        ProcessGroupNCCL,
    )
except ImportError:
    pass

"""
This file contains the registration logic and Python APIs for
``ProcessGroupCudaP2P`` (experimental).

``ProcessGroupCudaP2P`` is a thin wrapper around ``ProcessGroupNCCL``. By
default, it routes all collectives to the underlying ``ProcessGroupNCCL``. In
addition, ``ProcessGroupCudaP2P`` initializes a P2P workspace that allows
direct GPU memory access among the members. The workspace can be used in Python
to optimize intra-node communication patterns or to create custom intra-node
collectives in CUDA.

``ProcessGroupCudaP2P`` aims to bridge the gap where certain important patterns
can be better optimized via fine-grained P2P memory access than with
collectives in the latest version of NCCL. It is meant to complement NCCL
rather than replacing it.

Usage:

    # Using ProcessGroupCudaP2P
    dist.init_process_group(backend="cuda_p2p", ...)

    # Using ProcessGroupCudaP2P while specifying ProcessGroupCudaP2P.Options
    pg_options = ProcessGroupCudaP2P.Options()
    dist.init_process_group(backend="cuda_p2p", pg_options=pg_options, ...)

    # Using ProcessGroupCudaP2P while specifying ProcessGroupNCCL.Options
    pg_options = ProcessGroupNCCL.Options()
    dist.init_process_group(backend="cuda_p2p", pg_options=pg_options, ...)

    # Using ProcessGroupCudaP2P while specifying both
    # ProcessGroupCudaP2P.Options and ProcessGroupNCCL.Options
    pg_options = ProcessGroupCudaP2P.Options()
    pg_options.nccl_options = ProcessGroupNCCL.Options()
    dist.init_process_group(backend="cuda_p2p", pg_options=pg_options, ...)

    # Down-casting the backend to access p2p buffers for cuda_p2p specific
    # optimizations
    if is_cuda_p2p_group(group):
        backend = get_cuda_p2p_backend(group)
        if required_p2p_buffer_size > backend.get_buffer_size():
            # fallback
        p2p_buffer = backend.get_p2p_buffer(...)
    else:
        # fallback
"""


def _create_cuda_p2p_group(
    dist_backend_opts: "_DistributedBackendOptions",
    options: Union["ProcessGroupCudaP2P.Options", "ProcessGroupNCCL.Options", None],
) -> "Backend":
    if options is None:
        options = ProcessGroupCudaP2P.Options()
        options.nccl_options = ProcessGroupNCCL.Options()
    elif isinstance(options, ProcessGroupNCCL.Options):
        nccl_options = options
        options = ProcessGroupCudaP2P.Options()
        options.nccl_options = nccl_options
    elif isinstance(options, ProcessGroupCudaP2P.Options):
        if options.nccl_options is None:
            options.nccl_options = ProcessGroupNCCL.Options()
    else:
        raise TypeError(
            "options for cuda_p2p must be ProcessGroupCudaP2P.Options "
            f"or ProcessGroupNCCL.Options (got: {type(options)})"
        )

    return ProcessGroupCudaP2P(
        dist_backend_opts.store,
        dist_backend_opts.group_rank,
        dist_backend_opts.group_size,
        options,
    )


def is_cuda_p2p_group(group: c10d.ProcessGroup) -> bool:
    try:
        backend = group._get_backend(torch.device("cuda"))
    except Exception:
        return False
    return isinstance(backend, ProcessGroupCudaP2P) and backend.is_p2p_available()


def get_cuda_p2p_backend(group: c10d.ProcessGroup) -> "ProcessGroupCudaP2P":
    if not is_cuda_p2p_group(group):
        raise TypeError("group is not a cuda_p2p process group.")
    return cast(
        ProcessGroupCudaP2P,
        group._get_backend(torch.device("cuda")),
    )


def get_p2p_buffer_size(group: c10d.ProcessGroup) -> int:
    if not is_cuda_p2p_group(group):
        return 0
    backend = get_cuda_p2p_backend(group)
    return backend.get_buffer_size()


c10d.Backend.register_backend(
    "cuda_p2p",
    _create_cuda_p2p_group,
    extended_api=True,
    devices=["cuda"],
)
