from typing import Union

import torch
import torch.distributed.distributed_c10d as c10d
from torch._C._distributed_c10d import (
    _DistributedBackendOptions,
    Backend,
    ProcessGroupCudaP2P,
    ProcessGroupNCCL,
)


def _create_cuda_p2p_group(
    dist_backend_opts: _DistributedBackendOptions,
    options: Union[ProcessGroupCudaP2P.Options, ProcessGroupNCCL.Options, None],
) -> Backend:
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


c10d.Backend.register_backend(
    "cuda_p2p",
    _create_cuda_p2p_group,
    extended_api=True,
    devices=["cuda"],
)
