from contextlib import contextmanager
from typing import Optional

import torch
from torch.distributed import distributed_c10d
from torch.distributed import rpc

# Tracks the current process group in the load context manager.
_CURRENT_PROCESS_GROUP = None

@contextmanager
def load_with_process_group(process_group):
    """
    Context manager to set the process group with which to load a ShardedTensor.
    """
    global _CURRENT_PROCESS_GROUP
    if _CURRENT_PROCESS_GROUP is not None:
        raise RuntimeError(
            'ProcessGroup already set by previous "load_with_process_group" '
            'context manager')
    _CURRENT_PROCESS_GROUP = process_group
    try:
        yield process_group
    finally:
        _CURRENT_PROCESS_GROUP = None

def _parse_and_validate_remote_device(pg, remote_device):

    worker_name = remote_device.worker_name()
    rank = remote_device.rank()
    device = remote_device.device()

    # Validate rank, skip validation if rank is not part of process group.
    if not distributed_c10d._rank_not_in_group(pg):
        if rank is not None and (rank < 0 or rank >= distributed_c10d.get_world_size(pg)):
            raise ValueError(f'Invalid rank: {rank}')

    if worker_name is not None:
        if not rpc._is_current_rpc_agent_set():
            raise RuntimeError(f'RPC framework needs to be initialized for using worker names: {worker_name}')

        workers = rpc._get_current_rpc_agent().get_worker_infos()
        for worker in workers:
            if worker.name == worker_name:
                return worker.id, device

        raise ValueError(f'Invalid worker name: {worker_name}')

    return rank, device

def _validate_output_tensor_for_gather(
    my_rank: int,
    dst_rank: int,
    size: torch.Size,
    dst_tensor: Optional[torch.Tensor],
) -> None:
    if dst_rank == my_rank:
        if dst_tensor is None:
            raise ValueError(
                f"Argument ``dst_tensor`` must be specified on destination rank {dst_rank}"
            )
        if tuple(size) != (dst_tensor.size()):
            raise ValueError(
                f"Argument ``dst_tensor`` have size {tuple(dst_tensor.size())},"
                f"but should be {tuple(size)}"
            )
    elif dst_tensor:
        raise ValueError(
            "Argument ``dst_tensor`` must NOT be specified "
            "on non-destination ranks."
        )
