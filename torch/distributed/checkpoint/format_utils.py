import os
from typing import Union

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    Metadata,
    STATE_DICT_TYPE,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

__all__ = ["dcp_to_torch_save"]


class _DCPToTorchLoadPlanner(DefaultLoadPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        # rebuild the state dict from the metadata
        for k, v in metadata.state_dict_metadata.items():
            if isinstance(v, TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)
            if k in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[k], v)
            else:
                state_dict[k] = v

        super().set_up_planner(state_dict, metadata, is_coordinator)


def dcp_to_torch_save(
    dcp_checkpoint_dir: Union[str, os.PathLike],
    torch_save_fn: Union[str, os.PathLike],
    coordinator_rank: int = 0,
):
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch save file.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        torch_save_fn: Filename to store the converted Torch save file.
        coordinator_rank: Rank that will perform the conversion.
    """
    if not dist.is_initialized() or dist.get_rank() == coordinator_rank:
        sd = {}
        storage_reader = FileSystemReader(dcp_checkpoint_dir)

        _load_state_dict(
            sd,
            storage_reader=storage_reader,
            planner=_DCPToTorchLoadPlanner(),
            no_dist=True,
        )
        torch.save(sd, torch_save_fn)

    if dist.is_initialized():
        dist.barrier()
