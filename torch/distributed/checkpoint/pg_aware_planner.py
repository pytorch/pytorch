# Copyright (c) Meta Platforms, Inc. and affiliates
import io
from typing import Any, Dict, List

import torch
import torch.distributed.distributed_c10d as c10d
from torch import distributed as dist
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
    Metadata,
)
from torch.distributed.checkpoint.planner import ReadItem
from torch.distributed._shard.sharded_tensor.api import ShardedTensor


def get_ranks(pg: c10d.ProcessGroup) -> List[int]:
    """
    Return an array of global ranks for a given process group.
    """
    return [c10d.get_global_rank(pg, i) for i in range(pg.size())]


def create_state_dict_copy(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the ``state_dict_copy`` used by ProcessGroupAwareSavePlanner and ProcessGroupAwareLoadPlanner.
    It identifies ShardedTensor instances that are using the non-default
    process groups and change their FQNs to be prefixed by a PG-specific string.
    """
    state_dict_copy = {}

    for k, v in state_dict.items():
        # Only rename the fqn if the current process group is a sub-process group.
        if (
            isinstance(v, ShardedTensor)
            and v._process_group != dist.distributed_c10d._get_default_group()
        ):
            pg_global_ranks = get_ranks(v._process_group)
            fqn = "_".join([str(rank) for rank in pg_global_ranks]) + "_" + k
            state_dict_copy[fqn] = v
        else:
            state_dict_copy[k] = v

    return state_dict_copy


class ProcessGroupAwareSavePlanner(DefaultSavePlanner):
    """
    ProcessGroupAwareSavePlanner extends DefaultSavePlanner and re-write the state_dict.
    """

    def set_up_planner(
        self, state_dict: Dict[str, Any], is_coordinator: bool
    ) -> None:
        """
        Rename all keys of sharded tensors from sub-process groups by prefixing it
        with a PG specific string.
        """
        self.state_dict = create_state_dict_copy(state_dict)  # pyre-ignore[16]
        super().set_up_planner(self.state_dict, is_coordinator)


class ProcessGroupAwareLoadPlanner(DefaultLoadPlanner):
    """
    ProcessGroupAwareSaveLoader extends DefaultLoadPlanner and re-write the state_dict.
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        """
        Rename all keys of sharded tensors from sub-process groups by prefixing it
        with a PG specific string.
        """
        self.pg_original_state_dict = state_dict  # pyre-ignore[16]
        self.state_dict = create_state_dict_copy(state_dict)  # pyre-ignore[16]
        super().set_up_planner(self.state_dict, metadata, is_coordinator)

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        """
        This method makes sure that the non sharded_tensor value of the original_state_dict
        also gets loaded properly.
        """
        fqn: str = read_item.dest_index.fqn
        self.pg_original_state_dict[fqn] = torch.load(value)  # pyre-ignore[16]
