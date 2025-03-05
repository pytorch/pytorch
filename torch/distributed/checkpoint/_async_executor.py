# pyre-strict
# mypy: allow-untyped-defs
import abc
import os
from concurrent.futures import Future
from typing import Optional, Union

import torch.distributed as dist
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter


class _AsyncCheckpointExecutor(abc.ABC):
    @abc.abstractmethod
    def execute_save(
        self,
        staged_state_dict: STATE_DICT_TYPE,
        *,
        checkpoint_id: Union[str, os.PathLike, None] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Future:
        """
        Execute the checkpoint save request asynchronously.

        This method is intended to be used as an abstraction for
        implementing async checkpointing. The actual checkpoint save
        operation is executed in a separate thread or process depending
        on the implementation of this interface.
        """
