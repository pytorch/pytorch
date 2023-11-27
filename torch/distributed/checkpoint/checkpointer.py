from typing import Optional, Dict, Any

import torch.distributed as dist
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.storage import SavePlanner, LoadPlanner
from torch.distributed.checkpoint.storage import StorageReader, StorageWriter
import torch.distributed.checkpoint.state_dict_saver as saver
import torch.distributed.checkpoint.state_dict_loader as loader

__all__ = ["Checkpointer"]

class Checkpointer:
    """
    This base class specefies a high level API for saving and loading
    distributed `state_dict` 's. It provides an abstraction over the low-level APIs
    provided by :py:mod:`torch.distributed.checkpoint.storage`, essentially calling
    :py:meth: `torch.distributed.state_dict_saver.save` and
    :py:meth: `torch.distributed.state_dict_loader.load` with the provided storage
    readers and writers.

    """

    def __init__(
        self,
        storage_writer: StorageReader,
        storage_reader: StorageWriter,
        process_group: Optional[dist.ProcessGroup] = None,
        coordinator_rank: int = 0,
        no_dist: bool = False,
        planner: Optional[LoadPlanner] = None,
    ):
    """
    Args:
        storage_writer (StorageWriter):
            Instance of StorageWrite use to perform writes.
        storage_reader (StorageReader): StorageReader used to load data from.
        process_group (ProcessGroup):
            ProcessGroup to be used for cross-rank synchronization.
        coordinator_rank (int):
            Rank to use to coordinate the checkpoint.
            rank0 is used by default.
        no_dist (bool): If ``True``, distributed checkpoint will not load
            in SPMD style. (Default: ``False``)
    """
        self.storage_writer = storage_writer
        self.storage_reader = storage_reader
        self.process_group = process_group
        self.coordinator_rank = coordinator_rank
        self.no_dist = no_dist
        self.planner = planner

    def save(
        self,
        state_dict: STATE_DICT_TYPE,
        *,
        storage_writer: Optional[StorageWriter] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        coordinator_rank: int = 0,
        no_dist: bool = False,
        planner: Optional[SavePlanner] = None,
    ):
        """
            Calls :py:meth: `torch.distributed.state_dict_saver.save`.
            This method will default to the values passed during initialization.

        """
        storage_writer = storage_writer or self.storage_writer
        process_group = process_group or self.process_group
        coordinator_rank = coordinator_rank or self.coordinator_rank
        no_dist = no_dist or self.no_dist
        planner = planner or self.planner

        saver.save(
            state_dict,
            storage_writer,
            process_group,
            coordinator_rank,
            no_dist,
            planner
        )

    def load(
            self,
            state_dict: Dict[str, Any],
            *,
            storage_reader: Optional[StorageReader] = None,
            process_group: Optional[dist.ProcessGroup] = None,
            coordinator_rank: int = 0,
            no_dist: bool = False,
            planner: Optional[LoadPlanner] = None,
    ):
        """
        Calls :py:meth: `torch.distributed.state_dict_loader.load`.
        This method will default to the values passed during initialization.
        """
        storage_reader = storage_reader or self.storage_reader
        process_group = process_group or self.process_group
        coordinator_rank = coordinator_rank or self.coordinator_rank
        no_dist = no_dist or self.no_dist
        planner = planner or self.planner

        loader.load(
            state_dict,
            storage_reader,
            process_group,
            coordinator_rank,
            no_dist,
            planner
        )
