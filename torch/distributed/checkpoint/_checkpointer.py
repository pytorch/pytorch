from concurrent.futures import Future
from typing import Any, Dict, List, Optional

import torch.distributed as dist
import torch.distributed.checkpoint.state_dict_loader as loader
import torch.distributed.checkpoint.state_dict_saver as saver
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.storage import (
    LoadPlanner,
    SavePlanner,
    StorageReader,
    StorageWriter,
)


__all__: List[str] = []


class _Checkpointer:
    """This base class specefies a high level API for saving and loading
    distributed `state_dict` 's. It provides an abstraction over the low-level APIs
    provided by :py:mod:`torch.distributed.checkpoint.storage`, essentially calling
    :py:meth: `torch.distributed.state_dict_saver.save` and
    :py:meth: `torch.distributed.state_dict_loader.load` with the provided storage
    readers and writers.

    .. warning::
        This feature is experimental and subject to removal/change.

    """

    def __init__(
        self,
        storage_writer: StorageWriter,
        storage_reader: StorageReader,
        *,
        process_group: Optional[dist.ProcessGroup] = None,
        coordinator_rank: int = 0,
        no_dist: bool = False,
        load_planner: Optional[LoadPlanner] = None,
        save_planner: Optional[SavePlanner] = None,
    ):
        """Initializes the Checkpointer instance.

        Args:
            storage_writer: Instance of StorageWrite use to perform writes.
            storage_reader: StorageReader used to load data from.
            process_group: ProcessGroup to be used for cross-rank synchronization.
            coordinator_rank: Rank to use to coordinate the checkpoint. rank0 is used by default.
            no_dist: If ``True``, distributed checkpoint will not load in SPMD style. (Default: ``False``)
            loader_planner: Instance of LoadPlanner to use when loading.
            save_planner: Instance of SavePlanner to use when saving.
        """
        self.storage_writer = storage_writer
        self.storage_reader = storage_reader
        self.process_group = process_group
        self.coordinator_rank = coordinator_rank
        self.no_dist = no_dist
        self.load_planner = load_planner
        self.save_planner = save_planner

    def save(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Metadata:
        """Calls :py:meth: `torch.distributed.state_dict_saver.save`. Utilizing values passed during initialization."""
        return saver.save(
            state_dict,
            self.storage_writer,
            process_group=self.process_group,
            coordinator_rank=self.coordinator_rank,
            no_dist=self.no_dist,
            planner=self.save_planner,
        )

    def async_save(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Future:
        """
        Calls :py:meth: `torch.distributed.state_dict_saver._async_save`. Utilizing values passed during initialization.

        Returns:
            Future: A future holding the resultant Metadata object from `save`.
        """
        return saver.async_save(
            state_dict,
            storage_writer=self.storage_writer,
            process_group=self.process_group,
            planner=self.save_planner,
        )

    def load(self, state_dict: Dict[str, Any]) -> None:
        """Calls :py:meth: `torch.distributed.state_dict_loader.load`. Utilizing values passed during initialization."""
        loader.load(
            state_dict,
            storage_reader=self.storage_reader,
            process_group=self.process_group,
            planner=self.load_planner,
        )
