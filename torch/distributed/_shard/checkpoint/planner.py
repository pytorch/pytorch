import abc
from dataclasses import dataclass
import io
from typing import List, Tuple, Any, Union, Optional

from enum import Enum, auto
import torch

from torch.distributed._shard.sharded_tensor.metadata import TensorProperties

from .metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    Metadata,
    STATE_DICT_TYPE
)

class WriteItemType(Enum):
    TENSOR = auto()
    SHARD = auto()
    BYTE_IO = auto()

class LoadItemType(Enum):
    TENSOR = auto()
    BYTE_IO = auto()

@dataclass(frozen=True)
class TensorWriteData:
    chunk: ChunkStorageMetadata
    properties: TensorProperties
    size: torch.Size

@dataclass(frozen=True)
class WriteItem:
    index: MetadataIndex
    type: WriteItemType

    # Value present if it's a tensor write
    tensor_data: Optional[TensorWriteData] = None

@dataclass(frozen=True)
class ReadItem:
    # Read Item
    type: LoadItemType

    # Index into the state_dict
    dest_index: MetadataIndex
    # Offsets into destination tensor
    dest_offsets: torch.Size

    # Index into the checkpoint
    storage_index: MetadataIndex
    # Offset into the checkpoint data
    storage_offsets: torch.Size

    # Size of the hypercube to copy
    lengths: torch.Size

@dataclass(frozen=True)
class SavePlan:
    items: List[WriteItem]
    storage_data: Any = None
    planner_data: Any = None

@dataclass
class LoadPlan:
    items: List[ReadItem]
    storage_data: Any = None
    planner_data: Any = None

class SavePlanner(abc.ABC):
    """
    Abstract class defining the protocol used by save_state_dict to plan the save process.
    """

    @abc.abstractmethod
    def init(self, state_dict: STATE_DICT_TYPE, is_coordinator: bool) -> None:
        """
        Intialize this planner to save ``state_dict``.

        This is called on all ranks.
        """
        pass

    @abc.abstractmethod
    def create_local_plan(self) -> SavePlan:
        """
        Compute the save plan for the current rank.
        This will be aggregated and passed to create_global_plan.
        Planner specific data can be passed through SavePlan::planner_data.

        This is called on all ranks.
        """
        pass

    @abc.abstractmethod
    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        """
        Compute the global checkpoint plan and return the local plan of each rank.

        This is called on the coordinator rank only.
        """
        pass

    @abc.abstractmethod
    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        """
        Merge the plan created by `create_local_plan` and the result of `create_global_plan`.

        This is called on all ranks.
        """
        pass

    @abc.abstractmethod
    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        """
        Lookup the object associated with ``write_item``in `state_dict` and apply any
        transformation (such as serialization) prior to the storage layer consuming it.

        Called on each rank multiple times, at least once per WriteItem in the final SavePlan.

        This method should be idepotent and thread-save. StorageWriter implementations
        are free to call it as frequently as they need.

        Any transformation that allocates memory should be lazily done when his method
        is called in order to reduce peak memory required by checkpointing.

        When returning tensors, they can be on any device or format, they can be views too.
        It's the storage layer responsiblity to figure out how to save them.
        """
        pass

class LoadPlanner:
    """
    Abstract class defining the protocol used by load_state_dict to plan the load process.

    """
    @abc.abstractmethod
    def init(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        """
        Initialize this instance to load data into ``state_dict``

        . N.B. This is called on every rank.
        """
        pass

    @abc.abstractmethod
    def create_local_plan(self) -> LoadPlan:
        """
        Create a LoadPlan based on state_dict and metadata provided by init.

        . N.B. This is called on every rank.
        """
        pass

    @abc.abstractmethod
    def create_global_plan(self, globla_plan: List[LoadPlan]) -> List[LoadPlan]:
        """
        Compute the global load plan and return plans for each rank.

        . N.B. This is called on the coordinator rank only
        """
        pass

    @abc.abstractmethod
    def finish_plan(self, central_plan: LoadPlan) -> LoadPlan:
        """
        Accept the plan from coordinator and return final LoadPlan.
        """
        pass

    @abc.abstractmethod
    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        """
        Load the item described by ``read_item``and ``value``.

        This method is expected to modify in-place the underlying state_dict.

        The contents of ``value`` are defined by the SavePlanner used to produce
        the checkpoint being loaded.
        """
        pass

    @abc.abstractmethod
    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        """
        Return the tensor described by ``read_item`` to be used by the StorageReader to load `read_item`.

        The tensor should alias with one on the underlying state_dict as StorageReader will replace its contents.
        If, for any reason, that's not possible, the planner can use the ``commit_tensor`` method to copy the data
        back to the one in state_dict.
        """
        pass

    @abc.abstractmethod
    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        """
        This method is called once the StorageReader finished loading data into ``tensor``.

        The provided tensor is the same one returned by the call to ``resolve_tensor``.
        This method is only needed if this LoadPlanner needs to post process ``tensor`` prior to
        copying it back to the one in the state_dict.

        The contents of tensor will follow its device synchronization model.
        """
        pass
