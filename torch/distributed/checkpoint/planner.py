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
    STATE_DICT_TYPE,
)


__all__ = [
    "WriteItemType",
    "LoadItemType",
    "TensorWriteData",
    "WriteItem",
    "ReadItem",
    "SavePlan",
    "LoadPlan",
    "SavePlanner",
    "LoadPlanner",
]


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

    SavePlanners are stateful objects that can be used to customize the whole save process.

    SavePlanner acts as an access proxy to the state_dict, so any transfomation done to it
    will be visible to the whole process.

    A planner subclass can expect the following sequence of calls during save_state_dict:

    1) set_up_planner - called on all ranks.
        Signals the start of a checkpoint save.

    2) create_local_plan - called on all ranks.
        Process the state_dict and produces a `SavePlan` that will be sent for global planning.

    3) create_global_plan - called on the coordinator rank only.
        Takes the SavePlan from all ranks and make any global decision.

    4) finish_plan - called on all ranks.
        This gives each rank a chance to adjust to global planning decisions.

    5) resolve_data - called multiple times on each rank
        Lookups a value on the `state_dict` for the storage layer to write.

    Users are recomended to extend DefaultSavePlanner instead of this interface directly as
    most changes can be expressed by changes in a single method.

    There are 3 usual patterns of extension:

    Rewriting state_dict. This is the simplest way to extend the save process as it
    doesn't requite understanding the intrincacies of how SavePlan works:

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class RenamePlanner(DefaultSavePlanner):
    >>>     def set_up_planner(self, state_dict, is_coordinator):
    >>>         # prefix all keys with `foo_``
    >>>         super().set_up_planner(self, {"foo_" + k: v for k, v in state_dict.items()}, is_coordinator)

    Modifying local plan and lookup in tandem. This is useful when fine control of how data is persisted

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class FP16Planner(DefaultSavePlanner):
    >>>     def create_local_plan(self):
    >>>         plan = super().create_local_plan()
    >>>         for p in plan:
    >>>             if p.tensor_data is not None:
    >>>                 p.tensor_data.properties.dtype = torch.float16
    >>>
    >>>     def resolve_data(self, write_item):
    >>>         item = super().resolve_data(write_item)
    >>>         return item if write_item.type == WriteItemType.BYTE_IO else item.to(torch.float16)

    Using the global planning step to make central decisions that can't be made individually by each rank

    >>> # xdoctest: +SKIP("undefined vars")
    >>> from itertools import islice
    >>> from dataclasses import replace
    >>> class DDPLoadBalancingPlanner(DefaultSavePlanner):
    >>>     # This uses the default local plan behavior of having all non-sharded writes in rank 0
    >>>     # This sample doesn't handle ShardedTensors
    >>>     def create_global_plan(self, all_plans):
    >>>         def chunk(it, size):
    >>>             it = iter(it)
    >>>         return list(iter(lambda: tuple(islice(it, size)), ()))
    >>>         all_plans = [
    >>>             replace(plan, items=items) for plan, items in
    >>>                 zip(all_plans, chunk(all_plans[0].items, len(all_plans)))
    >>>         ]
    >>>         return super().create_global_plan(all_plans)

    Finally, some planners need to save additional metadata in the checkpoint, this is
    accomplished by having each rank contribute their data items in the local plan and
    the global planner aggregate them:

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class SaveExtraDataPlanner(DefaultSavePlanner):
    >>>     def create_local_plan(self) -> SavePlan:
    >>>         plan = super().create_local_plan()
    >>>         return replace(plan, planner_data="per-rank-data")
    >>>
    >>>     def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
    >>>         global_plan, metadata = super().create_global_plan(all_plans)
    >>>         merged_data = [p.planner_data for p in global_plan]
    >>>         metadata = replace(metadata, planner_data=merged_data)
    >>>         return global_plan, metadata
    """

    @abc.abstractmethod
    def set_up_planner(self, state_dict: STATE_DICT_TYPE, is_coordinator: bool) -> None:
        """
        Intialize this planner to save ``state_dict``.

        Implementations should save those values as they won't be provided lated in the save process.

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
    def create_global_plan(
        self, all_plans: List[SavePlan]
    ) -> Tuple[List[SavePlan], Metadata]:
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
    def resolve_data(
        self, write_item: WriteItem
    ) -> Union[torch.Tensor, io.BytesIO]:
        """
        Lookup the object associated with ``write_item`` in ``state_dict`` and apply any
        transformation (such as serialization) prior to the storage layer consuming it.

        Called on each rank multiple times, at least once per WriteItem in the final SavePlan.

        This method should be idepotent and thread-save. StorageWriter implementations
        are free to call it as frequently as they need.

        Any transformation that allocates memory should be lazily done when his method
        is called in order to reduce peak memory required by checkpointing.

        When returning tensors, they can be on any device or format, they can be views too.
        It's the storage layer responsibility to figure out how to save them.
        """
        pass


class LoadPlanner:
    """
    Abstract class defining the protocol used by load_state_dict to plan the load process.

    LoadPlanner are stateful objects that can be used to customize the whole load process.

    LoadPlanner acts as an access proxy to the state_dict, so any transfomation done to it
    will be visible to the whole process.

    A planner subclass can expect the following sequence of calls during load_state_dict:

    1) set_up_planner - called on all ranks.
        Signals the start of loading a checkpoint.

    2) create_local_plan - called on all ranks.
        Process the state_dict and produces a `LoadPlan` that will be sent for global planning.

    3) create_global_plan - called on the coordinator rank only.
        Takes the LoadPlan from all ranks and make any global decision.

    4) load_bytes - called multiple times on each rank
        This is called once per non-tensor value in state_dict.

    5) resolve_tensor and commit_tensor - called multiple times on each rank
        They are called in pair for each Tensor value in state_dict.

    Users are recomended to extend DefaultLoadPlanner instead of this interface directly as
    most changes can be expressed by changes in a single method.

    There are two usual patterns of extension:

    Rewriting state_dict. This is the simplest way to extend the load process as it
    doesn't requite understanding the intrincacies of how LoadPlan works. We need
    to keep a reference to the original state_dict as load happens in place so
    we need to be able to perform it in place

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class RenamePlanner(DefaultLoadPlanner):
    >>>     def set_up_planner(self, state_dict, metadata, is_coordinator):
    >>>         self.original_state_dict = state_dict
    >>>         super().set_up_planner(self, {"foo_" + k: v for k, v in state_dict.items()}, is_coordinator)
    >>>
    >>>     def load_bytes(self, read_item, value):
    >>>         # Remove the "foo_" prefix
    >>>         self.original_state_dict[read_item.dest_index.fqn[4:]] = torch.load(value)


    Modifying resolve_tensor and commit_tensor to handle load time transformation.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class MetaModelMaterialize(DefaultSavePlanner):
    >>>     def resolve_tensor(self, read_item):
    >>>         tensor = super().resolve_tensor(read_item)
    >>>         return torch.empty_like(tensor, device="cpu")
    >>>
    >>>     def commit_tensor(self, read_item, tensor):
    >>>         self.state_dict[read_item.dest_index.fqn] = tensor
    """

    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        """
        Initialize this instance to load data into ``state_dict``

        . N.B. This is called on every rank.
        """
        pass

    @abc.abstractmethod
    def create_local_plan(self) -> LoadPlan:
        """
        Create a LoadPlan based on state_dict and metadata provided by set_up_planner.

        . N.B. This is called on every rank.
        """
        pass

    @abc.abstractmethod
    def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
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
