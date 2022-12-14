import dataclasses
import io
from typing import List, Tuple, Dict, Any, Union, cast

import torch

from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._shard.sharded_tensor import ShardedTensor


from .planner import (
    SavePlanner,
    LoadPlanner,
    SavePlan,
    LoadPlan,
    ReadItem,
    WriteItem,
    WriteItemType,
)

from .metadata import (
    BytesStorageMetadata,
    TensorStorageMetadata,
    MetadataIndex,
    Metadata,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
)

from .planner_helpers import (
    _create_read_items,
    _create_write_items,
    _create_default_metadata_only_plan,
)

from .utils import find_state_dict_object

__all__ = [
    "DefaultSavePlanner",
    "DefaultLoadPlanner",
    "create_default_local_load_plan",
    "create_default_global_load_plan",
    "create_default_local_save_plan",
    "create_default_global_save_plan",
]


class DefaultSavePlanner(SavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        self.plan = create_default_local_save_plan(
            self.state_dict, self.is_coordinator
        )
        return self.plan

    def create_global_plan(
        self, all_plans: List[SavePlan]
    ) -> Tuple[List[SavePlan], Metadata]:
        self.global_plan, self.metadata = create_default_global_save_plan(
            all_plans
        )
        return self.global_plan, self.metadata

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        self.plan = new_plan
        return new_plan

    def resolve_data(
        self, write_item: WriteItem
    ) -> Union[torch.Tensor, io.BytesIO]:
        object = self.lookup_object(write_item.index)
        return self.transform_object(write_item, object)

    def lookup_object(self, index: MetadataIndex) -> Any:
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return find_state_dict_object(self.state_dict, index)

    def transform_object(self, write_item: WriteItem, object: Any):
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        if write_item.type == WriteItemType.BYTE_IO:
            bytes = io.BytesIO()
            torch.save(object, bytes)
            object = bytes
        return object


class DefaultLoadPlanner(LoadPlanner):
    def init(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        return create_default_local_load_plan(self.state_dict, self.metadata)

    def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return create_default_global_load_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        self.state_dict[read_item.dest_index.fqn] = torch.load(value)

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.dest_index)
        return self.transform_tensor(read_item, tensor)

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        pass

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return find_state_dict_object(self.state_dict, index)

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return narrow_tensor_by_index(
            tensor, read_item.dest_offsets, read_item.lengths
        )


def create_default_local_load_plan(
    state_dict: Dict[str, Any],
    metadata: Metadata,
) -> LoadPlan:
    requests = []
    """
    Create the ``LoadPlan`` used by DefaultLoadPlanner.

    It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.

    The default behavior is to match key exactly between state_dict and metadata.
    It handles resharding by issuing multiple read requests against storage in order to match
    load requirements.
    """
    for fqn, obj in state_dict.items():
        md = metadata.state_dict_metadata[fqn]
        requests += _create_read_items(fqn, md, obj)

    return LoadPlan(requests)


def create_default_global_load_plan(
    all_plans: List[LoadPlan],
) -> List[LoadPlan]:
    """
    Create global load plan used by DefaultLoadPlanner.

    The default load behavior involved no global coordination and this function
    currently doesn't change the local plans.
    """
    return all_plans


def create_default_local_save_plan(
    state_dict: Dict[str, Any], is_coordinator: bool
) -> SavePlan:
    """
    Create the ``SavePlan`` used by DefaultSavePlanner.

    On non-coordinator ranks, this function ignores tensors and non-tensor objects,
    only producing writes for ShardedTensor objects.

    On the coordinator rank, produce writes for all values.
    """
    requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, ShardedTensor) or is_coordinator:
            requests += _create_write_items(fqn, obj)
    return SavePlan(requests)


def create_default_global_save_plan(
    all_plans: List[SavePlan],
) -> Tuple[List[SavePlan], Metadata]:
    """
    Create the global plan and metadata used by DefaultSavePlanner.

    Metadata is produced by concatenating the metadata of all ``WriteItem`` from the supplied plans.

    The only global planning change is to update index hints in all ``MetadataIndex`` objects.
    """
    md: Dict[str, STORAGE_TYPES] = {}
    new_plans = []
    for plan in all_plans:
        new_items = []
        for item in plan.items:
            if not item.type == WriteItemType.SHARD:
                assert item.index.fqn not in md

            if item.type == WriteItemType.BYTE_IO:
                md[item.index.fqn] = BytesStorageMetadata()
                new_items.append(item)
            else:
                assert item.tensor_data is not None
                tensor_md = cast(
                    TensorStorageMetadata,
                    md.setdefault(
                        item.index.fqn,
                        TensorStorageMetadata(
                            properties=item.tensor_data.properties,
                            size=item.tensor_data.size,
                            chunks=[],
                        ),
                    ),
                )
                new_index = dataclasses.replace(
                    item.index, index=len(tensor_md.chunks)
                )
                new_item = dataclasses.replace(item, index=new_index)
                new_items.append(new_item)

                assert (
                    item.tensor_data.chunk is not None
                ), f"Cannot create MD for tensor without bounds. FQN: {item.index.fqn}"
                tensor_md.chunks.append(item.tensor_data.chunk)
        new_plans.append(dataclasses.replace(plan, items=new_items))
    return (new_plans, Metadata(md))


def _create_default_local_metadata(state_dict: STATE_DICT_TYPE) -> Metadata:
    """
    Return the ``Metadata`` if DefaultSavePlanner was used to checkpoint ``state_dict``.
    """
    plan = _create_default_metadata_only_plan(state_dict)
    _, md = create_default_global_save_plan([plan])
    return md
