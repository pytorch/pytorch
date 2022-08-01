import io
from typing import List, Tuple, Dict, Any, Union

import torch

from torch.distributed._shard._utils import narrow_tensor_by_index

from .resharding import (
    create_default_local_save_plan,
    create_default_global_save_plan,
    create_default_local_load_plan,
    create_default_global_load_plan,
)

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
    MetadataIndex,
    Metadata,
    STATE_DICT_TYPE
)

from .utils import (
    find_state_dict_object
)


class DefaultSavePlanner(SavePlanner):
    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        self.plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        return self.plan

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        self.global_plan, self.metadata = create_default_global_save_plan(all_plans)
        return self.global_plan, self.metadata

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        self.plan = new_plan
        return new_plan

    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
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
    def init(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
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
        return narrow_tensor_by_index(tensor, read_item.dest_offsets, read_item.lengths)
