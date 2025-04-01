# mypy: allow-untyped-defs
from dataclasses import dataclass

from torch.distributed.checkpoint._dedup_save_plans import (
    dedup_save_plans_with_fqn_to_index_mapping,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.planner import ReadItem, SavePlan


__all__ = ["_HuggingFaceSavePlanner", "_HuggingFaceLoadPlanner"]


@dataclass
class _FqnToFileMapping:
    fqn_to_file_index_mapping: dict[str, int]


class _HuggingFaceSavePlanner(DefaultSavePlanner):
    """
    A save planner that dedups the save plans based on the fqn to file index mapping.
    """

    def _dedup_save_plans(self, all_plans: list[SavePlan]) -> list[SavePlan]:
        assert len(all_plans) > 0, "all_plans should not be empty"
        assert all_plans[0].storage_data is not None, "storage_data should not be None"
        assert isinstance(all_plans[0].storage_data, _FqnToFileMapping), (
            "storage_data should be of type _FqnToFileMapping"
        )

        fqn_to_index_mapping: dict[str, int] = all_plans[
            0
        ].storage_data.fqn_to_file_index_mapping

        return dedup_save_plans_with_fqn_to_index_mapping(
            all_plans, fqn_to_index_mapping
        )


class _HuggingFaceLoadPlanner(DefaultLoadPlanner):
    def __init__(self, allow_tensor_resize: bool = False):
        super().__init__()
        self.allow_tensor_resize = allow_tensor_resize

    def resolve_tensor(self, read_item: ReadItem):
        return self.lookup_tensor(read_item.dest_index)
