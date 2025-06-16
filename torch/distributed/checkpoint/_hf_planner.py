# mypy: allow-untyped-defs

from torch.distributed.checkpoint.default_planner import (
    create_default_local_load_plan,
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.planner import LoadPlan


__all__ = ["_HuggingFaceSavePlanner", "_HuggingFaceLoadPlanner"]


class _HuggingFaceSavePlanner(DefaultSavePlanner):
    """
    A planner to work with HuggingFace's safetensors format.
    This is a placeholder, as it is likely that the DefaultSavePlanner is enough.
    """


class _HuggingFaceLoadPlanner(DefaultLoadPlanner):
    def __init__(self, allow_tensor_resize: bool = False):
        super().__init__()
        self.allow_tensor_resize = allow_tensor_resize

    def create_local_plan(self) -> LoadPlan:
        assert self.metadata is not None

        # check_md_size is added to avoid the check if we're allowing tensor resize.
        # This will be deprecated in favor of _load_state_dict_from_keys and then we
        # can remove this planner all together.
        return create_default_local_load_plan(
            self.state_dict,
            self.metadata,
            not self.allow_partial_load,
            check_md_size=not self.allow_tensor_resize,
        )
