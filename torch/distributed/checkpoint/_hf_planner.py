# mypy: allow-untyped-defs

from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.planner import ReadItem


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

    def resolve_tensor(self, read_item: ReadItem):
        return self.lookup_tensor(read_item.dest_index)
