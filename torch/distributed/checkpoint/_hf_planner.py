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
    """
    A planner to work with HuggingFace's safetensors format.
    This is a placeholder, as it is likely that the DefaultSavePlanner is enough.
    """
    
