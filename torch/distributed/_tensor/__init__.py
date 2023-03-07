# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Callable, cast, Optional, Sequence

# Import all builtin dist tensor ops
import torch.distributed._tensor.ops
from torch.distributed._tensor.api import DTensor, distribute_tensor, distribute_module
from torch.distributed._tensor.device_mesh import DeviceMesh, get_global_device_mesh
from torch.distributed._tensor.placement_types import Placement, Replicate, Shard


# All public APIs from dtensor package
__all__ = [
    "DTensor",
    "DeviceMesh",
    "distribute_tensor",
    "distribute_module",
    "Shard",
    "Replicate",
]
