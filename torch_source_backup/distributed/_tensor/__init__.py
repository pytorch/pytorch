"""
NOTICE: DTensor has moved to torch.distributed.tensor

This file is a shim to redirect to the new location, and
we keep the old import path starts with `_tensor` for
backward compatibility. We will remove this folder once
we resolve all the BC issues.
"""

import sys
from importlib import import_module


submodules = [
    # TODO: _shards_wrapper/_utils here mainly for checkpoint BC, remove them
    "_shards_wrapper",
    "_utils",
    "experimental",
    "device_mesh",
]

# Redirect imports
for submodule in submodules:
    full_module_name = f"torch.distributed.tensor.{submodule}"
    sys.modules[f"torch.distributed._tensor.{submodule}"] = import_module(
        full_module_name
    )

from torch.distributed.tensor import (  # noqa: F401
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    empty,
    full,
    init_device_mesh,
    ones,
    Partial,
    Placement,
    rand,
    randn,
    Replicate,
    Shard,
    zeros,
)
