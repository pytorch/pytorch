# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.distributed.tensor._ops  # force import all built-in dtensor ops
from typing_extensions import TypeAliasType
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor._api import (
    distribute_module,
    distribute_tensor,
    DTensor,
    empty,
    full,
    ones,
    rand,
    randn,
    zeros,
)
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.optim.optimizer import (
    _foreach_supported_types as _optim_foreach_supported_types,
)
from torch.utils._foreach_utils import (
    _foreach_supported_types as _util_foreach_supported_types,
)


# All public APIs from dtensor package
__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "Shard",
    "Replicate",
    "Partial",
    "Placement",
    "ones",
    "empty",
    "full",
    "rand",
    "randn",
    "zeros",
]

# For weights_only torch.load
from ._dtensor_spec import (
    DTensorSpec as _DTensorSpec,
    ShardOrderEntry as _ShardOrderEntry,
    TensorMeta as _TensorMeta,
)


torch.serialization.add_safe_globals(
    [
        DeviceMesh,
        _DTensorSpec,
        _TensorMeta,
        _ShardOrderEntry,
        DTensor,
        Partial,
        Replicate,
        Shard,
        _StridedShard,
    ]
)


# Append DTensor to the list of supported types for foreach implementation for optimizer
# and clip_grad_norm_ so that we will try to use foreach over the for-loop implementation on CUDA.
if DTensor not in _optim_foreach_supported_types:
    _optim_foreach_supported_types.append(DTensor)

if DTensor not in _util_foreach_supported_types:
    _util_foreach_supported_types.append(DTensor)  # type: ignore[arg-type]


# Replace __module__ reassignments with TypeAliasType for better type checker / linter compatibility
# DTensor is a class used in isinstance() checks, so keep the __module__ assignment for it
DTensor.__module__ = "torch.distributed.tensor"
distribute_tensor: TypeAliasType = TypeAliasType("distribute_tensor", distribute_tensor)
distribute_module: TypeAliasType = TypeAliasType("distribute_module", distribute_module)
ones: TypeAliasType = TypeAliasType("ones", ones)
empty: TypeAliasType = TypeAliasType("empty", empty)
full: TypeAliasType = TypeAliasType("full", full)
rand: TypeAliasType = TypeAliasType("rand", rand)
randn: TypeAliasType = TypeAliasType("randn", randn)
zeros: TypeAliasType = TypeAliasType("zeros", zeros)

# Register DTensor dispatch for higher order operators
from torch._higher_order_ops.print import _register_dtensor_impl


_register_dtensor_impl()
