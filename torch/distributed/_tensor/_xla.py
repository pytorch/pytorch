# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch

import torch.nn as nn
from torch.distributed._tensor.device_mesh import DeviceMesh, mesh_resources
from torch.distributed._tensor.placement_types import Placement, Replicate

log = logging.getLogger(__name__)

TORCH_XLA_INITIALIZED = False
try:
    import torch_xla.core.xla_model as xm  # type:ignore[import]  # noqa: F401
    import torch_xla.runtime as xr  # type:ignore[import]
    from torch_xla.experimental.xla_sharded_tensor import (  # type:ignore[import]
        XLAShardedTensor,
    )
    from torch_xla.experimental.xla_sharding import (  # type:ignore[import]
        mark_sharding,
        Mesh,
        ShardingType,
    )

    TORCH_XLA_INITIALIZED = True
except ImportError as e:
    log.warning(e.msg)


# wrapper to check xla test requirements
def with_xla(func: Callable) -> Callable:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self, *args: Tuple[object], **kwargs: Dict[str, Any]  # type: ignore[misc]
    ) -> None:
        if TORCH_XLA_INITIALIZED:
            # TODO(yeounoh) replace this with xr.use_spmd() when we deprecate the flag.
            os.environ["XLA_USE_SPMD"] = "1"
            func(self, *args, **kwargs)  # type: ignore[misc]
        else:
            raise ImportError(
                "torch.distributed._tensor._xla API requires torch_xla package installation."
            )

    return wrapper


@with_xla
def convert_to_xla_mesh(dt_mesh: DeviceMesh) -> "Mesh":
    assert dt_mesh.size() == xr.global_runtime_device_count()
    return Mesh(
        dt_mesh.mesh.flatten(), list(dt_mesh.mesh.size()), dt_mesh.mesh_dim_names
    )


@with_xla
def convert_to_xla_partition_spec(
    tensor: torch.Tensor, placement_spec: Sequence[Placement]
) -> Tuple[Union[Tuple, int, None]]:
    """
    Transform DTensor `placement_spec` into XLAShardedTensor `partitoin_spec`.
    This supports Placement type Shard and Replicate, and Partial type is restricted
    as it is only applicable to the intermediary results for aggregation. Note that
    this notion is different from PyTorch/XLA SPMD's partial replication scheme.
    """
    # per tensor dimension sharding
    sharding_spec = [None] * len(tensor.shape)
    for mesh_idx, spec in enumerate(placement_spec):
        if spec.is_shard:  # type:ignore[truthy-function]
            # mesh_idx to tensor_idx (spec.dim)
            tensor_idx = spec.dim  # type:ignore[attr-defined]
            sharding_spec[tensor_idx] = mesh_idx  # type:ignore[call-overload]
        elif spec.is_replicate:
            # spec.dim is already set to None by default
            continue
        else:
            raise ValueError(f"Unsupported placement type: {type(spec).__name__}")
    return tuple(sharding_spec)  # type:ignore[return-value]


@with_xla
def xla_distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> "XLAShardedTensor":
    """
    Distribute a torch.Tensor to the `device_mesh` according to the `placements`
    specified. The rank of `device_mesh` and `placements` must be the same.

    Args:
        tensor (torch.Tensor): torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use `torch.chunk`
            semantic to shard the tensor and scatter the shards.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as `device_mesh.ndim`. If not specified, we will
            by default replicate the tensor across the `device_mesh` from the
            first rank of each dimension of the `device_mesh`.

    Returns:
        A :class:`XLAShardedTensor` object
    """
    # get default device mesh if there's nothing specified
    dt_mesh = device_mesh or mesh_resources.get_current_mesh()
    assert dt_mesh.device_type == "xla"

    # convert to XLA device mesh
    xla_mesh = convert_to_xla_mesh(dt_mesh)
    assert xla_mesh.mesh_shape == list(dt_mesh.mesh.size())

    # convert tensor to the corresponding device type if it's not in that device type
    if not tensor.is_meta:
        tensor = tensor.to(dt_mesh.device_type)
    # set default placements to replicated if not specified
    if placements is None:
        placements = [Replicate() for _ in range(dt_mesh.ndim)]
    assert (
        len(placements) == dt_mesh.ndim
    ), "`placements` must have the same length as `device_mesh.ndim`! "
    f"Found placements length: {len(placements)}, and device_mesh.ndim: {dt_mesh.ndim}."
    # convert placements to xla partition spec
    partition_spec = convert_to_xla_partition_spec(tensor, placements)
    assert len(tensor.shape) == len(
        partition_spec
    ), "`partition_spec` from `placements` must have the same length as `tensor.length`! "
    f"Found tensor shape length: {len(tensor.shape)}, and partition_spec length: {len(partition_spec)}."

    global_tensor = tensor
    if type(tensor).__name__ == "DTensor":
        raise ValueError(
            "Cannot distribute a DTensor with local tensor on xla devices."
            "The input tensor must be global."
        )
    if type(tensor).__name__ == "XLAShardedTensor":
        sharding_type = tensor.sharding_type  # type:ignore[attr-defined]
        assert (
            sharding_type is None or sharding_type == ShardingType.REPLICATED
        ), "XLAShardedTensor `tensor` is already annotated with non-replication sharding. "
        "Clear the existing sharding annotation first, by callling torch_xla.experimental.xla_sharding.clear_sharding API."
        global_tensor = tensor.global_tensor  # type:ignore[attr-defined]
    assert global_tensor is not None, "distributing a tensor should not be None"

    # Annotates sharding and returns an XLAShardedTensor
    xla_tensor = mark_sharding(global_tensor, xla_mesh, partition_spec)
    return xla_tensor


@with_xla
def xla_distribute_module(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]] = None,
    input_fn: Optional[Callable[..., None]] = None,
    output_fn: Optional[Callable[..., None]] = None,
) -> nn.Module:
    raise NotImplementedError
