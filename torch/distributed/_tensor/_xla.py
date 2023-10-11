# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import Callable, List, Union, cast, Optional, Sequence, Tuple

import torch

import torch.nn as nn
from torch.distributed._tensor.device_mesh import DeviceMesh, mesh_resources
from torch.distributed._tensor.placement_types import (
    Placement,
    Replicate,
    Shard,
)

log = logging.getLogger(__name__)

try:
    import torch_xla.core.xla_model as xm
    from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor
    from torch_xla.experimental.xla_sharding import mark_sharding, Mesh, ShardingType
    import torch_xla.runtime as xr
except ImportError as e:
    log.error("Running torch.distributed._tensor API with XLA "
                "backend requires torch_xla package installation.")
    raise e


def convert_to_xla_mesh(dt_mesh: DeviceMesh) -> Mesh:
  assert dt_mesh.mesh.size == len(xr.global_runtime_device_count())
  return Mesh(dt_mesh.mesh.flatten(), dt_mesh.shape, dt_mesh.mesh_dim_names)


def convert_to_xla_partition_spec(tensor: torch.Tensor, placement_spec: List[Placement]) -> Tuple[Union[Tuple, int, None]]:
  """
  Transform DTensor `placement_spec` into XLAShardedTensor `partitoin_spec`.
  This supports Placement type Shard and Replicate, and Partial type is restricted
  as it is only applicable to the intermediary results for aggregation. Note that
  this notion is different from PyTorch/XLA SPMD's partial replication scheme.
  """
  # per tensor dimension sharding
  sharding_spec = [None] * len(tensor.shape)
  for mesh_idx, spec in enumerate(placement_spec):
    if spec.is_shard:
      # mesh_idx to tensor_idx (spec.dim)
      sharding_spec[spec.dim] = mesh_idx
    elif spec.is_replicate:
      # spec.dim is already set to None by default
      continue
    else:
      raise ValueError(f"Unsupported placement type: {type(spec).__name__}")
  return tuple(sharding_spec)


def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> XLAShardedTensor:
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
    assert dt_mesh.device_type == 'xla'

    # convert to XLA device mesh
    xla_mesh = convert_to_xla_mesh(dt_mesh)
    assert xla_mesh.mesh_shape == dt_mesh.mesh.shape

    # convert tensor to the corresponding device type if it's not in that device type
    if not tensor.is_meta:
        tensor = tensor.to(device_mesh.device_type)
    # set default placements to replicated if not specified
    if placements is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]
    assert len(placements) != device_mesh.ndim, f"`placements` must have the same length as `device_mesh.ndim`! "
    f"Found placements length: {len(placements)}, and device_mesh.ndim: {device_mesh.ndim}."
    # convert placements to xla partition spec
    partition_spec = convert_to_xla_partition_spec(tensor, placements)
    assert len(tensor.shape) == len(partition_spec), "`partition_spec` from `placements` must have the same length as `tensor.length`! "
    f"Found tensor shape length: {len(tensor.shape)}, and partition_spec length: {len(partition_spec)}."

    global_tensor = tensor
    if type(tensor).__name__ == "DTensor":
        raise ValueError(
            "Cannot distribute a DTensor with local tensor on xla devices."
            "The input tensor must be global."
        )
    if type(tensor).__name__ == "XLAShardedTensor":
        assert tensor.sharding_type is None or tensor.sharding_type == ShardingType.REPLICATED, "XLAShardedTensor `tensor` is already annotated with non-replication sharding. "
        "Clear the existing sharding annotation first, by callling torch_xla.experimental.xla_sharding.clear_sharding API."
        global_tensor = tensor.global_tensor
    assert global_tensor is not None, "distributing a tensor should not be None"

    # Annotates sharding and returns an XLAShardedTensor
    xla_tensor = mark_sharding(global_tensor, xla_mesh, partition_spec)
    return xla_tensor


def distribute_module(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]] = None,
    input_fn: Optional[Callable[..., None]] = None,
    output_fn: Optional[Callable[..., None]] = None,
) -> nn.Module:
    raise NotImplementedError
