# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.nn as nn
from typing import Sequence, Tuple
from torch.distributed._tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    Shard,
    Replicate,
    DeviceMesh,
    Placement,
)
from torch.distributed._tensor.parallel import TensorParallelMultiheadAttention
from torch.distributed._tensor.parallel.style import ParallelStyle, PairwiseParallel
from torch.distributed._tensor.parallel.utils import _create_1d_device_mesh


def replicate_input(
    inputs: Sequence[torch.Tensor], device_mesh: DeviceMesh
) -> Tuple[DTensor, ...]:
    replicate = [Replicate()] * device_mesh.ndim
    return tuple(
        DTensor.from_local(tensor, device_mesh, replicate) for tensor in inputs
    )


def replicate_output(output: DTensor, device_mesh: DeviceMesh) -> torch.Tensor:
    if isinstance(output, DTensor):
        replicate = [Replicate()] * output.device_mesh.ndim
        # TODO: can the output be left incontiguous?
        return (
            output.redistribute(output.device_mesh, replicate)
            .to_local()
            .contiguous()
        )


def tp_shard_self_attn(
    name: str, module: nn.Module, device_mesh: DeviceMesh
) -> None:
    col_wise_sharding: Sequence[Placement] = [Shard(0)]
    row_wise_sharding: Sequence[Placement] = [Shard(1)]
    replicate: Sequence[Placement] = [Replicate()] * device_mesh.ndim

    def _shard_self_attn_params(name: str, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if name == "qkv":
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, col_wise_sharding
                    )
                )
                module.register_parameter("weight", sharded_weight)
                if module.bias is not None:
                    sharded_bias = nn.Parameter(
                        distribute_tensor(
                            module.bias, device_mesh, col_wise_sharding
                        )
                    )
                    module.register_parameter("bias", sharded_bias)
            elif name == "proj":
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, row_wise_sharding
                    )
                )
                module.register_parameter("weight", sharded_weight)
                if module.bias is not None:
                    replicated_bias = nn.Parameter(
                        distribute_tensor(module.bias, device_mesh, replicate)
                    )
                    module.register_parameter("bias", replicated_bias)

    if isinstance(module, TensorParallelMultiheadAttention):  # shard TPMA
        for n, m in module.named_children():
            _shard_self_attn_params(n, m)
    else:
        for n, m in module.named_children():  # replace with TPMA
            if isinstance(m, nn.MultiheadAttention):
                tp_multi_head_attention = TensorParallelMultiheadAttention(
                    m.embed_dim,
                    m.num_heads,
                    device=torch.device(device_mesh.device_type),
                    tp_size=device_mesh.size(0),  # group size on dim 0
                    add_bias_kv=m.bias_k is not None,
                )
                tp_multi_head_attention.copy(m)
                module.register_module(n, tp_multi_head_attention)


def _has_even_num_linears(module: nn.Module) -> bool:
    """
    We traverse through all the children of the given module and count the
    number of Linear module. If the number is even, we return True.

    Args:
        module (nn.Module):
            :class:``nn.Module`` object to be traversed and counted.

    Return:
        A boolean object which specifies whether the module contains
        event-number of Linears in its children.

    .. warning::
        The traversal is not recursive for now.
    """
    linear_submodules = list(
        filter(lambda x: isinstance(x, nn.Linear), module.children())
    )
    return len(linear_submodules) > 0 and len(linear_submodules) % 2 == 0


def _parallelize_mlp(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallel_style: ParallelStyle = PairwiseParallel(),
    tp_mesh_dim: int = 0,
) -> None:
    """
    This function assumes the input module is a sequence of nn.Linear
    and we parallelize the module based on the given parallel style.
    We don't change the FQN of each sub-module and replace each parameter
    in place.

    Args:
        module (nn.Module):
            :class:``nn.Module`` object to be parallelized.
        device_mesh (DeviceMesh):
            :class:``DeviceMesh`` object which describes the mesh topology
            of devices for the DTensor.
        parallel_style (ParallelStyle):
            :class:``ParallelStyle`` object which contains how
            we prepare input/output for Tensor Parallelism.
        tp_mesh_dim (int):
            the dimension of ``device_mesh`` where we perform
            Tensor Parallelism on.

    Return:
        None

    .. warning::
        We only support ``PairwiseParallel`` right now.
    """

    # Define partition functions needed.
    def _rowwise_parallelize_fn(name, module, device_mesh):  # pyre-ignore[2, 3]
        for name, param in module.named_parameters():
            dist_spec = (
                [Shard(1)] if name == "weight" else [Replicate()]  # type: ignore[list-item]
            )
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, dist_spec)
            )
            module.register_parameter(name, dist_param)

    def _colwise_parallelize_fn(name, module, device_mesh):  # pyre-ignore[2, 3]
        for name, param in module.named_parameters():
            dist_param = torch.nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)])
            )
            module.register_parameter(name, dist_param)

    if not isinstance(parallel_style, PairwiseParallel):
        raise NotImplementedError(
            "Only support PairwiseParallel for MLP parallelization."
        )

    if not _has_even_num_linears(module):
        raise RuntimeError("We only support even number of Linear for MLP.")

    if device_mesh.ndim > 1:
        device_mesh = _create_1d_device_mesh(device_mesh, tp_mesh_dim)

    linear_submodules = list(
        filter(lambda x: isinstance(x, nn.Linear), module.children())
    )
    for i, m in enumerate(linear_submodules):
        if i % 2 == 0:
            # Col-wise Parallelize the linear layer
            distribute_module(
                m,
                device_mesh,
                _colwise_parallelize_fn,
                input_fn=parallel_style._prepare_input  # type: ignore[arg-type, misc] # pyre-ignore[6]
                if i == 0
                else None,
            )
        else:
            # Row-wise Parallelize the linear layer
            distribute_module(
                m,
                device_mesh,
                _rowwise_parallelize_fn,
                output_fn=parallel_style._prepare_output  # type: ignore[arg-type, misc] # pyre-ignore[6]
                if i == (len(linear_submodules) - 1)
                else None,
            )
