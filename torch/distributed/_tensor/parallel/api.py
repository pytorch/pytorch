# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.nn as nn
from typing import Union, Dict
from torch.distributed._tensor import (
    distribute_module,
    distribute_tensor,
    Shard,
    Replicate,
    DeviceMesh,
)
from torch.distributed._tensor.parallel import TensorParallelMultiheadAttention
from torch.distributed._tensor.parallel.style import PairwiseParallel, ParallelStyle
from torch.distributed._tensor.parallel.utils import _create_1d_device_mesh


__all__ = [
    "parallelize_module",
]


def parallelize_module(  # type: ignore[return]
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallelize_plan: Union[ParallelStyle, Dict[str, ParallelStyle]],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    """
    The API to apply Tensor Parallelism (TP) in PyTorch. We parallelize module
    or sub_modules based on a parallelize_plan which contains the parallel_style
    which indicates how user want the module or sub_module to be parallelized.
    User can also specify different parallel_style per module fully qualifed name (FQN).
    The API supports 2D parallelism natively by accepting an n-dimension device_mesh
    and users just need to specify the dimension where we perform tensor parallelism on.

    Args:
        module (nn.Module):
            :class:`nn.Module` object to be parallelized.
        device_mesh (DeviceMesh):
            :class:`DeviceMesh` object which describes the mesh topology
            of devices for the DTensor.
        parallelize_plan (Union[ParallelStyle, Dict[str, ParallelStyle]]):
            The plan used to parallelize the module. It can be either a
            :class:`ParallelStyle` object which contains how
            we prepare input/output for Tensor Parallelism or it can be a
            dict of module FQN and its corresponding :class:`ParallelStyle` object.
        tp_mesh_dim (int):
            the dimension of ``device_mesh`` where we perform
            Tensor Parallelism on.

    Return:
        A :class:`nn.Module` object parallelized.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from from torch.distributed._tensor.parallel import parallelize_module, PairwiseParallel
        >>>
        >>> # Define the module.
        >>> m = Model(...)
        >>> m = parallelize_module(m, PairwiseParallel())
        >>>

    .. warning::
        ``PairwiseParallel`` comes with constraints for now. If you need finer
        granularity, you need to pass in a dict of module FQN and parallel style instead.
    """

    if device_mesh.ndim > 1:
        device_mesh = _create_1d_device_mesh(device_mesh, tp_mesh_dim)

    if isinstance(parallelize_plan, ParallelStyle):
        if _is_mha_for_pairwise_parallel(module):
            return _parallelize_multihead_attn(module, device_mesh)
        elif _is_mlp_for_pairwise_parallel(module):
            return _parallelize_mlp(module, device_mesh)
        else:
            for n, m in module.named_children():
                module.register_module(
                    n, parallelize_module(m, device_mesh, parallelize_plan)
                )
            return module
    # TODO: Add parallelize linear logic when https://github.com/pytorch/tau/pull/624/ merged.
    elif isinstance(parallelize_plan, dict):
        for module_path, parallelize_style in parallelize_plan.items():
            sub_module = module.get_submodule(module_path)
            module.register_module(  # type: ignore[call-arg] # pyre-ignore[20]
                parallelize_module(  # type: ignore[arg-type]
                    module_path, sub_module, device_mesh, parallelize_style  # type: ignore[arg-type] # pyre-ignore[6]
                )
            )
            return module
    else:
        raise RuntimeError(  # pyre-ignore[7]
            f"Expect Union[ParallelStyle, Dict[str, ParallelStyle]] for parallelize_plan, {type(parallelize_plan)} found!"
        )


def _is_mha_for_pairwise_parallel(module: nn.Module) -> bool:
    """
    Check whether the mha module is the one can be handled for Pairwise parallel.

    Args:
        module (nn.Module):
            :class:``nn.Module`` object to be checked.

    Return:
        A boolean object which specifies whether the module is MHA supported by Pairwise parallel or not.
    """
    return isinstance(module, TensorParallelMultiheadAttention) or isinstance(
        module, nn.MultiheadAttention
    )


def _is_mlp_for_pairwise_parallel(module: nn.Module) -> bool:
    """
    Traverse through all the immediate children of the given module and count the
    number of Linear module. If the number is more than one, we return True.

    Args:
        module (nn.Module):
            :class:``nn.Module`` object to be traversed and counted.

    Return:
        A boolean object which specifies whether the module is MLP or not.

    .. warning::
        The traversal is not recursive for now.
    """
    linear_submodules = list(
        filter(lambda x: isinstance(x, nn.Linear), module.children())
    )
    return len(linear_submodules) > 1


def _rowwise_parallelize_linear_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    """
    This function parallelizes the input :class:``nn.Linear`` module in :class:``RowwiseParallel`` style.

    Args:
        name (str): name of the input module.
        module (nn.Module): the :class:``nn.Linear`` object to be parallelized.
        device_mesh (DeviceMesh): :class:``DeviceMesh`` object which describes the mesh topology
            of devices for the DTensor.

    Return:
        None
    """
    for name, param in module.named_parameters():
        dist_spec = (
            [Shard(1)] if name == "weight" else [Replicate()]  # type: ignore[list-item]
        )
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, dist_spec)
        )
        module.register_parameter(name, dist_param)


def _colwise_parallelize_linear_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    """
    This function parallelizes the input :class:``nn.Linear`` module in :class:``ColwiseParallel`` style.

    Args:
        name (str): name of the input module.
        module (nn.Module): the :class:``nn.Linear`` object to be parallelized.
        device_mesh (DeviceMesh): :class:``DeviceMesh`` object which describes the mesh topology
            of devices for the DTensor.

    Return:
        None
    """
    for name, param in module.named_parameters():
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, [Shard(0)])
        )
        module.register_parameter(name, dist_param)


def _parallelize_multihead_attn(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallel_style: ParallelStyle = PairwiseParallel(),
    tp_mesh_dim: int = 0,
) -> nn.Module:
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
        A :class:``nn.Module`` object parallelized.

    .. warning::
        We only support ``PairwiseParallel`` right now.
    """

    if not isinstance(parallel_style, PairwiseParallel):
        raise NotImplementedError(
            "Only support PairwiseParallel for Multihead Attention parallelization."
        )

    if device_mesh.ndim > 1:
        device_mesh = _create_1d_device_mesh(device_mesh, tp_mesh_dim)

    if isinstance(module, nn.MultiheadAttention):
        tp_multi_head_attention = TensorParallelMultiheadAttention(
            module.embed_dim,
            module.num_heads,
            device=torch.device(device_mesh.device_type),
            tp_size=device_mesh.size(tp_mesh_dim),
            add_bias_kv=module.bias_k is not None,
        )
        tp_multi_head_attention.copy(module)
        module = tp_multi_head_attention

    if isinstance(module, TensorParallelMultiheadAttention):  # shard TPMA
        for n, m in module.named_children():
            if n == "qkv":
                # Col-wise Parallelize the qkv layer.
                distribute_module(
                    m,
                    device_mesh,
                    _colwise_parallelize_linear_fn,
                    input_fn=parallel_style._prepare_input,  # type: ignore[arg-type, misc] # pyre-ignore[6]
                )
            elif n == "proj":
                # Row-wise Parallelize the proj layer
                distribute_module(
                    m,
                    device_mesh,
                    _rowwise_parallelize_linear_fn,
                    output_fn=parallel_style._prepare_output,  # type: ignore[arg-type, misc] # pyre-ignore[6]
                )
    return module


def _parallelize_mlp(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallel_style: ParallelStyle = PairwiseParallel(),
    tp_mesh_dim: int = 0,
) -> nn.Module:
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
        A :class:``nn.Module`` object parallelized.

    .. warning::
        We only support ``PairwiseParallel`` right now.
    """
    if not isinstance(parallel_style, PairwiseParallel):
        raise NotImplementedError(
            "Only support PairwiseParallel for MLP parallelization."
        )

    if not _is_mlp_for_pairwise_parallel(module):
        raise RuntimeError("More than one nn.Linear needed for a MLP.")

    if device_mesh.ndim > 1:
        device_mesh = _create_1d_device_mesh(device_mesh, tp_mesh_dim)

    linear_submodules = list(
        filter(lambda x: isinstance(x, nn.Linear), module.children())
    )
    mlp_last_even_layer = (len(linear_submodules) // 2) * 2
    for i in range(mlp_last_even_layer):
        m = linear_submodules[i]
        if i % 2 == 0:
            # Col-wise Parallelize the linear layer
            distribute_module(
                m,
                device_mesh,
                _colwise_parallelize_linear_fn,
                input_fn=parallel_style._prepare_input  # type: ignore[arg-type, misc] # pyre-ignore[6]
                if i == 0
                else None,
            )
        else:
            # Row-wise Parallelize the linear layer
            distribute_module(
                m,
                device_mesh,
                _rowwise_parallelize_linear_fn,
                output_fn=parallel_style._prepare_output  # type: ignore[arg-type, misc] # pyre-ignore[6]
                if i == (mlp_last_even_layer - 1)
                else None,
            )
    return module
