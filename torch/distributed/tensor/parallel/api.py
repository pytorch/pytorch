# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, Union

import torch
import torch.distributed._tensor.random as random
import torch.nn as nn
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)
from torch.distributed._tensor.random import (
    is_rng_supported_mesh,
    TensorParallelRNGTracker,
)
from torch.distributed.tensor.parallel._utils import _create_1d_device_mesh, _validate_tp_mesh_dim
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    PairwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)


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
    Apply Tensor Parallelism (TP) in PyTorch by parallelizing modules or sub-modules based on a user-specified plan.

    We parallelize module or sub_modules based on a parallelize_plan. The parallelize_plan contains
    :class:`ParallelStyle`, which indicates how user wants the module or sub_module
    to be parallelized.

    User can also specify different parallel style per module fully qualified name (FQN).
    The API supports 2D parallelism natively by accepting an n-dimension device_mesh
    and users just need to specify the dimension where we perform tensor parallelism on.

    Args:
        module (:class:`nn.Module`):
            Module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        parallelize_plan (Union[:class:`ParallelStyle`, Dict[str, :class:`ParallelStyle`]]):
            The plan used to parallelize the module. It can be either a
            :class:`ParallelStyle` object which contains how
            we prepare input/output for Tensor Parallelism or it can be a
            dict of module FQN and its corresponding :class:`ParallelStyle` object.
        tp_mesh_dim (int):
            The dimension of ``device_mesh`` where we perform
            Tensor Parallelism on.

    Return:
        A :class:`nn.Module` object parallelized.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>>
        >>> # Define the module.
        >>> m = Model(...)
        >>> m = parallelize_module(m, ColwiseParallel())
        >>>

    .. warning::
        Currently, there are some constraints which makes it hard for complicated modules
        like ``MultiheadAttention`` to work out of box for Tensor or Sequence Parallelism.
        We recommend users to try ``ColwiseParallel`` and ``RowwiseParallel`` for each parameter
        or submodule and there might be some code changes needed now.
    """
    torch._C._log_api_usage_once("torch.distributed.tensor.parallel.parallelize_module")

    # instantiate a TP RNG state tracker if it's not there
    if is_rng_supported_mesh(device_mesh) and not isinstance(
        random._rng_tracker, TensorParallelRNGTracker
    ):
        random._rng_tracker = TensorParallelRNGTracker(device_mesh.device_type)
        # TODO: we should allow user to pass in the default seed from a config
        random._rng_tracker._manual_seed(
            device_mesh, base_seed=1234, tp_dim=tp_mesh_dim
        )
        # By default we execute random ops in non-tensor-parallel region. If users want
        # to execute in tensor-parallel region, they can manually set this field to True
        # after parallelizing the model.
        random._rng_tracker.distribute_region_enabled = False

    if device_mesh.ndim > 1:
        device_mesh = _create_1d_device_mesh(device_mesh, tp_mesh_dim)
    else:
        _validate_tp_mesh_dim(device_mesh)


    if isinstance(parallelize_plan, ParallelStyle):
        # RowwiseParallel or ColwiseParallel
        if isinstance(parallelize_plan, (ColwiseParallel, RowwiseParallel)):
            return _parallelize_linear_like_module(
                module, device_mesh, parallelize_plan
            )
        elif isinstance(parallelize_plan, PrepareModuleInput):
            module.register_forward_pre_hook(lambda _, inputs: parallelize_plan._prepare_input(inputs, device_mesh))  # type: ignore[misc, call-arg]
            return module
        elif isinstance(parallelize_plan, PrepareModuleOutput):
            module.register_forward_hook(lambda _, _inputs, outputs: parallelize_plan._prepare_output(outputs, device_mesh))  # type: ignore[misc, call-arg]
            return module
        # PairwiseParallel
        if _is_mlp_for_pairwise_parallel(module):
            return _parallelize_mlp(module, device_mesh, parallelize_plan)
        else:
            for n, m in module.named_children():
                module.register_module(
                    n, parallelize_module(m, device_mesh, parallelize_plan)
                )
            return module
    elif isinstance(parallelize_plan, dict):
        for module_path, parallelize_style in parallelize_plan.items():
            sub_module = module.get_submodule(module_path)
            parent_module = module
            if "." in module_path:
                parent_module_path = ".".join(module_path.split(".")[:-1])
                parent_module = module.get_submodule(parent_module_path)
                module_path = module_path.split(".")[-1]
            parent_module.register_module(  # type: ignore[call-arg] # pyre-ignore[20]
                module_path,
                parallelize_module(  # type: ignore[arg-type]
                    sub_module, device_mesh, parallelize_style  # type: ignore[arg-type] # pyre-ignore[6]
                ),
            )
        return module
    else:
        raise RuntimeError(  # pyre-ignore[7]
            "Expect Union[ParallelStyle, Dict[str, ParallelStyle]] for"
            f" parallelize_plan, {type(parallelize_plan)} found!"
        )


def _is_mlp_for_pairwise_parallel(module: nn.Module) -> bool:
    """
    Traverse through all the immediate children of the given module and count the number of Linear module.

    If the number is more than one, we return True.

    Args:
        module (:class:`nn.Module`):
            Module to be traversed and counted.

    Return:
        A bool which specifies whether the module is MLP supported or not.

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
    Parallelize the input :class:`nn.Linear` module in :class:`RowwiseParallel` style.

    Args:
        name (str):
            Name of the input module.
        module (:class:`nn.Module`):
            The :class:`nn.Linear` module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.

    Returns:
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


def _colwise_parallelize_embedding_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    _colwise_parallelize_linear_fn(name, module, device_mesh, sharding_dim=1)


def _colwise_parallelize_linear_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
    sharding_dim: int = 0,
) -> None:
    """
    Parallelize the input :class:`nn.Linear` or :class:`nn.Embedding` module in :class:`ColwiseParallel` style.

    Args:
        name (str):
            Name of the input module.
        module (:class:`nn.Module`):
            The :class:`nn.Linear` or :class:`nn.Embedding` module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.

    Returns:
        None
    """
    for name, param in module.named_parameters():
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, [Shard(sharding_dim)])
        )
        module.register_parameter(name, dist_param)


def _parallelize_linear_like_module(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallel_style: ParallelStyle = ColwiseParallel(),
    tp_mesh_dim: int = 0,
) -> nn.Module:
    """
    Parallelize an nn.Linear module over a 1-d DeviceMesh based on a specified ParallelStyle for Tensor Parallelism.

    This function requires that the input module be an object
    of :class:`nn.Linear`.
    The module will be parallelized over a 1-d :class:`DeviceMesh`
    based on the :class:`ParallelStyle`.

    Args:
        module (:class:`nn.Module`):
            The module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices for the :class:`DTensor`.
            If the mesh is more than 1-dimensional, we will use the mesh dim of
            `device_mesh` specified by `tp_mesh_dim`.
        parallel_style (:class:`ParallelStyle`, optional):
            The object which describes how the :class:`nn.Linear` module
            should be distributed over :class:`DeviceMesh` and how the input
            and output should be prepared for Tensor Parallelism.
            :class:`RowwiseStyle`: weight is sharded on dim 1 and bias is
            replicate.
            :class:`ColwiseStyle`: weight and bias are both sharded on dim 0.
            Default: :class:`ColwiseParallel`
        tp_mesh_dim (int):
            The dimension of :class:`DeviceMesh` on which we
            perform Tensor Parallelism.
            Default: 0

    Return:
        A :class:`nn.Module` object parallelized.
    """
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        raise RuntimeError(
            f"Expect a torch.nn.Linear or a torch.nn.Embedding module but received {type(module)}!"
        )

    if not isinstance(parallel_style, ParallelStyle):
        raise RuntimeError(
            "Expect a ParallelStyle object but received" f" {type(parallel_style)}!"
        )

    if isinstance(module, nn.Embedding) and not isinstance(
        parallel_style, (ColwiseParallel)
    ):
        raise NotImplementedError(
            "Only support ColwiseParallel when parallelizing Embedding now."
        )

    if device_mesh.ndim > 1:
        device_mesh = _create_1d_device_mesh(device_mesh, tp_mesh_dim)

    if isinstance(parallel_style, (RowwiseParallel)):
        distribute_module(
            module,
            device_mesh,
            _rowwise_parallelize_linear_fn,
            input_fn=parallel_style._prepare_input,  # type: ignore[arg-type, misc] # pyre-ignore[6]
            output_fn=parallel_style._prepare_output,  # type: ignore[arg-type, misc] # pyre-ignore[6]
        )
    elif isinstance(parallel_style, (ColwiseParallel)):
        distribute_module(
            module,
            device_mesh,
            _colwise_parallelize_embedding_fn
            if isinstance(module, nn.Embedding)
            else _colwise_parallelize_linear_fn,
            input_fn=parallel_style._prepare_input,  # type: ignore[arg-type, misc] # pyre-ignore[6]
            output_fn=parallel_style._prepare_output,  # type: ignore[arg-type, misc] # pyre-ignore[6]
        )
    else:
        raise RuntimeError(f"{type(parallel_style)} is not supported!")
    return module


def _parallelize_mlp(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallel_style: ParallelStyle = PairwiseParallel(),
    tp_mesh_dim: int = 0,
) -> nn.Module:
    """
    Parallelize a sequence of nn.Linear modules over a 1-d DeviceMesh based on a specified ParallelStyle.

    This function assumes the input module is a sequence of nn.Linear
    and we parallelize the module based on the given parallel style.
    We don't change the FQN of each sub-module and replace each parameter
    in place.

    Args:
        module (:class:`nn.Module`):
            Module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.
        parallel_style (:class:`ParallelStyle`):
            Object which contains how we prepare input/output
            for Tensor Parallelism.
        tp_mesh_dim (int):
            The dimension of `device_mesh` where we perform
            Tensor Parallelism on.

    Return:
        A :class:`nn.Module` object parallelized.

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
