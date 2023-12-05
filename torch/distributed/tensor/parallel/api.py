# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, Union

import torch
import torch.distributed._tensor.random as random
import torch.nn as nn
from torch.distributed._tensor import (
    DeviceMesh,
)
from torch.distributed._tensor.random import (
    is_rng_supported_mesh,
    TensorParallelRNGTracker,
)
from torch.distributed.tensor.parallel._utils import _create_1d_device_mesh, _validate_tp_mesh_dim, _deprecate_warnings
from torch.distributed.tensor.parallel.style import (
    ParallelStyle,
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
    Apply Tensor Parallelism in PyTorch by parallelizing modules or sub-modules based on a user-specified plan.

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
        _deprecate_warnings("tp_mesh_dim", "If you have a 2-D or N-D device_mesh, consider passing in device_mesh[\"tp\"]")
        device_mesh = _create_1d_device_mesh(device_mesh, tp_mesh_dim)
    else:
        _validate_tp_mesh_dim(device_mesh)


    if isinstance(parallelize_plan, ParallelStyle):
        return parallelize_plan._apply(module, device_mesh)
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
