# Copyright (c) Meta Platforms, Inc. and affiliates
import warnings
from fnmatch import fnmatch
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle


__all__ = ["parallelize_module"]


def parallelize_module(  # type: ignore[return]
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    parallelize_plan: Optional[Union[ParallelStyle, dict[str, ParallelStyle]]] = None,
    *,
    src_data_rank: Optional[int] = 0,
) -> nn.Module:
    """
    Apply Tensor Parallelism in PyTorch by parallelizing modules or sub-modules based on a user-specified plan.

    We parallelize module or sub_modules based on a parallelize_plan. The parallelize_plan contains
    :class:`ParallelStyle`, which indicates how user wants the module or sub_module
    to be parallelized.

    User can also specify different parallel style per module fully qualified name (FQN).

    Note that ``parallelize_module`` only accepts a 1-D :class:`DeviceMesh`, if you have a 2-D or N-D :class:`DeviceMesh`,
    slice the DeviceMesh to a 1-D sub DeviceMesh first then pass to this API(i.e. ``device_mesh[\"tp\"]``)

    Args:
        module (:class:`nn.Module`):
            Module to be parallelized.
        device_mesh (:class:`DeviceMesh`, optional):
            Object which describes the mesh topology of devices for the DTensor.
            If not specified, the call must be under a DeviceMesh context.
        parallelize_plan (Union[:class:`ParallelStyle`, Dict[str, :class:`ParallelStyle`]], optional):
            The plan used to parallelize the module. It can be either a
            :class:`ParallelStyle` object which contains how we prepare
            input/output for Tensor Parallelism or it can be a dict of module
            FQN and its corresponding :class:`ParallelStyle` object. If not
            specified, the call will do nothing at the moment.
    Keyword args:
        src_data_rank (int, optional): the rank of the source data for the logical/global tensor, it is used by
            :meth:`distribute_tensor` to scatter/broadcast the shards/replicas to other ranks. By default,
            we use ``group_rank=0`` on each DeviceMesh dimension as the source data to preserve the single-device
            semantic. If passing ``None`` explicitly, :meth:`parallelize_module` simply uses its local data instead
            of trying to preserve the single-device semantic via scatter/broadcast. Default: 0
    Return:
        A :class:`nn.Module` object parallelized.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>>
        >>> # Define the module.
        >>> m = Model(...)
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>> m = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})
        >>>

    .. note:: For complex module architecture like Attention, MLP layers, we recommend composing
        different ParallelStyles together (i.e. ``ColwiseParallel`` and ``RowwiseParallel``) and pass
        as a parallelize_plan, to achieves the desired sharding computation.
    """
    torch._C._log_api_usage_once("torch.distributed.tensor.parallel.parallelize_module")

    device_mesh = device_mesh or _mesh_resources.get_current_mesh()

    if parallelize_plan is None:
        warnings.warn(
            "No parallelize_plan is provided and auto-parallel is not supported "
            "at the moment, so this parallelize_module call will do nothing."
        )
        return module

    # note: The RNG tracker will be initialized in distribute_tensor() call if it hasn't
    # been initialized.

    if isinstance(parallelize_plan, ParallelStyle):
        parallelize_plan.src_data_rank = src_data_rank
        return parallelize_plan._apply(module, device_mesh)
    elif isinstance(parallelize_plan, dict):
        for module_path, parallelize_style in parallelize_plan.items():
            if module_path == "":
                # shortcut: empty string means to apply the plan to the current module
                parallelize_module(module, device_mesh, parallelize_style)
                continue

            path_splits = module_path.split(".")
            # Instead of blindly popping tokens, first check the match,
            # we only consume/pop the token if we found a match.
            token = path_splits[0]

            matched_children = list(
                filter(
                    # `t[0]` is child name
                    lambda t: fnmatch(t[0], token),
                    module.named_children(),
                )
            )
            if not matched_children:
                # No match at this level. Log a warning and process next plan entry.
                warnings.warn(
                    f"Parallelize plan key '{module_path}' could not be resolved: "
                    f"no submodule matching token '{token}' in module {module}, "
                    f"skipping this plan entry."
                )
                continue

            # Now that we have a match, we can consume the token.
            path_splits.pop(0)
            # apply the plan to all matched submodules
            for _, submodule in matched_children:
                if path_splits:
                    # we haven't reached the leaf, apply in dict style
                    leaf_path = ".".join(path_splits)  # rest of the path after `token`
                    parallelize_module(
                        submodule,
                        device_mesh,
                        {leaf_path: parallelize_style},
                        src_data_rank=src_data_rank,
                    )
                else:
                    # otherwise, directly apply style to this submodule
                    parallelize_module(
                        submodule,
                        device_mesh,
                        parallelize_style,
                        src_data_rank=src_data_rank,
                    )
        return module
    else:
        raise TypeError(  # pyre-ignore[7]
            "Expect Union[ParallelStyle, Dict[str, ParallelStyle]] for"
            f" parallelize_plan, {type(parallelize_plan)} found!"
        )
