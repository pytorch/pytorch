# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn as nn

from .flex_shard import FlexShardModule
from .placement_contract import compose_global_layouts, GlobalLayout, set_global_layout
from .sharded_param import is_flex_shard_param
from .utils import _strip_checkpoint_wrapped_module_path


# Their states other than ``step`` are elementwise, with the parameter's shape.
_CHECKPOINTABLE_OPTIMIZERS = (torch.optim.Adam, torch.optim.AdamW)
_OPTIMIZER_CHECKPOINT_HOOK_ATTR = "_flex_shard_optimizer_checkpoint_hook"


def get_flex_shard_global_layouts(module: nn.Module) -> dict[str, GlobalLayout]:
    """Return where each FlexShard parameter's local shard sits in the full parameter.

    Keys are parameter FQNs relative to ``module``. Each layout composes the
    parameter's placement layout with the outer layout the parameter declared
    before ``flex_shard()``, and is computed from the current FlexShard state on
    every call.

    Raises ``NotImplementedError`` for parameters whose placement does not
    implement ``Placement.global_layout``.
    """
    storages = [
        (prefix, storage)
        for prefix, submodule in module.named_modules()
        if isinstance(submodule, FlexShardModule)
        for storage in submodule.sharded_bucket_storages
    ]
    layouts = {}
    for prefix, storage in storages:
        rank, world_size = storage._mesh.get_local_rank(), storage._mesh.size()
        for fqn, info in storage.param_infos.items():
            layout = info.placement.global_layout(info, rank, world_size)
            if info.outer_layout is not None:
                layout = compose_global_layouts(info.outer_layout, layout)
            # ``state_dict()`` lists every name of a shared parameter, each as
            # its own detached tensor, so each name needs the layout.
            for shared_fqn in (fqn, *info.shared_fqns):
                layouts[f"{prefix}.{shared_fqn}" if prefix else shared_fqn] = layout
    return layouts


def set_state_dict_global_layouts(
    module: nn.Module, state_dict: dict[str, Any]
) -> None:
    """Declare each FlexShard parameter's layout on its ``state_dict`` tensor.

    Call on the state dict passed to both ``dcp.save`` and ``dcp.load``.
    Without it, DCP treats each local shard as the full parameter. Parameters
    missing from ``state_dict`` are skipped.
    """
    for fqn, layout in get_flex_shard_global_layouts(module).items():
        key = fqn if fqn in state_dict else _strip_checkpoint_wrapped_module_path(fqn)
        if key in state_dict:
            set_global_layout(state_dict[key], layout)


def register_optimizer_checkpoint_hook(
    optimizer: torch.optim.Optimizer, module: nn.Module
) -> None:
    """Declare FlexShard layouts on the states of ``optimizer.state_dict()``.

    Registers a ``state_dict`` post-hook that declares each FlexShard parameter's
    layout on its state tensors, so DCP saves and loads each as the same shard
    of the full state. ``step`` is left without a layout, so DCP treats it as
    replicated. ``optimizer.state_dict()`` returns the live state tensors, so
    ``dcp.load`` into it restores them in place before
    ``optimizer.load_state_dict``.

    ``module`` must contain every FlexShard parameter of ``optimizer``; call this
    while its parameters are sharded, as after ``flex_shard()``. Calling it again
    replaces the hook.

    Raises ``NotImplementedError`` for optimizers other than Adam and AdamW, and
    ``ValueError`` if ``optimizer`` holds FlexShard parameters that are not
    sharded parameters of ``module``. The hook raises ``NotImplementedError``
    for a state that does not have its parameter's shape.
    """
    if not isinstance(optimizer, _CHECKPOINTABLE_OPTIMIZERS):
        raise NotImplementedError(
            "FlexShard optimizer checkpointing supports Adam and AdamW, got "
            f"{type(optimizer).__name__}."
        )
    layouts = {
        _strip_checkpoint_wrapped_module_path(fqn): layout
        for fqn, layout in get_flex_shard_global_layouts(module).items()
    }
    # Keyed by parameter identity: the optimizer holds the sharded parameters.
    param_layouts: dict[torch.Tensor, tuple[str, GlobalLayout]] = {}
    for fqn, param in module.named_parameters():
        layout = layouts.get(_strip_checkpoint_wrapped_module_path(fqn))
        if layout is not None:
            param_layouts[param] = (fqn, layout)
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            _optimizer_param_layout(param, param_layouts)

    handle = getattr(optimizer, _OPTIMIZER_CHECKPOINT_HOOK_ATTR, None)
    if handle is not None:
        handle.remove()
    handle = optimizer.register_state_dict_post_hook(
        partial(_set_optimizer_state_global_layouts, param_layouts=param_layouts)
    )
    setattr(optimizer, _OPTIMIZER_CHECKPOINT_HOOK_ATTR, handle)


def _optimizer_param_layout(
    param: torch.Tensor,
    param_layouts: dict[torch.Tensor, tuple[str, GlobalLayout]],
) -> tuple[str, GlobalLayout] | None:
    entry = param_layouts.get(param)
    if entry is None and is_flex_shard_param(param):
        raise ValueError(
            "The optimizer holds a FlexShard parameter that is not a sharded "
            "parameter of the module passed to register_optimizer_checkpoint_hook."
        )
    return entry


def _set_optimizer_state_global_layouts(
    optimizer: torch.optim.Optimizer,
    state_dict: dict[str, Any],
    *,
    param_layouts: dict[torch.Tensor, tuple[str, GlobalLayout]],
) -> None:
    # ``state_dict`` lists each group's parameters as indices, in the order of
    # the optimizer's live parameter groups.
    for param_group, saved_param_group in zip(
        optimizer.param_groups, state_dict["param_groups"], strict=True
    ):
        for param, param_id in zip(
            param_group["params"], saved_param_group["params"], strict=True
        ):
            entry = _optimizer_param_layout(param, param_layouts)
            if entry is None:
                continue
            fqn, layout = entry
            for state_name, value in state_dict["state"].get(param_id, {}).items():
                if state_name == "step" or not isinstance(value, torch.Tensor):
                    continue
                if value.shape != param.shape:
                    raise NotImplementedError(
                        f"Optimizer state {state_name!r} of FlexShard parameter "
                        f"{fqn!r} has shape {tuple(value.shape)}, but the local "
                        f"shard has shape {tuple(param.shape)}."
                    )
                set_global_layout(value, layout)


__all__ = [
    "get_flex_shard_global_layouts",
    "register_optimizer_checkpoint_hook",
    "set_state_dict_global_layouts",
]
