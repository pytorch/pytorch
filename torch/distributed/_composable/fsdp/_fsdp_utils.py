"""
This file contains utility functions meant to be exposed to the user.
"""
import functools
from typing import Any, cast, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from torch.utils._pytree import tree_map
from torch.utils.hooks import RemovableHandle

from ._fsdp_common import _cast_floating_point_tensor, FSDP_IGNORED


def register_forward_cast_hooks(
    module: nn.Module,
    input_dtype: Optional[torch.dtype] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[Optional[RemovableHandle], Optional[RemovableHandle]]:
    """
    Registers a forward pre-hook to cast floating-point input tensors to
    ``input_dtype`` if specified and a forward post-hook to cast floating-point
    output tensors to ``output_dtype`` if specified.

    Returns:
        Tuple[Optional[RemovableHandle], Optional[RemovableHandle]]: The
        forward pre-hook handle and forward hook handle if ``input_dtype``
        and if ``output_dtype`` are not ``None``, respectively, or ``None`` for
        each otherwise.
    """

    def forward_pre_hook(
        _module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ):
        input_cast_fn = functools.partial(
            _cast_floating_point_tensor, cast(torch.dtype, input_dtype)
        )
        return tree_map(input_cast_fn, (args, kwargs))

    def forward_hook(_module: nn.Module, input: Any, output: Any):
        output_cast_fn = functools.partial(
            _cast_floating_point_tensor, cast(torch.dtype, output_dtype)
        )
        return tree_map(output_cast_fn, output)

    forward_pre_hook_handle = (
        module.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)
        if input_dtype is not None
        else None
    )
    forward_hook_handle = (
        module.register_forward_hook(forward_hook) if output_dtype is not None else None
    )
    return forward_pre_hook_handle, forward_hook_handle


def _set_module_states_to_ignore(states_to_ignore: List[torch.Tensor]):
    for state in states_to_ignore:
        setattr(state, FSDP_IGNORED, True)
