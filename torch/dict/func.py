# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from torch.nn.modules.module import _global_parameter_registration_hooks
import torch
from torch import nn, Tensor

def set_tensor_dict(  # noqa: F811
    module_dict, module, name: str, tensor: torch.Tensor
) -> None:
    """Simplified version of torch.nn.utils._named_member_accessor."""
    was_buffer = False
    out = module_dict["_parameters"].pop(name, None)  # type: ignore[assignment]
    if out is None:
        out = module_dict["_buffers"].pop(name, None)
        was_buffer = out is not None
    if out is None:
        out = module_dict.pop(name, None)

    if isinstance(tensor, nn.Parameter):
        # module.register_parameter(name, tensor)
        for hook in _global_parameter_registration_hooks.values():
            output = hook(module, name, tensor)
            if output is not None:
                tensor = output
        module_dict["_parameters"][name] = tensor
    elif was_buffer and isinstance(tensor, Tensor):
        module_dict["_buffers"][name] = tensor
    else:
        module_dict[name] = tensor
    return out
