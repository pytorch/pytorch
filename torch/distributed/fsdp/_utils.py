from typing import cast

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.normalization import LayerNorm
from torch.utils._mode_utils import no_dispatch

from torch.distributed.utils import _apply_to_tensors
from functools import partial


def _contains_batchnorm(module):
    return any(isinstance(mod, _BatchNorm) for mod in module.modules())


def _contains_module(root, module_to_check):
    return any(isinstance(mod, module_to_check) for mod in root.modules())

def _override_module_mixed_precision(root, module_to_override, wrap_override_dict={"mixed_precision": None}):
    for mod in root.modules():
        if isinstance(mod, module_to_override):
            mod._wrap_overrides = wrap_override_dict  # type: ignore[assignment]

        if isinstance(mod, LayerNorm): # TODO: generalize this logic if additional types need to be supported

            old_dtype = None

            def cast_fn(dtype, x: torch.Tensor) -> torch.Tensor:
                if not torch.is_floating_point(x) or x.dtype == dtype:
                    return x
                nonlocal old_dtype
                old_dtype = x.dtype
                return x.to(dtype)

            def ln_forward_pre_hook(module, args):
                return _apply_to_tensors(partial(cast_fn, torch.float32), args)

            def ln_forward_post_hook(module, args, output):
                nonlocal old_dtype
                if old_dtype is not None:
                    return _apply_to_tensors(partial(cast_fn, old_dtype), output)

            mod.register_forward_pre_hook(ln_forward_pre_hook)
            mod.register_forward_hook(ln_forward_post_hook)

def _override_batchnorm_mixed_precision(module):
    for mod in module.modules():
        if isinstance(mod, _BatchNorm):
            mod._wrap_overrides = {"mixed_precision": None}  # type: ignore[assignment]


def _same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Returns if ``x`` and ``y`` share the same storage."""
    # NOTE: CPU and GPU tensors are ensured to have different data pointers.
    return x._typed_storage()._data_ptr() == y._typed_storage()._data_ptr()


def _same_storage_as_data_ptr(x: torch.Tensor, data_ptr: int) -> bool:
    return x._typed_storage()._data_ptr() == data_ptr


def _no_dispatch_record_stream(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    with no_dispatch():
        tensor.record_stream(cast(torch._C.Stream, stream))
