from functools import partial
from typing import Any, Dict, Type

import torch

from torch.distributed.utils import _apply_to_tensors
from torch.utils._mode_utils import no_dispatch


def _override_module_mixed_precision(
    root: torch.nn.Module,
    module_cls_to_override: Type[torch.nn.Module],
    wrap_override_dict: Dict[str, Any] = {"mixed_precision": None},  # noqa: B006
):
    for mod in root.modules():
        if isinstance(mod, module_cls_to_override):
            mod._wrap_overrides = wrap_override_dict  # type: ignore[assignment]
            # TODO: We need to run this mixed precision ignored module in fp32,
            # but ensure subsequent modules, that may possibly be running with
            # mixed precision, still receive the appropriate precision inputs
            # without user having to adjust mixed precision config too much.
            # As a result, we attach pre and post forward hooks to up / down
            # cast. We should revisit this design.
            old_dtype = None

            def cast_fn(dtype, x: torch.Tensor) -> torch.Tensor:
                if not torch.is_floating_point(x) or x.dtype == dtype:
                    return x
                nonlocal old_dtype
                old_dtype = x.dtype
                return x.to(dtype)

            def forward_pre_hook(module, args):
                return _apply_to_tensors(partial(cast_fn, torch.float32), args)

            def forward_post_hook(module, args, output):
                nonlocal old_dtype
                if old_dtype is not None:
                    return _apply_to_tensors(partial(cast_fn, old_dtype), output)

            # We intentionally append both of these hooks so that they run after
            # all other hooks.
            mod.register_forward_pre_hook(forward_pre_hook, prepend=False)
            mod.register_forward_hook(forward_post_hook, prepend=False)


def _same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Returns if ``x`` and ``y`` share the same storage."""
    # NOTE: CPU and GPU tensors are ensured to have different data pointers.
    return x._typed_storage()._data_ptr() == y._typed_storage()._data_ptr()


def _same_storage_as_data_ptr(x: torch.Tensor, data_ptr: int) -> bool:
    return x._typed_storage()._data_ptr() == data_ptr


def _no_dispatch_record_stream(tensor: torch.Tensor, stream: torch.Stream) -> None:
    # FIXME record_stream doesn't work with non-cuda tensors
    if tensor.device.type not in ["cuda", torch._C._get_privateuse1_backend_name()]:
        return
    with no_dispatch():
        tensor.record_stream(stream)
