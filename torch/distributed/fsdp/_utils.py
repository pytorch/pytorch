from typing import cast

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils._mode_utils import no_dispatch


def _contains_batchnorm(module):
    return any(isinstance(mod, _BatchNorm) for mod in module.modules())


def _override_batchnorm_mixed_precision(module):
    for mod in module.modules():
        if isinstance(mod, _BatchNorm):
            mod._wrap_overrides = {"mixed_precision": None}  # type: ignore[assignment]


def _same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Returns if ``x`` and ``y`` share the same storage."""
    # NOTE: CPU and GPU tensors are ensured to have different data pointers.
    return x._typed_storage()._data_ptr() == y._typed_storage()._data_ptr()


def _no_dispatch_record_stream(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    with no_dispatch():
        tensor.record_stream(cast(torch._C.Stream, stream))
