import torch

from torch.utils._mode_utils import no_dispatch


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
