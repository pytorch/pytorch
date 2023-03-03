from typing import Any, Optional

from torch import device, Tensor
from ..modules import Module
from .common_types import _device_t, _devices_t

class DataParallel(Module):
    module: Module = ...
    device_ids: _devices_t = ...
    dim: int = ...
    output_device: _device_t = ...
    src_device_obj: device = ...

    def __init__(
        self,
        module: Module,
        device_ids: Optional[_devices_t] = ...,
        output_device: Optional[_device_t] = ...,
        dim: int = ...,
    ) -> None: ...

def data_parallel(
    module: Module,
    inputs: Any,
    device_ids: Optional[_devices_t] = ...,
    output_device: Optional[_device_t] = ...,
    dim: int = ...,
    module_kwargs: Optional[Any] = ...,
) -> Tensor: ...
