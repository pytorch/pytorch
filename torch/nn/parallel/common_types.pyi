from typing import Sequence, Union

from torch import device

_device_t = Union[int, device]
_devices_t = Sequence[_device_t]
