from typing import List, Sequence, Union

from ..modules import Module
from .common_types import _devices_t

def replicate(
    network: Module,
    devices: Union[_devices_t, Sequence[_devices_t]],
    detach: bool = ...,
) -> List[Module]: ...
