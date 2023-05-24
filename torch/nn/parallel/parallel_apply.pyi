from typing import Any, List, Optional, Sequence

from ..modules import Module
from .common_types import _devices_t

def parallel_apply(
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Optional[Any] = ...,
    devices: Optional[_devices_t] = ...,
) -> List[Any]: ...
