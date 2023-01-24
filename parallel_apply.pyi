from typing import Any, Optional, Sequence, List
from .common_types import _devices_t
from ..modules import Module


def parallel_apply(modules: Sequence[Module], inputs: Sequence[Any], kwargs_tup: Optional[Any] = ...,
                   devices: Optional[_devices_t] = ...) -> List[Any]: ...
