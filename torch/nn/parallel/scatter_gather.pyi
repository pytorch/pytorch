from typing import Any, Dict, List, Tuple, overload, TypeVar
from ... import Tensor
from .common_types import _device_t, _devices_t


T = TypeVar('T', Dict, List, Tuple)

# For some reason, 'scatter' returns a tuple when given a single Tensor input but a list otherwise.
@overload
def scatter(inputs: Tensor, target_gpus: _devices_t, dim: int = ...) -> Tuple[Tensor, ...]: ...

@overload
def scatter(inputs: T, target_gpus: _devices_t, dim: int = ...) -> List[T]: ...


# TODO More precise types here.
def scatter_kwargs(inputs: Any, kwargs: Any, target_gpus: _devices_t, dim: int = ...) -> Any: ...


def gather(outputs: Any, target_device: _device_t, dim: int = ...) -> Any: ...
