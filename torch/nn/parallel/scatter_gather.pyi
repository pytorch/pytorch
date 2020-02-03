from typing import Any, Dict, List, Tuple, overload, TypeVar
from ... import Tensor
from .common_types import _device_t, _devices_t


T = TypeVar('T', Dict, List, Tuple)

# For some reason, 'scatter' returns a tuple when given a single Tensor input but a list otherwise.
@overload
def scatter(inputs: Tensor, target_gpus: _devices_t, dim: int = ...) -> Tuple[Tensor, ...]: ...

# flake8 will raise a spurious error here since `torch/__init__.pyi` has not been generated yet
# so mypy will interpret `Tensor` as `Any` since it is an import from what it believes to be an
# untyped module. Thus to mypy, the first definition of `scatter` looks strictly more general
# than this overload.
@overload
def scatter(inputs: T, target_gpus: _devices_t, dim: int = ...) -> List[T]: ...  # type: ignore 


# TODO More precise types here.
def scatter_kwargs(inputs: Any, kwargs: Any, target_gpus: _devices_t, dim: int = ...) -> Any: ...


def gather(outputs: Any, target_device: _device_t, dim: int = ...) -> Any: ...
