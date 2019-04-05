from typing import Any, Callable, TypeVar, Generic, overload, Sequence, List, Union
from . import Dataset, Sampler
from ... import Tensor

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Union[Tensor, Sequence[Tensor]]]

class DataLoader(Generic[T]):
    dataset: Dataset[T_co]
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float

    @overload
    def __init__(self, dataset: Dataset[T_co], batch_size: int=..., shuffle: bool=..., sampler: Sampler[int]=...,
                 num_workers: int=..., collate_fn: _collate_fn_t=..., pin_memory: bool=...,
                 drop_last: bool=..., timeout: float=..., worker_init_fn: _worker_init_fn_t=...) -> None: ...
    @overload
    def __init__(self, dataset: Dataset[T_co], batch_sampler: Sampler[Sequence[int]]=..., num_workers: int=...,
                 collate_fn: _collate_fn_t=..., pin_memory: bool=..., timeout: float=...,
                 worker_init_fn: _worker_init_fn_t=...) -> None: ...

    def __len__(self) -> int: ...
    # We quote '_DataLoaderIter' since it isn't defined yet and the definition can't be moved up since
    # '_DataLoaderIter' references 'DataLoader'. Pending updates of PEP 484 will fix this.
    def __iter__(self) -> '_DataLoaderIter':...

class _DataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:...
    def __len__(self) -> int: ...
    def __iter__(self) -> _DataLoaderIter: ...
    # Whether a single tensor or a list of tensors returned depends on if the
    # underlying dataloader returns a sequence or scalar in '__getitem__.
    # There is no way to exactly express that in the Python type system so
    # we give this loose return type.
    def __next__(self) -> Union[Tensor, List[Tensor]]: ...
