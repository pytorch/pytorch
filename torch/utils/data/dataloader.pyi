from typing import Any, Callable, TypeVar, Generic, overload, Sequence, List, Optional
from . import Dataset, Sampler

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`, but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[List[T]], Any]

def default_collate(batch: List[T]) -> Any: ...

class DataLoader(Generic[T_co]):
    dataset: Dataset[T_co]
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float

    @overload
    def __init__(self, dataset: Dataset[T_co], batch_size: int=..., shuffle: bool=...,
                 sampler: Optional[Sampler[int]]=..., num_workers: int=..., collate_fn: _collate_fn_t=...,
                 pin_memory: bool=..., drop_last: bool=..., timeout: float=...,
                 worker_init_fn: _worker_init_fn_t=...) -> None: ...
    @overload
    def __init__(self, dataset: Dataset[T_co], batch_sampler: Optional[Sampler[Sequence[int]]]=...,
                 num_workers: int=..., collate_fn: _collate_fn_t=..., pin_memory: bool=..., timeout: float=...,
                 worker_init_fn: _worker_init_fn_t=...) -> None: ...

    def __len__(self) -> int: ...
    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'. In mypy 0.720 and newer a new semantic
    # analyzer is used that obviates the need for this but we leave the quoting in to support older
    # versions of mypy
    def __iter__(self) -> '_BaseDataLoaderIter':...

class _BaseDataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:...
    def __len__(self) -> int: ...
    def __iter__(self) -> _BaseDataLoaderIter: ...
    def __next__(self) -> Any: ...
