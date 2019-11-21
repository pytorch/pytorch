from typing import TypeVar, Generic, Iterable, Sequence, List, Tuple
from ... import Tensor

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
class Dataset(Generic[T_co]):
    def __getitem__(self, index: int) -> T_co: ...
    def __len__(self) -> int: ...
    def __add__(self, other: T_co) -> 'ConcatDataset[T_co]': ...

class IterableDataset(Dataset[T_co]):
    def __iter__(self) -> Iterable[T_co]: ...

 
class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    tensors: List[Tensor]

    def __init__(self, *tensors: Tensor) -> None: ...

class ConcatDataset(Dataset[T_co]):
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    def __init__(self, datasets: Iterable[Dataset]) -> None: ...

class Subset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None: ...

def random_split(dataset: Dataset[T], lengths: Sequence[int]) -> List[Subset[T]]: ...
