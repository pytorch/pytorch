from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

T_co = TypeVar('T_co', covariant=True)
class Sampler(Generic[T_co]):
    def __init__(self, data_source: Sized) -> None: ...
    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...

class SequentialSampler(Sampler[int]):
    pass

class RandomSampler(Sampler[int]):
    num_samples: int

    def __init__(self, data_source: Sized, replacement: bool=..., num_samples: Optional[int]=...) -> None: ...

class SubsetRandomSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]) -> None: ...

class WeightedRandomSampler(Sampler[int]):
    def __init__(self, weights: Sequence[float], num_samples: int, replacement: bool=...) -> None: ...

class BatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None: ...
