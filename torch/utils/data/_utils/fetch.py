r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""

from collections.abc import Iterable, Iterator
from typing import Any, Callable, Generic, TypeVar, Union


_T = TypeVar("_T")
IndexType = Union[int, slice, list[int]]


class _BaseDatasetFetcher(Generic[_T]):
    def __init__(
        self,
        dataset: object,
        auto_collation: bool,
        collate_fn: Callable[[Any], _T],
        drop_last: bool,
    ) -> None:
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index: IndexType) -> _T:
        raise NotImplementedError


class _IterableDatasetFetcher(_BaseDatasetFetcher[_T]):
    def __init__(
        self,
        dataset: Iterable[Any],
        auto_collation: bool,
        collate_fn: Callable[[Any], _T],
        drop_last: bool,
    ) -> None:
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter: Iterator[Any] = iter(dataset)
        self.ended: bool = False

    def fetch(self, possibly_batched_index: IndexType) -> _T:
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data: list[Any] = []
            if isinstance(possibly_batched_index, list):
                for _ in possibly_batched_index:
                    try:
                        data.append(next(self.dataset_iter))
                    except StopIteration:
                        self.ended = True
                        break
                if len(data) == 0 or (
                    self.drop_last and len(data) < len(possibly_batched_index)
                ):
                    raise StopIteration
            else:
                # Handle single index case for auto_collation
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    raise
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher[_T]):
    def __init__(
        self,
        dataset: Any,  # Dataset that supports indexing
        auto_collation: bool,
        collate_fn: Callable[[Any], _T],
        drop_last: bool,
    ) -> None:
        super().__init__(dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index: IndexType) -> _T:
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                if isinstance(possibly_batched_index, list):
                    data = [self.dataset[idx] for idx in possibly_batched_index]  # type: ignore[index]
                else:
                    # Single index but auto_collation is True - wrap in list
                    data = [self.dataset[possibly_batched_index]]  # type: ignore[index]
        else:
            data = self.dataset[possibly_batched_index]  # type: ignore[index]
        return self.collate_fn(data)
