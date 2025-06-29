r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""
from typing import Any, Callable, Generic, TYPE_CHECKING, TypeVar, Union

from torch.utils.data.dataset import Dataset, IterableDataset


if TYPE_CHECKING:
    from collections.abc import Iterator


_T = TypeVar("_T")
_collate_fn_t = Callable[[Union[list[_T], _T]], Any]


class _BaseDatasetFetcher(Generic[_T]):
    def __init__(
        self,
        dataset: Union[Dataset[_T], IterableDataset[_T]],
        auto_collation: bool,
        collate_fn: _collate_fn_t[_T],
        drop_last: bool,
    ) -> None:
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index: Any) -> Any:
        raise NotImplementedError


class _IterableDatasetFetcher(_BaseDatasetFetcher[_T]):
    def __init__(
        self,
        dataset: IterableDataset[_T],
        auto_collation: bool,
        collate_fn: _collate_fn_t[_T],
        drop_last: bool,
    ) -> None:
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter: Iterator[_T] = iter(dataset)
        self.ended: bool = False

    def fetch(self, possibly_batched_index: Any) -> Any:
        if self.ended:
            raise StopIteration

        data: Union[list[_T], _T]
        if self.auto_collation:
            batch_data: list[_T] = []
            for _ in possibly_batched_index:
                try:
                    batch_data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(batch_data) == 0 or (
                self.drop_last and len(batch_data) < len(possibly_batched_index)
            ):
                raise StopIteration
            data = batch_data
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher[_T]):
    def fetch(self, possibly_batched_index: Any) -> Any:
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
