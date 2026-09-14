# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""

from typing import NoReturn


class _BaseDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index) -> NoReturn:
        raise NotImplementedError


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
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
            data = next(self.dataset_iter)
        return self.collate_fn(data)


def _is_contiguous_batch(indices):
    # Check whether indices form a contiguous increasing range, which allows
    # a zero-copy slice instead of per-sample indexing.
    return (
        len(indices) > 0
        and indices[-1] - indices[0] + 1 == len(indices)
        and all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1))
    )


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        # Detect TensorDataset + default_collate once at construction so fetch()
        # pays no isinstance/identity cost per batch.
        from torch.utils.data._utils.collate import default_collate
        from torch.utils.data.dataset import TensorDataset

        self._tensor_fast_path = (
            auto_collation
            and isinstance(dataset, TensorDataset)
            and collate_fn is default_collate
        )

    def fetch(self, possibly_batched_index):
        if self._tensor_fast_path and _is_contiguous_batch(possibly_batched_index):
            start = possibly_batched_index[0]
            end = possibly_batched_index[-1] + 1
            return tuple(t[start:end] for t in self.dataset.tensors)

        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
