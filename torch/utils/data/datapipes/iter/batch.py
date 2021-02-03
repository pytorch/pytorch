import warnings
from torch.utils.data import IterDataPipe
from typing import TypeVar, Optional, Iterator, List, Sized, Callable

T_co = TypeVar('T_co', covariant=True)


class BatchIterDataPipe(IterDataPipe[List[T_co]]):
    r""" :class:`BatchIterDataPipe`.

    Iterable DataPipe to create mini-batches of data. An outer dimension will be added as
    `batch_size` if `drop_last` is set to `True`, or `length % batch_size` for the
    last batch if `drop_last` is set to `False`.
    args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
    """
    datapipe: IterDataPipe[T_co]
    batch_size: int
    drop_last: bool
    length: Optional[int]

    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 *,
                 batch_size: int,
                 drop_last: bool = False,
                 ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length = None

    def __iter__(self) -> Iterator[List[T_co]]:
        batch: List[T_co] = []
        for x in self.datapipe:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield batch
                batch.clear()
        if len(batch) > 0:
            if not self.drop_last:
                yield batch
            batch.clear()

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            if self.drop_last:
                self.length = len(self.datapipe) // self.batch_size
            else:
                self.length = (len(self.datapipe) + self.batch_size - 1) // self.batch_size
            return self.length
        raise NotImplementedError


class BucketBatchIterDataPipe(IterDataPipe[List[T_co]]):
    r""" :class:`BucketBatchIterDataPipe`.

    Iterable DataPipe to create mini-batches of data from sorted bucket. An outer
    dimension will be added as `batch_size` if `drop_last` is set to `True`,
    or `length % batch_size` for the last batch if `drop_last` is set to `False`.
        args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
        bucket_size_mul: The multiplier to specify the size of bucket
        sort_key: Callable to specify the comparison key for sorting within bucket
    """
    datapipe: IterDataPipe[T_co]
    batch_size: int
    drop_last: bool
    bucket_size_mul: int
    sort_key: Optional[Callable]
    length: Optional[int]

    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 *,
                 batch_size: int,
                 drop_last: bool = False,
                 bucket_size_mul: int = 100,
                 sort_key: Optional[Callable] = None,
                 ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket_size = batch_size * bucket_size_mul
        self.sort_key = sort_key
        if sort_key is not None and sort_key.__name__ == '<lambda>':
            warnings.warn("Lambda function is not supported for pickle, "
                          "please use regular python function instead.")
        self.bucket_ds = BatchIterDataPipe(datapipe, batch_size=self.bucket_size, drop_last=False)
        self.length = None

    def __iter__(self) -> Iterator[List[T_co]]:
        # Bucket without sorting remains same order, directly returns BatchDataset
        if self.sort_key is None:
            yield from BatchIterDataPipe(self.datapipe, batch_size=self.batch_size, drop_last=self.drop_last)
        else:
            bucket: List[T_co]
            batch: List[T_co] = []
            for bucket in self.bucket_ds:
                # In-place sort within bucket
                bucket.sort(key=self.sort_key)
                for start in range(0, len(bucket), self.batch_size):
                    batch = bucket[start: start + self.batch_size]
                    if len(batch) == self.batch_size or not self.drop_last:
                        yield batch

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            if self.drop_last:
                self.length = len(self.datapipe) // self.batch_size
            else:
                self.length = (len(self.datapipe) + self.batch_size - 1) // self.batch_size
            return self.length
        raise NotImplementedError
