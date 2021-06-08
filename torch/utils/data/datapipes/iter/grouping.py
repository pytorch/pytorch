import functools
import os
import warnings

from torch.utils.data import IterDataPipe, functional_datapipe
from typing import Any, Callable, Dict, Iterator, List, Optional, Sized, Tuple, TypeVar

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('batch')
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
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                self.length = len(self.datapipe) // self.batch_size
            else:
                self.length = (len(self.datapipe) + self.batch_size - 1) // self.batch_size
            return self.length
        raise NotImplementedError


@functional_datapipe('bucket_batch')
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
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                self.length = len(self.datapipe) // self.batch_size
            else:
                self.length = (len(self.datapipe) + self.batch_size - 1) // self.batch_size
            return self.length
        raise NotImplementedError


# defaut group key is the file pathname without the extension.
# Assuming the passed in data is a tuple and 1st item is file's pathname.
def default_group_key_fn(dataitem: Tuple[str, Any]):
    return os.path.splitext(dataitem[0])[0]


def default_sort_data_fn(datalist: List[Tuple[str, Any]]):
    txt_ext = ['.json', '.jsn', '.txt', '.text']

    def cmp_fn(a : Tuple[str, Any], b : Tuple[str, Any]):
        a_is_txt = os.path.splitext(a[0])[1] in txt_ext
        b_is_txt = os.path.splitext(b[0])[1] in txt_ext

        # if a is txt but b is not, b go front
        if a_is_txt and not b_is_txt:
            return 1
        # if a is not txt but b is txt, a go front
        if not a_is_txt and b_is_txt:
            return -1
        # if a and b both are or are not txt, sort in alphabetic order
        if a[0] < b[0]:
            return -1
        elif a[0] > b[0]:
            return 1
        return 0

    return sorted(datalist, key=functools.cmp_to_key(cmp_fn))


@functional_datapipe('group_by_key')
class GroupByKeyIterDataPipe(IterDataPipe[list]):
    r""" :class:`GroupByKeyIterDataPipe`.

    Iterable datapipe to group data from input iterable by keys which are generated from `group_key_fn`,
    yields a list with `group_size` items in it, each item in the list is a tuple of key and data

    args:
        datapipe: Iterable datapipe that provides data. (typically str key (eg. pathname) and data stream in tuples)
        group_size: the size of group
        max_buffer_size: the max size of stream buffer which is used to store not yet grouped but iterated data
        group_key_fn: a function which is used to generate group key from the data in the input datapipe
        sort_data_fn: a function which is used to sort the grouped data before yielding back
        length: a nominal length of the datapipe
    """
    datapipe: IterDataPipe[Tuple[str, Any]]
    group_size: int
    max_buffer_size: int
    group_key_fn: Callable
    sort_data_fn: Callable
    curr_buffer_size: int
    stream_buffer: Dict[str, List[Tuple[str, Any]]]
    length: int

    def __init__(
            self,
            datapipe: IterDataPipe[Tuple[str, Any]],
            *,
            group_size: int,
            max_buffer_size: Optional[int] = None,
            group_key_fn: Callable = default_group_key_fn,
            sort_data_fn: Callable = default_sort_data_fn,
            length: int = -1):
        super().__init__()

        assert group_size > 0
        self.datapipe = datapipe
        self.group_size = group_size

        # default max buffer size is group_size * 10
        self.max_buffer_size = max_buffer_size if max_buffer_size is not None else group_size * 10
        assert self.max_buffer_size >= self.group_size

        self.group_key_fn = group_key_fn  # type: ignore[assignment]
        self.sort_data_fn = sort_data_fn  # type: ignore[assignment]
        self.curr_buffer_size = 0
        self.stream_buffer = {}
        self.length = length

    def __iter__(self) -> Iterator[list]:
        if self.group_size == 1:
            for data in self.datapipe:
                yield [data]
        else:
            for data in self.datapipe:
                key = self.group_key_fn(data)
                if key not in self.stream_buffer:
                    self.stream_buffer[key] = []
                res = self.stream_buffer[key]
                res.append(data)
                if len(res) == self.group_size:
                    yield self.sort_data_fn(res)
                    del self.stream_buffer[key]
                    self.curr_buffer_size = self.curr_buffer_size - self.group_size + 1
                else:
                    if self.curr_buffer_size == self.max_buffer_size:
                        raise OverflowError(
                            "stream_buffer is overflow, please adjust the order of data "
                            "in the input datapipe or increase the buffer size!")
                    self.curr_buffer_size = self.curr_buffer_size + 1

            if self.curr_buffer_size > 0:
                msg = "Not able to group [{}] with group size {}.".format(
                    ','.join([v[0] for _, vs in self.stream_buffer.items() for v in vs]), str(self.group_size))
                raise RuntimeError(msg)

    def __len__(self) -> int:
        if self.length == -1:
            raise NotImplementedError
        return self.length
