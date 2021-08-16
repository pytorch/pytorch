import functools
import os
import warnings

from collections import defaultdict

from torch.utils.data import IterDataPipe, functional_datapipe
from typing import Any, Callable, Dict, Iterator, List, Optional, Sized, Tuple, TypeVar, DefaultDict

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('sharding_filter')
class ShardingFilterIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe
        self.num_of_instances = 1
        self.instance_id = 0

    def is_shardable(self):
        return True

    def apply_sharding(self, num_of_instances, instance_id):
        self.num_of_instances = num_of_instances
        self.instance_id = instance_id

    def __iter__(self):
        for i, item in enumerate(self.source_datapipe):
            if i % self.num_of_instances == self.instance_id:
                yield item


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
        unbatch_level: Specifies if it necessary to unbatch source data before
            applying new batching rule
    """
    datapipe: IterDataPipe[T_co]
    batch_size: int
    drop_last: bool
    length: Optional[int]

    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 batch_size: int,
                 drop_last: bool = False,
                 unbatch_level: int = 0,
                 ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super().__init__()
        if unbatch_level == 0:
            self.datapipe = datapipe
        else:
            self.datapipe = datapipe.unbatch(unbatch_level=unbatch_level)
        self.unbatch_level = unbatch_level
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length = None

    def __iter__(self) -> Iterator[List[T_co]]:
        batch: List[T_co] = []
        for x in self.datapipe:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            if not self.drop_last:
                yield batch
            batch = []

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        if isinstance(self.datapipe, Sized) and self.unbatch_level == 0:
            if self.drop_last:
                self.length = len(self.datapipe) // self.batch_size
            else:
                self.length = (len(self.datapipe) + self.batch_size - 1) // self.batch_size
            return self.length
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


@functional_datapipe('unbatch')
class UnBatchIterDataPipe(IterDataPipe):
    r""" :class:`UnBatchIterDataPipe`.

    Iterable DataPipe to undo batching of data. In other words, it flattens the data up to the specified level
    within a batched DataPipe.
    args:
        datapipe: Iterable DataPipe being un-batched
        unbatch_level: Defaults to `1` (only flattening the top level). If set to `2`, it will flatten the top 2 levels,
        and `-1` will flatten the entire DataPipe.
    """

    def __init__(self, datapipe, unbatch_level: int = 1):
        self.datapipe = datapipe
        self.unbatch_level = unbatch_level

    def __iter__(self):
        for element in self.datapipe:
            for i in self._dive(element, unbatch_level=self.unbatch_level):
                yield i

    def _dive(self, element, unbatch_level):
        if unbatch_level < -1:
            raise ValueError("unbatch_level must be -1 or >= 0")
        if unbatch_level == -1:
            if isinstance(element, list):
                for item in element:
                    for i in self._dive(item, unbatch_level=-1):
                        yield i
            else:
                yield element
        elif unbatch_level == 0:
            yield element
        else:
            if not isinstance(element, list):
                raise IndexError(f"unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe")
            for item in element:
                for i in self._dive(item, unbatch_level=unbatch_level - 1):
                    yield i


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
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


# defaut group key is the file pathname without the extension.
# Assuming the passed in data is a tuple and 1st item is file's pathname.
def default_group_key_fn(dataitem: Tuple[str, Any]):
    return os.path.splitext(dataitem[0])[0]


def default_sort_data_fn(datalist: List[Tuple[str, Any]]):
    txt_ext = ['.json', '.jsn', '.txt', '.text']

    def cmp_fn(a: Tuple[str, Any], b: Tuple[str, Any]):
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


@functional_datapipe('groupby')
class GroupByIterDataPipe(IterDataPipe):
    # TODO(VtalyFedyunin): Add inline docs and tests (they are partially available in notebooks)
    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 group_key_fn: Callable,
                 *,
                 buffer_size: int = 10000,
                 group_size: Optional[int] = None,
                 unbatch_level: int = 0,
                 guaranteed_group_size: Optional[int] = None,
                 drop_remaining: bool = False):
        if unbatch_level == 0:
            self.datapipe = datapipe
        else:
            self.datapipe = datapipe.unbatch(unbatch_level=unbatch_level)
        self.group_key_fn = group_key_fn
        self.buffer_size = buffer_size
        self.group_size = group_size
        self.guaranteed_group_size = None
        if group_size is not None and buffer_size is not None:
            assert group_size > 0 and group_size <= buffer_size
            self.guaranteed_group_size = group_size
        if guaranteed_group_size is not None:
            assert guaranteed_group_size > 0 and group_size is not None and guaranteed_group_size <= group_size
            self.guaranteed_group_size = guaranteed_group_size
        self.drop_remaining = drop_remaining

    def _remove_biggest_key(self, buffer_elements, buffer_size):
        biggest_key = None
        biggest_size = 0
        result_to_yield = None
        for findkey in buffer_elements.keys():
            if len(buffer_elements[findkey]) > biggest_size:
                biggest_size = len(buffer_elements[findkey])
                biggest_key = findkey

        if self.guaranteed_group_size is not None and biggest_size < self.guaranteed_group_size and not self.drop_remaining:
            raise RuntimeError('Failed to group items', str(buffer_elements[biggest_key]))

        if self.guaranteed_group_size is None or biggest_size >= self.guaranteed_group_size:
            result_to_yield = buffer_elements[biggest_key]

        new_buffer_size = buffer_size - biggest_size
        del buffer_elements[biggest_key]

        return (result_to_yield, new_buffer_size)

    def __iter__(self):
        buffer_elements: DefaultDict[Any, List] = defaultdict(list)
        buffer_size = 0
        for x in self.datapipe:
            key = self.group_key_fn(x)

            if self.group_size is not None and self.group_size == len(buffer_elements[key]):
                yield buffer_elements[key]
                buffer_size -= len(buffer_elements[key])
                del buffer_elements[key]

            if buffer_size == self.buffer_size:
                (result_to_yield, buffer_size) = self._remove_biggest_key(buffer_elements, buffer_size)
                if result_to_yield is not None:
                    yield result_to_yield

            buffer_elements[key].append(x)
            buffer_size += 1

        while buffer_size:
            (result_to_yield, buffer_size) = self._remove_biggest_key(buffer_elements, buffer_size)
            if result_to_yield is not None:
                yield result_to_yield


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
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
        return self.length
