from torch.utils.data import IterDataPipe
from typing import Dict, List, Tuple, Any, Callable, Iterable, Iterator, Union

import os
import functools


# defaut group key is the file pathname without the extension.
# Assuming the passed in data is a tuple and 1st item is file's pathname.
def default_group_key_fn(dataitem : Tuple[str, Any]):
    return os.path.splitext(dataitem[0])[0]


def default_sort_data_fn(datalist : List[Tuple[str, Any]]):
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


class GroupByKeyIterDataPipe(IterDataPipe):
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

    def __init__(
            self,
            datapipe : Iterable[Tuple[str, Any]],
            *,
            group_size : int,
            max_buffer_size : Union[int, None] = None,
            group_key_fn : Callable = default_group_key_fn,
            sort_data_fn : Callable = default_sort_data_fn,
            length: int = -1):
        super().__init__()

        assert group_size > 0
        self.datapipe : Iterable[Tuple[str, Any]] = datapipe
        self.group_size : int = group_size

        # default max buffer size is group_size * 10
        self.max_buffer_size = max_buffer_size if max_buffer_size is not None else group_size * 10
        assert self.max_buffer_size >= self.group_size

        self.group_key_fn : Callable = group_key_fn
        self.sort_data_fn : Callable = sort_data_fn
        self.curr_buffer_size : int = 0
        self.stream_buffer : Dict[str, List[Tuple[str, Any]]] = {}
        self.length : int = length


    def __iter__(self) -> Iterator[list]:
        if self.group_size == 1:
            for data in self.datapipe:
                yield [data]
        else:
            for data in self.datapipe:
                key = self.group_key_fn(data)
                res = self.stream_buffer.get(key, []) + [data]
                if len(res) == self.group_size:
                    yield self.sort_data_fn(res)
                    del self.stream_buffer[key]
                    self.curr_buffer_size = self.curr_buffer_size - self.group_size + 1
                else:
                    if self.curr_buffer_size == self.max_buffer_size:
                        raise OverflowError(
                            "stream_buffer is overflow, please adjust the order of data "
                            "in the input datapipe or increase the buffer size!")
                    self.stream_buffer[key] = res
                    self.curr_buffer_size = self.curr_buffer_size + 1

            if self.curr_buffer_size > 0:
                msg = "Not able to group [{}] with group size {}.".format(
                    ','.join([v[0] for _, vs in self.stream_buffer.items() for v in vs]), str(self.group_size))
                raise RuntimeError(msg)


    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
