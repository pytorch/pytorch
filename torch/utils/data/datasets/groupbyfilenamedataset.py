from torch.utils.data.dataset import IterableDataset
from typing import List, Iterable, Iterator
from collections import OrderedDict

import os

class GroupByFilenameIterableDataset(IterableDataset):
    r""" :class:`GroupByFilenameIterableDataset`.

    IterableDataset to group binary streams from input iterables by pathname without extension,
    yield a list with `group_size` items in it, each item in the list is a tuple of
    pathname and binary stream

    args:
        dataset: Iterable dataset that provides pathname and zip binary stream in tuples
        group_size: the size of group
        buffer_size: the size of buffer which is used to store non-grouped but iterated streams
        length: a nominal length of the dataset
    """

    def __init__(
            self,
            dataset : Iterable,
            group_size : int = 1,
            buffer_size : int = 10,
            length: int = -1):
        super().__init__()
        self.dataset : Iterable = dataset
        self.group_size : int = group_size
        self.buffer_size : int = buffer_size
        self.stream_buffer : OrderedDict = OrderedDict()
        self.length : int = length


    def __scan_stream_buffer(self, group_size : int):
        # scan current stream buffer, return a group of data or partial group of data
        stream_buffer = self.stream_buffer
        res = []
        removed_pathnames = []
        key = None
        for pathname, stream in stream_buffer.items():
            name_ext = os.path.splitext(pathname)
            if key is None:
                key = name_ext[0]
                res.append((pathname, stream))
                removed_pathnames.append(pathname)
            elif key == name_ext[0]:
                res.append((pathname, stream))
                removed_pathnames.append(pathname)
            if len(res) == group_size:
                break
        for pathname in removed_pathnames:
            del stream_buffer[pathname]
        return res


    def __iter__(self) -> Iterator[list]:
        assert self.group_size > 0
        assert self.buffer_size >= 0

        group_size = self.group_size
        buffer_size = self.buffer_size
        stream_buffer = self.stream_buffer
        res : List[tuple] = []
        for data in self.dataset:
            # scan stream buffer before reading new data from dataset.
            # try finding a group in each iteration.
            # note that stream_buffer won't be scanned if size of `res` is not 0, since
            # we need to find the rest of the items in current group from input dataset
            while len(res) == 0 and len(stream_buffer) > 0:
                res = self.__scan_stream_buffer(group_size)
                if len(res) == group_size:
                    yield res
                    res = []

            # try grouping data from input dataset if stream_buffer has no enough items to group
            key = None
            if len(res) == 0:
                # start finding a new group
                key = os.path.splitext(data[0])[0]
                res.append((data[0], data[1]))
            else:
                # find the rest of item in a group that initiated from stream_buffer
                if key is None:
                    key = os.path.splitext(res[0][0])[0]
                if key == os.path.splitext(data[0])[0]:
                    res.append((data[0], data[1]))
                elif len(stream_buffer) < buffer_size:
                    stream_buffer[data[0]] = data[1]
                else:
                    raise OverflowError(
                        "stream_buffer is full, please adjust the order of data "
                        "in the input dataset or increase the buffer size!")

            if len(res) == group_size:
                yield res
                res = []

        # finally handle the rest of items in the stream_buffer if any
        while len(res) == 0 and len(stream_buffer) >= group_size:
            res = self.__scan_stream_buffer(group_size)
            if len(res) == group_size:
                yield res
                res = []

        if len(res) > 0:
            msg = "Not able to group [{}] with group size {}.".format(
                ','.join([v[0] for v in res]), str(group_size))
            raise RuntimeError(msg)
        if len(stream_buffer) > 0:
            msg = "Not able to group [{}] with group size {}".format(
                ','.join(v for v in list(stream_buffer)), str(group_size))
            raise RuntimeError(msg)


    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
