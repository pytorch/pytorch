#!/usr/bin/env python3

from collections.abc import Iterator
from typing import Callable, TypeVar
from typing_extensions import Self


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_T = TypeVar("_T")

__all__ = ["CyclingIterator"]


class CyclingIterator(Iterator[_T]):
    """
    An iterator decorator that cycles through the
    underlying iterator "n" times. Useful to "unroll"
    the dataset across multiple training epochs.

    The generator function is called as ``generator_fn(epoch)``
    to obtain the underlying iterator, where ``epoch`` is a
    number less than or equal to ``n`` representing the ``k``th cycle

    For example if ``generator_fn`` always returns ``[1,2,3]``
    then ``CyclingIterator(n=2, generator_fn)`` will iterate through
    ``[1,2,3,1,2,3]``
    """

    def __init__(
        self,
        n: int,
        generator_fn: Callable[[int], Iterator[_T]],
        start_epoch: int = 0,
    ):
        self._n = n
        self._epoch = start_epoch
        self._generator_fn = generator_fn
        self._iter = generator_fn(self._epoch)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> _T:
        try:
            return next(self._iter)
        except StopIteration as eod:  # eod == end of data
            if self._epoch < self._n - 1:
                self._epoch += 1
                self._iter = self._generator_fn(self._epoch)
                return self.__next__()
            else:
                raise eod
