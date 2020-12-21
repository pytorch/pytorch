# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Provides isolated namespace of skip tensors."""
import abc
from functools import total_ordering
from typing import Any
import uuid

__all__ = ["Namespace"]


@total_ordering
class Namespace(metaclass=abc.ABCMeta):
    """Namespace for isolating skip tensors used by :meth:`isolate()
    <torchpipe.skip.skippable.Skippable.isolate>`.
    """

    __slots__ = ("id",)

    def __init__(self) -> None:
        self.id = uuid.uuid4()

    def __repr__(self) -> str:
        return f"<Namespace '{self.id}'>"

    def __hash__(self) -> int:
        return hash(self.id)

    # Namespaces should support ordering, since SkipLayout will sort tuples
    # including a namespace. But actual order between namespaces is not
    # important. That's why they are ordered by version 4 UUID which generates
    # random numbers.
    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Namespace):
            return self.id < other.id
        return False

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Namespace):
            return self.id == other.id
        return False


# 'None' is the default namespace,
# which means that 'isinstance(None, Namespace)' is 'True'.
Namespace.register(type(None))
