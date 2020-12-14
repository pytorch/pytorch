# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Manipulation of micro-batches."""
import typing
from typing import Callable, Iterable, Iterator, List, Union, cast, Sequence

import torch
from torch import Tensor
import torch.cuda.comm

__all__: List[str] = []


Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]
Function = Callable[[TensorOrTensors], TensorOrTensors]


class Batch:
    """An abstraction of an atomic tensor or a tuple of tensors. This
    eliminates every boilerplate code to classify an atomic tensor or a tuple
    of tensors.
    ::

        x = generate_tensor_or_tensors()
        x = Batch(x)

        # in-place update
        x[0] = F.apply(x[0])
        x[:] = F.apply(*x)

        # f(x) if x is a tensor.
        # f(*x) if x is a tuple of tensors.
        # y is also a batch.
        y = x.call(f)

    """

    def __init__(self, value: TensorOrTensors) -> None:
        self.value = value
        self.atomic = torch.is_tensor(value)

    @property
    def tensor(self) -> Tensor:
        """Retrieves the underlying tensor."""
        if not self.atomic:
            raise AttributeError("not atomic batch")
        return cast(Tensor, self.value)

    @property
    def tensors(self) -> Tensors:
        """Retrieves the underlying tensors."""
        if self.atomic:
            raise AttributeError("batch is atomic")
        return cast(Tensors, self.value)

    @property
    def tensor_or_tensors(self) -> TensorOrTensors:
        """Retrieves the underlying tensor or tensors regardless of type."""
        return self.value

    def call(self, function: Function) -> "Batch":
        """Calls a function by the underlying tensor or tensors. It also wraps
        the output with :class:`Batch`.
        """
        return Batch(function(self.value))

    def __repr__(self) -> str:
        return f"Batch[atomic={self.atomic!r}]({self.value!r})"

    def __iter__(self) -> Iterator[Tensor]:
        if self.atomic:
            yield self.tensor
        else:
            yield from self.tensors

    def __len__(self) -> int:
        return 1 if self.atomic else len(self.tensors)

    def __getitem__(self, index: int) -> Tensor:
        if not self.atomic:
            return self.tensors[index]

        if index != 0:
            raise IndexError("atomic batch allows index 0 only")

        return self.tensor

    # NOTE(sublee): pyflakes can't detect "overload" instead of "typing.overload".
    @typing.overload
    def __setitem__(self, index: int, value: Tensor) -> None:
        ...

    @typing.overload
    def __setitem__(self, index: slice, value: Tensors) -> None:
        ...

    def __setitem__(self, index: Union[int, slice], value: TensorOrTensors) -> None:
        if isinstance(index, int):
            value = cast(Tensor, value)
            self._setitem_by_index(index, value)
        else:
            value = cast(Tensors, value)
            self._setitem_by_slice(index, value)

    def _setitem_by_index(self, index: int, value: Tensor) -> None:
        if not self.atomic:
            i = index
            self.value = self.value[:i] + (value,) + self.value[i + 1 :]  # type: ignore
            return

        if index != 0:
            raise IndexError("atomic batch allows index 0 only")

        self.value = value

    def _setitem_by_slice(self, index: slice, value: Tensors) -> None:
        if not (index.start is index.stop is index.step is None):
            raise NotImplementedError("only slice [:] supported")

        if not self.atomic:
            self.value = value
            return

        if len(value) != 1:
            raise IndexError("atomic batch cannot be replaced with multiple tensors")

        self.value = value[0]


def check(input: TensorOrTensors) -> None:
    """Checks whether the input is a tensor or tensors.

    Raises:
        TypeError: input is not a tensor or tensors.

    """
    if isinstance(input, Sequence):
        for x in input:
            if not isinstance(x, Tensor):
                raise TypeError(f"expected Tensor, but got {input.__class__.__name__}")
        return

    if not isinstance(input, Tensor):
        raise TypeError(f"expected Tensor, but got {input.__class__.__name__}")


def scatter(input: TensorOrTensors, chunks: int) -> List[Batch]:
    """Splits an input mini-batch into multiple micro-batches."""
    inputs: Iterable[TensorOrTensors]

    if isinstance(input, Tensor):
        inputs = input.chunk(chunks)
    else:
        rotated: List[Tensors] = []

        for tensor in input:
            tensors = tensor.chunk(chunks)
            rotated.append(cast(Tensors, tensors))

        inputs = zip(*rotated)

    return [Batch(x) for x in inputs]


def gather(outputs: List[Batch]) -> TensorOrTensors:
    """Concatenates output micro-batches into a mini-batch."""
    output: TensorOrTensors

    if outputs[0].atomic:
        tensors = tuple(b.tensor for b in outputs)
        output = torch.cat(tensors)
    else:
        rotated = [b.tensors for b in outputs]
        output_buf = []

        for tensors in zip(*rotated):
            output_buf.append(torch.cat(tensors))

        output = tuple(output_buf)

    return output
