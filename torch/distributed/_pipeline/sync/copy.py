# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Autograd functions for stream-aware CUDA copy. It is used to overlap copy
and computation on the same GPU.
"""
from collections import deque
from typing import Deque, List, Optional, Tuple

import torch
from torch import Tensor

from .stream import AbstractStream, current_stream, get_device, record_stream, use_stream, wait_stream

__all__: List[str] = []


Tensors = Tuple[Tensor, ...]


# Common interface between :class:`Copy` and :class:`Wait`.
class Context:
    prev_stream: AbstractStream
    next_stream: AbstractStream


class Copy(torch.autograd.Function):
    """Copies tensors on specific streams."""

    @staticmethod
    # type: ignore
    def forward(ctx: Context, prev_stream: AbstractStream, next_stream: AbstractStream, *input: Tensor,) -> Tensors:
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        output = []
        output_stream = current_stream(get_device(next_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            for x in input:
                y = x.to(get_device(next_stream), non_blocking=True)
                output.append(y)

                # 'prev_stream' is not where 'x' has been allocated.
                record_stream(x, prev_stream)
                # 'y' has been allocated on 'next_stream'.
                # It might be used on the current stream captured as 'output_stream'.
                record_stream(y, output_stream)

        return tuple(output)

    @staticmethod
    def backward(ctx: Context, *grad_output: Tensor,) -> Tuple[Optional[Tensor], ...]:
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        grad_input: Deque[Tensor] = deque(maxlen=len(grad_output))
        input_stream = current_stream(get_device(prev_stream))

        with use_stream(prev_stream), use_stream(next_stream):
            for x in reversed(grad_output):
                y = x.to(get_device(prev_stream), non_blocking=True)
                grad_input.appendleft(y)

                # 'next_stream' is not where 'x' has been allocated.
                record_stream(x, next_stream)
                # 'y' has been allocated on 'prev_stream'.
                # It might be used on the current stream captured as 'input_stream'.
                record_stream(y, input_stream)

        grad_streams: Tuple[Optional[Tensor], ...] = (None, None)
        return grad_streams + tuple(grad_input)


class Wait(torch.autograd.Function):
    """Synchronizes a stream to another stream.

    Place it just before you want to start an operation on the next stream,
    provided that all operations on the previous stream are done.

    """

    @staticmethod
    # type: ignore
    def forward(ctx: Context, prev_stream: AbstractStream, next_stream: AbstractStream, *input: Tensor,) -> Tensors:
        ctx.prev_stream = prev_stream
        ctx.next_stream = next_stream

        wait_stream(next_stream, prev_stream)

        return tuple(x.detach() for x in input)

    @staticmethod
    def backward(ctx: Context, *grad_input: Tensor,) -> Tuple[Optional[Tensor], ...]:
        prev_stream = ctx.prev_stream
        next_stream = ctx.next_stream

        wait_stream(prev_stream, next_stream)

        grad_streams: Tuple[Optional[Tensor], ...] = (None, None)
        return grad_streams + grad_input
