# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Checkpointing with preceding recomputation.

PyTorch already provides the official checkpointing utilities in
:mod:`torch.utils.checkpoint`. The official checkpointing combines
recomputation and recursive backpropagation into one autograd function named
``CheckpointFunction``. Hence, the recomputation can be started only when the
gradients arrive to the function. In Pipe, the recomputation needs to precede
the gradient arrival to minimize the GPU idle time.

We solve this problem by introducing separate autograd functions named
:class:`Recompute` and :class:`Checkpoint`. Each function represents
recomputation and recursive backpropagation, respectively. We can manipulate
the control flow in aspect of both the autograd engine and CUDA with a pair of
the functions.

Specifically, we place CUDA stream synchronization between :class:`Recompute`
and :class:`Checkpoint` to delay only :class:`Checkpoint` until the gradient is
copied entirely.

"""
from collections import deque
from contextlib import contextmanager
import threading
from typing import (
    Any,
    Deque,
    Generator,
    List,
    Optional,
    Protocol,
    Union,
    Sequence,
    Tuple
)

import torch
from torch import Tensor
import torch.autograd

from .dependency import fork, join
from .microbatch import Batch
from .phony import get_phony

__all__ = ["Function", "checkpoint", "Checkpointing", "ThreadLocal", "enable_checkpointing",
           "enable_recomputing", "is_checkpointing", "is_recomputing", "Context", "save_rng_states",
           "restore_rng_states", "Checkpoint", "Recompute"]


Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

# Types for shared memory between Checkpoint and Recompute.
Recomputed = Tuple[TensorOrTensors, Tensors]  # (output, input_leaf)
RNGStates = Tuple[Tensor, Optional[Tensor]]  # (cpu_rng_state, gpu_rng_state)


# Protocol with __call__ instead of Callable can be used as an attribute type.
# See: https://github.com/python/mypy/issues/708#issuecomment-561735949
class Function(Protocol):
    def __call__(self, input: TensorOrTensors) -> TensorOrTensors:
        ...


def checkpoint(function: Function, input):
    """Makes a checkpoint with a simple interface like
    :func:`torch.utils.checkpoint.checkpoint`. It's only used to test or debug
    :class:`Checkpoint` and :class:`Recompute` without boilerplate.
    """
    batch = Batch(input)

    chk = Checkpointing(function, batch)
    batch = chk.checkpoint()
    chk.recompute(batch)

    return batch.values


class Checkpointing:
    """Generates a pair of :class:`Checkpoint` and :class:`Recompute`."""

    def __init__(self, function: Function, batch: Batch) -> None:
        self.function = function
        self.batch = batch

        # Shared memory between Checkpoint and Recompute. 1-length deque is
        # used for mutability and length limitation.
        self.recomputed: Deque[Recomputed] = deque(maxlen=1)
        self.rng_states: Deque[RNGStates] = deque(maxlen=1)

    def checkpoint(self) -> Batch:
        """Returns a batch applied by :class:`Checkpoint`."""
        input_atomic = self.batch.atomic
        inputs = tuple(self.batch)

        # Use a phony which requires grad to ensure that Checkpoint can be
        # tracked by the autograd engine even when none of the input tensors
        # require grad.
        phony = get_phony(self.batch.get_device(), requires_grad=True)

        output = Checkpoint.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *inputs)

        # Gradients are only supported for float Tensors.
        if isinstance(output, tuple):
            output = tuple([x.detach() if torch.is_tensor(x) and not x.is_floating_point() else x for x in output])

        return Batch(output)

    def recompute(self, batch: Batch) -> None:
        """Applies :class:`Recompute` to the batch in place."""
        input_atomic = self.batch.atomic
        inputs = tuple(self.batch)

        # Use a tensor in the batch to tie together fork-join
        tensor_idx = batch.find_tensor_idx()
        # batch[tensor_idx] is always requiring grad, because it has been passed
        # checkpoint with a phony requiring grad.
        batch[tensor_idx], phony = fork(batch[tensor_idx])
        phony = Recompute.apply(phony, self.recomputed, self.rng_states, self.function, input_atomic, *inputs)
        batch[tensor_idx] = join(batch[tensor_idx], phony)


class ThreadLocal(threading.local):
    def __init__(self) -> None:
        self.is_checkpointing = False
        self.is_recomputing = False


thread_local = ThreadLocal()


@contextmanager
def enable_checkpointing() -> Generator[None, None, None]:
    """Makes :func:`is_checkpointing` return :data:`True` within a context."""
    orig = thread_local.is_checkpointing
    thread_local.is_checkpointing = True
    try:
        yield
    finally:
        thread_local.is_checkpointing = orig


@contextmanager
def enable_recomputing() -> Generator[None, None, None]:
    """Makes :func:`is_recomputing` return :data:`True` within a context."""
    orig = thread_local.is_recomputing
    thread_local.is_recomputing = True
    try:
        yield
    finally:
        thread_local.is_recomputing = orig


def is_checkpointing() -> bool:
    """Whether the current forward propagation is under checkpointing.

    Returns:
        bool: :data:`True` if it's under checkpointing.

    """
    return thread_local.is_checkpointing


def is_recomputing() -> bool:
    """Whether the current forward propagation is under checkpoint
    recomputation. Use this to prevent duplicated side-effects at forward
    propagation::

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, input):
                if not is_recomputing():
                    self.counter += 1
                return input

    Returns:
        bool: :data:`True` if it's under checkpoint recomputation.

    .. seealso:: :ref:`Detecting Recomputation`

    """
    return thread_local.is_recomputing


class Context:
    """The common interface between the :class:`Checkpoint` and
    :class:`Recompute` context.
    """

    recomputed: Deque[Recomputed]
    rng_states: Deque[RNGStates]
    function: Function
    input_atomic: bool
    inputs: Sequence[Any]

    saved_tensors: Tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None:  # pragma: no cover
        pass


def save_rng_states(device: torch.device, rng_states: Deque[RNGStates],) -> None:
    """:meth:`Checkpoint.forward` captures the current PyTorch's random number
    generator states at CPU and GPU to reuse in :meth:`Recompute.backward`.

    .. seealso:: :ref:`Referential Transparency`

    """
    cpu_rng_state = torch.get_rng_state()

    gpu_rng_state: Optional[Tensor]
    if device.type == "cuda":
        gpu_rng_state = torch.cuda.get_rng_state(device)
    else:
        gpu_rng_state = None

    rng_states.append((cpu_rng_state, gpu_rng_state))


@contextmanager
def restore_rng_states(device: torch.device, rng_states: Deque[RNGStates],) -> Generator[None, None, None]:
    """:meth:`Recompute.backward` restores the random number generator states
    captured by :func:`save_rng_states` within its context.

    .. seealso:: :ref:`Referential Transparency`

    """
    cpu_rng_state, gpu_rng_state = rng_states.pop()

    gpu_devices: List[torch.device] = []
    if device.type == "cuda":
        gpu_devices.append(device)

    with torch.random.fork_rng(gpu_devices):
        torch.set_rng_state(cpu_rng_state)
        if gpu_rng_state is not None:
            torch.cuda.set_rng_state(gpu_rng_state, device)
        yield


class Checkpoint(torch.autograd.Function):
    @staticmethod
    # type: ignore[override]
    def forward(
        ctx: Context,
        phony: Tensor,
        recomputed: Deque[Recomputed],
        rng_states: Deque[RNGStates],
        function: Function,
        input_atomic: bool,
        *inputs,
    ):
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states

        save_rng_states(phony.device, ctx.rng_states)

        ctx.function = function
        ctx.input_atomic = input_atomic
        if input_atomic:
            tensors = [inputs[0]]
        else:
            tensors = []
            for input in inputs:
                if torch.is_tensor(input):
                    tensors.append(input)

        ctx.save_for_backward(*tensors)

        with torch.no_grad(), enable_checkpointing():
            if input_atomic:
                assert len(inputs) == 1
                output = function(inputs[0])
            else:
                output = function(*inputs)
        return output

    @staticmethod
    def backward(ctx: Context, *grad_output: Tensor,) -> Tuple[Optional[Tensor], ...]:  # pragma: no cover
        output, input_leaf = ctx.recomputed.pop()

        if isinstance(output, tuple):
            outputs = output
        else:
            outputs = (output,)
        if any(torch.is_tensor(y) and y.requires_grad for y in outputs):
            tensors = tuple([x for x in outputs if torch.is_tensor(x) and x.requires_grad])
            torch.autograd.backward(tensors, grad_output)

        grad_input: List[Optional[Tensor]] = [None, None, None, None, None]
        grad_input.extend(x.grad if torch.is_tensor(x) else None for x in input_leaf)
        return tuple(grad_input)


class Recompute(torch.autograd.Function):
    @staticmethod
    # type: ignore[override]
    def forward(
        ctx: Context,
        phony: Tensor,
        recomputed: Deque[Recomputed],
        rng_states: Deque[RNGStates],
        function: Function,
        input_atomic: bool,
        *inputs,
    ) -> Tensor:
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states

        ctx.function = function
        ctx.input_atomic = input_atomic
        ctx.inputs = inputs
        if input_atomic:
            tensors = [inputs[0]]
        else:
            tensors = []
            for input in inputs:
                if torch.is_tensor(input):
                    tensors.append(input)
        ctx.save_for_backward(*tensors)

        return phony

    @staticmethod
    def backward(ctx: Context, *grad_output: Tensor) -> Tuple[None, ...]:  # pragma: no cover
        inputs = ctx.inputs
        inputs_leaf = tuple(x.detach().requires_grad_(x.requires_grad) if torch.is_tensor(x) else x for x in inputs)

        # Get the device for the inputs from a tensor
        device = None
        for input in inputs:
            if torch.is_tensor(input):
                device = input.device
                break

        if device is None:
            raise RuntimeError(f'No tensors found in {inputs}')

        with restore_rng_states(device, ctx.rng_states):
            with torch.enable_grad(), enable_recomputing():
                if ctx.input_atomic:
                    assert len(inputs_leaf) == 1
                    output = ctx.function(inputs_leaf[0])
                else:
                    output = ctx.function(*inputs_leaf)

        ctx.recomputed.append((output, inputs_leaf))

        grad_input: List[None] = [None, None, None, None, None]
        grad_input.extend(None for _ in ctx.inputs)
        return tuple(grad_input)
