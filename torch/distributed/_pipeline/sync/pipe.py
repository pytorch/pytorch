# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""The Pipe interface."""
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda

from . import microbatch
from .batchnorm import DeferredBatchNorm
from .pipeline import Pipeline
from .skip.layout import inspect_skip_layout
from .skip.skippable import verify_skippables
from .stream import AbstractStream, new_stream

__all__ = ["Pipe"]


Device = Union[torch.device, int, str]
Devices = Union[Iterable[Device], List[Device]]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

if TYPE_CHECKING:
    Module = nn.Module[TensorOrTensors]
    NamedModules = OrderedDict[str, Module]
else:
    Module = nn.Module
    NamedModules = OrderedDict


def recommend_auto_balance(message: str) -> str:
    """Expands a message with recommendation to :mod:`torchpipe.balance`."""
    return f"""{message}

If your model is still under development, its optimal balance would change
frequently. In this case, we highly recommend 'torch.distributed._pipeline.sync.balance' for
naive automatic balancing:

  from torch.distributed._pipeline.sync import Pipe
  from torch.distributed._pipeline.sync.balance import balance_by_time

  partitions = torch.cuda.device_count()
  sample = torch.empty(...)
  balance = balance_by_time(partitions, model, sample)

  model = Pipe(model, balance, ...)
"""


def verify_module(module: nn.Sequential) -> None:
    if not isinstance(module, nn.Sequential):
        raise TypeError("module must be nn.Sequential to be partitioned")

    named_children = list(module.named_children())
    if len(named_children) != len(module):
        raise ValueError("module with duplicate children is not supported")


def verify_splitting(
    module: nn.Sequential, partitions: List[nn.Sequential], balance: Iterable[int], devices: List[torch.device]
) -> None:
    num_parameters = len(list(module.parameters()))
    num_child_parameters = sum(len(list(child.parameters())) for child in module.children())
    if num_parameters == num_child_parameters:
        return

    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            parti = partitions[i]
            partj = partitions[j]
            if devices[i] == devices[j]:
                continue
            for p in parti.parameters():
                for q in partj.parameters():
                    if p is q:
                        raise ValueError("module with duplicate parameters on distinct devices is not supported")


class BalanceError(ValueError):
    pass


def split_module(
    module: nn.Sequential, balance: Iterable[int], devices: List[torch.device],
) -> Tuple[List[nn.Sequential], List[int], List[torch.device]]:
    """Splits a module into multiple partitions.

    Returns:
        A tuple of (partitions, balance, devices).

        Partitions are represented as a :class:`~torch.nn.ModuleList` whose
        item is a partition. All layers in a partition are placed in the
        same device.

    Raises:
        BalanceError:
            wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """
    balance = list(balance)

    if len(module) != sum(balance):
        raise BalanceError(
            "module and sum of balance have different length "
            f"(module: {len(module)}, sum of balance: {sum(balance)})"
        )

    if any(x <= 0 for x in balance):
        raise BalanceError(f"all balance numbers must be positive integer (balance: {balance})")

    if len(balance) > len(devices):
        raise IndexError(
            "too few devices to hold given partitions " f"(devices: {len(devices)}, partitions: {len(balance)})"
        )

    j = 0
    partitions = []
    layers: NamedModules = OrderedDict()

    for name, layer in module.named_children():
        layers[name] = layer

        if len(layers) == balance[j]:
            # Group buffered layers as a partition.
            partition = nn.Sequential(layers)

            device = devices[j]
            partition.to(device)

            partitions.append(partition)

            # Prepare for the next partition.
            layers.clear()
            j += 1

    partitions = cast(List[nn.Sequential], nn.ModuleList(partitions))
    del devices[j:]

    return partitions, balance, devices


MOVING_DENIED = TypeError("denied to move parameters and buffers, " "because Pipe should manage device placement")


class Pipe(Module):
    """Wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    to train on Pipe_. If the module requires lots of memory, Pipe will be
    very efficient.
    ::

        model = nn.Sequential(a, b, c, d)
        model = Pipe(model, balance=[1, 1, 1, 1], chunks=8)
        output = model(input)

    .. _Pipe: https://arxiv.org/abs/1811.06965

    Pipe combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should determine the balance when defining a :class:`Pipe` module, as
    balancing will not be done automatically. The module will be partitioned
    into multiple devices according to the given balance. You may rely on
    heuristics to find your own optimal configuration.

    Args:
        module (torch.nn.Sequential):
            sequential module to be parallelized
        balance (ints):
            list of number of layers in each partition

    Keyword Args:
        devices (iterable of devices):
            devices to use (default: all CUDA devices)
        chunks (int):
            number of micro-batches (default: ``1``)
        checkpoint (str):
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``)
        deferred_batch_norm (bool):
            whether to use deferred BatchNorm moving statistics (default:
            :data:`False`, see :ref:`Deferred Batch Normalization` for more
            details)

    Raises:
        TypeError:
            the module is not a :class:`nn.Sequential <torch.nn.Sequential>`.
        ValueError:
            invalid arguments, or wrong balance
        IndexError:
            the number of devices is fewer than the number of partitions.

    """

    #: The number of layers in each partition.
    balance: List[int] = []
    #                    ^^
    # The default value [] required for Sphinx's autoattribute.

    #: The devices mapped to each partition.
    #:
    #: ``devices[-1]`` refers to the device of the last partition, which means
    #: it is the output device. Probably, you need to use it to transfer the
    #: target to calculate the loss without a device mismatch
    #: :exc:`RuntimeError`. For example::
    #:
    #:     out_device = pipe.devices[-1]
    #:
    #:     for input, target in loader:
    #:         target = target.to(out_device, non_blocking=True)
    #:         output = pipe(input)
    #:         loss = F.cross_entropy(output, target)
    #:
    devices: List[torch.device] = []

    #: The number of micro-batches.
    chunks: int = 1

    #: The checkpoint mode to determine when to enable checkpointing. It is one
    #: of ``'always'``, ``'except_last'``, or ``'never'``.
    checkpoint: str = "except_last"

    def __init__(
        self,
        module: nn.Sequential,
        balance: Optional[Iterable[int]] = None,
        *,
        devices: Optional[Devices] = None,
        chunks: int = chunks,
        checkpoint: str = checkpoint,
        deferred_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        chunks = int(chunks)
        checkpoint = str(checkpoint)

        if balance is None:
            raise ValueError(recommend_auto_balance("balance is required"))
        if chunks <= 0:
            raise ValueError("number of chunks must be positive integer")
        if checkpoint not in ["always", "except_last", "never"]:
            raise ValueError("checkpoint is not one of 'always', 'except_last', or 'never'")

        verify_module(module)

        # Verify if the underlying skippable modules satisfy integrity. The
        # integrity can be verified before forward() because it is static.
        verify_skippables(module)

        self.chunks = chunks
        self.checkpoint = checkpoint

        if deferred_batch_norm:
            module = DeferredBatchNorm.convert_deferred_batch_norm(module, chunks)

        if devices is None:
            devices = range(torch.cuda.device_count())
        devices = [torch.device(d) for d in devices]
        devices = cast(List[torch.device], devices)

        try:
            self.partitions, self.balance, self.devices = split_module(module, balance, devices)
        except BalanceError as exc:
            raise ValueError(recommend_auto_balance(str(exc)))

        verify_splitting(module, self.partitions, self.balance, self.devices)

        self._copy_streams: List[List[AbstractStream]] = []
        self._skip_layout = inspect_skip_layout(self.partitions)

        # Separate CUDA streams for copy.
        copy_streams = self._ensure_copy_streams()

        # The micro-batch index where the checkpointing stops.
        checkpoint_stop = {"always": self.chunks, "except_last": self.chunks - 1, "never": 0}[self.checkpoint]

        self.pipeline = Pipeline(self.partitions, self.devices, copy_streams, self._skip_layout, checkpoint_stop)

    def __len__(self) -> int:
        """Counts the length of the underlying sequential module."""
        return sum(len(p) for p in self.partitions)

    def __getitem__(self, index: int) -> nn.Module:
        """Gets a layer in the underlying sequential module."""
        partitions = self.partitions
        if index < 0:
            partitions = partitions[::-1]

        for partition in partitions:
            try:
                return partition[index]
            except IndexError:
                pass

            shift = len(partition)

            if index < 0:
                index += shift
            else:
                index -= shift

        raise IndexError

    def __iter__(self) -> Iterable[nn.Module]:
        """Iterates over children of the underlying sequential module."""
        for partition in self.partitions:
            yield from partition

    # Pipe should manage the device of each partition.
    # Deny cuda(), cpu(), and to() with device, by TypeError.
    def cuda(self, device: Optional[Device] = None) -> "Pipe":
        raise MOVING_DENIED

    def cpu(self) -> "Pipe":
        raise MOVING_DENIED

    def to(self, *args: Any, **kwargs: Any) -> "Pipe":
        # Deny these usages:
        #
        # - to(device[, dtype, non_blocking])
        # - to(tensor[, non_blocking])
        #
        # But allow this:
        #
        # - to(dtype[, non_blocking])
        #
        if "device" in kwargs or "tensor" in kwargs:
            raise MOVING_DENIED

        if args:
            if isinstance(args[0], (torch.device, int, str)):
                raise MOVING_DENIED
            if torch.is_tensor(args[0]):
                raise MOVING_DENIED

        return super().to(*args, **kwargs)

    def _ensure_copy_streams(self) -> List[List[AbstractStream]]:
        """Ensures that :class:`Pipe` caches CUDA streams for copy.

        It's worth to cache CUDA streams although PyTorch already manages a
        pool of pre-allocated CUDA streams, because it may reduce GPU memory
        fragementation when the number of micro-batches is small.

        """
        if not self._copy_streams:
            for device in self.devices:
                self._copy_streams.append([new_stream(device) for _ in range(self.chunks)])

        return self._copy_streams

    def forward(self, input: TensorOrTensors) -> TensorOrTensors:  # type: ignore
        """:class:`Pipe` is a fairly transparent module wrapper. It doesn't
        modify the input and output signature of the underlying module. But
        there's type restriction. Input and output have to be a
        :class:`~torch.Tensor` or a tuple of tensors. This restriction is
        applied at partition boundaries too.

        Args:
            input (torch.Tensor or tensors): input mini-batch

        Returns:
            tensor or tensors: output mini-batch

        Raises:
            TypeError: input is not a tensor or tensors.

        """
        microbatch.check(input)

        if not self.devices:
            # Empty sequential module is not illegal.
            return input

        # Divide a mini-batch into micro-batches.
        batches = microbatch.scatter(input, self.chunks)

        # Run pipeline parallelism.
        self.pipeline.run(batches)

        # Merge the micro-batches into one mini-batch.
        output = microbatch.gather(batches)
        return output
