# -*- coding: utf-8 -*-
# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""The pipeline parallelism of Pipe."""
from queue import Queue
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Type, Union, cast, Sequence

import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function

from .checkpoint import Checkpointing
from .copy import Copy, Wait
from .dependency import fork, join
from .microbatch import Batch
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .stream import AbstractStream, current_stream, use_device
from .worker import Task, create_workers

__all__: List[str] = []


Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]

# Queue is generic only in stubs.
# https://mypy.readthedocs.io/en/latest/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    InQueue = Queue[Optional["Task"]]
    OutQueue = Queue[Tuple[bool, Union[Tuple["Task", Batch], ExcInfo, None]]]
else:
    InQueue = Queue
    OutQueue = Queue


def _depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from[0], phony = fork(fork_from[0])
    join_to[0] = join(join_to[0], phony)


def _copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x if x.is_floating_point() else x.detach() for x in batch])


def _wait(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)
    # Gradients are only supported for float Tensors.
    batch[:] = tuple([x if x.is_floating_point() else x.detach() for x in batch])


def _clock_cycles(m: int, n: int) -> Iterable[List[Tuple[int, int]]]:
    """Generates schedules for each clock cycle."""
    # m: number of micro-batches
    # n: number of partitions
    # i: index of micro-batch
    # j: index of partition
    # k: clock number
    #
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1) (0,2)
    # 3       (2,1) (1,2)
    # 4             (2,2)
    for k in range(m + n - 1):
        yield [(k - j, j) for j in range(max(1 + k - m, 0), min(1 + k, n))]


class Pipeline:
    """The pipeline parallelism for Pipe."""

    def __init__(
        self,
        partitions: List[nn.Sequential],
        devices: List[torch.device],
        copy_streams: List[List[AbstractStream]],
        skip_layout: SkipLayout,
        checkpoint_stop: int,
    ) -> None:
        self.partitions = partitions
        self.devices = devices
        self.copy_streams = copy_streams
        self.skip_layout = skip_layout
        self.checkpoint_stop = checkpoint_stop
        (self.in_queues, self.out_queues) = create_workers(devices)

    def run(self, batches: List[Batch]) -> None:
        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """
        partitions = self.partitions
        devices = self.devices
        skip_layout = self.skip_layout

        m = len(batches)
        n = len(partitions)

        skip_trackers = [SkipTrackerThroughPotals(skip_layout) for _ in batches]

        for schedule in _clock_cycles(m, n):
            self.fence(batches, schedule, skip_trackers)
            self.compute(batches, schedule, skip_trackers)

    def fence(
        self, batches: List[Batch], schedule: List[Tuple[int, int]], skip_trackers: List[SkipTrackerThroughPotals],
    ) -> None:
        """Copies micro-batches after computation for the previous
        micro-batches.
        """
        copy_streams = self.copy_streams
        skip_layout = self.skip_layout

        for i, j in schedule:
            # Ensure that batches[i-1] is executed after batches[i] in
            # backpropagation by an explicit dependency.
            if i != 0 and j != 0:
                _depend(batches[i - 1], batches[i])

            next_stream = copy_streams[j][i]

            for prev_j, ns, name in skip_layout.copy_policy(j):
                prev_stream = copy_streams[prev_j][i]
                skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

            if j != 0:
                prev_stream = copy_streams[j - 1][i]
                _copy(batches[i], prev_stream, next_stream)

    def compute(
        self, batches: List[Batch], schedule: List[Tuple[int, int]], skip_trackers: List[SkipTrackerThroughPotals],
    ) -> None:
        """Runs tasks with synchronization to copy streams."""
        partitions = self.partitions
        devices = self.devices
        copy_streams = self.copy_streams
        checkpoint_stop = self.checkpoint_stop

        # Disable checkpointing if in eval mode.
        if not self.partitions[0].training:
            checkpoint_stop = 0

        n = len(partitions)
        streams = [current_stream(d) for d in devices]
        exc_info: Optional[ExcInfo] = None

        # With checkpointing, the autograd graph looks like this diagram:
        # ┌─────┸──────┐
        # │    Copy    │
        # └─────┰──────┘   (fence)
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        #       ┃          (compute)
        # ┌─────┸──────┐
        # │    Wait    │ [1] Synchronize the current stream with the copy stream.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │ Checkpoint │ [2] Compute a partition within checkpointing.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │    Wait    │ [3] Synchronize the copy stream with the current stream.
        # └─────┰──────┘
        #       ┠ ─ ─ ─ ┐
        #       ┃ ┌─────┴─────┐
        #       ┃ │ Recompute │ [4] Schedule the recomputation at backpropagation.
        #       ┃ └─────┬─────┘
        #       ┠ ─ ─ ─ ┘
        #       ┃
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # ┌─────┸──────┐   (fence)
        # │    Copy    │
        # └─────┰──────┘
        for i, j in schedule:
            batch = batches[i]
            partition = partitions[j]

            # Synchronize with the copied input. ([1] in the diagram)
            if j != 0:
                _wait(batch, copy_streams[j][i], streams[j])

            # Determine whether checkpointing or not.
            checkpoint = i < checkpoint_stop
            if checkpoint:

                def function(
                    input: TensorOrTensors,
                    partition: nn.Sequential = partition,
                    skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                    chunk_id: int = i,
                    part_id: int = j,
                ) -> TensorOrTensors:
                    with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                        return partition(input)

                chk = Checkpointing(function, batch)
                task = Task(streams[j], compute=chk.checkpoint, finalize=chk.recompute)
                del function, chk

            else:

                def compute(
                    batch: Batch = batch,
                    partition: nn.Sequential = partition,
                    skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                    chunk_id: int = i,
                    part_id: int = j,
                ) -> Batch:
                    with use_skip_tracker(skip_tracker), record_function("chunk%d-part%d" % (chunk_id, part_id)):
                        return batch.call(partition)

                task = Task(streams[j], compute=compute, finalize=None)
                del compute

            # Compute tasks in parallel. ([2] in the diagram)
            self.in_queues[j].put(task)

        for i, j in schedule:
            ok, payload = self.out_queues[j].get()

            # Hold the first exception.
            if exc_info is not None:
                continue
            elif not ok:
                exc_info = cast(ExcInfo, payload)
                continue

            task, batch = cast(Tuple[Task, Batch], payload)

            # The copy stream synchronizes to copy the output. ([3] in the
            # diagram)
            if j != n - 1:
                _wait(batch, streams[j], copy_streams[j][i])

            # Finalize tasks. If checkpointing is enabled, here the
            # recomputation is scheduled at backpropagation. ([4] in the
            # diagram)
            with use_device(devices[j]):
                task.finalize(batch)

            batches[i] = batch

        # Fail at the first exception.
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
