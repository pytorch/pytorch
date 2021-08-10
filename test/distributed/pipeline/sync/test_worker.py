# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import threading

import pytest
import torch

from torch.distributed.pipeline.sync.microbatch import Batch
from torch.distributed.pipeline.sync.stream import CPUStream
from torch.distributed.pipeline.sync.worker import Task, spawn_workers


class fake_device:
    """A test double for :class:`torch.device`. Every fake device is different
    with each other.
    """

    type = "fake"
    index = None

def test_compute_multithreading():
    """Task.compute should be executed on multiple threads."""
    thread_ids = set()

    def log_thread_id():
        thread_id = threading.current_thread().ident
        thread_ids.add(thread_id)
        return Batch(())

    with spawn_workers([fake_device() for _ in range(2)]) as (in_queues, out_queues):
        for i in range(2):
            t = Task(CPUStream, compute=log_thread_id, finalize=None)
            in_queues[i].put(t)
        for i in range(2):
            out_queues[i].get()

    assert len(thread_ids) == 2


def test_compute_success():
    """Task.compute returns (True, (task, batch)) on success."""

    def _42():
        return Batch(torch.tensor(42))

    with spawn_workers([torch.device("cpu")]) as (in_queues, out_queues):
        t = Task(CPUStream, compute=_42, finalize=None)
        in_queues[0].put(t)
        ok, (task, batch) = out_queues[0].get()

        assert ok
        assert task is t
        assert isinstance(batch, Batch)
        assert batch[0].item() == 42


def test_compute_exception():
    """Task.compute returns (False, exc_info) on failure."""

    def zero_div():
        0 / 0

    with spawn_workers([torch.device("cpu")]) as (in_queues, out_queues):
        t = Task(CPUStream, compute=zero_div, finalize=None)
        in_queues[0].put(t)
        ok, exc_info = out_queues[0].get()

        assert not ok
        assert isinstance(exc_info, tuple)
        assert issubclass(exc_info[0], ZeroDivisionError)


@pytest.mark.parametrize("grad_mode", [True, False])
def test_grad_mode(grad_mode):
    def detect_grad_enabled():
        x = torch.rand(1, requires_grad=torch.is_grad_enabled())
        return Batch(x)

    with torch.set_grad_enabled(grad_mode):
        with spawn_workers([torch.device("cpu")]) as (in_queues, out_queues):
            task = Task(CPUStream, compute=detect_grad_enabled, finalize=None)
            in_queues[0].put(task)

            ok, (_, batch) = out_queues[0].get()

            assert ok
            assert batch[0].requires_grad == grad_mode


def test_worker_per_device():
    cpu = torch.device("cpu")
    cpu0 = torch.device("cpu", index=0)
    fake1 = fake_device()
    fake2 = fake_device()

    with spawn_workers([cpu, cpu, cpu0, fake1, fake2]) as (in_queues, out_queues):
        assert len(in_queues) == len(out_queues) == 5

        # 0: cpu, 1: cpu, 2: cpu0
        assert in_queues[0] is in_queues[1] is in_queues[2]
        assert out_queues[0] is out_queues[1] is out_queues[2]

        # 3: fake1, 4: fake2
        assert in_queues[3] is not in_queues[4]
        assert out_queues[3] is not out_queues[4]
