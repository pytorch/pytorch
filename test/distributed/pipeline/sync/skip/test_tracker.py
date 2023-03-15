# Owner(s): ["oncall: distributed"]

# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from queue import Queue
import threading

import pytest
import torch
from torch import nn

from torch.distributed.pipeline.sync.checkpoint import enable_checkpointing, enable_recomputing
from torch.distributed.pipeline.sync.microbatch import Batch
from torch.distributed.pipeline.sync.skip import pop, skippable, stash
from torch.distributed.pipeline.sync.skip.layout import SkipLayout
from torch.distributed.pipeline.sync.skip.tracker import SkipTracker, SkipTrackerThroughPotals, current_skip_tracker
from torch.testing._internal.common_utils import run_tests


def test_default_skip_tracker():
    q = Queue()

    def f():
        q.put(current_skip_tracker())

    t = threading.Thread(target=f)
    t.start()
    t.join()

    skip_tracker = q.get()

    assert type(skip_tracker) is SkipTracker
    assert type(skip_tracker) is not SkipTrackerThroughPotals


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_default_skip_tracker_by_data_parallel():
    @skippable(stash=["foo"])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("foo", input)
            return input * 2  # noqa: B901

    @skippable(pop=["foo"])
    class Pop(nn.Module):
        def forward(self, input):
            foo = yield pop("foo")
            return foo

    model = nn.Sequential(Stash(), Pop())
    model = nn.DataParallel(model, device_ids=[0, 0], output_device=0)

    input = torch.rand(10, device=0)
    output = model(input)

    assert torch.allclose(output, input)


def test_reuse_portal():
    skip_layout = SkipLayout(num_partitions=2, skip_routes={(None, "test"): (0, 1)})
    skip_tracker = SkipTrackerThroughPotals(skip_layout)

    batch = Batch(torch.tensor([1.0]))
    a = torch.tensor([2.0])
    b = torch.tensor([2.0])

    skip_tracker.save(batch, None, "test", a)
    portal = skip_tracker.portals[(None, "test")]

    skip_tracker.save(batch, None, "test", b)
    assert portal is skip_tracker.portals[(None, "test")]


def test_no_copy_no_portal():
    skip_layout = SkipLayout(num_partitions=2, skip_routes={(None, "copy"): (0, 1), (None, "not_copy"): (0, 0)})
    skip_tracker = SkipTrackerThroughPotals(skip_layout)

    batch = Batch(torch.tensor([1.0]))
    a = torch.tensor([2.0])
    b = torch.tensor([2.0])

    skip_tracker.save(batch, None, "copy", a)
    skip_tracker.save(batch, None, "not_copy", b)

    assert (None, "copy") in skip_tracker.portals
    assert (None, "copy") not in skip_tracker.tensors
    assert (None, "not_copy") in skip_tracker.tensors
    assert (None, "not_copy") not in skip_tracker.portals


def test_tensor_life_without_checkpointing():
    skip_layout = SkipLayout(num_partitions=2, skip_routes={(None, "test"): (0, 1)})
    skip_tracker = SkipTrackerThroughPotals(skip_layout)

    batch = Batch(torch.tensor([1.0]))
    tensor = torch.tensor([2.0])

    skip_tracker.save(batch, None, "test", tensor)
    assert skip_tracker.portals[(None, "test")].tensor_life == 1

    skip_tracker.load(batch, None, "test")
    assert skip_tracker.portals[(None, "test")].tensor_life == 0


def test_tensor_life_with_checkpointing():
    skip_layout = SkipLayout(num_partitions=2, skip_routes={(None, "test"): (0, 1)})
    skip_tracker = SkipTrackerThroughPotals(skip_layout)

    batch = Batch(torch.tensor([1.0]))
    tensor = torch.tensor([2.0])

    with enable_checkpointing():
        skip_tracker.save(batch, None, "test", tensor)
    assert skip_tracker.portals[(None, "test")].tensor_life == 2

    with enable_checkpointing():
        skip_tracker.load(batch, None, "test")
    assert skip_tracker.portals[(None, "test")].tensor_life == 1

    with enable_recomputing():
        skip_tracker.load(batch, None, "test")
    assert skip_tracker.portals[(None, "test")].tensor_life == 0

    with enable_recomputing():
        skip_tracker.save(batch, None, "test", tensor)
    assert skip_tracker.portals[(None, "test")].tensor_life == 0


if __name__ == "__main__":
    run_tests()
