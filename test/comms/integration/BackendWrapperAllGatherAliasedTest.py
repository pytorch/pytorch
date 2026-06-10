#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Verify ``BackendWrapper::allgather`` handles the legacy c10d
list-of-tensors API in both the fast (distinct outputs) and slow (aliased
outputs) paths.

Aliased outputs come up in real callers (e.g. vLLM EPLB profiling passes
``[buffer] * world_size``) — the wrapper must not race writes from N ranks
into a single buffer; it must allocate a contiguous staging tensor and
copy each rank's slice back.
"""

import os
import unittest

from integration.helpers.TorchCommTestHelpers import get_device, get_rank_and_size

import torch
import torch.distributed as dist


class TestBackendWrapperAllGatherAliased(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.config.use_torchcomms = True
        rank, world_size = get_rank_and_size()
        dist.init_process_group(
            backend=os.environ["TEST_BACKEND"], rank=rank, world_size=world_size
        )
        device = get_device(os.environ["TEST_BACKEND"], dist.get_rank())
        torch.set_default_device(device)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def test_distinct_outputs_fast_path(self):
        """When per-rank output tensors are distinct buffers, the gather
        result is the per-rank inputs in rank order."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        input_tensor = torch.tensor([float(rank)], dtype=torch.float32)
        output_list = [torch.empty(1, dtype=torch.float32) for _ in range(world_size)]

        dist.all_gather(output_list, input_tensor)

        for r in range(world_size):
            self.assertEqual(
                output_list[r].item(),
                float(r),
                f"slot {r}: expected {float(r)}, got {output_list[r].item()}",
            )

    def test_aliased_outputs_no_crash(self):
        """When per-rank outputs all alias the same buffer (``[buf]*N``),
        the wrapper must not race writes into the shared buffer. The slow
        path stages into a temporary tensor and copies each slice back; the
        last rank's value wins (matches stock ``ProcessGroupNCCL``)."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        input_tensor = torch.tensor([float(rank)], dtype=torch.float32)
        shared = torch.full((1,), -1.0, dtype=torch.float32)
        output_list = [shared for _ in range(world_size)]

        dist.all_gather(output_list, input_tensor)

        # Buffer was written by some rank, not the sentinel anymore.
        self.assertNotEqual(
            shared.item(),
            -1.0,
            "shared aliased buffer was never written — gather skipped",
        )
        # The slow path copies in rank order, so the last rank wins.
        self.assertEqual(
            shared.item(),
            float(world_size - 1),
            f"expected last-rank-wins value {float(world_size - 1)}, "
            f"got {shared.item()}",
        )

    def test_aliased_then_distinct_does_not_pollute(self):
        """Calling allgather once with aliased outputs then again with
        distinct outputs must produce correct results on the second call —
        the wrapper's staging buffer must not leak into the distinct path."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Aliased call first (slow path).
        shared = torch.zeros(1, dtype=torch.float32)
        dist.all_gather(
            [shared for _ in range(world_size)],
            torch.tensor([float(rank)], dtype=torch.float32),
        )

        # Distinct call (fast path) — must not be polluted.
        output_list = [torch.empty(1, dtype=torch.float32) for _ in range(world_size)]
        dist.all_gather(
            output_list,
            torch.tensor([float(rank) + 100.0], dtype=torch.float32),
        )

        for r in range(world_size):
            self.assertEqual(output_list[r].item(), float(r) + 100.0)


if __name__ == "__main__":
    unittest.main()
