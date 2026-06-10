#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Verify ``BackendWrapper`` implements c10d's coalescing hooks so
``dist.batch_isend_irecv`` issues each batch as one ``ncclGroupStart``/
``ncclGroupEnd`` pair via the underlying ``TorchCommBatch`` instead of
running each P2POp ungrouped (which can deadlock for mixed send/recv
batches like the PP 1F1B middle stage).
"""

import os
import unittest

from integration.helpers.TorchCommTestHelpers import get_device, get_rank_and_size

import torch
import torch.distributed as dist


class TestBackendWrapperCoalescing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.config.use_torchcomms = True
        rank, world_size = get_rank_and_size()
        dist.init_process_group(
            backend=os.environ["TEST_BACKEND"], rank=rank, world_size=world_size
        )
        device = get_device(os.environ["TEST_BACKEND"], dist.get_rank())
        torch.set_default_device(device)
        cls.device = device

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def test_supports_coalescing_is_true(self):
        """``supportsCoalescing`` is overridden to ``True`` so c10d's
        ``_coalescing_manager`` (used by ``batch_isend_irecv``) calls
        ``startCoalescing`` / ``endCoalescing`` instead of issuing each
        P2POp individually."""
        pg = dist.distributed_c10d._get_default_group()
        backend = pg._get_backend(self.device)
        self.assertTrue(
            backend.supports_coalescing,
            "BackendWrapper.supports_coalescing must be True so "
            "batch_isend_irecv takes the coalescing path",
        )

    def test_batch_isend_irecv_mixed_send_recv(self):
        """A mixed isend+irecv batch in a single ``batch_isend_irecv``
        call delivers correctly. Without coalescing this pattern (mirrors
        PP 1F1B middle stage) deadlocks because each tc.send / tc.recv is
        enqueued ungrouped on the same NCCL stream."""
        world_size = dist.get_world_size()
        if world_size < 2:
            self.skipTest("need at least 2 ranks for batch_isend_irecv")

        rank = dist.get_rank()
        peer = (rank + 1) % world_size
        recv_peer = (rank - 1) % world_size
        send_tensor = torch.full((4,), float(rank), dtype=torch.float32)
        recv_tensor = torch.empty(4, dtype=torch.float32)

        ops = [
            dist.P2POp(dist.isend, send_tensor, peer),
            dist.P2POp(dist.irecv, recv_tensor, recv_peer),
        ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        expected = torch.full((4,), float(recv_peer), dtype=torch.float32)
        self.assertTrue(
            torch.equal(recv_tensor, expected),
            f"recv mismatch: got {recv_tensor.tolist()}, expected {expected.tolist()}",
        )

    def test_batch_isend_irecv_multiple_peers(self):
        """A batch of N sends + N recvs across multiple peers in a single
        coalesced batch — exercises the ncclGroupStart/End grouping over
        more than one P2P pair."""
        world_size = dist.get_world_size()
        if world_size < 3:
            self.skipTest("need at least 3 ranks for multi-peer batch")

        rank = dist.get_rank()
        # Send to (rank+1) and (rank+2), receive from (rank-1) and (rank-2).
        send_tensors = [
            torch.full((4,), float(rank * 10 + i), dtype=torch.float32)
            for i in range(2)
        ]
        recv_tensors = [torch.empty(4, dtype=torch.float32) for _ in range(2)]
        send_peers = [(rank + 1) % world_size, (rank + 2) % world_size]
        recv_peers = [(rank - 1) % world_size, (rank - 2) % world_size]

        ops = []
        for i in range(2):
            ops.append(dist.P2POp(dist.isend, send_tensors[i], send_peers[i]))
            ops.append(dist.P2POp(dist.irecv, recv_tensors[i], recv_peers[i]))

        for req in dist.batch_isend_irecv(ops):
            req.wait()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        for i in range(2):
            expected_value = float(recv_peers[i] * 10 + i)
            expected = torch.full((4,), expected_value, dtype=torch.float32)
            self.assertTrue(
                torch.equal(recv_tensors[i], expected),
                f"slot {i} (from rank {recv_peers[i]}): got "
                f"{recv_tensors[i].tolist()}, expected {expected.tolist()}",
            )


if __name__ == "__main__":
    unittest.main()
