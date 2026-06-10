# Copyright (c) Meta Platforms, Inc. and affiliates.
# Owner(s): ["oncall: distributed"]
# pyre-unsafe

"""Integration tests for dist.batch_isend_irecv routed through BackendWrapper.

Pipeline parallel uses dist.batch_isend_irecv to issue mixed (send + recv)
batches on a single PG. Without coalescing support in BackendWrapper, each
ungrouped tc.send/tc.recv is enqueued back-to-back on the same NCCL stream
and bidirectional patterns deadlock. The tests below exercise exactly that
pattern to guard the BackendWrapper coalescing path.
"""

import os
import unittest

from integration.helpers.TorchCommTestHelpers import get_device, get_rank_and_size

import torch
import torch.distributed as dist


class TestC10dBackendWrapperBatchIsendIrecv(unittest.TestCase):
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

    def setUp(self):
        if dist.get_world_size() < 2:
            self.skipTest("batch_isend_irecv tests require world_size >= 2")
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.next = (self.rank + 1) % self.world
        self.prev = (self.rank - 1 + self.world) % self.world

    def test_supports_coalescing(self):
        """BackendWrapper must advertise coalescing for batch_isend_irecv to
        use the grouped fast path instead of the per-op fallback."""
        device = get_device(os.environ["TEST_BACKEND"], self.rank)
        backend = dist.group.WORLD._get_backend(device)
        self.assertTrue(backend.supports_coalescing)

    def test_batch_isend_irecv_mixed_ring(self):
        """Each rank concurrently sends to next and receives from prev — the
        canonical PP 1F1B middle-stage pattern. Without ncclGroupStart/End
        coalescing this deadlocks; with it, every rank gets the prev rank's
        value back."""
        send_tensor = torch.tensor([self.rank], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)

        ops = [
            dist.P2POp(dist.isend, send_tensor, self.next),
            dist.P2POp(dist.irecv, recv_tensor, self.prev),
        ]
        works = dist.batch_isend_irecv(ops)
        for w in works:
            w.wait()

        self.assertEqual(recv_tensor.item(), float(self.prev))

    def test_batch_isend_irecv_recv_first(self):
        """Same as above but with the recv listed before the send. Order
        within a batch shouldn't matter once the ops are coalesced into a
        single ncclGroup."""
        send_tensor = torch.tensor([self.rank * 10 + 1], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)

        ops = [
            dist.P2POp(dist.irecv, recv_tensor, self.prev),
            dist.P2POp(dist.isend, send_tensor, self.next),
        ]
        works = dist.batch_isend_irecv(ops)
        for w in works:
            w.wait()

        self.assertEqual(recv_tensor.item(), float(self.prev * 10 + 1))

    def test_batch_isend_irecv_multiple_ops_per_peer(self):
        """Multiple sends and recvs to/from the same peers in one batch —
        exercises the batch with >2 ops, the realistic PP middle-stage shape
        when activations and gradients overlap on the same neighbor."""
        send1 = torch.tensor([self.rank], dtype=torch.float32)
        send2 = torch.tensor([self.rank + 100], dtype=torch.float32)
        recv1 = torch.empty(1, dtype=torch.float32)
        recv2 = torch.empty(1, dtype=torch.float32)

        ops = [
            dist.P2POp(dist.isend, send1, self.next),
            dist.P2POp(dist.irecv, recv1, self.prev),
            dist.P2POp(dist.isend, send2, self.next),
            dist.P2POp(dist.irecv, recv2, self.prev),
        ]
        works = dist.batch_isend_irecv(ops)
        for w in works:
            w.wait()

        self.assertEqual(recv1.item(), float(self.prev))
        self.assertEqual(recv2.item(), float(self.prev + 100))

    def test_individual_isend_irecv_outside_coalescing(self):
        """Verify the non-coalesced path still works after coalescing was
        used — coalescing_batch_ must be cleared on endCoalescing."""
        # First do a coalesced batch.
        send_tensor = torch.tensor([self.rank], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)
        ops = [
            dist.P2POp(dist.isend, send_tensor, self.next),
            dist.P2POp(dist.irecv, recv_tensor, self.prev),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()

        # Then a separate ordinary send/recv ring — must not see leftover
        # coalescing state.
        send2 = torch.tensor([self.rank + 1000], dtype=torch.float32)
        recv2 = torch.empty(1, dtype=torch.float32)
        if self.rank % 2 == 0:
            dist.send(send2, dst=self.next)
            dist.recv(recv2, src=self.prev)
        else:
            dist.recv(recv2, src=self.prev)
            dist.send(send2, dst=self.next)
        self.assertEqual(recv2.item(), float(self.prev + 1000))


if __name__ == "__main__":
    unittest.main()
