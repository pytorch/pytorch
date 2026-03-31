# Owner(s): ["oncall: distributed"]

import unittest

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _TORCHCOMM_AVAILABLE
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import C10dTorchCommsTestBase
from torch.testing._internal.common_utils import parametrize, run_tests, subtest


@unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
class TestC10dTorchCommsBasic(C10dTorchCommsTestBase):
    REDUCE_OPS = [
        subtest(dist.ReduceOp.SUM, name="SUM"),
        subtest(dist.ReduceOp.AVG, name="AVG"),
        subtest(dist.ReduceOp.MIN, name="MIN"),
        subtest(dist.ReduceOp.MAX, name="MAX"),
    ]

    def _expected_reduce_result(self, op):
        """Return the expected scalar result for a rank+1 input reduced across all ranks."""
        total = sum(range(1, self.world_size + 1))
        if op == dist.ReduceOp.SUM:
            return total
        elif op == dist.ReduceOp.AVG:
            return total / self.world_size
        elif op == dist.ReduceOp.MIN:
            return 1
        elif op == dist.ReduceOp.MAX:
            return self.world_size
        raise ValueError(f"Unsupported op: {op}")

    @parametrize("op", REDUCE_OPS)
    def test_allreduce(self, op):
        tensor = torch.tensor([self.rank + 1], dtype=torch.float32)
        dist.all_reduce(tensor, op=op, group=self.pg)
        self.assertEqual(tensor.item(), self._expected_reduce_result(op))

    def test_all_gather(self):
        input_tensor = torch.tensor([self.rank], dtype=torch.float32)
        gather_list = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, input_tensor, group=self.pg)
        expected = list(range(self.world_size))
        self.assertEqual([t.item() for t in gather_list], expected)

    def test_all_gather_into_tensor(self):
        input_tensor = torch.tensor([self.rank], dtype=torch.float32)
        output_tensor = torch.empty(self.world_size, dtype=torch.float32)
        dist.all_gather_into_tensor(output_tensor, input_tensor, group=self.pg)
        expected = list(range(self.world_size))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_broadcast(self):
        tensor = torch.tensor([self.rank + 1], dtype=torch.float32)
        dist.broadcast(tensor, src=0, group=self.pg)
        self.assertEqual(tensor.item(), 1)

    def test_gather(self):
        tensor = torch.tensor([self.rank], dtype=torch.float32)
        gather_list = None
        if self.rank == 0:
            gather_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.gather(tensor, gather_list=gather_list, dst=0, group=self.pg)
        if self.rank == 0:
            expected = list(range(self.world_size))
            self.assertEqual([t.item() for t in gather_list], expected)

    def test_scatter(self):
        if self.rank == 0:
            scatter_list = [
                torch.tensor([i], dtype=torch.float32) for i in range(self.world_size)
            ]
        else:
            scatter_list = None
        tensor = torch.empty(1, dtype=torch.float32)
        dist.scatter(tensor, scatter_list=scatter_list, src=0, group=self.pg)
        self.assertEqual(tensor.item(), self.rank)

    @parametrize("op", REDUCE_OPS)
    def test_reduce(self, op):
        input_tensor = torch.tensor([self.rank + 1], dtype=torch.float32)
        dist.reduce(input_tensor, dst=0, op=op, group=self.pg)
        if self.rank == 0:
            self.assertEqual(input_tensor.item(), self._expected_reduce_result(op))

    @parametrize("op", REDUCE_OPS)
    def test_reduce_scatter(self, op):
        input_tensor = [
            torch.tensor([self.rank + 1], dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        output_tensor = torch.empty(1, dtype=torch.float32)
        dist.reduce_scatter(output_tensor, input_tensor, op=op, group=self.pg)
        self.assertEqual(output_tensor.item(), self._expected_reduce_result(op))

    @parametrize("op", REDUCE_OPS)
    def test_reduce_scatter_tensor(self, op):
        input_tensor = torch.full(
            (self.world_size,), self.rank + 1, dtype=torch.float32
        )
        output_tensor = torch.empty(1, dtype=torch.float32)
        dist.reduce_scatter_tensor(output_tensor, input_tensor, op=op, group=self.pg)
        self.assertEqual(output_tensor.item(), self._expected_reduce_result(op))

    def test_all_to_all(self):
        input_tensor = [
            torch.tensor([self.rank + 1], dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        output_tensor = [
            torch.empty(1, dtype=torch.float32) for _ in range(self.world_size)
        ]
        dist.all_to_all(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_all_to_all_single(self):
        input_tensor = torch.full(
            (self.world_size,), self.rank + 1, dtype=torch.float32
        )
        output_tensor = torch.empty([self.world_size], dtype=torch.float32)
        dist.all_to_all_single(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_send_recv(self):
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank + self.world_size - 1) % self.world_size
        send_tensor = torch.tensor([self.rank], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)
        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            dist.send(send_tensor, dst=send_rank, group=self.pg)
            dist.recv(recv_tensor, src=recv_rank, group=self.pg)
        else:
            # Odd ranks: receive first, then send
            dist.recv(recv_tensor, src=recv_rank, group=self.pg)
            dist.send(send_tensor, dst=send_rank, group=self.pg)
        # Each rank receives the rank number of the sender
        self.assertEqual(recv_tensor.item(), recv_rank)

    def test_barrier(self):
        dist.barrier(group=self.pg)
        # If we reach this point, the barrier succeeded without deadlock
        self.assertTrue(True)


devices = ["cpu", "cuda", "xpu"]
instantiate_device_type_tests(
    TestC10dTorchCommsBasic, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
