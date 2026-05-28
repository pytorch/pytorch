import os

# Owner(s): ["oncall: distributed"]
import unittest

from integration.helpers.TorchCommTestHelpers import get_device, get_rank_and_size

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)


class TestC10dTorchCommsBasic(unittest.TestCase):
    REDUCE_OPS = [
        subtest(dist.ReduceOp.SUM, name="SUM"),
        subtest(dist.ReduceOp.AVG, name="AVG"),
        subtest(dist.ReduceOp.MIN, name="MIN"),
        subtest(dist.ReduceOp.MAX, name="MAX"),
        subtest(dist.ReduceOp.PRODUCT, name="PRODUCT"),
    ]

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

    def _rank_value(self):
        return dist.get_rank() + 1

    def _skip_if_product_overflows(self, op):
        if op == dist.ReduceOp.PRODUCT and dist.get_world_size() > 12:
            self.skipTest(
                f"world_size={dist.get_world_size()} > 12: PRODUCT is world_size! "
                "and only up to 12! is exactly representable in float32"
            )

    def _expected_reduce_result(self, op):
        """Return the expected scalar result for a rank+1 input reduced across all ranks."""
        total = sum(range(1, dist.get_world_size() + 1))
        if op == dist.ReduceOp.SUM:
            return total
        elif op == dist.ReduceOp.AVG:
            return total / dist.get_world_size()
        elif op == dist.ReduceOp.MIN:
            return 1
        elif op == dist.ReduceOp.MAX:
            return dist.get_world_size()
        elif op == dist.ReduceOp.PRODUCT:
            product = 1
            for i in range(1, dist.get_world_size() + 1):
                product *= i
            return product
        raise ValueError(f"Unsupported op: {op}")

    @parametrize("op", REDUCE_OPS)
    def test_allreduce(self, op):
        self._skip_if_product_overflows(op)
        tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
        dist.all_reduce(tensor, op=op)
        self.assertEqual(tensor.item(), self._expected_reduce_result(op))

    def test_all_gather(self):
        input_tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
        gather_list = [
            torch.empty_like(input_tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gather_list, input_tensor)
        expected = list(range(1, dist.get_world_size() + 1))
        self.assertEqual([t.item() for t in gather_list], expected)

    def test_all_gather_into_tensor(self):
        input_tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
        output_tensor = torch.empty(dist.get_world_size(), dtype=torch.float32)
        dist.all_gather_into_tensor(output_tensor, input_tensor)
        expected = list(range(1, dist.get_world_size() + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_broadcast(self):
        tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
        dist.broadcast(tensor, src=0)
        self.assertEqual(tensor.item(), 1)

    def test_gather(self):
        tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
        gather_list = None
        if dist.get_rank() == 0:
            gather_list = [
                torch.empty_like(tensor) for _ in range(dist.get_world_size())
            ]
        dist.gather(tensor, gather_list=gather_list, dst=0)
        if dist.get_rank() == 0:
            expected = list(range(1, dist.get_world_size() + 1))
            self.assertEqual([t.item() for t in gather_list], expected)

    def test_scatter(self):
        if dist.get_rank() == 0:
            scatter_list = [
                torch.tensor([i], dtype=torch.float32)
                for i in range(dist.get_world_size())
            ]
        else:
            scatter_list = None
        tensor = torch.empty(1, dtype=torch.float32)
        dist.scatter(tensor, scatter_list=scatter_list, src=0)
        self.assertEqual(tensor.item(), dist.get_rank())

    @parametrize("op", REDUCE_OPS)
    def test_reduce(self, op):
        self._skip_if_product_overflows(op)
        input_tensor = torch.tensor([self._rank_value()], dtype=torch.float32)
        dist.reduce(input_tensor, dst=0, op=op)
        if dist.get_rank() == 0:
            self.assertEqual(input_tensor.item(), self._expected_reduce_result(op))

    @parametrize("op", REDUCE_OPS)
    def test_reduce_scatter(self, op):
        self._skip_if_product_overflows(op)
        input_tensor = [
            torch.tensor([self._rank_value()], dtype=torch.float32)
            for _ in range(dist.get_world_size())
        ]
        output_tensor = torch.empty(1, dtype=torch.float32)
        dist.reduce_scatter(output_tensor, input_tensor, op=op)
        self.assertEqual(output_tensor.item(), self._expected_reduce_result(op))

    @parametrize("op", REDUCE_OPS)
    def test_reduce_scatter_tensor(self, op):
        self._skip_if_product_overflows(op)
        input_tensor = torch.full(
            (dist.get_world_size(),), self._rank_value(), dtype=torch.float32
        )
        output_tensor = torch.empty(1, dtype=torch.float32)
        dist.reduce_scatter_tensor(output_tensor, input_tensor, op=op)
        self.assertEqual(output_tensor.item(), self._expected_reduce_result(op))

    def test_all_to_all(self):
        input_tensor = [
            torch.tensor([self._rank_value()], dtype=torch.float32)
            for _ in range(dist.get_world_size())
        ]
        output_tensor = [
            torch.empty(1, dtype=torch.float32) for _ in range(dist.get_world_size())
        ]
        dist.all_to_all(output_tensor, input_tensor)
        expected = list(range(1, dist.get_world_size() + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_all_to_all_single(self):
        input_tensor = torch.full(
            (dist.get_world_size(),), self._rank_value(), dtype=torch.float32
        )
        output_tensor = torch.empty([dist.get_world_size()], dtype=torch.float32)
        dist.all_to_all_single(output_tensor, input_tensor)
        expected = list(range(1, dist.get_world_size() + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_all_to_all_single_with_split_sizes(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Each rank sends (rank + 1) elements to every other rank,
        # so rank r's input_split_sizes are all (rank + 1).
        input_split_sizes = [rank + 1] * world_size
        # Rank r receives (sender_rank + 1) elements from each sender,
        # so output_split_sizes[i] = i + 1.
        output_split_sizes = [i + 1 for i in range(world_size)]

        input_tensor = torch.empty(sum(input_split_sizes), dtype=torch.float32)
        offset = 0
        for dst in range(world_size):
            input_tensor[offset : offset + input_split_sizes[dst]].fill_(rank + dst)
            offset += input_split_sizes[dst]

        output_tensor = torch.empty(sum(output_split_sizes), dtype=torch.float32)
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )

        # Verify: section from sender i should contain value (i + rank)
        offset = 0
        for src in range(world_size):
            section = output_tensor[offset : offset + output_split_sizes[src]]
            expected = torch.full_like(section, src + rank)
            self.assertTrue(
                torch.equal(section, expected),
                f"Mismatch in section from rank {src}: got {section}, expected {expected}",
            )
            offset += output_split_sizes[src]

    def test_send_recv(self):
        send_rank = (dist.get_rank() + 1) % dist.get_world_size()
        recv_rank = (
            dist.get_rank() + dist.get_world_size() - 1
        ) % dist.get_world_size()
        send_tensor = torch.tensor([dist.get_rank()], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)
        if dist.get_rank() % 2 == 0:
            # Even ranks: send first, then receive
            dist.send(send_tensor, dst=send_rank)
            dist.recv(recv_tensor, src=recv_rank)
        else:
            # Odd ranks: receive first, then send
            dist.recv(recv_tensor, src=recv_rank)
            dist.send(send_tensor, dst=send_rank)
        # Each rank receives the rank number of the sender
        self.assertEqual(recv_tensor.item(), recv_rank)

    def test_barrier(self):
        dist.barrier()
        # If we reach this point, the barrier succeeded without deadlock
        self.assertTrue(True)


instantiate_parametrized_tests(TestC10dTorchCommsBasic)

if __name__ == "__main__":
    unittest.main()
