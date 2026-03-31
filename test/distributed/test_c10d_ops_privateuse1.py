# Owner(s): ["oncall: distributed"]
# This test file contains positive tests for c10d collective ops on privateuse1 backends.
# Adapted from test_c10d_ops_nccl.py — strips NCCL-specific tests (CUDAGraph, float8,
# NCCL version-gated Premul, tensor register hook) and keeps portable collective ops.
# Uses MultiProcessTestCase so each test gets fresh processes (no poison pill cascade).

import math
import sys

import torch
import torch.distributed as c10d
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_accelerator_dist_backend,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

if torch.accelerator.current_accelerator() is None:
    print("No accelerator available, skipping tests", file=sys.stderr)
    sys.exit(0)

DEVICE_TYPE = torch.accelerator.current_accelerator().type
BACKEND = dist.get_default_backend_for_device(DEVICE_TYPE)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)


class ProcessGroupPrivateUse1OpTest(MultiProcessTestCase):
    world_size = 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def rank_to_accelerator(self):
        return init_multigpu_helper(self.world_size, BACKEND)

    def _init_pg(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend=BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        return c10d.distributed_c10d._get_default_group()

    @requires_accelerator_dist_backend()
    def test_empty_tensors(self):
        pg = self._init_pg()
        local_device_idx = self.rank_to_accelerator[self.rank][0]
        device = torch.device(DEVICE_TYPE, local_device_idx)

        xs = [torch.FloatTensor([]).to(device)]
        pg.broadcast(xs).wait()
        self.assertEqual(0, xs[0].numel())

        pg.allreduce(xs).wait()
        self.assertEqual(0, xs[0].numel())

        pg.reduce(xs).wait()
        self.assertEqual(0, xs[0].numel())

        ys = [[torch.FloatTensor([]).to(device) for _ in range(self.world_size)]]
        pg.allgather(ys, xs).wait()
        for y in ys[0]:
            self.assertEqual(0, y.numel())

        ys = [torch.FloatTensor([]).to(device)]
        xs = [[torch.FloatTensor([]).to(device) for _ in range(self.world_size)]]
        pg.reduce_scatter(ys, xs).wait()
        self.assertEqual(0, ys[0].numel())

    @requires_accelerator_dist_backend()
    def test_broadcast_ops(self):
        pg = self._init_pg()

        def broadcast(xs, rootRank, rootTensor):
            opts = dist.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()
            return xs

        for i in range(self.world_size):
            x = torch.tensor([self.rank]).to(
                torch.device(DEVICE_TYPE, self.rank_to_accelerator[self.rank][0])
            )
            output = broadcast([x], i, 0)
            self.assertEqual(torch.tensor([i]), output[0])

            expected_tensor = torch.empty([i + 1, i + 1]).fill_(i + 1)
            xs = [
                torch.empty([i + 1, i + 1]).fill_(-1).to(
                    torch.device(DEVICE_TYPE, device_idx)
                )
                for device_idx in self.rank_to_accelerator[self.rank]
            ]

            for j in range(len(xs)):
                if self.rank == i:
                    xs[j] = expected_tensor.to(
                        torch.device(DEVICE_TYPE, self.rank_to_accelerator[self.rank][j])
                    )
                broadcast(xs, i, j)
                for tensor in xs:
                    self.assertEqual(tensor, expected_tensor)

    @requires_accelerator_dist_backend()
    def test_allreduce_ops(self):
        pg = self._init_pg()
        local_device_id = self.rank_to_accelerator[self.rank][0]
        device = torch.device(DEVICE_TYPE, local_device_id)

        def allreduce(tensors, op):
            opts = dist.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce(tensors, opts)
            work.wait()

        # Sum
        tensors = [torch.tensor([self.rank + 1]).to(device)]
        allreduce(tensors, dist.ReduceOp.SUM)
        ndev = self.world_size
        self.assertEqual(torch.tensor([ndev * (ndev + 1) // 2]), tensors[0])

        # Product
        tensors = [torch.tensor([self.rank + 1]).to(device)]
        allreduce(tensors, dist.ReduceOp.PRODUCT)
        self.assertEqual(torch.tensor([math.factorial(self.world_size)]), tensors[0])

        # Min
        tensors = [torch.tensor([self.rank + 1]).to(device)]
        allreduce(tensors, dist.ReduceOp.MIN)
        self.assertEqual(torch.tensor([1]), tensors[0])

        # Max
        tensors = [torch.tensor([self.rank + 1]).to(device)]
        allreduce(tensors, dist.ReduceOp.MAX)
        self.assertEqual(torch.tensor([self.world_size]), tensors[0])

    @requires_accelerator_dist_backend()
    def test_reduce_ops(self):
        pg = self._init_pg()
        local_device_id = self.rank_to_accelerator[self.rank][0]
        device = torch.device(DEVICE_TYPE, local_device_id)

        def reduce(xs, rootRank, rootTensor, op=None):
            opts = dist.ReduceOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            if op:
                opts.reduceOp = op
            work = pg.reduce(xs, opts)
            work.wait()

        for rt in range(self.world_size):
            tensors = [torch.tensor([self.rank + 1]).to(device)]
            reduce(tensors, rt, 0)
            if self.rank == rt:
                self.assertEqual(
                    torch.tensor([self.world_size * (self.world_size + 1) // 2]),
                    tensors[0],
                )
            else:
                self.assertEqual(torch.tensor([self.rank + 1]), tensors[0])

    @requires_accelerator_dist_backend()
    def test_allgather_ops(self):
        pg = self._init_pg()
        local_device_ids = self.rank_to_accelerator[self.rank]

        def allgather(output_ts, input_ts):
            work = pg.allgather(output_ts, input_ts)
            return work.wait()

        tensors = [
            torch.empty(2, 2).fill_(2).to(torch.device(DEVICE_TYPE, i))
            for i in local_device_ids
        ]
        output_tensors = []
        expected_output = []

        output_per_device = (
            [torch.empty(2, 2).fill_(-1)] * len(local_device_ids) * self.world_size
        )
        expected_per_device = (
            [torch.empty(2, 2).fill_(2)] * len(local_device_ids) * self.world_size
        )

        for device in local_device_ids:
            output_tensors.append(
                [t.to(torch.device(DEVICE_TYPE, device)) for t in output_per_device]
            )
            expected_output.append(
                [t.to(torch.device(DEVICE_TYPE, device)) for t in expected_per_device]
            )

        allgather(output_tensors, tensors)
        self.assertEqual(output_tensors, expected_output)

    @requires_accelerator_dist_backend()
    def test_allgather_base_ops(self):
        pg = self._init_pg()
        local_device_id = self.rank_to_accelerator[self.rank][0]
        device = torch.device(DEVICE_TYPE, local_device_id)

        def allgather_base(output_t, input_t):
            work = pg._allgather_base(output_t, input_t)
            work.wait()

        tensor = torch.tensor([self.rank]).to(device)
        output_t = torch.empty((self.world_size), dtype=tensor.dtype).to(device)
        allgather_base(output_t, tensor)
        self.assertEqual(torch.arange(self.world_size), output_t) 

    @requires_accelerator_dist_backend()
    def test_gather_ops(self):
        pg = self._init_pg()
        local_device_ids = self.rank_to_accelerator[self.rank]
        num_devices = len(local_device_ids)

        def gather(output_t, input_t, rootRank):
            opts = dist.GatherOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.gather(output_t, input_t, opts)
            else:
                work = pg.gather([], input_t, opts)
            work.wait()

        tensors = []
        for device_id in local_device_ids:
            tensors.append(
                torch.tensor([self.rank]).to(torch.device(DEVICE_TYPE, device_id))
            )

        output_ts = []
        for idx in range(num_devices):
            device_idx = local_device_ids[idx]
            output_ts.append([])
            for rank in range(self.world_size):
                output_ts[idx].append(
                    torch.tensor([-1]).to(torch.device(DEVICE_TYPE, device_idx))
                )

        expected = [[torch.tensor([rank]) for rank in range(self.world_size)]]
        for rank in range(self.world_size):
            gather(output_ts, tensors, rank)
            if rank == self.rank:
                self.assertEqual(expected, output_ts)

    @requires_accelerator_dist_backend()
    def test_gather_stress(self):
        pg = self._init_pg()
        local_device_ids = self.rank_to_accelerator[self.rank]
        num_devices = len(local_device_ids)

        def gather(output_t, input_t, rootRank):
            opts = dist.GatherOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.gather(output_t, input_t, opts)
            else:
                work = pg.gather([], input_t, opts)
            work.wait()

        stress_length = 1000

        tensors = []
        for i in range(stress_length):
            tensors.append([])
            for device_id in local_device_ids:
                tensors[i].append(
                    torch.tensor([self.rank]).to(torch.device(DEVICE_TYPE, device_id))
                )

        output_ts = []
        for i in range(stress_length):
            output_ts.append([[] for _ in range(num_devices)])
            for idx, ls in enumerate(output_ts[i]):
                device_idx = local_device_ids[idx]
                for _ in range(self.world_size):
                    ls.append(
                        torch.tensor([-1]).to(torch.device(DEVICE_TYPE, device_idx))
                    )

        expected = [[torch.tensor([rank]) for rank in range(self.world_size)]]
        for i in range(stress_length):
            for rank in range(self.world_size):
                gather(output_ts[i], tensors[i], rank)
                if rank == self.rank:
                    self.assertEqual(output_ts[i], expected)

    @requires_accelerator_dist_backend()
    def test_scatter_ops(self):
        pg = self._init_pg()
        local_device_ids = self.rank_to_accelerator[self.rank]
        num_devices = len(local_device_ids)

        def scatter(output_t, input_t, rootRank):
            opts = dist.ScatterOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.scatter(output_t, input_t, opts)
            else:
                work = pg.scatter(output_t, [], opts)
            work.wait()

        tensors = []
        for device_id in local_device_ids:
            tensors.append(
                torch.tensor([-1]).to(torch.device(DEVICE_TYPE, device_id))
            )

        scatter_list = []
        for idx in range(num_devices):
            device_idx = local_device_ids[idx]
            scatter_list.append([])
            for rank in range(self.world_size):
                scatter_list[idx].append(
                    torch.tensor([rank]).to(torch.device(DEVICE_TYPE, device_idx))
                )

        expected = [torch.tensor([self.rank])]
        for rank in range(self.world_size):
            scatter(tensors, scatter_list, rank)
            self.assertEqual(expected, tensors)

    @requires_accelerator_dist_backend()
    def test_scatter_stress(self):
        pg = self._init_pg()
        local_device_ids = self.rank_to_accelerator[self.rank]
        num_devices = len(local_device_ids)

        def scatter(output_t, input_t, rootRank):
            opts = dist.ScatterOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.scatter(output_t, input_t, opts)
            else:
                work = pg.scatter(output_t, [], opts)
            work.wait()

        stress_length = 1000

        tensors = []
        for i in range(stress_length):
            tensors.append([])
            for device_id in local_device_ids:
                tensors[i].append(
                    torch.tensor([-1]).to(torch.device(DEVICE_TYPE, device_id))
                )

        scatter_list = []
        for i in range(stress_length):
            scatter_list.append([[] for _ in range(num_devices)])
            for idx, ls in enumerate(scatter_list[i]):
                device_idx = local_device_ids[idx]
                for rank in range(self.world_size):
                    ls.append(
                        torch.tensor([rank]).to(torch.device(DEVICE_TYPE, device_idx))
                    )

        expected = [torch.tensor([self.rank])]
        for i in range(stress_length):
            for rank in range(self.world_size):
                scatter(tensors[i], scatter_list[i], rank)
                self.assertEqual(tensors[i], expected)

    @requires_accelerator_dist_backend()
    def test_reduce_scatter_ops(self):
        pg = self._init_pg()
        local_device_ids = self.rank_to_accelerator[self.rank]
        num_devices = len(local_device_ids)

        def reduce_scatter(outputs, input_lists, op):
            opts = dist.ReduceScatterOptions()
            opts.reduceOp = op
            work = pg.reduce_scatter(outputs, input_lists, opts)
            work.wait()

        output = [
            torch.tensor([0]).to(torch.device(DEVICE_TYPE, i))
            for i in local_device_ids
        ]

        tensor_lists = []
        input_per_device = []
        for i in range(self.world_size):
            input_per_device.append(torch.tensor([self.rank + i + 1]))
        for device_idx in local_device_ids:
            tensor_lists.append(
                [t.to(torch.device(DEVICE_TYPE, device_idx)) for t in input_per_device]
            )

        # Sum
        reduce_scatter(output, tensor_lists, dist.ReduceOp.SUM)
        for i in range(num_devices):
            expected = torch.tensor(
                [(1 + self.world_size) * self.world_size // 2 + self.world_size * self.rank]
            )
            self.assertEqual(expected, output[i])

        # Min
        reduce_scatter(output, tensor_lists, dist.ReduceOp.MIN)
        for i in range(num_devices):
            expected = torch.tensor([self.rank + 1 + i])
            self.assertEqual(expected, output[i])

        # Max
        reduce_scatter(output, tensor_lists, dist.ReduceOp.MAX)
        for i in range(num_devices):
            expected = torch.tensor([self.rank + self.world_size + i])
            self.assertEqual(expected, output[i])

        # Product
        reduce_scatter(output, tensor_lists, dist.ReduceOp.PRODUCT)
        for i in range(num_devices):
            prod_val = math.perm(self.rank + self.world_size, self.world_size)
            expected = torch.tensor([prod_val])
            self.assertEqual(expected, output[i])

        # Test the input params overridden scenarios
        device = torch.device(DEVICE_TYPE, self.rank)
        output_tensor = torch.empty_like(input_per_device[0][0]).to(device)
        input_list = [tensor[0].to(device) for tensor in input_per_device]

        pg.reduce_scatter(output_tensor, input_list, dist.ReduceOp.SUM).wait()
        expected = torch.tensor(
            (1 + self.world_size) * self.world_size // 2 + self.world_size * self.rank
        )
        self.assertEqual(expected, output_tensor)

        pg.reduce_scatter(output_tensor, input_list, dist.ReduceOp.MIN).wait()
        self.assertEqual(torch.tensor(self.rank + 1), output_tensor)

        pg.reduce_scatter(output_tensor, input_list, dist.ReduceOp.MAX).wait()
        self.assertEqual(torch.tensor(self.rank + self.world_size), output_tensor)

        pg.reduce_scatter(output_tensor, input_list, dist.ReduceOp.PRODUCT).wait()
        prod_val = self.rank + 1
        for k in range(1, self.world_size):
            prod_val = prod_val * (self.rank + 1 + k)
        self.assertEqual(torch.tensor(prod_val), output_tensor)

    @requires_accelerator_dist_backend()
    def test_reduce_scatter_base_ops(self):
        pg = self._init_pg()
        local_device_id = self.rank_to_accelerator[self.rank][0]
        device = torch.device(DEVICE_TYPE, local_device_id)

        def reduce_scatter_base(output_t, input_t):
            work = pg._reduce_scatter_base(output_t, input_t)
            work.wait()

        output_t = torch.empty([1]).to(device)
        tensor = torch.arange(self.world_size, dtype=output_t.dtype).to(device)
        reduce_scatter_base(output_t, tensor)
        self.assertEqual(output_t[0], self.rank * self.world_size)

    @requires_accelerator_dist_backend()
    def test_reduce_scatter_bfloat16(self):
        self._init_pg()
        device = torch.device(DEVICE_TYPE, self.rank_to_accelerator[self.rank][0])

        numel = 1024
        output_tensor = torch.zeros(numel, dtype=torch.float32, device=device).to(
            torch.bfloat16
        )
        input_tensor = torch.ones(
            self.world_size * numel, dtype=torch.float32, device=device
        ).to(torch.bfloat16)
        dist.reduce_scatter_tensor(output_tensor, input_tensor)

        expected = (
            torch.empty_like(output_tensor).fill_(self.world_size).to(torch.bfloat16)
        )
        torch.testing.assert_close(output_tensor, expected)

    @requires_accelerator_dist_backend()
    def test_barrier(self):
        pg = self._init_pg()
        local_device_ids = self.rank_to_accelerator[self.rank]

        def allreduce(tensors):
            opts = dist.AllreduceOptions()
            work = pg.allreduce(tensors, opts)
            return work

        tensors_list = [[] for _ in range(len(local_device_ids))]
        for i in range(1, len(local_device_ids) + 1):
            for j in range(i):
                tensors_list[i - 1].append(
                    torch.tensor([j + 1]).to(
                        torch.device(DEVICE_TYPE, local_device_ids[j])
                    )
                )

        works = []
        for tensors in tensors_list:
            work = allreduce(tensors)
            works.append(work)

        pg.barrier().wait()

        for i in range(1, len(local_device_ids) + 1):
            for j in range(i):
                self.assertEqual(
                    torch.tensor([(j + 1) * self.world_size]), tensors_list[i - 1][j]
                )

    @requires_accelerator_dist_backend()
    def test_send_recv(self):
        self._init_pg()
        device = self.rank_to_accelerator[self.rank][0]

        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10, device=device)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)

    @requires_accelerator_dist_backend()
    def test_send_recv_complex(self):
        self._init_pg()
        device = self.rank_to_accelerator[self.rank][0]

        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10, dtype=torch.cfloat, device=device)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, dtype=torch.cfloat, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)

    @requires_accelerator_dist_backend()
    def test_send_recv_object_list(self):
        self._init_pg()
        device = self.rank_to_accelerator[self.rank][0]

        val = 99 if self.rank == 0 else None
        object_list = [val] * self.world_size
        if self.rank == 0:
            dist.send_object_list(object_list, 1, device=device)
        if self.rank == 1:
            dist.recv_object_list(object_list, 0, device=device)
            self.assertEqual(object_list[0], 99)


if __name__ == "__main__":
    run_tests()
