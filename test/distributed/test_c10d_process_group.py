# Owner(s): ["oncall: distributed"]

import contextlib
import math
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10D_BACKENDS,
    C10dBackendTestContinuous,
    instantiate_backend_tests,
)

from torch._C._distributed_c10d import WorkResult
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import get_cycles_per_ms, run_tests


class AbstractProcessGroupTest(C10dBackendTestContinuous):
    def test_new_group_rank_normalization(self):
        ranks = torch.arange(self.world_size)
        group = dist.new_group(ranks=ranks)
        self.assertEqual(
            dist.get_process_group_ranks(group), list(range(self.world_size))
        )

        tensor = torch.tensor([self.rank], device=self.device)
        dist.broadcast(tensor, src=0, group=group)
        self.assertEqual(tensor, torch.zeros_like(tensor))

        with self.assertRaisesRegex(TypeError, "ranks must be a sequence of integers"):
            dist.new_group(ranks=torch.arange(self.world_size, dtype=torch.float))

        with self.assertRaisesRegex(ValueError, "must not contain duplicate entries"):
            dist.new_group(ranks=[0, torch.tensor(0)])

    def test_sequence_numbers(self):
        if not self.supports_sequence_numbers:
            self.skipTest(f"{self.backend_name} does not support sequence numbers")
        default_pg = dist.distributed_c10d._get_default_group()
        self.assertEqual(default_pg._get_sequence_number_for_group(), 0)
        for sequence_number in range(1, 4):
            dist.all_reduce(torch.ones(1, device=self.device))
            self.assertEqual(
                default_pg._get_sequence_number_for_group(), sequence_number
            )

        subgroup = dist.new_group(list(range(self.world_size)))
        self.assertEqual(subgroup._get_sequence_number_for_group(), 0)
        dist.all_reduce(torch.ones(1, device=self.device), group=subgroup)
        self.assertEqual(subgroup._get_sequence_number_for_group(), 1)
        self.assertEqual(default_pg._get_sequence_number_for_group(), 3)

    def test_work_sequence_number(self):
        if not self.supports_work_sequence_number:
            self.skipTest(f"{self.backend_name} does not report work seq numbers")
        default_pg = dist.distributed_c10d._get_default_group()
        for expected in range(1, 4):
            work = dist.all_reduce(torch.ones(1, device=self.device), async_op=True)
            self.assertEqual(work._get_sequence_number(), expected)
            self.assertEqual(default_pg._get_sequence_number_for_group(), expected)
            work.wait()

        subgroup = dist.new_group(list(range(self.world_size)))
        work = dist.all_reduce(
            torch.ones(1, device=self.device), group=subgroup, async_op=True
        )
        self.assertEqual(work._get_sequence_number(), 1)
        work.wait()

    def test_work_duration(self):
        if not self.supports_collectives_timing:
            self.skipTest(f"{self.backend_name} does not support collectives timing")
        default_pg = dist.distributed_c10d._get_default_group()
        tensor = torch.ones(1024, device=self.device)

        untimed_work = dist.all_reduce(tensor, async_op=True)
        untimed_work.wait()
        default_pg._enable_collectives_timing()
        # Timing is only enabled for collectives issued afterwards.
        with self.assertRaises(RuntimeError):
            untimed_work._get_duration()

        work = dist.all_reduce(tensor, async_op=True)
        work.wait()
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        duration = work._get_duration()
        self.assertGreater(duration, 0)
        self.assertTrue(math.isfinite(duration))

    def test_different_group_initialization_order(self):
        if self.device_type == "cuda":
            torch.cuda.set_device(0)
        full_group = dist.new_group(list(range(self.world_size)))
        singleton = dist.new_group([0])
        if self.rank == 0:
            tensor = torch.ones(1, device=self.device)
            dist.all_reduce(tensor, group=singleton)
            self.assertEqual(tensor, torch.ones_like(tensor))
        dist.barrier(group=full_group)

    def test_wait_unregisters_work(self):
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            tensor = torch.ones(1, device=self.device)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            work = dist.all_reduce(tensor, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            work.wait()
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

    def test_async_work_lifetime(self):
        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        del work
        dist.barrier()
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(tensor, expected)

        tensor = torch.ones(4, device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        del tensor
        del work
        dist.barrier()

    def test_work_wait_is_stream_ordered(self):
        if self.device_type != "cuda":
            self.skipTest(f"{self.backend_name} does not use CUDA")
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        torch.cuda._sleep(int(250 * get_cycles_per_ms()))
        work = dist.all_reduce(torch.ones(4, device=self.device), async_op=True)

        work.wait()

        self.assertFalse(torch.cuda.current_stream().query())
        torch.cuda.synchronize()

    @contextlib.contextmanager
    def _hold_peers(self, key):
        # Peers enter only after rank 0 leaves, so rank 0's collectives issued inside
        # stay pending and nothing inside may wait on them. Run each op once first:
        # a first use may connect to peers. Only rank 0's observations are reliable;
        # broadcast them before asserting, as a rank failing alone hangs teardown.
        store = dist.distributed_c10d._get_default_store()
        if self.rank != 0:
            store.wait([key], timedelta(seconds=60))
        yield
        if self.rank == 0:
            store.set(key, "1")

    def test_work_get_future_does_not_block_current_stream(self):
        if self.device_type != "cuda":
            self.skipTest(f"{self.backend_name} does not use CUDA")
        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        with self._hold_peers("get_future"):
            with torch.cuda.stream(side):
                work = dist.all_reduce(tensor, async_op=True)

            future = work.get_future()

            idle = torch.cuda.current_stream().query()
        result = future.then(lambda fut: fut.value()[0].clone()).wait()
        idle = torch.tensor(idle, device=self.device)
        dist.broadcast(idle, src=0)
        self.assertTrue(idle)
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(result, expected)

    def test_work_block_current_stream_is_stream_ordered(self):
        if self.device_type != "cuda":
            self.skipTest(f"{self.backend_name} does not use CUDA")
        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        with self._hold_peers("block_current_stream"):
            with torch.cuda.stream(side):
                work = dist.all_reduce(tensor, async_op=True)

            work.block_current_stream()

            result = tensor.clone()
        dist.broadcast(result, src=0)
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(result, expected)

    def test_aliased_all_gather_future_is_stream_ordered(self):
        if self.device_type != "cuda":
            self.skipTest(f"{self.backend_name} does not use CUDA")
        # Large enough that a read racing the copy-back sees stale rows.
        tensor = torch.full((1 << 22,), float(self.rank + 1), device=self.device)
        output = torch.full_like(tensor, -1.0)
        dist.all_gather([torch.empty_like(tensor)] * self.world_size, tensor)
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        with self._hold_peers("aliased_all_gather"):
            with torch.cuda.stream(side):
                work = dist.all_gather(
                    [output] * self.world_size, tensor, async_op=True
                )

            future = work.get_future().then(lambda fut: output.clone())
        result = future.wait()
        dist.broadcast(result, src=0)

        # Every rank's row is copied into the same output; the last rank's wins.
        self.assertEqual(result, torch.full_like(tensor, float(self.world_size)))

    def test_work_result_future(self):
        if not self.supports_work_result:
            self.skipTest(f"{self.backend_name} does not report work results")
        work = dist.all_reduce(torch.ones(4, device=self.device), async_op=True)

        result = work.get_future_result().wait()

        self.assertEqual(WorkResult(result), WorkResult.SUCCESS)
        self.assertTrue(work.is_completed())

    def test_work_result_future_p2p_and_reduce_scatter(self):
        # gloo completes p2p results on wait() and composes reduce_scatter
        # from allreduce works.
        if not self.supports_work_result:
            self.skipTest(f"{self.backend_name} does not report work results")
        send = torch.ones(4, device=self.device)
        recv = torch.empty(4, device=self.device)
        works = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send, (self.rank + 1) % self.world_size),
                dist.P2POp(dist.irecv, recv, (self.rank - 1) % self.world_size),
            ]
        )
        output = torch.empty(1, device=self.device)
        inputs = [torch.ones(1, device=self.device)] * self.world_size
        works.append(dist.reduce_scatter(output, inputs, async_op=True))

        for work in works:
            work.wait()
            result = work.get_future_result().wait()
            self.assertEqual(WorkResult(result), WorkResult.SUCCESS)

    def test_work_future(self):
        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        future = work.get_future()
        future.wait()
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(tensor, expected)


instantiate_backend_tests(
    globals(),
    "ProcessGroup",
    AbstractProcessGroupTest,
    C10D_BACKENDS,
    harness=MultiProcContinuousTest,
)


if __name__ == "__main__":
    run_tests()
