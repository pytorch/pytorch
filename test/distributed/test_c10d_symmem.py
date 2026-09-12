# Owner(s): ["oncall: distributed"]

import os
import unittest

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TestCase


HAS_CUDA = torch.cuda.is_available()

try:
    from torch._C._distributed_c10d import _SymmetricMemory  # noqa: F401

    HAS_SYMM_MEM = True
except ImportError:
    HAS_SYMM_MEM = False


def skip_unless_symmem(func):
    return unittest.skipUnless(
        HAS_CUDA and HAS_SYMM_MEM, "CUDA and SymmetricMemory required"
    )(func)


@skip_unless_symmem
class TestSymmemBackendUnit(TestCase):
    """Single-process smoke tests (no real communication)."""

    def test_registration(self):
        import torch.distributed.pysymmem  # noqa: F401

        self.assertIn("symmem", dist.Backend.backend_list)

    def test_backend_class_import(self):
        from torch._C._distributed_c10d import Backend as C10DBackend
        from torch.distributed.pysymmem import SymmemBackend

        self.assertTrue(issubclass(SymmemBackend, C10DBackend))

    def test_helpers(self):
        from torch.distributed.pysymmem import cast_buffer, nbytes_of, reduce_op_name

        t = torch.zeros(10, dtype=torch.float32)
        self.assertEqual(nbytes_of(t), 40)

        buf = torch.zeros(40, dtype=torch.uint8)
        casted = cast_buffer(buf, t)
        self.assertEqual(casted.dtype, torch.float32)
        self.assertEqual(casted.numel(), 10)

        self.assertEqual(reduce_op_name(dist.ReduceOp.SUM), "sum")
        self.assertEqual(reduce_op_name(dist.ReduceOp.MAX), "max")
        self.assertEqual(reduce_op_name(dist.ReduceOp.MIN), "min")
        self.assertEqual(reduce_op_name(dist.ReduceOp.PRODUCT), "product")
        self.assertEqual(reduce_op_name(dist.ReduceOp.AVG), "avg")


@skip_unless_symmem
class TestSymmemBackendCollectives(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    def _init_pg(self):
        import torch.distributed.pysymmem  # noqa: F401

        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "symmem", store=store, rank=self.rank, world_size=self.world_size
        )

    def _destroy_pg(self):
        dist.destroy_process_group()

    @property
    def device(self):
        return torch.device(f"cuda:{self.rank}")

    @skip_if_lt_x_gpu(2)
    def test_allreduce_sum_float32(self):
        self._init_pg()
        t = torch.ones(4, device=self.device) * (self.rank + 1)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.full((4,), 3.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_sum_bfloat16(self):
        self._init_pg()
        t = torch.ones(4, dtype=torch.bfloat16, device=self.device) * (self.rank + 1)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(self.device)
        expected = torch.full((4,), 3.0, dtype=torch.bfloat16, device=self.device)
        self.assertEqual(t, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_async(self):
        self._init_pg()
        t = torch.ones(4, device=self.device) * (self.rank + 1)
        work = dist.all_reduce(t, async_op=True)
        self.assertIsNotNone(work)
        work.wait()
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.full((4,), 3.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_large(self):
        self._init_pg()
        n = 1024 * 1024
        t = torch.ones(n, device=self.device) * (self.rank + 1)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.full((n,), 3.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_empty(self):
        self._init_pg()
        t = torch.ones(0, device=self.device)
        dist.all_reduce(t)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_broadcast_root0(self):
        self._init_pg()
        t = torch.ones(4, device=self.device) * (self.rank + 1)
        dist.broadcast(t, src=0)
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.ones(4, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_broadcast_root1(self):
        self._init_pg()
        t = torch.ones(4, device=self.device) * (self.rank + 1)
        dist.broadcast(t, src=1)
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.full((4,), 2.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_broadcast_2d(self):
        self._init_pg()
        t = torch.ones(3, 4, device=self.device) * (self.rank + 1)
        dist.broadcast(t, src=0)
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.ones(3, 4, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_reduce_root0(self):
        self._init_pg()
        t = torch.ones(4, device=self.device) * (self.rank + 1)
        dist.reduce(t, dst=0)
        torch.cuda.synchronize(self.device)
        if self.rank == 0:
            self.assertEqual(t, torch.full((4,), 3.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_reduce_root1(self):
        self._init_pg()
        t = torch.ones(4, device=self.device) * (self.rank + 1)
        dist.reduce(t, dst=1)
        torch.cuda.synchronize(self.device)
        if self.rank == 1:
            self.assertEqual(t, torch.full((4,), 3.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_allgather(self):
        self._init_pg()
        inp = torch.ones(4, device=self.device) * (self.rank + 1)
        out = [torch.zeros(4, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(out, inp)
        torch.cuda.synchronize(self.device)
        self.assertEqual(out[0], torch.ones(4, device=self.device))
        self.assertEqual(out[1], torch.full((4,), 2.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_allgather_2d(self):
        self._init_pg()
        inp = torch.ones(2, 3, device=self.device) * (self.rank + 1)
        out = [torch.zeros(2, 3, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(out, inp)
        torch.cuda.synchronize(self.device)
        self.assertEqual(out[0], torch.ones(2, 3, device=self.device))
        self.assertEqual(out[1], torch.full((2, 3), 2.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_all_gather_into_tensor(self):
        self._init_pg()
        inp = torch.ones(4, device=self.device) * (self.rank + 1)
        out = torch.zeros(8, device=self.device)
        dist.all_gather_into_tensor(out, inp)
        torch.cuda.synchronize(self.device)
        expected = torch.cat(
            [
                torch.ones(4, device=self.device),
                torch.full((4,), 2.0, device=self.device),
            ]
        )
        self.assertEqual(out, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter(self):
        self._init_pg()
        inp = [
            torch.ones(4, device=self.device) * (self.rank + 1)
            for _ in range(self.world_size)
        ]
        out = torch.zeros(4, device=self.device)
        dist.reduce_scatter(out, inp)
        torch.cuda.synchronize(self.device)
        expected = torch.full((4,), 3.0, device=self.device)
        self.assertEqual(out, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor(self):
        self._init_pg()
        inp = torch.ones(8, device=self.device) * (self.rank + 1)
        out = torch.zeros(4, device=self.device)
        dist.reduce_scatter_tensor(out, inp)
        torch.cuda.synchronize(self.device)
        expected = torch.full((4,), 3.0, device=self.device)
        self.assertEqual(out, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single(self):
        self._init_pg()
        inp = torch.ones(8, device=self.device) * (self.rank + 1)
        out = torch.zeros(8, device=self.device)
        dist.all_to_all_single(out, inp)
        torch.cuda.synchronize(self.device)
        expected = torch.cat(
            [
                torch.ones(4, device=self.device),
                torch.full((4,), 2.0, device=self.device),
            ]
        )
        self.assertEqual(out, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single_uneven(self):
        self._init_pg()
        if self.rank == 0:
            inp = torch.tensor([1.0, 2.0, 3.0], device=self.device)
            out = torch.zeros(2, device=self.device)
            in_splits = [1, 2]
            out_splits = [1, 1]
        else:
            inp = torch.tensor([4.0, 5.0, 6.0], device=self.device)
            out = torch.zeros(4, device=self.device)
            in_splits = [1, 2]
            out_splits = [2, 2]
        dist.all_to_all_single(out, inp, out_splits, in_splits)
        torch.cuda.synchronize(self.device)
        if self.rank == 0:
            self.assertEqual(out, torch.tensor([1.0, 4.0], device=self.device))
        else:
            self.assertEqual(
                out, torch.tensor([2.0, 3.0, 5.0, 6.0], device=self.device)
            )
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_alltoall(self):
        self._init_pg()
        inp = [
            torch.full((4,), float(self.rank + 1), device=self.device)
            for _ in range(self.world_size)
        ]
        out = [torch.zeros(4, device=self.device) for _ in range(self.world_size)]
        dist.all_to_all(out, inp)
        torch.cuda.synchronize(self.device)
        for i in range(self.world_size):
            self.assertEqual(out[i], torch.full((4,), float(i + 1), device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_scatter(self):
        self._init_pg()
        out = torch.zeros(4, device=self.device)
        if self.rank == 0:
            inp = [
                torch.ones(4, device=self.device),
                torch.full((4,), 2.0, device=self.device),
            ]
            dist.scatter(out, inp, src=0)
        else:
            dist.scatter(out, src=0)
        torch.cuda.synchronize(self.device)
        expected = torch.full((4,), float(self.rank + 1), device=self.device)
        self.assertEqual(out, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_gather(self):
        self._init_pg()
        inp = torch.ones(4, device=self.device) * (self.rank + 1)
        if self.rank == 0:
            out = [torch.zeros(4, device=self.device) for _ in range(self.world_size)]
            dist.gather(inp, out, dst=0)
        else:
            dist.gather(inp, dst=0)
        torch.cuda.synchronize(self.device)
        if self.rank == 0:
            self.assertEqual(out[0], torch.ones(4, device=self.device))
            self.assertEqual(out[1], torch.full((4,), 2.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_barrier(self):
        self._init_pg()
        dist.barrier()
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_send_recv(self):
        self._init_pg()
        peer = (self.rank + 1) % self.world_size
        send_t = torch.ones(4, device=self.device) * (self.rank + 1)
        recv_t = torch.zeros(4, device=self.device)

        if self.rank == 0:
            dist.send(send_t, dst=1)
            dist.recv(recv_t, src=1)
        else:
            dist.recv(recv_t, src=0)
            dist.send(send_t, dst=0)
        torch.cuda.synchronize(self.device)
        expected = torch.full((4,), float(peer + 1), device=self.device)
        self.assertEqual(recv_t, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_isend_irecv(self):
        self._init_pg()
        peer = (self.rank + 1) % self.world_size
        send_t = torch.ones(4, device=self.device) * (self.rank + 1)
        recv_t = torch.zeros(4, device=self.device)

        works = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_t, peer),
                dist.P2POp(dist.irecv, recv_t, peer),
            ]
        )
        for w in works:
            w.wait()
        torch.cuda.synchronize(self.device)
        expected = torch.full((4,), float(peer + 1), device=self.device)
        self.assertEqual(recv_t, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_split_single_rank(self):
        self._init_pg()
        subgroup = dist.new_group([0])
        if self.rank == 0:
            t = torch.ones(4, device=self.device) * 42.0
            dist.all_reduce(t, group=subgroup)
            torch.cuda.synchronize(self.device)
            self.assertEqual(t, torch.full((4,), 42.0, device=self.device))
        dist.barrier()
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_split_all_ranks(self):
        self._init_pg()
        subgroup = dist.new_group([0, 1])
        t = torch.ones(4, device=self.device) * (self.rank + 1)
        dist.all_reduce(t, group=subgroup)
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.full((4,), 3.0, device=self.device))
        dist.barrier()
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_allgather_large(self):
        self._init_pg()
        n = 256 * 1024
        inp = torch.ones(n, device=self.device) * (self.rank + 1)
        out = [torch.zeros(n, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(out, inp)
        torch.cuda.synchronize(self.device)
        self.assertEqual(out[0], torch.ones(n, device=self.device))
        self.assertEqual(out[1], torch.full((n,), 2.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_broadcast_large(self):
        self._init_pg()
        n = 256 * 1024
        t = torch.ones(n, device=self.device) * (self.rank + 1)
        dist.broadcast(t, src=0)
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.ones(n, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_get_future(self):
        self._init_pg()
        t = torch.ones(4, device=self.device) * (self.rank + 1)
        work = dist.all_reduce(t, async_op=True)
        fut = work.get_future()
        fut.wait()
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.full((4,), 3.0, device=self.device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_multiple_allreduce(self):
        self._init_pg()
        for _ in range(10):
            t = torch.ones(4, device=self.device) * (self.rank + 1)
            dist.all_reduce(t)
            torch.cuda.synchronize(self.device)
            self.assertEqual(t, torch.full((4,), 3.0, device=self.device))
        self._destroy_pg()


if __name__ == "__main__":
    run_tests()
