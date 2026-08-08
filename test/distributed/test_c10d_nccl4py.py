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


try:
    import nccl.core  # noqa: F401

    from torch.distributed.nccl4py_backend import (
        _create_nccl4py_backend,
        NCCL4PyBackend,
    )

    HAS_NCCL4PY = True
except ImportError:
    HAS_NCCL4PY = False

HAS_CUDA = torch.cuda.is_available()


def skip_unless_nccl4py(func):
    return unittest.skipUnless(HAS_NCCL4PY and HAS_CUDA, "nccl4py and CUDA required")(
        func
    )


@skip_unless_nccl4py
class TestNCCL4PyBackendUnit(TestCase):
    """Single-process smoke tests (no real NCCL communication)."""

    def test_registration(self):
        dist.Backend.register_backend(
            "nccl4py_test", _create_nccl4py_backend, devices=["cuda"]
        )
        self.assertIn("nccl4py_test", dist.Backend.backend_list)

    def test_backend_name(self):
        store = dist.HashStore()
        backend = NCCL4PyBackend(
            store, 0, 1, timeout=torch.distributed.default_pg_timeout
        )
        self.assertEqual(backend.name(), "nccl4py")
        self.assertEqual(backend.rank(), 0)
        self.assertEqual(backend.size(), 1)
        self.assertTrue(backend.supports_splitting)
        self.assertEqual(backend.options.backend, "nccl4py")
        backend.shutdown()


@skip_unless_nccl4py
class TestNCCL4PyBackendCollectives(MultiProcessTestCase):
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
        dist.Backend.register_backend(
            "nccl4py", _create_nccl4py_backend, devices=["cuda"]
        )
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl4py", store=store, rank=self.rank, world_size=self.world_size
        )

    def _destroy_pg(self):
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_allreduce(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        t = torch.ones(4, device=device) * (self.rank + 1)
        dist.all_reduce(t)
        # SUM of [1,1,1,1] and [2,2,2,2] = [3,3,3,3]
        expected = torch.full((4,), 3.0, device=device)
        torch.cuda.synchronize(device)
        self.assertEqual(t, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_async(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        t = torch.ones(4, device=device) * (self.rank + 1)
        work = dist.all_reduce(t, async_op=True)
        self.assertIsNotNone(work)
        work.wait()
        expected = torch.full((4,), 3.0, device=device)
        torch.cuda.synchronize(device)
        self.assertEqual(t, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_broadcast(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        t = torch.ones(4, device=device) * (self.rank + 1)
        dist.broadcast(t, src=0)
        expected = torch.ones(4, device=device)
        torch.cuda.synchronize(device)
        self.assertEqual(t, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_reduce(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        t = torch.ones(4, device=device) * (self.rank + 1)
        dist.reduce(t, dst=0)
        torch.cuda.synchronize(device)
        if self.rank == 0:
            expected = torch.full((4,), 3.0, device=device)
            self.assertEqual(t, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_allgather(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        inp = torch.ones(4, device=device) * (self.rank + 1)
        out = [torch.zeros(4, device=device) for _ in range(self.world_size)]
        dist.all_gather(out, inp)
        torch.cuda.synchronize(device)
        self.assertEqual(out[0], torch.ones(4, device=device))
        self.assertEqual(out[1], torch.full((4,), 2.0, device=device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_all_gather_into_tensor(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        inp = torch.ones(4, device=device) * (self.rank + 1)
        out = torch.zeros(8, device=device)
        dist.all_gather_into_tensor(out, inp)
        torch.cuda.synchronize(device)
        expected = torch.cat(
            [torch.ones(4, device=device), torch.full((4,), 2.0, device=device)]
        )
        self.assertEqual(out, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        inp = torch.ones(8, device=device) * (self.rank + 1)
        out = torch.zeros(4, device=device)
        dist.reduce_scatter_tensor(out, inp)
        torch.cuda.synchronize(device)
        # SUM: each rank gets chunk of (1+2)=3
        expected = torch.full((4,), 3.0, device=device)
        self.assertEqual(out, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_scatter(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        out = torch.zeros(4, device=device)
        if self.rank == 0:
            inp = [
                torch.ones(4, device=device),
                torch.full((4,), 2.0, device=device),
            ]
            dist.scatter(out, inp, src=0)
        else:
            dist.scatter(out, src=0)
        torch.cuda.synchronize(device)
        expected = torch.full((4,), float(self.rank + 1), device=device)
        self.assertEqual(out, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_gather(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        inp = torch.ones(4, device=device) * (self.rank + 1)
        if self.rank == 0:
            out = [torch.zeros(4, device=device) for _ in range(self.world_size)]
            dist.gather(inp, out, dst=0)
        else:
            dist.gather(inp, dst=0)
        torch.cuda.synchronize(device)
        if self.rank == 0:
            self.assertEqual(out[0], torch.ones(4, device=device))
            self.assertEqual(out[1], torch.full((4,), 2.0, device=device))
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        inp = torch.ones(8, device=device) * (self.rank + 1)
        out = torch.zeros(8, device=device)
        dist.all_to_all_single(out, inp)
        torch.cuda.synchronize(device)
        expected = torch.cat(
            [torch.ones(4, device=device), torch.full((4,), 2.0, device=device)]
        )
        self.assertEqual(out, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_send_recv(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        peer = (self.rank + 1) % self.world_size
        send_t = torch.ones(4, device=device) * (self.rank + 1)
        recv_t = torch.zeros(4, device=device)

        works = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_t, peer),
                dist.P2POp(dist.irecv, recv_t, peer),
            ]
        )
        for w in works:
            w.wait()
        torch.cuda.synchronize(device)
        expected = torch.full((4,), float(peer + 1), device=device)
        self.assertEqual(recv_t, expected)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_barrier(self):
        self._init_pg()
        dist.barrier()
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_allreduce_rejects_noncontiguous(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        t = torch.ones(4, 4, device=device)[::2]
        with self.assertRaises(ValueError):
            dist.all_reduce(t)
        self._destroy_pg()

    @skip_if_lt_x_gpu(2)
    def test_send_recv_rejects_noncontiguous(self):
        self._init_pg()
        device = torch.device(f"cuda:{self.rank}")
        t = torch.ones(4, 4, device=device)[::2]
        if self.rank == 0:
            with self.assertRaises(ValueError):
                dist.send(t, dst=1)
        elif self.rank == 1:
            with self.assertRaises(ValueError):
                dist.recv(t, src=0)
        self._destroy_pg()


if __name__ == "__main__":
    run_tests()
