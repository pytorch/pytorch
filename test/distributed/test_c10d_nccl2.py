# Owner(s): ["oncall: distributed"]
#
# Basic sanity checks for the in-tree torchcomms NCCL backend
# (c10d::nccl2::ProcessGroupNCCL), selected via init_process_group(
# backend="nccl2"). Modeled on the torchcomms c10d tests; exercises the core
# collectives / point-to-point over real NCCL on multiple GPUs.

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


# The "nccl2" backend is normally discovered via the torch.distributed.backends
# entry point. Register it explicitly here too so the test is self-contained
# under editable installs, where a stale repo egg-info can shadow the dist-info
# entry points. _ensure_backend_registered short-circuits if already present.
try:
    from torch.distributed.distributed_c10d import _register_builtin_nccl2_backend

    _register_builtin_nccl2_backend()
except Exception:
    pass


class ProcessGroupNCCL2Test(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_allreduce(self) -> None:
        t = torch.full((10,), float(self.rank + 1), device=self.device)
        dist.all_reduce(t)
        expected = float(sum(range(1, self.world_size + 1)))
        self.assertEqual(t, torch.full((10,), expected, device=self.device))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_broadcast(self) -> None:
        t = torch.full((10,), float(self.rank + 1), device=self.device)
        dist.broadcast(t, src=0)
        self.assertEqual(t, torch.full((10,), 1.0, device=self.device))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_all_gather(self) -> None:
        t = torch.full((4,), float(self.rank), device=self.device)
        out = [torch.empty((4,), device=self.device) for _ in range(self.world_size)]
        dist.all_gather(out, t)
        for r in range(self.world_size):
            self.assertEqual(out[r], torch.full((4,), float(r), device=self.device))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_all_gather_into_tensor(self) -> None:
        t = torch.full((4,), float(self.rank), device=self.device)
        out = torch.empty((4 * self.world_size,), device=self.device)
        dist.all_gather_into_tensor(out, t)
        expected = torch.cat(
            [
                torch.full((4,), float(r), device=self.device)
                for r in range(self.world_size)
            ]
        )
        self.assertEqual(out, expected)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor(self) -> None:
        inp = torch.arange(4 * self.world_size, dtype=torch.float32, device=self.device)
        out = torch.empty((4,), device=self.device)
        dist.reduce_scatter_tensor(out, inp)
        base = torch.arange(
            4 * self.world_size, dtype=torch.float32, device=self.device
        )
        chunk = base[self.rank * 4 : (self.rank + 1) * 4]
        self.assertEqual(out, chunk * self.world_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_all_to_all_single(self) -> None:
        inp = torch.full((self.world_size,), float(self.rank), device=self.device)
        out = torch.empty((self.world_size,), device=self.device)
        dist.all_to_all_single(out, inp)
        # Rank r receives, at position s, the value sent by rank s -> float(s).
        self.assertEqual(
            out,
            torch.arange(self.world_size, dtype=torch.float32, device=self.device),
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_barrier(self) -> None:
        # Bind this rank's device first (barrier has no tensor to infer it from).
        torch.cuda.set_device(self.rank)
        dist.barrier()  # should not hang or raise

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_send_recv(self) -> None:
        # Ring: rank r sends to r+1, receives from r-1.
        send_t = torch.full((8,), float(self.rank), device=self.device)
        recv_t = torch.empty((8,), device=self.device)
        nxt = (self.rank + 1) % self.world_size
        prev = (self.rank - 1) % self.world_size
        if self.rank % 2 == 0:
            dist.send(send_t, nxt)
            dist.recv(recv_t, prev)
        else:
            dist.recv(recv_t, prev)
            dist.send(send_t, nxt)
        self.assertEqual(recv_t, torch.full((8,), float(prev), device=self.device))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_batch_isend_irecv(self) -> None:
        # Mixed batch on a single PG (the PP 1F1B pattern that needs coalescing).
        send_t = torch.full((1,), float(self.rank), device=self.device)
        recv_t = torch.empty((1,), device=self.device)
        nxt = (self.rank + 1) % self.world_size
        prev = (self.rank - 1) % self.world_size
        ops = [
            dist.P2POp(dist.isend, send_t, nxt),
            dist.P2POp(dist.irecv, recv_t, prev),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()
        self.assertEqual(recv_t, torch.full((1,), float(prev), device=self.device))


if __name__ == "__main__":
    if TEST_CUDA:
        run_tests()
