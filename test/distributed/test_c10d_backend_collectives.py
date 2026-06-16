# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    load_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

load_tests = load_tests  # noqa: PLW0127

BACKEND_ALLOWLIST = ("gloo", "nccl", "nccl2")
CUDA_BACKENDS = ("nccl", "nccl2")
ALL_TO_ALL_BACKENDS = ("nccl", "nccl2")
ALL_GATHER_COALESCED_BACKENDS = ("gloo",)
COLLECTIVE_DTYPES = (torch.float32, torch.int32, torch.int8)


def _sum_rank_values(world_size):
    return world_size * (world_size + 1) // 2


class BackendCollectivesRegistrationTest(TestCase):
    def test_nccl2_backend_registration(self):
        if not dist.is_nccl_available():
            self.skipTest("NCCL is not available")
        self.assertTrue(dist.is_backend_available("nccl2"))
        self.assertEqual(dist.Backend.NCCL2, "nccl2")


@instantiate_parametrized_tests
class BackendCollectivesTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _device_for_backend(self, backend):
        if backend not in BACKEND_ALLOWLIST:
            self.skipTest(f"backend {backend} is not allowlisted")
        if not dist.is_backend_available(backend):
            self.skipTest(f"{backend} backend is not available")
        if backend in CUDA_BACKENDS:
            if not dist.is_nccl_available():
                self.skipTest("NCCL is not available")
            if not torch.cuda.is_available():
                self.skipTest("CUDA is not available")
            if torch.cuda.device_count() < self.world_size:
                self.skipTest(f"{backend} tests require {self.world_size} GPUs")
            device = torch.device("cuda", self.rank)
            torch.cuda.set_device(device)
            return device
        if backend == "gloo":
            if not dist.is_gloo_available():
                self.skipTest("Gloo is not available")
            return torch.device("cpu")
        raise AssertionError(f"unhandled backend {backend}")

    def _init_process_group(self, backend):
        device = self._device_for_backend(backend)
        store = dist.FileStore(self.file_name, self.world_size)
        kwargs = {"device_id": device} if device.type == "cuda" else {}
        dist.init_process_group(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            **kwargs,
        )
        return device

    def _rank_tensor(self, device, size=4, dtype=torch.float32):
        return torch.full((size,), self.rank + 1, device=device, dtype=dtype)

    @parametrize("backend", BACKEND_ALLOWLIST)
    @parametrize("dtype", COLLECTIVE_DTYPES)
    def test_all_reduce(self, backend, dtype):
        device = self._init_process_group(backend)

        tensor = self._rank_tensor(device, dtype=dtype)
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        expected = torch.full_like(tensor, _sum_rank_values(self.world_size))
        self.assertEqual(tensor, expected)

        tensor = self._rank_tensor(device, dtype=dtype)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        self.assertEqual(tensor, torch.full_like(tensor, self.world_size))

    @parametrize("backend", BACKEND_ALLOWLIST)
    @parametrize("dtype", COLLECTIVE_DTYPES)
    def test_broadcast_and_reduce(self, backend, dtype):
        device = self._init_process_group(backend)

        root = self.world_size - 1
        tensor = self._rank_tensor(device, dtype=dtype)
        dist.broadcast(tensor, src=root)
        self.assertEqual(tensor, torch.full_like(tensor, root + 1))

        tensor = self._rank_tensor(device, dtype=dtype)
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            self.assertEqual(
                tensor, torch.full_like(tensor, _sum_rank_values(self.world_size))
            )

    @parametrize("backend", BACKEND_ALLOWLIST)
    @parametrize("dtype", COLLECTIVE_DTYPES)
    def test_all_gather(self, backend, dtype):
        device = self._init_process_group(backend)

        tensor = self._rank_tensor(device, dtype=dtype)
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        work = dist.all_gather(gathered, tensor, async_op=True)
        work.wait()
        for rank, output in enumerate(gathered):
            self.assertEqual(output, torch.full_like(output, rank + 1))

        output = torch.empty(
            self.world_size * tensor.numel(), device=device, dtype=dtype
        )
        dist.all_gather_single(output, tensor)
        expected = torch.cat(
            [torch.full_like(tensor, rank + 1) for rank in range(self.world_size)]
        )
        self.assertEqual(output, expected)

    @parametrize("backend", BACKEND_ALLOWLIST)
    @parametrize("dtype", COLLECTIVE_DTYPES)
    def test_reduce_scatter(self, backend, dtype):
        device = self._init_process_group(backend)
        chunk_size = 4

        input_list = [
            torch.full(
                (chunk_size,),
                self.rank + 1 + chunk,
                device=device,
                dtype=dtype,
            )
            for chunk in range(self.world_size)
        ]
        output = torch.empty(chunk_size, device=device, dtype=dtype)
        work = dist.reduce_scatter(output, input_list, async_op=True)
        work.wait()
        expected_value = _sum_rank_values(self.world_size) + self.world_size * self.rank
        self.assertEqual(output, torch.full_like(output, expected_value))

        input_tensor = torch.cat(input_list)
        output = torch.empty(chunk_size, device=device, dtype=dtype)
        dist.reduce_scatter_single(output, input_tensor)
        self.assertEqual(output, torch.full_like(output, expected_value))

    @parametrize("backend", BACKEND_ALLOWLIST)
    @parametrize("dtype", COLLECTIVE_DTYPES)
    def test_all_to_all_single(self, backend, dtype):
        device = self._init_process_group(backend)
        chunk_size = 3

        input_tensor = (
            torch.arange(self.world_size * chunk_size, device=device, dtype=dtype)
            + self.rank * 100
        )
        output = torch.empty_like(input_tensor)
        dist.all_to_all_single(output, input_tensor)

        expected_chunks = []
        for src in range(self.world_size):
            start = self.rank * chunk_size
            expected = torch.arange(
                start, start + chunk_size, device=device, dtype=dtype
            )
            expected_chunks.append(expected + src * 100)
        self.assertEqual(output, torch.cat(expected_chunks))

    @parametrize("backend", ALL_TO_ALL_BACKENDS)
    @parametrize("dtype", COLLECTIVE_DTYPES)
    def test_all_to_all(self, backend, dtype):
        device = self._init_process_group(backend)

        inputs = [
            torch.full((4,), self.rank * 10 + peer, device=device, dtype=dtype)
            for peer in range(self.world_size)
        ]
        outputs = [torch.empty_like(inputs[0]) for _ in range(self.world_size)]
        work = dist.all_to_all(outputs, inputs, async_op=True)
        work.wait()
        for peer, output in enumerate(outputs):
            self.assertEqual(output, torch.full_like(output, peer * 10 + self.rank))

    @parametrize("backend", BACKEND_ALLOWLIST)
    @parametrize("dtype", COLLECTIVE_DTYPES)
    def test_gather_and_scatter(self, backend, dtype):
        device = self._init_process_group(backend)

        root = 0
        tensor = self._rank_tensor(device, dtype=dtype)
        gathered = None
        if self.rank == root:
            gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.gather(tensor, gather_list=gathered, dst=root)
        if self.rank == root:
            for rank, output in enumerate(gathered):
                self.assertEqual(output, torch.full_like(output, rank + 1))

        output = torch.empty_like(tensor)
        scatter_list = None
        if self.rank == root:
            scatter_list = [
                torch.full_like(tensor, rank + 11) for rank in range(self.world_size)
            ]
        dist.scatter(output, scatter_list=scatter_list, src=root)
        self.assertEqual(output, torch.full_like(output, self.rank + 11))

    @parametrize("backend", BACKEND_ALLOWLIST)
    @parametrize("dtype", COLLECTIVE_DTYPES)
    def test_send_recv(self, backend, dtype):
        device = self._init_process_group(backend)

        send_peer = (self.rank + 1) % self.world_size
        recv_peer = (self.rank - 1) % self.world_size
        send = self._rank_tensor(device, dtype=dtype)
        recv = torch.empty_like(send)
        ops = [
            dist.P2POp(dist.isend, send, send_peer),
            dist.P2POp(dist.irecv, recv, recv_peer),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        self.assertEqual(recv, torch.full_like(recv, recv_peer + 1))

    @parametrize("backend", BACKEND_ALLOWLIST)
    @parametrize("dtype", COLLECTIVE_DTYPES)
    def test_coalesced_collectives(self, backend, dtype):
        device = self._init_process_group(backend)

        tensors = [self._rank_tensor(device, size=size, dtype=dtype) for size in (2, 4)]
        dist.all_reduce_coalesced(tensors, op=dist.ReduceOp.SUM)
        for tensor in tensors:
            self.assertEqual(
                tensor, torch.full_like(tensor, _sum_rank_values(self.world_size))
            )

        if backend not in ALL_GATHER_COALESCED_BACKENDS:
            return

        inputs = [self._rank_tensor(device, size=size, dtype=dtype) for size in (2, 4)]
        outputs = [
            [torch.empty_like(input) for input in inputs]
            for _ in range(self.world_size)
        ]
        dist.all_gather_coalesced(outputs, inputs)
        for rank, output_list in enumerate(outputs):
            for output in output_list:
                self.assertEqual(output, torch.full_like(output, rank + 1))

    @parametrize("backend", BACKEND_ALLOWLIST)
    def test_barrier(self, backend):
        device = self._init_process_group(backend)

        dist.barrier()
        tensor = self._rank_tensor(device)
        dist.all_reduce(tensor)
        expected = torch.full_like(tensor, _sum_rank_values(self.world_size))
        self.assertEqual(tensor, expected)


if __name__ == "__main__":
    run_tests()
