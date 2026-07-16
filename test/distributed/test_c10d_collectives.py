# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10D_BACKENDS,
    C10dBackendTest,
    instantiate_backend_tests,
)

from torch.testing._internal.common_utils import run_tests


class AbstractCollectivesTest(C10dBackendTest):
    @property
    def rank_sum(self):
        return self.world_size * (self.world_size - 1) // 2

    def test_all_reduce(self):
        self._init_pg()
        for async_op in (False, True):
            tensor = torch.full((4,), float(self.rank), device=self.device)
            work = dist.all_reduce(tensor, async_op=async_op)
            if work is not None:
                work.wait()
            self.assertEqual(tensor, torch.full_like(tensor, self.rank_sum))

    def test_broadcast(self):
        self._init_pg()
        tensor = torch.full((4,), float(self.rank), device=self.device)
        dist.broadcast(tensor, src=0)
        self.assertEqual(tensor, torch.zeros_like(tensor))

    def test_all_gather(self):
        self._init_pg()
        tensor = torch.full((4,), float(self.rank), device=self.device)
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(output, tensor)
        for rank, result in enumerate(output):
            self.assertEqual(result, torch.full_like(result, rank))

    def test_all_gather_single(self):
        self._init_pg()
        tensor = torch.full((4,), float(self.rank), device=self.device)
        output = torch.empty(self.world_size * tensor.numel(), device=self.device)
        dist.all_gather_single(output, tensor)
        expected = torch.arange(
            self.world_size, dtype=output.dtype, device=self.device
        ).repeat_interleave(tensor.numel())
        self.assertEqual(output, expected)

    def test_reduce_scatter_single(self):
        self._init_pg()
        input = torch.arange(
            4 * self.world_size, dtype=torch.float32, device=self.device
        )
        output = torch.empty(4, device=self.device)
        dist.reduce_scatter_single(output, input)
        expected = input.chunk(self.world_size)[self.rank] * self.world_size
        self.assertEqual(output, expected)

    def test_all_to_all_single(self):
        self._init_pg()
        input = torch.full((self.world_size,), float(self.rank), device=self.device)
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input)
        expected = torch.arange(self.world_size, dtype=output.dtype, device=self.device)
        self.assertEqual(output, expected)

    def test_barrier(self):
        self._init_pg()
        dist.barrier()

    def test_all_reduce_coalesced(self):
        self._init_pg()
        for async_op in (False, True):
            batches = [
                [
                    torch.full(
                        (i + 1,), float(self.rank + batch + i), device=self.device
                    )
                    for i in range(3)
                ]
                for batch in range(2)
            ]
            works = [
                dist.all_reduce_coalesced(tensors, async_op=async_op)
                for tensors in batches
            ]
            for work in works:
                if work is not None:
                    work.wait()
            for batch, tensors in enumerate(batches):
                for i, tensor in enumerate(tensors):
                    expected_value = self.rank_sum + (batch + i) * self.world_size
                    self.assertEqual(tensor, torch.full_like(tensor, expected_value))

    def test_coalescing_manager(self):
        if not self.supports_coalescing:
            self.skipTest(f"{self.backend_name} does not support coalescing")
        self._init_pg()

        for async_ops in (False, True):
            tensors = [
                torch.full((i + 1,), float(self.rank + i), device=self.device)
                for i in range(3)
            ]
            with dist._coalescing_manager(
                device=self.device, async_ops=async_ops
            ) as cm:
                for tensor in tensors:
                    dist.all_reduce(tensor)
            self.assertEqual(len(cm.works), 1 if async_ops else 0)
            cm.wait()
            for i, tensor in enumerate(tensors):
                expected_value = self.rank_sum + i * self.world_size
                self.assertEqual(tensor, torch.full_like(tensor, expected_value))

        inputs = [
            torch.full((i + 1,), float(self.rank + i), device=self.device)
            for i in range(3)
        ]
        for async_ops in (False, True):
            gathered = [
                torch.empty(input.numel() * self.world_size, device=self.device)
                for input in inputs
            ]
            with dist._coalescing_manager(
                device=self.device, async_ops=async_ops
            ) as cm:
                for output, input in zip(gathered, inputs):
                    dist.all_gather_single(output, input)
            self.assertEqual(len(cm.works), 1 if async_ops else 0)
            cm.wait()
            for i, output in enumerate(gathered):
                expected = torch.arange(
                    i, self.world_size + i, dtype=output.dtype, device=self.device
                ).repeat_interleave(i + 1)
                self.assertEqual(output, expected)

        inputs = [
            torch.full(
                (self.world_size * (i + 1),),
                float(self.rank + i),
                device=self.device,
            )
            for i in range(3)
        ]
        for async_ops in (False, True):
            outputs = [torch.empty(i + 1, device=self.device) for i in range(3)]
            with dist._coalescing_manager(
                device=self.device, async_ops=async_ops
            ) as cm:
                for output, input in zip(outputs, inputs):
                    dist.reduce_scatter_single(output, input)
            self.assertEqual(len(cm.works), 1 if async_ops else 0)
            cm.wait()
            for i, output in enumerate(outputs):
                expected_value = self.rank_sum + i * self.world_size
                self.assertEqual(output, torch.full_like(output, expected_value))

    def test_float8_all_gather(self):
        if self.device_type != "cuda":
            self.skipTest("Float8 collectives require CUDA")
        if torch.cuda.get_device_capability(self.device) < (9, 0):
            self.skipTest("Float8 collectives require sm90 or newer")
        self._init_pg()
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            input = torch.full((16,), float(self.rank), device=self.device).to(dtype)
            output = torch.empty(
                self.world_size * input.numel(), dtype=dtype, device=self.device
            )
            dist.all_gather_single(output, input)
            expected = torch.arange(
                self.world_size, dtype=torch.float32, device=self.device
            ).repeat_interleave(input.numel())
            self.assertEqual(output.float(), expected)

    def test_noncontiguous_all_to_all_error(self):
        self._init_pg()
        input = torch.ones(self.world_size, self.world_size, device=self.device).t()
        output = torch.empty_like(input.contiguous())
        with self.assertRaisesRegex(ValueError, "Tensors must be contiguous"):
            dist.all_to_all_single(output, input)


instantiate_backend_tests(
    globals(), "Collectives", AbstractCollectivesTest, C10D_BACKENDS
)


if __name__ == "__main__":
    run_tests()
