# Owner(s): ["oncall: distributed"]

import os
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


BACKENDS = [
    ("gloo", "cpu", False),
    ("nccl", "cuda", True),
    ("nccl2", "cuda", True),
]


class AbstractBackendContractTest:
    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        index = self.rank if self.device_type == "cuda" else None
        return torch.device(self.device_type, index)

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )

    def test_all_reduce_coalesced(self):
        self._init_pg()
        rank_sum = self.world_size * (self.world_size - 1) // 2
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
                    expected_value = rank_sum + (batch + i) * self.world_size
                    self.assertEqual(tensor, torch.full_like(tensor, expected_value))

    def test_coalescing_manager(self):
        if not self.supports_coalescing:
            self.skipTest(f"{self.backend_name} does not support coalescing")
        self._init_pg()
        rank_sum = self.world_size * (self.world_size - 1) // 2

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
                expected_value = rank_sum + i * self.world_size
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

        scatter_inputs = [
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
                for output, input in zip(outputs, scatter_inputs):
                    dist.reduce_scatter_single(output, input)
            self.assertEqual(len(cm.works), 1 if async_ops else 0)
            cm.wait()
            for i, output in enumerate(outputs):
                expected_value = rank_sum + i * self.world_size
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

    def test_sequence_numbers(self):
        self._init_pg()
        pg = dist.distributed_c10d._get_default_group()
        self.assertEqual(pg._get_sequence_number_for_group(), 0)
        dist.all_reduce(torch.ones(1, device=self.device))
        self.assertEqual(pg._get_sequence_number_for_group(), 1)

        subgroup = dist.new_group(list(range(self.world_size)))
        self.assertEqual(subgroup._get_sequence_number_for_group(), 0)
        dist.all_reduce(torch.ones(1, device=self.device), group=subgroup)
        self.assertEqual(subgroup._get_sequence_number_for_group(), 1)

    def test_singleton_subgroup_before_full_group(self):
        self._init_pg()
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
        self._init_pg()
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            tensor = torch.ones(1, device=self.device)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            work = dist.all_reduce(tensor, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            work.wait()
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)


def _make_backend_contract_test_class(
    backend_name, device_type, supports_coalescing
):
    class BackendContractTest(AbstractBackendContractTest, MultiProcessTestCase):
        pass

    BackendContractTest.backend_name = backend_name
    BackendContractTest.device_type = device_type
    BackendContractTest.supports_coalescing = supports_coalescing
    BackendContractTest.__name__ = f"{backend_name.capitalize()}BackendContractTest"
    BackendContractTest.__qualname__ = BackendContractTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(BackendContractTest)
    if device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 2,
            f"{backend_name} requires 2+ GPUs",
        )(cls)
    return cls


for backend_name, device_type, supports_coalescing in BACKENDS:
    globals()[f"{backend_name.capitalize()}BackendContractTest"] = (
        _make_backend_contract_test_class(
            backend_name, device_type, supports_coalescing
        )
    )


if __name__ == "__main__":
    run_tests()
