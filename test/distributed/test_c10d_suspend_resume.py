# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for the c10d memory offload API (Backend.suspend /
# resume / memory_stats). Parameterized over backends; today both entries are
# NCCL-based (the API needs communicator memory offload support, NCCL
# 2.29.7+), but other backends can be enabled by extending
# SUSPEND_RESUME_BACKENDS.

import os
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


SUSPEND_RESUME_BACKENDS = [
    ("nccl", "cuda"),
    ("nccl2", "cuda"),
]


def _has_nccl_offload():
    try:
        return dist.is_nccl_available() and torch.cuda.nccl.version() >= (2, 29, 7)
    except Exception:
        return False


class AbstractSuspendResumeTest:
    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        return torch.device(f"{self.device_type}:{self.rank}")

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
        self.backend = dist.get_backend_impl(device=self.device)
        # Run a large collective so the communicator allocates internal
        # (suspendable) memory.
        dist.all_reduce(torch.zeros(1024 * 1024 * 64, device=self.device))
        torch.cuda.synchronize()

    def test_memory_stats(self):
        self._init_pg()
        stats = self.backend.memory_stats()
        self.assertIsInstance(stats, dict)
        for key in ("suspend", "suspended", "persist", "total"):
            self.assertIn(key, stats)
        self.assertEqual(stats["suspended"], 0)

    def test_suspend(self):
        self._init_pg()
        self.backend.suspend()
        stats = self.backend.memory_stats()
        self.assertEqual(stats["suspended"], 1)

    def test_suspend_resume_cycle(self):
        self._init_pg()
        self.backend.suspend()
        self.backend.resume()
        stats = self.backend.memory_stats()
        self.assertEqual(stats["suspended"], 0)

        # The communicator must still work after a suspend/resume cycle.
        tensor = torch.ones(1024, device=self.device)
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        expected = torch.full((1024,), float(self.world_size), device=self.device)
        self.assertEqual(tensor, expected)


def _make_suspend_resume_test_class(backend_name, device_type):
    class SuspendResumeTest(AbstractSuspendResumeTest, MultiProcessTestCase):
        pass

    SuspendResumeTest.backend_name = backend_name
    SuspendResumeTest.device_type = device_type
    SuspendResumeTest.__name__ = f"{backend_name.capitalize()}SuspendResumeTest"
    SuspendResumeTest.__qualname__ = SuspendResumeTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(SuspendResumeTest)
    if device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 2 or not _has_nccl_offload(),
            "suspend/resume requires 2+ GPUs and NCCL 2.29.7+",
        )(cls)
    return cls


for backend_name, device_type in SUSPEND_RESUME_BACKENDS:
    globals()[f"{backend_name.capitalize()}SuspendResumeTest"] = (
        _make_suspend_resume_test_class(backend_name, device_type)
    )


if __name__ == "__main__":
    run_tests()
