# Owner(s): ["oncall: distributed"]

import functools
import os
import sys
import unittest

import torch
import torch.distributed as dist


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.distributed import config as dist_config
from torch.distributed.distributed_c10d import _TORCHCOMM_AVAILABLE
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests


def _with_torchcomm_env(func):
    """Sets TORCHCOMM env vars needed by torchcomms before init_pg runs."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        os.environ["TORCHCOMM_RANK"] = str(self.rank)
        os.environ["TORCHCOMM_SIZE"] = str(self.world_size)
        os.environ["TORCHCOMM_STORE_PATH"] = self.file_name
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        try:
            return func(self, *args, **kwargs)
        finally:
            os.environ.pop("TORCHCOMM_RANK", None)
            os.environ.pop("TORCHCOMM_SIZE", None)
            os.environ.pop("TORCHCOMM_STORE_PATH", None)
            os.environ.pop("MASTER_ADDR", None)
            os.environ.pop("MASTER_PORT", None)

    return wrapper


class SplitGroupTestBase(MultiProcessTestCase):
    """Base class with common setup, teardown, and helpers for split_group tests."""

    def setUp(self):
        super().setUp()
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    _use_device_id = True

    def _init_pg(self, backend):
        """Initialize default process group with the given backend string."""
        store = dist.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device if self._use_device_id else None,
        )
        return device

    def _verify_gpu_allreduce(self, group, device, expected_world_size):
        """Verify all_reduce works on GPU for a process group."""
        gpu_tensor = torch.ones(10, device=device)
        dist.all_reduce(gpu_tensor, group=group)
        self.assertTrue(torch.all(gpu_tensor == expected_world_size))

    def _verify_cpu_allreduce(self, group, expected_world_size):
        """Verify all_reduce works on CPU for a process group."""
        gpu_tensor = torch.ones(10)
        dist.all_reduce(gpu_tensor, group=group)
        self.assertTrue(torch.all(gpu_tensor == expected_world_size))

    def _verify_allreduce_on_split(self, group, device, expected_world_size):
        """Verify all_reduce works on both GPU and CPU for a process group."""
        self._verify_gpu_allreduce(group, device, expected_world_size)
        self._verify_cpu_allreduce(group, expected_world_size)

    def _split_into_halves(self):
        """Return split_ranks that divide world into two equal halves."""
        split_size = self.world_size // 2
        return [
            list(range(i * split_size, (i + 1) * split_size))
            for i in range(self.world_size // split_size)
        ]

    def _test_split_group_mixed_backend(self):
        """Test split_group with cpu:gloo,cuda:nccl splits both backends on default pg."""
        device = self._init_pg("cpu:gloo,cuda:nccl")

        all_ranks = list(range(self.world_size))
        split_pg = dist.split_group(split_ranks=[all_ranks])

        self._verify_allreduce_on_split(split_pg, device, self.world_size)

        dist.destroy_process_group()

    def _test_split_group_mixed_backend_subgroups(self):
        """Test split_group with mixed backend creates correct subgroups on default pg."""
        device = self._init_pg("cpu:gloo,cuda:nccl")

        split_ranks = self._split_into_halves()
        split_size = self.world_size // 2
        split_pg = dist.split_group(split_ranks=split_ranks)

        self.assertEqual(split_pg.size(), split_size)
        self._verify_allreduce_on_split(split_pg, device, split_size)

        dist.destroy_process_group()


class SplitGroupNativeBackendTest(SplitGroupTestBase):
    """Tests for split_group using native NCCL/Gloo backends (no torchcomms)."""

    @property
    def world_size(self):
        return 4

    @skip_if_lt_x_gpu(4)
    def test_split_group_mixed_backend(self):
        self._test_split_group_mixed_backend()

    @skip_if_lt_x_gpu(4)
    def test_split_group_mixed_backend_subgroups(self):
        self._test_split_group_mixed_backend_subgroups()


@unittest.skipIf(
    not _TORCHCOMM_AVAILABLE, "TorchComms is not installed, skipping torchcomms tests"
)
class SplitGroupBackendWrapperTest(SplitGroupTestBase):
    """Tests for split_group via BackendWrapper (torchcomms-backed process groups)."""

    _use_device_id = False

    @property
    def world_size(self):
        return 4

    @skip_if_lt_x_gpu(4)
    @dist_config.patch(use_torchcomms=True)
    @_with_torchcomm_env
    def test_split_group_mixed_backend(self):
        self._test_split_group_mixed_backend()

    @skip_if_lt_x_gpu(4)
    @dist_config.patch(use_torchcomms=True)
    @_with_torchcomm_env
    def test_split_group_mixed_backend_subgroups(self):
        self._test_split_group_mixed_backend_subgroups()


if __name__ == "__main__":
    run_tests()
