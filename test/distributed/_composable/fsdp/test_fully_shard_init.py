# Owner(s): ["oncall: distributed"]

import unittest
from typing import List

import torch
import torch.nn as nn
from _test_fully_shard_common import MLP
from torch.distributed._composable import replicate

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._composable.fsdp._fsdp_init import (
    _get_managed_modules,
    _get_managed_states,
    _normalize_device,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_fsdp import FSDPTestMultiThread
from torch.testing._internal.common_utils import run_tests


class TestFullyShardInitDevice(FSDPTestMultiThread):
    """Tests the ``device`` argument."""

    @property
    def world_size(self) -> int:
        return 1

    def test_normalize_device_cpu(self):
        for device in ("cpu", torch.device("cpu")):
            self.assertEqual(_normalize_device(device), torch.device("cpu"))

    def test_normalize_device_meta(self):
        for device in ("meta", torch.device("meta")):
            self.assertEqual(_normalize_device(device), torch.device("meta"))

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_normalize_device_cuda(self):
        for device in (
            "cuda",
            0,
            "cuda:0",
            torch.device("cuda"),
            torch.device("cuda", 0),
        ):
            self.assertEqual(_normalize_device(device), torch.device("cuda", 0))
        if torch.cuda.device_count() > 1:
            with torch.cuda.device(1):
                for device in ("cuda", torch.device("cuda")):
                    self.assertEqual(_normalize_device(device), torch.device("cuda", 1))


class TestFullyShardInitMesh(FSDPTestMultiThread):
    """Tests the ``mesh`` argument."""

    @property
    def world_size(self) -> int:
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_invalid_mesh_ndim(self):
        mesh = init_device_mesh("cuda", (self.world_size, 1, 1))
        model = MLP(8)
        regex = r"fully\_shard expects a 1D or 2D DeviceMesh but got DeviceMesh\(\[\[\[0\]\], \[\[1\]\]\]\)"
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model, mesh=mesh, device="cuda")

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_mesh_device_mismatch(self):
        mesh = init_device_mesh("cuda", (self.world_size,))
        model = MLP(8)
        regex = "device and mesh must be of the same type but got cpu for device and cuda for mesh"
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model, mesh=mesh, device="cpu")


class TestFullyShardInitManagedModules(FSDPTestMultiThread):
    """Tests getting the managed modules for a ``fully_shard`` module."""

    @property
    def world_size(self) -> int:
        return 1

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_managed_modules_single_fully_shard(self):
        model = MLP(8)
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules(model)
        expected_managed_modules = list(model.modules())
        self._check_managed_modules(managed_modules, expected_managed_modules)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_managed_modules_nested_fully_shard(self):
        model = nn.Sequential(*[MLP(8) for _ in range(2)])
        fully_shard(model[0])
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules(model)
        expected_managed_modules = list(model[1].modules()) + [model]
        self._check_managed_modules(managed_modules, expected_managed_modules)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_managed_modules_nested_fully_shard_and_replicate(self):
        model = nn.Sequential(*[MLP(8) for _ in range(3)])
        replicate(model[0])
        fully_shard(model[2])
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules(model)
        expected_managed_modules = list(model[1].modules()) + [model]
        self._check_managed_modules(managed_modules, expected_managed_modules)

    def _check_managed_modules(
        self,
        managed_modules: List[nn.Module],
        expected_managed_modules: List[nn.Module],
    ):
        self.assertEqual(len(managed_modules), len(expected_managed_modules))
        # Check set comparison since we do not require anything about the order
        self.assertEqual(set(managed_modules), set(expected_managed_modules))

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_managed_states_shared_params(self):
        model = nn.Sequential(*[MLP(8, with_buffer=True) for _ in range(3)])
        model[0].in_proj.weight = model[1].in_proj.weight
        model[2].in_proj.weight = model[1].in_proj.weight
        model[1].buffer = model[2].buffer
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules(model)
        params, buffers = _get_managed_states(managed_modules)
        expected_params = list(model.parameters())  # de-dups shared
        expected_buffers = list(model.buffers())  # de-dups shared
        self._check_managed_states(params, buffers, expected_params, expected_buffers)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_managed_states_nested_fully_shard(self):
        model = nn.Sequential(*[MLP(8, with_buffer=True) for _ in range(2)])
        fully_shard(model[0])
        # Assume calling `fully_shard` on `model`
        managed_modules = _get_managed_modules(model)
        params, buffers = _get_managed_states(managed_modules)
        expected_params = list(model[1].parameters())
        expected_buffers = list(model[1].buffers())
        self._check_managed_states(params, buffers, expected_params, expected_buffers)

    def _check_managed_states(
        self,
        managed_params: List[nn.Parameter],
        managed_buffers: List[torch.Tensor],
        expected_managed_params: List[nn.Parameter],
        expected_managed_buffers: List[torch.Tensor],
    ):
        self.assertEqual(len(managed_params), len(expected_managed_params))
        self.assertEqual(len(managed_buffers), len(expected_managed_buffers))
        self.assertEqual(set(managed_params), set(expected_managed_params))
        self.assertEqual(set(managed_buffers), set(expected_managed_buffers))


if __name__ == "__main__":
    run_tests()
