# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import os
import unittest
from functools import wraps
from typing import Any, Callable, Dict, Tuple

import numpy as np

import torch
from torch import nn
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# wrapper to check xla test requirements
def with_xla(func: Callable) -> Callable:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self, *args: Tuple[object], **kwargs: Dict[str, Any]  # type: ignore[misc]
    ) -> None:
        # TODO(yeounoh) replace this with xr.use_spmd() when we deprecate the flag.
        os.environ["XLA_USE_SPMD"] = "1"
        try:
            import torch_xla  # type:ignore[import]  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("torch_xla is not installed.") from exc
        self.device_type = "xla"
        func(self, *args, **kwargs)  # type: ignore[misc]
        os.environ["XLA_USE_SPMD"] = "0"

    return wrapper


class DTensorXLAIntegrationTest(TestCase):
    class SimpleLinear(nn.Module):
        def __init__(self) -> None:
            super(DTensorXLAIntegrationTest.SimpleLinear, self).__init__()
            self.fc1 = nn.Linear(128, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):
            y = self.relu(self.fc1(x))
            z = self.fc2(y)
            return z

    @with_xla
    def test_xla_distribute_tensor_1d_shard(self):
        import torch_xla.runtime as xr  # type:ignore[import]

        device_count = xr.global_runtime_device_count()
        if device_count > 1:
            device_mesh = DeviceMesh("xla", list(range(device_count)))
            shard_spec = [Shard(0)]

            for requires_grad in [True, False]:
                tensor_to_shard = torch.randn(
                    3 * device_count, 3, requires_grad=requires_grad
                )
                dist_tensor = distribute_tensor(
                    tensor_to_shard, device_mesh, shard_spec
                )
                # TODO(yeounoh) switch to DTensor API when XLAShardedTensor inherits DTensor
                assert type(dist_tensor).__name__ == "XLAShardedTensor"
                global_tensor = dist_tensor.global_tensor  # type:ignore[attr-defined]
                self.assertEqual(
                    global_tensor.size(), torch.Size([3 * device_count, 3])
                )
                local_tensor = dist_tensor.local_shards[0].data
                self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
                if requires_grad:
                    self.assertTrue(dist_tensor.global_tensor.requires_grad)
                    self.assertTrue(dist_tensor.is_leaf)

    @with_xla
    def test_xla_distribute_tensor_1d_replicate(self):
        import torch_xla.runtime as xr  # type:ignore[import]

        device_count = xr.global_runtime_device_count()
        device_mesh = DeviceMesh("xla", list(range(device_count)))
        shard_spec = [Replicate()]

        for requires_grad in [True, False]:
            tensor_to_shard = torch.randn(
                3 * device_count, 3, requires_grad=requires_grad
            )
            dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            # TODO(yeounoh) switch to DTensor API when XLAShardedTensor inherits DTensor
            assert type(dist_tensor).__name__ == "XLAShardedTensor"
            global_tensor = dist_tensor.global_tensor  # type:ignore[attr-defined]
            self.assertEqual(global_tensor.size(), torch.Size([3 * device_count, 3]))
            local_tensor = dist_tensor.local_shards[0].data
            self.assertEqual(local_tensor.size(), torch.Size([3 * device_count, 3]))
            if requires_grad:
                self.assertTrue(dist_tensor.global_tensor.requires_grad)
                self.assertTrue(dist_tensor.is_leaf)

    @with_xla
    def test_xla_distribute_tensor_2d(self):
        import torch_xla.runtime as xr  # type:ignore[import]

        device_count = xr.global_runtime_device_count()
        if device_count > 1:
            device_mesh = DeviceMesh(
                "xla", np.array(range(device_count)).reshape(2, device_count // 2)
            )
            shard_spec = [Replicate(), Shard(0)]

            for requires_grad in [True, False]:
                tensor_to_shard = torch.randn(
                    3 * device_count // 2, 3, requires_grad=requires_grad
                )
                dist_tensor = distribute_tensor(
                    tensor_to_shard, device_mesh, shard_spec
                )
                # TODO(yeounoh) switch to DTensor API when XLAShardedTensor inherits DTensor
                assert type(dist_tensor).__name__ == "XLAShardedTensor"
                global_tensor = dist_tensor.global_tensor  # type:ignore[attr-defined]
                self.assertEqual(
                    global_tensor.size(), torch.Size([3 * device_count // 2, 3])
                )
                local_tensor = dist_tensor.local_shards[0].data
                self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
                if requires_grad:
                    self.assertTrue(dist_tensor.global_tensor.requires_grad)
                    self.assertTrue(dist_tensor.is_leaf)

    @with_xla
    def text_xla_distribute_module(self):
        import torch_xla  # type:ignore[import]
        import torch_xla.core.xla_model as xm  # type:ignore[import]
        import torch_xla.runtime as xr  # type:ignore[import]

        model = self.SimpleLinear().to(xm.xla_device())

        device_count = xr.global_runtime_device_count()
        device_mesh = DeviceMesh("xla", list(range(device_count)))

        def shard_params(mod_name, mod, mesh):
            shard_spec = [Shard(0)]
            # annoate fc1 and fc2
            if isinstance(mod, nn.Linear):
                for _, param in mod.named_parameters():
                    # annotate the parameter tensors directly
                    distribute_tensor(param, mesh, shard_spec)

        sharded_model = distribute_module(model, device_mesh, shard_params)
        self.assertTrue(
            torch_xla._XLAC._get_xla_sharding_spec(sharded_model.fc1.weight) != ""
        )
        self.assertTrue(
            torch_xla._XLAC._get_xla_sharding_spec(sharded_model.fc2.weight) != ""
        )


if __name__ == "__main__":
    run_tests()
