# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from functools import wraps
from typing import Callable, Tuple, Any, Dict
import unittest
import torch
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    Shard,
)
from torch.distributed._tensor import _xla
from torch.testing._internal.common_utils import (
  TestCase,
  run_tests
)

# wrapper to check xla test requirements
def with_xla(func: Callable) -> Callable:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self, *args: Tuple[object], **kwargs: Dict[str, Any]  # type: ignore[misc]
    ) -> None:
        try:
          import torch_xla.core.xla_model as xm  # noqa
          from torch_xla.experimental.xla_sharded_tensor import XLAShardedTensor  # noqa
          from torch_xla.experimental.xla_sharding import mark_sharding, Mesh, ShardingType  # noqa
          import torch_xla.runtime as xr  # noqa
        except ImportError as exc:
          raise unittest.SkipTest('torch_xla is not installed.') from exc
        self.device_type = "xla"
        func(self, *args, **kwargs)  # type: ignore[misc]

    return wrapper


class DTensorXLAIntegrationTest(TestCase):
    @property
    def device_count(self) -> int:
        return 4

    @with_xla
    def test_mesh_conversion(self):
        # 1 x 4 device mesh with 2 axes.
        device_mesh = DeviceMesh('xla', list(list(range(self.device_count))),  # noqa: C414
                                 mesh_dim_names=('x', 'y'))
        self.assertEqual(device_mesh.size(), self.device_count)

        xla_mesh = _xla.convert_to_xla_mesh(device_mesh)
        self.assertEqual(xla_mesh.size(), device_mesh.size())
        self.assertEqual(xla_mesh.axis_names, device_mesh.mesh_dim_names)

    def test_placement_conversion(self):
        t = torch.ones(4,2)
        # partition input dim 1 over mesh dim 0
        partition_spec = _xla.convert_to_xla_partition_spec(t, [Shard(1)])
        self.assertEqual(partition_spec, (None, 0))

    @with_xla
    def test_xla_distribute_tensor(self):
        device_mesh = DeviceMesh('xla', list(range(self.world_size)))
        shard_spec = [Shard(0)]

        for requires_grad in [True, False]:
            tensor_to_shard = torch.randn(
                3 * self.world_size, 3, requires_grad=requires_grad
            )
            dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            # TODO(yeounoh) switch to DTensor API when XLAShardedTensor integrates
            global_tensor = dist_tensor.global_tensor
            self.assertEqual(global_tensor.size(), torch.Size([3 * self.world_size, 3]))
            local_tensor = dist_tensor.local_shards()[0].data.size()
            self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
            if requires_grad:
                self.assertTrue(dist_tensor.requires_grad)
                self.assertTrue(dist_tensor.is_leaf)


if __name__ == "__main__":
    run_tests()
