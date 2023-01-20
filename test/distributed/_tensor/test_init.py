# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    Shard,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class DTensorInitOpsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _run_init_op(self, init_op, *args, **kwargs):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        input_size = (8, 4)
        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        local_tensor_clone = torch.clone(input_tensor)
        torch.manual_seed(self.rank)
        local_tensor_clone = init_op(local_tensor_clone, *args, **kwargs)
        torch.manual_seed(self.rank)
        dtensor = init_op(dtensor, *args, **kwargs)
        dtensor_clone = DTensor.from_local(local_tensor_clone, device_mesh, shard_spec)
        self.assertEqual(dtensor_clone.to_local(), dtensor.to_local())

    @with_comms
    def test_init_ops(self):
        self._run_init_op(
            torch.nn.init.kaiming_uniform_,
            a=0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        self._run_init_op(torch.nn.init.normal_, mean=1.5, std=0.8)
        self._run_init_op(torch.nn.init.uniform_, a=0, b=1.2)
        self._run_init_op(torch.nn.init.constant_, 2.4)


if __name__ == "__main__":
    run_tests()
