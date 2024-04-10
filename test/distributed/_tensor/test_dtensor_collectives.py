# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed._functional_collectives as ft_c

from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed._tensor.placement_types import Replicate, Shard

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


@instantiate_parametrized_tests
class TestDTensorCollectives(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_all_to_all_single_sharded(self) -> None:
        mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])
        placement = [Shard(2), Replicate()]
        BS = 2
        seq_len = 8
        dim = 16
        tensor = torch.ones(
            BS, seq_len, dim, device=self.device_type, requires_grad=True
        )
        dtensor = distribute_tensor(tensor, mesh, placement)
        self.assertEqual(dtensor.to_local().shape, (BS, seq_len, dim // 2))

        group = mesh.get_group(1)
        rank = group.rank()

        # this is a "ring" but with only 2 workers it's just a swap
        output_sizes = [0, len(dtensor)] if rank == 0 else [len(dtensor), 0]
        input_sizes = [0, len(dtensor)] if rank == 0 else [len(dtensor), 0]

        out = ft_c.all_to_all_single_autograd(dtensor, output_sizes, input_sizes, group)
        out = out.wait()
        self.assertEqual(out.to_local().shape, (BS, seq_len, dim // 2))
        out.sum().backward()
        self.assertIsNotNone(dtensor.grad)


if __name__ == "__main__":
    run_tests()
