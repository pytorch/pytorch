# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import itertools

import torch
from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed._tensor.experimental import register_sharding
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed._tensor._op_schema import RuntimeSchemaInfo


aten = torch.ops.aten


class TestRegisterSharding(DTensorTestBase):
    @with_comms
    def test_softmax_fwd(self):
        # After registering the custom softmax sharding strategy,
        # the original entry would have been replaced.
        # The following line is for showcasing purpose only.
        DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs.pop(aten._softmax.default, None)

        @register_sharding(aten._softmax.default)
        def custom_softmax_sharding(
            x: DTensorSpec,
            dim: int,
            half_to_float: torch.dtype,
        ):
            softmax_dim = dim if dim >= 0 else dim + x.ndim

            acceptable_shardings = []

            all_replicate = ([Replicate()], [Replicate(), None, None])
            acceptable_shardings.append(all_replicate)

            for sharding_dim in range(x.ndim):
                if sharding_dim != softmax_dim:
                    all_sharding = ([Shard(sharding_dim)], [Shard(sharding_dim), None, None])
                    acceptable_shardings.append(all_sharding)
            
            return acceptable_shardings

        device_mesh = self.build_device_mesh()

        x = torch.rand(8, 12, 16, device=self.device_type)
        dims = range(3)  # used to convert -1 to the actual dim
        softmax_dims = [-1, 0, 1]
        shard_dims = [0, 1, 2]
        test_list = list(itertools.product(softmax_dims, shard_dims))

        for softmax_dim, shard_dim in test_list:
            local_y = torch.nn.functional.softmax(
                x, dim=softmax_dim, dtype=torch.float32
            )
            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            dist_y = torch.nn.functional.softmax(
                dist_x, dim=softmax_dim, dtype=torch.float32
            )
            if dims[shard_dim] == dims[softmax_dim]:
                self.assertTrue(dist_y.placements[0].is_replicate())
                self.assertEqual(dist_y.to_local(), local_y)
            else:
                self.assertTrue(dist_y.placements[0].is_shard(dim=shard_dim))
                self.assertEqual(dist_y.full_tensor(), local_y)


if __name__ == "__main__":
    run_tests()
