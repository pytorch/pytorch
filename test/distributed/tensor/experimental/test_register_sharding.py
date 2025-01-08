# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import itertools

import torch
from torch.distributed._tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed._tensor.experimental import register_sharding
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


aten = torch.ops.aten


class TestRegisterSharding(DTensorTestBase):
    @with_comms
    def test_softmax_fwd(self):
        # After registering the custom softmax sharding strategy,
        # the original entry would have been replaced.
        # The following line is for showcasing purpose only.
        DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs.pop(
            aten._softmax.default, None
        )

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
                    all_sharded = (
                        [Shard(sharding_dim)],
                        [Shard(sharding_dim), None, None],
                    )
                    acceptable_shardings.append(all_sharded)

            return acceptable_shardings

        # check if the RuntimeSchemaInfo is derived correctly
        schema_info = DTensor._op_dispatcher.sharding_propagator.op_to_schema_info[
            aten._softmax.default
        ]
        self.assertEqual(schema_info.static_argnum, 1)

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

    @with_comms
    def test_argmax(self):
        @register_sharding(aten.argmax.default)
        def custom_argmax_sharding(x, dim, keepdim):
            acceptable_shardings = []

            all_replicate = ([Replicate()], [Replicate(), None, None])
            acceptable_shardings.append(all_replicate)

            if keepdim:
                for sharding_dim in range(x.ndim):
                    if sharding_dim != dim:
                        all_sharded = (
                            [Shard(sharding_dim)],
                            [Shard(sharding_dim), None, None],
                        )
                        acceptable_shardings.append(all_sharded)

            return acceptable_shardings

        # check if the RuntimeSchemaInfo is derived correctly
        # when the first int arg is optional
        schema_info = DTensor._op_dispatcher.sharding_propagator.op_to_schema_info[
            aten.argmax.default
        ]
        self.assertEqual(schema_info.static_argnum, 1)

        device_mesh = self.build_device_mesh()

        x = torch.rand(8, 12, device=self.device_type)
        dist_x = distribute_tensor(x, device_mesh, [Shard(0)])

        local_y = torch.argmax(x, dim=1, keepdim=True)
        dist_y = torch.argmax(dist_x, dim=1, keepdim=True)

        self.assertTrue(dist_y.placements[0].is_shard(dim=0))
        self.assertEqual(dist_y.full_tensor(), local_y)


if __name__ == "__main__":
    run_tests()
