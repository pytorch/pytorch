# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools

import torch

from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_unless_torch_gpu,
    with_comms,
)


class DistMathOpsTest(DTensorTestBase):
    @with_comms
    def test_sum(self):
        device_mesh = self.build_device_mesh()

        shard_spec = [Shard(0)]

        tensor_to_sum = torch.randn(12, 8, 8)

        mat1 = distribute_tensor(tensor_to_sum, device_mesh, shard_spec)

        keep_dim_or_not = [True, False, None]
        for dim in range(tensor_to_sum.ndim):
            for keep_dim in keep_dim_or_not:
                sum_args = (dim, keep_dim) if keep_dim is not None else (dim,)
                dim_sumed_tensor = tensor_to_sum.sum(*sum_args)
                dt_dim_sumed_tensor = mat1.sum(*sum_args).full_tensor()
                self.assertEqual(dt_dim_sumed_tensor, dim_sumed_tensor)

        full_sumed_tensor = tensor_to_sum.sum()
        dt_sum = mat1.sum().full_tensor()
        self.assertEqual(dt_sum, full_sumed_tensor)

    # TODO: forward test can be removed once test_softmax_with_bwd passes on CPU
    @with_comms
    def test_softmax_fwd(self):
        device_mesh = self.build_device_mesh()

        x = torch.rand(8, 12, 16, device=self.device_type)
        dims = range(3)  # used to convert -1 to the actual dim
        softmax_dims = [-1, 0, 1, 2]
        shard_dims = [-1, 0, 1, 2]
        test_list = list(itertools.product(softmax_dims, shard_dims))

        for softmax_dim, shard_dim in test_list:
            local_y = torch.nn.functional.softmax(
                x, dim=softmax_dim, dtype=torch.float32
            )
            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            if dims[shard_dim] == dims[softmax_dim]:
                with self.assertRaisesRegex(
                    Exception, "Cannot run .* on sharding dimension!$"
                ):
                    dist_y = torch.nn.functional.softmax(
                        dist_x, dim=softmax_dim, dtype=torch.float32
                    )
            else:
                dist_y = torch.nn.functional.softmax(
                    dist_x, dim=softmax_dim, dtype=torch.float32
                )
                shard_dim = shard_dim + dist_y.ndim if shard_dim < 0 else shard_dim
                self.assertTrue(dist_y.placements[0].is_shard(dim=shard_dim))
                self.assertEqual(dist_y.full_tensor(), local_y)

    # TODO: get test_softmax_with_bwd pass on CPU
    # DTensor's _softmax_backward_data produces wrong result on CPU on certain dimension.
    # fail_on_cpu_list = [(0, -1), (1, -1)]
    @with_comms
    @skip_unless_torch_gpu
    def test_softmax_with_bwd(self):
        device_mesh = self.build_device_mesh()

        dims = range(3)  # used to convert -1 to the actual dim
        softmax_dims = [-1, 0, 1, 2]
        shard_dims = [-1, 0, 1, 2]
        test_list = list(itertools.product(softmax_dims, shard_dims))

        for params in test_list:
            softmax_dim, shard_dim = params
            x = torch.rand(8, 12, 16, device=self.device_type, requires_grad=True)
            self.assertTrue(x.requires_grad)
            local_y = torch.nn.functional.softmax(
                x, dim=softmax_dim, dtype=torch.float32
            ).sum()
            local_y.backward()

            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            self.assertTrue(dist_x.requires_grad)
            if dims[softmax_dim] == dims[shard_dim]:
                with self.assertRaisesRegex(
                    Exception, "Cannot run .* on sharding dimension!$"
                ):
                    dist_softmax = dist_x.softmax(dim=softmax_dim)
            else:
                dist_softmax = dist_x.softmax(dim=softmax_dim)
                shard_dim = shard_dim + dist_x.ndim if shard_dim < 0 else shard_dim
                self.assertTrue(dist_softmax.placements[0].is_shard(dim=shard_dim))
                dist_y = dist_softmax.sum()
                dist_y = dist_y.redistribute(device_mesh, [Replicate()])
                self.assertEqual(dist_y.to_local(), local_y)
                self.assertIsNone(dist_x.grad)
                dist_y.backward()
                self.assertIsNotNone(dist_x.grad)
                self.assertEqual(dist_x.grad.full_tensor(), x.grad)

    @with_comms
    def test_full_shard_math_ops(self):
        mesh_shape = (2, self.world_size // 2)
        mesh = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(*mesh_shape),
        )
        global_tensor = torch.ones(4, 4)
        double_shard_tensor = distribute_tensor(
            global_tensor, mesh, [Shard(0), Shard(0)]
        )
        fully_shard_tensor = distribute_tensor(
            global_tensor, mesh, [Shard(0), Shard(1)]
        )

        # for op in [torch.add, torch.sub, torch.mul, torch.div]:
        for op in [torch.add, torch.sub, torch.mul, torch.div]:
            expect_rs = op(global_tensor, 2)
            actual_rs = op(double_shard_tensor, 2).redistribute(
                mesh, [Replicate(), Replicate()]
            )
            actual_local_res = actual_rs.to_local()
            self.assertEqual(actual_local_res, expect_rs)


if __name__ == "__main__":
    run_tests()
