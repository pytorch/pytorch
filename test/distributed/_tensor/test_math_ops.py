# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy
import itertools

import torch
from torch.distributed._tensor import DeviceMesh, distribute_module, distribute_tensor
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.tensor._ops.utils import is_tensor_partial, normalize_dim
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_unless_torch_gpu,
    with_comms,
)


funcol = torch.ops.c10d_functional


class DistMathOpsTest(DTensorTestBase):
    def linear_op_reductions(self, op_str):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        tensor = torch.randn(12, 8, 8)
        dtensor = distribute_tensor(tensor, device_mesh, shard_spec)

        op = getattr(tensor, op_str)
        op_dt = getattr(dtensor, op_str)

        keep_dim_or_not = [True, False, None]
        for dim in range(tensor.ndim):
            for keep_dim in keep_dim_or_not:
                args = (dim, keep_dim) if keep_dim is not None else (dim,)
                if op_str in ("max", "min"):
                    # min and max return a tuple when dim specified
                    dim_reduced_tensor, _ = op(*args)
                    dt_reduced, _ = op_dt(*args)
                else:
                    dim_reduced_tensor = op(*args)
                    dt_reduced = op_dt(*args)
                dt_dim_reduced_tensor = dt_reduced.full_tensor()
                self.assertEqual(dt_dim_reduced_tensor, dim_reduced_tensor)

        full_reduced_tensor = op()
        dt_full_reduced = op_dt().full_tensor()
        self.assertEqual(dt_full_reduced, full_reduced_tensor)

    @with_comms
    def test_linear_op_reductions(self):
        for op_str in ("all", "sum", "prod", "max", "min"):
            self.linear_op_reductions(op_str)

    @with_comms
    @skip_unless_torch_gpu
    def test_mean(self):
        self.linear_op_reductions("mean")

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
            dist_y = torch.nn.functional.softmax(
                dist_x, dim=softmax_dim, dtype=torch.float32
            )
            shard_dim = normalize_dim(shard_dim, dist_x.ndim)
            if dims[shard_dim] == dims[softmax_dim]:
                self.assertTrue(dist_y.placements[0].is_replicate())
                self.assertEqual(dist_y.to_local(), local_y)
            else:
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
            dist_softmax = dist_x.softmax(dim=softmax_dim)
            shard_dim = normalize_dim(shard_dim, dist_x.ndim)
            if dims[softmax_dim] == dims[shard_dim]:
                self.assertTrue(dist_softmax.placements[0].is_replicate())
            else:
                self.assertTrue(dist_softmax.placements[0].is_shard(dim=shard_dim))
            dist_y = dist_softmax.sum()
            if dims[softmax_dim] == dims[shard_dim]:
                self.assertTrue(dist_y.placements[0].is_replicate())
            else:
                self.assertTrue(dist_y.placements[0].is_partial())
                dist_y = dist_y.redistribute(device_mesh, [Replicate()])
            self.assertEqual(dist_y.to_local(), local_y)
            self.assertIsNone(dist_x.grad)
            dist_y.backward()
            self.assertIsNotNone(dist_x.grad)
            if dims[softmax_dim] == dims[shard_dim]:
                self.assertTrue(dist_x.grad.placements[0].is_replicate())
            else:
                self.assertTrue(dist_x.grad.placements[0].is_shard(dim=shard_dim))
            self.assertEqual(dist_x.grad.full_tensor(), x.grad)

    @with_comms
    @skip_unless_torch_gpu
    def test_nll_loss_and_cross_entropy(self):
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        channel_size, channel_dim = 16, 1
        test_setup = [
            (2, (8, channel_size), (8,)),  # calling aten.nll_loss_forward
            (3, (8, channel_size, 12), (8, 12)),  # calling aten.nll_loss2d_forward
        ]
        for input_ndim, input_size, target_size in test_setup:
            x = torch.rand(*input_size, device=self.device_type, requires_grad=True)
            target = torch.randint(channel_size, target_size, device=self.device_type)
            dist_target = distribute_tensor(target, device_mesh, [Replicate()])

            shard_dims = list(range(input_ndim))
            reductions = ["none", "mean", "sum"]
            # Compared with nll_loss, cross_entropy additionally calls log_softmax first.
            # Testing them together as code can be reused.
            loss_functions = [
                torch.nn.functional.nll_loss,
                torch.nn.functional.cross_entropy,
            ]
            for shard_dim, reduction, loss_fn in itertools.product(
                shard_dims, reductions, loss_functions
            ):
                dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
                y = loss_fn(x, target, reduction=reduction)
                if reduction == "none":
                    y.sum().backward()
                else:
                    y.backward()
                with comm_mode:
                    dist_y = loss_fn(dist_x, dist_target, reduction=reduction)
                    if shard_dim == channel_dim:
                        self.assertEqual(comm_mode.get_total_counts(), 1)
                        self.assertEqual(
                            comm_mode.get_comm_counts()[funcol.all_gather_into_tensor],
                            1,
                        )
                        self.assertTrue(dist_y.placements[0].is_replicate())
                        self.assertEqual(dist_y.to_local(), y)
                    else:
                        self.assertEqual(comm_mode.get_total_counts(), 0)
                        if reduction == "none":
                            output_shard_dim = (
                                shard_dim if shard_dim < channel_dim else shard_dim - 1
                            )
                            self.assertTrue(
                                dist_y.placements[0].is_shard(dim=output_shard_dim)
                            )
                        else:
                            self.assertTrue(dist_y.placements[0].is_partial())
                        self.assertEqual(dist_y.full_tensor(), y)

                    if reduction == "none":
                        dist_y.sum().backward()
                    else:
                        dist_y.backward()
                    if shard_dim == channel_dim:
                        self.assertTrue(dist_x.grad.placements[0].is_replicate())
                        self.assertEqual(dist_x.grad.to_local(), x.grad)
                    else:
                        self.assertTrue(
                            dist_x.grad.placements[0].is_shard(dim=shard_dim)
                        )
                        self.assertEqual(dist_x.grad.full_tensor(), x.grad)
                    x.grad.zero_()

    @with_comms
    def test_shard_math_ops(self):
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
            double_shard_full_tensor = op(double_shard_tensor, 2).full_tensor()
            self.assertEqual(double_shard_full_tensor, expect_rs)

            fully_shard_full_tensor = op(fully_shard_tensor, 2).full_tensor()
            self.assertEqual(fully_shard_full_tensor, expect_rs)

    @with_comms
    def test_layer_norm_fwd(self):
        device_mesh = self.build_device_mesh()

        # NLP example from pytorch docs
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        batch, sentence_length, embedding_dim = 20, 5, 10
        x = torch.rand(batch, sentence_length, embedding_dim, device=self.device_type)
        norm_shape_idx_list = list(range(x.ndim))
        shard_dims = [-1, 0, 1, 2]
        elementwise_affine_list = [False, True]
        test_config_list = list(
            itertools.product(shard_dims, norm_shape_idx_list, elementwise_affine_list)
        )

        # normalized shape is a torch.Size object
        for shard_dim, norm_idx, elementwise_affine in test_config_list:
            normalized_shape = x.shape[norm_idx:]
            layer_norm = torch.nn.LayerNorm(
                normalized_shape,
                elementwise_affine=elementwise_affine,
                device=self.device_type,
            )
            layer_norm_local = copy.deepcopy(layer_norm).to(self.device_type)

            def _replicate_fn(name, module, device_mesh):
                for name, param in module.named_parameters():
                    if name in ["weight", "bias"]:
                        param_dist = torch.nn.Parameter(
                            distribute_tensor(param, device_mesh, [Replicate()])
                        )
                        module.register_parameter(name, param_dist)

            layer_norm_dist = distribute_module(layer_norm, device_mesh, _replicate_fn)

            x_local = x
            x_dist = distribute_tensor(x, device_mesh, [Shard(shard_dim)])

            y_local = layer_norm_local(x_local)
            # make sure that forward layer norm does not introduce extra collectives
            comm_mode = CommDebugMode()
            with comm_mode:
                y_dist = layer_norm_dist(x_dist)

            self.assertLessEqual(
                comm_mode.get_total_counts(),
                1,  # TODO: This should be 0!
                f"comm count={comm_mode.get_total_counts()}, "
                f"shard_dim={shard_dim}, norm_shape={normalized_shape}, elem_affine={elementwise_affine}",
            )

            from torch.distributed._tensor.placement_types import TensorMeta

            dtensor_meta = y_dist._spec.tensor_meta
            assert isinstance(dtensor_meta, TensorMeta)
            # make sure the right shape in sharding prop
            self.assertEqual(y_local.shape, dtensor_meta.shape)
            self.assertEqual(y_local, y_dist.full_tensor())

    @with_comms
    def test_layer_norm_bwd(self):
        device_mesh = self.build_device_mesh()

        # NLP example from pytorch docs
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        batch, sentence_length, embedding_dim = 20, 5, 10
        norm_shape_idx_list = list(range(3))
        shard_dims = [0, 1, 2]
        elementwise_affine_list = [False, True]
        test_config_list = list(
            itertools.product(shard_dims, norm_shape_idx_list, elementwise_affine_list)
        )

        # normalized shape is a torch.Size object
        for shard_dim, norm_idx, elementwise_affine in test_config_list:
            x = torch.rand(
                batch,
                sentence_length,
                embedding_dim,
                device=self.device_type,
                requires_grad=True,
            )
            normalized_shape = x.shape[norm_idx:]
            layer_norm = torch.nn.LayerNorm(
                normalized_shape,
                elementwise_affine=elementwise_affine,
                device=self.device_type,
            )
            layer_norm_local = copy.deepcopy(layer_norm).to(self.device_type)

            def _replicate_fn(name, module, device_mesh):
                for name, param in module.named_parameters():
                    if name in ["weight", "bias"]:
                        param_dist = torch.nn.Parameter(
                            distribute_tensor(param, device_mesh, [Replicate()])
                        )
                        module.register_parameter(name, param_dist)

            layer_norm_dist = distribute_module(layer_norm, device_mesh, _replicate_fn)

            if elementwise_affine:
                self.assertEqual(
                    layer_norm_local.weight, layer_norm_dist.weight.full_tensor()
                )
                self.assertEqual(
                    layer_norm_local.bias, layer_norm_dist.bias.full_tensor()
                )

            x_local = x.detach().clone().requires_grad_(True)
            x_dist = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            self.assertEqual(x_local, x_dist.full_tensor())

            y_local = layer_norm_local(x_local)
            # make sure that backward layer norm does not introduce extra collectives
            comm_mode = CommDebugMode()
            with comm_mode:
                y_dist = layer_norm_dist(x_dist)
                y_dist.sum().backward()

            expected_fwd_comm = 0 if shard_dim < norm_idx else 1

            self.assertEqual(
                sum(comm_mode.comm_module_counts["Global"]["forward"].values()),
                expected_fwd_comm,
                f"comm count={comm_mode.get_total_counts()}, "
                f"shard_dim={shard_dim}, norm_shape={normalized_shape}, elem_affine={elementwise_affine}",
            )

            self.assertEqual(y_local, y_dist.full_tensor())

            # backward step
            y_local.sum().backward()

            expected_bwd_comm = 0 if shard_dim < norm_idx else 1

            self.assertEqual(
                sum(comm_mode.comm_module_counts["Global"]["backward"].values()),
                expected_bwd_comm,
                f"comm count={comm_mode.get_total_counts()}, "
                f"shard_dim={shard_dim}, norm_shape={normalized_shape}, elem_affine={elementwise_affine}",
            )

            if elementwise_affine:
                # if input is sharded on any outer dimension, the gradient of weight
                # and bias should be Partial
                dim_map = x_dist._spec.dim_map
                outer_dims = range(norm_idx)
                needs_reduction = any(dim_map[d] >= 0 for d in outer_dims)
                self.assertEqual(
                    is_tensor_partial(layer_norm_dist.weight.grad._spec),
                    needs_reduction,
                )
                self.assertEqual(
                    is_tensor_partial(layer_norm_dist.bias.grad._spec),
                    needs_reduction,
                )
                self.assertEqual(
                    layer_norm_local.weight.grad,
                    layer_norm_dist.weight.grad.full_tensor(),
                )
                self.assertEqual(
                    layer_norm_local.bias.grad,
                    layer_norm_dist.bias.grad.full_tensor(),
                )

            self.assertEqual(x_local.grad, x_dist.grad.full_tensor())

    @with_comms
    def test_topk(self):
        device_mesh = self.build_device_mesh()
        placement_combs = [Shard(0), Shard(1), Shard(2), Replicate()]

        comm_mode = CommDebugMode()

        tensor = torch.randn(12, 8, 8, requires_grad=True)
        global_topk = tensor.topk(3, dim=0)

        for placement in placement_combs:
            dtensor = distribute_tensor(tensor, device_mesh, (placement,))
            with comm_mode:
                out_dt = dtensor.topk(3, dim=0)
            if placement.is_shard(0):
                self.assertEqual(comm_mode.get_total_counts(), 1)
                self.assertEqual(
                    comm_mode.get_comm_counts()[funcol.all_gather_into_tensor],
                    1,
                )
            out_full_values = out_dt.values.full_tensor()
            self.assertEqual(global_topk.values, out_full_values)

            # TODO: support backward scatter
            # global_topk.values.sum().backward()
            # out_full_values.sum().backward()

    @with_comms
    def test_shard0_svd(self):
        device_mesh = self.build_device_mesh()
        torch.manual_seed(42)
        replicated_x = torch.randn((8, 8), device=self.device_type)
        sharded_x = distribute_tensor(replicated_x, device_mesh, (Shard(0),))
        with CommDebugMode() as comm_mode:
            U, S, V = torch.linalg.svd(sharded_x, full_matrices=False)
        ref_U, ref_S, ref_V = torch.linalg.svd(replicated_x, full_matrices=False)
        self.assertEqual(U.to_local(), ref_U)
        self.assertEqual(S.to_local(), ref_S)
        self.assertEqual(V.to_local(), ref_V)
        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(len(comm_counts), 1)
        self.assertEqual(comm_counts[funcol.all_gather_into_tensor], 1)

    @with_comms
    def test_foreach_norm(self):
        device_mesh = self.build_device_mesh()

        grad0 = torch.randn(12, 8)
        grad1 = torch.randn(8, 8)

        sharded_grad0 = distribute_tensor(grad0, device_mesh, [Shard(0)])
        sharded_grad1 = distribute_tensor(grad1, device_mesh, [Shard(0)])

        # non-sharded op
        out = torch.ops.aten._foreach_norm([grad0, grad1], 2)

        # sharded op
        sharded_out = torch.ops.aten._foreach_norm([sharded_grad0, sharded_grad1], 2)

        for o, so in zip(out, sharded_out):
            self.assertEqual(so.full_tensor(), o)


if __name__ == "__main__":
    run_tests()
