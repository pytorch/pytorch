# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools
import unittest
from typing import cast, Optional

import torch
import torch.nn.functional as F
from torch.distributed import init_device_mesh
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor._ops._matrix_ops import (
    gen_single_dim_einsum_strategies,
    mm_single_dim_strategy,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    register_single_dim_strategy,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8, SM90OrLater
from torch.testing._internal.common_device_type import E4M3_MAX_POS, e4m3_type
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_ROCM,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    skip_unless_torch_gpu,
    with_comms,
)


funcol = torch.ops.c10d_functional


def scale_for_fp8(
    t: torch.Tensor, scale_shape: tuple[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    if all(d == 1 for d in scale_shape):
        t = t.unsqueeze(0).unsqueeze(-2)
    else:
        t = t.unflatten(0, (scale_shape[0], -1)).unflatten(-1, (scale_shape[1], -1))

    scale = t.abs().amax(dim=[1, -1]).float() / E4M3_MAX_POS
    t_fp8 = (t / scale[:, None, :, None]).to(e4m3_type)

    return t_fp8.flatten(end_dim=1).flatten(start_dim=-2), scale.view(scale_shape)


class DistMatrixOpsTest(DTensorTestBase):
    @with_comms
    def test_addmm(self):
        """
        Test addmm with all sharding strategies from addmm_single_dim_strategy.

        The single dim strategy generates these cases for addmm(bias, mat1, mat2):
        - Contracting dim k: mat1=Shard(1), mat2=Shard(0) -> output=Partial
        - LHS free dim m: mat1=Shard(0), mat2=Replicate -> output=Shard(0)
        - RHS free dim n: mat1=Replicate, mat2=Shard(1) -> output=Shard(1)

        The bias placement depends on output placement and broadcast dims.
        """
        device_mesh = self.build_device_mesh()
        M, K, N = 12, 8, 4  # mat1: (M, K), mat2: (K, N), output: (M, N)

        mat1_tensor = torch.randn(M, K)
        mat2_tensor = torch.randn(K, N)
        bias_1d = torch.randn(N)  # 1D bias, broadcasts on M dim
        bias_2d = torch.randn(M, N)  # 2D bias, no broadcast

        local_res_1d = torch.addmm(bias_1d, mat1_tensor, mat2_tensor)
        local_res_2d = torch.addmm(bias_2d, mat1_tensor, mat2_tensor)

        # Case 1: LHS free dim m - mat1=Shard(0), mat2=Replicate -> output=Shard(0)
        # With 1D bias: bias should be Replicate (broadcast on m dim)
        mat1_s0 = distribute_tensor(mat1_tensor, device_mesh, [Shard(0)])
        mat2_r = distribute_tensor(mat2_tensor, device_mesh, [Replicate()])
        bias_1d_r = distribute_tensor(bias_1d, device_mesh, [Replicate()])

        dist_res = torch.addmm(bias_1d_r, mat1_s0, mat2_r)
        self.assertEqual(dist_res.full_tensor(), local_res_1d)
        self.assertEqual(dist_res.placements[0], Shard(0))

        # Case 1b: LHS free dim m with 2D bias - bias should be Shard(0)
        bias_2d_s0 = distribute_tensor(bias_2d, device_mesh, [Shard(0)])
        dist_res = torch.addmm(bias_2d_s0, mat1_s0, mat2_r)
        self.assertEqual(dist_res.full_tensor(), local_res_2d)
        self.assertEqual(dist_res.placements[0], Shard(0))

        # Case 2: RHS free dim n - mat1=Replicate, mat2=Shard(1) -> output=Shard(1)
        # With 1D bias: bias should be Shard(0) (its dim 0 corresponds to n)
        mat1_r = distribute_tensor(mat1_tensor, device_mesh, [Replicate()])
        mat2_s1 = distribute_tensor(mat2_tensor, device_mesh, [Shard(1)])
        bias_1d_s0 = distribute_tensor(bias_1d, device_mesh, [Shard(0)])

        dist_res = torch.addmm(bias_1d_s0, mat1_r, mat2_s1)
        self.assertEqual(dist_res.full_tensor(), local_res_1d)
        self.assertEqual(dist_res.placements[0], Shard(1))

        # Case 2b: RHS free dim n with 2D bias - bias should be Shard(1)
        bias_2d_s1 = distribute_tensor(bias_2d, device_mesh, [Shard(1)])
        dist_res = torch.addmm(bias_2d_s1, mat1_r, mat2_s1)
        self.assertEqual(dist_res.full_tensor(), local_res_2d)
        self.assertEqual(dist_res.placements[0], Shard(1))

        # Case 3: Contracting dim k - mat1=Shard(1), mat2=Shard(0) -> output=Partial
        # bias should be Partial
        mat1_s1 = distribute_tensor(mat1_tensor, device_mesh, [Shard(1)])
        mat2_s0 = distribute_tensor(mat2_tensor, device_mesh, [Shard(0)])
        bias_1d_p = distribute_tensor(bias_1d, device_mesh, [Partial()])

        dist_res = torch.addmm(bias_1d_p, mat1_s1, mat2_s0)
        self.assertIsInstance(dist_res.placements[0], Partial)
        self.assertEqual(dist_res.full_tensor(), local_res_1d)

        # Case 3b: Contracting dim k with 2D bias - bias should be Partial
        bias_2d_p = distribute_tensor(bias_2d, device_mesh, [Partial()])
        dist_res = torch.addmm(bias_2d_p, mat1_s1, mat2_s0)
        self.assertIsInstance(dist_res.placements[0], Partial)
        self.assertEqual(dist_res.full_tensor(), local_res_2d)

        # Case 4: All-Replicate case
        mat1_r = distribute_tensor(mat1_tensor, device_mesh, [Replicate()])
        mat2_r = distribute_tensor(mat2_tensor, device_mesh, [Replicate()])
        bias_1d_r = distribute_tensor(bias_1d, device_mesh, [Replicate()])
        bias_2d_r = distribute_tensor(bias_2d, device_mesh, [Replicate()])

        dist_res = torch.addmm(bias_1d_r, mat1_r, mat2_r)
        self.assertEqual(dist_res.full_tensor(), local_res_1d)
        self.assertEqual(dist_res.placements[0], Replicate())

        dist_res = torch.addmm(bias_2d_r, mat1_r, mat2_r)
        self.assertEqual(dist_res.full_tensor(), local_res_2d)
        self.assertEqual(dist_res.placements[0], Replicate())

        # Case 5: Scalar bias - broadcasts on all dims
        bias_scalar = torch.randn(())
        local_res_scalar = torch.addmm(bias_scalar, mat1_tensor, mat2_tensor)

        # Scalar with all strategies - should always be Replicate
        bias_scalar_r = distribute_tensor(bias_scalar, device_mesh, [Replicate()])

        dist_res = torch.addmm(bias_scalar_r, mat1_s0, mat2_r)
        self.assertEqual(dist_res.full_tensor(), local_res_scalar)
        self.assertEqual(dist_res.placements[0], Shard(0))

        dist_res = torch.addmm(bias_scalar_r, mat1_r, mat2_s1)
        self.assertEqual(dist_res.full_tensor(), local_res_scalar)
        self.assertEqual(dist_res.placements[0], Shard(1))

        # Case 6: (1, N) bias - broadcasts on M dim, similar to 1D
        bias_1n = torch.randn(1, N)
        local_res_1n = torch.addmm(bias_1n, mat1_tensor, mat2_tensor)

        # With LHS sharding: output=Shard(0), bias broadcasts on M so bias=Replicate
        bias_1n_r = distribute_tensor(bias_1n, device_mesh, [Replicate()])
        dist_res = torch.addmm(bias_1n_r, mat1_s0, mat2_r)
        self.assertEqual(dist_res.full_tensor(), local_res_1n)
        self.assertEqual(dist_res.placements[0], Shard(0))

        # With RHS sharding: output=Shard(1), bias dim 1 corresponds to N
        bias_1n_s1 = distribute_tensor(bias_1n, device_mesh, [Shard(1)])
        dist_res = torch.addmm(bias_1n_s1, mat1_r, mat2_s1)
        self.assertEqual(dist_res.full_tensor(), local_res_1n)
        self.assertEqual(dist_res.placements[0], Shard(1))

        # Case 7: (M, 1) bias - broadcasts on N dim
        bias_m1 = torch.randn(M, 1)
        local_res_m1 = torch.addmm(bias_m1, mat1_tensor, mat2_tensor)

        # With LHS sharding: output=Shard(0), bias dim 0 corresponds to M
        bias_m1_s0 = distribute_tensor(bias_m1, device_mesh, [Shard(0)])
        dist_res = torch.addmm(bias_m1_s0, mat1_s0, mat2_r)
        self.assertEqual(dist_res.full_tensor(), local_res_m1)
        self.assertEqual(dist_res.placements[0], Shard(0))

        # With RHS sharding: output=Shard(1), bias broadcasts on N so bias=Replicate
        bias_m1_r = distribute_tensor(bias_m1, device_mesh, [Replicate()])
        dist_res = torch.addmm(bias_m1_r, mat1_r, mat2_s1)
        self.assertEqual(dist_res.full_tensor(), local_res_m1)
        self.assertEqual(dist_res.placements[0], Shard(1))

    @with_comms
    def test_addmm_empty_operand(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_shard = torch.randn(12, 0)
        mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        tensor_to_replicate = torch.randn(0, 4)
        mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)
        input_tensor = torch.randn(4)
        inp = distribute_tensor(input_tensor, device_mesh, replica_spec)

        dist_res = torch.addmm(inp, mat1, mat2)
        local_res = torch.addmm(input_tensor, tensor_to_shard, tensor_to_replicate)
        self.assertEqual(dist_res.full_tensor(), local_res)

    @with_comms
    def test_addmm_auto_redistribute(self):
        device_mesh = self.build_device_mesh()
        shard0_spec = [Shard(0)]
        shard1_spec = [Shard(1)]
        replica_spec = [Replicate()]

        tensor_to_shard1 = torch.randn(12, 8, requires_grad=True)
        mat1 = distribute_tensor(tensor_to_shard1, device_mesh, shard1_spec)
        tensor_to_shard0 = torch.randn(8, 4, requires_grad=True)
        mat2 = distribute_tensor(tensor_to_shard0, device_mesh, shard0_spec)
        input_tensor = torch.randn(4, requires_grad=True)
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)

        local_res = torch.addmm(input_tensor, tensor_to_shard1, tensor_to_shard0)
        dist_res = torch.addmm(input, mat1, mat2)

        # test if addmm output is a partial
        self.assertIsInstance(dist_res, DTensor)
        self.assertIsInstance(dist_res.placements[0], Partial)

        # test if result is the same as tensor
        dist_local_res = dist_res.full_tensor()
        self.assertEqual(local_res, dist_local_res)

        # backward checks
        dist_local_res.sum().backward()
        local_res.sum().backward()
        self.assertIsNotNone(mat2.grad)
        self.assertEqual(mat2.grad.full_tensor(), tensor_to_shard0.grad)

    def test_gen_single_dim_einsum_strategies_bias_reduce_op(self):
        """Test that bias Partial placements preserve reduce_op from output Partial."""
        # Test addmm strategy: "mk,kn->mn" with bias
        # For contracting dim k: output=Partial, bias should also be Partial with same reduce_op
        bias_shape_1d = torch.Size([4])  # 1D bias
        bias_shape_2d = torch.Size([12, 4])  # 2D bias

        strategies_1d = gen_single_dim_einsum_strategies(
            "mk,kn->mn", bias_shape=bias_shape_1d
        )
        strategies_2d = gen_single_dim_einsum_strategies(
            "mk,kn->mn", bias_shape=bias_shape_2d
        )

        # Find strategies where output is Partial (contracting dim case)
        # Strategy format: [output, bias, mat1, mat2]
        for strategies, bias_shape in [
            (strategies_1d, bias_shape_1d),
            (strategies_2d, bias_shape_2d),
        ]:
            for strategy in strategies:
                output_placement = strategy[0]
                bias_placement = strategy[1]

                if isinstance(output_placement, Partial):
                    # Bug: _derive_bias_placement was returning Partial() without
                    # preserving reduce_op from output_placement
                    self.assertIsInstance(bias_placement, Partial)
                    self.assertEqual(
                        bias_placement.reduce_op,
                        output_placement.reduce_op,
                        f"Bias Partial should have same reduce_op as output Partial. "
                        f"Got bias={bias_placement.reduce_op}, output={output_placement.reduce_op}",
                    )

    @with_comms
    def test_mm(self):
        device_mesh = self.build_device_mesh()
        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        replica_spec = Replicate()

        t1 = torch.randn(12, 8, requires_grad=True)
        t2 = torch.randn(8, 16, requires_grad=True)
        local_res = torch.mm(t1, t2)

        def test_placement_comb(
            placements1: list[Placement], placements2: list[Placement]
        ) -> None:
            dt1 = distribute_tensor(t1, device_mesh, placements1)
            dt2 = distribute_tensor(t2, device_mesh, placements2)
            dist_res: DTensor = cast(DTensor, torch.mm(dt1, dt2)).redistribute(
                device_mesh, [replica_spec]
            )
            self.assertEqual(dist_res.to_local(), local_res)
            # backward
            grad_dist_res = torch.ones_like(dist_res)
            dist_res.backward(grad_dist_res)
            self.assertIsNotNone(dt1.grad)

        placement_specs = [shard0_spec, shard1_spec, replica_spec]
        shard_specs_comb = list(itertools.product(placement_specs, placement_specs))
        for spec in shard_specs_comb:
            test_placement_comb([spec[0]], [spec[1]])

    @with_comms
    def test_mm_single_dim_strategy(self):
        register_single_dim_strategy(torch.ops.aten.mm.default)(mm_single_dim_strategy)
        # unshardable input where some rank have empty _local_tensor
        # eg sharding tensor (world_size - 1) over world_size
        device_mesh = self.build_device_mesh()
        global_inps_viewed = (
            torch.arange((self.world_size - 1) * self.world_size)
            .float()
            .view(self.world_size - 1, self.world_size)
        )
        inps_viewed = distribute_tensor(
            global_inps_viewed,
            device_mesh,
            (Shard(dim=0),),
        )
        global_weight = (
            torch.arange(self.world_size * self.world_size)
            .float()
            .view(self.world_size, self.world_size)
        )
        weight = distribute_tensor(global_weight, device_mesh, (Replicate(),))
        out = torch.mm(inps_viewed, weight)
        expected_placements = (Replicate(),)
        self.assertEqual(out.placements, expected_placements)

    @with_comms
    @skip_unless_torch_gpu
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    def test_scaled_mm(self):
        device_mesh = self.build_device_mesh()
        shrd0 = Shard(0)
        shrd1 = Shard(1)
        repl = Replicate()
        part = Partial()

        ws = self.world_size
        # _scaled_mm requires all dimensions to be multiples of 16. Since we'll
        # shard along n and k, we need to ensure this stays true on each rank.
        m, n, k = 16, 32 * ws, 16 * ws

        t1 = torch.randn(m, k, device=self.device_type, dtype=torch.bfloat16)
        t2 = torch.randn(n, k, device=self.device_type, dtype=torch.bfloat16)

        for (
            output_spec,
            t1_spec,
            t2_spec,
            scale1_shape,
            scale2_shape,
            scale1_spec,
            scale2_spec,
        ) in [
            # Tensor-wise scaling
            # Replicated, zero-dim scale
            (repl, repl, repl, (), (), repl, repl),
            # Column-parallel, two-dim scale
            (shrd1, repl, shrd0, (1, 1), (1, 1), repl, repl),
            # Row-parallel, one-dim scale
            (part, shrd1, shrd1, (1,), (1,), repl, repl),
            # Row-wise scaling
            # Replicated
            (repl, repl, repl, (m, 1), (n, 1), repl, repl),
            # Column-parallel
            (shrd1, repl, shrd0, (m, 1), (n, 1), repl, shrd0),
            # Row-parallel (which actually ends up doing sub-row-wise scaling)
            (part, shrd1, shrd1, (m, ws), (n, ws), shrd1, shrd1),
        ]:
            full_ref_res = t1 @ t2.t()

            t1_fp8, scale1 = scale_for_fp8(t1, scale1_shape)
            t2_fp8, scale2 = scale_for_fp8(t2, scale2_shape)

            dist_t1_fp8 = distribute_tensor(t1_fp8, device_mesh, [t1_spec])
            dist_t2_fp8 = distribute_tensor(t2_fp8, device_mesh, [t2_spec])
            dist_scale1 = distribute_tensor(scale1, device_mesh, [scale1_spec])
            dist_scale2 = distribute_tensor(scale2, device_mesh, [scale2_spec])

            with CommDebugMode() as comm_mode:
                dist_res = cast(
                    DTensor,
                    torch._scaled_mm(
                        dist_t1_fp8,
                        dist_t2_fp8.t(),
                        scale_a=dist_scale1,
                        scale_b=dist_scale2.t(),
                        out_dtype=torch.bfloat16,
                    ),
                )

            self.assertEqual(dist_res.placements[0], output_spec)

            full_dist_res = dist_res.full_tensor()
            # Fp8 matmuls are quite inaccurate, we need high tolerances
            self.assertEqual(full_dist_res, full_ref_res, atol=1.5, rtol=7e-2)

            self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_matmul(self):
        device_mesh = self.build_device_mesh()
        dim = 128
        x = torch.randn(8, dim)
        A = torch.randn(dim, dim)
        y = torch.matmul(x, A)

        # Prepare DTensors
        dx = distribute_tensor(x, device_mesh, [Replicate()])
        dA = distribute_tensor(A, device_mesh, [Shard(0)])

        # Use `inference_mode` to test DTensor's capability of decomposing
        # `matmul` op
        with torch.inference_mode():
            dy = torch.matmul(dx, dA)

        self.assertEqual(y, dy.full_tensor())

    @with_comms
    def test_t(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        tensor_to_transpose = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_transpose, device_mesh, shard_spec)
        tranposed_mat = mat.t()
        self.assertEqual(tranposed_mat.size(), torch.Size([8, 12]))
        self.assertEqual(tranposed_mat.placements, [Shard(1)])
        tranposed_mat2 = tranposed_mat.t()
        self.assertEqual(tranposed_mat2.size(), torch.Size([12, 8]))
        self.assertEqual(tranposed_mat2.placements, shard_spec)

    @with_comms
    def test_t_partial(self):
        device_mesh = self.build_device_mesh()

        a = torch.randn(12, 8)
        b = torch.randn(8, 4)
        c = torch.mm(a, b).t()

        da = distribute_tensor(a, device_mesh, [Shard(1)])
        db = distribute_tensor(b, device_mesh, [Shard(0)])

        # mm(da, db) should return a Partial tensor.
        # transposing it should keep it Partial
        dc = torch.mm(da, db).t()

        self.assertTrue(isinstance(dc.placements[0], Partial))

        # check that the local and distributed op results match
        self.assertEqual(
            c,
            dc.redistribute(device_mesh, [Replicate()]).to_local(),
        )

    # baddbmm introduces nan occasionally on CPU: https://github.com/pytorch/pytorch/issues/80588
    @with_comms
    @skip_unless_torch_gpu
    def test_baddbmm(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        batch_1 = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        batch_2 = torch.rand(4, 8, 8, device=self.device_type, requires_grad=True)

        def test_placement_comb(
            tensor_placements: list[Placement],
            batch_1_placements: list[Placement],
            batch_2_placements: list[Placement],
            beta: int,
            alpha: int,
            batch_1_grad: Optional[torch.Tensor],
        ) -> None:
            tensor_dt = distribute_tensor(tensor, device_mesh, tensor_placements)
            batch_1_dt = distribute_tensor(batch_1, device_mesh, batch_1_placements)
            batch_2_dt = distribute_tensor(batch_2, device_mesh, batch_2_placements)
            dist_res = cast(
                DTensor,
                torch.baddbmm(
                    tensor_dt, batch_1_dt, batch_2_dt, beta=beta, alpha=alpha
                ),
            ).redistribute(device_mesh, [Replicate()])
            dist_local_res = dist_res.to_local()
            assert not torch.isnan(local_result).any()
            assert not torch.isnan(dist_local_res).any()
            self.assertEqual(dist_local_res.detach(), local_result.detach())

            # TODO: add test backward
            # grad_dist_res = torch.ones_like(dist_res)
            # dist_res.backward(grad_dist_res)
            # self.assertIsNotNone(batch_1_dt.grad)
            # batch_1_grad_local = batch_1_dt.grad.redistribute(
            #     device_mesh, [Replicate()]
            # ).to_local()
            # self.assertEqual(batch_1_grad_local, batch_1_grad)

        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        shard2_spec = Shard(2)
        replica_spec = Replicate()
        shard_specs = [shard0_spec, shard1_spec, shard2_spec, replica_spec]
        shard_specs_comb = list(
            itertools.product(shard_specs, shard_specs, shard_specs)
        )
        # If beta is 0, input tensor will be ignored
        numeric_params_comb = [
            (0.0, 0.5),  # zero-beta
            (0.8, 0.5),  # non-zero-beta
        ]

        for beta, alpha in numeric_params_comb:
            local_result = torch.baddbmm(
                tensor, batch_1, batch_2, beta=beta, alpha=alpha
            )
            grad_local_res = torch.ones_like(local_result)
            local_result.backward(grad_local_res)
            # test all combos
            for spec in shard_specs_comb:
                test_placement_comb(
                    [spec[0]], [spec[1]], [spec[2]], beta, alpha, batch_1.grad
                )

    @with_comms
    def test_bmm(self):
        device_mesh = self.build_device_mesh()
        mat1 = torch.rand(4, 8, 4, device=self.device_type, requires_grad=True)
        mat2 = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        local_result = torch.bmm(mat1, mat2)
        grad_local_res = torch.ones_like(local_result)
        local_result.backward(grad_local_res)

        def test_placement_comb(
            placements1: list[Placement],
            placements2: list[Placement],
        ) -> None:
            mat1_dt = distribute_tensor(mat1, device_mesh, placements1)
            mat2_dt = distribute_tensor(mat2, device_mesh, placements2)
            dist_res = cast(DTensor, torch.bmm(mat1_dt, mat2_dt)).redistribute(
                device_mesh, [Replicate()]
            )
            dist_local_res = dist_res.to_local()
            self.assertEqual(dist_local_res, local_result)

            # test backward
            # TODO: figure out (replicate, shard1) fail on backward
            # it generates a different grad shape
            grad_dist_res = torch.ones_like(dist_res)
            dist_res.backward(grad_dist_res)
            self.assertIsNotNone(mat1_dt.grad)
            mat1_dt_grad = cast(DTensor, mat1_dt.grad)
            mat1_grad_local = mat1_dt_grad.redistribute(
                device_mesh, [Replicate()]
            ).to_local()
            self.assertEqual(mat1_grad_local, mat1.grad)

        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        shard2_spec = Shard(2)
        replica_spec = Replicate()
        placement_specs = [shard0_spec, shard1_spec, shard2_spec, replica_spec]
        shard_specs_comb = list(itertools.product(placement_specs, placement_specs))

        # tests that currently pass
        for spec in shard_specs_comb:
            test_placement_comb([spec[0]], [spec[1]])

    @with_comms
    @skip_unless_torch_gpu
    def test_scaled_dot_product_attention(self):
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()
        # bsz, n_heads, slen, head_dim
        query = torch.rand(
            (4, 8, 8, 8),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.rand(
            (4, 8, 8, 8),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.rand(
            (4, 8, 8, 8),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        from torch.nn.attention import sdpa_kernel, SDPBackend

        available_backends = []
        dropout_p = 0.0
        # TODO: Add test cases where is_causal=False and an attention mask is provided.
        #       Gaps include missing op support for aten.masked_fill_.Scalar.
        is_causal = True
        enable_gqa = False
        params = torch.backends.cuda.SDPAParams(
            query, key, value, None, dropout_p, is_causal, enable_gqa
        )
        if torch.backends.cuda.can_use_flash_attention(params, debug=False):
            available_backends.append(SDPBackend.FLASH_ATTENTION)
        if torch.backends.cuda.can_use_efficient_attention(params, debug=False):
            available_backends.append(SDPBackend.EFFICIENT_ATTENTION)

        placement_specs = [(Replicate(),), (Shard(0),), (Shard(1),)]
        for backend, input_placements in itertools.product(
            available_backends, placement_specs
        ):
            dist_query = distribute_tensor(query, device_mesh, input_placements)
            dist_key = distribute_tensor(key, device_mesh, input_placements)
            dist_value = distribute_tensor(value, device_mesh, input_placements)
            with sdpa_kernel(backends=[backend]):
                out = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=dropout_p, is_causal=is_causal
                )
                with comm_mode:
                    dist_out = F.scaled_dot_product_attention(
                        dist_query,
                        dist_key,
                        dist_value,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                    )
                    self.assertEqual(comm_mode.get_total_counts(), 0)
                    self.assertEqual(dist_out.placements, input_placements)
                    self.assertEqual(dist_out.full_tensor(), out)

                out.sum().backward()
                with comm_mode:
                    dist_out.sum().backward()
                    self.assertEqual(comm_mode.get_total_counts(), 0)
                    self.assertEqual(dist_query.grad.placements, input_placements)
                    self.assertEqual(dist_query.grad.full_tensor(), query.grad)
                    self.assertEqual(dist_key.grad.placements, input_placements)
                    self.assertEqual(dist_key.grad.full_tensor(), key.grad)
                    self.assertEqual(dist_value.grad.placements, input_placements)
                    self.assertEqual(dist_value.grad.full_tensor(), value.grad)
                    query.grad.zero_()
                    key.grad.zero_()
                    value.grad.zero_()

    @skip_unless_torch_gpu
    @with_comms()
    def test_dtensor_mm(self):
        """
        Test mm with DTensor with 2D mesh.
        We need to add the test here since we only test 1D mesh in test_dtensor_ops.py.
        Also, we added tests for the corner case where one of the 2D dimension is 1.

        # TODO: we need to test more DTensor ops with 2D mesh, especially when 1 of the
        mesh dimension of the 2D mesh is 1.
        """
        mesh_0 = init_device_mesh(self.device_type, (self.world_size // 2, 2))
        mesh_1 = init_device_mesh(self.device_type, (self.world_size, 1))
        mesh_2 = init_device_mesh(self.device_type, (1, self.world_size))

        for mesh in [mesh_0, mesh_1, mesh_2]:
            lhs = torch.randn(256, 128)
            rhs = torch.randn(128, 256)
            mm_result = lhs @ rhs

            lhs_dtensor = distribute_tensor(lhs, mesh, [Shard(dim=0), Replicate()])
            rhs_dtensor = distribute_tensor(rhs, mesh, [Replicate(), Shard(dim=1)])
            dtensor_result = lhs_dtensor @ rhs_dtensor
            self.assertEqual(
                dtensor_result.full_tensor(), mm_result, atol=1.5e-5, rtol=1e-6
            )

    @with_comms
    @skip_unless_torch_gpu
    def test_tensordot_shampoo(self):
        """
        Create a simple test for Shampoo's use case.
        """
        device_mesh = self.build_device_mesh()

        local_a = torch.randn(4, 4)
        local_b = torch.randn(4, 15)
        dims = ([0], [0])
        local_result = torch.tensordot(local_a, local_b, dims=(dims))

        placements = [Replicate(), Shard(0), Shard(1)]
        placements_tuples = itertools.product(placements, repeat=2)

        for placement1, placement2 in placements_tuples:
            dist_a = distribute_tensor(local_a, device_mesh, [placement1])
            dist_b = distribute_tensor(local_b, device_mesh, [placement2])
            dist_result = torch.tensordot(dist_a, dist_b, dims=dims)
            dist_result_full = dist_result.full_tensor()
            self.assertEqual(local_result, dist_result_full)

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @unittest.skipIf(not SM90OrLater, "Grouped gemm supported on SM90")
    @with_comms
    @skip_unless_torch_gpu
    @parametrize(
        "kwargs",
        [
            {
                # 2D x 3D case from MoE layer
                "inp_shape": (64, 16),
                "w1_shape": (2, 16, 32),
                "w2_shape": (2, 32, 16),
                "inp_placements": [Replicate()],
                "w1_placements": [Shard(2)],
                "w2_placements": [Shard(1)],
                "expected_comm_counts_fwd": 0,
                "expected_comm_counts_bwd": 1,
                "expected_out_placements": [Partial()],
            },
            {
                # Case that would have invalid strides on inp * mat1 when sharded
                "inp_shape": (64, 16),
                "w1_shape": (2, 16, 16),
                "w2_shape": (2, 16, 16),
                "inp_placements": [Replicate()],
                "w1_placements": [Shard(2)],
                "w2_placements": [Shard(1)],
                "expected_comm_counts_fwd": 2,
                "expected_comm_counts_bwd": 4,
                "expected_out_placements": [Replicate()],
            },
        ],
    )
    def test_grouped_mm(self, kwargs):
        # TODO: torch.nn.functional.grouped_mm can take inputs of dimension (2D, 3D) x (2D, 3D)
        # More tests need to be added.
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()
        dtype = torch.bfloat16
        inp = torch.rand(
            *kwargs["inp_shape"],
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        w1 = torch.rand(
            *kwargs["w1_shape"],
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        w2 = torch.rand(
            *kwargs["w2_shape"],
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        offs = torch.tensor([16, 64], device=self.device_type, dtype=torch.int32)

        h = F.grouped_mm(inp, w1, offs=offs)
        out = F.grouped_mm(h, w2, offs=offs)

        dist_inp = distribute_tensor(inp, device_mesh, kwargs["inp_placements"])
        # colwise sharded
        dist_w1 = distribute_tensor(w1, device_mesh, kwargs["w1_placements"])
        # rowwise sharded
        dist_w2 = distribute_tensor(w2, device_mesh, kwargs["w2_placements"])
        dist_offs = distribute_tensor(offs, device_mesh, [Replicate()])

        with comm_mode:
            dist_h = F.grouped_mm(dist_inp, dist_w1, offs=dist_offs)
            dist_out = F.grouped_mm(dist_h, dist_w2, offs=dist_offs)
            self.assertEqual(
                comm_mode.get_total_counts(), kwargs["expected_comm_counts_fwd"]
            )
            self.assertEqual(dist_out.placements, kwargs["expected_out_placements"])
            self.assertEqual(dist_out.full_tensor(), out)

        out_grad = torch.ones_like(out)
        out.backward(out_grad)

        dist_out = dist_out.redistribute(device_mesh, [Shard(0)])
        dist_out_grad = distribute_tensor(out_grad, device_mesh, [Shard(0)])

        with comm_mode:
            dist_out.backward(dist_out_grad)
            self.assertEqual(
                comm_mode.get_total_counts(), kwargs["expected_comm_counts_bwd"]
            )
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.all_gather_into_tensor],
                kwargs["expected_comm_counts_bwd"],
            )
        self.assertEqual(dist_inp.grad.full_tensor(), inp.grad)
        self.assertEqual(dist_w1.grad.full_tensor(), w1.grad)
        self.assertEqual(dist_w2.grad.full_tensor(), w2.grad)


instantiate_parametrized_tests(DistMatrixOpsTest)

DistMatrixOpsTestWithLocalTensor = create_local_tensor_test_class(
    DistMatrixOpsTest,
)

if __name__ == "__main__":
    run_tests()
