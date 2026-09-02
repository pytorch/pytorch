# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools
import math
import unittest
from typing import cast
from unittest.mock import patch

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
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_cuda import (
    _get_torch_cuda_version,
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MX_GEMM,
    SM90OrLater,
)
from torch.testing._internal.common_device_type import E4M3_MAX_POS, e4m3_type
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfRocm,
    TEST_WITH_ROCM,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    skip_unless_torch_gpu,
    with_comms,
)


funcol = torch.ops.c10d_functional
mx_skip_msg = "MX gemm is only supported on CUDA capability 10.0+"


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


def _current_rank_int_for_mesh(
    mesh,
    value: int | torch.SymInt,  # pyrefly: ignore[bad-parameter-annotation]
) -> int:
    from torch.distributed.tensor._dispatch import _current_rank_int

    return _current_rank_int(value, current_rank=mesh._rank)


def _make_blockwise_scale_2d(
    rows: int, sf_k: int, dtype: torch.dtype, device: str
) -> torch.Tensor:
    base = torch.tensor(
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
        device=device,
        dtype=torch.float32,
    )
    idx = torch.arange(rows * sf_k, device=device).remainder(base.numel())
    return base[idx].reshape(rows, sf_k).to(dtype)


def _flat_blockwise_scale_slice(
    scale_2d: torch.Tensor,
    *,
    row_start: int = 0,
    row_count: int | None = None,
    sf_k_start: int = 0,
    sf_k_count: int | None = None,
) -> torch.Tensor:
    from torch._vendor.quack.blockscaled_layout_utils import (
        pack_scale_2d_to_blocked_contig,
        scale_blocked_for_cublas,
    )

    row_count = row_count if row_count is not None else scale_2d.shape[0] - row_start
    sf_k_count = (
        sf_k_count if sf_k_count is not None else scale_2d.shape[1] - sf_k_start
    )
    return scale_blocked_for_cublas(
        pack_scale_2d_to_blocked_contig(
            scale_2d[
                row_start : row_start + row_count,
                sf_k_start : sf_k_start + sf_k_count,
            ]
        ),
        row_count,
        sf_k_count,
    )


def _make_rowwise_mxfp8_tp_inputs(
    device_mesh,
    device_type: str,
    *,
    m: int,
    n: int,
    k: int,
    block_size: int = 32,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    DTensor,
    DTensor,
    DTensor,
    DTensor,
    torch.Tensor,
    torch.Tensor,
]:
    from torch.testing._internal.common_quantized import to_blocked, to_mxfp

    t1 = (
        torch.randn(m, k, device=device_type, dtype=torch.bfloat16) * (k**-0.5)
    ).contiguous()
    t2 = (
        torch.randn(n, k, device=device_type, dtype=torch.bfloat16) * (k**-0.5)
    ).contiguous()
    scale1, t1_fp8 = to_mxfp(t1, block_size=block_size, format="mxfp8")
    scale2, t2_fp8 = to_mxfp(t2, block_size=block_size, format="mxfp8")
    dist_t1_fp8 = distribute_tensor(t1_fp8, device_mesh, [Shard(1)])
    dist_t2_fp8 = distribute_tensor(t2_fp8, device_mesh, [Shard(1)])
    dist_scale1 = DTensor.from_local(to_blocked(scale1), device_mesh, [Replicate()])
    dist_scale2 = DTensor.from_local(to_blocked(scale2), device_mesh, [Replicate()])
    return t1, t2, dist_t1_fp8, dist_t2_fp8, dist_scale1, dist_scale2, scale1, scale2


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
                        lambda msg: f"{msg}\nBias Partial should have same reduce_op as output Partial. "
                        f"Got bias={bias_placement.reduce_op}, output={output_placement.reduce_op}",
                    )

    def test_gen_single_dim_einsum_strategies_batch_linearity(self):
        """Test that batch-only equations auto-detect all-Partial linearity."""
        S = _ShardingPlaceholder

        # For "abcd,abcd->abcd": all dims are batch, no contracting/free.
        # Expect: 4 batch-dim + 4 per-input linearity + 2 all-Partial linearity = 10
        strategies = gen_single_dim_einsum_strategies("abcd,abcd->abcd")

        # Convert to repr tuples for comparison since _ShardingPlaceholder lacks __eq__
        actual = [tuple(repr(p) for p in s) for s in strategies]
        expected = [
            # batch dims: shard output and both inputs on same dim
            (repr(S(0)), repr(S(0)), repr(S(0))),
            (repr(S(1)), repr(S(1)), repr(S(1))),
            (repr(S(2)), repr(S(2)), repr(S(2))),
            (repr(S(3)), repr(S(3)), repr(S(3))),
            # per-input linearity: one input Partial, other Replicate
            ("Partial(sum)", "Partial(sum)", "Replicate()"),
            ("Partial(sum)", "Replicate()", "Partial(sum)"),
            ("Partial(avg)", "Partial(avg)", "Replicate()"),
            ("Partial(avg)", "Replicate()", "Partial(avg)"),
            # batch-dimension linearity: all inputs Partial
            ("Partial(sum)", "Partial(sum)", "Partial(sum)"),
            ("Partial(avg)", "Partial(avg)", "Partial(avg)"),
        ]
        self.assertEqual(actual, expected)

        # For "mk,kn->mn": has contracting dim k, no all-Partial linearity.
        mm_strategies = gen_single_dim_einsum_strategies("mk,kn->mn")
        mm_all_partial = [
            s for s in mm_strategies if all(isinstance(p, Partial) for p in s)
        ]
        self.assertEqual(len(mm_all_partial), 0)

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_mm_with_strided_input(self):
        # Case 1: 1D mesh with StridedShard
        # Tests mm where input has _StridedShard(dim=0, split_factor=2) placement.
        # Input shape: (batch_size * seq_len, contract_dim), weight is Replicate.
        # Output should preserve the same StridedShard placement.
        mesh = self.build_device_mesh()
        batch_size, seq_len, contract_dim, out_dim = 2, self.world_size, 3, 7
        global_inps_viewed = (
            torch.arange(batch_size * seq_len * contract_dim)
            .float()
            .view(batch_size * seq_len, contract_dim)
        )
        inps_viewed = distribute_tensor(
            global_inps_viewed,
            mesh,
            (_StridedShard(dim=0, split_factor=2),),
            src_data_rank=None,
        )
        global_weight = (
            torch.arange(contract_dim * out_dim).float().view(contract_dim, out_dim)
        )
        weight = distribute_tensor(global_weight, mesh, (Replicate(),))
        out = torch.mm(inps_viewed, weight)
        expected_placements = (_StridedShard(dim=0, split_factor=2),)
        self.assertEqual(out.placements, expected_placements)

        # Case 2: 2D mesh (2x2) with nested StridedShard on both mesh dimensions
        # Tests mm where input has StridedShard on both mesh dims with different split_factors.
        # This simulates a more complex sharding pattern (e.g., from a reshaped 4D tensor).
        # Output should preserve both StridedShard placements.
        mesh = init_device_mesh(self.device_type, (2, 2))
        tensor_dims = (4, mesh.size(0) * mesh.size(1), 6, 8)
        global_inps_viewed = (
            torch.arange(math.prod(tensor_dims))
            .float()
            .view(math.prod(tensor_dims[:3]), 8)
        )
        inps_viewed = distribute_tensor(
            global_inps_viewed,
            mesh,
            (
                _StridedShard(dim=0, split_factor=4),
                _StridedShard(
                    dim=0,
                    split_factor=tensor_dims[0] * (tensor_dims[1] // mesh.size(0)),
                ),
            ),
            src_data_rank=None,
        )
        global_weight = (
            torch.arange(tensor_dims[-1] * tensor_dims[-1])
            .float()
            .view(tensor_dims[-1], tensor_dims[-1])
        )
        weight = distribute_tensor(global_weight, mesh, (Replicate(), Replicate()))
        out = torch.mm(inps_viewed, weight)
        expected_placements = (
            _StridedShard(dim=0, split_factor=4),
            _StridedShard(
                dim=0, split_factor=tensor_dims[0] * (tensor_dims[1] // mesh.size(0))
            ),
        )
        self.assertEqual(out.placements, expected_placements)

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
    def test_aten_linear(self):
        device_mesh = self.build_device_mesh()
        x = distribute_tensor(
            torch.randn(1, 47, 2048),
            device_mesh,
            [Replicate()],
        )
        w = distribute_tensor(
            torch.randn(2048, 2048),
            device_mesh,
            [Shard(0)],
        )

        with torch.inference_mode():  # call aten::linear
            out = torch.nn.functional.linear(x, w)

        self.assertEqual(out.placements, (Shard(2),))

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

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180006")
    @with_comms
    @skip_unless_torch_gpu
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @unittest.skip(
        "Disabled due to CI failures on B200; see "
        "https://github.com/pytorch/pytorch/issues/190086"
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

    def test_scaled_mm_blockwise_1d_scale_placement(self):
        """Test that _scaled_mm_scale_placement handles 1D blockwise scales correctly.

        1D blockwise scales arise in MX (microscaling) formats where a data
        tensor [M, K] has a flattened scale of shape [M * K / block_size].
        Shard(>=1) is invalid on a 1D tensor, so the strategy maps
        non-contracting shards to Shard(0) and keeps contracting-dim shards
        Replicate so dispatch can localize the K slice later. The dispatch-time
        half of that behavior is covered separately by the localization tests.
        """
        from torch.distributed.tensor._ops._matrix_ops import _scaled_mm_scale_placement

        # --- Tensor-wise scale (single element) -> always Replicate ---
        result = _scaled_mm_scale_placement(
            Shard(0), torch.Size([1]), contracting_dim=1
        )
        self.assertEqual(result, Replicate())
        result = _scaled_mm_scale_placement(Shard(0), torch.Size([]), contracting_dim=1)
        self.assertEqual(result, Replicate())

        # --- 2D scale -> copy data placement directly (row-wise) ---
        result = _scaled_mm_scale_placement(
            Shard(0), torch.Size([16, 1]), contracting_dim=1
        )
        self.assertEqual(result, Shard(0))

        # --- 1D blockwise + non-contracting shard -> Shard(0) ---
        # A (mk): dim 0 = m (non-contracting), dim 1 = k (contracting)
        result = _scaled_mm_scale_placement(
            Shard(0), torch.Size([64]), contracting_dim=1
        )
        self.assertEqual(result, Shard(0))

        # B_t (kn): dim 1 = n (non-contracting), dim 0 = k (contracting)
        result = _scaled_mm_scale_placement(
            Shard(1), torch.Size([64]), contracting_dim=0
        )
        self.assertEqual(result, Shard(0))

        # --- 1D blockwise + contracting shard -> Replicate (localized later) ---
        result = _scaled_mm_scale_placement(
            Shard(1), torch.Size([64]), contracting_dim=1
        )
        self.assertEqual(result, Replicate())
        result = _scaled_mm_scale_placement(
            Shard(0), torch.Size([64]), contracting_dim=0
        )
        self.assertEqual(result, Replicate())

        # --- 1D blockwise + Replicate -> Replicate ---
        result = _scaled_mm_scale_placement(
            Replicate(), torch.Size([64]), contracting_dim=1
        )
        self.assertEqual(result, Replicate())

        # --- 1D blockwise + Partial -> Replicate ---
        result = _scaled_mm_scale_placement(
            Partial(), torch.Size([64]), contracting_dim=0
        )
        self.assertEqual(result, Replicate())

    @skipIfRocm
    @with_comms
    @skip_unless_torch_gpu
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    def test_scaled_mm_blockwise_1d_scale_localization(self):
        from torch._vendor.quack.blockscaled_layout_utils import MX_BLOCK_SIZE
        from torch.distributed.tensor._dispatch import (
            _localize_blockwise_scaled_mm_scale,
        )
        from torch.distributed.tensor._utils import (
            compute_local_shape_and_global_offset,
        )

        device_mesh = self.build_device_mesh()
        block_size = MX_BLOCK_SIZE
        m, n = 128, 96
        k = block_size * self.world_size * 2
        sf_k = k // block_size
        scale_a_2d = _make_blockwise_scale_2d(
            m, sf_k, torch.float8_e8m0fnu, self.device_type
        )
        scale_b_2d = _make_blockwise_scale_2d(
            n, sf_k, torch.float8_e8m0fnu, self.device_type
        )
        scale_a_flat = _flat_blockwise_scale_slice(scale_a_2d)
        scale_b_flat = _flat_blockwise_scale_slice(scale_b_2d)

        data_a = distribute_tensor(
            torch.zeros(m, k, device=self.device_type, dtype=torch.float8_e4m3fn),
            device_mesh,
            [Shard(1)],
        )
        data_b = distribute_tensor(
            torch.zeros(k, n, device=self.device_type, dtype=torch.float8_e4m3fn),
            device_mesh,
            [Shard(0)],
        )
        localized_a = _localize_blockwise_scaled_mm_scale(
            scale_a_flat,
            data_a._local_tensor,
            data_a._spec,
            contracting_dim=1,
            non_contracting_dim=0,
        )
        localized_b = _localize_blockwise_scaled_mm_scale(
            scale_b_flat,
            data_b._local_tensor,
            data_b._spec,
            contracting_dim=0,
            non_contracting_dim=1,
        )

        local_shape_a, global_offset_a = compute_local_shape_and_global_offset(
            data_a._spec.tensor_meta.shape, data_a.device_mesh, data_a.placements
        )
        local_shape_a = tuple(data_a._local_tensor.shape)
        global_offset_a = tuple(
            _current_rank_int_for_mesh(device_mesh, offset)
            for offset in global_offset_a
        )
        local_sf_k_a = local_shape_a[1] // block_size
        start_sf_k_a = global_offset_a[1] // block_size
        expected_a = _flat_blockwise_scale_slice(
            scale_a_2d,
            row_count=local_shape_a[0],
            sf_k_start=start_sf_k_a,
            sf_k_count=local_sf_k_a,
        )
        self.assertEqual(localized_a, expected_a)
        local_shape_b, global_offset_b = compute_local_shape_and_global_offset(
            data_b._spec.tensor_meta.shape, data_b.device_mesh, data_b.placements
        )
        local_shape_b = tuple(data_b._local_tensor.shape)
        global_offset_b = tuple(
            _current_rank_int_for_mesh(device_mesh, offset)
            for offset in global_offset_b
        )
        local_sf_k_b = local_shape_b[0] // block_size
        start_sf_k_b = global_offset_b[0] // block_size
        expected_b = _flat_blockwise_scale_slice(
            scale_b_2d,
            row_count=local_shape_b[1],
            sf_k_start=start_sf_k_b,
            sf_k_count=local_sf_k_b,
        )
        self.assertEqual(localized_b, expected_b)

    @skipIfRocm
    @with_comms
    @skip_unless_torch_gpu
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    def test_scaled_mm_blockwise_1d_scale_localization_nvfp4(self):
        from torch.distributed.tensor._dispatch import (
            _localize_blockwise_scaled_mm_scale,
        )
        from torch.distributed.tensor._utils import (
            compute_local_shape_and_global_offset,
        )
        from torch.testing._internal.common_quantized import to_blocked

        device_mesh = self.build_device_mesh()
        block_size = 16
        m, n = 128, 96
        logical_k = block_size * self.world_size * 2
        packed_k = logical_k // 2
        sf_k = logical_k // block_size
        scale_a_2d = _make_blockwise_scale_2d(
            m, sf_k, torch.float8_e4m3fn, self.device_type
        )
        scale_b_2d = _make_blockwise_scale_2d(
            n, sf_k, torch.float8_e4m3fn, self.device_type
        )
        scale_a_flat = to_blocked(scale_a_2d)
        scale_b_flat = to_blocked(scale_b_2d)

        local_packed_k = packed_k // self.world_size
        data_a = DTensor.from_local(
            torch.empty(
                m,
                local_packed_k,
                device=self.device_type,
                dtype=torch.float4_e2m1fn_x2,
            ),
            device_mesh,
            [Shard(1)],
            shape=torch.Size([m, packed_k]),
            stride=(packed_k, 1),
        )
        data_b = DTensor.from_local(
            torch.empty(
                local_packed_k,
                n,
                device=self.device_type,
                dtype=torch.float4_e2m1fn_x2,
            ),
            device_mesh,
            [Shard(0)],
            shape=torch.Size([packed_k, n]),
            stride=(n, 1),
        )
        localized_a = _localize_blockwise_scaled_mm_scale(
            scale_a_flat,
            data_a._local_tensor,
            data_a._spec,
            contracting_dim=1,
            non_contracting_dim=0,
        )
        localized_b = _localize_blockwise_scaled_mm_scale(
            scale_b_flat,
            data_b._local_tensor,
            data_b._spec,
            contracting_dim=0,
            non_contracting_dim=1,
        )

        local_shape_a, global_offset_a = compute_local_shape_and_global_offset(
            data_a._spec.tensor_meta.shape, data_a.device_mesh, data_a.placements
        )
        local_shape_a = tuple(data_a._local_tensor.shape)
        global_offset_a = tuple(
            _current_rank_int_for_mesh(device_mesh, offset)
            for offset in global_offset_a
        )
        local_sf_k_a = (local_shape_a[1] * 2) // block_size
        start_sf_k_a = (global_offset_a[1] * 2) // block_size
        self.assertEqual(
            localized_a,
            to_blocked(scale_a_2d[:, start_sf_k_a : start_sf_k_a + local_sf_k_a]),
        )

        local_shape_b, global_offset_b = compute_local_shape_and_global_offset(
            data_b._spec.tensor_meta.shape, data_b.device_mesh, data_b.placements
        )
        local_shape_b = tuple(data_b._local_tensor.shape)
        global_offset_b = tuple(
            _current_rank_int_for_mesh(device_mesh, offset)
            for offset in global_offset_b
        )
        local_sf_k_b = (local_shape_b[0] * 2) // block_size
        start_sf_k_b = (global_offset_b[0] * 2) // block_size
        self.assertEqual(
            localized_b,
            to_blocked(scale_b_2d[:, start_sf_k_b : start_sf_k_b + local_sf_k_b]),
        )

    @skipIfRocm
    @with_comms
    @skip_unless_torch_gpu
    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    def test_scaled_mm_blockwise_1d_scale_localization_mixed_mk(self):
        from torch._vendor.quack.blockscaled_layout_utils import MX_BLOCK_SIZE
        from torch.distributed.tensor._dispatch import (
            _localize_blockwise_scaled_mm_scale,
        )
        from torch.distributed.tensor._utils import (
            compute_local_shape_and_global_offset,
        )

        mesh_shape = (2, self.world_size // 2)
        device_mesh = init_device_mesh(self.device_type, mesh_shape)
        block_size = MX_BLOCK_SIZE
        m = 128 * mesh_shape[0]
        n = 128 * mesh_shape[0]
        k = block_size * mesh_shape[1] * 4
        sf_k = k // block_size
        scale_a_2d = _make_blockwise_scale_2d(
            m, sf_k, torch.float8_e8m0fnu, self.device_type
        )
        scale_b_2d = _make_blockwise_scale_2d(
            n, sf_k, torch.float8_e8m0fnu, self.device_type
        )

        data_a = distribute_tensor(
            torch.zeros(m, k, device=self.device_type, dtype=torch.float8_e4m3fn),
            device_mesh,
            [Shard(0), Shard(1)],
        )
        local_shape_a, global_offset_a = compute_local_shape_and_global_offset(
            data_a._spec.tensor_meta.shape, data_a.device_mesh, data_a.placements
        )
        local_shape_a = tuple(data_a._local_tensor.shape)
        global_offset_a = tuple(
            _current_rank_int_for_mesh(device_mesh, offset)
            for offset in global_offset_a
        )
        m_start = global_offset_a[0]
        scale_a_current = _flat_blockwise_scale_slice(
            scale_a_2d,
            row_start=m_start,
            row_count=local_shape_a[0],
            sf_k_count=sf_k,
        )
        localized_a = _localize_blockwise_scaled_mm_scale(
            scale_a_current,
            data_a._local_tensor,
            data_a._spec,
            contracting_dim=1,
            non_contracting_dim=0,
        )
        local_sf_k_a = local_shape_a[1] // block_size
        start_sf_k_a = global_offset_a[1] // block_size
        expected_a = _flat_blockwise_scale_slice(
            scale_a_2d,
            row_start=m_start,
            row_count=local_shape_a[0],
            sf_k_start=start_sf_k_a,
            sf_k_count=local_sf_k_a,
        )
        self.assertEqual(localized_a, expected_a)

        data_b = distribute_tensor(
            torch.zeros(k, n, device=self.device_type, dtype=torch.float8_e4m3fn),
            device_mesh,
            [Shard(1), Shard(0)],
        )
        local_shape_b, global_offset_b = compute_local_shape_and_global_offset(
            data_b._spec.tensor_meta.shape, data_b.device_mesh, data_b.placements
        )
        local_shape_b = tuple(data_b._local_tensor.shape)
        global_offset_b = tuple(
            _current_rank_int_for_mesh(device_mesh, offset)
            for offset in global_offset_b
        )
        n_start = global_offset_b[1]
        scale_b_current = _flat_blockwise_scale_slice(
            scale_b_2d,
            row_start=n_start,
            row_count=local_shape_b[1],
            sf_k_count=sf_k,
        )
        localized_b = _localize_blockwise_scaled_mm_scale(
            scale_b_current,
            data_b._local_tensor,
            data_b._spec,
            contracting_dim=0,
            non_contracting_dim=1,
        )
        local_sf_k_b = local_shape_b[0] // block_size
        start_sf_k_b = global_offset_b[0] // block_size
        expected_b = _flat_blockwise_scale_slice(
            scale_b_2d,
            row_start=n_start,
            row_count=local_shape_b[1],
            sf_k_start=start_sf_k_b,
            sf_k_count=local_sf_k_b,
        )
        self.assertEqual(localized_b, expected_b)

    @skipIfRocm
    @skip_unless_torch_gpu
    def test_scaled_mm_blockwise_1d_scale_localization_rejects_unaligned_offsets(self):
        from torch._vendor.quack.blockscaled_layout_utils import (
            pack_scale_2d_to_blocked_contig,
            scale_blocked_for_cublas,
        )
        from torch.distributed.tensor._dispatch import (
            _localize_blockwise_scaled_mm_scale_from_offsets,
        )

        rows, global_k, local_k, offset, block_size = 128, 96, 48, 48, 32
        scale_2d = torch.ones(
            rows,
            global_k // block_size,
            device=self.device_type,
            dtype=torch.float8_e8m0fnu,
        )
        scale_flat = scale_blocked_for_cublas(
            pack_scale_2d_to_blocked_contig(scale_2d),
            rows,
            global_k // block_size,
        )
        with self.assertRaisesRegex(ValueError, "block-aligned K offsets"):
            _localize_blockwise_scaled_mm_scale_from_offsets(
                scale_flat,
                rows,
                global_k,
                local_k,
                offset,
                block_size,
            )
        with self.assertRaisesRegex(ValueError, "block-aligned local K sizes"):
            _localize_blockwise_scaled_mm_scale_from_offsets(
                scale_flat,
                rows,
                global_logical_k=128,
                local_logical_k=48,
                logical_k_offset=0,
                block_size=block_size,
            )

    @skipIfRocm
    @with_comms
    @skip_unless_torch_gpu
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    def test_scaled_mm_blockwise_1d_rowwise_tp(self):
        """Exercise the DTensor rowwise MXFP8 path through sharding propagation.

        This asserts the user-visible DTensor behavior this fix owns: the
        rowwise TP inputs propagate to Partial(sum), and the handler localizes
        the flat blockwise MX scales to the correct per-rank payloads before
        the local tensor call. We stop short of a full local _scaled_mm kernel
        execution here because that path is still cuBLASLt-shape-sensitive on
        the current H200/CUDA test environment.
        """
        from torch.distributed.tensor._dispatch import (
            _current_op_schema_for_local_args,
            _maybe_localize_scaled_mm_blockwise_args,
        )
        from torch.distributed.tensor._utils import (
            compute_local_shape_and_global_offset,
        )

        device_mesh = self.build_device_mesh()
        block_size = 32
        ws = self.world_size
        m, n, k = 128, 128, block_size * ws * 8
        _, _, dist_t1_fp8, dist_t2_fp8, dist_scale1, dist_scale2, scale1, scale2 = (
            _make_rowwise_mxfp8_tp_inputs(
                device_mesh,
                self.device_type,
                m=m,
                n=n,
                k=k,
                block_size=block_size,
            )
        )

        op_call = torch.ops.aten._scaled_mm.default
        op_info = DTensor._op_dispatcher.unwrap_to_op_info(
            op_call,
            (
                dist_t1_fp8,
                dist_t2_fp8.t(),
                dist_scale1,
                dist_scale2,
                None,
                None,
                torch.bfloat16,
                False,
            ),
            {},
        )
        DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
        output_sharding = op_info.output_sharding
        if output_sharding is None:
            raise AssertionError("output sharding should not be None")

        self.assertFalse(output_sharding.needs_redistribute)
        self.assertEqual(output_sharding.output_spec.placements, (Partial(),))

        current_op_schema = _current_op_schema_for_local_args(
            op_call, op_info, output_sharding
        )
        local_args = cast(tuple[object, ...], tuple(op_info.local_args))
        localized_args = _maybe_localize_scaled_mm_blockwise_args(
            local_args, current_op_schema
        )
        localized_scale1 = cast(torch.Tensor, localized_args[2])
        localized_scale2 = cast(torch.Tensor, localized_args[3])

        local_shape_a, global_offset_a = compute_local_shape_and_global_offset(
            dist_t1_fp8._spec.tensor_meta.shape,
            dist_t1_fp8.device_mesh,
            dist_t1_fp8.placements,
        )
        local_shape_a = tuple(dist_t1_fp8._local_tensor.shape)
        global_offset_a = tuple(
            _current_rank_int_for_mesh(device_mesh, offset)
            for offset in global_offset_a
        )
        local_sf_k_a = local_shape_a[1] // block_size
        start_sf_k_a = global_offset_a[1] // block_size
        self.assertEqual(
            localized_scale1,
            _flat_blockwise_scale_slice(
                scale1,
                row_count=scale1.shape[0],
                sf_k_start=start_sf_k_a,
                sf_k_count=local_sf_k_a,
            ),
        )

        local_shape_b, global_offset_b = compute_local_shape_and_global_offset(
            dist_t2_fp8.t()._spec.tensor_meta.shape,
            dist_t2_fp8.t().device_mesh,
            dist_t2_fp8.t().placements,
        )
        local_shape_b = tuple(dist_t2_fp8.t()._local_tensor.shape)
        global_offset_b = tuple(
            _current_rank_int_for_mesh(device_mesh, offset)
            for offset in global_offset_b
        )
        local_sf_k_b = local_shape_b[0] // block_size
        start_sf_k_b = global_offset_b[0] // block_size
        self.assertEqual(
            localized_scale2,
            _flat_blockwise_scale_slice(
                scale2,
                row_count=scale2.shape[0],
                sf_k_start=start_sf_k_b,
                sf_k_count=local_sf_k_b,
            ),
        )

    @skipIfRocm
    @with_comms
    @skip_unless_torch_gpu
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(not PLATFORM_SUPPORTS_MX_GEMM, mx_skip_msg)
    def test_scaled_mm_blockwise_1d_rowwise_tp_e2e_mx_gemm(self):
        device_mesh = self.build_device_mesh()
        block_size = 32
        ws = self.world_size
        m = 128
        n = 128
        k = 128 * ws
        t1, t2, dist_t1_fp8, dist_t2_fp8, dist_scale1, dist_scale2, _, _ = (
            _make_rowwise_mxfp8_tp_inputs(
                device_mesh,
                self.device_type,
                m=m,
                n=n,
                k=k,
                block_size=block_size,
            )
        )
        full_ref_res = t1 @ t2.t()

        with CommDebugMode() as comm_mode:
            dist_res = cast(
                DTensor,
                torch._scaled_mm(
                    dist_t1_fp8,
                    dist_t2_fp8.t(),
                    scale_a=dist_scale1,
                    scale_b=dist_scale2,
                    out_dtype=torch.bfloat16,
                ),
            )

        self.assertEqual(dist_res.placements, (Partial(),))
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(dist_res.full_tensor(), full_ref_res, atol=1.5, rtol=7e-2)

    @skipIfRocm
    @with_comms
    @skip_unless_torch_gpu
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    def test_scaled_mm_blockwise_1d_uses_custom_handler(self):
        class ScaledMmHandlerCalled(RuntimeError):
            pass

        device_mesh = self.build_device_mesh()
        block_size = 32
        ws = self.world_size
        m, n, k = 128, 128, block_size * ws * 4
        _, _, dist_t1_fp8, dist_t2_fp8, dist_scale1, dist_scale2, _, _ = (
            _make_rowwise_mxfp8_tp_inputs(
                device_mesh,
                self.device_type,
                m=m,
                n=n,
                k=k,
                block_size=block_size,
            )
        )

        def sentinel_handler(op_call, args, kwargs):
            raise ScaledMmHandlerCalled

        with patch.dict(
            DTensor._op_dispatcher._custom_op_handlers,
            {torch.ops.aten._scaled_mm.default: sentinel_handler},
            clear=False,
        ):
            with self.assertRaises(ScaledMmHandlerCalled):
                torch._scaled_mm(
                    dist_t1_fp8,
                    dist_t2_fp8.t(),
                    scale_a=dist_scale1,
                    scale_b=dist_scale2,
                    out_dtype=torch.bfloat16,
                )

    @skipIfRocm
    @with_comms
    @skip_unless_torch_gpu
    @skip_if_lt_x_gpu(2)
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

    @with_comms
    def test_t_1d(self):
        # t() on a 1D tensor is a no-op and should preserve the shard placement
        device_mesh = self.build_device_mesh()

        tensor_1d = torch.randn(8)
        mat = distribute_tensor(tensor_1d, device_mesh, [Shard(0)])
        transposed = mat.t()
        # t() on 1D is a no-op, should stay Shard(0)
        self.assertEqual(transposed.size(), torch.Size([8]))
        self.assertEqual(transposed.placements, (Shard(0),))
        # Verify values match
        self.assertEqual(transposed.full_tensor(), tensor_1d)

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
            batch_1_grad: torch.Tensor | None,
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
            if torch.isnan(local_result).any():
                raise AssertionError("NaN values found in local_result")
            if torch.isnan(dist_local_res).any():
                raise AssertionError("NaN values found in dist_local_res")
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
    def test_mm_partial_inputs(self):
        # mm with Partial inputs should produce Partial output via per-input
        # linearity, for both the default and single-dim strategy paths,
        # across various mesh dimensionalities and reduce ops (sum, avg).
        mesh_shapes = [
            (self.world_size,),
            (self.world_size // 2, 2),
            (self.world_size // 2, 2, 1),
            (1, self.world_size // 2, 2, 1),
            (1, 1, self.world_size // 2, 2, 1),
        ]

        def _run_mm(device_mesh, reduce_op="sum"):
            placements = [Partial(reduce_op)] * device_mesh.ndim
            a_local = torch.randn(16, 12, device=self.device_type)
            b_local = torch.randn(12, 20, device=self.device_type)
            dt1 = DTensor.from_local(
                a_local,
                device_mesh,
                placements,
                run_check=False,
            )
            dt2 = DTensor.from_local(
                b_local,
                device_mesh,
                placements,
                run_check=False,
            )
            comm_mode = CommDebugMode()
            with comm_mode:
                dist_res = torch.mm(dt1, dt2)
            expected_placements = tuple(
                Partial(reduce_op) for _ in range(device_mesh.ndim)
            )
            self.assertEqual(dist_res.placements, expected_placements)
            # Per-input linearity keeps one input as-is and redistributes
            # the other from Partial to Replicate via one all-reduce per
            # Partial mesh dim with size > 1.
            expected_allreduce_count = sum(
                1
                for i, p in enumerate(placements)
                if isinstance(p, Partial) and device_mesh.size(i) > 1
            )
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.all_reduce],
                expected_allreduce_count,
            )
            self.assertEqual(comm_mode.get_total_counts(), expected_allreduce_count)
            # Numeric check: redistribute to Replicate to materialize the full
            # result, then compare against the ground truth computed from the
            # full (all-reduced) inputs.
            full_res = dist_res.full_tensor()
            full_a = dt1.full_tensor()
            full_b = dt2.full_tensor()
            expected_val = torch.mm(full_a, full_b)
            self.assertEqual(full_res, expected_val)

        for mesh_shape in mesh_shapes:
            device_mesh = init_device_mesh(self.device_type, mesh_shape)
            for reduce_op in Partial.LINEAR_REDUCE_OPS:
                _run_mm(device_mesh, reduce_op)

        # Also verify mixed Partial placements across mesh dims: on a 2D mesh,
        # left=P(op)R and right=RP(op) should produce output=P(op)P(op)
        # with no communication, matching the full tensor result.
        device_mesh = init_device_mesh(self.device_type, (self.world_size // 2, 2))
        M, K, N = 16, 12, 20
        for reduce_op in Partial.LINEAR_REDUCE_OPS:
            a_local = torch.randn(M, K, device=self.device_type)
            b_local = torch.randn(K, N, device=self.device_type)

            dt_a = DTensor.from_local(
                a_local,
                device_mesh,
                [Partial(reduce_op), Replicate()],
                run_check=False,
            )
            dt_b = DTensor.from_local(
                b_local,
                device_mesh,
                [Replicate(), Partial(reduce_op)],
                run_check=False,
            )

            comm_mode = CommDebugMode()
            with comm_mode:
                dist_res = torch.mm(dt_a, dt_b)

            self.assertEqual(
                dist_res.placements,
                (Partial(reduce_op), Partial(reduce_op)),
            )
            self.assertEqual(comm_mode.get_total_counts(), 0)

            full_res = dist_res.full_tensor()
            full_a = dt_a.full_tensor()
            full_b = dt_b.full_tensor()
            self.assertEqual(full_res, torch.mm(full_a, full_b))

    @with_comms
    @skip_unless_torch_gpu
    def test_scaled_dot_product_attention(self):
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()
        head_dim = 8
        if self.device_type == "xpu":
            head_dim = 64
        # bsz, n_heads, slen, head_dim
        query = torch.rand(
            (4, 8, 8, head_dim),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.rand(
            (4, 8, 8, head_dim),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.rand(
            (4, 8, 8, head_dim),
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
    @parametrize("backend", ["cublaslt", "cutlass"])
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
                # Keep the local BF16 row stride unaligned on 2- and 4-rank meshes
                "inp_shape": (64, 16),
                "w1_shape": (2, 16, 8),
                "w2_shape": (2, 8, 16),
                "inp_placements": [Replicate()],
                "w1_placements": [Shard(2)],
                "w2_placements": [Shard(1)],
                "expected_comm_counts_fwd": 2,
                "expected_comm_counts_bwd": 4,
                "expected_out_placements": [Replicate()],
            },
        ],
    )
    def test_grouped_mm(self, backend, kwargs):
        if backend == "cublaslt":
            if _get_torch_cuda_version() < (13, 3):
                self.skipTest("cublaslt grouped gemm requires CUDA Toolkit >= 13.3")
            sm_major = torch.cuda.get_device_capability()[0]
            if sm_major < 9 or sm_major >= 12:
                self.skipTest("cublaslt grouped gemm requires SM 9.0-11.0")
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

        prev = torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm
        torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm = backend == "cublaslt"
        self.addCleanup(
            setattr,
            torch.backends.cuda.matmul,
            "prefer_cublaslt_grouped_gemm",
            prev,
        )

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

    @with_comms
    def test_constant_pad_nd(self):
        """constant_pad_nd: shard non-padded, replicate padded, Partial iff value==0."""
        device_mesh = self.build_device_mesh()
        t = torch.randn(8, 6, device=self.device_type)
        pad = [1, 1]  # pad last dim only
        expected = torch.nn.functional.pad(t, pad, value=0.0)

        # Shard on non-padded dim (dim 0) — should work directly
        dt = distribute_tensor(t, device_mesh, [Shard(0)])
        result = torch.nn.functional.pad(dt, pad, value=0.0)
        self.assertEqual(result.full_tensor(), expected)

        # Shard on padded dim (dim 1) — forces redistribute to Replicate
        dt = distribute_tensor(t, device_mesh, [Shard(1)])
        result = torch.nn.functional.pad(dt, pad, value=0.0)
        self.assertEqual(result.full_tensor(), expected)

        # Partial input with value=0 — Partial passes through
        dt = distribute_tensor(t, device_mesh, [Partial()])
        result = torch.nn.functional.pad(dt, pad, value=0.0)
        self.assertEqual(result.placements, (Partial(),))
        self.assertEqual(result.full_tensor(), expected)

        # Partial input with value!=0 — forces redistribute to Replicate
        expected_nz = torch.nn.functional.pad(t, pad, value=1.0)
        dt = distribute_tensor(t, device_mesh, [Partial()])
        result = torch.nn.functional.pad(dt, pad, value=1.0)
        self.assertNotEqual(result.placements, (Partial(),))
        self.assertEqual(result.full_tensor(), expected_nz)


instantiate_parametrized_tests(DistMatrixOpsTest)

DistMatrixOpsTestWithLocalTensor = create_local_tensor_test_class(
    DistMatrixOpsTest,
)

if __name__ == "__main__":
    run_tests()
