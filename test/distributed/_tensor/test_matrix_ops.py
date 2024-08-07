# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools
from typing import cast, List, Optional

import torch
import torch.nn.functional as F
from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_unless_torch_gpu,
    with_comms,
)


class DistMatrixOpsTest(DTensorTestBase):
    @with_comms
    def test_addmm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_shard = torch.randn(12, 8)
        mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        tensor_to_replicate = torch.randn(8, 4)
        mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)
        input_tensor = torch.randn(4)
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)

        dist_res = torch.addmm(input, mat1, mat2)
        local_res = torch.addmm(input_tensor, tensor_to_shard, tensor_to_replicate)
        self.assertEqual(dist_res.full_tensor(), local_res)

    @with_comms
    def test_addmm_empty_operand(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
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
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
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

    @with_comms
    def test_mm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = Shard(0)
        shard1_spec = Shard(1)
        replica_spec = Replicate()

        t1 = torch.randn(12, 8, requires_grad=True)
        t2 = torch.randn(8, 16, requires_grad=True)
        local_res = torch.mm(t1, t2)

        def test_placement_comb(
            placements1: List[Placement], placements2: List[Placement]
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
    def test_t(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
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
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

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
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        tensor = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        batch_1 = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        batch_2 = torch.rand(4, 8, 8, device=self.device_type, requires_grad=True)

        def test_placement_comb(
            tensor_placements: List[Placement],
            batch_1_placements: List[Placement],
            batch_2_placements: List[Placement],
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
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        mat1 = torch.rand(4, 8, 4, device=self.device_type, requires_grad=True)
        mat2 = torch.rand(4, 4, 8, device=self.device_type, requires_grad=True)
        local_result = torch.bmm(mat1, mat2)
        grad_local_res = torch.ones_like(local_result)
        local_result.backward(grad_local_res)

        def test_placement_comb(
            placements1: List[Placement],
            placements2: List[Placement],
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
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
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

        dist_query = distribute_tensor(query, device_mesh, [Shard(1)])
        dist_key = distribute_tensor(key, device_mesh, [Shard(1)])
        dist_value = distribute_tensor(value, device_mesh, [Shard(1)])

        from torch.nn.attention import sdpa_kernel, SDPBackend

        available_backends = []
        dropout_p = 0.0
        # TODO: Add test cases where is_causal=False and an attention mask is provided.
        #       Gaps include missing op support for aten.masked_fill_.Scalar.
        is_causal = True
        params = torch.backends.cuda.SDPAParams(
            query, key, value, None, dropout_p, is_causal
        )
        if torch.backends.cuda.can_use_flash_attention(params, debug=False):
            available_backends.append(SDPBackend.FLASH_ATTENTION)
        if torch.backends.cuda.can_use_efficient_attention(params, debug=False):
            available_backends.append(SDPBackend.EFFICIENT_ATTENTION)

        for backend in available_backends:
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
                    self.assertTrue(dist_out.placements[0].is_shard(dim=1))
                    self.assertEqual(dist_out.full_tensor(), out)

                out.sum().backward()
                with comm_mode:
                    dist_out.sum().backward()
                    self.assertEqual(comm_mode.get_total_counts(), 0)
                    self.assertTrue(dist_query.grad.placements[0].is_shard(dim=1))
                    self.assertEqual(dist_query.grad.full_tensor(), query.grad)
                    self.assertTrue(dist_key.grad.placements[0].is_shard(dim=1))
                    self.assertEqual(dist_key.grad.full_tensor(), key.grad)
                    self.assertTrue(dist_value.grad.placements[0].is_shard(dim=1))
                    self.assertEqual(dist_value.grad.full_tensor(), value.grad)


if __name__ == "__main__":
    run_tests()
