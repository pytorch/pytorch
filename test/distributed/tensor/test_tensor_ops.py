# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, skipIfRocm
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorConverter,
    DTensorTestBase,
    with_comms,
)


class DistTensorOpsTest(DTensorTestBase):
    @with_comms
    def test_aten_contiguous(self):
        # this op not covered by dtensor_ops
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        self._test_op(
            mesh,
            lambda x: torch.ops.aten.contiguous(x),
            torch.randn(16, 32),
        )

    @with_comms
    def test_detach(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        tensor_to_detach = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_detach, device_mesh, shard_spec)
        detached_mat = mat.detach()
        self.assertFalse(detached_mat is mat)

    @with_comms
    def test_clone(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        specs = [[Replicate()], [Shard(0)]]
        tensor_to_clone = torch.randn(12, 8, requires_grad=True)
        for spec in specs:
            mat = distribute_tensor(tensor_to_clone, device_mesh, spec)
            cloned_mat = mat.clone()
            self.assertFalse(cloned_mat is mat)
            self.assertEqual(cloned_mat.to_local(), mat.to_local())

    @with_comms
    def test_contiguous(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        tensor = torch.rand(3, 5, 6, requires_grad=True)
        sharding = [Shard(0)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        self.assertTrue(dist_tensor.is_contiguous())
        # shard on dim 0 should not change stride (30, 6, 1)
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        new_dt = dist_tensor.transpose(0, 2)
        self.assertFalse(new_dt.is_contiguous())
        self.assertFalse(new_dt.to_local().is_contiguous())
        # check stride
        self.assertEqual(new_dt.stride(), (1, 6, 30))

        new_dt = new_dt.contiguous()
        self.assertTrue(new_dt.is_contiguous())
        self.assertTrue(new_dt.to_local().is_contiguous())
        # check stride
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        # check backward
        new_dt.to_local().sum().backward()
        self.assertEqual(tensor.grad, torch.ones(3, 5, 6))

    @with_comms
    def test_inplace_op(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_tensor = torch.randn((12, 3), device=self.device_type)
        dt_to_add = distribute_tensor(input_tensor, mesh, [Shard(0)])
        dt_to_mul = dt_to_add.clone()
        expected_add_dt = dt_to_add.clone() + 3
        add_res = dt_to_add.add_(3)
        expected_mul_dt = dt_to_mul.clone() * 3
        mul_res = dt_to_mul.mul_(3)
        # inplace op should be the same instance before and after
        self.assertTrue(add_res is dt_to_add)
        self.assertEqual(add_res.to_local(), expected_add_dt.to_local())

        self.assertTrue(mul_res is dt_to_mul)
        self.assertEqual(mul_res.to_local(), expected_mul_dt.to_local())

        # test inplace op self and other dtensor with other specs
        # and make sure out spec not change
        shard_spec = [Shard(0)]
        partial_spec = [Partial()]
        dt_to_inplace_add = distribute_tensor(input_tensor, mesh, shard_spec)
        partial_grad = DTensor.from_local(torch.randn(12, 3), mesh, partial_spec)
        res = dt_to_inplace_add.add_(partial_grad)
        self.assertTrue(res is dt_to_inplace_add)
        self.assertTrue(res.placements == tuple(shard_spec))

    @with_comms
    def test_op_out_variant(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_tensor = torch.randn((12, 3), device=self.device_type)
        sharded_dt_input = distribute_tensor(input_tensor, mesh, [Shard(0)])
        expected_dt = sharded_dt_input.clone() + 3
        sharded_dt_out = sharded_dt_input.clone()
        res = torch.add(sharded_dt_input, 3, out=sharded_dt_out)
        # op out variant should be the same instance before and after
        self.assertTrue(res is sharded_dt_out)
        self.assertEqual(sharded_dt_out.to_local(), expected_dt.to_local())

        # test op out variant with other spec and make sure out spec not change
        replica_spec = [Replicate()]
        replicate_out = distribute_tensor(input_tensor, mesh, replica_spec)
        expected_dt = replicate_out.clone() + 3
        res = torch.add(sharded_dt_input, 3, out=replicate_out)
        self.assertTrue(res is replicate_out)
        self.assertTrue(res.placements == tuple(replica_spec))
        self.assertEqual(replicate_out.to_local(), expected_dt.to_local())

    @with_comms
    def test_empty_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        empty_like_dt = torch.empty_like(dist_tensor)
        # empty is not deterministic, so we only check that the shard propagation worked
        self.assertEqual((4, 8), empty_like_dt.to_local().shape)

    @with_comms
    def test_fill_inplace(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.fill_(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())
        self.assertEqual(full_expected, dist_tensor.to_local())

    @with_comms
    def test_full_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.full_like(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())

    @with_comms
    def test_ones_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        ones_like_dt = torch.ones_like(dist_tensor)
        ones_expected = torch.ones(4, 8)
        self.assertEqual(ones_expected, ones_like_dt.to_local())

    @with_comms
    def test_ones_like_partial_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        assert dist_tensor.shape == (4, 8)

        ones_like_dt = torch.ones_like(dist_tensor)
        ones_expected = torch.ones(dist_tensor.shape)
        self.assertEqual(ones_expected, ones_like_dt.full_tensor())

    @with_comms
    def test_fill_inplace_partial_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        assert dist_tensor.shape == (4, 8)

        # inplace partial sum should keep partial
        torch.fill_(dist_tensor, 8)
        fill_expected = torch.full(
            dist_tensor.shape, 8 * self.world_size, dtype=input_tensor.dtype
        )
        self.assertEqual(fill_expected, dist_tensor.full_tensor())

    @with_comms
    def test_zeros_like_partial_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        assert dist_tensor.shape == (4, 8)

        zeros_like_dt = torch.zeros_like(dist_tensor)
        zeros_expected = torch.zeros(dist_tensor.shape)
        self.assertEqual(zeros_expected, zeros_like_dt.full_tensor())

    @with_comms
    def test_zero_inplace(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zero_(dist_tensor)
        zeros_expected = torch.zeros(4, 8)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())
        self.assertEqual(zeros_expected, dist_tensor.to_local())

    @with_comms
    def test_zeros_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zeros_like(dist_tensor, dtype=torch.bfloat16)
        zeros_expected = torch.zeros(4, 8, dtype=torch.bfloat16)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())
        # make sure there is no side effect on the input tensor dtype
        self.assertEqual(dist_tensor.dtype, torch.float32)
        self.assertEqual(zeros_like_dt.dtype, torch.bfloat16)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_stack(self):
        mesh_2d = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        partial_replicate_placement = [Partial(), Replicate()]
        partial_placement = [Partial(), Partial()]

        partial_replicate_dt = DTensor.from_local(
            torch.randn(4, 8), mesh_2d, partial_replicate_placement
        )
        partial_dt = DTensor.from_local(torch.randn(4, 8), mesh_2d, partial_placement)

        stack_dt = torch.stack([partial_replicate_dt, partial_dt])
        self.assertEqual(stack_dt.placements, tuple(partial_placement))
        self.assertEqual(stack_dt.shape, (2, 4, 8))

        mesh_1d = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # stack before/after shard dim
        global_input = torch.randn(8, 8)
        shard1_input = distribute_tensor(global_input, mesh_1d, [Shard(1)])
        cloned_shard1_input = shard1_input.clone()
        stack_shard1_dt = torch.stack([shard1_input, cloned_shard1_input])
        self.assertEqual(stack_shard1_dt.placements, (Shard(2),))
        self.assertEqual(stack_shard1_dt.shape, (2, 8, 8))
        self.assertEqual(
            stack_shard1_dt.full_tensor(), torch.stack([global_input, global_input])
        )

        stack_dim1_shard1_dt = torch.stack([shard1_input, cloned_shard1_input], dim=1)
        self.assertEqual(stack_dim1_shard1_dt.placements, (Shard(2),))
        self.assertEqual(stack_dim1_shard1_dt.shape, (8, 2, 8))
        self.assertEqual(
            stack_dim1_shard1_dt.full_tensor(),
            torch.stack([global_input, global_input], dim=1),
        )

    @with_comms
    def test_equal(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor_1 = torch.ones(4, 4)
        dist_tensor_1 = DTensor.from_local(input_tensor_1, device_mesh, shard_spec)

        # tensors are equal
        input_tensor_2 = torch.ones(4, 4)
        dist_tensor_2 = DTensor.from_local(input_tensor_2, device_mesh, shard_spec)

        eq_result = dist_tensor_1.equal(dist_tensor_2)
        self.assertTrue(eq_result)

        # tensors are different on some shards
        if self.rank == 0:
            input_tensor_2 = torch.ones(4, 4)
        else:
            input_tensor_2 = torch.randn(4, 4)
        dist_tensor_2 = DTensor.from_local(input_tensor_2, device_mesh, shard_spec)

        eq_result = dist_tensor_1.equal(dist_tensor_2)
        # equal op all reduces each shard's local result
        self.assertFalse(eq_result)
        self.assertTrue(dist_tensor_1.is_same_size(dist_tensor_2))

        # test if sharding are different
        replica_spec = [Replicate()]
        global_input = torch.ones(4 * self.world_size, 4)
        dist_tensor_3 = DTensor.from_local(
            global_input, device_mesh, replica_spec, run_check=False
        )

        self.assertTrue(dist_tensor_1.equal(dist_tensor_3))
        self.assertTrue(dist_tensor_1.is_same_size(dist_tensor_3))

        # test sharding difference with only some shards content difference
        self.assertFalse(dist_tensor_2.equal(dist_tensor_3))
        self.assertTrue(dist_tensor_1.is_same_size(dist_tensor_3))
        self.assertFalse(input_tensor_2.is_same_size(dist_tensor_3))

    def _test_op(self, mesh, op_call, *args, **kwargs):
        out = op_call(*args, **kwargs)
        dtc = DTensorConverter(mesh, args, kwargs)
        for d_args, d_kwargs in dtc:
            self.assertTrue(dtc.successful())
            d_out = op_call(*d_args, **d_kwargs)
            self.assertEqual(d_out.full_tensor(), out)

    @with_comms
    def test_new_full(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        comm_mode = CommDebugMode()

        global_tensor = torch.randn(12, 8)
        placements = [[Shard(0)], [Replicate()]]
        for placement in placements:
            input_dt = distribute_tensor(global_tensor, device_mesh, placement)
            with comm_mode:
                new_full_diff_dt = input_dt.new_full((4, 8), 42.0)
                # new_full_diff_dt creates a replicated tensor, regardless of input_dt placement,
                # which should not trigger any communication.
                self.assertEqual(comm_mode.get_total_counts(), 0)
            new_full_diff_expected = torch.full((4, 8), 42.0)
            self.assertTrue(new_full_diff_dt.placements[0].is_replicate())
            self.assertEqual(new_full_diff_expected, new_full_diff_dt.to_local())

            with comm_mode:
                new_full_same_dt = input_dt.new_full((12, 8), 42.0)
                # new_full_same_dt creates a tensor with the same placement as input_dt,
                # which should not trigger any communication.
                self.assertEqual(comm_mode.get_total_counts(), 0)
            new_full_same_expected = torch.full((12, 8), 42.0)
            self.assertEqual(new_full_same_dt.placements, placement)
            self.assertEqual(new_full_same_expected, new_full_same_dt.full_tensor())

    @with_comms
    def test_new_empty_strided(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        comm_mode = CommDebugMode()

        shard_dim = 1
        placement = (Shard(shard_dim),)

        # output shape same as input shape, evenly sharded input -> output same sharding as input
        global_tensor = torch.randn(12, 8)
        input_dt = distribute_tensor(global_tensor, device_mesh, placement)
        self.assertTrue(input_dt.shape[shard_dim] % self.world_size == 0)
        with comm_mode:
            new_empty_strided_dt = input_dt.new_empty_strided((12, 8), (8, 1))
            self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(new_empty_strided_dt.placements, placement)
        self.assertEqual(
            new_empty_strided_dt._local_tensor.size(), (12, 8 // self.world_size)
        )
        self.assertEqual(
            new_empty_strided_dt._local_tensor.stride(), (8 // self.world_size, 1)
        )
        self.assertTrue(new_empty_strided_dt.contiguous() is new_empty_strided_dt)

        # output shape same as input shape, unevenly sharded input -> output replicated
        global_tensor = torch.randn(12, 7)
        input_dt = distribute_tensor(global_tensor, device_mesh, placement)
        self.assertTrue(input_dt.shape[shard_dim] % self.world_size != 0)
        with comm_mode:
            new_empty_strided_dt = input_dt.new_empty_strided((12, 7), (7, 1))
            self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(new_empty_strided_dt.placements, (Replicate(),))
        self.assertEqual(new_empty_strided_dt._local_tensor.size(), (12, 7))
        self.assertEqual(new_empty_strided_dt._local_tensor.stride(), (7, 1))

        # output shape different from input shape -> output replicated
        global_tensor = torch.randn(12, 8)
        input_dt = distribute_tensor(global_tensor, device_mesh, placement)
        with comm_mode:
            new_empty_strided_dt = input_dt.new_empty_strided((12, 4), (4, 1))
            self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(new_empty_strided_dt.placements, (Replicate(),))
        self.assertEqual(new_empty_strided_dt._local_tensor.size(), (12, 4))
        self.assertEqual(new_empty_strided_dt._local_tensor.stride(), (4, 1))

    @with_comms
    def test_scatter(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        comm_mode = CommDebugMode()

        # case 1 all replicate: input replicated, index/src replicated, output replicated
        global_indexs = [
            torch.tensor([[0, 1, 2, 0]]),
            torch.tensor([[0, 1, 2], [0, 1, 4]]),
        ]
        for scatter_dim in [0, 1]:
            srcs = [torch.arange(1, 11).reshape((2, 5)), 4]
            for global_src in srcs:
                global_input = torch.zeros(3, 5, dtype=torch.int64)
                global_index = global_indexs[scatter_dim]

                input_dt = distribute_tensor(
                    global_input.clone(), device_mesh, [Replicate()]
                )
                index_dt = distribute_tensor(global_index, device_mesh, [Replicate()])
                if isinstance(global_src, torch.Tensor):
                    src_dt = distribute_tensor(global_src, device_mesh, [Replicate()])
                else:
                    src_dt = global_src
                global_output = torch.scatter(
                    global_input, scatter_dim, global_index, global_src
                )
                with comm_mode:
                    output_dt = torch.scatter(input_dt, scatter_dim, index_dt, src_dt)

                self.assertEqual(comm_mode.get_total_counts(), 0)
                self.assertEqual(output_dt.placements, [Replicate()])
                self.assertEqual(output_dt.to_local(), global_output)

    @with_comms
    def test_gather(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        comm_mode = CommDebugMode()

        # case 1 all replicate: input replicated, index replicated, output replicated
        global_input = torch.randn(12, 8, 16)
        global_index = torch.randint(8, (4, 4, 8))
        input_dt = distribute_tensor(global_input, device_mesh, [Replicate()])
        index_dt = distribute_tensor(global_index, device_mesh, [Replicate()])
        for gather_dim in [0, 1, 2]:
            global_output = torch.gather(global_input, gather_dim, global_index)
            with comm_mode:
                output_dt = torch.gather(input_dt, gather_dim, index_dt)
                self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(output_dt.placements, [Replicate()])
            self.assertEqual(output_dt.to_local(), global_output)

        # case 2 input sharding: input sharded, index replicated, output mask partial
        # only works when index has size 1 on the gather dimension and
        # input is sharded on the gather dimension
        from torch.distributed.tensor._ops._embedding_ops import _MaskPartial

        gather_dim = 1
        global_input = torch.randn(12, 8, 16)
        global_index = torch.randint(8, (4, 1, 8))
        global_output = torch.gather(global_input, gather_dim, global_index)
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(gather_dim)])
        index_dt = distribute_tensor(global_index, device_mesh, [Replicate()])
        with comm_mode:
            output_dt = torch.gather(input_dt, gather_dim, index_dt)
            self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertIsInstance(output_dt.placements[0], _MaskPartial)
        self.assertEqual(output_dt.full_tensor(), global_output)

        # case 3 index sharding: input replicated, index sharded, output sharded
        # only works when the sharding dimension is the gather dimension
        global_input = torch.randn(12, 8, 16)
        global_index = torch.randint(8, (4, 4, 8))
        for gather_dim in range(len(global_index.shape)):
            input_dt = distribute_tensor(global_input, device_mesh, [Replicate()])
            index_dt = distribute_tensor(global_index, device_mesh, [Shard(gather_dim)])
            global_output = torch.gather(global_input, gather_dim, global_index)
            with comm_mode:
                output_dt = torch.gather(input_dt, gather_dim, index_dt)
                self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(output_dt.placements, [Shard(gather_dim)])
            self.assertEqual(output_dt.full_tensor(), global_output)

    @skipIfRocm
    @with_comms
    def test_index(self):
        meshes = [
            DeviceMesh(self.device_type, list(range(self.world_size))),  # 1D mesh
            # TODO(@azzolini): un-comment when DTensorConverter supports N-D mesh
            # DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, -1)), # 2D mesh
        ]
        for mesh in meshes:
            self._test_op(
                mesh,
                lambda x, y: x[y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 8)),
            )
            self._test_op(
                mesh,
                lambda x, y: x.index_select(1, y),
                torch.randn(16, 32, 16),
                torch.randint(5, (4,)),
            )
            self._test_op(
                mesh,
                lambda x, y: x.index_select(0, y),
                torch.randn(16, 32, 16),
                torch.randint(5, (4,)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[y],
                torch.randn(16, 32, 16),
                torch.randint(5, (12,)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[:, y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 8)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[..., y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[..., y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 8, 16)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, y],
                torch.randn(16, 32, 16),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 8, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, :, y],
                torch.randn(16, 32, 16),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 8, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[:, z, :, y],
                torch.randn(16, 32, 16, 12),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 8, 12)),
            )
            # broadcast in inner dimensions
            self._test_op(
                mesh,
                lambda x, y, z: x[:, z, :, y],
                torch.randn(16, 32, 16, 12),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 1, 12)),
            )
            # implicit (left-padded) broadcast
            self._test_op(
                mesh,
                lambda x, y, z: x[:, z, :, y],
                torch.randn(16, 32, 16, 12),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (8, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, y, :, :],
                torch.randn(16, 32, 16, 12),
                torch.randint(2, (8, 12)),
                torch.randint(5, (12, 8, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, :, y, :],
                torch.randn(16, 32, 16, 12),
                torch.randint(2, (8, 12)),
                torch.randint(5, (12, 8, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, :, :, y],
                torch.randn(16, 32, 16, 12),
                torch.randint(2, (8, 1)),
                torch.randint(5, (12, 8, 12)),
            )

    @with_comms
    def test_index_put_scalar(self):
        device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        global_input = torch.randn(2, 4, 8, device=self.device_type)
        global_index = [
            torch.randint(global_input.shape[i], size=(), device=self.device_type)
            for i in range(3)
        ]
        global_value = torch.randn(size=(), device=self.device_type)
        value_dt = distribute_tensor(
            global_value, device_mesh, [Replicate(), Replicate()]
        )
        placement_choice_pool = [Shard(0), Shard(1), Replicate()]
        for i in placement_choice_pool:
            for j in placement_choice_pool:
                input_dt = distribute_tensor(global_input, device_mesh, [i, j])
                ref = torch.index_put(global_input, global_index, global_value)
                output_dt = torch.index_put(input_dt, global_index, value_dt)
                assert isinstance(output_dt, DTensor)
                # for value is a scalar case, output placement must be replicate
                self.assertEqual(output_dt.placements, (Replicate(), Replicate()))
                self.assertEqual(output_dt.full_tensor(), ref)

    @with_comms
    def test_index_put_tensor(self):
        device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        global_input = torch.randn(2, 4, 8, device=self.device_type)
        global_index = [
            torch.randint(global_input.shape[0], size=(), device=self.device_type)
        ]
        global_value = torch.zeros([4, 8], device=self.device_type)
        value_dt = distribute_tensor(global_value, device_mesh, [Shard(1), Replicate()])
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(0), Replicate()])
        ref = torch.index_put(global_input, global_index, global_value)
        output_dt = torch.index_put(input_dt, global_index, value_dt)
        assert isinstance(output_dt, DTensor)
        # `input_dt` follows `value_dt`'s Shard(1) plus a offset value of
        # global_value.ndim-global_input.ndim, which results in Shard(2)
        self.assertEqual(output_dt.placements, (Shard(2), Replicate()))
        self.assertEqual(output_dt.full_tensor(), ref)

    @with_comms
    def test_where_type_promotion(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))  # 1D mesh

        specs = [[Shard(0)], [Replicate()]]
        for spec in specs:
            global_tensor = torch.randn(12, 8)
            mat = distribute_tensor(global_tensor, mesh, spec)
            res = torch.where(mat > 0, 1, 0)
            ref = torch.where(global_tensor > 0, 1, 0)
            self.assertEqual(res.full_tensor(), ref)

    @with_comms
    def test_dtensor_dtype_conversion(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        # by default we start from bf16 dtype
        local_tenor = torch.randn(2, 8, dtype=torch.bfloat16)
        bf16_sharded_dtensor = DTensor.from_local(local_tenor, device_mesh, shard_spec)
        self.assertEqual(bf16_sharded_dtensor.dtype, torch.bfloat16)
        self.assertEqual(bf16_sharded_dtensor.to_local().dtype, torch.bfloat16)

        # convert to float dtype
        fp32_sharded_dtensor = bf16_sharded_dtensor.float()
        self.assertEqual(fp32_sharded_dtensor.dtype, torch.float32)
        self.assertEqual(fp32_sharded_dtensor.to_local().dtype, torch.float32)

        # convert back to bf16 dtype
        bf16_sharded_dtensor1 = fp32_sharded_dtensor.type_as(bf16_sharded_dtensor)
        self.assertEqual(bf16_sharded_dtensor1.dtype, torch.bfloat16)
        self.assertEqual(bf16_sharded_dtensor1.to_local().dtype, torch.bfloat16)

        from torch.distributed.tensor.debug import _get_sharding_prop_cache_info

        # by this point we only have cache misses
        hits, misses, _, _ = _get_sharding_prop_cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 2)

        # convert to fp32 again and see if there's cache hit
        bf16_sharded_dtensor1.float()
        hits, misses, _, _ = _get_sharding_prop_cache_info()
        # by now we should have cache hit
        self.assertEqual(hits, 1)
        self.assertEqual(misses, 2)

    @with_comms
    def test_slice(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))  # 1D mesh
        comm_mode = CommDebugMode()

        shard_spec = [Shard(1)]
        global_tensor = torch.randn(8, 16, requires_grad=True)
        sharded_dtensor = distribute_tensor(global_tensor, mesh, shard_spec)

        global_out = global_tensor[:, 8:]
        with comm_mode:
            sharded_out = sharded_dtensor[:, 8:]

        self.assertEqual(comm_mode.get_total_counts(), 1)

        global_out.backward(gradient=torch.ones_like(global_out))
        with comm_mode:
            sharded_out_grad = torch.distributed.tensor.ones(
                sharded_out.shape, device_mesh=mesh, placements=shard_spec
            )
            sharded_out.backward(gradient=sharded_out_grad)

        self.assertEqual(comm_mode.get_total_counts(), 1)

        self.assertEqual(sharded_out.full_tensor(), global_out)
        self.assertEqual(sharded_dtensor.grad.full_tensor(), global_tensor.grad)


if __name__ == "__main__":
    run_tests()
