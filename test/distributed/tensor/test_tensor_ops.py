# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools
import unittest

import torch
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._sharding_prop import ShardingPropagator
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import MI200_ARCH, run_tests, skipIfRocmArch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorConverter,
    DTensorTestBase,
    LocalDTensorTestBase,
    with_comms,
)


class DistTensorOpsTest(DTensorTestBase):
    @with_comms
    def test_aten_contiguous(self):
        # this op not covered by dtensor_ops
        mesh = self.build_device_mesh()
        self._test_op(
            mesh,
            lambda x: torch.ops.aten.contiguous(x),
            torch.randn(16, 32),
        )

    @with_comms
    def test_detach(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        tensor_to_detach = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_detach, device_mesh, shard_spec)
        detached_mat = mat.detach()
        self.assertFalse(detached_mat is mat)

    @with_comms
    def test_clone(self):
        device_mesh = self.build_device_mesh()
        specs = [[Replicate()], [Shard(0)]]
        tensor_to_clone = torch.randn(12, 8, requires_grad=True)
        for spec in specs:
            mat = distribute_tensor(tensor_to_clone, device_mesh, spec)
            cloned_mat = mat.clone()
            self.assertFalse(cloned_mat is mat)
            self.assertEqual(cloned_mat.to_local(), mat.to_local())

    @with_comms
    def test_copy_(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # basic test
        src_tensor = torch.randn((12, 12))
        dst_tensor = torch.zeros(12, 12)
        src_specs = [[Replicate()], [Shard(0)]]
        dst_specs = [[Replicate()], [Shard(0)]]
        for dst_spec, src_spec in zip(dst_specs, src_specs):
            src_dtensor = distribute_tensor(src_tensor, device_mesh, dst_spec)
            dst_dtensor = distribute_tensor(dst_tensor, device_mesh, src_spec)
            dst_dtensor.copy_(src_dtensor)
            dst_tensor.copy_(src_tensor)
            self.assertEqual(dst_dtensor.full_tensor(), dst_tensor)

        # simple broadcasting
        src_tensor = torch.randn((128,))
        dst_tensor = torch.zeros(128, 128)
        src_specs = [[Replicate()], [Shard(0)]]
        dst_specs = [[Replicate()], [Shard(1)]]
        for dst_spec, src_spec in zip(dst_specs, src_specs):
            src_dtensor = distribute_tensor(src_tensor, device_mesh, src_spec)
            dst_dtensor = distribute_tensor(dst_tensor, device_mesh, dst_spec)
            dst_dtensor.copy_(src_dtensor)
            dst_tensor.copy_(src_tensor)
            self.assertEqual(dst_dtensor.full_tensor(), dst_tensor)

        # The src specs in this case are designed to not be compatible with the dst_specs, redistribute should happen
        src_tensor = torch.randn((64, 1))
        dst_tensor = torch.zeros(16, 32, 64, 128)
        src_specs = [[Shard(1)], [Shard(1)], [Shard(1)], [Shard(1)]]
        dst_specs = [[Replicate()], [Shard(0)], [Shard(1)], [Shard(2)]]
        for dst_spec, src_spec in zip(dst_specs, src_specs):
            src_dtensor = distribute_tensor(src_tensor, device_mesh, src_spec)
            dst_dtensor = distribute_tensor(dst_tensor, device_mesh, dst_spec)
            dst_dtensor.copy_(src_dtensor)
            dst_tensor.copy_(src_tensor)
            self.assertEqual(dst_dtensor.full_tensor(), dst_tensor)

        # as a pointwise op, need to keep Partial placements without redistribute
        src_tensor = torch.randn((64, 1))
        dst_tensor = torch.zeros(16, 32, 64, 128)
        src_specs = [[Partial()]]
        dst_specs = [[Partial()]]
        for dst_spec, src_spec in zip(dst_specs, src_specs):
            src_dtensor = DTensor.from_local(src_tensor, device_mesh, src_spec)
            dst_dtensor = DTensor.from_local(dst_tensor, device_mesh, dst_spec)
            dst_dtensor.copy_(src_dtensor)
            dst_tensor.copy_(src_tensor)
            self.assertEqual(dst_dtensor.placements, (Partial(),))
            self.assertEqual(dst_dtensor._local_tensor, dst_tensor)

        # test that copy_ preserves any Partial type, not just sum/avg
        for reduce_op in ["max", "min"]:
            src_tensor = torch.randn((64, 1))
            dst_tensor = torch.zeros(16, 32, 64, 128)
            partial_placement = Partial(reduce_op)
            src_dtensor = DTensor.from_local(
                src_tensor, device_mesh, [partial_placement]
            )
            dst_dtensor = DTensor.from_local(
                dst_tensor, device_mesh, [partial_placement]
            )
            dst_dtensor.copy_(src_dtensor)
            dst_tensor.copy_(src_tensor)
            self.assertEqual(dst_dtensor.placements, (partial_placement,))
            self.assertEqual(dst_dtensor._local_tensor, dst_tensor)

    @with_comms
    def test_contiguous(self):
        device_mesh = self.build_device_mesh()
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
        mesh = self.build_device_mesh()
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
        mesh = self.build_device_mesh()
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
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        empty_like_dt = torch.empty_like(dist_tensor)
        # empty is not deterministic, so we only check that the shard propagation worked
        self.assertEqual((4, 8), empty_like_dt.to_local().shape)

    @with_comms
    def test_meta_init_partial(self):
        device_mesh = self.build_device_mesh()
        partial_spec = [Partial()]

        class ToyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "scalar_buffer", torch.tensor(0.0, dtype=torch.float32)
                )

        with torch.device("meta"):
            module = ToyModule()
            module._buffers["scalar_buffer"] = DTensor.from_local(
                module.scalar_buffer,
                device_mesh=device_mesh,
                placements=partial_spec,
            )
        module.to_empty(device=None)

        # check that to_empty preserves partial
        self.assertEqual(module.scalar_buffer.placements, (Partial(),))

    @with_comms
    def test_fill_inplace(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.fill_(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())
        self.assertEqual(full_expected, dist_tensor.to_local())

    @with_comms
    def test_full_like(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.full_like(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())

    @with_comms
    def test_ones_like(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        ones_like_dt = torch.ones_like(dist_tensor)
        ones_expected = torch.ones(4, 8)
        self.assertEqual(ones_expected, ones_like_dt.to_local())

    @with_comms
    def test_ones_like_partial_sum(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        if not (dist_tensor.shape == (4, 8)):
            raise AssertionError(f"Expected shape (4, 8), got {dist_tensor.shape}")

        ones_like_dt = torch.ones_like(dist_tensor)
        ones_expected = torch.ones(dist_tensor.shape)
        self.assertEqual(ones_expected, ones_like_dt.full_tensor())

    @with_comms
    def test_fill_inplace_partial_sum(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        if not (dist_tensor.shape == (4, 8)):
            raise AssertionError(f"Expected shape (4, 8), got {dist_tensor.shape}")

        # inplace partial sum should keep partial
        torch.fill_(dist_tensor, 8)
        fill_expected = torch.full(
            dist_tensor.shape, 8 * self.world_size, dtype=input_tensor.dtype
        )
        self.assertEqual(fill_expected, dist_tensor.full_tensor())

    @with_comms
    def test_zeros_like_partial_sum(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        if not (dist_tensor.shape == (4, 8)):
            raise AssertionError(f"Expected shape (4, 8), got {dist_tensor.shape}")

        zeros_like_dt = torch.zeros_like(dist_tensor)
        zeros_expected = torch.zeros(dist_tensor.shape)
        self.assertEqual(zeros_expected, zeros_like_dt.full_tensor())

    @with_comms
    def test_zero_inplace(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zero_(dist_tensor)
        zeros_expected = torch.zeros(4, 8)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())
        self.assertEqual(zeros_expected, dist_tensor.to_local())

    @with_comms
    def test_zeros_like(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zeros_like(dist_tensor, dtype=torch.bfloat16)
        zeros_expected = torch.zeros(4, 8, dtype=torch.bfloat16)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())
        # make sure there is no side effect on the input tensor dtype
        self.assertEqual(dist_tensor.dtype, torch.float32)
        self.assertEqual(zeros_like_dt.dtype, torch.bfloat16)

    @skip_if_lt_x_gpu(4)
    @with_comms
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

        # stack with negative dim: dim=-1 inserts at the last position of the
        # output (ndim+1), so Shard(1) should stay Shard(1)
        stack_neg_dim_dt = torch.stack([shard1_input, cloned_shard1_input], dim=-1)
        self.assertEqual(stack_neg_dim_dt.placements, (Shard(1),))
        self.assertEqual(stack_neg_dim_dt.shape, (8, 8, 2))
        self.assertEqual(
            stack_neg_dim_dt.full_tensor(),
            torch.stack([global_input, global_input], dim=-1),
        )

    @with_comms
    def test_stack_cache(self):
        device_mesh = self.build_device_mesh()

        shape = (4, 8)
        placements = [Replicate()]
        dtensor_list = []
        for _ in range(3):
            local_tensor = torch.randn(shape)
            dt = DTensor.from_local(local_tensor, device_mesh, placements)
            dtensor_list.append(dt)

        _ = torch.stack(dtensor_list)

        dtensor_list2 = []
        for _ in range(3):
            local_tensor = torch.randn(shape)
            dt = DTensor.from_local(local_tensor, device_mesh, placements)
            dtensor_list2.append(dt)

        def error(*args, **kwargs):
            raise AssertionError

        with unittest.mock.patch.object(
            ShardingPropagator, "_propagate_tensor_meta_non_cached", error
        ):
            _ = torch.stack(dtensor_list2)

    @with_comms
    def test_equal(self):
        device_mesh = self.build_device_mesh()
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
        device_mesh = self.build_device_mesh()
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
        device_mesh = self.build_device_mesh()
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
        device_mesh = self.build_device_mesh()
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
        device_mesh = self.build_device_mesh()
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
        from torch.distributed.tensor.placement_types import _MaskPartial

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

    @skipIfRocmArch(MI200_ARCH)
    @with_comms
    def test_index(self):
        meshes = [
            self.build_device_mesh(),  # 1D mesh
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
            # Commented out to fix distributed CI timeout: each 3-tensor call
            # generates 40-80 sharding combinations via itertools.product,
            # causing combinatorial explosion.
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[z, :, y],
            #     torch.randn(16, 32, 16),
            #     torch.randint(5, (12, 8, 12)),
            #     torch.randint(2, (12, 8, 12)),
            # )
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[:, z, :, y],
            #     torch.randn(16, 32, 16, 12),
            #     torch.randint(5, (12, 8, 12)),
            #     torch.randint(2, (12, 8, 12)),
            # )
            # broadcast in inner dimensions
            self._test_op(
                mesh,
                lambda x, y, z: x[:, z, :, y],
                torch.randn(16, 32, 16, 12),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 1, 12)),
            )
            # Commented out to fix distributed CI timeout: each 3-tensor call
            # generates 40-80 sharding combinations via itertools.product,
            # causing combinatorial explosion.
            # # implicit (left-padded) broadcast
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[:, z, :, y],
            #     torch.randn(16, 32, 16, 12),
            #     torch.randint(5, (12, 8, 12)),
            #     torch.randint(2, (8, 12)),
            # )
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[z, y, :, :],
            #     torch.randn(16, 32, 16, 12),
            #     torch.randint(2, (8, 12)),
            #     torch.randint(5, (12, 8, 12)),
            # )
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[z, :, y, :],
            #     torch.randn(16, 32, 16, 12),
            #     torch.randint(2, (8, 12)),
            #     torch.randint(5, (12, 8, 12)),
            # )
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[z, :, :, y],
            #     torch.randn(16, 32, 16, 12),
            #     torch.randint(2, (8, 1)),
            #     torch.randint(5, (12, 8, 12)),
            # )

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
                if not isinstance(output_dt, DTensor):
                    raise AssertionError(f"Expected DTensor, got {type(output_dt)}")
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
        for accumulate in [True, False]:
            ref = torch.index_put(global_input, global_index, global_value, accumulate)
            output_dt = torch.index_put(input_dt, global_index, value_dt, accumulate)
            if not isinstance(output_dt, DTensor):
                raise AssertionError(f"Expected DTensor, got {type(output_dt)}")
            # Output should be sharded on non-indexed dims (dim 1 or 2).
            # The exact placement depends on cost-based strategy selection.
            for p in output_dt.placements:
                if isinstance(p, Shard):
                    self.assertIn(p.dim, [1, 2])
            self.assertEqual(output_dt.full_tensor(), ref)

    @with_comms
    def test_index_put_requires_replicated_index(self):
        """Test that index_put correctly replicates sharded indices."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.randn(4, 8, device=self.device_type)
        global_value = torch.zeros([8], device=self.device_type)

        # Create sharded index - should be redistributed to replicated
        idx_tensor = torch.tensor([0, 1], device=self.device_type)
        idx_dt = distribute_tensor(idx_tensor, device_mesh, [Shard(0)])

        input_dt = distribute_tensor(global_input, device_mesh, [Replicate()])
        value_dt = distribute_tensor(global_value, device_mesh, [Replicate()])

        # The op should work - index gets redistributed to replicated internally
        ref = torch.index_put(global_input, [idx_tensor], global_value)
        output_dt = torch.index_put(input_dt, [idx_dt], value_dt)
        self.assertEqual(output_dt.full_tensor(), ref)

    @with_comms
    def test_index_put_no_shard_on_indexed_dim(self):
        """Test that index_put output is not sharded on indexed dims."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.randn(4, 8, device=self.device_type)
        global_value = torch.zeros([8], device=self.device_type)
        global_index = [torch.tensor(0, device=self.device_type)]

        # Shard input on indexed dim 0
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(0)])
        value_dt = distribute_tensor(global_value, device_mesh, [Replicate()])

        ref = torch.index_put(global_input, global_index, global_value)
        output_dt = torch.index_put(input_dt, global_index, value_dt)

        # Verify output is not sharded on the indexed dim (dim 0)
        for p in output_dt.placements:
            if isinstance(p, Shard):
                self.assertNotEqual(
                    p.dim, 0, "Output should not be sharded on indexed dim"
                )
        self.assertEqual(output_dt.full_tensor(), ref)

    @with_comms
    def test_index_put_partial_numerics(self):
        """Test index_put with Partial placements produces correct numerics."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.randn(4, 8, device=self.device_type)
        global_index = [torch.tensor(1, device=self.device_type)]
        global_value = torch.randn(8, device=self.device_type)

        # Create Partial tensors - each rank has partial sum
        local_input = global_input / self.world_size
        local_value = global_value / self.world_size
        input_dt = DTensor.from_local(
            local_input, device_mesh, [Partial()], run_check=False
        )
        value_dt = DTensor.from_local(
            local_value, device_mesh, [Partial()], run_check=False
        )

        for accumulate in [True, False]:
            ref = torch.index_put(global_input, global_index, global_value, accumulate)
            output_dt = torch.index_put(input_dt, global_index, value_dt, accumulate)
            self.assertIsInstance(output_dt, DTensor)
            self.assertEqual(output_dt.placements, (Partial(),))
            self.assertEqual(output_dt.full_tensor(), ref)

    @with_comms
    def test_index_put_duplicated_indices(self):
        """Test index_put with duplicated indices for both accumulate modes."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.zeros(4, 8, device=self.device_type)
        # Duplicated indices - index 1 appears twice
        idx_tensor = torch.tensor([1, 1, 2], device=self.device_type)
        global_index = [idx_tensor]
        global_value = torch.ones(3, 8, device=self.device_type)

        # Shard on non-indexed dim, index must be replicated
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(1)])
        idx_dt = distribute_tensor(idx_tensor, device_mesh, [Replicate()])
        value_dt = distribute_tensor(global_value, device_mesh, [Shard(1)])

        # accumulate=False: last write wins (non-deterministic with duplicates, but
        # since values are the same, result is deterministic)
        ref_no_accum = torch.index_put(
            global_input, global_index, global_value, accumulate=False
        )
        output_no_accum = torch.index_put(
            input_dt, [idx_dt], value_dt, accumulate=False
        )
        self.assertEqual(output_no_accum.full_tensor(), ref_no_accum)

        # accumulate=True: values are added (index 1 gets 2x the value)
        ref_accum = torch.index_put(
            global_input, global_index, global_value, accumulate=True
        )
        output_accum = torch.index_put(input_dt, [idx_dt], value_dt, accumulate=True)
        self.assertEqual(output_accum.full_tensor(), ref_accum)

    @with_comms
    def test_index_put_broadcast_values(self):
        """Test index_put where values has size-1 broadcast dims."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        # self shape (4, 8), index on dim 0 -> non-indexed dim is 1
        global_input = torch.randn(4, 8, device=self.device_type)
        idx = torch.tensor([0, 1], device=self.device_type)
        global_index = [idx]
        # values shape (2, 1) — broadcast on the non-indexed dim
        global_value = torch.randn(2, 1, device=self.device_type)

        # Shard self on non-indexed dim 1, values should be replicated (size 1)
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(1)])
        idx_dt = distribute_tensor(idx, device_mesh, [Replicate()])
        value_dt = distribute_tensor(global_value, device_mesh, [Replicate()])

        ref = torch.index_put(global_input, global_index, global_value)
        output_dt = torch.index_put(input_dt, [idx_dt], value_dt)
        self.assertEqual(output_dt.full_tensor(), ref)

    @with_comms
    def test_where_type_promotion(self):
        mesh = self.build_device_mesh()  # 1D mesh

        specs = [[Shard(0)], [Replicate()]]
        for spec in specs:
            global_tensor = torch.randn(12, 8)
            mat = distribute_tensor(global_tensor, mesh, spec)
            res = torch.where(mat > 0, 1, 0)
            ref = torch.where(global_tensor > 0, 1, 0)
            self.assertEqual(res.full_tensor(), ref)

    @with_comms
    def test_dtensor_dtype_conversion(self):
        from torch.distributed.tensor.debug import (
            _clear_sharding_prop_cache,
            _get_fast_path_sharding_prop_cache_stats,
        )

        _clear_sharding_prop_cache()
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        # by default we start from bf16 dtype
        local_tensor = torch.randn(2, 8, dtype=torch.bfloat16)
        bf16_sharded_dtensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
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

        # by this point we only have cache misses
        hits, misses = _get_fast_path_sharding_prop_cache_stats()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 2)

        # convert to fp32 again and see if there's cache hit
        bf16_sharded_dtensor1.float()
        hits, misses = _get_fast_path_sharding_prop_cache_stats()
        # by now we should have cache hit
        self.assertEqual(hits, 1)
        self.assertEqual(misses, 2)

    @with_comms
    def test_single_dim_strategy_dtype_cache_key(self):
        """Test that schema_info from single-dim strategy affects cache key.

        When @register_single_dim_strategy specifies static_kwargkey=["dtype"],
        the C++ dispatch path should include dtype in the cache key. This ensures
        calls with different dtypes don't return the same cached result.
        """
        from unittest.mock import patch

        from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
        from torch.distributed.tensor._ops.single_dim_strategy import (
            _ShardingPlaceholder,
            _SingleDimStrategyInfo,
        )
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        call_count = [0]

        def to_copy_single_dim_strategy(op, args_schema, kwargs_schema):
            call_count[0] += 1
            self_meta = args_schema[0]
            if not isinstance(self_meta, TensorMeta):
                raise AssertionError(f"Expected TensorMeta, got {type(self_meta)}")
            single_dim_strategies = []
            for dim in range(len(self_meta.shape)):
                single_dim_strategies.append(
                    [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
                )
            return single_dim_strategies

        _clear_sharding_prop_cache()
        mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(2, 8, dtype=torch.float32)
        sharded_dtensor = DTensor.from_local(local_tensor, mesh, shard_spec)

        propagator = DTensor._op_dispatcher.sharding_propagator
        op = torch.ops.aten._to_copy.default
        schema_info = RuntimeSchemaInfo(static_kwargkey=["dtype"])

        with (
            patch.dict(
                propagator.op_single_dim_strategy_funcs,
                {op: _SingleDimStrategyInfo(func=to_copy_single_dim_strategy)},
            ),
            patch.dict(
                propagator.op_to_schema_info_for_single_dim_strategy, {op: schema_info}
            ),
            patch.dict(propagator.op_strategy_funcs, clear=True),
            patch.dict(propagator.op_to_schema_info, clear=True),
        ):
            call_count[0] = 0
            sharded_dtensor.to(torch.int32)
            sharded_dtensor.to(torch.float64)

            # With dtype in cache key, strategy should be called twice (different dtypes)
            self.assertEqual(call_count[0], 2)

    @with_comms
    def test_slice(self):
        mesh = self.build_device_mesh()  # 1D mesh
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

    @with_comms
    def test_slice_full_size_on_sharded_dim(self):
        """
        Test for the issue #170427 where slicing with a size that equals or
        exceeds the full dimension size should work correctly on sharded
        dimensions.

        So when slicing [:, :N] where N >= dim_size on a tensor sharded on that
        dimension, the operation may be optimized to use aten.alias.default,
        which must have a proper sharding strategy registered.
        """
        mesh = self.build_device_mesh()

        global_tensor = torch.randn(2, 4)
        sharded_dtensor = distribute_tensor(global_tensor, mesh, [Shard(1)])

        result1 = sharded_dtensor[:, :2]  # partial slice
        self.assertEqual(result1.full_tensor(), global_tensor[:, :2])

        result2 = sharded_dtensor[:, 2:]  # partial slice from middle
        self.assertEqual(result2.full_tensor(), global_tensor[:, 2:])

        # This used to fail with: NotImplementedError: Operator aten.alias.default
        # does not have a sharding strategy registered
        result3 = sharded_dtensor[:, :4]  # full dimension slice
        self.assertEqual(result3.full_tensor(), global_tensor[:, :4])

        result4 = sharded_dtensor[:, :8]  # beyond dimension size
        self.assertEqual(result4.full_tensor(), global_tensor[:, :8])

        result5 = sharded_dtensor[:2, :]  # full slice on dim 0
        self.assertEqual(result5.full_tensor(), global_tensor[:2, :])

    @with_comms
    def test_split_on_partial(self):
        self.run_subtests(
            {
                "reduce_op": ["sum", "avg", "product", "min", "max"],
                "split_size": [2, 3, 4],
                "split_dim": [0, 1],
            },
            self._test_split_on_partial,
        )

    def _test_split_on_partial(self, reduce_op: str, split_size: int, split_dim: int):
        self.init_manual_seed_for_rank()
        mesh = self.build_device_mesh()

        partial_tensor = torch.randn(8, 8, device=self.device_type)
        partial_dt = DTensor.from_local(
            local_tensor=partial_tensor,
            device_mesh=mesh,
            placements=[Partial(reduce_op=reduce_op)],
        )
        self._test_op_on_dtensor(
            torch.split,
            partial_dt,
            split_size,
            dim=split_dim,
        )

    @with_comms
    def test_unbind(self):
        device_mesh = self.build_device_mesh()
        shard_dims = [0, 1]
        unbind_dims = [0, 1]
        local_tensor = torch.randn(4, 8, requires_grad=True)
        for shard_dim, unbind_dim in itertools.product(shard_dims, unbind_dims):
            dist_tensor = distribute_tensor(
                local_tensor, device_mesh, (Shard(shard_dim),)
            )

            if shard_dim == unbind_dim:
                with self.assertRaisesRegex(
                    RuntimeError, "Sharding propagation failed"
                ):
                    dist_tensor.unbind(dim=unbind_dim)
            else:
                unbinded_dist_tensors = dist_tensor.unbind(dim=unbind_dim)
                new_shard_dim = shard_dim if shard_dim < unbind_dim else shard_dim - 1
                self.assertTrue(
                    all(
                        elem.placements[0].is_shard(dim=new_shard_dim)
                        for elem in unbinded_dist_tensors
                    )
                )
                for x, y in zip(
                    unbinded_dist_tensors, local_tensor.unbind(dim=unbind_dim)
                ):
                    self.assertEqual(x.full_tensor(), y)


class DistBucketizeTest(LocalDTensorTestBase):
    @with_comms
    def test_bucketize_partial_input(self):
        # Bucketize is non-linear, so Partial("sum")/Partial("avg") inputs
        # must be converted to Replicate. But bucketize is monotone, so
        # Partial("max") and Partial("min") can propagate directly.
        with LocalTensorMode(ranks=self.world_size):
            mesh = self.build_device_mesh()
            boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0], device=self.device_type)
            input_tensor = torch.tensor(
                [[2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 5.0, 9.0]],
                device=self.device_type,
            )

            # Non-linear reductions: must redistribute to Replicate
            for reduce_op in ("sum", "avg"):
                partial_input = DTensor.from_local(
                    input_tensor, mesh, [Partial(reduce_op)]
                )
                dist_boundaries = distribute_tensor(boundaries, mesh, [Replicate()])
                result = torch.bucketize(partial_input, dist_boundaries)

                self.assertTrue(
                    result.placements[0].is_replicate(),
                    f"Expected Replicate output but got {result.placements[0]} "
                    f"for Partial({reduce_op}) input",
                )
                global_input = partial_input.full_tensor()
                expected = torch.bucketize(global_input, boundaries)
                self.assertEqual(result.to_local(), expected)

            # Monotone reductions: output inherits the same partial type
            for reduce_op in ("max", "min"):
                partial_input = DTensor.from_local(
                    input_tensor, mesh, [Partial(reduce_op)]
                )
                dist_boundaries = distribute_tensor(boundaries, mesh, [Replicate()])
                result = torch.bucketize(partial_input, dist_boundaries)

                self.assertTrue(
                    result.placements[0].is_partial(),
                    f"Expected Partial output but got {result.placements[0]} "
                    f"for Partial({reduce_op}) input",
                )
                self.assertEqual(
                    result.placements[0].reduce_op,
                    reduce_op,
                    f"Expected Partial({reduce_op}) output but got {result.placements[0]}",
                )
                expected = torch.bucketize(input_tensor, boundaries)
                self.assertEqual(result.full_tensor(), expected)

    @with_comms
    def test_bucketize_sharded_input(self):
        # Sharded inputs should propagate sharding to output normally.
        with LocalTensorMode(ranks=self.world_size):
            mesh = self.build_device_mesh()
            boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0], device=self.device_type)
            input_tensor = torch.randn(8, 4, device=self.device_type)
            expected = torch.bucketize(input_tensor, boundaries)

            for shard_dim in range(2):
                dist_input = distribute_tensor(input_tensor, mesh, [Shard(shard_dim)])
                dist_boundaries = distribute_tensor(boundaries, mesh, [Replicate()])
                result = torch.bucketize(dist_input, dist_boundaries)

                self.assertTrue(result.placements[0].is_shard(shard_dim))
                self.assertEqual(result.full_tensor(), expected)

    @with_comms
    def test_bucketize_sharded_boundaries(self):
        # When boundaries are sharded on dim 0, each rank counts how many of
        # its local boundary values each input exceeds. The sum across ranks
        # (Partial("sum")) gives the correct global bucket index.
        with LocalTensorMode(ranks=self.world_size):
            mesh = self.build_device_mesh()
            boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0], device=self.device_type)
            input_tensor = torch.tensor(
                [[2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 5.0, 9.0]],
                device=self.device_type,
            )
            expected = torch.bucketize(input_tensor, boundaries)

            dist_input = distribute_tensor(input_tensor, mesh, [Replicate()])
            dist_boundaries = distribute_tensor(boundaries, mesh, [Shard(0)])
            result = torch.bucketize(dist_input, dist_boundaries)
            self.assertEqual(result.full_tensor(), expected)


class DistArgMaxArgMinTest(DTensorTestBase):
    _ops = [torch.argmax, torch.argmin]
    sample = [
        [0, 2, 1, 11, 5, 9, -2, -23],
        [3, 5, 7, 9, 0, -1, 4, 2],
        [8, 4, 6, -5, -10, 12, 7, 1],
        [13, 6, 9, -5, 0, 4, 2, 8],
        [4, 9, 2, 1, -6, -3, 5, 7],
        [0, -4, -2, 8, 6, 3, 12, -7],
        [20, 6, -3, 1, -8, 4, 2, 0],
        [5, 9, 11, -1, -4, 2, 3, 8],
    ]
    placements_tuples = (
        [Partial(), Shard(1)],
        [Partial(), Shard(0)],
        [Shard(0), Shard(1)],
        [Replicate(), Shard(0)],
        [Replicate(), Shard(1)],
    )

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_argmax_argmin_with_placements(self):
        device_mesh = self.build_device_mesh()
        local_tensor = torch.tensor(self.sample, device=self.device_type)
        for placements in self.placements_tuples:
            dtensor_input = distribute_tensor(local_tensor, device_mesh, placements)
            for op in self._ops:
                d_result = op(dtensor_input, dim=1)
                full_dresult = d_result.full_tensor()
                local_result = op(local_tensor, dim=1)
                self.assertEqual(full_dresult, local_result)

    @with_comms
    def test_argmax_argmin_sharded_reduction_dim(self):
        """Unlike max/min which use reduction_linear=True and produce
        Partial("max")/Partial("min") outputs, argmax/argmin return indices
        that can't be combined across shards with an element-wise max/min.
        The strategy sets reduction_linear=False, which forces the input to
        be redistributed to Replicate on the sharded reduction dim before
        the op runs. No Partial placement appears in the output."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        tensor = torch.tensor(self.sample, device=self.device_type, dtype=torch.float)
        dtensor = distribute_tensor(tensor, mesh, [Shard(0)])

        for op in self._ops:
            self.assertEqual(op(dtensor, dim=0).full_tensor(), op(tensor, dim=0))
            self.assertEqual(op(dtensor).full_tensor(), op(tensor))

    def build_device_mesh(self):
        return init_device_mesh(self.device_type, (2, 2))


DistArgMaxArgMinTestWithLocalTensor = create_local_tensor_test_class(
    DistArgMaxArgMinTest,
)

DistTensorOpsTestWithLocalTensor = create_local_tensor_test_class(
    DistTensorOpsTest,
)

if __name__ == "__main__":
    run_tests()
