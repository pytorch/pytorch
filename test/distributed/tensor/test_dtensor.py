# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import pathlib
import tempfile
import unittest

from numpy.testing import assert_array_equal

import torch
import torch.nn.functional as F
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._api import _shard_tensor
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    ShardOrderEntry,
    TensorMeta,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental import implicit_replication
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing import make_tensor
from torch.testing._internal.common_utils import IS_FBCODE, run_tests, skipIfHpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    map_local_tensor_for_rank,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional


class DummyMLP(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net1 = torch.nn.Linear(5, 1024, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(1024, 4, device=device)

    def forward(self, x):
        return self.net2(F.relu(self.net1(x)))

    def reset_parameters(self, *args, **kwargs):
        with torch.no_grad():
            self.net1.weight.fill_(0.5)
            self.net2.weight.fill_(1)
            self.net1.bias.fill_(1.5)
            self.net2.bias.fill_(1.2)


class DTensorTest(DTensorTestBase):
    @with_comms
    def test_dtensor_constructor(self):
        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]
        local_tensor = torch.randn(3, 3, requires_grad=True)

        spec = DTensorSpec(
            device_mesh,
            tuple(placements),
            tensor_meta=TensorMeta(
                torch.Size([self.world_size * 3, 3]),
                local_tensor.stride(),
                local_tensor.dtype,
            ),
        )

        dist_tensor = DTensor(
            local_tensor,
            spec,
            requires_grad=True,
        )
        self.assertEqual(dist_tensor.size(), torch.Size((self.world_size * 3, 3)))

        with self.assertWarnsRegex(UserWarning, "To construct"):
            DTensor(
                local_tensor,
                spec,
                requires_grad=False,
            )

    @with_comms
    def test_meta_dtensor(self):
        device_mesh = self.build_device_mesh()
        dist_specs = [[Shard(0)], [Replicate()]]
        meta_tensor = torch.randn(1024, 2048, device="meta")
        for dist_spec in dist_specs:
            # Test distribute_tensor on meta tensor
            meta_dtensor = distribute_tensor(meta_tensor, device_mesh, dist_spec)
            self.assertTrue(meta_dtensor.is_meta)
            meta_dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
            torch.nn.init.constant_(meta_dtensor, 1.2)
            value_tensor = torch.empty_like(meta_dtensor.to_local()).fill_(1.2)
            self.assertFalse(meta_dtensor.is_meta)
            self.assertEqual(meta_dtensor.device.type, self.device_type)
            self.assertEqual(meta_dtensor.to_local(), value_tensor)
            # Test from_local on meta tensor
            meta_dtensor = DTensor.from_local(meta_tensor, device_mesh, dist_spec)
            meta_dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
            torch.nn.init.constant_(meta_dtensor, 1.5)
            self.assertEqual(meta_dtensor.device.type, self.device_type)
            value_tensor = torch.empty_like(meta_dtensor.to_local()).fill_(1.5)
            self.assertEqual(meta_dtensor.to_local(), value_tensor)

    @with_comms
    def test_modules_w_meta_dtensor(self):
        model = DummyMLP("meta")
        device_mesh = self.build_device_mesh()
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_tp = parallelize_module(model, device_mesh, parallelize_plan)
        model_tp.to_empty(device=self.device_type)
        model_tp.reset_parameters()
        optim = torch.optim.SGD(model_tp.parameters(), lr=0.1)
        model_regular = DummyMLP(self.device_type)
        model_regular_tp = parallelize_module(
            model_regular, device_mesh, parallelize_plan
        )
        optim_regular = torch.optim.SGD(model_regular_tp.parameters(), lr=0.1)
        model_regular_tp.reset_parameters()
        torch.manual_seed(0)
        inp = torch.randn(20, 5, device=self.device_type)

        output = model_tp(inp)
        output_regular = model_regular_tp(inp)
        self.assertEqual(output, output_regular)

        output.sum().backward()
        output_regular.sum().backward()

        optim.step()
        optim_regular.step()

        torch.manual_seed(1)
        inp = torch.randn(20, 5, device=self.device_type)
        self.assertEqual(model_tp(inp), model_regular_tp(inp))

    @with_comms
    def test_dtensor_stride(self):
        device_mesh = self.build_device_mesh()
        shard0_spec = [Shard(0)]
        local_tensor = torch.randn(4, 8)
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard0_spec)
        # won't affect stride
        self.assertEqual(dist_tensor.stride(), (8, 1))

        shard1_spec = [Shard(1)]
        local_tensor = torch.randn(8, 4)
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard1_spec)
        # will affect stride after DT initialized
        self.assertEqual(dist_tensor.stride(), (4 * self.world_size, 1))

        # if initialized from a transposed mat
        local_tensor = torch.randn(8, 4, 8)
        local_tensor_t = local_tensor.permute(1, 2, 0)
        self.assertEqual(local_tensor_t.stride(), (8, 1, 32))
        dist_tensor = DTensor.from_local(local_tensor_t, device_mesh, shard1_spec)
        global_stride = (8 * self.world_size, 1, 32 * self.world_size)
        self.assertEqual(dist_tensor.stride(), global_stride)

    @with_comms
    def test_from_local(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        self.assertEqual(sharded_tensor.size(), torch.Size([self.world_size * 3, 3]))

        replica_spec = [Replicate()]
        ddp_tensor = DTensor.from_local(local_tensor, device_mesh, replica_spec)
        self.assertEqual(ddp_tensor.size(), local_tensor.size())

        partial_spec = [Partial()]
        partial_tensor = DTensor.from_local(local_tensor, device_mesh, partial_spec)
        self.assertEqual(partial_tensor.size(), local_tensor.size())

        # test dist tensor works with torch.Tensor during backwards
        local_tensor_with_grad = torch.randn(3, 3, requires_grad=True)
        # do some operations on local tensor
        local_tensor_temp = local_tensor_with_grad * 3
        # create the dist tensor with non leaf local tensor, dist tensor created
        # should also be non leaf node
        dist_tensor = DTensor.from_local(local_tensor_temp, device_mesh, shard_spec)
        self.assertFalse(dist_tensor.is_leaf)
        # do some random operations on dist tensor
        output = dist_tensor * 3
        self.assertIsInstance(output, DTensor)
        # trigger .backward() on dist tensor directly
        local_grad = torch.ones(3, 3)
        grad_output = DTensor.from_local(local_grad, device_mesh, shard_spec)
        # run backward directly on dist tensor
        output.backward(grad_output)
        # check it gradients flow back to original torch.Tensor
        self.assertIsNotNone(local_tensor_with_grad.grad)
        expected_grad = torch.ones(3, 3) * 9
        self.assertEqual(local_tensor_with_grad.grad, expected_grad)

        # DTensor.from_local should raise error if the `local_tensor`
        # argument is a DTensor
        local_tensor = torch.ones(2, 2)
        dtensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)

        with self.assertRaisesRegex(
            RuntimeError, "the local_tensor argument only accepts torch.Tensor"
        ):
            DTensor.from_local(dtensor, device_mesh, shard_spec)

    @with_comms
    def test_from_local_backward(self):
        """Test that from_local backward gives the correct gradient placements.

        Verifies the gradient placements meet the following guarantees:
        - Shard(fwd) - Shard(grad_output) - Shard(grad_input)
        - Replicate(fwd) - Replicate(grad_output) - Replicate(grad_input)
        - Replicate(fwd) - Partial(grad_output) - Replicate(grad_input)
        - Partial(fwd) - Partial(grad_output) - Replicate(grad_input)
        - Partial(fwd) - Replicate(grad_output) - Replicate(grad_input)
        """
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        # Test 1: Shard(fwd) - Shard(grad_output) - Shard(grad_input): no redistribution needed in backward
        with comm_mode:
            local_tensor = torch.randn(
                3, 3, device=self.device_type, requires_grad=True
            )
            dt = DTensor.from_local(local_tensor, device_mesh, [Shard(0)])
            output = dt * 2
            grad = DTensor.from_local(
                torch.ones(3, 3, device=self.device_type), device_mesh, [Shard(0)]
            )
            output.backward(grad)
            self.assertEqual(local_tensor.grad, torch.ones(3, 3) * 2)
        # Gradient should flow back without redistribution
        self.assertEqual(comm_mode.get_total_counts(), 0)

        # Test 2: Replicate(fwd) - Replicate(grad_output) - Replicate(grad_input): no redistribution needed in backward
        comm_mode = CommDebugMode()
        with comm_mode:
            local_tensor = torch.randn(
                3, 3, device=self.device_type, requires_grad=True
            )
            dt = DTensor.from_local(local_tensor, device_mesh, [Replicate()])
            output = dt * 2
            grad = DTensor.from_local(
                torch.ones(3, 3, device=self.device_type),
                device_mesh,
                [Replicate()],
                run_check=False,
            )
            output.backward(grad)
            self.assertEqual(local_tensor.grad, torch.ones(3, 3) * 2)
        # Gradient should flow back without redistribution
        self.assertEqual(comm_mode.get_total_counts(), 0)

        # Test 3: Replicate(fwd) - Partial(grad_output) - Replicate(grad_input): gradient redistributed to Replicate
        comm_mode = CommDebugMode()
        with comm_mode:
            local_tensor = torch.randn(
                3, 3, device=self.device_type, requires_grad=True
            )
            dt = DTensor.from_local(local_tensor, device_mesh, [Replicate()])
            output = dt * 2
            # Manually create a Partial gradient to simulate grad coming from a reduce operation
            grad = DTensor.from_local(
                torch.ones(3, 3, device=self.device_type), device_mesh, [Partial()]
            )
            output.backward(grad)
            # The gradient should be redistributed from Partial to Replicate
            # Since grad was Partial with ones, after all-reduce it becomes world_size * ones
            # Then multiplied by 2 from the backward of (dt * 2)
            expected_grad = torch.ones(3, 3) * 2 * self.world_size
            self.assertEqual(local_tensor.grad, expected_grad)
        # Verify that an all-reduce happened for the Partial -> Replicate case
        self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 1)

        # Test 4: Partial(fwd) - Partial(grad_output) - Replicate(grad_input): gradient redistributed to Replicate
        comm_mode = CommDebugMode()
        with comm_mode:
            local_tensor = torch.randn(
                3, 3, device=self.device_type, requires_grad=True
            )
            dt = DTensor.from_local(local_tensor, device_mesh, [Partial()])
            output = dt * 2
            grad = DTensor.from_local(
                torch.ones(3, 3, device=self.device_type), device_mesh, [Partial()]
            )
            output.backward(grad)
            # The gradient should be redistribute from Partial to Replicate
            # Since grad was Partial with ones, after all-reduce it becomes world_size * ones
            # Then multiplied by 2 from the backward of (dt * 2)
            expected_grad = torch.ones(3, 3) * 2 * self.world_size
            self.assertEqual(local_tensor.grad, expected_grad)
        # Verify that an all-reduce happened for the Partial -> Replicate case
        self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 1)

        # Test 5: Partial(fwd) - Replicate(grad_output) - Replicate(grad_input): no redistribution needed in backward
        # When forward was Partial but grad_output is already Replicate, no redistribution is needed
        comm_mode = CommDebugMode()
        with comm_mode:
            local_tensor = torch.randn(
                3, 3, device=self.device_type, requires_grad=True
            )
            dt = DTensor.from_local(local_tensor, device_mesh, [Partial()])
            output = dt * 2
            # Grad output is already Replicate
            grad = DTensor.from_local(
                torch.ones(3, 3, device=self.device_type),
                device_mesh,
                [Replicate()],
                run_check=False,
            )
            output.backward(grad)
            # The gradient is already Replicate which matches the normalized target (Partial->Replicate)
            # So no redistribution should happen
            self.assertEqual(local_tensor.grad, torch.ones(3, 3) * 2)
        # Gradient should flow back without redistribution since grad is already Replicate
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_from_local_uneven_sharding(self):
        device_mesh = self.build_device_mesh()

        uneven_dim0_size = self.world_size + 1
        global_tensor = torch.randn(uneven_dim0_size, 2)
        shard_placement = Shard(0)
        tensor_list, _ = shard_placement._split_tensor(
            global_tensor,
            device_mesh.size(mesh_dim=0),
            with_padding=False,
            contiguous=True,
        )

        dtensor = DTensor.from_local(
            map_local_tensor_for_rank(tensor_list, self.rank, lambda tl, r: tl[r]),
            device_mesh,
            (Shard(0),),
            shape=global_tensor.size(),
            stride=global_tensor.stride(),
        )

        self.assertEqual(dtensor.size(), global_tensor.size())
        self.assertEqual(dtensor.stride(), global_tensor.stride())

    @with_comms
    def test_from_local_uneven_sharding_raise_error(self):
        device_mesh = self.build_device_mesh()

        uneven_dim0_size = self.world_size + 1
        global_tensor = torch.randn(uneven_dim0_size, 2)
        shard_placement = Shard(0)
        tensor_list, _ = shard_placement._split_tensor(
            global_tensor,
            device_mesh.size(mesh_dim=0),
            with_padding=False,
            contiguous=True,
        )

        with self.assertRaisesRegex(
            RuntimeError, "Please pass both shape and stride at the same time."
        ):
            DTensor.from_local(
                map_local_tensor_for_rank(tensor_list, self.rank, lambda tl, r: tl[r]),
                device_mesh,
                (Shard(0),),
                shape=global_tensor.size(),
            )

        with self.assertRaisesRegex(
            RuntimeError, "Please pass both shape and stride at the same time."
        ):
            DTensor.from_local(
                map_local_tensor_for_rank(tensor_list, self.rank, lambda tl, r: tl[r]),
                device_mesh,
                (Shard(0),),
                stride=global_tensor.stride(),
            )

    @with_comms
    def test_from_local_negative_dim(self):
        device_mesh = self.build_device_mesh()
        placements = [Shard(-1)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)
        self.assertEqual(sharded_tensor.placements[0].dim, 1)

    @with_comms
    def test_to_local(self):
        device_mesh = self.build_device_mesh()
        placements = (Shard(0),)
        local_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        dist_tensor_shape = torch.Size([self.world_size * 3, 3])
        spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=TensorMeta(
                dist_tensor_shape,
                local_tensor_with_grad.stride(),
                local_tensor_with_grad.dtype,
            ),
        )
        sharded_tensor = DTensor(
            local_tensor_with_grad,
            spec,
            requires_grad=True,
        )
        self.assertEqual(sharded_tensor.size(), dist_tensor_shape)
        self.assertEqual(sharded_tensor.to_local(), local_tensor_with_grad)

        # test dist tensor works with torch.Tensor during backwards
        # dist tensor created is a leaf node, do some operation on dist tensor
        temp_st = sharded_tensor * 3

        # do some operation on local tensor of the dist tensor
        new_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        res = temp_st.to_local() + new_tensor_with_grad
        # call backward directly on torch.Tensor, and see if it works by
        # propagating through dist tensor
        res.sum().backward()
        self.assertIsNotNone(sharded_tensor.grad)

        self.assertEqual(sharded_tensor.grad.to_local(), torch.ones(3, 3) * 3)

        # test the case when grad stride is different from fwd input.
        res = sharded_tensor.to_local()
        model = torch.nn.ReLU()
        res.register_hook(lambda grad: grad.t())
        target = torch.randn(3, 3, device=self.device_type)
        mae_loss = torch.nn.L1Loss()
        output = mae_loss(model(res), target)
        # The manual change to grad stride leads to the failure of the copy op afterwards.
        # so that we need a try-catch here.
        try:
            output.backward()
        except RuntimeError:
            self.assertEqual(sharded_tensor.grad.stride(), [1, 3 * self.world_size])

        # test the case under no-grad we directly return the local tensor
        with torch.no_grad():
            local_no_grad = sharded_tensor.to_local()
            assert local_no_grad is sharded_tensor._local_tensor

    @with_comms
    def test_to_local_grad_hint(self):
        device_mesh = self.build_device_mesh()
        placements = (Shard(0),)
        global_tensor = torch.ones(8, 3, requires_grad=True)

        sharded_dtensor = distribute_tensor(global_tensor, device_mesh, placements)
        comm_mode = CommDebugMode()

        with comm_mode:
            local_out = sharded_dtensor.redistribute(placements=[Replicate()]).to_local(
                grad_placements=[Partial()]
            )
            local_out.backward(torch.ones_like(local_out))

        self.assertEqual(
            comm_mode.comm_counts[c10d_functional.all_gather_into_tensor], 1
        )
        self.assertEqual(
            comm_mode.comm_counts[c10d_functional.reduce_scatter_tensor], 1
        )

        replica_grad = sharded_dtensor.grad.full_tensor()
        self.assertEqual(replica_grad, global_tensor * self.world_size)

    @with_comms
    def test_full_tensor_sync(self):
        device_mesh = self.build_device_mesh()
        placements = (Shard(0),)
        global_tensor = torch.ones(8, 3, requires_grad=True)

        sharded_dtensor = distribute_tensor(global_tensor, device_mesh, placements)
        full_out = sharded_dtensor.full_tensor()
        self.assertFalse(isinstance(full_out, AsyncCollectiveTensor))
        self.assertEqual(full_out, global_tensor)

    @with_comms
    def test_full_tensor_grad_hint(self):
        device_mesh = self.build_device_mesh()
        placements = (Shard(0),)
        global_tensor = torch.ones(8, 3, requires_grad=True)

        sharded_dtensor = distribute_tensor(global_tensor, device_mesh, placements)
        local_out = sharded_dtensor.full_tensor(grad_placements=[Partial()])
        local_out.sum().backward()

        replica_grad = sharded_dtensor.grad.full_tensor()
        self.assertEqual(replica_grad, global_tensor * self.world_size)

    @with_comms
    def test_dtensor_new_empty_strided(self):
        device_mesh = self.build_device_mesh()
        local_tensor = torch.randn(8, 8, requires_grad=True, device=self.device_type)
        my_dtensor = distribute_tensor(local_tensor, device_mesh, [Shard(0)])
        new_strided_dtensor = my_dtensor.new_empty_strided(
            (8, 8), (8, 1), requires_grad=True
        )
        # test the op produces new dtensor and autograd works
        self.assertEqual(new_strided_dtensor.shape, my_dtensor.shape)
        new_strided_dtensor.sum().backward()
        self.assertIsNotNone(new_strided_dtensor.grad)
        self.assertIsInstance(new_strided_dtensor.grad, DTensor)

        # test backward new_empty_strided with sharding works correctly
        my_dtensor.to_local().sum().backward()
        local_tensor.sum().backward()
        self.assertEqual(my_dtensor.grad, new_strided_dtensor.grad)
        self.assertEqual(
            my_dtensor.grad.redistribute(placements=[Replicate()]).to_local(),
            local_tensor.grad,
        )

    @with_comms
    def test_dtensor_async_output(self):
        # Tests that if the output of some dtensor operations  isn't used in any compute,
        # the output should be an AsyncCollectiveTensor (representing the fact that
        # we haven't synced the collective yet).
        mesh = self.build_device_mesh()

        def fn(dt):
            dt_out_redistribute = dt.redistribute(mesh, [Replicate()], async_op=True)
            # Make sure we haven't synced yet
            # TODO: figure out why this is returning None
            # self.assertTrue(_tensor_needs_wait(dt_out_redistribute))
            dt_out_redistribute_view = dt_out_redistribute.view(
                dt_out_redistribute.shape
            )
            local_tensor = dt_out_redistribute_view.to_local()
            return local_tensor

        x = torch.ones((4, 2), device=self.device_type)
        dt = distribute_tensor(x, mesh, [Shard(0)])
        out = fn(dt)
        # Make sure we haven't synced yet
        self.assertEqual(type(out), AsyncCollectiveTensor)
        self.assertFalse(out.completed)
        out_view = out.view(-1)

        # Assert that output is a `AsyncCollectiveTensor`
        self.assertEqual(type(out_view), AsyncCollectiveTensor)
        self.assertFalse(out.completed)

        # Use the data, requiring a sync
        ref = torch.ones((4, 2), device=self.device_type) + 1
        ref = ref.view(-1)
        out_data = out_view + 1
        self.assertEqual(type(out_data), torch.Tensor)
        self.assertEqual(out_data, ref)

        # test async_op = False default
        sync_out = dt.redistribute(mesh, [Replicate()])
        self.assertFalse(isinstance(sync_out, AsyncCollectiveTensor))
        self.assertEqual(sync_out.to_local(), x)

    @with_comms
    def test_from_local_then_to_local(self):
        # this test ensure end to end from torch.Tensor -> dist tensor -> torch.Tensor works
        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]

        # step 1. construct from construct local tensor
        local_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        # do some operations on local tensor
        local_tensor_temp = local_tensor_with_grad + 8
        # step 2. create the dist tensor with non leaf local tensor, dist tensor
        # created should also be non leaf node
        dist_tensor = DTensor.from_local(local_tensor_temp, device_mesh, placements)
        self.assertFalse(dist_tensor.is_leaf)
        # do some random operations on dist tensor
        output = dist_tensor * 6
        self.assertIsInstance(output, DTensor)

        # step 3. do some operation on local tensor of the dist tensor
        new_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        res = output.to_local() + new_tensor_with_grad
        # call backward directly on torch.Tensor, and see if it works by
        # propagating all the way back to the original torch.Tensor
        res.sum().backward()
        self.assertIsNotNone(local_tensor_with_grad.grad)

        expected_grad = torch.ones(3, 3) * 6
        self.assertEqual(local_tensor_with_grad.grad, expected_grad)

    @with_comms
    def test_dtensor_spec_read_only_after_set(self):
        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)

        # modify placements, and dist_tensor's spec should not be changed
        placements[0] = Replicate()
        self.assertTrue(sharded_tensor.placements is not placements)
        self.assertNotEqual(sharded_tensor.placements, placements)

    @with_comms
    def test_dtensor_spec_hash(self):
        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        local_tensor2 = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)
        sharded_tensor2 = DTensor.from_local(local_tensor2, device_mesh, placements)
        # note that DTensorSpec without real tensor data, so the hash would be the same
        # as long as the mesh, placements and tensor properties are the same
        self.assertEqual(hash(sharded_tensor._spec), hash(sharded_tensor2._spec))

        # change the placements would change the hash
        local_tensor3 = torch.ones(3, 3)
        replica_spec = [Replicate()]
        replica_tensor = DTensor.from_local(
            local_tensor3, device_mesh, replica_spec, run_check=False
        )
        self.assertNotEqual(hash(sharded_tensor._spec), hash(replica_tensor._spec))

    @with_comms
    def test_dtensor_properties(self):
        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)
        self.assertEqual(sharded_tensor.device.type, self.device_type)

    @with_comms
    def test_dtensor_save_load(self):
        import io

        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)
        buffer = io.BytesIO()
        torch.save(sharded_tensor, buffer)
        buffer.seek(0)
        reloaded_st = torch.load(buffer, weights_only=False)
        self.assertEqual(sharded_tensor, reloaded_st)
        buffer.seek(0)
        reloaded_st = torch.load(buffer, weights_only=True)
        self.assertEqual(sharded_tensor, reloaded_st)

    @skipIfHpu
    @with_comms
    @unittest.skipIf(
        IS_FBCODE,
        "subprocess import torch fails with ModuleNotFoundError: No module named 'torch' in fbcode",
    )
    def test_dtensor_save_load_import(self):
        for should_import in [True, False]:
            device_mesh = self.build_device_mesh()
            placements = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = DTensor.from_local(local_tensor, device_mesh, placements)
            with tempfile.NamedTemporaryFile() as f:
                torch.save(sharded_tensor, f)
                import_string = (
                    "import torch.distributed.tensor;" if should_import else ""
                )
                filename = pathlib.Path(f.name)
                err_msg = (
                    (
                        "_pickle.UnpicklingError: Weights only load failed. "
                        "``torch.distributed.tensor`` must be imported to load DTensors"
                    )
                    if not should_import
                    else None
                )
                self._attempt_load_from_subprocess(filename, import_string, err_msg)

    @with_comms
    def test_shard_tensor(self):
        ws = self.world_size
        device_mesh = self.build_device_mesh()
        full_tensor = torch.arange(ws * ws).reshape(ws, ws)

        # Shard by row
        placements = [Shard(0)]
        sharded_tensor = _shard_tensor(full_tensor, placements, device_mesh)
        self.assertEqual(sharded_tensor.size(), torch.Size([ws, ws]))
        self.assertEqual(sharded_tensor.placements, placements)
        local_tensor = sharded_tensor.to_local()
        self.assertEqual(
            local_tensor,
            map_local_tensor_for_rank(
                full_tensor, self.rank, lambda ft, r: ft[range(r, r + 1), :]
            ),
        )

        # Shard by column
        placements = [Shard(1)]
        sharded_tensor = _shard_tensor(full_tensor, placements, device_mesh)
        self.assertEqual(sharded_tensor.size(), torch.Size([ws, ws]))
        self.assertEqual(sharded_tensor.placements, placements)
        local_tensor = sharded_tensor.to_local()
        self.assertEqual(
            local_tensor,
            map_local_tensor_for_rank(
                full_tensor, self.rank, lambda ft, r: ft[:, range(r, r + 1)]
            ),
        )

        # assert full tensor is not changed
        self.assertEqual(full_tensor, torch.arange(ws * ws).reshape(ws, ws))

    @with_comms
    def test_shard_tensor_2d(self):
        ws = self.world_size
        full_tensor = torch.arange(ws).reshape(2, ws // 2)
        device_mesh = DeviceMesh(self.device_type, full_tensor)

        # Shard by row and column
        placements = [Shard(0), Shard(1)]
        sharded_tensor = _shard_tensor(full_tensor, placements, device_mesh)
        self.assertEqual(sharded_tensor.size(), torch.Size([2, ws // 2]))
        self.assertEqual(sharded_tensor.placements, placements)
        local_tensor = sharded_tensor.to_local()
        self.assertEqual(local_tensor.item(), self.rank)


DTensorTestWithLocalTensor = create_local_tensor_test_class(
    DTensorTest,
    skipped_tests=[
        # Async output in local mode is not supported
        "test_dtensor_async_output",
        # Disabling saving and loading in local mode since it requires a deeper
        # integration
        "test_dtensor_save_load",
        "test_dtensor_save_load_import",
    ],
)


class DTensorMeshTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def sub_mesh_assert_equal(self, mesh, exp_in_mesh, exp_out_of_mesh, tensor):
        if self.rank in mesh:
            self.assertEqual(tensor, exp_in_mesh)
        else:
            self.assertEqual(tensor, exp_out_of_mesh)

    @with_comms
    def test_dtensor_device_mesh_device_conversion(self):
        # construct a gpu device mesh
        mesh = self.build_device_mesh()

        # construct from a cpu local tensor with gpu device mesh
        # should automatically convert the dist tensor to gpu
        placements = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, placements)
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

    @with_comms
    def test_dtensor_api_device_mesh_context_manager(self):
        with self.build_device_mesh() as mesh:
            placements = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = DTensor.from_local(
                local_tensor, device_mesh=mesh, placements=placements
            )

        with self.build_device_mesh():
            placements = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = DTensor.from_local(local_tensor, placements=placements)
            replica_spec = [Replicate()]
            replica_tensor = sharded_tensor.redistribute(placements=replica_spec)
            self.assertEqual(
                replica_tensor.size(), torch.Size([3 * self.world_size, 3])
            )

        with self.build_device_mesh():
            placements = [Shard(0)]
            global_shape = torch.Size([3 * self.world_size, 3])
            global_tensor = torch.randn(global_shape)
            sharded_tensor = distribute_tensor(global_tensor, placements=placements)
            self.assertEqual(sharded_tensor.to_local().shape, torch.Size([3, 3]))

            mesh_2d = DeviceMesh(
                self.device_type, torch.arange(self.world_size).reshape(2, 4)
            )

            with mesh_2d:
                shard_2d_spec = [Shard(0), Replicate()]
                tensor_2d = distribute_tensor(global_tensor, placements=shard_2d_spec)

                self.assertEqual(tensor_2d.to_local().shape, torch.Size([3 * 4, 3]))

            sharded_after_2d = distribute_tensor(global_tensor, placements=placements)
            self.assertEqual(sharded_after_2d.to_local().shape, torch.Size([3, 3]))

    @with_comms
    def test_dtensor_2d_mesh(self):
        mesh_tensor = torch.arange(self.world_size).reshape(2, 4)
        # construct a gpu device mesh
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # construct a dist tensor on 2d device mesh and test if works
        placements = [Shard(0), Shard(1)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, placements)
        self.assertEqual(
            dist_tensor.size(), torch.Size([3 * mesh.size(0), 3 * mesh.size(1)])
        )
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

        # if shard on the same tensor dimension
        # we should correctly construct the global tensor size
        shard_same_dim_spec = [Shard(0), Shard(0)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_same_dim_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([3 * self.world_size, 3]))

    @with_comms
    def test_device_mesh_nd(self):
        # construct a gpu device mesh
        mesh_tensor = torch.arange(self.world_size).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        # construct a dist tensor on 3d device mesh and test if works
        placements = [Shard(0), Shard(1), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, placements)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

        # construct a dist tensor on 3d device mesh with some shards on same dim
        placements = [Shard(0), Shard(0), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, placements)
        self.assertEqual(dist_tensor.size(), torch.Size([12, 3, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

    @with_comms
    def test_dtensor_spec_local_shard_offset(self):
        device_mesh = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 4)
        )
        tensor_shape = (3 * self.world_size, 3 * self.world_size)
        # sharding specs and its corresponding local shard offsets
        shard_spec_and_offsets = [
            (
                [Shard(0), Replicate()],
                (3 * (self.world_size // 2) * (self.rank // 4), 0),
            ),
            (
                [Shard(1), Replicate()],
                (0, 3 * (self.world_size // 2) * (self.rank // 4)),
            ),
            (
                [Replicate(), Shard(0)],
                (3 * (self.world_size // 4) * (self.rank % 4), 0),
            ),
            (
                [Replicate(), Shard(1)],
                (0, 3 * (self.world_size // 4) * (self.rank % 4)),
            ),
        ]

        from torch.distributed.tensor._utils import (
            compute_local_shape_and_global_offset,
        )

        # loop through all sharding specs and check local shard offsets
        logical_tensor = torch.randn(tensor_shape)
        for placements, expected_shard_offsets in shard_spec_and_offsets:
            dtensor = distribute_tensor(logical_tensor, device_mesh, placements)
            _, offset = compute_local_shape_and_global_offset(
                dtensor.shape, device_mesh, dtensor.placements
            )
            self.assertEqual(expected_shard_offsets, offset)

    @with_comms
    def test_from_local_sub_mesh(self):
        mesh = DeviceMesh(self.device_type, [0, 2])
        local_tensor = torch.ones(3, 4)

        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        self.assertEqual(dtensor.size(), torch.Size([6, 4]))

        self.sub_mesh_assert_equal(
            mesh.mesh,
            torch.ones(3, 4),
            torch.tensor([]),
            dtensor.to_local(),
        )

        # test dtensor created in submesh, the operation should only
        # be applied to the local shard inside the mesh, not the whole
        # world, so only 0/2 really run the computation
        dtensor = dtensor + 2

        self.sub_mesh_assert_equal(
            mesh.mesh,
            torch.ones(3, 4) + 2,
            torch.tensor([]),
            dtensor.to_local(),
        )

    @with_comms
    def test_default_value_sub_mesh(self):
        mesh = DeviceMesh(self.device_type, [0, 2])

        # test scalar return value
        local_tensor1 = torch.ones(4, 3)
        local_tensor2 = torch.ones(4, 3)
        dtensor1 = DTensor.from_local(local_tensor1, mesh, [Shard(0)])
        dtensor2 = DTensor.from_local(local_tensor2, mesh, [Shard(0)])
        local_res = dtensor1.equal(dtensor2)  # equal returns local result
        self.sub_mesh_assert_equal(
            mesh.mesh,
            True,
            True,
            local_res,
        )

        # test 0-d tensor return value
        local_tensor = torch.ones(4, 3)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)]).sum()
        self.sub_mesh_assert_equal(
            mesh.mesh,
            torch.tensor(12.0),
            torch.tensor(0.0),
            dtensor.to_local(),
        )

        # test List[torch.Tensor] return value
        local_tensor = torch.ones(3, 4)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        dtensor_list = dtensor.split([2, 2], dim=1)
        self.sub_mesh_assert_equal(
            mesh.mesh,
            [torch.ones(3, 2)] * 2,
            [torch.tensor([])] * 2,
            [dt.to_local() for dt in dtensor_list],
        )

    @with_comms
    def test_redistribute_sub_mesh(self):
        mesh = DeviceMesh(self.device_type, [0, 2])

        # test redistribute on a submesh
        local_tensor1 = torch.ones(4, 3)
        sharded_dtensor = DTensor.from_local(local_tensor1, mesh, [Shard(0)])
        replicated_dtensor = sharded_dtensor.redistribute(placements=[Replicate()])
        self.sub_mesh_assert_equal(
            mesh.mesh, torch.ones(8, 3), torch.tensor([]), replicated_dtensor.to_local()
        )
        sharded_again = replicated_dtensor.redistribute(placements=[Shard(0)])
        self.sub_mesh_assert_equal(
            mesh.mesh, torch.ones(4, 3), torch.tensor([]), sharded_again.to_local()
        )

    @with_comms
    def test_implicit_replication(self):
        mesh = self.build_device_mesh()
        local_tensor1 = torch.ones(4, 3)
        sharded_dtensor = DTensor.from_local(local_tensor1, mesh, [Shard(0)])

        with implicit_replication():
            # We put the scalar tensor as the left operand so we can test out
            # when a non-dtensor is a the arg in the args list.
            out_dt = torch.ones(3, device=self.device_type) + sharded_dtensor
            self.assertEqual(out_dt.placements, [Shard(0)])
            self.assertEqual(out_dt.shape, (4 * self.world_size, 3))
            local_shard = out_dt.to_local()
            self.assertEqual(local_shard.shape, (4, 3))
            self.assertEqual(local_shard, torch.ones(4, 3) + torch.ones(3))

    @with_comms
    def test_vmap_embedding(self):
        mesh = self.build_device_mesh()
        batch_size, seq_len = 2, 6
        output_dim = 32

        indices = torch.zeros(*(batch_size, seq_len), dtype=torch.int64)
        indices[0, 1] = 1
        indices[1, 3] = 1
        indices[1, 5] = 1
        indices = DTensor.from_local(indices, mesh, [Shard(0)])

        emb = torch.randn(
            *(batch_size, 8, output_dim),
            dtype=torch.float32,
        )
        emb = DTensor.from_local(emb, mesh, [Shard(0)])
        result = torch.vmap(F.embedding)(indices, emb)
        expected = [F.embedding(indices[i], emb[i]) for i in range(batch_size)]
        expected = torch.stack(expected)
        local_result = result.to_local()
        local_expected = expected.to_local()
        self.assertEqual(local_result, local_expected)

    @unittest.expectedFailure
    @with_comms
    def test_inplace_on_local_tensor_view(self):
        mesh = self.build_device_mesh()
        seq = 8
        vocab = 16
        leaf = torch.randn((seq, vocab), device=self.device_type, requires_grad=True)
        dtensor_leaf = DTensor.from_local(leaf, mesh, [Shard(1)])
        dtensor_vocab_parallel_logits = dtensor_leaf * 2  # make this non-leaf
        vocab_parallel_logits = dtensor_vocab_parallel_logits.to_local()
        logits_max = torch.randn(seq, device=self.device_type)
        vocab_parallel_logits -= logits_max.unsqueeze(dim=1)

    @with_comms
    def test_auto_implicit_replication(self):
        mesh = self.build_device_mesh()

        local_tensor = torch.ones(self.world_size, 3, device=self.device_type)
        sharded_dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])

        # automatically turn tensor to DTensor replicate when ndim = 0 and numel = 1
        ndim_0_tensor = torch.tensor(1, device=self.device_type)

        def add_scalar_tensor_with_dtensor():
            return ndim_0_tensor + sharded_dtensor

        result = add_scalar_tensor_with_dtensor().to_local()
        self.assertEqual(result, local_tensor + ndim_0_tensor)
        self.assertNotWarn(
            add_scalar_tensor_with_dtensor,
            "Found a non-scalar tensor with numel=1 and ndim!=0",
        )

        # automatically turn tensor to DTensor replicate when ndim = 1 and numel = 1
        numel_1_tensor = torch.tensor([1], device=self.device_type)
        self.assertEqual(
            (numel_1_tensor + sharded_dtensor).to_local(), numel_1_tensor + local_tensor
        )

    @unittest.expectedFailure
    @with_comms
    def test_dtensor_cond(self):
        mesh = self.build_device_mesh()

        def make_dtensor(*shape, dtype, device):
            return distribute_tensor(
                make_tensor(*shape, dtype=dtype, device=device),
                device_mesh=mesh,
                placements=None,
            )

        x = make_dtensor(1, 1, dtype=torch.bfloat16, device="cuda")

        # Fails with AssertionError: P1972527564
        torch.cond(
            x > 0,
            lambda: 1.0 / x,
            lambda: torch.zeros_like(x),
        )

    @with_comms
    def test_metadata_consistency_check(self):
        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]

        # Create a local tensor with specific metadata and check dtype change
        local_tensor = torch.randn(3, 3, requires_grad=True, dtype=torch.float32)

        if self.rank == 0:
            local_tensor = local_tensor.to(dtype=torch.float64)

        with self.assertRaises(ValueError):
            DTensor.from_local(local_tensor, device_mesh, placements, run_check=True)

        try:
            DTensor.from_local(local_tensor, device_mesh, placements, run_check=False)
        except ValueError:
            self.fail("Unexpected ValueError raised with run_check=False")

        # Create a local tensor with specific metadata and check requires_grad change
        local_tensor = torch.randn(3, 3, requires_grad=True, dtype=torch.float32)

        if self.rank == 0:
            local_tensor.requires_grad = False

        with self.assertRaises(ValueError):
            DTensor.from_local(local_tensor, device_mesh, placements, run_check=True)

        try:
            DTensor.from_local(local_tensor, device_mesh, placements, run_check=False)
        except ValueError:
            self.fail("Unexpected ValueError raised with run_check=False")

        # Create a local tensor with specific metadata and check stride change
        local_tensor = torch.randn(3, 4, requires_grad=True, dtype=torch.float32)

        if self.rank == 0:
            local_tensor = local_tensor.t()  # transpose changes the stride

        with self.assertRaises(ValueError):
            DTensor.from_local(local_tensor, device_mesh, placements, run_check=True)

        try:
            DTensor.from_local(local_tensor, device_mesh, placements, run_check=False)
        except ValueError:
            self.fail("Unexpected ValueError raised with run_check=False")

    @with_comms
    def test_as_strided_identity(self):
        # Test calling as_strided with the same size/stride/offset as input tensor
        # This should be a no-op but currently fails
        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]
        local_tensor = torch.randn(3, 4, device=self.device_type)
        dtensor = DTensor.from_local(local_tensor, device_mesh, placements)

        # Get the current size, stride, and storage_offset
        size = dtensor.size()
        stride = dtensor.stride()
        storage_offset = dtensor.storage_offset()

        # Call as_strided with the exact same parameters
        result = dtensor.as_strided(size, stride, storage_offset)

        # The result should be identical to the input
        self.assertEqual(result.size(), dtensor.size())
        self.assertEqual(result.stride(), dtensor.stride())
        self.assertEqual(result.to_local(), dtensor.to_local())


DTensorMeshTestWithLocalTensor = create_local_tensor_test_class(
    DTensorMeshTest,
    skipped_tests=[
        # Test asserts must be rewritten for local tensor
        "test_from_local_sub_mesh",
        "test_default_value_sub_mesh",
        "test_redistribute_sub_mesh",
        # Local tensor mode doesn't support tensors of different types on different ranks
        "test_metadata_consistency_check",
    ],
)


class TestDTensorPlacementTypes(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def _create_tensor(self, size):
        # Keep everything deterministic.
        torch.manual_seed(0)
        tensor = torch.rand(size)
        if self.device_type != "cpu":
            return tensor.to(self.device_type)
        else:
            return tensor

    @with_comms
    def test_split_tensor_1D(self) -> None:
        mesh = self.build_device_mesh()
        shard_placement = Shard(0)

        for size in range(8):
            tensor = self._create_tensor(size)
            splitted_tensor_list, pad_sizes = shard_placement._split_tensor(
                tensor,
                mesh.size(),
                with_padding=True,
                contiguous=True,
            )
            if size == 0:
                # when tensor size is 0, there is no padding needed for all the ranks.
                expected_pad_sizes = [0] * self.world_size
                assert_array_equal(expected_pad_sizes, pad_sizes)

                is_tensor_empty = [
                    not splitted_tensor.numel() > 0
                    for splitted_tensor in splitted_tensor_list
                ]
                expected_is_tensor_empty = [True] * self.world_size
                assert_array_equal(expected_is_tensor_empty, is_tensor_empty)
            else:
                expected_pad_sizes = [
                    0 if idx < size else 1
                    for idx, _ in enumerate(range(self.world_size))
                ]
                assert_array_equal(expected_pad_sizes, pad_sizes)

                from torch.distributed.tensor._collective_utils import unpad_tensor

                unpadded_list = [
                    (
                        unpad_tensor(tensor, shard_placement.dim, pad_sizes[i])
                        if pad_sizes[i] > 0
                        else tensor
                    )
                    for i, tensor in enumerate(splitted_tensor_list)
                ]
                expected_is_tensor_empty = [
                    not idx < size for idx, _ in enumerate(range(self.world_size))
                ]
                is_tensor_empty = [
                    not unpadded_tensor.numel() > 0 for unpadded_tensor in unpadded_list
                ]
                assert_array_equal(expected_is_tensor_empty, is_tensor_empty)


TestDTensorPlacementTypesWithLocalTensor = create_local_tensor_test_class(
    TestDTensorPlacementTypes,
)


class TestDTensorSpec(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def test_dtensor_spec_print(self):
        self.assertExpectedInline(
            DTensorSpec.format_shard_order_str((Shard(2), Shard(1), Shard(0)), None),
            """S(2)S(1)S(0)""",
        )
        self.assertExpectedInline(
            DTensorSpec.format_shard_order_str(
                (Shard(2), Shard(1), Shard(0)),
                (
                    ShardOrderEntry(tensor_dim=0, mesh_dims=(2,)),
                    ShardOrderEntry(tensor_dim=1, mesh_dims=(1,)),
                    ShardOrderEntry(tensor_dim=2, mesh_dims=(0,)),
                ),
            ),
            """S(2)S(1)S(0)""",
        )
        self.assertExpectedInline(
            DTensorSpec.format_shard_order_str(
                (Shard(1), Shard(1), Shard(1)),
                (ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 0, 1)),),
            ),
            """S(1)[1]S(1)[2]S(1)[0]""",
        )
        self.assertExpectedInline(
            DTensorSpec.format_shard_order_str(
                (Replicate(), Replicate(), Replicate()), None
            ),
            """RRR""",
        )
        self.assertExpectedInline(
            DTensorSpec.format_shard_order_str(
                (Replicate(), Replicate(), Shard(1)), None
            ),
            """RRS(1)""",
        )

    @with_comms
    def test_dtensor_spec_with_invalid_shard_order(self):
        mesh_shape = (2, 2, self.world_size // 4)
        mesh = init_device_mesh(self.device_type, mesh_shape)
        tensor_local = torch.randn(8, 6, 5, device=self.device_type)
        tensor_global = DTensor.from_local(
            tensor_local, mesh, [Shard(1), Shard(1), Shard(0)]
        )
        tensor_global._spec.shard_order = (
            ShardOrderEntry(tensor_dim=0, mesh_dims=(2,)),
            ShardOrderEntry(tensor_dim=1, mesh_dims=(1, 0)),
        )
        with self.assertRaisesRegex(
            AssertionError, r"shard_order .* has empty mesh dim"
        ):
            tensor_global._spec.shard_order = (
                ShardOrderEntry(tensor_dim=1, mesh_dims=()),
                ShardOrderEntry(tensor_dim=0, mesh_dims=(2,)),
            )
        with self.assertRaisesRegex(
            AssertionError, "tensor dim should be sorted in shard_order"
        ):
            tensor_global._spec.shard_order = (
                ShardOrderEntry(tensor_dim=1, mesh_dims=(1, 0)),
                ShardOrderEntry(tensor_dim=0, mesh_dims=(2,)),
            )
        with self.assertRaisesRegex(
            AssertionError,
            r"placement\[\d+\] doesn't have a matching shard in shard_order",
        ):
            tensor_global._spec.shard_order = (
                ShardOrderEntry(tensor_dim=0, mesh_dims=(1,)),
                ShardOrderEntry(tensor_dim=1, mesh_dims=(1, 0)),
            )
        with self.assertRaisesRegex(
            AssertionError, r"shard_order .* has invalid mesh dim \([\d,]+\)"
        ):
            tensor_global._spec.shard_order = (
                ShardOrderEntry(tensor_dim=0, mesh_dims=(3,)),
                ShardOrderEntry(tensor_dim=1, mesh_dims=(1, 0)),
            )
        with self.assertRaisesRegex(
            AssertionError, r"shard_order .* has invalid tensor dim -?\d+"
        ):
            tensor_global._spec.shard_order = (
                ShardOrderEntry(tensor_dim=0, mesh_dims=(2,)),
                ShardOrderEntry(tensor_dim=-1, mesh_dims=(1, 0)),
            )

    @with_comms
    def test_dtensor_spec_update(self):
        mesh_shape = (2, 2, self.world_size // 4)
        mesh = init_device_mesh(self.device_type, mesh_shape)
        tensor_local = torch.randn(8, 6, 5, device=self.device_type)
        tensor_global_1 = DTensor.from_local(
            tensor_local, mesh, [Shard(1), Shard(1), Shard(0)]
        )
        tensor_global_2 = DTensor.from_local(
            tensor_local, mesh, [Shard(1), Shard(1), Shard(0)]
        )
        self.assertNotEqual(id(tensor_global_1), id(tensor_global_2))
        self.assertEqual(hash(tensor_global_1._spec), hash(tensor_global_2._spec))
        self.assertEqual(tensor_global_1._spec, tensor_global_2._spec)
        # not using the default shard_order
        tensor_global_1._spec.shard_order = (
            ShardOrderEntry(tensor_dim=0, mesh_dims=(2,)),
            ShardOrderEntry(tensor_dim=1, mesh_dims=(1, 0)),
        )
        # hash should be recomputed in DTensorSpec.__setattr__()
        self.assertNotEqual(hash(tensor_global_1._spec), hash(tensor_global_2._spec))
        self.assertNotEqual(tensor_global_1._spec, tensor_global_2._spec)

    @with_comms
    def test_dtensor_spec_default_shard_order_generation(self):
        mesh_shape = (2, 2, self.world_size // 4)
        mesh = init_device_mesh(self.device_type, mesh_shape)
        tensor_local = torch.randn(8, 6, 5, device=self.device_type)

        tensor_global = DTensor.from_local(
            tensor_local, mesh, [Shard(1), Shard(1), Shard(0)]
        )
        self.assertEqual(
            tensor_global._spec.shard_order,
            (
                ShardOrderEntry(tensor_dim=0, mesh_dims=(2,)),
                ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 1)),
            ),
        )

        tensor_global = DTensor.from_local(
            tensor_local, mesh, [Replicate(), Replicate(), Replicate()]
        )
        self.assertEqual(tensor_global._spec.shard_order, ())

        # shard order omit partial
        tensor_global = DTensor.from_local(
            tensor_local, mesh, [Partial(), Replicate(), Replicate()]
        )
        self.assertEqual(tensor_global._spec.shard_order, ())

    @with_comms
    def test_default_shard_order(self):
        mesh_shape = (2, 2, self.world_size // 4)
        mesh = init_device_mesh(self.device_type, mesh_shape)
        tensor_local = torch.randn(8, 6, 5, device=self.device_type)

        tensor_global = DTensor.from_local(
            tensor_local, mesh, [Shard(1), Shard(2), Shard(1)]
        )
        # DTensorSpec automatically builds the default left-to-right order
        self.assertEqual(
            tensor_global._spec.shard_order,
            (
                ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 2)),
                ShardOrderEntry(tensor_dim=2, mesh_dims=(1,)),
            ),
        )
        self.assertTrue(
            DTensorSpec.is_default_device_order(tensor_global._spec.shard_order)
        )
        # manually set the shard_order by exchange mesh dim 0 and 2
        tensor_global._spec.shard_order = (
            ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 0)),
            ShardOrderEntry(tensor_dim=2, mesh_dims=(1,)),
        )
        self.assertFalse(
            DTensorSpec.is_default_device_order(tensor_global._spec.shard_order)
        )


TestDTensorSpecWithLocalTensor = create_local_tensor_test_class(
    TestDTensorSpec,
)

if __name__ == "__main__":
    run_tests()
