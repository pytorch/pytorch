# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools

import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.placement_types import _Partial, Replicate, Shard

from torch.testing._internal.common_utils import run_tests

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

funcol = torch.ops.c10d_functional


class RedistributeTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_shard_to_replicate_forward_backward(self):
        # 1) test shard -> replicate forward
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        replica_spec = [Replicate()]

        input_sizes_and_shard_dim = [
            ((self.world_size * 3, 3), 0),
            ((self.world_size * 3 + 1, 3), 0),
            ((self.world_size * 3 + 2, 3), 0),
            ((3, self.world_size * 3), 1),
            ((3, self.world_size * 3 + 1), 1),
            ((3, self.world_size * 3 + 2), 1),
        ]

        comm_mode = CommDebugMode()
        for input_size, shard_dim in input_sizes_and_shard_dim:
            shard_spec = [Shard(shard_dim)]
            expected_tensor = torch.randn(
                input_size, device=self.device_type, requires_grad=True
            )
            dtensor = distribute_tensor(expected_tensor, device_mesh, shard_spec)
            with comm_mode:
                reshard_dtensor = dtensor.redistribute(device_mesh, replica_spec)
            self.assertEqual(reshard_dtensor.size(), torch.Size(input_size))
            self.assertEqual(expected_tensor, reshard_dtensor.to_local())
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.all_gather_into_tensor], 1
            )

            # 2) test shard -> replicate backward:
            # should give gradient as shard
            grad_output = torch.ones_like(reshard_dtensor)
            with comm_mode:
                reshard_dtensor.backward(grad_output)
            grad_input = dtensor.grad
            self.assertEqual(grad_input.placements, shard_spec)
            self.assertEqual(
                grad_input.to_local(), torch.ones(dtensor.to_local().size())
            )
            self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_replicate_to_replicate_forward_backward(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        replica_spec = [Replicate()]
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)

        comm_mode = CommDebugMode()

        # 1) test replicate -> replicate forward
        replica_tensor = distribute_tensor(local_tensor, device_mesh, replica_spec)
        with comm_mode:
            reshard_replica_tensor = replica_tensor.redistribute(
                device_mesh, replica_spec
            )
        self.assertEqual(replica_tensor.size(), local_tensor.size())
        self.assertEqual(replica_tensor, reshard_replica_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)

        # 2) test replicate -> replicate backward:
        # should give gradient as replicate
        grad_output = torch.ones_like(reshard_replica_tensor)
        with comm_mode:
            reshard_replica_tensor.backward(grad_output)
        grad_input = replica_tensor.grad
        self.assertEqual(grad_input.placements, replica_spec)
        self.assertEqual(grad_input.to_local(), torch.ones(12, 3))
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_replicate_to_shard_forward_backward(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        replica_spec = [Replicate()]

        input_sizes_and_shard_dim = [
            ((self.world_size * 3, 3), 0),
            ((self.world_size * 3 + 1, 3), 0),
            ((self.world_size * 3 + 2, 3), 0),
            ((3, self.world_size * 3), 1),
            ((3, self.world_size * 3 + 1), 1),
            ((3, self.world_size * 3 + 2), 1),
        ]

        comm_mode = CommDebugMode()
        for input_size, shard_dim in input_sizes_and_shard_dim:
            shard_spec = [Shard(shard_dim)]
            # 1) test replicate -> shard forward
            local_replica = torch.randn(
                input_size, device=self.device_type, requires_grad=True
            )
            splitted_list = list(
                torch.chunk(local_replica, self.world_size, dim=shard_dim)
            )

            # make local tensor as the element of the corresponding chunked list
            local_tensor = splitted_list[self.rank]
            replica_tensor = distribute_tensor(local_replica, device_mesh, replica_spec)
            with comm_mode:
                reshard_tensor = replica_tensor.redistribute(device_mesh, shard_spec)
            self.assertEqual(reshard_tensor.size(), replica_tensor.size())
            self.assertEqual(reshard_tensor.placements, shard_spec)
            self.assertEqual(reshard_tensor.to_local(), local_tensor)
            self.assertEqual(comm_mode.get_total_counts(), 0)

            # 2) test replicate -> shard backward:
            # should give gradient as replicate
            grad_output = torch.ones_like(reshard_tensor)
            with comm_mode:
                reshard_tensor.backward(grad_output)
            grad_input = replica_tensor.grad
            self.assertEqual(grad_input.placements, replica_spec)
            self.assertEqual(grad_input.to_local(), torch.ones(input_size))
            self.assertEqual(comm_mode.get_total_counts(), 1)
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.all_gather_into_tensor], 1
            )

    @with_comms
    def test_partial_to_replicate_forward_backward(self):
        # Although we don't allow user to reshard to produce a partial
        # placement (i.e. user can't reshard to partial), we do allow
        # replicate to partial internally, and also partial to replicate
        # backward should work as expected
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        partial_local = torch.ones(12, 3, device=self.device_type, requires_grad=True)
        partial_spec = [_Partial()]
        replica_spec = [Replicate()]

        comm_mode = CommDebugMode()
        # test partial -> replicate, which trigger all_reduce
        partial_tensor = DTensor.from_local(partial_local, device_mesh, partial_spec)
        with comm_mode:
            global_partial_tensor = partial_tensor.redistribute(
                device_mesh, replica_spec
            )

        self.assertEqual(partial_tensor.size(), partial_local.size())
        self.assertEqual(
            partial_local * self.world_size, global_partial_tensor.to_local()
        )
        self.assertEqual(comm_mode.get_comm_counts()[funcol.all_reduce], 1)

        # test backward to have replicate grad on partial
        # for from_local backward, we want the replicate() -> partial() to be
        # pass through.
        with comm_mode:
            global_partial_tensor.backward(torch.ones_like(global_partial_tensor))
        self.assertIsNotNone(partial_local.grad)
        self.assertEqual(partial_local.grad.size(), partial_local.size())
        self.assertEqual(partial_local.grad, torch.ones_like(partial_local))
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_replicate_to_partial(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)
        partial_spec = _Partial()
        replica_spec = Replicate()
        # 1) test replicate -> partial forward
        replica_tensor = distribute_tensor(local_tensor, device_mesh, [replica_spec])
        with self.assertRaisesRegex(RuntimeError, "Can not redistribute to _Partial"):
            partial_tensor = replica_tensor.redistribute(device_mesh, [partial_spec])

        from torch.distributed._tensor.redistribute import Redistribute

        comm_mode = CommDebugMode()

        with comm_mode:
            partial_tensor = Redistribute.apply(
                replica_tensor, device_mesh, [partial_spec]
            )
        self.assertEqual(partial_tensor.size(), local_tensor.size())
        # test it successfully zero out the contents on other ranks
        self.assertEqual(
            replica_tensor.to_local() / self.world_size, partial_tensor.to_local()
        )
        self.assertEqual(comm_mode.get_total_counts(), 0)

        # replicate to partial on sub groups
        local_tensor = torch.randn(12, 3, device=self.device_type)
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(self.world_size // 2, 2),
        )
        # 1) test replicate -> partial on 2d-mesh subgroups
        replica_tensor = distribute_tensor(
            local_tensor, device_mesh, [replica_spec, replica_spec]
        )
        with comm_mode:
            partial_tensor = Redistribute.apply(
                replica_tensor, device_mesh, [partial_spec, partial_spec]
            )
        self.assertEqual(partial_tensor.size(), local_tensor.size())

        self.assertEqual(
            replica_tensor.to_local() / self.world_size,
            partial_tensor.to_local(),
        )
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_partial_to_shard(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        partial_spec = [_Partial()]
        my_rank = device_mesh.get_rank()

        input_sizes_and_shard_dim = [
            ((self.world_size * 3, 3), 0),
            ((self.world_size * 3 + 1, 3), 0),
            ((self.world_size * 3 + 2, 3), 0),
            ((3, self.world_size * 3), 1),
            ((3, self.world_size * 3 + 1), 1),
            ((3, self.world_size * 3 + 2), 1),
        ]

        comm_mode = CommDebugMode()

        for input_size, shard_dim in input_sizes_and_shard_dim:
            shard_spec = [Shard(shard_dim)]

            partial_local = torch.ones(input_size, device=self.device_type)
            partial_tensor = DTensor.from_local(
                partial_local, device_mesh, partial_spec, run_check=False
            )

            full_chunk_size = (
                input_size[shard_dim] + self.world_size - 1
            ) // self.world_size
            chunk_sizes = [
                max(
                    min(input_size[shard_dim], full_chunk_size * (idx + 1))
                    - full_chunk_size * idx,
                    0,
                )
                for idx in range(self.world_size)
            ]

            local_shape = list(input_size)
            local_shape[shard_dim] = chunk_sizes[my_rank]

            # test partial to shard, trigger reduce_scatter
            with comm_mode:
                scatter_shard_tensor = partial_tensor.redistribute(
                    device_mesh, shard_spec
                )
            self.assertEqual(scatter_shard_tensor.size(), partial_tensor.size())
            self.assertEqual(scatter_shard_tensor.placements, shard_spec)
            self.assertEqual(
                scatter_shard_tensor.to_local(),
                torch.ones(local_shape) * self.world_size,
            )
            self.assertEqual(
                comm_mode.get_comm_counts()[funcol.reduce_scatter_tensor], 1
            )

    @with_comms
    def test_redistribute_negative_shard_dim(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)
        shard_spec = [Shard(1)]
        shard_minus_spec = [Shard(-1)]

        shard_tensor = distribute_tensor(local_tensor, device_mesh, shard_spec)
        self.assertEqual(shard_tensor.placements[0].dim, 1)
        reshard_tensor = shard_tensor.redistribute(device_mesh, shard_minus_spec)
        self.assertEqual(shard_tensor.placements[0].dim, 1)

    @with_comms
    def test_redistribute_uneven_sharding(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        data_to_test = [
            # uneven on last mesh dim
            torch.randn((10, 5), device=self.device_type),
            # uneven on both mesh dims
            torch.randn((9, 5), device=self.device_type),
            # smaller than mesh dim shape
            torch.randn((3, 5), device=self.device_type),
            torch.randn((1, 3), device=self.device_type),
        ]

        sharding_to_tests = [
            [Shard(0), Shard(0)],
            [Shard(0), Shard(1)],
        ]

        for input_tensor in data_to_test:
            for placements in sharding_to_tests:
                dt = distribute_tensor(input_tensor, mesh, placements)
                dt_full_tensor = dt.full_tensor()
                self.assertEqual(dt_full_tensor, input_tensor)


class MultiDimRedistributeTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 8

    @with_comms
    def test_multi_dim_mesh(self):
        devices = torch.arange(self.world_size)
        for mesh_shape in [devices, devices.view(4, 2), devices.view(2, 2, 2)]:
            mesh_shape = torch.arange(self.world_size).view(-1, 2)
            device_mesh = DeviceMesh(self.device_type, mesh_shape)
            tensor_shape = (16, 24)

            if torch.distributed.get_rank() == 0:
                full_tensor = torch.randn(*tensor_shape)
            else:
                # these should be entirely ignored
                # because distribute_tensor is expected to override shards in ranks != 0
                full_tensor = torch.ones(*tensor_shape)

            possibilities = [Replicate()] + [Shard(i) for i in range(full_tensor.ndim)]
            all_outputs = list(itertools.product(*(mesh_shape.ndim * [possibilities])))
            all_inputs = list(
                itertools.product(*(mesh_shape.ndim * [possibilities + [_Partial()]]))
            )

            for inputs in all_inputs:
                # if partial, temporarily make it Replicated, then replace replicated with partial afterwards
                repl_inputs = [Replicate() if s.is_partial() else s for s in inputs]
                dt = distribute_tensor(full_tensor, device_mesh, repl_inputs)

                if repl_inputs != inputs:
                    # create a new DTensor reinterpreting some of the replicated entires as "Partial"
                    dt = DTensor.from_local(
                        dt.to_local(), device_mesh, inputs, run_check=False
                    )

                for outputs in all_outputs:
                    # redistribute on target outputs
                    dt2 = dt.redistribute(device_mesh, outputs)

                    # replicate and then get first shard
                    local_full = dt2.full_tensor()

                    if torch.distributed.get_rank() == 0:
                        self.assertEqual(local_full.shape, full_tensor.shape)

                        num_sums = 1
                        for idx, input in enumerate(inputs):
                            if input.is_partial():
                                num_sums *= mesh_shape.size(idx)
                        expected = num_sums * full_tensor
                        self.assertEqual(local_full, expected)


if __name__ == "__main__":
    run_tests()
