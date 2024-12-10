# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools

import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor._collective_utils import shard_dim_alltoall
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TEST_HPU
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
    def test_replicate_to_local_partial_grad(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        replica_spec = [Replicate()]
        local_tensor = torch.randn(12, 3, device=self.device_type, requires_grad=True)

        replica_tensor = distribute_tensor(local_tensor, device_mesh, replica_spec)

        comm_mode = CommDebugMode()

        with comm_mode:
            out = replica_tensor.redistribute(placements=[Replicate()]).to_local(
                grad_placements=[Partial()]
            )
            out.backward(torch.ones_like(out))

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertEqual(comm_mode.get_comm_counts()[funcol.all_reduce], 1)

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
        partial_spec = [Partial()]
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
        partial_spec = Partial()
        replica_spec = Replicate()
        # 1) test replicate -> partial forward
        replica_tensor = distribute_tensor(local_tensor, device_mesh, [replica_spec])
        with self.assertRaisesRegex(RuntimeError, "Can not redistribute to Partial"):
            partial_tensor = replica_tensor.redistribute(device_mesh, [partial_spec])

        from torch.distributed.tensor._redistribute import Redistribute

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
        partial_spec = [Partial()]
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

    @with_comms
    def test_redistribute_shard_dim_change(self):
        # test 1d device mesh
        mesh_1d = DeviceMesh(self.device_type, torch.arange(self.world_size))
        data_to_test = [
            # evenly sharded case
            torch.randn((8, 8), device=self.device_type),
            # 3d or more dims
            torch.randn((8, 8, 8), device=self.device_type),
            # uneven case 1
            torch.randn((8, 5), device=self.device_type),
            # uneven case 2
            torch.randn((5, 8), device=self.device_type),
            # uneven case 3
            torch.randn((5, 5), device=self.device_type),
        ]

        sharding_src_dst_pairs = [([Shard(0)], [Shard(1)]), ([Shard(1)], [Shard(0)])]

        comm_mode = CommDebugMode()

        for input_data in data_to_test:
            for src, dst in sharding_src_dst_pairs:
                expected_dt = distribute_tensor(input_data.clone(), mesh_1d, dst)
                sharded_dt = distribute_tensor(input_data, mesh_1d, src)
                with comm_mode:
                    out_dt = sharded_dt.redistribute(mesh_1d, dst)
                self.assertEqual(out_dt.placements, expected_dt.placements)
                local_out_dt = out_dt.to_local()
                local_expected_dt = expected_dt.to_local()
                self.assertEqual(out_dt.to_local(), expected_dt.to_local())
                if TEST_HPU or TEST_CUDA:
                    self.assertEqual(
                        comm_mode.get_comm_counts()[
                            torch.ops._dtensor.shard_dim_alltoall
                        ],
                        1,
                    )
                else:
                    self.assertEqual(
                        comm_mode.get_comm_counts()[funcol.all_gather_into_tensor],
                        1,
                    )

        # test 2d device mesh
        mesh_2d = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        data_to_test_2d = [
            # evenly sharded case
            torch.randn((8, 8), device=self.device_type),
            # 3d or more dims
            torch.randn((8, 8, 8), device=self.device_type),
            # uneven case 1
            torch.randn((8, 5), device=self.device_type),
            # uneven case 2
            torch.randn((5, 8), device=self.device_type),
            # uneven case 3
            torch.randn((5, 5), device=self.device_type),
        ]
        sharding_src_dst_pairs_2d = [
            ([Shard(0), Shard(1)], [Shard(0), Shard(0)]),
            ([Shard(0), Shard(1)], [Shard(1), Shard(0)]),
            ([Shard(0), Shard(0)], [Shard(1), Shard(1)]),
        ]
        comm_counts_2d = [
            1,  # 1: S1 -> S0
            2,  # 1: S1 -> R, 0: S0 -> S1, 1: R -> S0
            2,  # 1: S0 -> R, 0: S0 -> S1, 1: R -> S1
        ]

        for input_data in data_to_test_2d:
            if input_data.ndim > 2:
                sharding_spec_combs = sharding_src_dst_pairs_2d + [
                    ([Shard(0), Shard(2)], [Shard(1), Shard(0)]),
                    ([Shard(1), Shard(1)], [Shard(1), Shard(2)]),
                ]
                comm_counts_2d = comm_counts_2d + [
                    2,  # 1. S2 -> R, 0: S0 -> S1, 1: R -> S0
                    1,  # 1: S1 -> S2
                ]
            else:
                sharding_spec_combs = sharding_src_dst_pairs_2d

            for idx, (src, dst) in enumerate(sharding_spec_combs):
                expected_dt = distribute_tensor(input_data.clone(), mesh_2d, dst)
                sharded_dt = distribute_tensor(input_data, mesh_2d, src)
                with comm_mode:
                    out_dt = sharded_dt.redistribute(mesh_2d, dst)

                self.assertEqual(out_dt.placements, expected_dt.placements)
                self.assertEqual(comm_mode.get_total_counts(), comm_counts_2d[idx])

                local_out_dt = out_dt.to_local()
                local_expected_dt = expected_dt.to_local()
                self.assertEqual(local_out_dt, local_expected_dt)

    @with_comms
    def test_shard_dim_alltoall(self):
        # init 2d mesh here so we can test when group_rank != global_rank
        mesh = init_device_mesh(self.device_type, (2, 2))
        tensor = torch.randn(12, self.world_size, device=self.device_type)
        new_tensor = shard_dim_alltoall(tensor, 0, 1, mesh, 0)

        meta_tensor = torch.randn(12, self.world_size, device="meta")
        new_meta_tensor = shard_dim_alltoall(meta_tensor, 0, 1, mesh, 0)

        self.assertEqual(new_tensor.shape, new_meta_tensor.shape)
        self.assertEqual(new_tensor.stride(), new_meta_tensor.stride())


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
                itertools.product(*(mesh_shape.ndim * [possibilities + [Partial()]]))
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

    @with_comms
    def test_redistribute_shard_dim_multi_dim_mesh(self):
        mesh = init_device_mesh(self.device_type, (2, 2, 2))
        input_data = torch.randn((8, 8, 8), device=self.device_type)

        sharding_src_dst_pairs_3d = [
            ([Shard(0), Shard(0), Shard(0)], [Shard(1), Shard(1), Shard(1)]),
            ([Shard(0), Shard(1), Shard(0)], [Shard(1), Shard(0), Shard(0)]),
            ([Shard(0), Shard(1), Shard(2)], [Shard(2), Shard(1), Shard(0)]),
            ([Shard(1), Shard(0), Shard(0)], [Replicate(), Shard(0), Shard(0)]),
            ([Shard(1), Replicate(), Shard(0)], [Replicate(), Shard(0), Shard(0)]),
            ([Shard(0), Shard(0), Shard(1)], [Shard(0), Shard(1), Shard(2)]),
        ]
        comm_counts_3d = [
            3,  # 2: S0 - R, 1: S1 -> R, 0: S0 -> S1
            3,  # 2: S0 -> R, 1: S1 -> R, 0: S0 -> S1, 1: R -> S0, 2: R -> S0
            2,  # 2: S2 -> R, 0: S1 -> S2
            1,  # 0: S1 -> R
            2,  # 2: S0 -> R, 1: R -> S0, 2: R -> S0, 0: S1 -> R
            2,  # 2: S1 -> S2, 1: S0 -> S1
        ]

        comm_mode = CommDebugMode()
        for idx, (src_placement, dst_placement) in enumerate(sharding_src_dst_pairs_3d):
            expected_dt = distribute_tensor(input_data.clone(), mesh, dst_placement)
            sharded_dt = distribute_tensor(input_data, mesh, src_placement)

            with comm_mode:
                out_dt = sharded_dt.redistribute(mesh, dst_placement)

            self.assertEqual(out_dt.placements, expected_dt.placements)
            self.assertEqual(comm_mode.get_total_counts(), comm_counts_3d[idx])

            local_out_dt = out_dt.to_local()
            local_expected_dt = expected_dt.to_local()
            self.assertEqual(local_out_dt, local_expected_dt)


if __name__ == "__main__":
    run_tests()
