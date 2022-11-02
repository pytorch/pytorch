# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch

from torch.distributed.distributed_c10d import (
    ProcessGroup,
    new_group,
    get_global_rank,
    get_world_size,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import Shard


class DeviceMeshTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_device_mesh_2d(self):
        mesh_tensor = torch.arange(4).reshape(2, 2)
        # construct a cuda device mesh
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()

        expected_ranks_by_dim = [[[0, 2], [1, 3]], [[0, 1], [2, 3]]]
        for dim, dim_group in enumerate(dim_to_subgroups):
            self.assertTrue(dim < 2)
            dim_ranks = expected_ranks_by_dim[dim]

            dim_group_size = get_world_size(dim_group)
            self.assertIsInstance(dim_group, ProcessGroup)
            self.assertEqual(dim_group_size, 2)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            current_rank_expected_group_ranks = (
                dim_ranks[0] if self.rank in dim_ranks[0] else dim_ranks[1]
            )
            self.assertEqual(global_ranks, current_rank_expected_group_ranks)

    @with_comms
    def test_device_mesh_2d_from_dim_groups(self):
        # construct a two dimension subgroups
        dim_groups = []
        expected_ranks_by_dim = [[[0, 2], [1, 3]], [[0, 1], [2, 3]]]
        for dim_group_ranks in expected_ranks_by_dim:
            for subgroup_ranks in dim_group_ranks:
                subgroup = new_group(ranks=subgroup_ranks)
                if self.rank in subgroup_ranks:
                    dim_groups.append(subgroup)

        # construct a device mesh from the subgroups
        mesh = DeviceMesh(
            self.device_type, [[0, 1], [2, 3]], dim_groups=dim_groups
        )

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            self.assertTrue(dim < 2)
            dim_ranks = expected_ranks_by_dim[dim]

            dim_group_size = get_world_size(dim_group)
            self.assertIsInstance(dim_group, ProcessGroup)
            self.assertEqual(dim_group_size, 2)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            current_rank_expected_group_ranks = (
                dim_ranks[0] if self.rank in dim_ranks[0] else dim_ranks[1]
            )
            self.assertEqual(global_ranks, current_rank_expected_group_ranks)

    @with_comms
    def test_device_mesh_dim_groups_error(self):
        # construct a two dimension subgroups
        dim_groups = []
        expected_ranks_by_dim = [[[0, 2], [1, 3]], [[0, 1], [2, 3]]]
        for dim_group_ranks in expected_ranks_by_dim:
            for subgroup_ranks in dim_group_ranks:
                subgroup = new_group(ranks=subgroup_ranks)
                if self.rank in subgroup_ranks:
                    dim_groups.append(subgroup)

        if len(dim_groups) > 0:
            # dim_groups is not a list
            self.assertRaises(
                RuntimeError,
                DeviceMesh,
                self.device_type,
                [[0, 1], [2, 3]],
                dim_groups=dim_groups[0],
            )

            # dim_groups is a list, but not a list of ProcessGroup
            self.assertRaises(
                RuntimeError,
                DeviceMesh,
                self.device_type,
                [[0, 1], [2, 3]],
                dim_groups=[dim_groups[0], "dummy"],
            )

            # dim_groups has incorrect length
            self.assertRaises(
                RuntimeError,
                DeviceMesh,
                self.device_type,
                [[0, 1], [2, 3]],
                dim_groups=[dim_groups[0]],
            )

    @with_comms
    def test_device_mesh_nd(self):
        # construct a cuda device mesh
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()

        for dim, dim_group in enumerate(dim_to_subgroups):
            self.assertTrue(dim < mesh_tensor.ndim)
            dim_ranks = mesh_tensor.swapdims(-1, dim).reshape(-1, 2)
            # print(dim_ranks)
            # dim_ranks = expected_ranks_by_dim[dim]

            dim_group_size = get_world_size(dim_group)
            self.assertIsInstance(dim_group, ProcessGroup)
            self.assertEqual(dim_group_size, 2)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            for ranks in dim_ranks:
                if self.rank in ranks:
                    self.assertEqual(global_ranks, ranks.tolist())


class DeviceMeshCollectiveTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_all_reduce_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
        mesh.all_reduce(local_tensor, mesh_dim=0)
        res_num = ((0 + self.world_size - 1) * self.world_size) / 2
        self.assertEqual(local_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_broadcast_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
        mesh.broadcast(local_tensor, mesh_dim=0)
        self.assertEqual(local_tensor, torch.zeros(3, 3))

    @with_comms
    def test_scatter_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        scatter_tensor_shape = [3, 3, 3]
        for scatter_dim in range(len(scatter_tensor_shape)):
            shard_placement = Shard(scatter_dim)
            scatter_tensor_shape[scatter_dim] *= self.world_size
            # make the random seed same across rank
            torch.manual_seed(0)
            global_tensor = torch.randn(
                scatter_tensor_shape, device=self.device_type
            )
            splitted_list, _ = shard_placement._split_tensor(
                global_tensor, mesh.size(), with_padding=True, contiguous=True
            )
            recv_tensor = torch.empty_like(splitted_list[mesh.get_rank()])
            # scatter on dim > 0 would generate non-contiguous tensor, verify that works
            mesh.scatter(recv_tensor, splitted_list, mesh_dim=0)
            self.assertEqual(recv_tensor, splitted_list[mesh.get_rank()])

    @with_comms
    def test_scatter_uneven(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()
        tensor_to_split = torch.randn(
            device_mesh.size() + 3, device_mesh.size() + 1
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)
            tensor_to_scatter = tensor_to_split.clone()
            tensor_splitted_list = tensor_to_split.tensor_split(
                device_mesh.size(), dim=shard_dim
            )
            padded_tensor_list, pad_idx = shard_placement._split_tensor(
                tensor_to_scatter,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )

            scattered_tensor = torch.empty_like(padded_tensor_list[my_rank])
            device_mesh.scatter(
                scattered_tensor, padded_tensor_list, mesh_dim=0
            )
            # unpad scattered_tensor
            if pad_idx != 0 and my_rank >= pad_idx:
                scattered_tensor = shard_placement._unpad_tensor(
                    scattered_tensor
                )

            self.assertEqual(
                scattered_tensor.size(), tensor_splitted_list[my_rank].size()
            )
            self.assertEqual(scattered_tensor, tensor_splitted_list[my_rank])

    @with_comms
    def test_all_gather_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        dims_to_gather = [0, 1]
        for dim in dims_to_gather:
            output_size = [3, 3]
            output_size[dim] *= self.world_size
            # each rank have its own tensor, all_gather gives a list
            local_tensor = torch.ones(3, 3, device=self.device_type)
            gathered_list = []
            for _ in range(self.world_size):
                gathered_list.append(torch.zeros_like(local_tensor))
            mesh.all_gather(gathered_list, local_tensor, mesh_dim=0)
            gathered_tensor = torch.cat(gathered_list, dim=dim)
            self.assertEqual(gathered_tensor, torch.ones(output_size))

    @with_comms
    def test_all_gather_uneven(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()
        tensor_to_split = torch.ones(
            device_mesh.size() + 3,
            device_mesh.size() + 1,
            device=self.device_type,
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)
            tensor_padded_list, pad_idx = shard_placement._split_tensor(
                tensor_to_split,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )
            local_tensor = tensor_padded_list[my_rank]
            gathered_list = []
            for _ in range(device_mesh.size()):
                gathered_list.append(torch.empty_like(local_tensor))

            device_mesh.all_gather(
                gathered_list,
                local_tensor,
                mesh_dim=0,
            )
            if pad_idx != 0:
                gathered_list = [
                    shard_placement._unpad_tensor(gathered_tensor)
                    if i >= pad_idx
                    else gathered_tensor
                    for i, gathered_tensor in enumerate(gathered_list)
                ]
            all_gathered_tensor = torch.cat(gathered_list, dim=shard_dim)
            self.assertEqual(all_gathered_tensor.size(), tensor_to_split.size())
            self.assertEqual(all_gathered_tensor, tensor_to_split)

    @with_comms
    def test_reduce_scatter_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        dims_to_scatter = [0, 1]
        for dim in dims_to_scatter:
            input_size = [3, 3]
            scattered_tensor = torch.empty(input_size, device=self.device_type)
            input_size[dim] *= self.world_size

            input_rs_list = (
                torch.ones(input_size, device=self.device_type) * self.rank
            ).tensor_split(self.world_size, dim=dim)
            res_num = ((0 + self.world_size - 1) * self.world_size) / 2
            mesh.reduce_scatter(scattered_tensor, input_rs_list, mesh_dim=0)
            self.assertEqual(scattered_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_reduce_scatter_uneven(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        my_rank = device_mesh.get_rank()
        tensor_to_split = (
            torch.ones(
                device_mesh.size() + 3,
                device_mesh.size() + 1,
                device=self.device_type,
            )
            * self.rank
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)
            tensor_to_scatter = tensor_to_split.clone()
            tensor_splitted_list = tensor_to_split.tensor_split(
                device_mesh.size(), dim=shard_dim
            )
            padded_tensor_list, pad_idx = shard_placement._split_tensor(
                tensor_to_scatter,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )

            res_num = ((0 + self.world_size - 1) * self.world_size) / 2
            scattered_tensor = torch.empty_like(padded_tensor_list[my_rank])
            device_mesh.reduce_scatter(
                scattered_tensor, padded_tensor_list, mesh_dim=0
            )
            # unpad scattered_tensor
            if pad_idx != 0 and my_rank >= pad_idx:
                scattered_tensor = shard_placement._unpad_tensor(
                    scattered_tensor
                )

            self.assertEqual(
                scattered_tensor.size(), tensor_splitted_list[my_rank].size()
            )
            self.assertEqual(
                scattered_tensor,
                torch.ones_like(tensor_splitted_list[my_rank]) * res_num,
            )

    @with_comms
    def test_all_gather_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            gathered_tensor_list = list(
                torch.empty(
                    (dim_group_size * 3, 3), device=self.device_type
                ).tensor_split(dim_group_size, dim=0)
            )
            mesh.all_gather(gathered_tensor_list, local_tensor, mesh_dim=dim)
            gathered_tensor = torch.cat(gathered_tensor_list)
            exp_tensor = torch.ones(3 * dim_group_size, 3)
            for i in range(len(global_ranks)):
                exp_tensor[i * 3 : (i + 1) * 3] = (
                    torch.ones(3, 3) * global_ranks[i]
                )
            self.assertEqual(gathered_tensor, exp_tensor)

    @with_comms
    def test_reduce_scatter_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            local_rs_list = (
                torch.ones(dim_group_size * 3, 3, device=self.device_type)
                * self.rank
            ).tensor_split(dim_group_size, dim=0)
            scattered_tensor = torch.empty_like(
                local_rs_list[mesh.get_coordinate_on_dim(dim)],
                device=self.device_type,
            )
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            mesh.reduce_scatter(scattered_tensor, local_rs_list, mesh_dim=dim)
            res_num = torch.sum(torch.tensor(global_ranks))
            self.assertEqual(scattered_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_all_reduce_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            cloned_local_tensor = local_tensor.clone()
            mesh.all_reduce(cloned_local_tensor, mesh_dim=dim)
            res_num = sum(global_ranks)
            self.assertEqual(cloned_local_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_broadcast_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            cloned_local_tensor = local_tensor.clone()
            mesh.broadcast(cloned_local_tensor, mesh_dim=dim)
            res_num = global_ranks[0]
            self.assertEqual(cloned_local_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_scatter_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            scattered_tensors = [
                torch.ones(3, 3, device=self.device_type) * global_rank
                for global_rank in global_ranks
            ]
            received_tensor = torch.empty_like(
                scattered_tensors[mesh.get_coordinate_on_dim(dim)]
            )
            mesh.scatter(received_tensor, scattered_tensors, mesh_dim=dim)
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)

    @with_comms
    def test_all_to_all_1d(self):
        # transpose on a 2D tensor distributed over N nodes:
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        tensor_shape = [3, 3]
        input_tensor_list = [
            torch.ones(*tensor_shape, device=self.device_type)
            * (rank + self.rank * self.world_size)
            for rank in range(self.world_size)
        ]
        expected_tensor_list = [
            torch.ones(tensor_shape, device=self.device_type)
            * (self.rank + rank * self.world_size)  # i.e. transpose
            for rank in range(self.world_size)
        ]
        for scatter_dim in range(len(tensor_shape)):
            output_tensor_list = [
                torch.empty_like(input_tensor_list[idx])
                for idx in range(len(input_tensor_list))
            ]
            # scatter on dim > 0 would generate non-contiguous tensor, verify that works
            mesh.all_to_all(output_tensor_list, input_tensor_list, mesh_dim=0)
            output_tensor = torch.cat(output_tensor_list, dim=scatter_dim)
            expected_tensor = torch.cat(expected_tensor_list, dim=scatter_dim)

            self.assertEqual(output_tensor, expected_tensor)

    @with_comms
    def test_all_to_all_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        tensor_shape = [3, 3, 3]
        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            my_coordinate = mesh.get_coordinate_on_dim(dim)
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            input_tensor_list = [
                torch.ones(*tensor_shape, device=self.device_type)
                * (i + self.rank * dim_group_size)
                for i in range(dim_group_size)
            ]
            expected_tensor_list = [
                torch.ones(*tensor_shape, device=self.device_type)
                * (
                    my_coordinate + global_rank * dim_group_size
                )  # i.e. transpose
                for global_rank in global_ranks
            ]
            for scatter_dim in range(len(tensor_shape)):
                # input_tensor = torch.cat(input_tensor_list, dim=scatter_dim)
                output_tensor_list = [
                    torch.empty_like(input_tensor_list[idx])
                    for idx in range(len(input_tensor_list))
                ]
                # scatter on dim > 0 would generate non-contiguous tensor, verify that works
                mesh.all_to_all(
                    output_tensor_list, input_tensor_list, mesh_dim=dim
                )
                output_tensor = torch.cat(output_tensor_list, dim=scatter_dim)
                expected_tensor = torch.cat(
                    expected_tensor_list, dim=scatter_dim
                )
                self.assertEqual(output_tensor, expected_tensor)


if __name__ == "__main__":
    run_tests()
