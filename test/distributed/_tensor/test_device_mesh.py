# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import os
import sys

import torch
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import Shard

from torch.distributed.distributed_c10d import (
    get_global_rank,
    get_process_group_ranks,
    get_world_size,
    is_initialized,
    ProcessGroup,
)
from torch.testing._internal.common_distributed import TEST_SKIPS
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def _get_device_type_and_backend():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "nccl" if device_type == "cuda" else "gloo"
    return device_type, backend


def _set_env_var(addr="localhost", port="25364", world_size=1, rank=0):
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["RANK"] = f"{rank}"


class DeviceMeshTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_eligible_default_pg_for_mesh(self):
        mesh_tensor = torch.arange(self.world_size).reshape(2, -1)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

    def test_init_process_group(self):
        device_type, backend = _get_device_type_and_backend()
        # skip the test if not enough GPUs
        if backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        mesh_tensor = torch.arange(4).reshape(2, 2)
        self.assertTrue(not is_initialized())
        _set_env_var(world_size=self.world_size, rank=self.rank)
        mesh = DeviceMesh(device_type, mesh_tensor)
        self.assertTrue(is_initialized())
        self.destroy_pg()

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
    def test_lazy_init_device_mesh(self):
        mesh = DeviceMesh(self.device_type, [1], _init_process_groups=False)

        with self.assertRaisesRegex(RuntimeError, "process groups not initialized!"):
            mesh.get_dim_groups()

        if self.rank == 1:
            assert mesh.get_coordinate() is not None
        else:
            assert mesh.get_coordinate() is None

    @with_comms
    def test_validate_device_mesh(self):
        mesh = torch.arange(self.world_size).reshape(2, -1)
        mesh_subgp_1 = mesh[0]
        mesh_subgp_2 = mesh[1]
        with self.assertRaisesRegex(RuntimeError, "different mesh"):
            if self.rank in mesh_subgp_1:
                mesh = DeviceMesh(self.device_type, mesh_subgp_1)
            else:
                mesh = DeviceMesh(self.device_type, mesh_subgp_2)


class DeviceMeshTestNDim(DTensorTestBase):
    @property
    def world_size(self):
        return 8

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

    @with_comms
    def test_device_mesh_hash(self):
        mesh_tensor_2d = torch.arange(8).reshape(4, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor_2d)
        mesh2 = DeviceMesh(self.device_type, mesh_tensor_2d)
        self.assertNotEqual(hash(mesh), hash(mesh2))
        mesh_tensor_3d = torch.arange(8).reshape(2, 2, 2)
        mesh3 = DeviceMesh(self.device_type, mesh_tensor_3d)
        self.assertNotEqual(hash(mesh), hash(mesh3))
        self.assertNotEqual(hash(mesh2), hash(mesh3))


class DeviceMeshCollectiveTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_all_reduce_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
        # multiply with 1 to trigger wait
        local_tensor = mesh.all_reduce(local_tensor, mesh_dim=0) * 1
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
            global_tensor = torch.randn(scatter_tensor_shape, device=self.device_type)
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
            device_mesh.size() + 3, device_mesh.size() + 1, device=self.device_type
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
            device_mesh.scatter(scattered_tensor, padded_tensor_list, mesh_dim=0)
            # unpad scattered_tensor
            if pad_idx != 0 and my_rank >= pad_idx:
                scattered_tensor = shard_placement._unpad_tensor(scattered_tensor)

            self.assertEqual(
                scattered_tensor.size(), tensor_splitted_list[my_rank].size()
            )
            self.assertEqual(scattered_tensor, tensor_splitted_list[my_rank])

    @with_comms
    def test_all_gather_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        dims_to_gather = [0, 1, 2]
        for dim in dims_to_gather:
            output_size = [3, 3, 3]
            output_size[dim] *= self.world_size
            # each rank have its own tensor, all_gather gives a list
            local_tensor = torch.ones([3, 3, 3], device=self.device_type)
            gathered_tensor = mesh.all_gather(local_tensor, mesh_dim=0, gather_dim=dim)
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
            big_tensor = device_mesh.all_gather(
                local_tensor, mesh_dim=0, gather_dim=shard_dim
            )
            if pad_idx != 0:
                big_tensor = shard_placement._unpad_concat_tensor(
                    big_tensor, pad_idx, device_mesh.size()
                )

            self.assertEqual(big_tensor.size(), tensor_to_split.size())
            self.assertEqual(big_tensor, tensor_to_split)

    @with_comms
    def test_reduce_scatter_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        dims_to_scatter = [0, 1]
        for dim in dims_to_scatter:
            input_size = [3, 3]
            input_size[dim] *= self.world_size
            input_tensor = torch.ones(input_size, device=self.device_type) * self.rank
            res_num = ((0 + self.world_size - 1) * self.world_size) / 2
            scattered_tensor = mesh.reduce_scatter(
                input_tensor, mesh_dim=0, scatter_dim=dim
            )
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
            tensor_splitted_list = tensor_to_split.tensor_split(
                device_mesh.size(), dim=shard_dim
            )

            shard_placement = Shard(shard_dim)
            padded_tensor_list, pad_idx = shard_placement._split_tensor(
                tensor_to_split, device_mesh.size(), with_padding=True, contiguous=True
            )
            tensor_to_reduce = torch.cat(padded_tensor_list, shard_dim)

            res_num = ((0 + self.world_size - 1) * self.world_size) / 2

            scattered_tensor = device_mesh.reduce_scatter(
                tensor_to_reduce, mesh_dim=0, scatter_dim=shard_dim
            )

            if pad_idx <= device_mesh.get_coordinate()[0]:
                scattered_tensor = shard_placement._unpad_tensor(scattered_tensor)

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
            global_ranks = get_process_group_ranks(dim_group)

            gathered_tensor = mesh.all_gather(local_tensor, mesh_dim=dim) * 1
            exp_tensor = torch.ones(3 * dim_group_size, 3)
            for i in range(len(global_ranks)):
                exp_tensor[i * 3 : (i + 1) * 3] = torch.ones(3, 3) * global_ranks[i]
            self.assertEqual(gathered_tensor, exp_tensor)

    @with_comms
    def test_reduce_scatter_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            input_size = [3, 3, 3]
            dim_group_size = get_world_size(dim_group)
            input_size[dim] *= dim_group_size

            input_tensor = torch.ones(input_size, device=self.device_type) * self.rank
            global_ranks = get_process_group_ranks(dim_group)

            scattered_tensor = mesh.reduce_scatter(
                input_tensor, mesh_dim=dim, scatter_dim=dim
            )

            res_num = torch.sum(torch.tensor(global_ranks))
            self.assertEqual(scattered_tensor, torch.ones(3, 3, 3) * res_num)

    @with_comms
    def test_all_reduce_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            global_ranks = get_process_group_ranks(dim_group)
            cloned_local_tensor = local_tensor.clone()
            cloned_local_tensor = mesh.all_reduce(cloned_local_tensor, mesh_dim=dim) * 1
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
                scattered_tensors[mesh.get_coordinate()[dim]]
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
            my_coordinate = mesh.get_coordinate()[dim]
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
                * (my_coordinate + global_rank * dim_group_size)  # i.e. transpose
                for global_rank in global_ranks
            ]
            for scatter_dim in range(len(tensor_shape)):
                # input_tensor = torch.cat(input_tensor_list, dim=scatter_dim)
                output_tensor_list = [
                    torch.empty_like(input_tensor_list[idx])
                    for idx in range(len(input_tensor_list))
                ]
                # scatter on dim > 0 would generate non-contiguous tensor, verify that works
                mesh.all_to_all(output_tensor_list, input_tensor_list, mesh_dim=dim)
                output_tensor = torch.cat(output_tensor_list, dim=scatter_dim)
                expected_tensor = torch.cat(expected_tensor_list, dim=scatter_dim)
                self.assertEqual(output_tensor, expected_tensor)


if __name__ == "__main__":
    run_tests()
