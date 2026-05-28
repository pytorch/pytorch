#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import os
import unittest

import torch
import torch.comms
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.comms import ReduceOp
from torch.comms.device_mesh import (
    _create_torchcomm_process_group,
    _flatten_with_comm,
    init_device_mesh,
)


try:
    from torch.distributed._mesh_layout import _MeshLayout

    HAS_MESH_LAYOUT = True
except ImportError:
    HAS_MESH_LAYOUT = False

try:
    from torch.distributed._mesh_layout import _FlatLayout  # noqa: F401

    HAS_FLAT_LAYOUT = True
except ImportError:
    HAS_FLAT_LAYOUT = False


class DeviceMeshTest(unittest.TestCase):
    """Test class for DeviceMesh compatibility in torch.comms."""

    def test_init(self) -> None:
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

        comm = torch.comms.new_comm(backend, device, name="comms_test_init")

        try:
            device_mesh = init_device_mesh(
                mesh_dim_comms=(comm,),
                mesh_dim_names=("main",),
            )
        except TypeError as e:
            # TODO: remove this once PT 2.10 is released
            if "_rank" in str(e):
                comm.finalize()
                return
            raise

        group = device_mesh.get_group("main")
        self.assertEqual(group.group_name, "main")

        t = torch.ones(10, device=device, dtype=torch.int32)
        dist.all_reduce(t, group=group)

        # Device-aware synchronization
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
        # No synchronization needed for CPU

        self.assertEqual(t[0].item(), comm.get_size())

        comm.finalize()

    @unittest.skipIf(
        torch.accelerator.device_count() < 4, "Skipping non GPU situations for now"
    )
    def test_2_d_parallel(self) -> None:
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        comm = torch.comms.new_comm(
            backend,
            device,
            name="comms_test_2_d_parallel",
            timeout=datetime.timedelta(seconds=60),
        )
        world_size = comm.get_size()
        dp_degree = 2
        tp_degree = world_size // dp_degree
        mesh = torch.arange(world_size, dtype=torch.int, device="cpu").view(
            dp_degree, tp_degree
        )

        # Get current rank to determine which groups this rank belongs to
        cur_rank = comm.get_rank()

        # For TP communication: find which row contains current rank
        tp_ranks = None
        for row in mesh.tolist():
            if cur_rank in row:
                tp_ranks = row
                break

        # For DP communication: find which column contains current rank
        dp_ranks = None
        mesh_transposed = mesh.transpose(0, 1)
        for col in mesh_transposed.tolist():
            if cur_rank in col:
                dp_ranks = col
                break

        # Create communicators using the new single-list API
        tp_comm = comm.split(tp_ranks, "tp")
        dp_comm = comm.split(dp_ranks, "dp")

        sub_comms = {"dp": dp_comm, "tp": tp_comm}

        try:
            device_mesh_2d = init_device_mesh(
                mesh_dim_comms=(dp_comm, tp_comm),
                mesh_dim_names=("dp", "tp"),
                _global_comm=comm,
            )
        except TypeError as e:
            # TODO: remove this once PT 2.10 is released
            if "_rank" in str(e):
                for sub_comm in sub_comms.values():
                    sub_comm.finalize()
                comm.finalize()
                return
            raise

        cur_rank = comm.get_rank()
        for dim, sub_comm in sub_comms.items():
            sub_mesh = device_mesh_2d[dim]
            self.assertEqual(sub_mesh.get_rank(), cur_rank)
            self.assertEqual(sub_mesh.size(), sub_comm.get_size())
            sub_group = sub_mesh.get_group()
            self.assertEqual(sub_group.group_name, dim)

            t = torch.ones(10, device=device, dtype=torch.int32)
            dist.all_reduce(t, group=sub_group)

            # Device-aware synchronization
            if torch.accelerator.is_available():
                torch.accelerator.synchronize()
            # No synchronization needed for CPU

            self.assertEqual(t[0].item(), sub_comm.get_size())
            sub_comm.finalize()
        comm.finalize()

    def _validate_sub_mesh(self, sub_mesh, sub_comm, dim_mesh_name, cur_rank, device):
        """Validate a sub-mesh and its associated communicator."""
        self.assertEqual(sub_mesh.get_rank(), cur_rank)
        self.assertEqual(sub_mesh.size(), sub_comm.get_size())
        sub_group = sub_mesh.get_group()
        self.assertEqual(sub_group.group_name, dim_mesh_name)

        t = torch.ones(10, device=device, dtype=torch.int32)
        dist.all_reduce(t, group=sub_group)

        # Device-aware synchronization
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
        # No synchronization needed for CPU

        self.assertEqual(t[0].item(), sub_comm.get_size())
        tag = c10d._get_group_tag(sub_group)
        self.assertEqual(tag, f"ptd:{dim_mesh_name}")
        pg_group_ranks = c10d.get_process_group_ranks(sub_group)
        self.assertEqual(len(pg_group_ranks), sub_comm.get_size())

    def _find_ranks_for_mesh_dims(self, mesh, mesh_dim_names, cur_rank):
        """Find the ranks for each mesh dimension that contain the current rank."""
        ranks_per_dim = {}
        for idx, dim_name in enumerate(mesh_dim_names):
            # Calculate global ranks mapping for this mesh dimension
            global_ranks = mesh.transpose(idx, -1).reshape(-1, mesh.size(idx)).tolist()
            for row in global_ranks:
                if cur_rank in row:
                    ranks_per_dim[dim_name] = row
                    break
        return ranks_per_dim

    def _setup_flattened_mesh_dim(
        self, device_mesh_3d, flatten_dim_name, flatten_mesh_dim_names, comm, ranks
    ):
        """Set up a flattened mesh dimension with proper layout."""
        if HAS_FLAT_LAYOUT:
            # New PyTorch API (2.13+): _MeshLayout is a Sequence[_FlatLayout]
            flat_layouts = []
            for dim_name in flatten_mesh_dim_names[flatten_dim_name]:
                sub_layout = device_mesh_3d[dim_name]._layout
                flat_layouts.append(sub_layout[0])
            flatten_layout = _MeshLayout(flat_layouts)
        else:
            # Old PyTorch API: _MeshLayout has .shape and .stride fields
            sizes = []
            strides = []
            for dim_name in flatten_mesh_dim_names[flatten_dim_name]:
                layout = device_mesh_3d[dim_name]._layout
                sizes.append(layout.shape)
                strides.append(layout.stride)
            flatten_layout = _MeshLayout(tuple(sizes), tuple(strides))
        _flatten_with_comm(
            device_mesh_3d,
            flatten_dim_name,
            comm,
            ranks,
            flatten_layout,
        )

    def _find_flatten_ranks_per_dim(
        self, flatten_mesh, flattened_mesh_dim_names, cur_rank
    ):
        """Find ranks for each flattened mesh dimension that contain the current rank."""
        flatten_ranks_per_dim = {}
        for idx, dim_name in enumerate(flattened_mesh_dim_names):
            # Calculate global ranks mapping for this mesh dimension
            global_ranks = flatten_mesh[idx].transpose(idx, -1).tolist()
            for row in global_ranks:
                if cur_rank in row:
                    flatten_ranks_per_dim[dim_name] = row
                    break
        return flatten_ranks_per_dim

    @unittest.skipIf(
        torch.accelerator.device_count() < 8 or not HAS_MESH_LAYOUT,
        "Skipping non GPU situations for now",
    )
    def test_n_d_parallel(self) -> None:
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        comm = torch.comms.new_comm(
            backend,
            device,
            name="comms_test_n_d_parallel",
            timeout=datetime.timedelta(seconds=60),
        )

        world_size = comm.get_size()
        pp_degree = 2
        ep_degree = 2
        cp_degree = world_size // (pp_degree * ep_degree)
        mesh = torch.arange(world_size, dtype=torch.int, device="cpu").view(
            pp_degree, cp_degree, ep_degree
        )

        # Get current rank to determine which groups this rank belongs to
        cur_rank = comm.get_rank()

        mesh_dim_names = ["pp", "cp", "ep"]
        ranks_per_dim = self._find_ranks_for_mesh_dims(mesh, mesh_dim_names, cur_rank)
        comm_per_dim = {}

        # Create communicators using the new single-list API
        for dim_name, ranks in ranks_per_dim.items():
            comm_per_dim[dim_name] = comm.split(ranks, dim_name)

        try:
            device_mesh_3d = init_device_mesh(
                mesh_dim_comms=(
                    comm_per_dim["pp"],
                    comm_per_dim["cp"],
                    comm_per_dim["ep"],
                ),
                mesh_dim_names=("pp", "cp", "ep"),
                _global_comm=comm,
            )
        except TypeError as e:
            # TODO: remove this once PT 2.10 is released
            if "_rank" in str(e):
                for sub_comm in comm_per_dim.values():
                    sub_comm.finalize()
                comm.finalize()
                return
            raise

        flatten_mesh = [
            mesh.view(pp_degree * cp_degree, ep_degree),
            mesh.view(pp_degree, cp_degree * ep_degree),
        ]

        flattened_mesh_dim_names = ["pp_cp", "cp_ep"]
        flatten_mesh_dim_names = {"pp_cp": ["pp", "cp"], "cp_ep": ["cp", "ep"]}
        flatten_ranks_per_dim = self._find_flatten_ranks_per_dim(
            flatten_mesh, flattened_mesh_dim_names, cur_rank
        )

        for flatten_dim_name, ranks in flatten_ranks_per_dim.items():
            comm_per_dim[flatten_dim_name] = comm.split(ranks, flatten_dim_name)
            self._setup_flattened_mesh_dim(
                device_mesh_3d,
                flatten_dim_name,
                flatten_mesh_dim_names,
                comm_per_dim[flatten_dim_name],
                ranks,
            )

        dims_to_test = ["cp", "pp_cp", "cp_ep"]
        cur_rank = comm.get_rank()
        for dim_mesh_name in dims_to_test:
            sub_comm = comm_per_dim[dim_mesh_name]
            sub_mesh = device_mesh_3d[dim_mesh_name]
            self._validate_sub_mesh(sub_mesh, sub_comm, dim_mesh_name, cur_rank, device)

        for sub_comm in comm_per_dim.values():
            sub_comm.finalize()
        comm.finalize()

    @unittest.skipIf(
        torch.accelerator.device_count() < 4,
        "Skipping not enough GPUs situations for now",
    )
    def test_backend_wrapper_split_group(self) -> None:
        """Test splitting a BackendWrapper process group and validating the result."""
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

        # Create a TorchComm communicator
        comm = torch.comms.new_comm(backend, device, name="comms_test_split_group")
        pg = _create_torchcomm_process_group(comm, "comms_test_split_group")

        # Get current rank and world size
        cur_rank = comm.get_rank()
        world_size = comm.get_size()

        split_size = world_size // 2
        pg_ranks_by_dim = torch.arange(world_size).view(2, split_size)[
            cur_rank // split_size
        ]
        # Split using the BackendWrapper's split_group method
        split_pg = pg.split_group(
            pg_ranks_by_dim.tolist(),
            # pyre-ignore Incompatible parameter type [6]
            group_name="split_test_group",
            opts=pg._get_backend(device).options,
        )

        # Validate the split process group
        self.assertIsNotNone(split_pg, "Split process group should not be None")
        self.assertEqual(split_pg._get_backend(device).name(), backend)

        # Verify split comm properties
        self.assertEqual(
            split_pg.size(),
            split_size,
            f"Split comm size should be {split_size}",
        )

        # Verify the rank within the split group is correct
        expected_comm_rank = pg_ranks_by_dim.tolist().index(cur_rank)
        self.assertEqual(
            split_pg.rank(),
            expected_comm_rank,
            f"Rank in split group should be {expected_comm_rank}",
        )

        # Compare with direct TorchComm split to ensure consistency
        direct_split_comm = comm.split(
            pg_ranks_by_dim.tolist(), name="direct_split_test"
        )

        # Test collective operation on the split process group
        t = torch.ones(10, device=device, dtype=torch.int32)
        duplicate_t = t.clone()
        dist.all_reduce(t, group=split_pg)
        direct_split_comm.all_reduce(duplicate_t, ReduceOp.SUM, False)

        # Device-aware synchronization
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()

        # Verify the all_reduce result
        torch.testing.assert_close(t, duplicate_t)

        # Clean up
        # pyre-ignore Undefined attribute [16]
        split_pg._get_backend(device).get_comm().finalize()
        direct_split_comm.finalize()

        comm.finalize()


if __name__ == "__main__":
    unittest.main()
