# Owner(s): ["oncall: distributed"]
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.tensor import Replicate
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(8, 4)
        self.net3 = nn.Linear(4, 12)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x

    def get_input(self):
        return torch.rand(4, 5, device=device_type)


class SimpleModelUneven(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        return x

    def get_input(self):
        return torch.rand(4, 5, device=device_type)


class TestHSDPCheckpoint(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def backend(self):
        curr_backend = dist.get_default_backend_for_device(self.device_type)
        return f"cpu:gloo,{self.device_type}:{curr_backend}"

    def _assert_state_dicts_equal(self, sd1, sd2, check_equal=True):
        """Helper to compare model state dicts."""
        for (k1, v1), (k2, v2) in zip(sd1.items(), sd2.items()):
            self.assertEqual(k1, k2)
            self.assertEqual(v1.device_mesh, v2.device_mesh)
            self.assertEqual(v1.placements, v2.placements)
            if check_equal:
                self.assertEqual(v1.to_local(), v2.to_local())
            else:
                self.assertNotEqual(v1.to_local(), v2.to_local())

    def _assert_optim_state_dicts_equal(self, optim_sd1, optim_sd2, check_equal=True):
        """Helper to compare optimizer state dicts."""
        # Compare state
        for (k1, v1), (k2, v2) in zip(
            optim_sd1["state"].items(), optim_sd2["state"].items()
        ):
            self.assertEqual(k1, k2)
            for (k3, v3), (k4, v4) in zip(v1.items(), v2.items()):
                self.assertEqual(k3, k4)
                if isinstance(v3, dist.tensor.DTensor):
                    self.assertEqual(v3.device_mesh, v4.device_mesh)
                    self.assertEqual(v3.placements, v4.placements)
                    if check_equal:
                        self.assertEqual(v3.to_local(), v4.to_local())
                    else:
                        self.assertNotEqual(v3.to_local(), v4.to_local())
                else:
                    if check_equal:
                        self.assertEqual(v3, v4)
                    else:
                        self.assertNotEqual(v3, v4)
        # Compare param_groups
        for kv1, kv2 in zip(optim_sd1["param_groups"], optim_sd2["param_groups"]):
            for (k1, v1), (k2, v2) in zip(kv1.items(), kv2.items()):
                self.assertEqual(k1, k2)
                self.assertEqual(v1, v2)

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    @parametrize("is_even_sharded_model", [True, False])
    @parametrize("broadcast_count", [0, 1, 2, 3, 4])
    def test_hsdp_checkpoint(self, is_even_sharded_model, broadcast_count) -> None:
        CHECKPOINT_DIR = self.temp_dir
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        mesh_2d = init_device_mesh(self.device_type, (self.world_size // 2, 2))

        replicate_dim_size = mesh_2d.size(0)
        # broadcast_count >= 2 requires replicate_dim_size > broadcast_count
        # and divisible by broadcast_count
        if broadcast_count >= 2 and (
            replicate_dim_size <= broadcast_count
            or replicate_dim_size % broadcast_count != 0
        ):
            self.skipTest(
                f"Skipping broadcast_count={broadcast_count}: replicate dim size "
                f"{replicate_dim_size} must be > {broadcast_count} and divisible by {broadcast_count}"
            )

        model = FSDP(
            simple_model().to(self.device_type),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=mesh_2d,
        )
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        model(model.get_input()).sum().backward()
        optim.step()

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        state_dict = {"model": model.state_dict(), "optim": optim.state_dict()}
        state_dict_to_save = deepcopy(state_dict)

        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # Update the parameters so current model state_dict now be different from state_dict_to_save.
        model(model.get_input()).sum().backward()
        optim.step()

        # At this point, the current state dict is different from state_dict_to_save.
        self._assert_state_dicts_equal(
            state_dict_to_save["model"], model.state_dict(), check_equal=False
        )
        self._assert_optim_state_dicts_equal(
            state_dict_to_save["optim"], optim.state_dict(), check_equal=False
        )

        if broadcast_count == 0:
            loadplanner = DefaultLoadPlanner()
        elif broadcast_count == 1:
            replicate_group = mesh_2d.get_group(mesh_dim=0)
            loadplanner = DefaultLoadPlanner(replicate_group=replicate_group)
        else:  # broadcast_count >= 2
            # Create a 3D mesh to get sub replicate group
            # e.g., broadcast_count=2 with replicate_dim_size=4 -> (2, 2, shard_dim)
            mesh_3d = init_device_mesh(
                self.device_type,
                (broadcast_count, replicate_dim_size // broadcast_count, 2),
            )
            sub_replicate_group = mesh_3d.get_group(mesh_dim=1)
            loadplanner = DefaultLoadPlanner(replicate_group=sub_replicate_group)

        dist_cp.load(
            state_dict=state_dict_to_save,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=loadplanner,
        )
        model.load_state_dict(state_dict_to_save["model"])
        optim.load_state_dict(state_dict_to_save["optim"])

        # After loading, the current model state dict should be the same as state_dict_to_save.
        self._assert_state_dicts_equal(
            state_dict_to_save["model"], model.state_dict(), check_equal=True
        )
        self._assert_optim_state_dicts_equal(
            state_dict_to_save["optim"], optim.state_dict(), check_equal=True
        )

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    @parametrize("is_even_sharded_model", [True, False])
    def test_hsdp_fsdp_checkpoint_conversion(self, is_even_sharded_model) -> None:
        CHECKPOINT_DIR = self.temp_dir
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        # save the hsdp model state_dict
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        hsdp_model = FSDP(
            simple_model().to(self.device_type),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=mesh_2d,
        )
        FSDP.set_state_dict_type(
            hsdp_model,
            StateDictType.SHARDED_STATE_DICT,
        )
        hsdp_state_dict = {"model": hsdp_model.state_dict()}
        dist_cp.save_state_dict(
            state_dict=hsdp_state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # initialize a fsdp model to load checkpoint into
        mesh_1d = init_device_mesh(self.device_type, (self.world_size,))
        fsdp_model = FSDP(
            simple_model().to(self.device_type),
            device_mesh=mesh_1d,
        )
        FSDP.set_state_dict_type(
            fsdp_model,
            StateDictType.SHARDED_STATE_DICT,
        )
        fsdp_state_dict = {"model": fsdp_model.state_dict()}

        # at this point, the hsdp model parameters are different from fsdp model parameters.
        for (k1, v1), (k2, v2) in zip(
            hsdp_state_dict["model"].items(), fsdp_state_dict["model"].items()
        ):
            self.assertEqual(k1, k2)
            self.assertNotEqual(v1.device_mesh, v2.device_mesh)
            self.assertNotEqual(v1.placements, v2.placements)
            v1_all_gather = v1.redistribute(
                mesh_2d, placements=(Replicate(), Replicate())
            )
            v2_all_gather = v2.redistribute(mesh_1d, placements=(Replicate(),))
            self.assertNotEqual(v1_all_gather.to_local(), v2_all_gather.to_local())

        # load the fsdp state_dict from storage
        dist_cp.load_state_dict(
            state_dict=fsdp_state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=DefaultLoadPlanner(),
        )
        fsdp_model.load_state_dict(fsdp_state_dict["model"])

        state_dict_after_load = fsdp_model.state_dict()
        # After loading, the current model state dict should be the same as hsdp_state_dict.
        for (k1, v1), (k2, v2) in zip(
            hsdp_state_dict["model"].items(), state_dict_after_load.items()
        ):
            self.assertEqual(k1, k2)
            self.assertNotEqual(v1.device_mesh, v2.device_mesh)
            self.assertNotEqual(v1.placements, v2.placements)
            v1_all_gather = v1.redistribute(
                mesh_2d, placements=(Replicate(), Replicate())
            )
            v2_all_gather = v2.redistribute(mesh_1d, placements=(Replicate(),))
            self.assertEqual(v1_all_gather.to_local(), v2_all_gather.to_local())


instantiate_parametrized_tests(TestHSDPCheckpoint)
if __name__ == "__main__":
    run_tests()
