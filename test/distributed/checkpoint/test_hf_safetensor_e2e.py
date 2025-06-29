# Owner(s): ["oncall: distributed checkpointing"]

import importlib

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard, zeros
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


CHECKPOINT_DIR = "checkpoint"


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


class TestSingleRankSaveLoad(TestCase):
    @with_temp_dir
    def test_save(self) -> None:
        try:
            from safetensors.torch import load_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.HuggingFaceStorageWriter(path=CHECKPOINT_DIR),
        )

        state_dict_loaded = load_file(
            CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )
        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_loaded.keys())
        )
        for key in state_dict_to_save.keys():
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_loaded[key])
            )

    @with_temp_dir
    def test_load(self) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        state_dict_to_load = MyTestModule().state_dict()
        save_file(
            state_dict_to_save, CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )

        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=dist_cp.HuggingFaceStorageReader(path=CHECKPOINT_DIR),
        )

        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_to_load.keys())
        )
        for key in state_dict_to_save.keys():
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_to_load[key])
            )

    @with_temp_dir
    def test_load_into_empty_dict(self) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        save_file(
            state_dict_to_save, CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )

        state_dict_loaded = _load_state_dict_from_keys(
            storage_reader=dist_cp.HuggingFaceStorageReader(path=CHECKPOINT_DIR),
        )

        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_loaded.keys())
        )
        for key in state_dict_to_save.keys():
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_loaded[key])
            )


ONE_D_PLACEMENTS = [
    [Shard(0)],
    [Replicate()],
]
ONE_D_TO_ONE_D_PLACEMENTS = [
    ([Replicate()], [Shard(0)]),
    ([Shard(0)], [Replicate()]),
]

TWO_D_PLACEMENTS = [
    [Replicate(), Replicate()],
    [Replicate(), Shard(0)],
    [Shard(0), Replicate()],
    [Shard(0), Shard(0)],
]
TWO_D_TO_TWO_D_PLACEMENTS = []
for p1 in TWO_D_PLACEMENTS:
    for p2 in TWO_D_PLACEMENTS:
        if p1 != p2:
            TWO_D_TO_TWO_D_PLACEMENTS.append((p1, p2))


@instantiate_parametrized_tests
class TestDTensorReshardPlacementChange(DTensorTestBase):
    """
    Test DCP reshard for DTensor with placements changes and without world_size change and mesh_tensor change.
    """

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_1d_to_1d_reshard_placement_change(self) -> None:
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        for one_d_to_one_d_placements in ONE_D_TO_ONE_D_PLACEMENTS:
            original_placement, new_placement = one_d_to_one_d_placements

            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (self.world_size,)
            device_mesh = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor, device_mesh, placements=original_placement
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.HuggingFaceStorageWriter(
                    path=CHECKPOINT_DIR,
                    save_sharded=True,
                ),
            )

            zero_dtensor = zeros(
                [4, 4], device_mesh=device_mesh, placements=new_placement
            )
            state_dict_to_load = {"dtensor": zero_dtensor}

            dist_cp.load(
                state_dict=state_dict_to_load,
                storage_reader=dist_cp.HuggingFaceStorageReader(
                    CHECKPOINT_DIR,
                ),
            )

            # materialize the whole tensor to compare with the original global_tensor
            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                device_mesh,
                placements=[Replicate()],
            )
            self.assertEqual(global_tensor, state_dict_to_load["dtensor"].to_local())

            # redistribute the tensor back to its original placement for comparison.
            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                device_mesh,
                placements=original_placement,
            )
            self.assertEqual(
                state_dict_to_save["dtensor"].to_local(),
                state_dict_to_load["dtensor"].to_local(),
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_2d_to_2d_reshard_placement_change(self) -> None:
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir
        for two_d_to_two_d_placements in TWO_D_TO_TWO_D_PLACEMENTS:
            original_placement, new_placement = two_d_to_two_d_placements

            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor,
                mesh_2d,
                placements=original_placement,
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.HuggingFaceStorageWriter(
                    path=CHECKPOINT_DIR, save_sharded=True
                ),
                planner=dist_cp.DefaultSavePlanner(),
            )

            zero_dtensor = zeros([4, 4], device_mesh=mesh_2d, placements=new_placement)
            state_dict_to_load = {"dtensor": zero_dtensor}

            dist_cp.load(
                state_dict=state_dict_to_load,
                storage_reader=dist_cp.HuggingFaceStorageReader(CHECKPOINT_DIR),
            )

            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                mesh_2d,
                placements=[Replicate(), Replicate()],
            )
            self.assertEqual(global_tensor, state_dict_to_load["dtensor"].to_local())

            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                mesh_2d,
                placements=original_placement,
            )
            self.assertEqual(
                state_dict_to_save["dtensor"].to_local(),
                state_dict_to_load["dtensor"].to_local(),
            )


class TestDTensorReshardMeshChange(DTensorTestBase):
    """
    Test DCP reshard for DTensor with placements changes and mesh_tensor change.
    """

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_1d_to_2d_reshard_mesh_change(self) -> None:
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir
        for placements_1d in ONE_D_PLACEMENTS:
            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (self.world_size,)
            mesh_1d = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor, mesh_1d, placements=placements_1d
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.HuggingFaceStorageWriter(
                    path=CHECKPOINT_DIR, save_sharded=True
                ),
            )

            for placements_2d in TWO_D_PLACEMENTS:
                mesh_shape = (2, self.world_size // 2)
                mesh_2d = init_device_mesh(self.device_type, mesh_shape)

                zero_dtensor = zeros(
                    [4, 4], device_mesh=mesh_2d, placements=placements_2d
                )
                state_dict_to_load = {"dtensor": zero_dtensor}

                dist_cp.load(
                    state_dict=state_dict_to_load,
                    storage_reader=dist_cp.HuggingFaceStorageReader(CHECKPOINT_DIR),
                    planner=dist_cp.DefaultLoadPlanner(),
                )

                # materialzie the whole tensor to compare with the original global_tensor
                state_dict_to_load["dtensor"] = state_dict_to_load[
                    "dtensor"
                ].redistribute(
                    mesh_2d,
                    placements=[Replicate(), Replicate()],
                )
                self.assertEqual(
                    global_tensor, state_dict_to_load["dtensor"].to_local()
                )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_2d_to_1d_reshard_mesh_change(self) -> None:
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir
        for placements_2d in TWO_D_PLACEMENTS:
            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor, mesh_2d, placements=placements_2d
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.HuggingFaceStorageWriter(
                    path=CHECKPOINT_DIR, save_sharded=True
                ),
                planner=dist_cp.DefaultSavePlanner(),
            )

            for placements_1d in ONE_D_PLACEMENTS:
                mesh_shape = (self.world_size,)
                mesh_1d = init_device_mesh(self.device_type, mesh_shape)

                zero_dtensor = zeros(
                    [4, 4], device_mesh=mesh_1d, placements=placements_1d
                )
                state_dict_to_load = {"dtensor": zero_dtensor}

                dist_cp.load(
                    state_dict=state_dict_to_load,
                    storage_reader=dist_cp.HuggingFaceStorageReader(CHECKPOINT_DIR),
                    planner=dist_cp.DefaultLoadPlanner(),
                )

                # materialzie the whole tensor to compare with the original global_tensor
                state_dict_to_load["dtensor"] = state_dict_to_load[
                    "dtensor"
                ].redistribute(
                    mesh_1d,
                    placements=[Replicate()],
                )
                self.assertEqual(
                    global_tensor, state_dict_to_load["dtensor"].to_local()
                )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_dtensor_checkpoint_resharding_with_empty_shard(self):
        """
        Test dtensor checkpoint resharding with dtensor containing empty shards.
        """
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        tensor = torch.rand(1).cuda()
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        dtensor = distribute_tensor(tensor, mesh, [Shard(0)])
        ref_state_dict = {"dtensor": dtensor}

        dist_cp.save(
            state_dict=ref_state_dict,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=self.temp_dir, save_sharded=True
            ),
        )

        tensor = torch.rand(1).cuda()
        mesh_2 = init_device_mesh(self.device_type, (2, self.world_size // 2))
        dtensor = distribute_tensor(tensor, mesh_2, [Shard(0), Shard(0)])
        state_dict = {"dtensor": dtensor}
        dist_cp.load(
            state_dict=state_dict,
            storage_reader=dist_cp.HuggingFaceStorageReader(self.temp_dir),
        )


if __name__ == "__main__":
    run_tests()
