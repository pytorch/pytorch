# Owner(s): ["oncall: distributed checkpointing"]

import importlib
import json
import os
import unittest

import torch
import torch.distributed.checkpoint as dist_cp
from torch import distributed as dist
from torch.distributed.checkpoint.quantized_hf_storage import (
    QuantizedHuggingFaceStorageReader,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard, zeros
from torch.testing._internal.common_device_type import (
    Capability,
    instantiate_device_type_tests,
    requires_capabilities,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


@unittest.skipUnless(HAS_SAFETENSORS, "safetensors is not installed")
class TestSingleRankSaveLoad(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @with_temp_dir
    def test_save(self) -> None:
        from safetensors.torch import load_file

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
        for key in state_dict_to_save:
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_loaded[key])
            )

    @with_temp_dir
    def test_load(self) -> None:
        from safetensors.torch import save_file

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
        for key in state_dict_to_save:
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_to_load[key])
            )

    @with_temp_dir
    def test_load_into_empty_dict(self) -> None:
        from safetensors.torch import save_file

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
        for key in state_dict_to_save:
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_loaded[key])
            )

    @with_temp_dir
    def test_load_with_multiple_threads(self) -> None:
        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        state_dict_to_load = MyTestModule().state_dict()

        # Create a mapping to split tensors across multiple files
        # This will force multiple files to be created, enabling multi-threading
        fqn_to_index_mapping = {}
        for i, fqn in enumerate(state_dict_to_save.keys()):
            fqn_to_index_mapping[fqn] = (i % 2) + 1  # Split across 2 files

        # Save using HuggingFaceStorageWriter with multiple files
        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=CHECKPOINT_DIR, fqn_to_index_mapping=fqn_to_index_mapping
            ),
        )

        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=dist_cp.HuggingFaceStorageReader(
                path=CHECKPOINT_DIR, thread_count=2
            ),
        )

        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_to_load.keys())
        )
        for key in state_dict_to_save:
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_to_load[key])
            )

    @with_temp_dir
    def test_quantized_checkpoint_loading(self) -> None:
        """Test end-to-end saving a quantizaed checkpoint and loading it."""
        from safetensors.torch import save_file

        CHECKPOINT_DIR = self.temp_dir

        # Create original (unquantized) tensors to validate against
        original_tensors = {
            "linear1.weight": torch.randn(256, 128, dtype=torch.float32) * 2.0,
            "linear2.weight": torch.randn(128, 64, dtype=torch.float32) * 1.5,
            "embedding.weight": torch.randn(512, 256, dtype=torch.float32) * 3.0,
        }

        # Create quantized tensors and scale tensors
        quantized_checkpoint = {}
        block_size = 128

        for tensor_name, original_tensor in original_tensors.items():
            # Simulate quantization: scale down the tensor for quantization
            # This is a simplified quantization - in real scenarios it would be more complex
            rows, cols = original_tensor.shape

            # Create scale tensor for block-wise dequantization
            block_rows = (rows + block_size - 1) // block_size
            block_cols = (cols + block_size - 1) // block_size

            # Create scale inverse tensor (used for dequantization)
            scale_inv = torch.ones(block_rows, block_cols, dtype=torch.float32) * 2.0

            # Create quantized version (divide by scale for quantization)
            quantized_tensor = original_tensor / 2.0  # Simplified quantization

            # Store quantized tensor and its scale
            quantized_checkpoint[tensor_name] = quantized_tensor
            quantized_checkpoint[f"{tensor_name}_scale_inv"] = scale_inv

        # Save quantized checkpoint to safetensors file
        safetensors_file = os.path.join(CHECKPOINT_DIR, "model.safetensors")
        save_file(quantized_checkpoint, safetensors_file)

        # Create model.safetensors.index.json with weight mapping
        weight_map = {}
        for key in quantized_checkpoint:
            weight_map[key] = "model.safetensors"

        index_data = {
            "metadata": {
                "total_size": sum(
                    t.numel() * t.element_size() for t in quantized_checkpoint.values()
                )
            },
            "weight_map": weight_map,
        }

        index_file = os.path.join(CHECKPOINT_DIR, "model.safetensors.index.json")
        with open(index_file, "w") as f:
            json.dump(index_data, f, indent=2)

        # Prepare state dict to load into
        state_dict_to_load = {}
        for tensor_name, original_tensor in original_tensors.items():
            state_dict_to_load[tensor_name] = torch.zeros_like(original_tensor)

        # Load using QuantizedHuggingFaceStorageReader
        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=QuantizedHuggingFaceStorageReader(
                path=CHECKPOINT_DIR,
                target_dtype=torch.float32,
                block_size=block_size,
                thread_count=2,
            ),
        )

        # Validate that loaded tensors match original tensors
        self.assertEqual(
            sorted(original_tensors.keys()), sorted(state_dict_to_load.keys())
        )

        for tensor_name in original_tensors:
            original = original_tensors[tensor_name]
            loaded = state_dict_to_load[tensor_name]

            # Verify shapes match
            self.assertEqual(
                original.shape,
                loaded.shape,
                lambda msg: f"{msg}\nShape mismatch for {tensor_name}: {original.shape} vs {loaded.shape}",
            )

            # Verify dtypes match
            self.assertEqual(
                original.dtype,
                loaded.dtype,
                lambda msg: f"{msg}\nDtype mismatch for {tensor_name}: {original.dtype} vs {loaded.dtype}",
            )

            # Verify dequantized values match original values
            # We expect exact match since we used simple 2x scaling
            torch.testing.assert_close(
                loaded,
                original,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Value mismatch for tensor {tensor_name}",
            )


def _test_consolidate_to_one_file(test_case, device) -> None:
    import safetensors

    device_type = torch.device(device).type
    global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
    mesh_shape = (test_case.world_size,)
    mesh_1d = init_device_mesh(device_type, mesh_shape)

    rows_per_rank = global_tensor.shape[0] // test_case.world_size
    start_row = test_case.rank * rows_per_rank
    end_row = start_row + rows_per_rank
    local_tensor = global_tensor[start_row:end_row].clone().to(device_type)

    dtensor = DTensor.from_local(
        local_tensor,
        device_mesh=mesh_1d,
        placements=[Shard(0)],
        shape=global_tensor.shape,
        stride=(4, 1),
    )

    state_dict_to_save = {"dtensor": dtensor}
    dist_cp.save(
        state_dict=state_dict_to_save,
        storage_writer=dist_cp.HuggingFaceStorageWriter(
            path=test_case.temp_dir,
            save_distributed=True,
            enable_consolidation=True,
        ),
    )
    dist.barrier()

    if test_case.rank == 0:
        file_path = os.path.join(test_case.temp_dir, "model-00001-of-00001.safetensors")
        loaded_dict = safetensors.torch.load_file(file_path)
        test_case.assertEqual(loaded_dict.keys(), {"dtensor"})
        test_case.assertTrue(torch.equal(loaded_dict["dtensor"], global_tensor))

    dist.barrier()


@unittest.skipUnless(HAS_SAFETENSORS, "safetensors is not installed")
class _CPUHFSafetensorsTestBase(DTensorTestBase):
    hw_classification = HardwareClassification.GENERIC

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def backend(self) -> str:
        return "gloo"


class TestDistributedHFSafetensorsConsolidation(_CPUHFSafetensorsTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    @with_temp_dir
    def test_consolidate_to_one_file(self) -> None:
        _test_consolidate_to_one_file(self, "cpu")


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


def _test_1d_to_1d_reshard_placement_change(test_case, device) -> None:
    device_type = torch.device(device).type
    for original_placement, new_placement in ONE_D_TO_ONE_D_PLACEMENTS:
        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
        mesh_shape = (test_case.world_size,)
        device_mesh = init_device_mesh(device_type, mesh_shape)
        dtensor = distribute_tensor(
            global_tensor, device_mesh, placements=original_placement
        )
        state_dict_to_save = {"dtensor": dtensor}

        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=test_case.temp_dir,
                save_distributed=True,
            ),
        )

        zero_dtensor = zeros([4, 4], device_mesh=device_mesh, placements=new_placement)
        state_dict_to_load = {"dtensor": zero_dtensor}

        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=dist_cp.HuggingFaceStorageReader(test_case.temp_dir),
        )

        state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
            device_mesh,
            placements=[Replicate()],
        )
        test_case.assertEqual(global_tensor, state_dict_to_load["dtensor"].to_local())

        state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
            device_mesh,
            placements=original_placement,
        )
        test_case.assertEqual(
            state_dict_to_save["dtensor"].to_local(),
            state_dict_to_load["dtensor"].to_local(),
        )


@instantiate_parametrized_tests
class TestDTensorReshardPlacementChange(_CPUHFSafetensorsTestBase):
    """
    Test DCP reshard for DTensor with placements changes and without world_size change and mesh_tensor change.
    """

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    @with_temp_dir
    def test_1d_to_1d_reshard_placement_change(self) -> None:
        _test_1d_to_1d_reshard_placement_change(self, "cpu")

    @with_comms
    @with_temp_dir
    def test_2d_to_2d_reshard_placement_change(self) -> None:
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
                    path=CHECKPOINT_DIR, save_distributed=True
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


class TestDTensorReshardMeshChange(_CPUHFSafetensorsTestBase):
    """
    Test DCP reshard for DTensor with placements changes and mesh_tensor change.
    """

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    @with_temp_dir
    def test_1d_to_2d_reshard_mesh_change(self) -> None:
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
                    path=CHECKPOINT_DIR, save_distributed=True
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
    def test_2d_to_1d_reshard_mesh_change(self) -> None:
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
                    path=CHECKPOINT_DIR, save_distributed=True
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
    def test_dtensor_checkpoint_resharding_with_empty_shard(self):
        """
        Test dtensor checkpoint resharding with dtensor containing empty shards.
        """
        tensor = torch.rand(1).to(self.device_type)
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        dtensor = distribute_tensor(tensor, mesh, [Shard(0)])
        ref_state_dict = {"dtensor": dtensor}

        dist_cp.save(
            state_dict=ref_state_dict,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=self.temp_dir, save_distributed=True
            ),
        )

        tensor = torch.rand(1).to(self.device_type)
        mesh_2 = init_device_mesh(self.device_type, (2, self.world_size // 2))
        dtensor = distribute_tensor(tensor, mesh_2, [Shard(0), Shard(0)])
        state_dict = {"dtensor": dtensor}
        dist_cp.load(
            state_dict=state_dict,
            storage_reader=dist_cp.HuggingFaceStorageReader(self.temp_dir),
        )


class TestHFSafetensorsAcceleratorSmoke(DTensorTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @requires_capabilities(
        Capability.lib.safetensors,
        Capability.distributed.backend,
        Capability.distributed.dtensor,
    )
    @with_comms
    @with_temp_dir
    def test_consolidate_to_one_file(self, device) -> None:
        _test_consolidate_to_one_file(self, device)

    @skip_if_lt_x_gpu(2)
    @requires_capabilities(
        Capability.lib.safetensors,
        Capability.distributed.backend,
        Capability.distributed.dtensor,
    )
    @with_comms
    @with_temp_dir
    def test_1d_to_1d_reshard_placement_change(self, device) -> None:
        _test_1d_to_1d_reshard_placement_change(self, device)


instantiate_device_type_tests(
    TestHFSafetensorsAcceleratorSmoke,
    globals(),
    except_for=("cpu",),
    allow_xpu=True,
)

if __name__ == "__main__":
    run_tests()
