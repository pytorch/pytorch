# Owner(s): ["oncall: distributed checkpointing"]

import importlib
import json
import os

import torch
import torch.distributed.checkpoint as dist_cp
from torch import distributed as dist
from torch.distributed.checkpoint.quantized_hf_storage import (
    QuantizedHuggingFaceStorageReader,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard, zeros
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorContinuousTestBase,
    DTensorTestBase,
    NUM_DEVICES,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import (
    random_compressed_tensors_mxfp4,
    save_hf_checkpoint,
    with_temp_dir,
)


CHECKPOINT_DIR = "checkpoint"


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


@instantiate_parametrized_tests
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
        for key in state_dict_to_save:
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_loaded[key])
            )

    @with_temp_dir
    def test_save_non_contiguous_tensor(self) -> None:
        try:
            from safetensors.torch import load_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        weight = torch.rand(4, 8, 6)
        state_dict_to_save = {"weight_t": weight.transpose(1, 2)}
        self.assertFalse(state_dict_to_save["weight_t"].is_contiguous())
        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.HuggingFaceStorageWriter(path=CHECKPOINT_DIR),
        )

        state_dict_loaded = load_file(
            CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )
        self.assertTrue(
            torch.equal(state_dict_loaded["weight_t"], weight.transpose(1, 2))
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
        for key in state_dict_to_save:
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
        for key in state_dict_to_save:
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_loaded[key])
            )

    @with_temp_dir
    def test_load_with_multiple_threads(self) -> None:
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

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
    def test_mxfp4_blocks_loading(self) -> None:
        """Load GPT-OSS MXFP4 _blocks/_scales end-to-end through the planner."""
        try:
            import safetensors  # noqa: F401
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir
        num_experts, rows, groups = 2, 3, 4
        blocks_fqn = "model.layers.0.mlp.experts.gate_up_proj_blocks"
        scales_fqn = "model.layers.0.mlp.experts.gate_up_proj_scales"
        blocks = torch.randint(256, (num_experts, rows, groups, 16), dtype=torch.uint8)
        scales = torch.randint(124, 131, (num_experts, rows, groups), dtype=torch.uint8)
        files = {"model.safetensors": {blocks_fqn: blocks, scales_fqn: scales}}
        save_hf_checkpoint(CHECKPOINT_DIR, files)

        # Reference decode: low nibble first, each value times 2**(scale - 127).
        fp4_values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        )
        expected = torch.stack(
            [fp4_values[(blocks & 0x0F).long()], fp4_values[(blocks >> 4).long()]],
            dim=-1,
        ).flatten(-2)
        expected = torch.ldexp(
            expected, scales.to(torch.int32).unsqueeze(-1) - 127
        ).flatten(-2)

        state_dict_to_load = {blocks_fqn: torch.zeros(num_experts, rows, groups * 32)}
        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=QuantizedHuggingFaceStorageReader(
                path=CHECKPOINT_DIR, target_dtype=torch.float32
            ),
        )

        torch.testing.assert_close(
            state_dict_to_load[blocks_fqn], expected, rtol=0, atol=0
        )

        # Empty-state-dict loads allocate their destination from the metadata,
        # so the result must come back dequantized, not truncated to uint8.
        loaded = _load_state_dict_from_keys(
            storage_reader=QuantizedHuggingFaceStorageReader(
                path=CHECKPOINT_DIR, target_dtype=torch.float32
            ),
        )
        self.assertEqual(loaded[blocks_fqn].dtype, torch.float32)
        torch.testing.assert_close(loaded[blocks_fqn], expected, rtol=0, atol=0)

    @with_temp_dir
    def test_quantized_checkpoint_loading(self) -> None:
        """Test end-to-end saving a quantizaed checkpoint and loading it."""
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("safetensors not installed")
            return

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

    @with_temp_dir
    def test_compressed_tensors_mxfp4_loading(self) -> None:
        """Dequantize compressed-tensors MXFP4 weight_packed using weight_scale."""
        try:
            import safetensors  # noqa: F401
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir
        rows, cols = 4, 96
        prefix = "model.layers.1.block_sparse_moe.experts.0"
        packed, scale, expected = random_compressed_tensors_mxfp4(rows, cols)
        unquantized = torch.randn(rows, cols)
        # The scale lives in a different file than the packed weight.
        save_hf_checkpoint(
            CHECKPOINT_DIR,
            {
                "model-00001-of-00002.safetensors": {
                    f"{prefix}.w1.weight_packed": packed,
                    "model.layers.1.mlp_res_proj.weight": unquantized,
                },
                "model-00002-of-00002.safetensors": {
                    f"{prefix}.w1.weight_scale": scale
                },
            },
        )

        state_dict_to_load = {
            f"{prefix}.w1.weight_packed": torch.zeros(rows, cols),
            "model.layers.1.mlp_res_proj.weight": torch.zeros(rows, cols),
        }
        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=QuantizedHuggingFaceStorageReader(path=CHECKPOINT_DIR),
        )
        torch.testing.assert_close(
            state_dict_to_load[f"{prefix}.w1.weight_packed"], expected
        )
        torch.testing.assert_close(
            state_dict_to_load["model.layers.1.mlp_res_proj.weight"], unquantized
        )

        # E2M1 values times these scales are exact in bf16.
        bf16_state_dict = {
            f"{prefix}.w1.weight_packed": torch.zeros(rows, cols, dtype=torch.bfloat16)
        }
        dist_cp.load(
            state_dict=bf16_state_dict,
            storage_reader=QuantizedHuggingFaceStorageReader(
                path=CHECKPOINT_DIR, target_dtype=torch.bfloat16
            ),
        )
        torch.testing.assert_close(
            bf16_state_dict[f"{prefix}.w1.weight_packed"],
            expected.to(torch.bfloat16),
            rtol=0,
            atol=0,
        )

        # Empty-state-dict loads allocate their destination from the metadata,
        # so the result must come back dequantized in target_dtype, not uint8.
        for dtype in (torch.float32, torch.bfloat16):
            loaded = _load_state_dict_from_keys(
                storage_reader=QuantizedHuggingFaceStorageReader(
                    path=CHECKPOINT_DIR, target_dtype=dtype
                ),
            )[f"{prefix}.w1.weight_packed"]
            self.assertEqual(loaded.dtype, dtype)
            torch.testing.assert_close(loaded, expected.to(dtype), rtol=0, atol=0)

    @parametrize(
        "case",
        [
            "nvfp4",
            "packed_not_uint8",
            "scale_not_uint8",
            "scale_not_per_32",
            "unaligned_width",
            "int4",
            "int4_3d",
        ],
    )
    @with_temp_dir
    def test_compressed_tensors_non_mxfp4_loads_raw(self, case: str) -> None:
        """Other formats with these names load raw, as before MXFP4 support."""
        try:
            import safetensors  # noqa: F401
        except ImportError:
            print("safetensors not installed")
            return

        rows = 4
        packed, scale = {
            # NVFP4: float8 scales, one per 16 values.
            "nvfp4": (
                torch.randint(0, 256, (rows, 48), dtype=torch.uint8),
                torch.rand(rows, 6).to(torch.float8_e4m3fn),
            ),
            # MXFP4 shapes, but one tensor is not uint8.
            "packed_not_uint8": (
                torch.randint(-128, 128, (rows, 48), dtype=torch.int8),
                torch.randint(0, 256, (rows, 3), dtype=torch.uint8),
            ),
            "scale_not_uint8": (
                torch.randint(0, 256, (rows, 48), dtype=torch.uint8),
                torch.rand(rows, 3).to(torch.float8_e4m3fn),
            ),
            "scale_not_per_32": (
                torch.randint(0, 256, (rows, 48), dtype=torch.uint8),
                torch.randint(0, 256, (rows, 6), dtype=torch.uint8),
            ),
            # Not a whole number of 16-byte groups.
            "unaligned_width": (
                torch.randint(0, 256, (rows, 40), dtype=torch.uint8),
                torch.randint(0, 256, (rows, 40 // 16), dtype=torch.uint8),
            ),
            # int4 pack-quantized: int32 [out, in/8] and a float scale per 128
            # values, which matches the MXFP4 scale shape.
            "int4": (
                torch.randint(0, 2**31 - 1, (rows, 32), dtype=torch.int32),
                torch.randn(rows, 2).to(torch.bfloat16),
            ),
            # int4 pack-quantized MoE weights stacked per expert are 3D.
            "int4_3d": (
                torch.randint(0, 2**31 - 1, (2, rows, 32), dtype=torch.int32),
                torch.randn(2, rows, 2).to(torch.bfloat16),
            ),
        }[case]
        prefix = "model.layers.1.block_sparse_moe.experts.0"
        tensors = {
            f"{prefix}.w1.weight_packed": packed,
            f"{prefix}.w1.weight_scale": scale,
        }
        save_hf_checkpoint(self.temp_dir, {"model.safetensors": tensors})

        state_dict = {fqn: torch.empty_like(t) for fqn, t in tensors.items()}
        dist_cp.load(
            state_dict=state_dict,
            storage_reader=QuantizedHuggingFaceStorageReader(path=self.temp_dir),
        )
        for fqn, t in tensors.items():
            self.assertEqual(state_dict[fqn].dtype, t.dtype)
            self.assertTrue(
                torch.equal(state_dict[fqn].view(torch.uint8), t.view(torch.uint8))
            )


class TestCompressedTensorsMXFP4ShardedLoad(DTensorTestBase):
    @with_comms
    @with_temp_dir
    def test_shard_starts_mid_group(self) -> None:
        """Each rank's Shard(1) slice starts partway into a 32-value group."""
        try:
            import safetensors  # noqa: F401
        except ImportError:
            print("safetensors not installed")
            return

        fqn = "model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"
        scale_fqn = "model.layers.1.block_sparse_moe.experts.0.w1.weight_scale"
        # With 160 columns, every shard boundary falls mid-group for 2, 3 or 4
        # ranks (80; 54 and 108; 40, 80 and 120).
        rows, cols = 4, 160
        torch.manual_seed(0)
        packed, scale, expected = random_compressed_tensors_mxfp4(rows, cols)
        if self.rank == 0:
            save_hf_checkpoint(
                self.temp_dir, {"model.safetensors": {fqn: packed, scale_fqn: scale}}
            )
        dist.barrier()

        mesh = init_device_mesh(self.device_type, (self.world_size,))
        state_dict = {
            fqn: distribute_tensor(
                torch.zeros(rows, cols, device=self.device_type), mesh, [Shard(1)]
            )
        }
        dist_cp.load(
            state_dict=state_dict,
            storage_reader=QuantizedHuggingFaceStorageReader(path=self.temp_dir),
        )
        torch.testing.assert_close(state_dict[fqn].full_tensor().cpu(), expected)


class TestDistributedHFSafetensorsConsolidation(DTensorTestBase):
    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_consolidate_to_one_file(self) -> None:
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

        import safetensors

        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
        mesh_shape = (self.world_size,)
        mesh_1d = init_device_mesh(self.device_type, mesh_shape)

        # Create local tensor with row-wise sharding
        rows_per_rank = global_tensor.shape[0] // self.world_size
        start_row = self.rank * rows_per_rank
        end_row = start_row + rows_per_rank
        local_tensor = global_tensor[start_row:end_row].clone()

        # Create DTensor with row-wise sharding
        dtensor = DTensor.from_local(
            local_tensor,
            device_mesh=mesh_1d,
            placements=[Shard(0)],
            shape=global_tensor.shape,
            stride=(4, 1),
        )

        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)

        checkpoint_dir = self.temp_dir

        state_dict_to_save = {"dtensor": dtensor}
        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=checkpoint_dir,
                save_distributed=True,
                enable_consolidation=True,
            ),
        )
        dist.barrier()

        if self.rank == 0:
            file_path = os.path.join(checkpoint_dir, "model-00001-of-00001.safetensors")
            loaded_dict = safetensors.torch.load_file(file_path)
            self.assertEqual(loaded_dict.keys(), {"dtensor"})
            self.assertTrue(torch.equal(loaded_dict["dtensor"], global_tensor))

        dist.barrier()


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
class TestDTensorReshardPlacementChange(DTensorContinuousTestBase):
    """
    Test DCP reshard for DTensor with placements changes and without world_size change and mesh_tensor change.
    """

    world_size = NUM_DEVICES

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
                    save_distributed=True,
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


class TestDTensorReshardMeshChange(DTensorContinuousTestBase):
    """
    Test DCP reshard for DTensor with placements changes and mesh_tensor change.
    """

    world_size = NUM_DEVICES

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
    @skip_if_lt_x_gpu(2)
    def test_dtensor_checkpoint_resharding_with_empty_shard(self):
        """
        Test dtensor checkpoint resharding with dtensor containing empty shards.
        """
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return

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


if __name__ == "__main__":
    run_tests()
