# Owner(s): ["oncall: distributed checkpointing"]

import importlib
import json
import os
import unittest

import torch
import torch.distributed.checkpoint as dist_cp
from torch import distributed as dist
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    _calculate_max_contiguous_elements,
    _FqnData,
    _InputFileData,
    _OutputFileData,
    _parse_input_metadata,
    _write_sub_tensor_to_file_optimized,
    consolidate_safetensors_files,
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint._hf_utils import _metadata_fn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
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
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class TestConsolidateHFSafeTensors(DTensorContinuousTestBase):
    world_size = NUM_DEVICES

    def _create_d_tensors(self) -> None:
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

        # Create local tensor with column-wise sharding
        cols_per_rank = global_tensor.shape[1] // self.world_size
        start_col = self.rank * cols_per_rank
        end_col = start_col + cols_per_rank
        local_tensor_col = global_tensor[:, start_col:end_col].clone()

        # Create DTensor with column-wise sharding
        dtensor_col = DTensor.from_local(
            local_tensor_col,
            device_mesh=mesh_1d,
            placements=[Shard(1)],  # Column-wise sharding
            shape=global_tensor.shape,
            stride=(4, 1),
        )

        state_dict_to_save = {"dtensor": dtensor, "dtensor_col": dtensor_col}
        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.HuggingFaceStorageWriter(
                path=self.temp_dir, save_distributed=True
            ),
        )
        dist.barrier()
        os.sync()

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_consolidate_to_one_file(self) -> None:
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return
        import safetensors

        checkpoint_dir = self.temp_dir
        output_dir = os.path.join(checkpoint_dir, "consolidated")
        os.makedirs(output_dir, exist_ok=True)

        self._create_d_tensors()

        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)

        if self.rank == 0:
            consolidate_safetensors_files(
                checkpoint_dir,
                output_dir,
                fqn_to_index_mapping={"dtensor": 1, "dtensor_col": 1},
            )

            file_path = os.path.join(output_dir, "model-00001-of-00001.safetensors")
            loaded_dict = safetensors.torch.load_file(file_path)
            self.assertEqual(loaded_dict.keys(), {"dtensor", "dtensor_col"})
            self.assertTrue(torch.equal(loaded_dict["dtensor"], global_tensor))
            self.assertTrue(torch.equal(loaded_dict["dtensor_col"], global_tensor))

            with open(os.path.join(output_dir, _metadata_fn)) as f:
                metadata = json.load(f)
                self.assertEqual(metadata["metadata"]["total_size"], 16 * 4 * 2)
                self.assertEqual(
                    metadata["weight_map"],
                    {
                        "dtensor": "model-00001-of-00001.safetensors",
                        "dtensor_col": "model-00001-of-00001.safetensors",
                    },
                )

        dist.barrier()

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_consolidate_to_two_files(self):
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return
        import safetensors

        checkpoint_dir = self.temp_dir
        output_dir = os.path.join(checkpoint_dir, "consolidated")
        os.makedirs(output_dir, exist_ok=True)

        self._create_d_tensors()

        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)

        if self.rank == 0:
            fqn_to_index_mapping = {"dtensor": 1, "dtensor_col": 2}
            consolidate_safetensors_files(
                checkpoint_dir, output_dir, fqn_to_index_mapping=fqn_to_index_mapping
            )

            file1_path = os.path.join(output_dir, "model-00001-of-00002.safetensors")
            file2_path = os.path.join(output_dir, "model-00002-of-00002.safetensors")

            loaded_dict = safetensors.torch.load_file(file1_path)
            self.assertEqual(loaded_dict.keys(), {"dtensor"})
            self.assertTrue(torch.equal(loaded_dict["dtensor"], global_tensor))

            loaded_dict_col = safetensors.torch.load_file(file2_path)
            self.assertEqual(loaded_dict_col.keys(), {"dtensor_col"})
            self.assertTrue(torch.equal(loaded_dict_col["dtensor_col"], global_tensor))

            with open(os.path.join(output_dir, _metadata_fn)) as f:
                metadata = json.load(f)
                self.assertEqual(metadata["metadata"]["total_size"], 16 * 4 * 2)
                self.assertEqual(
                    metadata["weight_map"],
                    {
                        "dtensor": "model-00001-of-00002.safetensors",
                        "dtensor_col": "model-00002-of-00002.safetensors",
                    },
                )
        dist.barrier()

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_consolidate_with_two_ranks(self):
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return
        import safetensors

        checkpoint_dir = self.temp_dir
        output_dir = os.path.join(checkpoint_dir, "consolidated")
        os.makedirs(output_dir, exist_ok=True)

        self._create_d_tensors()

        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)

        fqn_to_index_mapping = {"dtensor": 1, "dtensor_col": 2}
        consolidate_safetensors_files_on_every_rank(
            checkpoint_dir, output_dir, fqn_to_index_mapping=fqn_to_index_mapping
        )

        file1_path = os.path.join(output_dir, "model-00001-of-00002.safetensors")
        file2_path = os.path.join(output_dir, "model-00002-of-00002.safetensors")

        loaded_dict = safetensors.torch.load_file(file1_path)
        self.assertEqual(loaded_dict.keys(), {"dtensor"})
        self.assertTrue(torch.equal(loaded_dict["dtensor"], global_tensor))

        loaded_dict_col = safetensors.torch.load_file(file2_path)
        self.assertEqual(loaded_dict_col.keys(), {"dtensor_col"})
        self.assertTrue(torch.equal(loaded_dict_col["dtensor_col"], global_tensor))

        metadata_path = os.path.join(output_dir, _metadata_fn)
        self.assertTrue(os.path.exists(metadata_path))
        with open(metadata_path) as f:
            metadata = json.load(f)
            self.assertEqual(metadata["metadata"]["total_size"], 16 * 4 * 2)
            self.assertEqual(
                metadata["weight_map"],
                {
                    "dtensor": "model-00001-of-00002.safetensors",
                    "dtensor_col": "model-00002-of-00002.safetensors",
                },
            )

        dist.barrier()

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_consolidate_one_file_with_two_ranks(self):
        if importlib.util.find_spec("safetensors") is None:
            print("safetensors not installed")
            return
        import safetensors

        # this is testing the case where one rank has no data to write
        # and the other rank has two tensors to write.
        # the rank with no work should wait properly for the other rank to finish
        checkpoint_dir = self.temp_dir
        output_dir = os.path.join(checkpoint_dir, "consolidated")
        os.makedirs(output_dir, exist_ok=True)

        self._create_d_tensors()

        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)

        fqn_to_index_mapping = {"dtensor": 1, "dtensor_col": 1}
        consolidate_safetensors_files_on_every_rank(
            checkpoint_dir, output_dir, fqn_to_index_mapping=fqn_to_index_mapping
        )

        file1_path = os.path.join(output_dir, "model-00001-of-00001.safetensors")

        loaded_dict = safetensors.torch.load_file(file1_path)
        self.assertEqual(loaded_dict.keys(), {"dtensor", "dtensor_col"})
        self.assertTrue(torch.equal(loaded_dict["dtensor"], global_tensor))
        self.assertTrue(torch.equal(loaded_dict["dtensor_col"], global_tensor))


@instantiate_parametrized_tests
@unittest.skipIf(
    importlib.util.find_spec("safetensors") is None,
    "safetensors not installed",
)
class TestConsolidateHFSafeTensorsDtypes(TestCase):
    @parametrize(
        "code,expected_size",
        [
            ("I8", 1),
            ("U8", 1),
            ("I16", 2),
            ("I32", 4),
            ("I64", 8),
            ("BOOL", 1),
            ("F32", 4),
            ("F16", 2),
            ("BF16", 2),
            ("F64", 8),
            ("F8_E4M3", 1),
            ("F8_E5M2", 1),
            ("U16", 2),
            ("U32", 4),
            ("U64", 8),
            ("C64", 8),
        ],
    )
    def test_parse_input_metadata_dtype_sizes(
        self, code: str, expected_size: int
    ) -> None:
        from safetensors.torch import _TYPES

        if code not in _TYPES:
            self.skipTest(f"{code} not supported by installed safetensors")

        inputs = {
            "shard": _InputFileData(
                metadata={
                    "__metadata__": {
                        "DCP_SHARDING_INFO": json.dumps(
                            {"weight": {"saved_offsets": [0]}}
                        )
                    },
                    "weight": {"shape": [4], "dtype": code},
                }
            )
        }
        outputs = {"out": _OutputFileData(fqn_data={"weight": _FqnData()})}

        _parse_input_metadata(inputs, outputs)
        result = outputs["out"].fqn_data["weight"]

        self.assertEqual(result.shape_in_file, [4])
        self.assertEqual(result.dtype_str, code)
        self.assertEqual(result.dtype_size, expected_size)

    @with_temp_dir
    def test_consolidate_integer_bool_shards_cpu(self) -> None:
        from safetensors.torch import load_file, save_file

        expected = {
            "row": (torch.arange(16, dtype=torch.int32) - 8).view(4, 4),
            "col": (torch.arange(16, dtype=torch.int8) - 5).view(4, 4),
            "mask": (torch.arange(16).view(4, 4) % 3 == 0),
        }

        input_dir = os.path.join(self.temp_dir, "input")
        output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        for rank in range(2):
            start = rank * 2
            shard = {
                "row": expected["row"][start : start + 2].contiguous(),
                "col": expected["col"][:, start : start + 2].contiguous(),
                "mask": expected["mask"][:, start : start + 2].contiguous(),
            }
            offsets = {
                "row": [start, 0],
                "col": [0, start],
                "mask": [0, start],
            }
            metadata = {
                "DCP_SHARDING_INFO": json.dumps(
                    {
                        name: {"saved_offsets": offset}
                        for name, offset in offsets.items()
                    }
                )
            }
            save_file(
                shard,
                os.path.join(input_dir, f"shard{rank}.safetensors"),
                metadata=metadata,
            )

        consolidate_safetensors_files(
            input_dir,
            output_dir,
            fqn_to_index_mapping={name: 1 for name in expected},
        )

        loaded = load_file(
            os.path.join(output_dir, "model-00001-of-00001.safetensors")
        )

        self.assertEqual(loaded, expected)

        with open(os.path.join(output_dir, _metadata_fn)) as f:
            index = json.load(f)

        self.assertEqual(
            index["metadata"]["total_size"],
            16 * 4 + 16 * 1 + 16 * 1,
        )


class TestConsolidateHFSafeTensorsNoProcessGroup(DTensorTestBase):
    def test_calculate_max_contiguous_elements_validations(self) -> None:
        """Test validation logic in _calculate_max_contiguous_elements function."""

        # Test empty lists validation
        with self.assertRaisesRegex(ValueError, "Input lists cannot be empty"):
            _calculate_max_contiguous_elements([], [2, 3], [4, 5])

        # Test mismatched list lengths validation
        with self.assertRaisesRegex(
            ValueError, "All input lists must have the same length"
        ):
            _calculate_max_contiguous_elements([1], [2, 3], [4, 5])

        # Test indices out of bounds validation
        with self.assertRaisesRegex(
            ValueError, "Index .* at dimension .* is out of bounds for sub-tensor shape"
        ):
            _calculate_max_contiguous_elements(
                [2, 1], [2, 3], [4, 5]
            )  # indices[0] >= sub_tensor_shape[0]

        # Test sub-tensor dimensions exceeding tensor dimensions validation
        with self.assertRaisesRegex(
            ValueError,
            "Sub-tensor dimension .* at position .* exceeds tensor dimension",
        ):
            _calculate_max_contiguous_elements(
                [1, 2], [2, 6], [4, 5]
            )  # sub_tensor_shape[1] > tensor_shape[1]

    def test_calculate_max_contiguous_elements_valid_cases(self) -> None:
        """Test valid cases for _calculate_max_contiguous_elements function."""

        # Test 1D case - simple remaining elements
        result = _calculate_max_contiguous_elements([2], [5], [10])
        self.assertEqual(result, 3)  # 5 - 2 = 3 elements remaining

        # Test 2D case - at start of row, can write complete rows
        result = _calculate_max_contiguous_elements([1, 0], [3, 4], [6, 4])
        self.assertEqual(result, 8)  # 2 rows * 4 columns = 8 elements

        # Test 2D case - middle of row, only remaining in current row
        result = _calculate_max_contiguous_elements([1, 2], [3, 4], [6, 8])
        self.assertEqual(result, 2)  # 4 - 2 = 2 elements remaining in row

        # Test 3D case - at start of 2D slice, can write complete slices
        result = _calculate_max_contiguous_elements([1, 0, 0], [3, 2, 4], [5, 2, 4])
        self.assertEqual(result, 16)  # 2 slices * 2 rows * 4 columns = 16 elements

        # Test edge case - at last position
        result = _calculate_max_contiguous_elements([2, 3], [3, 4], [6, 8])
        self.assertEqual(result, 1)  # Only 1 element remaining

        # Test case where sub-tensor spans full width
        result = _calculate_max_contiguous_elements([0, 0], [2, 5], [4, 5])
        self.assertEqual(result, 10)  # 2 rows * 5 columns = 10 elements

        # Test column-wise sharded case - sub-tensor doesn't span full width
        # Even at start of row, can only write width of one row due to column sharding
        result = _calculate_max_contiguous_elements([1, 0], [3, 2], [4, 8])
        self.assertEqual(
            result, 2
        )  # Only 2 elements (width of sub-tensor) can be written contiguously

        # Test another column-wise sharded case - middle of tensor
        result = _calculate_max_contiguous_elements([0, 0], [2, 3], [6, 10])
        self.assertEqual(
            result, 3
        )  # Only 3 elements (width of sub-tensor) can be written contiguously

    def test_write_sub_tensor_to_file_optimized(self) -> None:
        """Test the _write_sub_tensor_to_file_optimized function with various scenarios."""

        # Test case 1: Simple 2D tensor, row-wise sharding
        full_tensor_shape = [4, 6]
        sub_tensor_shape = [2, 6]
        sub_tensor_offsets = [1, 0]
        element_size = 4  # float32

        # Create test data
        sub_tensor_data = torch.arange(12, dtype=torch.float32)
        sub_tensor_bytes = sub_tensor_data.numpy().tobytes()

        # Create full tensor buffer
        full_tensor_buffer = bytearray(4 * 6 * element_size)
        full_tensor_mv = memoryview(full_tensor_buffer)

        # Call the function
        _write_sub_tensor_to_file_optimized(
            full_tensor_mv,
            sub_tensor_bytes,
            element_size,
            full_tensor_shape,
            sub_tensor_offsets,
            sub_tensor_shape,
        )

        # Verify the result
        result_tensor = torch.frombuffer(full_tensor_buffer, dtype=torch.float32).view(
            4, 6
        )
        expected_tensor = torch.zeros(4, 6, dtype=torch.float32)
        expected_tensor[1:3, :] = sub_tensor_data.view(2, 6)

        self.assertTrue(torch.equal(result_tensor, expected_tensor))

        # Test case 2: Column-wise sharding
        full_tensor_shape = [3, 8]
        sub_tensor_shape = [3, 2]
        sub_tensor_offsets = [0, 3]

        sub_tensor_data = torch.arange(6, dtype=torch.float32)
        sub_tensor_bytes = sub_tensor_data.numpy().tobytes()

        full_tensor_buffer = bytearray(3 * 8 * element_size)
        full_tensor_mv = memoryview(full_tensor_buffer)

        _write_sub_tensor_to_file_optimized(
            full_tensor_mv,
            sub_tensor_bytes,
            element_size,
            full_tensor_shape,
            sub_tensor_offsets,
            sub_tensor_shape,
        )

        result_tensor = torch.frombuffer(full_tensor_buffer, dtype=torch.float32).view(
            3, 8
        )
        expected_tensor = torch.zeros(3, 8, dtype=torch.float32)
        expected_tensor[:, 3:5] = sub_tensor_data.view(3, 2)

        self.assertTrue(torch.equal(result_tensor, expected_tensor))


if __name__ == "__main__":
    run_tests()
