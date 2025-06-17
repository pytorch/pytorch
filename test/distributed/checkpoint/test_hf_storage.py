# Owner(s): ["oncall: distributed checkpointing"]

import json
import os
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock

import torch
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint._hf_utils import _HFStorageInfo
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.filesystem import _StorageInfo, FileSystem
from torch.distributed.checkpoint.hf_storage import (
    _metadata_fn,
    HuggingFaceStorageReader,
    HuggingFaceStorageWriter,
)
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    ReadItem,
    SavePlan,
)
from torch.distributed.checkpoint.planner_helpers import _create_write_item_for_tensor
from torch.distributed.checkpoint.storage import WriteResult
from torch.testing._internal.common_utils import run_tests, TestCase


class TestHfStorage(TestCase):
    def test_write_data_hf(self) -> None:
        mock_module = MagicMock()
        mock_module.save.return_value = b""
        sys.modules["safetensors.torch"] = mock_module

        with tempfile.TemporaryDirectory() as path:
            writer = HuggingFaceStorageWriter(
                path=path,
                fqn_to_index_mapping={"tensor_0": 1, "tensor_1": 2},
            )
            writer.fs = FileSystem()

            tensor0 = torch.rand(4)
            tensor1 = torch.rand(10)
            write_item_1 = _create_write_item_for_tensor("tensor_0", tensor0)
            write_item_2 = _create_write_item_for_tensor("tensor_1", tensor1)

            state_dict = {"tensor_0": tensor0, "tensor_1": tensor1}

            save_plan = SavePlan(
                [write_item_1, write_item_2],
                storage_data={"fqn_to_index_mapping": {"tensor_0": 1, "tensor_1": 2}},
            )
            save_planner = DefaultSavePlanner()
            save_planner.set_up_planner(state_dict=state_dict)

            write_results = writer.write_data(save_plan, save_planner)

            write_results.wait()
            actual_write_results = write_results.value()

            expected_write_results = [
                WriteResult(
                    index=MetadataIndex(
                        fqn="tensor_0", offset=torch.Size([0]), index=None
                    ),
                    size_in_bytes=tensor0.numel() * tensor0.element_size(),
                    storage_data=_StorageInfo(
                        relative_path="model-00001-of-00002.safetensors",
                        offset=0,
                        length=tensor0.numel() * tensor0.element_size(),
                    ),
                ),
                WriteResult(
                    index=MetadataIndex(
                        fqn="tensor_1", offset=torch.Size([0]), index=None
                    ),
                    size_in_bytes=tensor1.numel() * tensor1.element_size(),
                    storage_data=_StorageInfo(
                        relative_path="model-00002-of-00002.safetensors",
                        offset=0,
                        length=tensor1.numel() * tensor1.element_size(),
                    ),
                ),
            ]

            self.assertEqual(
                actual_write_results,
                expected_write_results,
            )

    def test_write_data_with_sharding(self) -> None:
        mock_module = MagicMock()
        mock_module.save.return_value = b""
        sys.modules["safetensors.torch"] = mock_module

        with tempfile.TemporaryDirectory() as path:
            writer = HuggingFaceStorageWriter(
                path=path,
                save_sharded=True,
            )
            writer.fs = FileSystem()

            tensor0 = torch.rand(4)
            tensor1 = torch.rand(10)
            write_item_1 = _create_write_item_for_tensor("tensor_0", tensor0)
            write_item_2 = _create_write_item_for_tensor("tensor_1", tensor1)

            state_dict = {"tensor_0": tensor0, "tensor_1": tensor1}

            save_plan = SavePlan(
                [write_item_1, write_item_2],
                storage_data={"shard_index": 1},
            )
            save_planner = DefaultSavePlanner()
            save_planner.set_up_planner(state_dict=state_dict)

            write_results = writer.write_data(save_plan, save_planner)

            write_results.wait()
            actual_write_results = write_results.value()

            expected_write_results = [
                WriteResult(
                    index=MetadataIndex(
                        fqn="tensor_0", offset=torch.Size([0]), index=None
                    ),
                    size_in_bytes=tensor0.numel() * tensor0.element_size(),
                    storage_data=_StorageInfo(
                        relative_path="shard-00001-model-00001-of-00001.safetensors",
                        offset=0,
                        length=tensor0.numel() * tensor0.element_size(),
                    ),
                ),
                WriteResult(
                    index=MetadataIndex(
                        fqn="tensor_1", offset=torch.Size([0]), index=None
                    ),
                    size_in_bytes=tensor1.numel() * tensor1.element_size(),
                    storage_data=_StorageInfo(
                        relative_path="shard-00001-model-00001-of-00001.safetensors",
                        offset=0,
                        length=tensor1.numel() * tensor1.element_size(),
                    ),
                ),
            ]

            self.assertEqual(
                actual_write_results,
                expected_write_results,
            )

    def test_read_data_hf(self) -> None:
        mock_safetensors = MagicMock()
        sys.modules["safetensors"] = mock_safetensors

        # Create test tensors
        tensor_0 = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # Mock the deserialize function to return our test tensors
        # The format matches what's expected in the read_data method
        mock_safetensors.deserialize.return_value = [
            (
                "tensor_0",
                {"data": tensor_0.numpy().tobytes(), "dtype": "F32", "shape": [4]},
            ),
        ]

        with tempfile.TemporaryDirectory() as path:
            # Create the reader
            reader = HuggingFaceStorageReader(path=path)
            reader.fs = FileSystem()

            # Create test file
            file_name = "model-00001-of-00001.safetensors"
            file_path = os.path.join(path, file_name)
            pathlib.Path(file_path).touch()

            # Set up storage data with _StorageInfo objects
            storage_data = {
                MetadataIndex(
                    fqn="tensor_0", offset=torch.Size([0]), index=None
                ): _HFStorageInfo(
                    file_path,
                    0,
                    tensor_0.numel() * tensor_0.element_size(),
                    tensor_0.shape,
                    tensor_0.dtype,
                ),
            }

            reader.storage_data = storage_data

            # Create target tensors that will be updated by read_data
            target_tensor_0 = torch.zeros(4)
            state_dict = {
                "tensor_0": target_tensor_0,
            }

            # Create read items for the load plan
            read_items = []
            for name, tensor in state_dict.items():
                storage_index = MetadataIndex(
                    fqn=name, offset=torch.Size([0]), index=None
                )
                dest_index = MetadataIndex(fqn=name, offset=torch.Size([0]), index=None)
                read_items.append(
                    ReadItem(
                        type=LoadItemType.TENSOR,
                        storage_index=storage_index,
                        dest_index=dest_index,
                        storage_offsets=[0, 0],
                        dest_offsets=[0, 0],
                        lengths=tensor.size(),
                    )
                )

            # Create load plan and planner
            load_plan = LoadPlan(read_items)
            load_planner = DefaultLoadPlanner()
            load_planner.set_up_planner(
                state_dict=state_dict,
                metadata=Metadata(
                    state_dict_metadata={
                        "tensor_0": TensorStorageMetadata(
                            properties=TensorProperties(dtype=torch.float32),
                            size=torch.Size([4]),
                            chunks=[
                                ChunkStorageMetadata(
                                    offsets=torch.Size([0]), sizes=torch.Size([4])
                                )
                            ],
                        )
                    },
                    storage_data=storage_data,
                ),
            )

            # Call read_data
            future = reader.read_data(load_plan, load_planner)
            future.wait()

            # Verify results - the target tensors should now contain the values from our test tensor
            self.assertTrue(torch.equal(state_dict["tensor_0"], tensor_0))

    def test_write_metadata_hf(self) -> None:
        mock_module = MagicMock()
        sys.modules["huggingface_hub"] = mock_module
        with tempfile.TemporaryDirectory() as path:
            file_name = "model-00001-of-00001"
            write_results = [
                WriteResult(
                    index=MetadataIndex(fqn="tensor_0", offset=None, index=None),
                    size_in_bytes=100,
                    storage_data=_StorageInfo(
                        relative_path=file_name,
                        offset=0,
                        length=100,
                    ),
                ),
                WriteResult(
                    index=MetadataIndex(fqn="tensor_1", offset=None, index=None),
                    size_in_bytes=100,
                    storage_data=_StorageInfo(
                        relative_path=file_name,
                        offset=0,
                        length=100,
                    ),
                ),
            ]

            writer = HuggingFaceStorageWriter(
                path=path,
            )
            writer.fs = FileSystem()
            writer.finish(
                Metadata(
                    state_dict_metadata={
                        "tensor_0": BytesStorageMetadata(),
                        "tensor_1": BytesStorageMetadata(),
                    }
                ),
                results=[write_results],
            )
            metadata_file = os.path.join(path, _metadata_fn)

            expected_metadata = {
                "metadata": {"total_size": 200},
                "weight_map": {
                    "tensor_0": "model-00001-of-00001",
                    "tensor_1": "model-00001-of-00001",
                },
            }
            with open(metadata_file) as f:
                metadata = json.load(f)
                self.assertEqual(metadata, expected_metadata)

    def test_read_metadata_hf(self):
        with tempfile.TemporaryDirectory() as path:
            reader = HuggingFaceStorageReader(path=path)

            key = "tensor_0"
            file_name = "test.safetensors"
            with open(os.path.join(path, file_name), "wb") as f:
                # write metadata the same way it would be in safetensors file
                metadata_contents = json.dumps(
                    {
                        "tensor_0": {
                            "dtype": "F32",
                            "shape": [5, 10],
                            "data_offsets": [0, 200],
                        }
                    }
                )
                metadata_bytes = metadata_contents.encode("utf-8")

                f.write(len(metadata_bytes).to_bytes(8, byteorder="little"))
                f.write(metadata_bytes)

            metadata = reader.read_metadata()

            self.assertEqual(
                metadata.state_dict_metadata,
                {
                    key: TensorStorageMetadata(
                        properties=TensorProperties(dtype=torch.float32),
                        size=torch.Size([5, 10]),
                        chunks=[
                            ChunkStorageMetadata(
                                offsets=torch.Size([0, 0]), sizes=torch.Size([5, 10])
                            )
                        ],
                    ),
                },
            )
            self.assertEqual(
                metadata.storage_data,
                {
                    MetadataIndex(
                        fqn=key, offset=torch.Size([0, 0]), index=None
                    ): _HFStorageInfo(
                        os.path.join(path, file_name),
                        0,
                        200,
                        torch.Size([5, 10]),
                        torch.float32,
                    )
                },
            )


if __name__ == "__main__":
    run_tests()
