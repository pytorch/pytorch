# Owner(s): ["oncall: distributed checkpointing"]

import json
import os
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock

import torch
from torch.distributed.checkpoint._hf_planner import (
    _FqnToFileMapping,
    _HuggingFaceLoadPlanner,
)
from torch.distributed.checkpoint._hf_storage import (
    _HuggingFaceStorageReader,
    _HuggingFaceStorageWriter,
    _metadata_fn,
)
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.filesystem import _StorageInfo, FileSystem
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    MetadataIndex,
)
from torch.distributed.checkpoint.planner import LoadPlan, SavePlan
from torch.distributed.checkpoint.planner_helpers import (
    _create_read_items,
    _create_write_item_for_tensor,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.testing._internal.common_utils import run_tests, TestCase


class TestHfStorage(TestCase):
    def test_write_data_hf(self) -> None:
        mock_module = MagicMock()
        sys.modules["safetensors"] = mock_module
        sys.modules["huggingface_hub"] = mock_module

        mock_module = MagicMock()
        mock_module.save.return_value = b""
        sys.modules["safetensors.torch"] = mock_module

        with tempfile.TemporaryDirectory() as path:
            writer = _HuggingFaceStorageWriter(
                path=path,
                fqn_to_index_mapping={"tensor_0": 1, "tensor_1": 1},
            )
            writer.fs = FileSystem()

            tensor0 = torch.rand(4)
            tensor1 = torch.rand(10)
            write_item_1 = _create_write_item_for_tensor("tensor_0", tensor0)
            write_item_2 = _create_write_item_for_tensor("tensor_1", tensor1)

            state_dict = {"tensor_0": tensor0, "tensor_1": tensor1}

            save_plan = SavePlan(
                [write_item_1, write_item_2],
                storage_data=_FqnToFileMapping({"tensor_0": 1, "tensor_1": 1}),
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
                        relative_path="model-00001-of-00001.safetensors",
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
                        relative_path="model-00001-of-00001.safetensors",
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
        mock_module = MagicMock()
        sys.modules["safetensors"] = mock_module
        sys.modules["huggingface_hub"] = mock_module

        name = "tensor_0"
        tensor_0 = torch.rand(4)
        mock_module = MagicMock()
        mock_module.load.return_value = {name: tensor_0}
        sys.modules["safetensors.torch"] = mock_module

        with tempfile.TemporaryDirectory() as path:
            reader = _HuggingFaceStorageReader(path=path)
            reader.fs = FileSystem()
            file_name = "model-00001-of-00001"

            pathlib.Path(os.path.join(path, file_name)).touch()

            reader.set_up_storage_reader(
                Metadata(
                    state_dict_metadata={name: BytesStorageMetadata()},
                    storage_data={name: file_name},
                ),
                is_coordinator=True,
            )

            read_items = _create_read_items(name, BytesStorageMetadata(), file_name)
            load_plan = LoadPlan(read_items)
            load_planner = _HuggingFaceLoadPlanner()
            load_planner.set_up_planner(state_dict={name: torch.rand(4)})

            read_data = reader.read_data(load_plan, load_planner)
            read_data.wait()

            loaded_tensor = load_planner.original_state_dict[name]
            self.assertEqual(loaded_tensor, tensor_0)

    def test_metadata_hf(self) -> None:
        mock_module = MagicMock()
        sys.modules["huggingface_hub"] = mock_module
        with tempfile.TemporaryDirectory() as path:
            file_name = "model-00001-of-00001"
            write_results = [
                WriteResult(
                    index=MetadataIndex(fqn="tensor_0", offset=None, index=None),
                    size_in_bytes=100,
                    storage_data=_StorageInfo(
                        relative_path=file_name, offset=0, length=100
                    ),
                ),
                WriteResult(
                    index=MetadataIndex(fqn="tensor_1", offset=None, index=None),
                    size_in_bytes=100,
                    storage_data=_StorageInfo(
                        relative_path=file_name, offset=0, length=100
                    ),
                ),
            ]

            writer = _HuggingFaceStorageWriter(
                path=path,
                fqn_to_index_mapping=_FqnToFileMapping({}),
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

            reader = _HuggingFaceStorageReader(path=path)
            reader.fs = FileSystem()
            metadata = reader.read_metadata()
            self.assertEqual(metadata.storage_data, expected_metadata["weight_map"])

    def test_read_metadata_when_metadata_file_does_not_exist(self) -> None:
        mock_module = MagicMock()
        sys.modules["huggingface_hub"] = mock_module

        with tempfile.TemporaryDirectory() as path:
            reader = _HuggingFaceStorageReader(path=path)
            reader.fs = FileSystem()
            # there is one safetensor file, but no metadata file,
            # so we create metadata from the safetensor file
            keys = ["tensor_0", "tensor_1"]
            file_name = "test.safetensors"
            with open(os.path.join(path, file_name), "wb") as f:
                # write metadata the same way it would be in safetensors file
                metadata_contents = json.dumps(
                    {"tensor_0": "value_0", "tensor_1": "value_1"}
                )
                metadata_bytes = metadata_contents.encode("utf-8")

                f.write(len(metadata_bytes).to_bytes(8, byteorder="little"))
                f.write(metadata_bytes)

            metadata = reader.read_metadata()

            self.assertEqual(
                metadata.state_dict_metadata,
                {
                    keys[0]: BytesStorageMetadata(),
                    keys[1]: BytesStorageMetadata(),
                },
            )
            self.assertEqual(
                metadata.storage_data,
                {keys[0]: file_name, keys[1]: file_name},
            )


if __name__ == "__main__":
    run_tests()
