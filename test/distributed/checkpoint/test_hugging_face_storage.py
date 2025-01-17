import io
import tempfile
from unittest.mock import ANY, patch

import torch
from torch.distributed.checkpoint._hugging_face_storage import (
    _StorageInfo,
    _StoragePrefix,
    HuggingFaceHubReader,
    HuggingFaceHubWriter,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    MetadataIndex,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    SavePlan,
)
from torch.distributed.checkpoint.planner_helpers import (
    _create_read_items,
    _create_write_item_for_tensor,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.testing._internal.common_utils import TestCase


class TestHuggingFaceStorage(TestCase):
    @patch("huggingface_hub.HfApi.upload_file")
    @patch("huggingface_hub.create_repo")
    def test_write_data_hf(self, _mock_create_repo, mock_upload_file) -> None:
        repo_id = "test_repo_id"
        writer = HuggingFaceHubWriter(
            repo_id=repo_id,
        )

        tensor0 = torch.rand(4)
        tensor1 = torch.rand(10)
        write_item_1 = _create_write_item_for_tensor("tensor_0", tensor0)
        write_item_2 = _create_write_item_for_tensor("tensor_1", tensor1)

        state_dict = {"tensor_0": tensor0, "tensor_1": tensor1}

        save_plan = SavePlan([write_item_1, write_item_2], storage_data=_StoragePrefix("test"))
        save_planner = DefaultSavePlanner()
        save_planner.set_up_planner(state_dict=state_dict)

        write_results = writer.write_data(save_plan, save_planner)

        mock_upload_file.assert_any_call(
                    path_or_fileobj=ANY,
                    path_in_repo="test0.distcp",
                    repo_id=repo_id,
                    repo_type="model",
                    token=None,)

        write_results.wait()
        actual_write_results = write_results.value()
        
        expected_write_results = [WriteResult(index=MetadataIndex(fqn='tensor_0', offset=torch.Size([0]), index = None), size_in_bytes=ANY,
        storage_data=_StorageInfo(relative_path="test0.distcp", offset=0, length=ANY)),
         WriteResult(index=MetadataIndex(fqn='tensor_1', offset=torch.Size([0]), index = None), size_in_bytes=ANY,
        storage_data=_StorageInfo(relative_path="test0.distcp", offset=ANY, length=ANY))]
        
        self.assertEqual(actual_write_results, expected_write_results)

    @patch("huggingface_hub.snapshot_download")
    def test_read_data_hf(self, mock_snapshot_download) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_snapshot_download.return_value = temp_dir

            reader = HuggingFaceHubReader(repo_id="test_repo_id")
            name = "blob_0"
            bytes_data_write = b"blob 0 text"
            with open(f"{temp_dir}/{name}.txt", "wb") as file:
                torch.save(bytes_data_write, file)

            metadata_index = MetadataIndex(fqn='blob_0', offset=None, index=None)
            reader.set_up_storage_reader(
                Metadata(
                    state_dict_metadata = {name: BytesStorageMetadata()},
                    storage_data={metadata_index: _StorageInfo(relative_path="blob_0.txt", offset=0, length=864)}),
                is_coordinator=True)

            
            read_items = _create_read_items(name, BytesStorageMetadata(), name + ".txt")
            load_plan = LoadPlan(read_items)
            load_planner = DefaultLoadPlanner()
            load_planner.set_up_planner(state_dict={name: io.BytesIO()})

            read_data = reader.read_data(load_plan, load_planner)
            read_data.wait()

            loaded_blob = load_planner.original_state_dict[name]
            self.assertEqual(loaded_blob, bytes_data_write)
