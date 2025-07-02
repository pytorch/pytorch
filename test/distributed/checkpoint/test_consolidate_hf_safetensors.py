# Owner(s): ["oncall: distributed checkpointing"]

import importlib
import json
import os

import torch
import torch.distributed.checkpoint as dist_cp
from torch import distributed as dist
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files,
)
from torch.distributed.checkpoint._hf_utils import _metadata_fn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class TestConsolidateHFSafeTensors(DTensorTestBase):
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
            consolidate_safetensors_files(checkpoint_dir, output_dir)

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


if __name__ == "__main__":
    run_tests()
