# Owner(s): ["oncall: distributed checkpointing"]

import io

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    TensorStorageMetadata,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class TestListStateDict(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_list_stored_sd_metadata(self) -> None:
        CHECKPOINT_DIR = self.temp_dir
        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(self.device_type, mesh_shape)
        # Save 1) DTensor 2) Blob
        dtensor = distribute_tensor(
            global_tensor,
            mesh_2d,
            placements=[Shard(0), Replicate()],
        )
        state_dict_to_save = {
            "distributed_weight": dtensor,
            "bytes_data": io.BytesIO(b"TrainingEpoch:4"),
        }

        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
            planner=dist_cp.DefaultSavePlanner(),
        )
        md = dist_cp.list_stored_state_dict(checkpoint_id=CHECKPOINT_DIR)

        # Verify  DTensor
        self.assertTrue("distributed_weight" in md)
        self.assertTrue(isinstance(md["distributed_weight"], TensorStorageMetadata))
        self.assertEqual(md["distributed_weight"].size, torch.Size([4, 4]))
        self.assertEqual(md["distributed_weight"].properties.dtype, torch.float32)

        # Verify Blob
        self.assertTrue("bytes_data" in md)
        self.assertTrue(isinstance(md["bytes_data"], BytesStorageMetadata))


if __name__ == "__main__":
    run_tests()
