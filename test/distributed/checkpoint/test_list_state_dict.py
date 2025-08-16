# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import (
    with_temp_dir,
)

CHECKPOINT_DIR = "checkpoint"

class TestListStateDict(DTensorTestBase):

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_list_stored_sd_metadata(self) -> None:
        """
        Verify if the listing works for saved disributed tensor.
        """
        global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
        mesh_shape = (2, self.world_size // 2)
        mesh_2d = init_device_mesh(self.device_type, mesh_shape)
        dtensor = distribute_tensor(
            global_tensor,
            mesh_2d,
            placements=[Shard(0), Replicate()],
        )
        state_dict_to_save = {"dtensor": dtensor}

        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
            planner=dist_cp.DefaultSavePlanner(),
        )
        md = dist_cp.list_stored_sd_metadata(checkpoint_id=CHECKPOINT_DIR)
        self.assertTrue("dtensor" in md)
        self.assertEqual(md["dtensor"].size, torch.Size([4, 4]))
        self.assertEqual(md["dtensor"].properties.dtype, torch.float32)
