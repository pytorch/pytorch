# Owner(s): ["oncall: distributed"]
import os
from unittest.mock import patch

import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class MyTestModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))


class TestSaveAndLoadAPI(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_auto_detect(self):
        model = FSDP(MyTestModule().cuda())
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = FSDP(model, device_mesh=device_mesh)
        dcp.save(model.state_dict(), checkpoint_id=os.path.join(self.temp_dir, "first"))
        dcp.load(model.state_dict(), checkpoint_id=os.path.join(self.temp_dir, "first"))

        with patch.object(
            dcp.FileSystemReader, "validate_checkpoint_id", return_value=False
        ):
            with patch.object(
                dcp.FileSystemWriter, "validate_checkpoint_id", return_value=False
            ):
                dcp.save(
                    model.state_dict(),
                    checkpoint_id=os.path.join(self.temp_dir, "second"),
                )
                dcp.load(
                    model.state_dict(),
                    checkpoint_id=os.path.join(self.temp_dir, "second"),
                )

        with self.assertRaisesRegex(RuntimeError, "Cannot detect"):
            dcp.save(model.state_dict(), checkpoint_id="abc://abc.abc")
        with self.assertRaisesRegex(RuntimeError, "Cannot detect"):
            dcp.load(model.state_dict(), checkpoint_id="abc://abc.abc")


if __name__ == "__main__":
    run_tests()
