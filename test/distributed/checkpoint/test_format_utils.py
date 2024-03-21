# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn

import torch.nn.functional as F
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.format_utils import (
    BroadcastingTorchSaveReader,
    dcp_to_torch_save,
    DynamicMetaLoadPlanner,
    torch_save_to_dcp,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class SimpleModelUneven(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = self.net4(x)
        return x

    def get_input(self):
        return torch.rand(4, 5, device="cuda")


class TestFormatUtils(DTensorTestBase):
    @with_temp_dir
    def test_dcp_to_torch_save(self) -> None:
        model = SimpleModelUneven()
        dcp.save({"model": model}, checkpoint_id=self.temp_dir)

        torch_path = self.temp_dir + "/model.pt"
        dcp_to_torch_save(self.temp_dir, torch_path)

        loaded_sd = torch.load(torch_path)
        self.assertEqual(loaded_sd, {"model": model.state_dict()})

    @with_temp_dir
    def test_torch_save_to_dcp(self) -> None:
        model = SimpleModelUneven()
        sd = {"model": model.state_dict()}
        torch_path = self.temp_dir + "/model.pt"
        torch.save(sd, torch_path)

        torch_save_to_dcp(torch_path, self.temp_dir)

        model = SimpleModelUneven()
        dcp.load({"model": model}, checkpoint_id=self.temp_dir)

        self.assertEqual({"model": model.state_dict()}, sd)

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_online_torch_save_to_dcp(self) -> None:
        """Tests loading a model saved by torch.save directly into a sharded model
        using dcp.load
        """
        # Save a model with torch.save
        model = SimpleModelUneven()
        sd = {"model": model.state_dict()}

        torch_fn = self.temp_dir + "/model.pt"
        if dist.get_rank() == 0:
            torch.save(sd, torch_fn)
        dist.barrier()

        # Load into a sharded model
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = SimpleModelUneven().cuda()
        model = FSDP(
            model,
            device_mesh=device_mesh,
            use_orig_params=True,
        )
        dcp.load(
            {"model": model},
            planner=DynamicMetaLoadPlanner(),
            storage_reader=BroadcastingTorchSaveReader(),
            checkpoint_id=torch_fn,
        )

        self.assertEqual(sd["model"], model.state_dict())


if __name__ == "__main__":
    run_tests()
