# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.format_utils import (
    BroadcastingTorchSaveReader,
    dcp_to_torch_save,
    DynamicMetaLoadPlanner,
    torch_save_to_dcp,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    ModelArgs,
    skip_if_lt_x_gpu,
    Transformer,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class TestFormatUtils(DTensorTestBase):
    @with_temp_dir
    def test_dcp_to_torch_save(self) -> None:
        # Using a transformer model to simulate a 'complicated enough' state dict w/ nested modules
        model = Transformer(ModelArgs())
        dcp.save({"model": model}, checkpoint_id=self.temp_dir)

        torch_path = self.temp_dir + "/model.pt"
        dcp_to_torch_save(self.temp_dir, torch_path)

        loaded_sd = torch.load(torch_path)
        self.assertEqual(loaded_sd, {"model": model.state_dict()})

    @with_temp_dir
    def test_torch_save_to_dcp(self) -> None:
        model = Transformer(ModelArgs())
        sd = {"model": model.state_dict()}
        torch_path = self.temp_dir + "/model.pt"
        torch.save(sd, torch_path)

        torch_save_to_dcp(torch_path, self.temp_dir)

        model = Transformer(ModelArgs())
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
        model = Transformer(ModelArgs())
        sd = {"model": model.state_dict()}

        torch_fn = self.temp_dir + "/model.pt"
        torch.save(sd, torch_fn)

        # Load into a sharded model
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = Transformer(ModelArgs()).cuda()
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
