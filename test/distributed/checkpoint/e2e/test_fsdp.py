# Owner(s): ["oncall: distributed"]
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.nn as nn
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

# Simple and boring model to test interface and some corner cases that do not
# require complicated wrapping strategy.
class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(8, 8, device="cuda")


class TestFSDP(DTensorTestBase):
    def _create_model(self, device_mesh=None):
        dummy_model = TestDummyModel()

        model = FSDP(dummy_model.cuda(), device_mesh=device_mesh, use_orig_params=True)
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        model(model.get_input()).sum().backward()
        optim.step()

        return model, optim

    def _equal_state_dict(self, model_0, model_1):
        for params_0, params_1 in zip(model_0.values(), model_1.values()):
            if not torch.equal(params_0, params_1):
                return False
        return True

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_e2e(self):
        checkpoint_dir = self.temp_dir

        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model, optim = self._create_model(device_mesh)

        model_state_dict_0, optim_state_dict = get_state_dict(model, optimizers=optim)
        DCP.save_state_dict(
            state_dict={"model": model_state_dict_0, "optimizer": optim_state_dict},
            storage_writer=DCP.FileSystemWriter(checkpoint_dir),
        )

        # load the checkpoint, starting with a new model
        model, optim = self._create_model(device_mesh)

        model_state_dict_1, optim_state_dict = get_state_dict(model, optimizers=optim)

        # sanity check, since we have not done any loading, the models should be different
        assert not self._equal_state_dict(model_state_dict_0, model_state_dict_1)

        DCP.load_state_dict(
            {"model": model_state_dict_1, "optimizer": optim_state_dict},
            storage_reader=DCP.FileSystemReader(checkpoint_dir),
        )
        set_state_dict(
            model,
            optimizers=optim,
            model_state_dict=model_state_dict_1,
            optim_state_dict=optim_state_dict,
        )

        # model state dict should be the same after loading
        assert self._equal_state_dict(model_state_dict_0, model_state_dict_1)


if __name__ == "__main__":
    run_tests()
