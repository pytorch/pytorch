# Owner(s): ["oncall: distributed"]

from enum import auto, Enum

import torch
import torch.distributed.checkpoint as DCP
import torch.nn as nn
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
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
from torch.testing._internal.common_state_dict import VerifyStateDictMixin

# Simple and boring model to test interface and some corner cases that do not
# require complicated wrapping strategy.
class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(8, 8, device="cuda")


class ModelType(Enum):
    FSDP = auto()
    HSDP = auto()
    FSDP_TP = auto()
    NONE = auto() # no parallelization

def _train(model, optim, train_steps=1):
    torch.manual_seed(0)
    loss = None
    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss

class TestE2ELoadAndSave(DTensorTestBase, VerifyStateDictMixin):
    def _create_model(self, compile, model_type, train_steps=2):
        dummy_model = TestDummyModel().cuda()

        assert model_type in ModelType, f"{model_type} is not supported."
        if model_type == ModelType.FSDP:
            device_mesh = init_device_mesh(self.device_type, (self.world_size,))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
            )
        elif model_type == ModelType.HSDP:
            device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )
        elif model_type == ModelType.FSDP_TP:
            mesh_2d = init_device_mesh(
                self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
            )
            tp_mesh = mesh_2d["tp"]
            dp_mesh = mesh_2d["dp"]
            model = parallelize_module(dummy_model, tp_mesh, PairwiseParallel())
            model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
        else:
            model = dummy_model
        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        if compile:
            model = torch.compile(model)

        return model, optim

    def _equal_state_dict(self, model_0, model_1):
        for params_0, params_1 in zip(model_0.values(), model_1.values()):
            if not torch.equal(params_0, params_1):
                return False
        return True

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    @parametrize("compile", [True, False])
    @parametrize("model_type", [ModelType.FSDP, ModelType.HSDP, ModelType.FSDP_TP])
    def test_e2e(self, compile, model_type):
        model, optim = self._create_model(compile, ModelType.NONE)
        _train(model, optim, train_steps=2)

        dist_model, dist_optim = self._create_model(compile, model_type)
        _train(dist_model, dist_optim, train_steps=2)

        # create and save a checkpoint for parallel model
        dist_msd, dist_osd = get_state_dict(dist_model, optimizers=dist_optim)
        DCP.save_state_dict(
            state_dict={"model": dist_msd, "optimizer": dist_osd},
            storage_writer=DCP.FileSystemWriter(self.temp_dir),
        )

        # load the checkpoint, starting with a new model
        dist_model, dist_optim = self._create_model(compile, model_type)
        dist_msd, dist_osd = get_state_dict(dist_model, optimizers=dist_optim)
        DCP.load_state_dict(
            {"model": dist_msd, "optimizer": dist_osd},
            storage_reader=DCP.FileSystemReader(self.temp_dir),
        )
        set_state_dict(
            dist_model,
            optimizers=dist_optim,
            model_state_dict=dist_msd,
            optim_state_dict=dist_osd,
        )

        # train one more step on both models
        loss = _train(model, optim, train_steps=1)
        dist_loss = _train(dist_model, dist_optim, train_steps=1)
        self.assertEqual(loss, dist_loss)

        dist_msd, dist_osd = get_state_dict(dist_model, optimizers=dist_optim)
        model_sd, optim_sd = get_state_dict(model, optimizers=optim)

        self._verify_msd(model_sd, dist_msd)
        self._verify_osd_by_load(
            model,
            optim,
            torch.optim.Adam(model.parameters(), lr=0.1),
            optim_sd
        )

instantiate_parametrized_tests(TestE2ELoadAndSave)
if __name__ == "__main__":
    run_tests()
