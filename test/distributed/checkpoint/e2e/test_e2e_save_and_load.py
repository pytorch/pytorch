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


# Simple and boring model to test interface and some corner cases that do not
# require complicated wrapping strategy.
class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
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


class TestE2ELoadAndSave(DTensorTestBase):
    def _create_model(self, compile, model_type):
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

        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        if compile:
            model = torch.compile(model)

        model(model.get_input()).sum().backward()
        optim.step()

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
        # first create and save a checkpoint
        model, optim = self._create_model(compile, model_type)
        model_state_dict_0, optim_state_dict_0 = get_state_dict(model, optimizers=optim)

        DCP.save_state_dict(
            state_dict={"model": model_state_dict_0, "optimizer": optim_state_dict_0},
            storage_writer=DCP.FileSystemWriter(self.temp_dir),
        )

        # load the checkpoint, starting with a new model
        model, optim = self._create_model(compile, model_type)
        model_state_dict_1, optim_state_dict_1 = get_state_dict(model, optimizers=optim)

        # sanity check, since we have not done any loading, state dicts should differ
        self.assertFalse(self._equal_state_dict(model_state_dict_0, model_state_dict_1))

        DCP.load_state_dict(
            {"model": model_state_dict_1, "optimizer": optim_state_dict_1},
            storage_reader=DCP.FileSystemReader(self.temp_dir),
        )
        set_state_dict(
            model,
            optimizers=optim,
            model_state_dict=model_state_dict_1,
            optim_state_dict=optim_state_dict_1,
        )

        # state dict should be the same following loading
        self.assertTrue(self._equal_state_dict(model_state_dict_0, model_state_dict_1))
        self.assertEqual(optim_state_dict_0, optim_state_dict_1)


instantiate_parametrized_tests(TestE2ELoadAndSave)
if __name__ == "__main__":
    run_tests()
