# Owner(s): ["oncall: distributed"]

import io
from copy import deepcopy

import torch
import torch.distributed.checkpoint as DCP
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor

from torch.distributed._tensor import DTensor, init_device_mesh, Replicate, Shard

from torch.distributed._tensor.device_mesh import _mesh_resources, init_device_mesh
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)

from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

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
        return torch.rand(4, 8, device="cuda")


class TestTpCheckpoint(DTensorTestBase):
    def _create_model(self, device_mesh=None):
        if device_mesh:
            model = FSDP(
                TestDummyModel().cuda(),
                device_mesh=device_mesh,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )
        else:
            mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
            intra_node_pg = mesh_2d.get_dim_groups(mesh_dim=1)
            inter_node_pg = mesh_2d.get_dim_groups(mesh_dim=0)
            model = FSDP(
                TestDummyModel().cuda(),
                process_group=(intra_node_pg, inter_node_pg),
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )

        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        model(model.get_input()).sum().backward()
        optim.step()

        return model, optim

    # Case 1: this test would fail because torch.save() and torch.load() does not understand
    # DTensor and save the DTensor as it is.
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_try_load_without_DCP(self):
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        model, optim = self._create_model(mesh_2d)

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            checkpoint = io.BytesIO()
            torch.save(model.state_dict(), checkpoint)
            # Deepcopy to save current state_dict to compare with the state_dict loaded back below.
            ref_state_dict = deepcopy(model.state_dict())

        # Update the parameters so model.state_dict() will be different from ref_dtensor_sd.
        model(model.get_input()).sum().backward()
        optim.step()

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            # Load ref_state_dict back.
            checkpoint.seek(0)
            load_ref_state_dict = torch.load(checkpoint)
            """
            We will see the errors:
            RuntimeError: Error(s) in loading state_dict for FullyShardedDataParallel:
            While copying the parameter named "_fsdp_wrapped_module.net1.0.weight", whose dimensions in the model are torch.Size([16, 8]) and whose dimensions in the checkpoint are torch.Size([16, 8]), an exception occurred : ('aten.copy_.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!',).
            While copying the parameter named "_fsdp_wrapped_module.net1.0.bias", whose dimensions in the model are torch.Size([16]) and whose dimensions in the checkpoint are torch.Size([16]), an exception occurred : ('aten.copy_.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!',).
            While copying the parameter named "_fsdp_wrapped_module.net2.0.weight", whose dimensions in the model are torch.Size([32, 16]) and whose dimensions in the checkpoint are torch.Size([32, 16]), an exception occurred : ('aten.copy_.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!',).
            While copying the parameter named "_fsdp_wrapped_module.net2.0.bias", whose dimensions in the model are torch.Size([32]) and whose dimensions in the checkpoint are torch.Size([32]), an exception occurred : ('aten.copy_.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!',).
            While copying the parameter named "_fsdp_wrapped_module.net3.weight", whose dimensions in the model are torch.Size([64, 32]) and whose dimensions in the checkpoint are torch.Size([64, 32]), an exception occurred : ('aten.copy_.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!',).
            While copying the parameter named "_fsdp_wrapped_module.net3.bias", whose dimensions in the model are torch.Size([64]) and whose dimensions in the checkpoint are torch.Size([64]), an exception occurred : ('aten.copy_.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!',).
            While copying the parameter named "_fsdp_wrapped_module.net4.1.weight", whose dimensions in the model are torch.Size([8, 64]) and whose dimensions in the checkpoint are torch.Size([8, 64]), an exception occurred : ('aten.copy_.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!',).
            While copying the parameter named "_fsdp_wrapped_module.net4.1.bias", whose dimensions in the model are torch.Size([8]) and whose dimensions in the checkpoint are torch.Size([8]), an exception occurred : ('aten.copy_.default: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!',).
            """
            model.load_state_dict(load_ref_state_dict)

            # new_state_dict = model.state_dict()
            # print(f"{self.rank=}, {new_state_dict=}")

    # Case 2: this test would succeed since DCP does not save DTensor as it is. Instead, it
    # saves DTensor as local tensor with a bunch of metadata. When loading back in full_state_dict,
    # DCP would paste the shards together to form tensor.
    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_try_load_with_DCP(self):
        CHECKPIONT_DIR = self.temp_dir
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        model, optim = self._create_model(mesh_2d)

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            # Deepcopy to save current state_dict to compare with the state_dict loaded back below.
            ref_state_dict = deepcopy(model.state_dict())
            DCP.save_state_dict(
                state_dict=ref_state_dict,
                storage_writer=DCP.FileSystemWriter(CHECKPIONT_DIR),
            )

        # Update the parameters so model.state_dict() will be different from ref_dtensor_sd.
        model(model.get_input()).sum().backward()
        optim.step()

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict_to_load = model.state_dict()
            DCP.load_state_dict(
                state_dict=state_dict_to_load,
                storage_reader=DCP.FileSystemReader(CHECKPIONT_DIR),
            )
            model.load_state_dict(state_dict_to_load)

            new_state_dict = model.state_dict()
            # print(f"{self.rank=}, {new_state_dict=}")


if __name__ == "__main__":
    run_tests()
