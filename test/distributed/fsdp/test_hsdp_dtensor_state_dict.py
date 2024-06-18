# Owner(s): ["oncall: distributed"]

import io
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor

from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
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


# Simple and boring model to test interface and some corner cases that do not
# require complicated wrapping strategy.
class DenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(4, 8, device="cuda")


# TODO: Consolidate DeviceMesh based FSDP and HSDP test cases.
class TestHSDPWithDeviceMeshAndDTensor(DTensorTestBase):
    def _create_model(self, device_mesh=None):
        if device_mesh:
            model = FSDP(
                DenseModel().cuda(),
                device_mesh=device_mesh,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )
        else:
            mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
            intra_node_pg = mesh_2d.get_group(mesh_dim=1)
            inter_node_pg = mesh_2d.get_group(mesh_dim=0)
            model = FSDP(
                DenseModel().cuda(),
                process_group=(intra_node_pg, inter_node_pg),
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )

        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        model(model.get_input()).sum().backward()
        optim.step()

        return model, optim

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_hsdp_init_with_device_mesh(self):
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        model, optim = self._create_model(mesh_2d)

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        state_dict = model.state_dict()
        optim_state_dict = FSDP.optim_state_dict(model, optim)

        for v in state_dict.values():
            self.assertEqual(type(v), DTensor)
            self.assertEqual(len(v.placements), 2)
            self.assertEqual(v.placements, (Replicate(), Shard(0)))
            self.assertEqual(v.device_mesh, mesh_2d)

        for state in optim_state_dict["state"].values():
            for k, v in state.items():
                if k != "step":
                    self.assertEqual(type(v), DTensor)
                    self.assertEqual(len(v.placements), 2)
                    self.assertEqual(v.placements, (Replicate(), Shard(0)))
                    self.assertEqual(v.device_mesh, mesh_2d)

        state_dict_type = model.get_state_dict_type(model)
        # If device_mesh is used when initializing FSDP, the field _use_dtensor will
        # automatically be set to True.
        self.assertEqual(state_dict_type.state_dict_config._use_dtensor, True)
        self.assertEqual(state_dict_type.optim_state_dict_config._use_dtensor, True)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("offload_to_cpu", [True, False])
    def test_dtensor_sharded_tensor_state_dict_identical(self, offload_to_cpu):
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        model, optim = self._create_model(mesh_2d)

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=offload_to_cpu),
            optim_state_dict_config=ShardedOptimStateDictConfig(
                offload_to_cpu=offload_to_cpu
            ),
        )
        dtensor_sd = model.state_dict()
        dtensor_osd = FSDP.optim_state_dict(model, optim)

        ref_model, ref_optim = self._create_model()
        FSDP.set_state_dict_type(
            ref_model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=offload_to_cpu),
            optim_state_dict_config=ShardedOptimStateDictConfig(
                offload_to_cpu=offload_to_cpu
            ),
        )
        sharded_tensor_sd = ref_model.state_dict()
        sharded_tensor_osd = FSDP.optim_state_dict(ref_model, ref_optim)

        # Check dtensor and sharded_tensor model state dict values are identical
        for dtensor_sd_item, sharded_tensor_sd_item in zip(
            dtensor_sd.items(), sharded_tensor_sd.items()
        ):
            k1, v1 = dtensor_sd_item
            k2, v2 = sharded_tensor_sd_item
            self.assertEqual(k1, k2)

            self.assertEqual(type(v1), DTensor)
            self.assertEqual(type(v2), ShardedTensor)
            # check whether local_tensor are the same
            self.assertEqual(v1.to_local(), v2.local_tensor())
            # check whether device are the same
            self.assertEqual(v1.to_local().device, v2.local_tensor().device)

        # Check dtensor and sharde_tensor optim state dict values are identical
        for dtensor_osd_state, sharded_tensor_osd_state in zip(
            dtensor_osd["state"].items(), sharded_tensor_osd["state"].items()
        ):
            # check FQN are the same
            self.assertEqual(dtensor_osd_state[0], sharded_tensor_osd_state[0])
            for dtensor_hyper_param, sharded_tensor_hyper_param in zip(
                dtensor_osd_state[1].items(),
                sharded_tensor_osd_state[1].items(),
            ):
                k1, v1 = dtensor_hyper_param
                k2, v2 = sharded_tensor_hyper_param
                self.assertEqual(k1, k2)

                if k1 != "step":
                    self.assertEqual(type(v1), DTensor)
                    self.assertEqual(type(v2), ShardedTensor)
                    # check whether local_tensor are the same
                    self.assertEqual(v1.to_local(), v2.local_tensor())
                    # check whether device are the same
                    self.assertEqual(v1.to_local().device, v2.local_tensor().device)
                else:
                    self.assertEqual(v1, v2)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("offload_to_cpu", [True, False])
    def test_dtensor_sharded_optim_load_state_dict(self, offload_to_cpu):
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        model, optim = self._create_model(mesh_2d)

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            optim_state_dict_config=ShardedOptimStateDictConfig(
                offload_to_cpu=offload_to_cpu
            ),
        )

        checkpoint = io.BytesIO()
        torch.save(FSDP.optim_state_dict(model, optim), checkpoint)
        # Deepcopy to save current optim_state_dict to compare with the optim_state_dict loaded back below.
        ref_optim_state_dict = deepcopy(FSDP.optim_state_dict(model, optim))

        # Update the parameters so FSDP.optim_state_dict() will be different from ref_optim_state_dict.
        model(model.get_input()).sum().backward()
        optim.step()

        # Load ref_optim_state_dict back.
        checkpoint.seek(0)
        load_ref_optim_state_dict = torch.load(checkpoint)
        optim.load_state_dict(
            FSDP.optim_state_dict_to_load(model, optim, load_ref_optim_state_dict)
        )
        new_optim_state_dict = FSDP.optim_state_dict(model, optim)

        # Check whether new_optim_state_dict is the same as ref_optim_state_dict.
        for new_optim_state_dict_item, ref_optim_state_dict_item in zip(
            new_optim_state_dict["state"].items(),
            ref_optim_state_dict["state"].items(),
        ):
            # check FQN are the same
            self.assertEqual(new_optim_state_dict_item[0], ref_optim_state_dict_item[0])
            for new_optim_hyper_param, ref_optim_hyper_param in zip(
                new_optim_state_dict_item[1].items(),
                ref_optim_state_dict_item[1].items(),
            ):
                k1, v1 = new_optim_hyper_param
                k2, v2 = ref_optim_hyper_param
                # check whether keys are the same
                self.assertEqual(k1, k2)
                # check whether DTensor are the same
                self.assertEqual(v1, v2)

                if k1 != "step":
                    self.assertEqual(type(v1), DTensor)
                    self.assertEqual(type(v2), DTensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("offload_to_cpu", [True, False])
    def test_dtensor_sharded_model_load_state_dict(self, offload_to_cpu):
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        model, optim = self._create_model(mesh_2d)

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=offload_to_cpu),
        )

        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        # Deepcopy to save current state_dict to compare with the state_dict loaded back below.
        ref_state_dict = deepcopy(model.state_dict())

        # Update the parameters so model.state_dict() will be different from ref_dtensor_sd.
        model(model.get_input()).sum().backward()
        optim.step()

        # Load ref_state_dict back.
        checkpoint.seek(0)
        load_ref_state_dict = torch.load(checkpoint)
        model.load_state_dict(load_ref_state_dict)
        new_state_dict = model.state_dict()

        # Check whether new_state_dict is the same as ref_state_dict.
        for (k1, v1), (k2, v2) in zip(ref_state_dict.items(), new_state_dict.items()):
            # check whether fqn are the same
            self.assertEqual(k1, k2)

            self.assertEqual(type(v1), DTensor)
            self.assertEqual(type(v2), DTensor)
            # check whether DTensor are the same
            self.assertEqual(v1, v2)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_root_module_is_not_FSDP(self):
        class FakeMPModel(torch.nn.Module):
            def __init__(self, device_mesh):
                super().__init__()
                torch.manual_seed(0)
                self.dense = FSDP(
                    DenseModel().cuda(),
                    use_orig_params=True,
                    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    device_mesh=device_mesh,
                )
                if dist.get_rank() == 0:
                    self.sparse0 = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
                else:
                    self.sparse1 = nn.Sequential(nn.Linear(8, 8), nn.ReLU())

            def forward(self, x):
                if dist.get_rank() == 0:
                    sparse = self.sparse0(x)
                else:
                    sparse = self.sparse1(x)
                dist.all_reduce(sparse)
                return self.dense(sparse)

        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        model = FakeMPModel(device_mesh=mesh_2d).cuda()
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        batch = torch.rand(5, 8, device=torch.device("cuda"))
        model(batch).sum().backward()
        optim.step()
        osd = optim.state_dict()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            osd = FSDP.optim_state_dict(model, optim, osd)

        for param, state in osd["state"].items():
            if "dense" in param:
                self.assertIsInstance(state["exp_avg"], DTensor)
                self.assertIsInstance(state["exp_avg_sq"], DTensor)
                self.assertEqual(state["exp_avg"].placements, (Replicate(), Shard(0)))
                self.assertEqual(
                    state["exp_avg_sq"].placements, (Replicate(), Shard(0))
                )
            else:
                self.assertIsInstance(state["exp_avg"], torch.Tensor)
                self.assertIsInstance(state["exp_avg_sq"], torch.Tensor)


instantiate_parametrized_tests(TestHSDPWithDeviceMeshAndDTensor)
if __name__ == "__main__":
    run_tests()
