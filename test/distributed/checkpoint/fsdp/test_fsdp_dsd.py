# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    StateDictOptions,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.wrap import always_wrap_policy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, MLP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.utils._pytree import tree_all_only


class TestFullyShardWithDistributedStateDict(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    def _get_base_model(self, mlp_dim: int = 2):
        base_model = nn.Sequential(
            MLP(mlp_dim),
            nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),
            MLP(mlp_dim),
        )
        return base_model

    @skip_if_lt_x_gpu(2)
    def test_1d_fsdp_get_model_state_dict(self):
        self.run_subtests(
            {"mlp_dim": [2, 3, 4, 5]},
            self._test_1d_fsdp_get_model_state_dict,
        )

    def _test_1d_fsdp_get_model_state_dict(self, mlp_dim: int):
        """
        Test model.state_dict() and distributed_state_dict parity.
        """
        base_model = self._get_base_model(mlp_dim)
        # Default is `reshard_after_forward=True`
        model1 = copy.deepcopy(base_model)
        for module in model1:
            fully_shard(module)
        fully_shard(model1)

        # osd: original state dict, dsd: distributed state dict
        osd = model1.state_dict()
        dsd = get_model_state_dict(model1)
        self.assertEqual(osd, dsd)

        # Check `reshard_after_forward=False` after a forward
        model2 = copy.deepcopy(base_model)
        for module in model2:
            fully_shard(module, reshard_after_forward=False)
        fully_shard(model2, reshard_after_forward=False)
        inp = torch.randn((2, mlp_dim), device="cuda")
        model2(inp)  # parameters are not resharded after this forward
        # Check that state dict hooks reshard
        osd_2 = model2.state_dict()
        dsd_2 = get_model_state_dict(model2)
        self.assertEqual(osd_2, dsd_2)

    @skip_if_lt_x_gpu(2)
    def test_1d_fsdp_cpu_offload_full_model_state_dict(self):
        """
        Test full_state_dict and cpu_offload works for FSDP2 state_dict.
        """
        orig_model = self._get_base_model()
        fsdp_model = copy.deepcopy(orig_model)
        for module in fsdp_model:
            fully_shard(module)
        fully_shard(fsdp_model)

        osd = orig_model.state_dict()
        dsd = get_model_state_dict(
            fsdp_model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )

        cpu_device = torch.device("cpu")

        def is_cpu(v):
            if isinstance(v, DTensor):
                return v.device == torch.device("cpu")
            else:
                return v.device == cpu_device

        if self.rank == 0:
            self.assertEqual(osd, dsd)
            self.assertTrue(tree_all_only((torch.Tensor, DTensor), is_cpu, osd))
        else:
            self.assertEqual(dsd, {})

    @skip_if_lt_x_gpu(2)
    def test_save_with_fsdp1_and_load_with_fsdp2(self):
        self.run_subtests(
            {
                "state_dict_type": [
                    StateDictType.FULL_STATE_DICT,
                    StateDictType.SHARDED_STATE_DICT,
                ]
            },
            self._test_save_with_fsdp1_and_load_with_fsdp2,
        )

    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def _test_save_with_fsdp1_and_load_with_fsdp2(self, state_dict_type: StateDictType):
        """
        Test that we can save a model with FSDP1 and load it with FSDP2.
        """

        # Save state dict with model wrapped with FSDP1
        fsdp1_model = FSDP(
            self._get_base_model().cuda(),
            use_orig_params=True,
            auto_wrap_policy=always_wrap_policy,
        )

        fsdp1_optim = torch.optim.Adam(fsdp1_model.parameters(), lr=0.1)

        fsdp1_model(torch.randn((2,), device=self.rank)).sum().backward()
        fsdp1_optim.step()

        with FSDP.state_dict_type(fsdp1_model, state_dict_type):
            fsdp1_state_dict = {
                "model": fsdp1_model.state_dict(),
                "optim": FSDP.sharded_optim_state_dict(fsdp1_model, fsdp1_optim),
            }
            dcp.save(
                fsdp1_state_dict,
                checkpoint_id=self.temp_dir,
            )

        fsdp1_full_msd = get_model_state_dict(
            fsdp1_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        fsdp1_full_osd = get_optimizer_state_dict(
            fsdp1_model,
            fsdp1_optim,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        # Load state dict into model with FSDP2 applied
        fsdp2_model = self._get_base_model()
        for module in fsdp2_model:
            fully_shard(module)
        fully_shard(fsdp2_model)
        fsdp2_optim = torch.optim.Adam(fsdp2_model.parameters(), lr=0.1)

        fsdp2_state_dict = {
            "model": get_model_state_dict(fsdp2_model),
            "optim": get_optimizer_state_dict(fsdp2_model, fsdp2_optim),
        }
        dcp.load(
            fsdp2_state_dict,
            checkpoint_id=self.temp_dir,
        )
        fsdp2_model.load_state_dict(fsdp2_state_dict["model"])
        fsdp2_optim.load_state_dict(fsdp2_state_dict["optim"])

        fsdp2_full_msd = get_model_state_dict(
            fsdp2_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        fsdp2_full_osd = get_optimizer_state_dict(
            fsdp2_model,
            fsdp2_optim,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        # Compare full state dict to make sure they are the same.
        self.assertEqual(fsdp2_full_msd, fsdp1_full_msd)
        self.assertEqual(fsdp1_full_osd, fsdp2_full_osd)


if __name__ == "__main__":
    run_tests()
