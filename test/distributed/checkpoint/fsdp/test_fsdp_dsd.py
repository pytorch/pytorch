# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, MLP
from torch.testing._internal.common_utils import run_tests
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


if __name__ == "__main__":
    run_tests()
