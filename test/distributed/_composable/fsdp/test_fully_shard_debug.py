# Owner(s): ["oncall: distributed"]

import unittest

import torch.nn as nn
from _test_fully_shard_common import MLP

from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_fsdp import FSDPTestMultiThread
from torch.testing._internal.common_utils import run_tests


class TestFullyShardDebug(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fully_shard_module_and_param_fqns(self):
        """
        Tests that the module and parameter FQNs are computed correctly after
        lazy initialization.

        FSDP(model(
            MLP(FSDP(in_proj), FSDP(out_proj)),
            MLP(in_proj, out_proj),
        ))
        """
        model = nn.Sequential(MLP(8), MLP(8))
        fully_shard(model[0].in_proj)
        fully_shard(model[0].out_proj)
        fully_shard(model)  # root gets `model[1]`
        root_state = fully_shard.state(model)
        root_state._lazy_init()

        root_param_group = root_state._fsdp_param_group
        self.assertIsNotNone(root_param_group)
        self.assertEqual(root_param_group._module_fqn, "")
        root_param_fqns = {
            fsdp_param._param_fqn for fsdp_param in root_param_group.fsdp_params
        }
        self.assertEqual(
            root_param_fqns,
            {
                "1.in_proj.weight",
                "1.in_proj.bias",
                "1.out_proj.weight",
                "1.out_proj.bias",
            },
        )

        model0_in_proj_state = fully_shard.state(model[0].in_proj)
        model0_in_proj_param_group = model0_in_proj_state._fsdp_param_group
        self.assertIsNotNone(model0_in_proj_param_group)
        self.assertEqual(model0_in_proj_param_group._module_fqn, "0.in_proj")
        model0_in_proj_param_fqns = {
            fsdp_param._param_fqn
            for fsdp_param in model0_in_proj_param_group.fsdp_params
        }
        self.assertEqual(
            model0_in_proj_param_fqns, {"0.in_proj.weight", "0.in_proj.bias"}
        )

        model0_out_proj_state = fully_shard.state(model[0].out_proj)
        model0_out_proj_param_group = model0_out_proj_state._fsdp_param_group
        self.assertIsNotNone(model0_out_proj_param_group)
        self.assertEqual(model0_out_proj_param_group._module_fqn, "0.out_proj")
        model0_out_proj_param_fqns = {
            fsdp_param._param_fqn
            for fsdp_param in model0_out_proj_param_group.fsdp_params
        }
        self.assertEqual(
            model0_out_proj_param_fqns, {"0.out_proj.weight", "0.out_proj.bias"}
        )


if __name__ == "__main__":
    run_tests()
