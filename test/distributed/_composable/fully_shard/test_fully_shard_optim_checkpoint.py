# Owner(s): ["oncall: distributed"]

import copy
import itertools
import sys

import torch
import torch.distributed as dist
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestOptimStateCheckpointing(FSDPTest):
    """Tests ``fully_shard`` optimizer state checkpointing."""

    @property
    def world_size(self) -> int:
        return 2

    def _test_optim_state_save_load(self, model1, optim1, model2, optim2) -> None:
        batch = torch.randn(2, 100, device="cuda")
        for model, optim in (
            (model1, optim1),
            (model2, optim2),
        ):
            optim.zero_grad(set_to_none=True)
            model(batch).sum().backward()
            optim.step()

        optim_state_dict1 = FSDP.optim_state_dict(model1, optim1)
        optim_state_dict2 = FSDP.optim_state_dict(model2, optim2)

        self.assertEqual(
            len(optim_state_dict1["state"]), len(optim_state_dict2["state"])
        )
        for fqn, state in optim_state_dict1["state"].items():
            self.assertEqual(state, optim_state_dict2["state"][fqn], fqn)

        for group1, group2 in itertools.zip_longest(
            optim_state_dict1["param_groups"], optim_state_dict2["param_groups"]
        ):
            for key, value in group1.items():
                self.assertEqual(value, group2[key])

    @skip_if_lt_x_gpu(2)
    def test_optim_state_dict_save_load(self):
        orig_model = CompositeParamModel(device=torch.device("cuda"))
        composable_model = copy.deepcopy(orig_model)
        fully_shard(composable_model, policy=ModuleWrapPolicy({UnitModule}))
        composable_optim = torch.optim.Adam(composable_model.parameters(), lr=1e-2)
        orig_model = FSDP(orig_model)
        orig_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-2)

        self._test_optim_state_save_load(
            orig_model, orig_optim, composable_model, composable_optim
        )

    @skip_if_lt_x_gpu(2)
    def test_optim_state_dict_submodule_fully_shard(self):
        orig_model = CompositeParamModel(device=torch.device("cuda"))
        composable_model = copy.deepcopy(orig_model)
        fully_shard(composable_model.u1)
        fully_shard(composable_model.u2)
        composable_optim = torch.optim.Adam(composable_model.parameters(), lr=1e-2)
        orig_model = FSDP(orig_model)
        orig_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-2)

        self._test_optim_state_save_load(
            orig_model, orig_optim, composable_model, composable_optim
        )


if __name__ == "__main__":
    run_tests()
