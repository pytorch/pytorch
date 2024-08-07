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

_optim_cls = torch.optim.Adam
_optim_lr = 1e-2


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

        o1_sd, o2_sd = optim1.state_dict(), optim2.state_dict()
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

        reload_o1 = _optim_cls(model1.parameters(), lr=_optim_lr)
        reload_o2 = _optim_cls(model2.parameters(), lr=_optim_lr)
        fsdp_o1_load = FSDP.optim_state_dict_to_load(
            model1, optim1, optim_state_dict1, is_named_optimizer=False
        )
        reload_o1.load_state_dict(fsdp_o1_load)
        fsdp_o2_load = FSDP.optim_state_dict_to_load(
            model2, optim2, optim_state_dict2, is_named_optimizer=False
        )
        reload_o2.load_state_dict(fsdp_o2_load)
        reload_o1_sd, reload_o2_sd = reload_o1.state_dict(), reload_o2.state_dict()
        for sd_pair in [(o1_sd, reload_o1_sd), (o2_sd, reload_o2_sd)]:
            sd1, sd2 = sd_pair
            for (k1, v1), (k2, v2) in zip(sd1.items(), sd2.items()):
                self.assertEqual(k1, k2, f"Mismatched keys: {k1} vs {k2}")
                self.assertEqual(v1, v2, f"Mismatched values {v1} vs {v2}")

    @skip_if_lt_x_gpu(2)
    def test_optim_state_dict_save_load(self):
        orig_model = CompositeParamModel(device=torch.device("cuda"))
        composable_model = copy.deepcopy(orig_model)
        fully_shard(composable_model, policy=ModuleWrapPolicy({UnitModule}))
        composable_optim = _optim_cls(composable_model.parameters(), lr=_optim_lr)
        orig_model = FSDP(orig_model)
        orig_optim = _optim_cls(orig_model.parameters(), lr=_optim_lr)

        self._test_optim_state_save_load(
            orig_model, orig_optim, composable_model, composable_optim
        )

    @skip_if_lt_x_gpu(2)
    def test_optim_state_dict_submodule_fully_shard(self):
        orig_model = CompositeParamModel(device=torch.device("cuda"))
        composable_model = copy.deepcopy(orig_model)
        fully_shard(composable_model.u1)
        fully_shard(composable_model.u2)
        composable_optim = _optim_cls(composable_model.parameters(), lr=_optim_lr)
        orig_model = FSDP(orig_model)
        orig_optim = _optim_cls(orig_model.parameters(), lr=_optim_lr)

        self._test_optim_state_save_load(
            orig_model, orig_optim, composable_model, composable_optim
        )


if __name__ == "__main__":
    run_tests()
