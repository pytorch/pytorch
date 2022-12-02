# Owner(s): ["oncall: distributed"]

import copy
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import checkpoint, fully_shard
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class UnitModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.l2 = nn.Linear(100, 100)

    def forward(self, x):
        return self.l2(self.seq(self.l1(x)))


class CompositeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.u1 = UnitModule()
        self.u2 = UnitModule()
        self.l2 = nn.Linear(100, 100)

    def forward(self, x):
        return self.l2(self.u2(self.u1(self.l1(x))))


class TestFSDPCheckpoint(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def _test_wrap_same_submodule(
        self,
        base_model: nn.Module,
        test_model: nn.Module,
        x: torch.Tensor,
        grad_to_none: bool,
    ):
        LR = 0.01
        base_optim = torch.optim.Adam(base_model.parameters(), lr=LR)
        test_optim = torch.optim.Adam(test_model.parameters(), lr=LR)

        for _ in range(5):
            test_loss = test_model(x).sum()
            base_loss = base_model(x).sum()

            self.assertEqual(test_loss, base_loss)

            test_loss.backward()
            test_optim.step()
            test_optim.zero_grad(set_to_none=grad_to_none)

            base_loss.backward()
            base_optim.step()
            base_optim.zero_grad(set_to_none=grad_to_none)

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_wrap_same_submodule(self, use_reentrant: bool):
        model = UnitModule().to("cuda")

        base_model = copy.deepcopy(model)

        test_model = copy.deepcopy(model)
        # compose checkpoint and fully_shard
        test_model.seq = checkpoint(test_model.seq, use_reentrant=use_reentrant)
        test_model.seq = fully_shard(
            test_model.seq,
            policy=ModuleWrapPolicy({nn.Linear}),
        )

        self.run_subtests(
            {
                "base_model": [base_model],
                "test_model": [test_model],
                "x": [torch.randn(2, 100, device="cuda")],
                "grad_to_none": [True, False],
            },
            self._test_wrap_same_submodule,
        )

    def _test_checkpoint_fsdp_submodules(self, use_reentrant):
        model = CompositeModel().to(torch.device("cuda"))

        base_model = copy.deepcopy(model)

        test_model = copy.deepcopy(model)
        test_model.u1 = fully_shard(
            test_model.u1,
            policy=ModuleWrapPolicy({UnitModule}),
        )
        test_model.u2 = fully_shard(
            test_model.u2,
            policy=ModuleWrapPolicy({UnitModule}),
        )

        test_model.u1.seq = checkpoint(
            test_model.u1.seq, use_reentrant=use_reentrant
        )
        test_model.u2.seq = checkpoint(
            test_model.u2.seq, use_reentrant=use_reentrant
        )

        self.run_subtests(
            {
                "base_model": [base_model],
                "test_model": [test_model],
                "x": [torch.randn(2, 100, device="cuda")],
                "grad_to_none": [True, False],
            },
            self._test_wrap_same_submodule,
        )

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_use_reentrant(self):
        with self.assertRaisesRegex(
            AssertionError,
            "Expects `Tensor` to have been saved in forward",
        ):
            self._test_checkpoint_fsdp_submodules(True)

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_non_reentrant(self):
        self._test_checkpoint_fsdp_submodules(False)


instantiate_parametrized_tests(TestFSDPCheckpoint)


if __name__ == "__main__":
    run_tests()
