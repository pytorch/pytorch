# Owner(s): ["oncall: distributed"]

import copy
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import checkpoint, fully_shard, replicate
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_dist_composable import (
    CompositeModel,
    CompositeParamModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
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


class TestFSDPCheckpoint(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def _test_parity(
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
        model = UnitModule(device=torch.device("cuda"))

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
            self._test_parity,
        )

    def _test_checkpoint_fsdp_submodules(self, use_reentrant):
        model = CompositeModel(device=torch.device("cuda"))

        base_model = copy.deepcopy(model)

        test_model = copy.deepcopy(model)
        test_model.u1 = fully_shard(test_model.u1, policy=None)
        test_model.u2 = fully_shard(test_model.u2)

        test_model.u1.seq = checkpoint(test_model.u1.seq, use_reentrant=use_reentrant)
        test_model.u2.seq = checkpoint(test_model.u2.seq, use_reentrant=use_reentrant)

        self.run_subtests(
            {
                "base_model": [base_model],
                "test_model": [test_model],
                "x": [torch.randn(2, 100, device="cuda")],
                "grad_to_none": [True, False],
            },
            self._test_parity,
        )

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_use_reentrant(self):
        # Escape the brackets like `\[` since `[` has special meaning in regex
        with self.assertRaisesRegex(
            RuntimeError,
            r"setStorage: sizes \[100, 100\], strides \[100, 1\], storage "
            "offset 0, and itemsize 4 requiring a storage size of 40000 are "
            "out of bounds for storage of size 0",
        ):
            self._test_checkpoint_fsdp_submodules(True)

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_non_reentrant(self):
        self._test_checkpoint_fsdp_submodules(False)

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_with_param(self):
        model = CompositeParamModel(device=torch.device("cuda"))

        base_model = copy.deepcopy(model)

        test_model = copy.deepcopy(model)
        test_model.u1.seq = checkpoint(test_model.u1.seq, use_reentrant=False)
        test_model.u2.seq = checkpoint(test_model.u2.seq, use_reentrant=False)
        test_model = fully_shard(test_model)

        self.run_subtests(
            {
                "base_model": [base_model],
                "test_model": [test_model],
                "x": [torch.randn(2, 100, device="cuda")],
                "grad_to_none": [True, False],
            },
            self._test_parity,
        )

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_with_param_no_shard(self):
        model = CompositeParamModel(device=torch.device("cuda"))

        base_model = copy.deepcopy(model)

        test_model = copy.deepcopy(model)
        test_model.u1.seq = checkpoint(test_model.u1.seq, use_reentrant=False)
        test_model.u2.seq = checkpoint(test_model.u2.seq, use_reentrant=False)
        test_model = fully_shard(test_model, strategy=ShardingStrategy.NO_SHARD)

        self.run_subtests(
            {
                "base_model": [base_model],
                "test_model": [test_model],
                "x": [torch.randn(2, 100, device="cuda")],
                "grad_to_none": [True, False],
            },
            self._test_parity,
        )

    @skip_if_lt_x_gpu(2)
    def test_composable_fsdp_replicate(self):
        # Verify how the APIs can be composed, e.g. if both `fully_shard` and
        # `replicate` are applied on the same module, it should raise exception.
        model = CompositeModel(device=torch.device("cpu"))
        fully_shard(model.l1)
        with self.assertRaisesRegex(AssertionError, "Cannot apply .*replicate"):
            replicate(model.l1)
        replicate(model.l2)  # should not raise

    @skip_if_lt_x_gpu(2)
    def test_fsdp_in_replicate(self):
        model = CompositeModel(device=torch.device("cuda"))
        base_model = copy.deepcopy(model)
        test_model = copy.deepcopy(model)
        fully_shard(test_model.l1)
        replicate(test_model)
        self.run_subtests(
            {
                "base_model": [base_model],
                "test_model": [test_model],
                "x": [torch.randn(2, 100, device="cuda")],
                "grad_to_none": [True, False],
            },
            self._test_parity,
        )


instantiate_parametrized_tests(TestFSDPCheckpoint)


if __name__ == "__main__":
    run_tests()
