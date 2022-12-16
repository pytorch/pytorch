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

    # NOTE: Define `fix_seed_per_iter` for now for BC since some test model
    # configs do not have a simple base model to compare against. In those
    # cases, we leverage the fact that the averaged gradient is approximately
    # equal to the local gradient to check for parity. Fixing the seed per
    # iteration helps ensure the closeness between averaged and local gradient.
    def _test_parity(
        self,
        base_model: nn.Module,
        test_model: nn.Module,
        inp_size: torch.Size,
        inp_device: torch.device,
        grad_to_none: bool,
        fix_seed_per_iter: bool,
    ):
        LR = 0.01
        base_optim = torch.optim.Adam(base_model.parameters(), lr=LR)
        test_optim = torch.optim.Adam(test_model.parameters(), lr=LR)

        for _ in range(5):
            if fix_seed_per_iter:
                torch.manual_seed(self.rank + 1)
            x = torch.randn(inp_size, device=inp_device)
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
                "inp_size": [torch.Size((2, 100))],
                "inp_device": [torch.device("cuda")],
                "grad_to_none": [True, False],
                "fix_seed_per_iter": [True],
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
                "inp_size": [torch.Size((2, 100))],
                "inp_device": [torch.device("cuda")],
                "grad_to_none": [True, False],
                "fix_seed_per_iter": [True],
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
                "inp_size": [torch.Size((2, 100))],
                "inp_device": [torch.device("cuda")],
                "grad_to_none": [True, False],
                "fix_seed_per_iter": [True],
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
                "inp_size": [torch.Size((2, 100))],
                "inp_device": [torch.device("cuda")],
                "grad_to_none": [True, False],
                "fix_seed_per_iter": [True],
            },
            self._test_parity,
        )

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_replicate_composability(self):
        """
        Tests composing ``fully_shard`` and ``replicate``. To save CI time, we
        run the different configs in subtests.

        TODO (awgu): Once https://github.com/pytorch/pytorch/pull/90711/ lands,
        add "1fm1fm,1r1r,1fm".
        """
        self.run_subtests(
            {
                "config": [
                    "1r,1fm",
                    "1r,1fa",
                    "1r1fm,1fm",
                    "1r1fa,1fm",
                ]
            },
            self._test_replicate_in_fully_shard,
        )

    def _test_replicate_in_fully_shard(self, config: str):
        """
        To interpret the config, each comma delineates a level in the module
        tree ordered bottom-up; 'r' means ``replicate``; 'f' means
        ``fully_shard``; 'a' means auto wrap; and 'm' means manual wrap.
        """
        # Set the seed to ensure that all ranks initialize the same model
        torch.manual_seed(0)
        if config == "1r,1fm":
            base_model = CompositeParamModel(torch.device("cuda"))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model)
        elif config == "1r,1fa":
            base_model = CompositeParamModel(torch.device("cuda"))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model, policy=ModuleWrapPolicy({UnitModule}))
        elif config == "1r1fm,1fm":
            base_model = CompositeParamModel(torch.device("cuda"))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model.u2)
            fully_shard(test_model)
        elif config == "1r1fa,1fm":
            base_model = CompositeParamModel(torch.device("cuda"))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model.u2, policy=ModuleWrapPolicy({UnitModule}))
            fully_shard(test_model)
        else:
            raise ValueError(f"Unknown config: {config}")
        # Apply data parallelism to the base model for parity since we apply
        # data parallelism to the test model
        replicate(base_model)

        # Set the seed to ensure that ranks get different input data
        torch.manual_seed(self.rank + 1)
        self._test_parity(
            base_model,
            test_model,
            torch.Size((2, 100)),
            torch.device("cuda"),
            True,
            False,
        )


instantiate_parametrized_tests(TestFSDPCheckpoint)


if __name__ == "__main__":
    run_tests()
