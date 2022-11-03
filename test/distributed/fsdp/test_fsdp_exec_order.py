# Owner(s): ["oncall: distributed"]

import sys
import warnings
from contextlib import suppress

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
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


class Model(torch.nn.Module):
    """
    Model that supports two computation paths: `layer0` -> `layer1` and
    `layer0` -> `layer2`. Notably, both `layer1` and `layer2` have 36 elements
    when flattened, which means that their corresponding all-gathers and
    reduce-scatters may be silently matched if we do not perform any checks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(5, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6, bias=False),
        )
        self.relu = torch.nn.ReLU()
        self.use_alt_path = False
        for param in self.layer2.parameters():
            param.requires_grad = False

    def forward(self, x):
        # `layer0` -> `layer1` (normal)
        # `layer0` -> `layer2` (alternate)
        z = self.relu(self.layer0(x))
        z = (
            self.relu(self.layer2(z))
            if self.use_alt_path
            else self.relu(self.layer1(z))
        )
        return z

    def get_input(self, device: torch.device):
        return (torch.randn((8, 5)).to(device),)

    def get_loss(self, input, output):
        return output.sum()

    def run_backward(self, loss):
        loss.backward()

    def flip_path(self):
        params_to_freeze = (
            self.layer2.parameters() if self.use_alt_path else self.layer1.parameters()
        )
        params_to_unfreeze = (
            self.layer1.parameters() if self.use_alt_path else self.layer2.parameters()
        )
        for param in params_to_freeze:
            param.requires_grad = False
        for param in params_to_unfreeze:
            param.requires_grad = True
        self.use_alt_path = not self.use_alt_path

    @staticmethod
    def wrap(sharding_strategy: ShardingStrategy, device: torch.device):
        model = Model()
        model.layer1 = FSDP(model.layer1, sharding_strategy=sharding_strategy)
        model.layer2 = FSDP(model.layer2, sharding_strategy=sharding_strategy)
        fsdp_model = FSDP(model, sharding_strategy=sharding_strategy)
        return fsdp_model.to(device)


class TestFSDPExecOrder(FSDPTest):
    def setUp(self):
        super().setUp()

    @property
    def device(self):
        return torch.device("cuda")

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    def test_invalid_first_iter_order(
        self,
        sharding_strategy: ShardingStrategy,
    ):
        """Tests that FSDP errors if the all-gather order differs across ranks
        in the first iteration."""
        dist.set_debug_level(dist.DebugLevel.INFO)
        # Rank 0 runs the forward pass in one order and all other ranks run in
        # different order
        dist.set_debug_level(dist.DebugLevel.INFO)
        fsdp_model = Model.wrap(sharding_strategy, self.device)
        if self.rank != 0:
            fsdp_model.flip_path()
        inp = fsdp_model.module.get_input(self.device)
        # Match the error message with the following prefix
        error_regex = "^(Forward order differs across ranks)"
        with self.assertRaisesRegex(RuntimeError, error_regex):
            fsdp_model(*inp)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    @parametrize("iters_before_path_change", [1, 3])
    def test_invalid_later_iter_order(
        self,
        sharding_strategy: ShardingStrategy,
        iters_before_path_change: int,
    ):
        """Tests that FSDP warns the user if the all-gather order changes after
        the first iteration."""
        dist.set_debug_level(dist.DebugLevel.INFO)
        # On the first iteration, all ranks run the same order, and on the next
        # iteration, all but rank 0 run in a different order
        fsdp_model = Model.wrap(sharding_strategy, self.device)
        for _ in range(iters_before_path_change):
            inp = fsdp_model.module.get_input(self.device)
            output = fsdp_model(*inp)
            loss = fsdp_model.module.get_loss(inp, output).to(self.device)
            fsdp_model.module.run_backward(loss)
        # Match the warning message with the following prefix
        regex = (
            "^(Forward order differs from that of the first iteration "
            f"on rank {self.rank}. Collectives are unchecked and may give "
            "incorrect results or hang)"
        )
        context = (
            self.assertWarnsRegex(
                expected_warning=UserWarning,
                expected_regex=regex,
            )
            if self.rank != 0
            else suppress()
        )
        if self.rank != 0:
            fsdp_model.flip_path()
        inp = fsdp_model.module.get_input(self.device)
        # Expect a warning for the forward pass all-gather
        with context:  # warning for forward pass all-gather
            output = fsdp_model(*inp)
        loss = fsdp_model.module.get_loss(inp, output).to(self.device)
        fsdp_model.module.run_backward(loss)
        # Run an additional iteration to check that there are no more warnings
        inp = fsdp_model.module.get_input(self.device)
        output = fsdp_model(*inp)
        loss = fsdp_model.module.get_loss(inp, output).to(self.device)
        fsdp_model.module.run_backward(loss)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "sharding_strategy",
        [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
    )
    def test_train_eval(self, sharding_strategy: ShardingStrategy):
        dist.set_debug_level(dist.DebugLevel.INFO)
        fsdp_model = Model.wrap(sharding_strategy, self.device)
        NUM_ITERS = 3
        NUM_EPOCHS = 2
        with warnings.catch_warnings(record=True) as w:  # records warnings to `w`
            for _ in range(NUM_EPOCHS):
                fsdp_model.train()
                for _ in range(NUM_ITERS):
                    inp = fsdp_model.module.get_input(self.device)
                    output = fsdp_model(*inp)
                    loss = fsdp_model.module.get_loss(inp, output).to(self.device)
                    fsdp_model.module.run_backward(loss)
                fsdp_model.eval()
                for _ in range(NUM_ITERS):
                    inp = fsdp_model.module.get_input(self.device)
                    output = fsdp_model(*inp)
                    fsdp_model.module.get_loss(inp, output).to(self.device)
        # Check that the order validation warning was not issued (errors do not
        # need to be checked since they will be directly reported)
        warning_prefix = "Forward order differs"
        for warning in w:
            if str(warning.message).startswith(warning_prefix):
                raise AssertionError(
                    f"Warning was incorrectly issued: {warning.message}"
                )
        # If we still validate the forward execution order in eval mode, then
        # an `AssertionError` will be raised above for both sharding strategies


instantiate_parametrized_tests(TestFSDPExecOrder)

if __name__ == "__main__":
    run_tests()
