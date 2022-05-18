# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    instantiate_parametrized_tests,
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


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(3, 5)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(5, 5),
            torch.nn.Linear(5, 4),
            torch.nn.Linear(4, 4),
        )
        self.layer2 = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.layer2(self.layer1(self.layer0(x)))

    def get_input(self, device):
        return (torch.randn((8, 3)).to(device),)

    def get_loss(self, input, output):
        return output.sum()

    def run_backward(self, loss):
        loss.backward()

class TestFSDPIgnoredModules(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_transformer(self):
        """Tests that ignored modules' parameters are not flattened for a
        transformer model with shared parameters."""
        # Initialize an FSDP-wrapped transformer model that has FSDP ignore
        # the `nn.Transformer` module's parameters
        group = dist.distributed_c10d._get_default_group()
        wrapped_model = self._get_wrapped_model(group, ignore_modules=True)
        # Check that the wrapped model's flattened parameter does not include
        # the ignored transformer module's parameters
        nonwrapped_model = self._get_nonwrapped_model(group)
        total_numel = sum(p.numel() for p in nonwrapped_model.parameters())
        ignored_numel = sum(
            p.numel() for p in nonwrapped_model.transformer.parameters()
        )
        nonignored_numel = total_numel - ignored_numel
        with FSDP.summon_full_params(wrapped_model):
            flat_param_numel = wrapped_model.params[0].numel()
            self.assertEqual(flat_param_numel, nonignored_numel)
        # Check that we can run a few iterations
        device = torch.device("cuda")
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        for _ in range(3):
            inp = wrapped_model.module.get_input(device)
            output = wrapped_model(*inp)
            loss = wrapped_model.module.get_loss(inp, output).to(device)
            wrapped_model.module.run_backward(loss)
            optim.step()

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_nested(self):
        """Tests that passing a module with nested FSDP modules does not
        error and still ignores non-FSDP modules' parameters."""
        # Initialize an FSDP-wrapped nested model that first wraps the nested
        # sequential's middle linear layer (`layer1[1]`) and then wraps the
        # overall model while ignoring the nested sequential (`layer1`)
        model = Model().cuda()
        model.layer1[1] = FSDP(model.layer1[1])
        wrapped_model = FSDP(model, ignored_modules=[model.layer1])
        # Check that the wrapped model's flattened parameter does not include
        # the ignored nested sequential's parameters
        nonwrapped_model = Model()
        total_numel = sum(p.numel() for p in nonwrapped_model.parameters())
        ignored_numel = sum(
            p.numel() for p in nonwrapped_model.layer1.parameters()
        )
        nonignored_numel = total_numel - ignored_numel
        with FSDP.summon_full_params(wrapped_model):
            flat_param_numel = wrapped_model.params[0].numel()
            self.assertEqual(flat_param_numel, nonignored_numel)
        # Check that we can run a few iterations
        device = torch.device("cuda")
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        for _ in range(3):
            inp = wrapped_model.get_input(device)
            output = wrapped_model(*inp)
            loss = wrapped_model.get_loss(inp, output).to(device)
            wrapped_model.run_backward(loss)
            optim.step()

    def test_ignored_modules_invalid(self):
        """Tests that passing an FSDP module as an ignored module errors."""
        model = Model()
        model.layer1 = FSDP(model.layer1)
        # Passing an FSDP module as an ignored module should error
        with self.assertRaises(
            ValueError,
            msg="`ignored_modules` should not include FSDP modules",
        ):
            FSDP(model, ignored_modules=[model.layer1])


instantiate_parametrized_tests(TestFSDPIgnoredModules)

if __name__ == "__main__":
    run_tests()
