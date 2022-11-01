# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
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
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(3, 5)
        layer1_modules = [
            torch.nn.Linear(5, 4),
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 4),
        ]
        self.layer1 = torch.nn.Sequential(*layer1_modules)
        self.layer2 = torch.nn.Linear(4, 2)
        self.layer3 = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        z = self.relu(self.layer0(x))
        z = self.relu(self.layer1(z))
        z = self.relu(self.layer2(z))
        z = self.relu(self.layer3(z))
        return z

    def get_input(self, device):
        return (torch.randn((8, 3)).to(device),)

    def get_loss(self, input, output):
        return output.sum()

    def run_backward(self, loss):
        loss.backward()


class IgnoredModule(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))

    def forward(self, x):
        return x @ self.weight


class ModelWithIgnoredModules(Model):
    """Adds a variable number of :class:`IgnoredModule` to ``self.layer1``."""

    def __init__(self, num_ignored: int) -> None:
        assert num_ignored >= 0
        super().__init__()
        layer1_modules = (
            [torch.nn.Linear(5, 4), torch.nn.Linear(4, 4)]
            + [IgnoredModule(4, 4) for _ in range(num_ignored)]
            + [torch.nn.Linear(4, 4)]
        )
        self.layer1 = torch.nn.Sequential(*layer1_modules)


class TestFSDPIgnoredModules(FSDPTest):
    def _train_model(self, model, optim, num_iters, device=torch.device("cuda")):
        for _ in range(num_iters):
            inp = model.module.get_input(device)
            output = model(*inp)
            loss = model.module.get_loss(inp, output).to(device)
            model.module.run_backward(loss)
            optim.step()

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_transformer(self):
        """Tests that ignored modules' parameters are not flattened for a
        transformer model with shared parameters."""
        # Initialize an FSDP-wrapped transformer model that has FSDP ignore
        # the `nn.Transformer` module's parameters
        model: nn.Module = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        wrapped_model = FSDP(
            model,
            self.process_group,
            ignored_modules=[model.transformer],
        )
        # Check that the wrapped model's flattened parameter does not include
        # the ignored transformer module's parameters
        nonwrapped_model: nn.Module = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        total_numel = sum(p.numel() for p in nonwrapped_model.parameters())
        ignored_numel = sum(
            p.numel() for p in nonwrapped_model.transformer.parameters()
        )
        nonignored_numel = total_numel - ignored_numel
        with FSDP.summon_full_params(wrapped_model):
            flat_param_numel = wrapped_model.params[0].numel()
            self.assertEqual(flat_param_numel, nonignored_numel)
        # Check that we can run a few iterations
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        self._train_model(wrapped_model, optim, 3)

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_nested(self):
        """Tests that passing a module with nested FSDP modules does not
        error and still ignores non-FSDP modules' parameters."""
        # Initialize an FSDP-wrapped nested model that first wraps the nested
        # sequential's second linear layer (`layer1[1]`) and then wraps the
        # overall model while ignoring the nested sequential (`layer1`)
        model = Model().cuda()
        model.layer1[1] = FSDP(model.layer1[1])
        wrapped_model = FSDP(model, ignored_modules=[model.layer1])
        # Check that the wrapped model's flattened parameter does not include
        # the ignored nested sequential's parameters
        nonwrapped_model = Model()
        total_numel = sum(p.numel() for p in nonwrapped_model.parameters())
        ignored_numel = sum(p.numel() for p in nonwrapped_model.layer1.parameters())
        nonignored_numel = total_numel - ignored_numel
        with FSDP.summon_full_params(wrapped_model):
            flat_param_numel = wrapped_model.params[0].numel()
            self.assertEqual(flat_param_numel, nonignored_numel)
        # Check that we can run a few iterations
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        self._train_model(wrapped_model, optim, 3)

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_invalid(self):
        """Tests that passing an FSDP module as an ignored module or the
        top-level module itself errors."""
        model = Model().cuda()
        model.layer1 = FSDP(model.layer1)
        # Passing an FSDP module as an ignored module should error
        with self.assertRaises(
            ValueError,
            msg="`ignored_modules` should not include FSDP modules",
        ):
            FSDP(model, ignored_modules=[model.layer1])
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="Trying to ignore the top-level module passed into "
            "the FSDP constructor itself will result in all parameters being "
            "ignored",
        ):
            FSDP(model, ignored_modules=[model])

    @skip_if_lt_x_gpu(2)
    @parametrize("pass_ignored_modules_to_root", [False, True])
    def test_diff_ignored_modules_across_ranks(
        self, pass_ignored_modules_to_root: bool
    ):
        """
        Tests ignoring different modules across ranks.

        Args:
            pass_ignored_modules_to_root (bool): If ``False``, does not pass
                any ignored modules (including those already ignored in child
                FSDP instances) to the root FSDP instance; if ``True``, passes
                all ignored modules (representing a superset of the children's
                ignored modules) to the root FSDP instance.
        """
        # To exercise different `FlatParameter` enumerations across ranks,
        # we wrap `layer3` with FSDP, where `layer3` is registered as a module
        # after `layer1`, which has the variable number of ignored modules
        model = ModelWithIgnoredModules(num_ignored=self.rank + 1).cuda()
        layer1_ignored_modules = [
            m for m in model.layer1.modules() if isinstance(m, IgnoredModule)
        ]
        model.layer1 = FSDP(model.layer1, ignored_modules=layer1_ignored_modules)
        model.layer3 = FSDP(model.layer3)
        model_ignored_modules = (
            [m for m in model.modules() if isinstance(m, IgnoredModule)]
            if pass_ignored_modules_to_root
            else []
        )
        wrapped_model = FSDP(model, ignored_modules=model_ignored_modules)
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        self._train_model(wrapped_model, optim, 3)


instantiate_parametrized_tests(TestFSDPIgnoredModules)

if __name__ == "__main__":
    run_tests()
