# Owner(s): ["oncall: distributed"]

import sys

from torch import distributed as dist
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


class TestFSDPIgnoredModules(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_ignored_modules(self):
        """Tests that ignored modules' parameters are not flattened."""
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
        with wrapped_model.summon_full_params():
            flat_param_numel = wrapped_model.params[0].numel()
            self.assertEqual(flat_param_numel, nonignored_numel)


instantiate_parametrized_tests(TestFSDPIgnoredModules)

if __name__ == "__main__":
    run_tests()
