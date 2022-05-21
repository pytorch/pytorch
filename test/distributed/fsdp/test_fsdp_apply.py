# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    NestedWrappedModule,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
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


class TestApply(FSDPTest):
    @property
    def world_size(self):
        return 2

    @torch.no_grad()
    def _init_linear_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.fill_(1.0)
            m.bias.fill_(1.0)

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    def check_weights(self, fsdp, expected_tensor_fn, check):
        with fsdp.summon_full_params(fsdp, recurse=True):
            linear_modules = [
                module for module in fsdp.modules() if type(module) == nn.Linear
            ]
            for module in linear_modules:
                for param in module.parameters():
                    expected = expected_tensor_fn(param)
                    check(param, expected, f"Got {param} but expected {expected}")

    def _check_apply(self, fsdp):
        # Assert linear weights are not all 1.0
        self.check_weights(
            fsdp, lambda param: torch.empty_like(param).fill_(1.0), self.assertNotEqual
        )

        fsdp.apply(self._init_linear_weights)

        # Ensure all weights are 1.0
        self.check_weights(
            fsdp, lambda param: torch.empty_like(param).fill_(1.0), self.assertEqual
        )

    @skip_if_lt_x_gpu(2)
    def test_nested_module_apply(self):
        """
        Checks apply() modifies weights appropriately on a nested FSDP instance.
        """
        nested_module = NestedWrappedModule(
            self.process_group, wrap_fsdp=True, wrap_everything=True
        )
        fsdp_module = FSDP(nested_module, self.process_group).cuda(self.rank)
        self._check_apply(fsdp_module)

    @skip_if_lt_x_gpu(2)
    def test_transformer_module_apply(self):
        """
        Checks apply() modifies weights appropriately on a wrapped Transformer
        module.
        """
        transformer = self._get_wrapped_model(group=self.process_group).cuda(self.rank)
        self._check_apply(transformer)

    @skip_if_lt_x_gpu(2)
    def test_apply_in_summon_raises_error(self):
        """
        Ensures that if user calls apply() on FSDP instance within full param
        summon context, appropriate error is raised.
        """
        transformer = self._get_wrapped_model(group=self.process_group).cuda(self.rank)
        with transformer.summon_full_params(transformer, recurse=True):
            with self.assertRaisesRegex(ValueError, "expected to be in states"):
                transformer.apply(self._init_linear_weights)


if __name__ == "__main__":
    run_tests()
