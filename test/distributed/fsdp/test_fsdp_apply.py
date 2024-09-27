# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
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


class TestApply(FSDPTest):
    @property
    def world_size(self):
        return 2

    @torch.no_grad()
    def _init_linear_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.fill_(1.0)
            m.bias.fill_(1.0)

    def check_weights(self, fsdp, expected_tensor_fn, check):
        with FSDP.summon_full_params(fsdp, recurse=True):
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
        """Tests that ``apply()`` modifies parameter values in-place on a
        non-FSDP-root nested FSDP-wrapped model."""
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
        )
        self._check_apply(nested_wrapped_module)

    @skip_if_lt_x_gpu(2)
    def test_transformer_module_apply(self):
        """Tests that ``apply()`` modifies parameter values in-place on an
        FSDP-wrapped transformer model with shared parameters."""
        transformer = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
        )
        self._check_apply(transformer)

    @skip_if_lt_x_gpu(2)
    def test_apply_in_summon_raises_error(self):
        """Tests that calling ``apply()`` on an FSDP instance inside the
        ``summon_full_params()`` context raises an error."""
        transformer = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_AFTER,
        )
        with transformer.summon_full_params(transformer):
            with self.assertRaisesRegex(ValueError, "expected to be in states"):
                transformer.apply(self._init_linear_weights)


if __name__ == "__main__":
    run_tests()
