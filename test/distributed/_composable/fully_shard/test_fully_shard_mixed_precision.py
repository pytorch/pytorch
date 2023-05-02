# Owner(s): ["oncall: distributed"]

import sys
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import MixedPrecision
from torch.testing._internal.common_distributed import (
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
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


class TestMixedPrecision(FSDPTest):
    """Tests ``fully_shard`` with mixed precision."""

    EXPECTED_CAST_DTYPES = {
        # (cast_root_forward_inputs_submodule, cast_forward_inputs_submodule, use_root_no_params)
        (True, True, True): (
            "cast_root_cast_child_no_root_params",
            torch.float16,
            torch.float16,
            torch.float16,
        ),
        (True, True, False): (
            "cast_root_cast_child_root_params",
            torch.float32,
            torch.float32,
            torch.float16,
        ),
        (True, False, True): (
            "cast_root_no_cast_child_no_root_params",
            torch.float16,
            torch.float16,
            torch.float16,
        ),
        (True, False, False): (
            "cast_root_no_cast_child_root_params",
            torch.float32,
            torch.float32,
            torch.float16,
        ),
        (False, True, True): (
            "no_cast_root_cast_child_no_root_params",
            torch.float32,
            torch.float16,
            torch.float16,
        ),
        (False, True, False): (
            "no_cast_root_cast_child_root_params",
            torch.float32,
            torch.float32,
            torch.float16,
        ),
        (False, False, True): (
            "no_cast_root_no_cast_child_no_root_params",
            torch.float32,
            torch.float32,
            None,
        ),
        # (False, False, True): ('no_cast_root_no_cast_child_no_root_params',
        #                        torch.float32, torch.float32, torch.float32),  # expected result after eval mode fix
        (False, False, False): (
            "no_cast_root_no_cast_child_root_params",
            torch.float32,
            torch.float32,
            torch.float32,
        ),
    }

    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_float16_cast_forward(self):
        self.run_subtests(
            {
                "cast_root_forward_inputs_submodule": [True, False],
                "cast_forward_inputs_submodule": [True, False],
                "use_root_no_params": [True, False],
            },
            self._test_float16_cast_forward,
        )

    def _test_float16_cast_forward(
        self,
        cast_root_forward_inputs_submodule: bool,
        cast_forward_inputs_submodule: bool,
        use_root_no_params: bool,
    ):
        self.cast_root_forward_inputs_submodule = cast_root_forward_inputs_submodule
        self.cast_forward_inputs_submodule = cast_forward_inputs_submodule
        x, fsdp = self._input_and_model_init(use_root_no_params=use_root_no_params)
        # self._validate_eval(x, fsdp)
        self._backward_or_validate_error(x, fsdp)
        self._assert_expected_dtypes(use_root_no_params)

    def _backward_or_validate_error(
        self, input: Dict[nn.Module, torch.Tensor], fsdp_model: nn.Module
    ):
        # cast_root_forward_inputs_submodule or cast_forward_inputs_submodule should be True
        if (
            not self.cast_root_forward_inputs_submodule
            and not self.cast_forward_inputs_submodule
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "mat1 and mat2 must have the same dtype",
            ):
                fsdp_model(input).sum().backward()
        else:
            fsdp_model(input).sum().backward()

    def _input_and_model_init(self, use_root_no_params: bool):
        self.forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(
            param_dtype=torch.float16,
            cast_root_forward_inputs=self.cast_root_forward_inputs_submodule,
            cast_forward_inputs=self.cast_forward_inputs_submodule,
        )

        self.model = SaveForwardInputsModel(
            forward_inputs=self.forward_inputs,
            cast_forward_inputs=False,
        ).cuda()
        self.c1, self.c2 = self.model.c1, self.model.c2
        x = torch.zeros(2, 100, device="cuda")

        # float16 on at least one submodule
        self.model.c2 = fully_shard(self.model.c2, mixed_precision=float16)

        # if not `use_root_no_params`, one float32 module will be left for root
        if use_root_no_params:
            # use float16 and wrap all submodules, leaving root without direct parameters
            self.model.c1 = fully_shard(self.model.c1, mixed_precision=float16)
            fsdp = fully_shard(self.model, mixed_precision=float16)
        else:
            fsdp = fully_shard(self.model)
        return x, fsdp

    def _validate_eval(
        self, input: Dict[nn.Module, torch.Tensor], fsdp_model: nn.Module
    ):
        # validate eval mode always forces full precision
        fsdp_model.eval()
        _ = fsdp_model(input)
        self.assertEqual(self.forward_inputs[self.c1].dtype, torch.float32)
        self.assertEqual(self.forward_inputs[self.c2].dtype, torch.float32)
        fsdp_model.train()

    def _assert_expected_dtypes(self, use_root_no_params):
        result_key = (
            self.cast_root_forward_inputs_submodule,
            self.cast_forward_inputs_submodule,
            use_root_no_params,
        )
        subtest_fail_msg = f"Subtest `{TestMixedPrecision.EXPECTED_CAST_DTYPES[result_key][0]}` failed."
        self.assertEqual(
            self.forward_inputs[self.model].dtype,
            TestMixedPrecision.EXPECTED_CAST_DTYPES[result_key][1],
            msg=subtest_fail_msg,
        )
        for i, mod in enumerate((self.c1, self.c2), start=2):
            if self.forward_inputs.get(mod, None) is not None:
                self.assertEqual(
                    self.forward_inputs[mod].dtype,
                    TestMixedPrecision.EXPECTED_CAST_DTYPES[result_key][i],
                    msg=subtest_fail_msg,
                )

if __name__ == "__main__":
    run_tests()
