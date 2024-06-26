# Owner(s): ["oncall: distributed"]

import sys
from typing import Dict, NamedTuple, Optional

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

    class SubtestKey(NamedTuple):
        cast_root_forward_inputs: bool
        cast_forward_inputs_submodule: bool
        use_root_no_params: bool

    class SubtestResult(NamedTuple):
        subtest_alias: str
        model_dtype: torch.dtype = torch.float16
        c1_dtype: torch.dtype = torch.float16
        c2_dtype: Optional[torch.dtype] = torch.float16

    EXPECTED_CAST_DTYPES = {
        SubtestKey(True, True, True): SubtestResult(
            "cast_root_cast_child_no_root_params"
        ),
        SubtestKey(True, True, False): SubtestResult(
            "cast_root_cast_child_root_params",
            model_dtype=torch.float32,
            c1_dtype=torch.float32,
        ),
        SubtestKey(True, False, True): SubtestResult(
            "cast_root_no_cast_child_no_root_params"
        ),
        SubtestKey(True, False, False): SubtestResult(
            "cast_root_no_cast_child_root_params",
            model_dtype=torch.float32,
            c1_dtype=torch.float32,
        ),
        SubtestKey(False, True, True): SubtestResult(
            "no_cast_root_cast_child_no_root_params", model_dtype=torch.float32
        ),
        SubtestKey(False, True, False): SubtestResult(
            "no_cast_root_cast_child_root_params",
            model_dtype=torch.float32,
            c1_dtype=torch.float32,
        ),
        SubtestKey(False, False, True): SubtestResult(
            "no_cast_root_no_cast_child_no_root_params",
            model_dtype=torch.float32,
            c1_dtype=torch.float32,
            c2_dtype=None,
        ),
        # SubtestKey(False, False, True): SubtestResult(
        #     "no_cast_root_no_cast_child_no_root_params",
        #     model_dtype=torch.float32,
        #     c1_dtype=torch.float32,
        #     c2_dtype=torch.float32), # expected result after eval mode fix
        SubtestKey(False, False, False): SubtestResult(
            "no_cast_root_no_cast_child_root_params",
            model_dtype=torch.float32,
            c1_dtype=torch.float32,
            c2_dtype=torch.float32,
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
        cast_forward_cfg = (
            cast_root_forward_inputs_submodule,
            cast_forward_inputs_submodule,
        )
        x, fsdp = self._input_and_model_init(*cast_forward_cfg, use_root_no_params)
        # self._validate_eval(x, fsdp)
        self._backward_or_validate_error(x, fsdp, *cast_forward_cfg)
        self._assert_expected_dtypes(fsdp, *cast_forward_cfg, use_root_no_params)

    def _input_and_model_init(
        self,
        cast_root_forward_inputs_submodule: bool,
        cast_forward_inputs_submodule: bool,
        use_root_no_params: bool,
    ):
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(
            param_dtype=torch.float16,
            cast_root_forward_inputs=cast_root_forward_inputs_submodule,
            cast_forward_inputs=cast_forward_inputs_submodule,
        )

        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs,
            cast_forward_inputs=False,
        ).cuda()
        x = torch.zeros(2, 100, device="cuda")

        # float16 on at least one submodule
        model.c2 = fully_shard(model.c2, mixed_precision=float16)

        # if not `use_root_no_params`, one float32 module will be left for root
        if use_root_no_params:
            # use float16 and wrap all submodules, leaving root without direct parameters
            model.c1 = fully_shard(model.c1, mixed_precision=float16)
            fsdp = fully_shard(model, mixed_precision=float16)
        else:
            fsdp = fully_shard(model)
        return x, fsdp

    def _validate_eval(
        self, input: Dict[nn.Module, torch.Tensor], fsdp_model: nn.Module
    ):
        # validate eval mode always forces full precision
        fsdp_model.eval()
        _ = fsdp_model(input)
        self.assertEqual(fsdp_model.forward_inputs[fsdp_model.c1].dtype, torch.float32)
        self.assertEqual(fsdp_model.forward_inputs[fsdp_model.c2].dtype, torch.float32)
        fsdp_model.train()

    def _backward_or_validate_error(
        self,
        input: Dict[nn.Module, torch.Tensor],
        fsdp_model: nn.Module,
        cast_root_forward_inputs_submodule: bool,
        cast_forward_inputs_submodule: bool,
    ):
        # cast_root_forward_inputs_submodule or cast_forward_inputs_submodule should be True
        if not cast_root_forward_inputs_submodule and not cast_forward_inputs_submodule:
            with self.assertRaisesRegex(
                RuntimeError,
                "mat1 and mat2 must have the same dtype",
            ):
                fsdp_model(input).sum().backward()
        else:
            fsdp_model(input).sum().backward()

    def _assert_expected_dtypes(
        self,
        fsdp_model: nn.Module,
        cast_root_forward_inputs_submodule: bool,
        cast_forward_inputs_submodule: bool,
        use_root_no_params: bool,
    ):
        subtest_key = TestMixedPrecision.SubtestKey(
            cast_root_forward_inputs_submodule,
            cast_forward_inputs_submodule,
            use_root_no_params,
        )
        subtest_fail_msg = f"Subtest `{TestMixedPrecision.EXPECTED_CAST_DTYPES[subtest_key].subtest_alias}` failed."
        self.assertEqual(
            fsdp_model.forward_inputs[fsdp_model].dtype,
            TestMixedPrecision.EXPECTED_CAST_DTYPES[subtest_key].model_dtype,
            msg=subtest_fail_msg,
        )
        for i, mod in enumerate((fsdp_model.c1, fsdp_model.c2), start=2):
            if fsdp_model.forward_inputs.get(mod, None) is not None:
                self.assertEqual(
                    fsdp_model.forward_inputs[mod].dtype,
                    TestMixedPrecision.EXPECTED_CAST_DTYPES[subtest_key][i],
                    msg=subtest_fail_msg,
                )


if __name__ == "__main__":
    run_tests()
