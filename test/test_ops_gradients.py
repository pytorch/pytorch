# Owner(s): ["module: unknown"]

from functools import partial

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
    skip,
    skipOps,
    xfail,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    TestGradients,
    unMarkDynamoStrictTest,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.hop_db import hop_db


# gradcheck requires double precision
_gradcheck_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=[torch.double, torch.cdouble]
)


# Device-agnostic skips migrated from OpInfo definitions (see #177259).
_bwd_grad_all = {
    skip("as_strided"),
    skip("as_strided_copy"),
    skip("round", variant_name="decimals_3"),
    skip("round", variant_name="decimals_neg_3"),
    skip("__rpow__"),
    skip("polygamma", variant_name="polygamma_n_1"),
    skip("polygamma", variant_name="polygamma_n_2"),
    skip("polygamma", variant_name="polygamma_n_3"),
    skip("polygamma", variant_name="polygamma_n_4"),
    xfail("bfloat16"),
    xfail("float"),
    xfail("half"),
    xfail("cfloat"),
    xfail("chalf"),
    skip("normal"),
    skip("normal", variant_name="number_mean"),
    skip("linalg.lstsq"),
}


@unMarkDynamoStrictTest
class TestBwdGradients(TestGradients):
    # Tests that gradients are computed correctly
    @skipOps(
        _bwd_grad_all
        | {
            xfail("cov"),
            xfail("istft"),
            xfail("native_group_norm"),
            skip("sparse.sampled_addmm"),
            skip("sparse.mm", variant_name="reduce"),
            xfail("as_strided_scatter"),
            skip("nn.functional.max_unpool1d"),
            skip("nn.functional.max_unpool2d"),
            skip("nn.functional.max_unpool3d"),
            xfail("linalg.norm", variant_name="subgradients_at_zero"),
        }
    )
    @_gradcheck_ops(op_db + hop_db + custom_op_db)
    def test_fn_grad(self, device, dtype, op):
        # This is verified by test_dtypes in test_ops.py
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest("Skipped! Dtype is not in supported backward dtypes!")
        else:
            self._grad_test_helper(device, dtype, op, op.get_op())

    # Method grad (and gradgrad, see below) tests are disabled since they're
    #   costly and redundant with function grad (and gradgad) tests
    # @_gradcheck_ops(op_db)
    # def test_method_grad(self, device, dtype, op):
    #     self._skip_helper(op, device, dtype)
    #     self._grad_test_helper(device, dtype, op, op.get_method())

    @skipOps(
        _bwd_grad_all
        | {
            skip("abs", dtypes=(torch.cdouble,)),
            xfail("as_strided", variant_name="partial_views"),
        }
    )
    @_gradcheck_ops(op_db + custom_op_db)
    def test_inplace_grad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if not op.inplace_variant:
            self.skipTest("Op has no inplace variant!")

        # Verifies an operation doesn't support inplace autograd if it claims not to
        if not op.supports_inplace_autograd:
            inplace = self._get_safe_inplace(op.get_inplace())
            for sample in op.sample_inputs(device, dtype, requires_grad=True):
                if sample.broadcasts_input:
                    continue
                with self.assertRaises(Exception):
                    result = inplace(sample)
                    result.sum().backward()
        else:
            self._grad_test_helper(
                device, dtype, op, self._get_safe_inplace(op.get_inplace())
            )

    # Test that gradients of gradients are computed correctly
    @skipOps(
        _bwd_grad_all
        | {
            xfail("cov"),
            skip("sparse.sampled_addmm"),
            skip("sparse.mm", variant_name="reduce"),
            xfail("native_layer_norm"),
            skip("nn.functional.max_unpool1d"),
            skip("nn.functional.max_unpool2d"),
            skip("nn.functional.max_unpool3d"),
            xfail("cat"),
            xfail("nn.functional.ctc_loss", dtypes=(torch.float64,)),
            xfail("nn.functional.linear_cross_entropy", variant_name="chunked_none"),
            xfail("nn.functional.linear_cross_entropy", variant_name="chunked"),
            xfail("linalg.norm"),
            xfail("linalg.norm", variant_name="subgradients_at_zero"),
            skip("masked.logaddexp"),
        }
    )
    @_gradcheck_ops(op_db + hop_db + custom_op_db)
    def test_fn_gradgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if not op.supports_gradgrad:
            self.skipTest(
                "Op claims it doesn't support gradgrad. This is not verified."
            )
        else:
            self._check_helper(device, dtype, op, op.get_op(), "bwgrad_bwgrad")

    # Test that gradients of gradients are properly raising
    @skipOps(_bwd_grad_all | {skip("sparse.mm", variant_name="reduce")})
    @_gradcheck_ops(op_db + custom_op_db)
    def test_fn_fail_gradgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if op.supports_gradgrad:
            self.skipTest("Skipped! Operation does support gradgrad")

        err_msg = r"derivative for .* is not implemented"
        with self.assertRaisesRegex(RuntimeError, err_msg):
            self._check_helper(device, dtype, op, op.get_op(), "bwgrad_bwgrad")

    # Method gradgrad (and grad, see above) tests are disabled since they're
    #   costly and redundant with function gradgrad (and grad) tests
    # @_gradcheck_ops(op_db)
    # def test_method_gradgrad(self, device, dtype, op):
    #     self._skip_helper(op, device, dtype)
    #     self._gradgrad_test_helper(device, dtype, op, op.get_method())

    @skipOps(
        _bwd_grad_all
        | {
            skip("abs", dtypes=(torch.cdouble,)),
            xfail("as_strided", variant_name="partial_views"),
            xfail("nn.functional.hardsigmoid"),
        }
    )
    @_gradcheck_ops(op_db)
    def test_inplace_gradgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")
        self._check_helper(
            device, dtype, op, self._get_safe_inplace(op.get_inplace()), "bwgrad_bwgrad"
        )


instantiate_device_type_tests(TestBwdGradients, globals())

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
