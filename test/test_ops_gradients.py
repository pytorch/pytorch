# Owner(s): ["module: unknown"]

from functools import partial

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
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


@unMarkDynamoStrictTest
class TestBwdGradients(TestGradients):
    # Tests that gradients are computed correctly
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
