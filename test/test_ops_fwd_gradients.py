# Owner(s): ["module: unknown"]

from functools import partial
import torch

from torch.testing._internal.common_utils import (
    TestGradients, run_tests, skipIfTorchInductor, IS_MACOS)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, OpDTypes)

# TODO: mitigate flaky issue on macOS https://github.com/pytorch/pytorch/issues/66033
# AFAIK, c10::ThreadPool looks correct in the way it uses condition_variable wait. The
# issue seems to point to macOS itself https://github.com/graphia-app/graphia/issues/33
if IS_MACOS:
    torch.set_num_threads(1)

# gradcheck requires double precision
_gradcheck_ops = partial(ops, dtypes=OpDTypes.supported,
                         allowed_dtypes=[torch.double, torch.cdouble])

class TestFwdGradients(TestGradients):
    # Test that forward-over-reverse gradgrad is computed correctly
    @_gradcheck_ops(op_db)
    def test_fn_fwgrad_bwgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        if op.supports_fwgrad_bwgrad:
            self._check_helper(device, dtype, op, op.get_op(), "fwgrad_bwgrad")
        else:
            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = ("Running forward-over-backward gradgrad for an OP that has does not support it did not "
                        "raise any error. If your op supports forward AD, you should set supports_fwgrad_bwgrad=True.")
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                self._check_helper(device, dtype, op, op.get_op(), "fwgrad_bwgrad")


    def _forward_grad_helper(self, device, dtype, op, variant, is_inplace):
        # TODO: clean up how attributes are passed to gradcheck from OpInfos
        def call_grad_test_helper():
            check_batched_forward_grad = ((op.check_batched_forward_grad and not is_inplace) or
                                          (op.check_inplace_batched_forward_grad and is_inplace))
            self._grad_test_helper(device, dtype, op, variant, check_forward_ad=True, check_backward_ad=False,
                                   check_batched_grad=False, check_batched_forward_grad=check_batched_forward_grad)
        if op.supports_forward_ad:
            call_grad_test_helper()
        else:
            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = ("Running forward AD for an OP that has does not support it did not "
                        "raise any error. If your op supports forward AD, you should set supports_forward_ad=True")
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                call_grad_test_helper()

    @_gradcheck_ops(op_db)
    def test_forward_mode_AD(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        self._forward_grad_helper(device, dtype, op, op.get_op(), is_inplace=False)

    @_gradcheck_ops(op_db)
    @skipIfTorchInductor("to be fixed")
    def test_inplace_forward_mode_AD(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")

        self._forward_grad_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()), is_inplace=True)

instantiate_device_type_tests(TestFwdGradients, globals())

if __name__ == '__main__':
    run_tests()
