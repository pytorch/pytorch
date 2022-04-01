# Owner(s): ["high priority"]

from functools import partial, wraps
import torch

from torch.testing._internal.common_utils import \
    (TestCase, is_iterable_of_tensors, run_tests, gradcheck, gradgradcheck, first_sample)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, OpDTypes)

# TODO: fixme https://github.com/pytorch/pytorch/issues/68972
torch.set_default_dtype(torch.float32)

# gradcheck requires double precision
_gradcheck_ops = partial(ops, dtypes=OpDTypes.supported,
                         allowed_dtypes=[torch.double, torch.cdouble])

class TestGradients(TestCase):
    exact_dtype = True

    # Copies inputs to inplace operations to avoid inplace modifications
    #   to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    def _check_helper(self, device, dtype, op, variant, check, *, check_forward_ad=False, check_backward_ad=True,
                      check_batched_grad=None, check_batched_forward_grad=False):
        assert check in ('gradcheck', 'bwgrad_bwgrad', 'fwgrad_bwgrad')
        # NB: check_backward_ad does not affect gradgradcheck (always True)
        if variant is None:
            self.skipTest("Skipped! Variant not implemented.")
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest(f"Skipped! {op.name} does not support dtype {str(dtype)}")

        def is_inplace(variant):
            if hasattr(variant, "__wrapped__"):
                return variant.__wrapped__ is op.get_inplace()
            return variant is op.get_inplace()

        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(device, dtype, requires_grad=True, include_conjugated_inputs=include_conjugated_inputs)

        for sample in samples:
            if sample.broadcasts_input and is_inplace(variant):
                continue

            # Note on TensorList inputs
            #
            # gradcheck does not support TensorList inputs so here we pass TensorList
            # inputs of size n as n single Tensor inputs to gradcheck and wrap the op
            # in a function that puts the n Tensor inputs back into a TensorList
            def fn(*inputs):
                # Put tensors back into TensorList since we splat them when passing to gradcheck
                if is_iterable_of_tensors(sample.input):
                    n = len(sample.input)
                    inputs = (inputs[:n], *inputs[n:])
                output = op.gradcheck_wrapper(variant, *inputs, **sample.kwargs)
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            # Splat TensorList inputs into single Tensor inputs
            gradcheck_args = (sample.input,) if isinstance(sample.input, torch.Tensor) else tuple(sample.input)
            gradcheck_args += sample.args

            if check == 'gradcheck':
                if check_batched_grad is None:
                    check_batched_grad = op.check_batched_grad
                self.assertTrue(gradcheck(fn, gradcheck_args,
                                          check_batched_grad=check_batched_grad,
                                          check_grad_dtypes=True,
                                          nondet_tol=op.gradcheck_nondet_tol,
                                          fast_mode=op.gradcheck_fast_mode,
                                          check_forward_ad=check_forward_ad,
                                          check_backward_ad=check_backward_ad,
                                          check_undefined_grad=True,
                                          check_batched_forward_grad=check_batched_forward_grad))
            elif check in ('bwgrad_bwgrad', 'fwgrad_bwgrad'):  # gradgrad check
                self.assertFalse(check_forward_ad, msg="Cannot run forward AD check for gradgradcheck")
                for gen_non_contig_grad_outputs in (False, True):
                    kwargs = {
                        "gen_non_contig_grad_outputs": gen_non_contig_grad_outputs,
                        "check_batched_grad": op.check_batched_gradgrad,
                        "check_grad_dtypes": True,
                        "nondet_tol": op.gradcheck_nondet_tol,
                        "fast_mode": op.gradcheck_fast_mode
                    }
                    if check == "fwgrad_bwgrad":
                        kwargs["check_fwd_over_rev"] = True
                        kwargs["check_rev_over_rev"] = False
                        kwargs["check_batched_grad"] = False
                        kwargs["check_undefined_grad"] = False

                    self.assertTrue(gradgradcheck(fn, gradcheck_args, **kwargs))
            else:
                self.assertTrue(False, msg="Unknown check requested!")

    def _grad_test_helper(self, device, dtype, op, variant, *, check_forward_ad=False, check_backward_ad=True,
                          check_batched_grad=None, check_batched_forward_grad=False):
        return self._check_helper(device, dtype, op, variant, 'gradcheck', check_forward_ad=check_forward_ad,
                                  check_backward_ad=check_backward_ad, check_batched_grad=check_batched_grad,
                                  check_batched_forward_grad=check_batched_forward_grad)

    def _skip_helper(self, op, device, dtype):
        if not op.supports_autograd and not op.supports_forward_ad:
            self.skipTest("Skipped! autograd not supported.")
        if not op.supports_complex_autograd(torch.device(device).type) and dtype.is_complex:
            self.skipTest("Skipped! Complex autograd not supported.")

    # Tests that gradients are computed correctly
    @_gradcheck_ops(op_db)
    def test_fn_grad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        self._grad_test_helper(device, dtype, op, op.get_op())

    # Method grad (and gradgrad, see below) tests are disabled since they're
    #   costly and redundant with function grad (and gradgad) tests
    # @_gradcheck_ops(op_db)
    # def test_method_grad(self, device, dtype, op):
    #     self._skip_helper(op, device, dtype)
    #     self._grad_test_helper(device, dtype, op, op.get_method())

    @_gradcheck_ops(op_db)
    def test_inplace_grad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")
        self._grad_test_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()))

    # Test that gradients of gradients are computed correctly
    @_gradcheck_ops(op_db)
    def test_fn_gradgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
        self._check_helper(device, dtype, op, op.get_op(), 'bwgrad_bwgrad')

    # Test that forward-over-reverse gradgrad is computed correctly
    @_gradcheck_ops(op_db)
    def test_fn_fwgrad_bwgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        if op.supports_fwgrad_bwgrad:
            self._check_helper(device, dtype, op, op.get_op(), "fwgrad_bwgrad")
        else:
            err_msg = r"Trying to use forward AD with .* that does not support it\."
            hint_msg = ("Running forward-over-backward gradgrad for an OP that has does not support it did not "
                        "raise any error. If your op supports forward AD, you should set supports_fwgrad_bwgrad=True.")
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                self._check_helper(device, dtype, op, op.get_op(), "fwgrad_bwgrad")

    # Test that gradients of gradients are properly raising
    @_gradcheck_ops(op_db)
    def test_fn_fail_gradgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if op.supports_gradgrad:
            self.skipTest("Skipped! Operation does support gradgrad")

        err_msg = r"derivative for .* is not implemented"
        with self.assertRaisesRegex(RuntimeError, err_msg):
            self._check_helper(device, dtype, op, op.get_op(), 'bwgrad_bwgrad')

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
        self._check_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()), "bwgrad_bwgrad")

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
            err_msg = r"Trying to use forward AD with .* that does not support it\."
            hint_msg = ("Running forward AD for an OP that has does not support it did not "
                        "raise any error. If your op supports forward AD, you should set supports_forward_ad=True")
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                call_grad_test_helper()

    @_gradcheck_ops(op_db)
    def test_forward_mode_AD(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        self._forward_grad_helper(device, dtype, op, op.get_op(), is_inplace=False)

    @_gradcheck_ops(op_db)
    def test_inplace_forward_mode_AD(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")

        self._forward_grad_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()), is_inplace=True)

    # Functions that do not support autograd should not fail in forward mode
    # Inplace functions (such as "resize_") are expected to fail in forward mode and should be skipped
    # Test only when supports_autograd=False and for double dtype
    @ops(filter(lambda op: not op.supports_autograd, op_db), dtypes=OpDTypes.supported, allowed_dtypes=(torch.double,))
    def test_nondifferentiable(self, device, dtype, op):
        # Expecting no errors
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        sample = first_sample(self, samples)
        result = op(sample.input, *sample.args, **sample.kwargs)



instantiate_device_type_tests(TestGradients, globals())

if __name__ == '__main__':
    run_tests()
