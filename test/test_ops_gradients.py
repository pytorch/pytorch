# Owner(s): ["module: unknown"]

from functools import partial, wraps
from itertools import chain
import torch

from torch.testing._internal.common_utils import (
    TestCase, is_iterable_of_tensors, run_tests, gradcheck, gradgradcheck, is_slow_gradcheck_env,
    skipIfTorchInductor)
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

        samples = op.sample_inputs(device, dtype, requires_grad=True, include_conjugated_inputs=include_conjugated_inputs,
                                   small_inputs_only=is_slow_gradcheck_env())

        for sample in samples:
            if sample.broadcasts_input and is_inplace(variant):
                continue

            # Gradcheck expects tensors as its input, but autograd actually supports tensorlists
            #   and tensors passed as kwargs. The following creates a function that accepts just
            #   the tensors that require grad as varargs, and then recomposes them back into the
            #   original input.

            # Creates gradcheck inputs by identifying tensors requiring grad
            all_args = None
            if is_iterable_of_tensors(sample.input):
                all_args = chain(sample.input, sample.args, sample.kwargs.values())
            else:
                all_args = tuple(chain((sample.input,), sample.args, sample.kwargs.values()))
            gradcheck_args = tuple(x for x in all_args if (isinstance(x, torch.Tensor) and x.requires_grad))

            def _input_recomposition_helper(inputs, inp, input_idx):
                if is_iterable_of_tensors(inp):
                    tensor_list = []
                    for x in inp:
                        if isinstance(x, torch.Tensor) and x.requires_grad:
                            tensor_list.append(inputs[input_idx])
                            input_idx = input_idx + 1
                        else:
                            tensor_list.append(x)
                    return tensor_list, input_idx
                elif isinstance(inp, torch.Tensor) and inp.requires_grad:
                    return inputs[input_idx], input_idx + 1
                else:
                    return inp, input_idx

            def fn(*inputs):
                # Puts inputs back into sample properly
                positional_args = []
                input_idx = 0
                inp, input_idx = _input_recomposition_helper(inputs, sample.input, input_idx)
                positional_args.append(inp)

                for x in sample.args:
                    inp, input_idx = _input_recomposition_helper(inputs, x, input_idx)
                    positional_args.append(inp)

                # Recreates kwargs
                kwargs = {}
                for k, v in sample.kwargs.items():
                    inp, input_idx = _input_recomposition_helper(inputs, v, input_idx)
                    kwargs[k] = inp

                output = op.gradcheck_wrapper(variant, *positional_args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

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
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest("Skipped! Op doesn't support autograd for this dtype.")
        if not op.supports_autograd and not op.supports_forward_ad:
            self.skipTest("Skipped! autograd not supported.")
        if op.name == "cat":
            self.skipTest("TODO(whc) fix pre-existing bug with cat for newly added opinfo for empty+nonempty")

    # Tests that gradients are computed correctly
    @_gradcheck_ops(op_db)
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

    @_gradcheck_ops(op_db)
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
            self._grad_test_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()))

    # Test that gradients of gradients are computed correctly
    @_gradcheck_ops(op_db)
    def test_fn_gradgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if not op.supports_gradgrad:
            self.skipTest("Op claims it doesn't support gradgrad. This is not verified.")
        else:
            self._check_helper(device, dtype, op, op.get_op(), 'bwgrad_bwgrad')

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


instantiate_device_type_tests(TestGradients, globals())

if __name__ == '__main__':
    run_tests()
