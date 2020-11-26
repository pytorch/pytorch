from functools import partial, wraps

import torch

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
from torch.testing._internal.common_methods_invocations import \
    (op_db)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, dtypes, onlyOnCPUAndCUDA, skipCUDAIfRocm)
from torch.autograd.gradcheck import gradcheck, gradgradcheck


# Tests that apply to all operators

class TestOpInfo(TestCase):
    exact_dtype = True

    # Verifies that ops have their unsupported dtypes
    #   registered correctly by testing that each claimed unsupported dtype
    #   throws a runtime error
    @skipCUDAIfRocm
    @onlyOnCPUAndCUDA
    @ops(op_db, unsupported_dtypes_only=True)
    def test_unsupported_dtypes(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)
        if len(samples) == 0:
            self.skipTest("Skipped! No sample inputs!")

        # NOTE: only tests on first sample
        sample = samples[0]
        with self.assertRaises(RuntimeError):
            op(sample.input, *sample.args, **sample.kwargs)

    # Verifies that ops have their supported dtypes
    #   registered correctly by testing that each claimed supported dtype
    #   does NOT throw a runtime error
    @skipCUDAIfRocm
    @onlyOnCPUAndCUDA
    @ops(op_db, all_dtypes=True)
    def test_supported_dtypes(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)
        if len(samples) == 0:
            self.skipTest("Skipped! No sample inputs!")

        # NOTE: only tests on first sample
        sample = samples[0]
        op(sample.input, *sample.args, **sample.kwargs)


class TestGradients(TestCase):
    exact_dtype = True

    # Copies inputs to inplace operations to avoid inplace modifications
    #   to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    def _check_helper(self, device, dtype, op, variant, check):
        if variant is None:
            self.skipTest("Skipped! Variant not implemented.")
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest(f"Skipped! {op.name} does not support dtype {str(dtype)}")

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for sample in samples:
            partial_fn = partial(variant, **sample.kwargs)
            if check == 'gradcheck':
                self.assertTrue(gradcheck(partial_fn, (sample.input,) + sample.args,
                                          check_grad_dtypes=True))
            elif check == 'gradgradcheck':
                self.assertTrue(gradgradcheck(partial_fn, (sample.input,) + sample.args,
                                              gen_non_contig_grad_outputs=False,
                                              check_grad_dtypes=True))
                self.assertTrue(gradgradcheck(partial_fn, (sample.input,) + sample.args,
                                              gen_non_contig_grad_outputs=True,
                                              check_grad_dtypes=True))
            else:
                self.assertTrue(False, msg="Unknown check requested!")

    def _grad_test_helper(self, device, dtype, op, variant):
        return self._check_helper(device, dtype, op, variant, 'gradcheck')

    def _gradgrad_test_helper(self, device, dtype, op, variant):
        return self._check_helper(device, dtype, op, variant, 'gradgradcheck')

    # Tests that gradients are computed correctly
    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_fn_grad(self, device, dtype, op):
        self._grad_test_helper(device, dtype, op, op.get_op())

    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_method_grad(self, device, dtype, op):
        self._grad_test_helper(device, dtype, op, op.get_method())

    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_inplace_grad(self, device, dtype, op):
        if not op.test_inplace_grad:
            self.skipTest("Skipped! Inplace gradcheck marked to skip.")
        self._grad_test_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()))

    # Test that gradients of gradients are computed correctly
    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_fn_gradgrad(self, device, dtype, op):
        self._gradgrad_test_helper(device, dtype, op, op.get_op())

    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_method_gradgrad(self, device, dtype, op):
        self._gradgrad_test_helper(device, dtype, op, op.get_method())

    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_inplace_gradgrad(self, device, dtype, op):
        if not op.test_inplace_grad:
            self.skipTest("Skipped! Inplace gradgradcheck marked to skip.")
        self._gradgrad_test_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()))


class TestOut(TestCase):
    exact_dtype = True

    @ops(op_db)
    def test_out(self, device, dtype, op):
        if not op.supports_tensor_out:
            self.skipTest("Skipped! Operator %s does not support out=..." % op.name)
        samples = op.sample_inputs(device, dtype)
        if len(samples) == 0:
            self.skipTest("Skipped! No sample inputs!")

        # NOTE: only tests on first sample
        sample = samples[0]
        # call it normally to get the expected result
        expected = op(sample.input, *sample.args, **sample.kwargs)
        # call it with out=... and check we get the expected result
        out_kwargs = sample.kwargs.copy()
        out_kwargs['out'] = out = torch.empty_like(expected)
        op(sample.input, *sample.args, **out_kwargs)
        self.assertEqual(expected, out)


instantiate_device_type_tests(TestOpInfo, globals())
instantiate_device_type_tests(TestGradients, globals())
instantiate_device_type_tests(TestOut, globals())

if __name__ == '__main__':
    run_tests()
