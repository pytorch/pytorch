from functools import partial
from copy import deepcopy

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
    @ops(op_db)
    def test_supported_dtypes(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)
        if len(samples) == 0:
            self.skipTest("Skipped! No sample inputs!")

        # NOTE: only tests on first sample
        sample = samples[0]
        op(sample.input, *sample.args, **sample.kwargs)


class TestGradients(TestCase):
    exact_dtype = True

    # Tests that gradients are computed correctly
    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_grad(self, device, dtype, op):
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest("Skipped!")

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for sample in samples:
            with self.subTest(sample=sample):
                partial_fn = partial(op, **sample.kwargs)
                self.assertTrue(gradcheck(partial_fn, (sample.input,) + sample.args))

            # NOTE: the following prevents the test from reporting a CUDA memory leak
            # TODO: extend to multiple input tensors requiring grad
            del sample.input.grad
            del sample.input

    # Test that gradients of gradients are computed correctly
    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_gradgrad(self, device, dtype, op):
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest("Skipped!")

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for sample in samples:
            with self.subTest(sample=sample):
                partial_fn = partial(op, **sample.kwargs)
                self.assertTrue(gradgradcheck(partial_fn, (sample.input,) + sample.args,
                                              gen_non_contig_grad_outputs=False))
                self.assertTrue(gradgradcheck(partial_fn, (sample.input,) + sample.args,
                                              gen_non_contig_grad_outputs=True))

                # NOTE: the following prevents the test from reporting a CUDA memory leak
                # TODO: extend to multiple input tensors requiring grad
                del sample.input.grad
                del sample.input

    # Compares gradients of functions and their inplace variants
    @dtypes(torch.double, torch.cdouble)
    @ops((op for op in op_db if op.get_inplace() is not None))
    def test_grad_variants(self, device, dtype, op):
        inplace = op.get_inplace()
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for sample in samples:
            inplace_input = deepcopy(sample.input)

            output = op(sample.input, *sample.args, **sample.kwargs)

            # NOTE: inplace inputs are cloned because leaves requiring
            #   requiring gradients cann't be transformed inplace
            inplace_input_clone = inplace_input.clone()
            inplace_output = inplace(inplace_input_clone, *sample.args, **sample.kwargs)

            t = torch.randn_like(output)
            output.backward(t)
            inplace_output.backward(t)

            self.assertEqual(sample.input.grad, inplace_input.grad)
            self.assertEqual(sample.input.dtype, sample.input.grad.dtype)


instantiate_device_type_tests(TestOpInfo, globals())
instantiate_device_type_tests(TestGradients, globals())

if __name__ == '__main__':
    run_tests()
