from functools import partial, wraps
from copy import deepcopy
from itertools import product

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

    # Copies inputs to inplace operations to avoid inplace modifications
    #   to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    # Tests that gradients are computed correctly
    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_grad(self, device, dtype, op):
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest("Skipped!")

        variants = (op.get_op(),
                    op.get_method(),
                    self._get_safe_inplace(op.get_inplace()))
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for variant, sample in product(variants, samples):
            sample_copy = deepcopy(sample)
            partial_fn = partial(variant, **sample_copy.kwargs)
            self.assertTrue(gradcheck(partial_fn, (sample_copy.input,) + sample_copy.args))

    # Test that gradients of gradients are computed correctly
    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_gradgrad(self, device, dtype, op):
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest("Skipped!")

        variants = (op.get_op(),
                    op.get_method(),
                    self._get_safe_inplace(op.get_inplace()))
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for variant, sample in product(variants, samples):
            sample_copy = deepcopy(sample)
            partial_fn = partial(variant, **sample.kwargs)
            self.assertTrue(gradgradcheck(partial_fn, (sample.input,) + sample.args,
                                          gen_non_contig_grad_outputs=False))
            self.assertTrue(gradgradcheck(partial_fn, (sample.input,) + sample.args,
                                          gen_non_contig_grad_outputs=True))


instantiate_device_type_tests(TestOpInfo, globals())
instantiate_device_type_tests(TestGradients, globals())

if __name__ == '__main__':
    run_tests()
