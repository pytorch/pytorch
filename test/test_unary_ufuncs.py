import torch

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
from torch.testing._internal.common_methods_invocations import \
    (unary_ufuncs)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, onlyOnCPUAndCUDA, skipCUDAIfRocm)


# Tests for unary "universal functions (ufuncs)" that accept a single
# tensor and have common properties like:
#   - they are elementwise functions
#   - the input shape is the output shape
#   - they typically have method and inplace variants
#   - they typically support the out kwarg
#   - they typically have NumPy or SciPy references

# See NumPy's universal function documentation
# (https://numpy.org/doc/1.18/reference/ufuncs.html) for more details
# about the concept of ufuncs.
class TestUnaryUfuncs(TestCase):
    exact_dtype = True

    # Verifies that the unary ufuncs have their supported dtypes
    #   registered correctly by testing that each unlisted dtype
    #   throws a runtime error
    @skipCUDAIfRocm
    @onlyOnCPUAndCUDA
    @ops(unary_ufuncs, unsupported_dtypes_only=True)
    def test_unsupported_dtypes(self, device, dtype, op):
        t = torch.empty(1, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            op(t)


instantiate_device_type_tests(TestUnaryUfuncs, globals())

if __name__ == '__main__':
    run_tests()
