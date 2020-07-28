import torch

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
from torch.testing._internal.common_methods_invocations import \
    (unary_ufuncs)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, onlyOnCPUAndCUDA)


# Returns a tensor of the requested shape, dtype, and device
# Initialization uses randint for non-float types and randn for float types.
def _make_tensor(shape, dtype, device, fill_ones=False) -> torch.Tensor:
    # Returns a tensor filled with ones
    if fill_ones:
        return torch.ones(*shape, dtype=_convert_t(dtype, device), device=device)

    # Returns a tensor with random integer values
    if not (dtype.is_floating_point or dtype.is_complex):
        t = torch.randint(0, 10, shape, device=device)
        if dtype != torch.uint8:
            t = t - 5  # generate negative values also
        return t.to(_convert_t(dtype, device))

    # Populates the CPU tensor with floats representable as half/bfloat16
    if dtype == torch.half and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).half()
    if dtype == torch.bfloat16 and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).bfloat16()

    # Default: returns a tensor with random float values
    return torch.randn(shape, dtype=dtype, device=device)

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
    @onlyOnCPUAndCUDA
    @ops(unary_ufuncs, unsupported_dtypes_only=True)
    def test_unsupported_dtypes(self, device, dtype, op_meta):
        t = torch.empty(1, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            op = op_meta.getOp()
            op(t)

    @ops(unary_ufuncs)
    def test_non_contig(self, device, dtype, op_meta):
        op = op_meta.getOp()

        shapes = [(5, 7), (1024,)]
        for shape in shapes:
            contig = _make_tensor(shape, device=device, dtype=dtype)
            non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
            non_contig.copy_(contig)
            self.assertFalse(non_contig.is_contiguous())
            self.assertEqual(op(contig), op(non_contig))


instantiate_device_type_tests(TestUnaryUfuncs, globals())

if __name__ == '__main__':
    run_tests()
