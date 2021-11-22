"""Tests for masked operations.
"""

import itertools
import torch

from torch.testing._internal.common_utils import \
    (TestCase, suppress_warnings)
from torch.testing._internal.common_methods_invocations import \
    (op_db,)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, onlyNativeDeviceTypes)


def apply_masked_normalization_along_dim(op, x, dim, dtype=None, mask=None):
    """Applies normalization op along given dimension to strided x
    elements that are valid according to mask tensor.
    """
    if x.ndim == 0:  # scalar input
        return op(x, dim, dtype=dtype)
    y = torch.zeros_like(x, dtype=dtype)
    inpmask = torch._masked._input_mask(x, mask=mask)
    dim_ = dim % x.ndim
    left_ranges = tuple(map(range, x.shape[:dim_]))
    right_ranges = tuple(map(range, x.shape[dim_ + 1:]))
    for s in itertools.product(*(left_ranges + ((slice(None),),) + right_ranges)):
        indices = inpmask[s].argwhere()
        y[s][indices] = op(x[s][indices], 0, dtype=dtype)
    return y


reference_functions = dict(
    softmax=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.softmax, *args, **kwargs),
    log_softmax=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.log_softmax, *args, **kwargs),
    softmin=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.nn.functional.softmin, *args, **kwargs),
)

masked_ops = [op for op in op_db if op.name.startswith('_masked.')]
masked_ops_with_references = [op for op in masked_ops if op.name.rsplit('.', 1)[-1] in reference_functions]


class TestMasked(TestCase):

    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(masked_ops_with_references)
    def test_reference_masked(self, device, dtype, op):
        ref_op = reference_functions[op.name.rsplit('.', 1)[-1]]
        sample_inputs = op.sample_inputs(device, dtype)
        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
            actual = op.op(t_inp, *t_args, **t_kwargs)
            expected = ref_op(t_inp, *t_args, **t_kwargs)
            outmask = torch._masked._output_mask(op.op, t_inp, *t_args, **t_kwargs)
            actual = torch.where(outmask, actual, actual.new_zeros([]))
            expected = torch.where(outmask, expected, expected.new_zeros([]))
            self.assertEqual(actual, expected, exact_device=False)


instantiate_device_type_tests(TestMasked, globals())
