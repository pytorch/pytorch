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


def apply_masked_reduction_along_dim(op, input, *args, **kwargs):
    """Applies reduction op along given dimension to strided x
    elements that are valid according to mask tensor.
    """
    mask = kwargs.pop('mask', None)
    dim_pos = kwargs.pop('dim_position', 0)
    if input.ndim == 0:
        # scalar input
        return op(input, *args, **kwargs)
    keepdim = kwargs.pop('keepdim', False)
    dtype = kwargs.get('dtype', input.dtype)
    if dim_pos < len(args):
        assert 'dim' not in kwargs, (args, kwargs)
        dim = args[dim_pos]
        args0 = args[:dim_pos] + (None,) + args[dim_pos + 1:]
    else:
        dim = kwargs.pop('dim', None)
        args0 = args
    dim_ = torch._masked._canonical_dim(dim, input.ndim)
    inpmask = torch._masked._input_mask(input, mask=mask)
    ranges = []
    shape = []
    for i in range(input.ndim):
        if i in dim_:
            ranges.append((slice(None),))
            shape.append(1)
        else:
            ranges.append(range(input.shape[i]))
            shape.append(input.shape[i])
    output = input.new_full(shape, float('nan') if dtype.is_floating_point else 0, dtype=dtype)
    for s in itertools.product(*ranges):
        data = input[s].flatten()[inpmask[s].flatten().argwhere()]
        if not data.numel():
            continue
        output[s][0] = op(data, *args0, **kwargs)
    if not keepdim:
        shape = [shape[i] for i in range(len(shape)) if i not in dim_]
        output = output.reshape(shape)
    return output


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
    norm=lambda *args, **kwargs: apply_masked_reduction_along_dim(torch.linalg.vector_norm, *args, **dict(kwargs, dim_position=1)),
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
