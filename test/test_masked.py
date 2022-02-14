# Owner(s): ["module: masked operators"]

"""Tests for masked operations.
"""

import itertools
import torch
from typing import List, Any

from torch.testing._internal.common_utils import \
    (TestCase, suppress_warnings)
from torch.testing._internal.common_methods_invocations import \
    (op_db,)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, onlyNativeDeviceTypes)


def apply_masked_reduction_along_dim(op, input, *args, **kwargs):
    """Applies reduction op along given dimension to strided x
    elements that are valid according to mask tensor.

    The op is applied to each elementary slice of input with args and
    kwargs with the following constraints:

    1. Prior applying the op:

      A. if kwargs contains an item with key 'dim_position' then it is
         removed from kwargs. The value of 'dim_position' is an
         integer that describes the dim argument position: while
         typically the dim argument appears at the 0-th position of
         the op arguments (excluding input), for instance, sum(input,
         dim), then there exists reductions that have extra arguments
         prior the dim argument, for instance, norm(input, ord, dim).

      B. if args or kwargs contains dim or keepdim arguments, these
         will be removed or replaced with None so that the op is
         applied to elementary slice using the default dim and keepdim
         value.

    2. The elementary slice of the input is defined as the flattened
      slice that has no masked out elements and when op is applied,
      the result will be a scalar value (assuming keepdim=False). For
      example, an input tensor to a reduction operation op having
      dim=0 and keepdim=True argument:

       [[1 * 2 * *]
        [* 3 4 * 5]]

      (* denotes masked out elements) has the following elementary
      slices: [1, 2] and [3, 4, 5]. The result of
      apply_masked_reduction_along_dim is

       [[op([1, 2], *args0, **kwargs, dim=None, keepdim=False)]
        [op([3, 4, 5], *args0, **kwargs, dim=None, keepdim=False)]]

      where args0 is args where dim value is replased with None if
      present.

      Using the same example data, if the op is called with dim=(0, 1)
      and keepdim=False, there is one elementary slice: [1, 2, 3, 4,
      5]; and the corresponding result of the op is:

        op([1, 2, 3, 4, 5], *args0, **kwargs, dim=None, keepdim=False)

    3. If the elementary slice is empty, the corresponding output
      value is nan if dtype is float, otherwise, 0.  An empty
      elementary slice corresponds to fully masked-out output, so, the
      corresponding specific value of the output will not be important
      because we used masked equality check for comparing the results
      of masked operations.
    """
    # eliminate mask and dim_position keyword arguments:
    mask = kwargs.pop('mask', None)
    dim_pos = kwargs.pop('dim_position', 0)

    dtype = kwargs.get('dtype', input.dtype)
    if input.ndim == 0:
        # scalar input is an elementary slice
        return op(input, *args, **kwargs).to(dtype=dtype)

    # eliminate keepdim keyword argument if specified:
    keepdim = kwargs.pop('keepdim', False)

    # eliminate dim argument that may appear both as args or kwargs
    # element:
    if dim_pos < len(args):
        # dim is specified in args
        assert 'dim' not in kwargs, (args, kwargs)
        dim = args[dim_pos]
        args0 = args[:dim_pos] + (None,) + args[dim_pos + 1:]
    else:
        # dim may be specified in kwargs
        dim = kwargs.pop('dim', None)
        args0 = args

    # dimensions along which the reduction operation is applied:
    dim_ = torch._masked._canonical_dim(dim, input.ndim)
    # slices in product(*ranges) define all elementary slices:
    ranges: List[Any] = []
    # shape of output for the keepdim=True case:
    shape = []
    for i in range(input.ndim):
        if i in dim_:
            ranges.append((slice(None),))
            shape.append(1)
        else:
            ranges.append(range(input.shape[i]))
            shape.append(input.shape[i])

    # keepdim=True version of the output, filled with nan or 0:
    output = input.new_full(shape, float('nan') if dtype.is_floating_point else 0, dtype=dtype)

    # apply op to all elementary slices:
    inpmask = torch._masked._input_mask(input, mask=mask)
    for s in itertools.product(*ranges):
        # data of an elementary slice is 1D sequence and has only
        # masked-in elements:
        data = input[s].flatten()[inpmask[s].flatten().argwhere()]
        if not data.numel():
            # empty elementary slice
            continue
        output[s][0] = op(data, *args0, **kwargs)

    if not keepdim:
        # reshape output for the keepdim=False case
        shape = [shape[i] for i in range(len(shape)) if i not in dim_]
        output = output.reshape(shape)
    return output


def apply_masked_normalization_along_dim(op, input, *args, **kwargs):
    """Applies normalization op along given dimension to strided x
    elements that are valid according to mask tensor.
    """
    mask = kwargs.pop('mask', None)
    dim_pos = kwargs.pop('dim_position', 0)
    if input.ndim == 0:  # scalar input
        return op(input, *args, **kwargs)
    dtype = kwargs.get('dtype', input.dtype)
    dim = args[dim_pos]
    args0 = args[:dim_pos] + (0,) + args[dim_pos + 1:]
    output = torch.zeros_like(input, dtype=dtype)
    inpmask = torch._masked._input_mask(input, mask=mask)
    dim_ = dim % input.ndim
    left_ranges = tuple(map(range, input.shape[:dim_]))
    right_ranges = tuple(map(range, input.shape[dim_ + 1:]))
    for s in itertools.product(*(left_ranges + ((slice(None),),) + right_ranges)):
        indices = inpmask[s].argwhere()
        output[s][indices] = op(input[s][indices], *args0, **kwargs)
    return output


reference_functions = dict(
    norm=lambda *args, **kwargs: apply_masked_reduction_along_dim(torch.linalg.vector_norm, *args, **dict(kwargs, dim_position=1)),
    var=lambda *args, **kwargs: apply_masked_reduction_along_dim(torch.var, *args, **dict(kwargs, dim_position=0)),
    softmax=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.softmax, *args, **kwargs),
    log_softmax=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.log_softmax, *args, **kwargs),
    softmin=lambda *args, **kwargs: apply_masked_normalization_along_dim(torch.nn.functional.softmin, *args, **kwargs),
    normalize=lambda *args, **kwargs: apply_masked_normalization_along_dim(
        torch.nn.functional.normalize, *args, **dict(kwargs, dim_position=1)),
)

masked_ops = [op for op in op_db if op.name.startswith('_masked.')]
masked_ops_with_references = [op for op in masked_ops if op.name.rsplit('.', 1)[-1] in reference_functions]
masked_ops_with_mixed_layouts_support = [op for op in masked_ops if op.supports_sparse or op.supports_sparse_csr]


def _tensor_to_strided(obj):
    # after gh-59958 is resolved, replace the usage of this function
    # with torch.Tensor.to_dense
    if obj.layout == torch.strided:
        return obj
    return obj.to_dense()


def to_strided(obj):
    """Convert the tensor content of object to strided tensor content.
    """
    return torch.utils._pytree.tree_map(_tensor_to_strided, obj)


class TestMasked(TestCase):

    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(masked_ops_with_references)
    def test_reference_masked(self, device, dtype, op):
        op_name = op.name.rsplit('.', 1)[-1]
        ref_op = reference_functions[op_name]
        sample_inputs = op.sample_inputs(device, dtype)
        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
            if op_name == 'var' and not (t_inp.dtype.is_floating_point or t_inp.dtype.is_complex):
                # torch.var does not support integer inputs
                continue
            actual = op.op(t_inp, *t_args, **t_kwargs)
            expected = ref_op(t_inp, *t_args, **t_kwargs)
            outmask = torch._masked._output_mask(op.op, t_inp, *t_args, **t_kwargs)
            actual = torch.where(outmask, actual, actual.new_zeros([]))
            expected = torch.where(outmask, expected, expected.new_zeros([]))
            self.assertEqual(actual, expected, exact_device=False)

    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(masked_ops_with_mixed_layouts_support)
    def test_mixed_layouts(self, device, dtype, op):
        sample_inputs = op.sample_inputs_mixed(device, dtype)
        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
            actual = op.op(t_inp, *t_args, **t_kwargs)

            # the first input defines the result layout
            assert actual.layout == t_inp.layout

            # check masked invariance:
            #  op(inp, mask).to_dense() == op(inp.to_dense(), mask.to_dense()) at outmask
            #
            r_inp, r_args, r_kwargs = map(to_strided, (t_inp, t_args, t_kwargs))
            outmask = torch._masked._output_mask(op.op, r_inp, *r_args, **r_kwargs)
            strided = to_strided(actual)
            strided = torch.where(outmask, strided, strided.new_zeros([]))
            reference = op.op(r_inp, *r_args, **r_kwargs)
            reference = torch.where(outmask, reference, reference.new_zeros([]))
            self.assertEqual(strided, reference, exact_device=False)


instantiate_device_type_tests(TestMasked, globals())
