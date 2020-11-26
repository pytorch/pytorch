from functools import reduce
from operator import mul, itemgetter
import collections

import torch
import numpy as np
from torch._six import inf, istuple
from torch.autograd import Variable

from typing import List, Tuple, Dict, Any

from torch.testing import \
    (make_non_contiguous, _dispatch_dtypes,
     floating_types, floating_types_and, floating_and_complex_types,
     floating_and_complex_types_and, all_types_and_complex_and, all_types_and)
from torch.testing._internal.common_device_type import \
    (skipCUDAIfNoMagma, skipCPUIfNoLapack,
     expectedAlertNondeterministic, precisionOverride)
from torch.testing._internal.common_utils import \
    (prod_single_zero, random_square_matrix_of_rank,
     random_symmetric_matrix, random_symmetric_psd_matrix,
     random_symmetric_pd_matrix, make_nonzero_det,
     random_fullrank_matrix_distinct_singular_value, set_rng_seed,
     TEST_WITH_ROCM, IS_WINDOWS, IS_MACOS, make_tensor)


class SkipInfo(object):
    """Describes which test, or type of tests, should be skipped when testing
       an operator. Any test that matches all provided arguments will be skipped.
       The skip will only be checked if the active_if argument is True."""

    __slots__ = ['cls_name', 'test_name', 'device_type', 'dtypes', 'active_if']

    def __init__(self, cls_name=None, test_name=None, *,
                 device_type=None, dtypes=None, active_if=True):
        self.cls_name = cls_name
        self.test_name = test_name
        self.device_type = device_type
        self.dtypes = dtypes
        self.active_if = active_if

class SampleInput(object):
    """Represents sample inputs to a function."""

    __slots__ = ['input', 'args', 'kwargs']

    def __init__(self, input, *, args=tuple(), kwargs=None):
        self.input = input
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}


# Classes and methods for the operator database
class OpInfo(object):
    """Operator information and helper functions for acquiring it."""

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 op=None,  # the function variant of the operation, populated as torch.<name> if None
                 dtypes=floating_types(),  # dtypes this function is expected to work with
                 dtypesIfCPU=None,  # dtypes this function is expected to work with on CPU
                 dtypesIfCUDA=None,  # dtypes this function is expected to work with on CUDA
                 dtypesIfROCM=None,  # dtypes this function is expected to work with on ROCM
                 tested_dtypes=None,  # dtypes to test with by default
                 test_inplace_grad=True,  # whether to gradcheck and gradgradcheck the inplace variant
                 supports_tensor_out=True,  # whether the op supports the out kwarg, returning a Tensor
                 skips=tuple(),  # information about which tests to skip
                 decorators=None):  # decorators to apply to generated tests

        # Validates the dtypes are generated from the dispatch-related functions
        for dtype_list in (dtypes, dtypesIfCPU, dtypesIfCUDA, dtypesIfROCM):
            assert isinstance(dtype_list, (_dispatch_dtypes, type(None)))

        self.name = name

        self.dtypes = set(dtypes)
        self.dtypesIfCPU = set(dtypesIfCPU) if dtypesIfCPU is not None else self.dtypes
        self.dtypesIfCUDA = set(dtypesIfCUDA) if dtypesIfCUDA is not None else self.dtypes
        self.dtypesIfROCM = set(dtypesIfROCM) if dtypesIfROCM is not None else self.dtypes
        self._tested_dtypes = set(tested_dtypes) if tested_dtypes is not None else None

        # NOTE: if the op is unspecified it is assumed to be under the torch namespace
        if op is None:
            assert hasattr(torch, self.name), f"Can't find torch.{self.name}"
        self.op = op if op else getattr(torch, self.name)
        self.method_variant = getattr(torch.Tensor, name) if hasattr(torch.Tensor, name) else None
        inplace_name = name + "_"
        self.inplace_variant = getattr(torch.Tensor, inplace_name) if hasattr(torch.Tensor, name) else None

        self.test_inplace_grad = test_inplace_grad
        self.supports_tensor_out = supports_tensor_out

        self.skips = skips
        self.decorators = decorators

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    def get_op(self):
        """Returns the function variant of the operator, torch.<op_name>."""
        return self.op

    def get_method(self):
        """Returns the method variant of the operator, torch.Tensor.<op_name>.
        Returns None if the operator has no method variant.
        """
        return self.method_variant

    def get_inplace(self):
        """Returns the inplace variant of the operator, torch.Tensor.<op_name>_.
        Returns None if the operator has no inplace variant.
        """
        return self.inplace_variant

    def sample_inputs(self, device, dtype, requires_grad=False):
        """Returns an iterable of SampleInputs."""
        return tuple()

    # Returns True if the test should be skipped and False otherwise
    def should_skip(self, cls_name, test_name, device_type, dtype):
        for si in self.skips:
            if not si.active_if:
                continue

            cls_name_match = si.cls_name is None or cls_name == si.cls_name
            name_match = si.test_name is None or test_name == si.test_name
            device_type_match = si.device_type is None or device_type == si.device_type
            dtype_match = si.dtypes is None or dtype in si.dtypes
            if cls_name_match and name_match and device_type_match and dtype_match:
                return True

        return False

    def supported_dtypes(self, device_type):
        if device_type == 'cpu':
            return self.dtypesIfCPU
        if device_type == 'cuda':
            return self.dtypesIfROCM if TEST_WITH_ROCM else self.dtypesIfCUDA
        else:
            return self.dtypes


    def supports_dtype(self, dtype, device_type):
        return dtype in self.supported_dtypes(device_type)

    def tested_dtypes(self, device_type):
        supported = self.supported_dtypes(device_type)
        return (supported if self._tested_dtypes is None
                else supported.intersection(self._tested_dtypes))


L = 20
M = 10
S = 5


# Metadata class for unary "universal functions (ufuncs)" that accept a single
# tensor and have common properties like:
class UnaryUfuncInfo(OpInfo):
    """Operator information for 'universal unary functions (unary ufuncs).'
    These are functions of a single tensor with common properties like:
      - they are elementwise functions
      - the input shape is the output shape
      - they typically have method and inplace variants
      - they typically support the out kwarg
      - they typically have NumPy or SciPy references
    See NumPy's universal function documentation
    (https://numpy.org/doc/1.18/reference/ufuncs.html) for more details
    about the concept of ufuncs.
    """

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref,  # a reference function
                 dtypes=floating_types(),
                 dtypesIfCPU=floating_and_complex_types_and(torch.bfloat16),
                 dtypesIfCUDA=floating_and_complex_types_and(torch.half),
                 dtypesIfROCM=floating_types_and(torch.half),
                 domain=(None, None),  # the [low, high) domain of the function
                 handles_large_floats=True,  # whether the op correctly handles large float values (like 1e20)
                 handles_extremals=True,  # whether the op correctly handles extremal values (like inf)
                 handles_complex_extremals=True,  # whether the op correct handles complex extremals (like inf -infj)
                 promotes_integers_to_float=False,  # whether op promotes unary output to float or not
                 **kwargs):
        super(UnaryUfuncInfo, self).__init__(name,
                                             dtypes=dtypes,
                                             dtypesIfCPU=dtypesIfCPU,
                                             dtypesIfCUDA=dtypesIfCUDA,
                                             dtypesIfROCM=dtypesIfROCM,
                                             **kwargs)
        self.ref = ref
        self.domain = domain
        self.handles_large_floats = handles_large_floats
        self.handles_extremals = handles_extremals
        self.handles_complex_extremals = handles_complex_extremals
        self.promotes_integers_to_float = promotes_integers_to_float

        # Epsilon to ensure grad and gradgrad checks don't test values
        #   outside a function's domain.
        self._domain_eps = 1e-5

    def sample_inputs(self, device, dtype, requires_grad=False):
        low, high = self.domain
        low = low if low is None else low + self._domain_eps
        high = high if high is None else high - self._domain_eps

        return (SampleInput(make_tensor((L,), device, dtype,
                                        low=low, high=high,
                                        requires_grad=requires_grad)),)



# Operator database (sorted alphabetically)
op_db = [
    # NOTE: CPU complex acos produces incorrect outputs (https://github.com/pytorch/pytorch/issues/42952)
    UnaryUfuncInfo('acos',
                   ref=np.arccos,
                   domain=(-1, 1),
                   handles_complex_extremals=False,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-1,
                                                  torch.complex64: 1e-2}),),
                   promotes_integers_to_float=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.float16],
                                active_if=TEST_WITH_ROCM),
                       SkipInfo('TestGradients', 'test_fn_grad',
                                dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_method_grad',
                                dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_inplace_grad',
                                dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                   )),
    # NOTE: the derivative for inplace acosh is not implemented
    UnaryUfuncInfo('acosh',
                   ref=np.arccosh,
                   domain=(1, float('inf')),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   promotes_integers_to_float=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   test_inplace_grad=False),
    UnaryUfuncInfo('asin',
                   ref=np.arcsin,
                   domain=(-1, 1),
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    # NOTE: derivative for inplace asinh is not implemented
    UnaryUfuncInfo('asinh',
                   ref=np.arcsinh,
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   promotes_integers_to_float=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   test_inplace_grad=False),
    UnaryUfuncInfo('atan',
                   ref=np.arctan,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   promotes_integers_to_float=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('atanh',
                   ref=np.arctanh,
                   domain=(-1, 1),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   promotes_integers_to_float=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   test_inplace_grad=False),
    UnaryUfuncInfo('cos',
                   ref=np.cos,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   handles_large_floats=False,
                   promotes_integers_to_float=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics', device_type='cpu',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.float], active_if=TEST_WITH_ROCM),
                   )),
    UnaryUfuncInfo('cosh',
                   ref=np.cosh,
                   dtypesIfCPU=floating_and_complex_types(),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics', device_type='cpu',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                   )),
    UnaryUfuncInfo('log',
                   ref=np.log,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   promotes_integers_to_float=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('log10',
                   ref=np.log10,
                   domain=(0, float('inf')),
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   promotes_integers_to_float=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('log1p',
                   ref=np.log1p,
                   domain=(-1, float('inf')),
                   dtypesIfCPU=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   decorators=(precisionOverride({torch.bfloat16: 1e-1}),)),
    UnaryUfuncInfo('log2',
                   ref=np.log2,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   promotes_integers_to_float=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble]),
                   )),
    UnaryUfuncInfo('neg',
                   ref=np.negative,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16)),
    UnaryUfuncInfo('sin',
                   ref=np.sin,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   handles_large_floats=False,
                   handles_complex_extremals=False,
                   promotes_integers_to_float=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.float], active_if=TEST_WITH_ROCM),
                   )),
    UnaryUfuncInfo('sinh',
                   ref=np.sinh,
                   dtypesIfCPU=floating_and_complex_types(),
                   decorators=(precisionOverride({torch.float16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('tan',
                   ref=np.tan,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   promotes_integers_to_float=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.float64],
                                active_if=TEST_WITH_ROCM),
                   )),
    UnaryUfuncInfo('tanh',
                   ref=np.tanh,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   promotes_integers_to_float=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                   )),
    UnaryUfuncInfo('exp2',
                   ref=np.exp2,
                   dtypes=floating_types_and(torch.half),
                   dtypesIfCPU=None,
                   dtypesIfCUDA=None),
    UnaryUfuncInfo('nan_to_num',
                   ref=np.nan_to_num,
                   dtypes=all_types_and(torch.half, torch.bool),
                   dtypesIfCPU=None,
                   dtypesIfCUDA=None),
    UnaryUfuncInfo('sqrt',
                   ref=np.sqrt,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   decorators=(precisionOverride({torch.bfloat16: 7e-2}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/47358
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_MACOS),
                       # Reference: https://github.com/pytorch/pytorch/pull/47293#issuecomment-721774436
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.bfloat16]),
                       # RuntimeError: sqrt does not support automatic differentiation for outputs with complex dtype.
                       SkipInfo('TestGradients', 'test_fn_grad',
                                dtypes=[torch.cdouble]),
                       SkipInfo('TestGradients', 'test_fn_gradgrad',
                                dtypes=[torch.cdouble]),
                       SkipInfo('TestGradients', 'test_method_grad',
                                dtypes=[torch.cdouble]),
                       SkipInfo('TestGradients', 'test_method_gradgrad',
                                dtypes=[torch.cdouble]),
                       SkipInfo('TestGradients', 'test_inplace_grad',
                                dtypes=[torch.cdouble]),
                       SkipInfo('TestGradients', 'test_inplace_gradgrad',
                                dtypes=[torch.cdouble]),),
                   promotes_integers_to_float=True,
                   handles_complex_extremals=False),
]

# Common operator groupings
unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo)]

def index_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = torch.rand(*shape).mul_(max_indices).floor_().long()
    return index


def index_perm_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)

    index = torch.randperm(max_indices).narrow(0, 0, reduce(mul, shape)).view(shape)
    return index


def gather_variable(shape, index_dim, max_indices, duplicate=False):
    assert len(shape) == 2
    assert index_dim < 2
    batch_dim = 1 - index_dim
    index = torch.LongTensor(*shape)
    for i in range(shape[index_dim]):
        index.select(index_dim, i).copy_(
            torch.randperm(max_indices)[:shape[batch_dim]])
    if duplicate:
        index.select(batch_dim, 0).copy_(index.select(batch_dim, 1))
    return index


def bernoulli_scalar():
    return torch.tensor(0, dtype=torch.bool).bernoulli_()


def mask_not_all_zeros(shape):
    assert len(shape) > 0
    while True:
        result = torch.randn(shape).gt(0)
        if result.sum() > 0:
            return result


def uniform_scalar(offset=0, requires_grad=False):
    v = torch.rand(()) + offset
    v.requires_grad = requires_grad
    return v


def normal_scalar_clamp(amin, amax, requires_grad=False):
    v = torch.randn(()).clamp(amin, amax)
    v.requires_grad = requires_grad
    return v


def prod_zeros(dim_size, dim_select):
    assert len(dim_select) == 2
    result = torch.randn(dim_size, dim_size, dim_size)
    result.narrow(dim_select[0], 0, 1).narrow(dim_select[1], 1, 1).zero_()
    result.narrow(dim_select[0], 2, 1).narrow(dim_select[1], 3, 1).zero_()
    result.narrow(dim_select[0], 4, 1).narrow(dim_select[1], 3, 1).zero_()
    return result


non_differentiable = collections.namedtuple('non_differentiable', ['tensor'])


class dont_convert(tuple):
    pass


class NoArgsClass(object):
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration()
    next = __next__  # Python 2 compatibility

    def __len__(self):
        return 0

NO_ARGS = NoArgsClass()

def ident(x):
    return x

# (
#   method name,
#   input size/constructing fn,
#   args (tuple represents shape of a tensor arg),
#   test variant name (will be used at test name suffix),    // optional
#   (should_check_autodiff[bool], nonfusible_nodes, fusible_nodes) for autodiff, // optional
#   indices for possible dim arg,                            // optional
#   fn mapping output to part that should be gradcheck'ed,   // optional
#   kwargs                                                   // optional
# )
# Note: some functions have separate schema for (Tensor other) and (Scalar other),
#       and it's possible that we only support AD for Scalar version but not Tensor
#       version, and vice versa.
#       When writing tests, only scalar(float/int) input triggers the Scalar schema.
#       uniform_scalar produces a scalar **Tensor** which won't match Scalar input.
def method_tests():
    set_rng_seed(0)
    return [
        ('acosh', torch.rand(S, S, S).add(1), NO_ARGS, ''),
        ('acosh', torch.rand(tuple()).add(1), NO_ARGS, 'scalar'),
        ('add', (S, S, S), ((S, S, S),), '', (True,)),
        ('add', (S, S, S), ((S, S),), 'broadcast_rhs', (True,)),
        ('add', (S, S), ((S, S, S),), 'broadcast_lhs', (True,)),
        ('add', (S, 1, S), ((M, S),), 'broadcast_all', (True,)),
        ('add', (), ((),), 'scalar', (True,)),
        ('add', (S, S, S), ((),), 'scalar_broadcast_rhs', (True,)),
        ('add', (), ((S, S, S),), 'scalar_broadcast_lhs', (True,)),
        ('add', (S, S, S), (3.14,), 'constant', (True,)),
        ('add', (), (3.14,), 'scalar_constant', (True,)),
        ('add', (S, S, S), (3.14j,), 'complex_scalar_constant', (True,)),
        ('asinh', (S, S, S), NO_ARGS, ''),
        ('asinh', (), NO_ARGS, 'scalar'),
        ('atanh', torch.rand(S, S, S), NO_ARGS, ''),
        ('atanh', torch.rand(tuple()), NO_ARGS, 'scalar'),
        ('__radd__', (S, S, S), (3.14,), 'constant', (True, 'aten::add')),
        ('__radd__', (), (3.14,), 'scalar_constant', (True, 'aten::add')),
        ('sub', (S, S, S), ((S, S, S),), '', (True,)),
        ('sub', (S, S, S), ((S, S),), 'broadcast_rhs', (True,)),
        ('sub', (S, S), ((S, S, S),), 'broadcast_lhs', (True,)),
        ('sub', (S, 1, S), ((M, S),), 'broadcast_all', (True,)),
        ('sub', (S, S, S), ((),), 'scalar_broadcast_rhs', (True,)),
        ('sub', (), ((S, S, S),), 'scalar_broadcast_lhs', (True,)),
        ('sub', (S, S, S), (3.14,), 'constant', (True,)),
        ('sub', (), (3.14,), 'scalar_constant', (True,)),
        ('sub', (S, S, S), (3.14j,), 'complex_scalar_constant', (True,)),
        ('__rsub__', (S, S, S), (3.14,), 'constant', (True, 'aten::rsub')),
        ('__rsub__', (), (3.14,), 'scalar_constant', (True, 'aten::rsub')),
        ('mul', (S, S, S), ((S, S, S),), '', (True,)),
        ('mul', (), ((),), 'scalar', (True,)),
        ('mul', (S, S, S), ((S, S),), 'broadcast_rhs', (True,)),
        ('mul', (S, S), ((S, S, S),), 'broadcast_lhs', (True,)),
        ('mul', (S, 1, S), ((M, S),), 'broadcast_all', (True,)),
        ('mul', (S, S, S), ((),), 'scalar_broadcast_rhs', (True,)),
        ('mul', (), ((S, S, S),), 'scalar_broadcast_lhs', (True,)),
        ('mul', (S, S, S), (3.14,), 'constant', (True,)),
        ('mul', (), (3.14,), 'scalar_constant', (True,)),
        # TODO(@anjali411): enable these tests
        # ('mul', (S, S, S), (3.14j,), 'imaginary_constant', (True,)),
        # ('mul', (), (3.14j,), 'imaginary_scalar_constant', (True,)),
        ('__rmul__', (S, S, S), (3.14,), 'constant', (True, 'aten::mul')),
        ('__rmul__', (), (3.14,), 'scalar_constant', (True, 'aten::mul')),
        ('div', (S, S, S), (torch.rand(S, S, S) + 0.1,), '', (True,)),
        ('div', (S, S, S), (torch.rand(S, S) + 0.1,), 'broadcast_rhs', (True,)),
        ('div', (S, S), (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs', (True,)),
        ('div', (S, 1, S), (torch.rand(M, S) + 0.1,), 'broadcast_all', (True,)),
        ('div', (), (uniform_scalar(0.1),), 'scalar', (True,)),
        ('div', (S, S, S), (uniform_scalar(0.1),), 'scalar_broadcast_rhs', (True,)),
        ('div', (), (uniform_scalar(0.1),), 'scalar_broadcast_lhs', (True,)),
        ('div', torch.rand(S, S, S) + 1e-1, (3.14,), 'constant', (True,)),
        ('div', uniform_scalar(1e-1, requires_grad=True), (3.14,), 'scalar_constant', (True,)),
        ('true_divide', (S, S, S), (torch.rand(S, S, S) + 0.1,), '', (True,)),
        ('true_divide', (S, S, S), (torch.rand(S, S) + 0.1,), 'broadcast_rhs', (True,)),
        ('true_divide', (S, S), (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs', (True,)),
        ('true_divide', (S, 1, S), (torch.rand(M, S) + 0.1,), 'broadcast_all', (True,)),
        ('true_divide', (), (uniform_scalar(0.1),), 'scalar', (True,)),
        ('true_divide', (S, S, S), (uniform_scalar(0.1),), 'scalar_broadcast_rhs', (True,)),
        ('true_divide', (), (uniform_scalar(0.1),), 'scalar_broadcast_lhs', (True,)),
        ('true_divide', torch.rand(S, S, S) + 1e-1, (3.14,), 'constant', (True,)),
        ('true_divide', uniform_scalar(1e-1, requires_grad=True), (3.14,), 'scalar_constant', (True,)),
        ('__rdiv__', torch.rand(S, S, S) + 1e-1, (3.14,), 'constant',
            (True, [], ['aten::mul', 'aten::reciprocal'])),
        ('__rdiv__', uniform_scalar(1e-1, requires_grad=True), (3.14,), 'scalar_constant',
            (True, [], ['aten::mul', 'aten::reciprocal'])),
        ('__rdiv__', torch.rand(S, S, S, dtype=torch.cdouble) + 1e-1, (3.14j,), 'complex_constant',
            (True, [], ['aten::mul', 'aten::reciprocal'])),
        ('__rdiv__', uniform_scalar(1e-1 * (1 + 1j), requires_grad=True), (3.14j,), 'complex_scalar_constant',
            (True, [], ['aten::mul', 'aten::reciprocal'])),
        ('div', (S, S, S), (torch.rand(S, S, S, dtype=torch.cdouble) + 0.1,), 'complex', (True,)),
        ('div', (S, S, S), (torch.rand(S, S, dtype=torch.cdouble) + 0.1,), 'complex_broadcast_rhs', (True,)),
        ('div', (S, S), (torch.rand(S, S, S, dtype=torch.cdouble) + 0.1,), 'complex_broadcast_lhs', (True,)),
        ('div', (S, 1, S), (torch.rand(M, S, dtype=torch.cdouble) + 0.1,), 'complex_broadcast_all', (True,)),
        ('div', (), (uniform_scalar(0.1j),), 'complex_scalar', (True,)),
        ('div', (S, S, S), (uniform_scalar(0.1j),), 'complex_scalar_broadcast_rhs', (True,)),
        ('div', (), (uniform_scalar(0.1j),), 'complex_scalar_broadcast_lhs', (True,)),
        ('div', torch.rand(S, S, S, dtype=torch.cdouble) + 1e-1, (3.14j,), 'complex_constant', (True,)),
        ('div', uniform_scalar(1e-1j, requires_grad=True), (3.14j,), 'complex_scalar_constant', (True,)),
        ('pow', torch.rand(S, S, S) + 1e-3, (torch.rand(S, S, S) + 0.1,), '', (True,)),
        ('pow', torch.rand(S, S, S) + 1e-3, (torch.rand(1,) + 0.1,), 'broadcast_rhs', (True,)),
        ('pow', torch.rand(1,) + 1e-3, (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs', (True,)),
        ('pow', torch.rand(S, 1, S) + 1e-3, (torch.rand(1, S, 1) + 0.1,), 'broadcast_all', (True,)),
        ('pow', uniform_scalar(1e-3, requires_grad=True), (uniform_scalar(0.1),), 'scalar', (True,)),
        ('pow', torch.rand(S, S, S) + 1e-3, (uniform_scalar(0.1),), 'scalar_broadcast_rhs', (True,)),
        ('pow', uniform_scalar(1e-3, requires_grad=True), (torch.rand(S, S, S) + 0.1,), 'scalar_broadcast_lhs', (True,)),
        ('pow', torch.rand(S, S, S) + 1e-3, (3.14,), 'constant', (True,)),
        ('pow', torch.rand(S, S, S, dtype=torch.cdouble) + 1e-3 * (1 + 1j), (3.14,), 'complex_constant', (True,)),
        ('__rpow__', torch.rand(S, S, S) + 1e-3, (3.14,), 'constant', (True, 'aten::pow')),
        ('pow', uniform_scalar(1e-3, requires_grad=True), (3.14,), 'scalar_constant', (True,)),
        ('pow', uniform_scalar(1e-3 * (1 + 1j), requires_grad=True), (3.14,), 'complex_scalar_constant', (True,)),
        ('pow', uniform_scalar(1e-3 * (1 + 1j), requires_grad=True), (3.14j,), 'complex_imaginary_exponent', (True,)),
        ('__rpow__', uniform_scalar(1e-3, requires_grad=True), (3.14,), 'scalar_constant', (True, 'aten::pow')),
        ('transpose', (1, 2, 3), (1, 2), 'dim', (False,), [0, 1]),
        ('transpose', (), (0, 0), 'scalar', (False,)),
        ('transpose', (1,), (0, 0), '1d', (False,)),
        ('transpose', (L, L), (0, 1), '2d', (False,)),
        ('transpose', (S, S, S), (2, 0), '3d', (False,)),
        ('swapdims', (1, 2, 3), (1, 2), 'dim', (False,), [0, 1]),
        ('swapdims', (), (0, 0), 'scalar', (False,)),
        ('swapdims', (1,), (0, 0), '1d', (False,)),
        ('swapdims', (L, L), (0, 1), '2d', (False,)),
        ('swapdims', (S, S, S), (2, 0), '3d', (False,)),
        ('swapaxes', (1, 2, 3), (1, 2), 'dim', (False,), [0, 1]),
        ('swapaxes', (), (0, 0), 'scalar', (False,)),
        ('swapaxes', (1,), (0, 0), '1d', (False,)),
        ('swapaxes', (L, L), (0, 1), '2d', (False,)),
        ('swapaxes', (S, S, S), (2, 0), '3d', (False,)),
        ('t', (1, 2), NO_ARGS, '', (False,)),
        ('view', (S, S, S), (S * S, S), '', (False,)),
        ('view', (torch.Size([S * S, S]),), (S, S, S), 'size', (False,)),
        ('view', (S,), (S,), '1d', (False,)),
        ('view', (), (dont_convert(()),), 'scalar_to_scalar', (False,)),
        ('view', (), (1,), 'scalar_to_1d', (False,)),
        ('ravel', (S, S, S), NO_ARGS, '', (False,)),
        ('reshape', (S, S, S), (S * S, S), '', (False,)),
        ('reshape', (torch.Size([S * S, S]),), (S, S, S), 'size', (False,)),
        ('reshape', (S,), (S,), '1d', (False,)),
        ('reshape', (), (dont_convert(()),), 'scalar_to_scalar', (False,)),
        ('reshape', (), (1,), 'scalar_to_1d', (False,)),
        ('reshape_as', (S, S, S), (non_differentiable(torch.rand(S * S, S)),)),
        ('reshape_as', (), (non_differentiable(torch.tensor(42.)),), 'scalar'),
        ('reshape_as', (), (non_differentiable(torch.rand(1, 1)),), 'scalar_to_dims'),
        ('flip', (S, S, S), ([0],), 'd0'),
        ('flip', (S, S, S), ([0, 1, 2],), 'd012'),
        ('flip', (S, S, S), ([0, 2],), 'd02'),
        ('flip', (S, S, S), ([2, 0],), 'd20'),
        ('flip', (S, S, S), ([-1],), 'neg_d'),
        ('fliplr', (S, S, S), ()),
        ('flipud', (S, S, S), ()),
        ('roll', (S, S, S), (0, 0), 'd0'),
        ('roll', (S, S, S), (1, 2), 'd12'),
        ('roll', (S, S, S), (0, 2,), 'd02'),
        ('roll', (S, S, S), (2, 0,), 'd20'),
        ('roll', (S, S, S), (-1, 0), 'neg_shift'),
        ('roll', (S, S, S), (10000, 1), 'loop_shift'),
        ('roll', (S, S, S), (2,), 'flattened'),
        ('roll', (S, S, S), ([1, 2, -1], [0, 1, 2]), 'three_dims'),
        ('rot90', (S, S, S), (1, [0, 1],), 'k1_d01'),
        ('rot90', (S, S, S), (1, [1, 2],), 'k1_d12'),
        ('rot90', (S, S, S), (1, [1, -1],), 'k1_neg_d'),
        ('rot90', (S, S, S), (), 'default'),
        ('view_as', (S, S, S), (non_differentiable(torch.rand(S * S, S)),)),
        ('view_as', (), (non_differentiable(torch.tensor(5.5)),), 'scalar'),
        ('view_as', (), (non_differentiable(torch.rand(1, 1)),), 'scalar_to_dims'),
        ('expand', (S, 1, 1), (S, S, S), '', (False,)),
        ('expand', (torch.Size([S, 1, S]),), (S, S, S), 'size', (False,)),
        ('expand', (S, 1), (S, S, S), 'new_dim', (False,)),
        ('expand', (1,), (S, S, S), '1_element', (False,)),
        ('expand', (1, S), (1, 1, S), 'new_dim_front_old_front_1', (False,)),
        ('expand', (), (dont_convert(()),), 'scalar_to_scalar'),
        ('expand', (), (1, 3, 2), 'scalar_to_dims', (False,)),
        ('expand_as', (S, 1, 1), (torch.rand(S, S, S),), '', (False,)),
        ('exp', (S, S, S), NO_ARGS, '', (True,)),
        ('exp', (), NO_ARGS, 'scalar', (True,)),
        ('exp2', (S, S, S), NO_ARGS, '', (False,)),
        ('exp2', (), NO_ARGS, 'scalar', (False,)),
        ('expm1', (S, S, S), NO_ARGS, '', (True,)),
        ('expm1', (), NO_ARGS, 'scalar', (True,)),
        ('erf', torch.rand(S, S, S), NO_ARGS, '', (True,)),
        ('erf', uniform_scalar(requires_grad=True), NO_ARGS, 'scalar', (True,)),
        ('erfc', torch.rand(S, S, S), NO_ARGS, '', (True,)),
        ('erfc', uniform_scalar(requires_grad=True), NO_ARGS, 'scalar', (True,)),
        ('erfinv', torch.rand(S, S, S).clamp(-0.9, 0.9), NO_ARGS),
        ('erfinv', normal_scalar_clamp(-0.9, 0.9, requires_grad=True), NO_ARGS, 'scalar'),
        ('log', torch.rand(S, S, S) + 1e-2, NO_ARGS, '', (True,)),
        ('log', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar', (True,)),
        ('log10', torch.rand(S, S, S) + 1e-2, NO_ARGS, '', (True,)),
        ('log10', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar', (True,)),
        ('log1p', torch.rand(S, S, S), NO_ARGS, '', (True,)),
        ('log1p', uniform_scalar(requires_grad=True), NO_ARGS, 'scalar', (True,)),
        ('log2', torch.rand(S, S, S) + 1e-2, NO_ARGS, '', (True,)),
        ('log2', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar', (True,)),
        ('log', torch.randn(S, S, S, dtype=torch.cdouble) + 1e-2, NO_ARGS, 'complex', (True,)),
        ('log', uniform_scalar(1e-2j, requires_grad=True), NO_ARGS, 'complex_scalar', (True,)),
        ('log10', torch.randn(S, S, S, dtype=torch.cdouble) + 1e-2, NO_ARGS, 'complex', (True,)),
        ('log10', uniform_scalar(1e-2j, requires_grad=True), NO_ARGS, 'complex_scalar', (True,)),
        ('log2', torch.randn(S, S, S, dtype=torch.cdouble) + 1e-2, NO_ARGS, 'complex', (True,)),
        ('log2', uniform_scalar(1e-2j, requires_grad=True), NO_ARGS, 'complex_scalar', (True,)),
        ('tanh', (S, S, S), NO_ARGS, '', (True,)),
        ('tanh', (), NO_ARGS, 'scalar', (True,)),
        ('sigmoid', (S, S, S), NO_ARGS, '', (True,)),
        ('sigmoid', (), NO_ARGS, 'scalar', (True,)),
        ('logit', torch.randn(S, S, S).clamp(0.1, 0.9).requires_grad_(True), NO_ARGS, ''),
        ('logit', torch.randn(S, S, S).clamp(0.1, 0.9).requires_grad_(True), (0.2,), 'eps'),
        ('logit', uniform_scalar().clamp(0.1, 0.9).requires_grad_(True), NO_ARGS, 'scalar'),
        ('logit', uniform_scalar().clamp(0.1, 0.9).requires_grad_(True), (0.2,), 'scalar_eps'),
        ('sinh', (S, S, S), NO_ARGS, '', (True,)),
        ('sinh', (), NO_ARGS, 'scalar', (True,)),
        ('cosh', (S, S, S), NO_ARGS, '', (True,)),
        ('cosh', (), NO_ARGS, 'scalar', (True,)),
        ('conj', (S, S, S), NO_ARGS),
        ('copysign', (S, S, S), ((S, S, S),), '', (False,)),
        ('copysign', (S, S, S), ((S, S),), 'broadcast_rhs', (False,)),
        ('copysign', (S, S), ((S, S, S),), 'broadcast_lhs', (False,)),
        ('copysign', (S, 1, S), ((M, S),), 'broadcast_all', (False,)),
        ('copysign', (S, S), (3.14,), 'scalar', (False,)),
        ('copysign', (S, S), (0.0,), 'scalar_pos_zero', (False,)),
        # TorchScript does not recognize -0.0: Issue #46848
        # https://github.com/pytorch/pytorch/issues/46848
        # ('copysign', (S, S), (-0.0,), 'scalar_neg_zero', (False,)),
        ('real', (S, S, S), NO_ARGS, 'complex'),
        ('imag', (S, S, S), NO_ARGS, 'complex'),
        ('view_as_real', (S, S, S), NO_ARGS, 'complex'),
        ('view_as_complex', (S, S, 2), NO_ARGS),
        ('complex', (S, S, S), ((S, S, S),), ''),
        ('abs', (S, S, S), NO_ARGS, '', (True,)),
        ('abs', (), NO_ARGS, 'scalar', (True,)),
        ('angle', (S, S, S), NO_ARGS, '', (True,)),
        ('angle', (), NO_ARGS, 'scalar', (True,)),
        ('clamp', (S, S, S), (0, 1), '', (True,)),
        ('clamp', (S, S, S), (None, 0.5), 'min', (True,)),
        ('clamp', (S, S, S), (0.5, None), 'max', (True,)),
        ('clamp', (), (0, 1), 'scalar', (True,)),
        ('clamp', (), (None, 0.5), 'min_scalar', (True,)),
        ('clamp', (), (0.5, None), 'max_scalar', (True,)),
        ('clamp', (S, S), (), 'max_scalar_kwarg', (True,), (), (), ident, {'max': 1}),
        ('sqrt', torch.rand(S, S, S) + 5e-4, NO_ARGS, '', (True,)),
        ('sqrt', uniform_scalar(5e-4, requires_grad=True), NO_ARGS, 'scalar', (True,)),
        ('sin', (S, S, S), NO_ARGS, '', (True,)),
        ('sin', (), NO_ARGS, 'scalar', (True,)),
        ('cos', (S, S, S), NO_ARGS, '', (True,)),
        ('cos', (), NO_ARGS, 'scalar', (True,)),
        ('tan', torch.randn(S, S, S).clamp(-1, 1), NO_ARGS, '', (True,)),
        ('tan', (S, S, S), NO_ARGS, 'complex', (True,)),
        ('asin', torch.randn(S, S, S).clamp(-0.9, 0.9), NO_ARGS, '', (True,)),
        ('acos', torch.randn(S, S, S).clamp(-0.9, 0.9), NO_ARGS, '', (True,)),
        ('atan', (S, S, S), NO_ARGS, '', (True,)),
        ('atan', (), NO_ARGS, 'scalar', (True,)),
        ('atan2', (S, S, S), ((S, S, S),)),
        ('atan2', (), ((),), 'scalar'),
        ('atan2', (S, S, S), ((S,),), 'broadcast_rhs'),
        ('atan2', (S,), ((S, S, S),), 'broadcast_lhs'),
        ('atan2', (S, 1, S), ((S, S),), 'broadcast_all'),
        ('reciprocal', torch.rand(S, S, S) + 0.1, NO_ARGS, '', (True,)),
        ('reciprocal', uniform_scalar(0.1, requires_grad=True), NO_ARGS, 'scalar', (True,)),
        ('reciprocal', torch.randn(S, S, S, dtype=torch.cdouble) + 0.1, NO_ARGS, 'complex', (True,)),
        ('reciprocal', uniform_scalar(0.1j), NO_ARGS, 'complex_scalar', (True,)),
        ('round', (S, S, S), NO_ARGS, '', (True,)),
        ('round', (), NO_ARGS, 'scalar', (True,)),
        ('sign', (S, S, S), NO_ARGS),
        ('sign', (), NO_ARGS, 'scalar'),
        ('sgn', (S, S, S), NO_ARGS),
        ('sgn', (), NO_ARGS, 'scalar'),
        ('trunc', (S, S, S), NO_ARGS, '', (True,)),
        ('trunc', (), NO_ARGS, 'scalar', (True,)),
        ('floor', (S, S, S), NO_ARGS, '', (True,)),
        ('floor', (), NO_ARGS, 'scalar', (True,)),
        ('ceil', (S, S, S), NO_ARGS, '', (True,)),
        ('ceil', (), NO_ARGS, 'scalar', (True,)),
        ('rad2deg', (S, S, S), NO_ARGS),
        ('deg2rad', (S, S, S), NO_ARGS),
        ('rsqrt', torch.rand(S, S, S) + 1e-2, NO_ARGS, '', (True,)),
        ('rsqrt', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar', (True,)),
        ('rsqrt', torch.rand(S, S, S, dtype=torch.cfloat) + 1e-2, NO_ARGS, 'complex', (True,)),
        ('rsqrt', uniform_scalar(1e-2 * (1 + 1j), requires_grad=True), NO_ARGS, 'complex_scalar', (True,)),
        ('frac', (S, S, S), NO_ARGS, '', (True,)),
        ('frac', (), NO_ARGS, 'scalar', (True,)),
        ('fmod', (S, S, S), (1.5,), '', (True,)),
        ('fmod', (), (1.5,), 'scalar', (True,)),
        ('fmod', (S, S, S), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor'),
        ('fmod', (S,), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor_broadcast_lhs'),
        ('fmod', (S, S, S), (non_differentiable(torch.rand(S) + 1.5),), 'tensor_broadcast_rhs'),
        ('fmod', (S, 1, S), (non_differentiable(torch.rand(S, S) + 1.5),), 'tensor_broadcast_all'),
        ('fmod', (), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor'),
        ('fmod', (), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'scalar_tensor_broadcast_lhs'),
        ('fmod', (S, S, S), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor_broadcast_rhs'),
        ('hypot', (S, S), ((S, S),)),
        ('remainder', (S, S, S), (1.5,), '', (True,)),
        ('remainder', (), (1.5,), 'scalar', (True,)),
        ('remainder', (S, S, S), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor'),
        ('remainder', (S,), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor_broadcast_lhs'),
        ('remainder', (S, 1, S), (non_differentiable(torch.rand(S, S) + 1.5),), 'tensor_broadcast_all'),
        ('remainder', (), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor'),
        ('remainder', (), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'scalar_tensor_broadcast_lhs'),
        ('lerp', (S, S, S), ((S, S, S), 0.4), 'scalar_no_broadcast', (True,)),
        ('lerp', (S, S, S), ((S,), 0.4), 'broadcast_rhs', (True,)),
        ('lerp', (S,), ((S, S, S), 0.4), 'broadcast_lhs', (True,)),
        ('lerp', (S, 1, S), ((S, S), 0.4), 'broadcast_all', (True,)),
        ('lerp', (), ((), 0.4), 'scalar', (True,)),
        ('lerp', (S, S, S), ((), 0.4), 'scalar_broadcast_rhs', (True,)),
        ('lerp', (), ((S, S, S), 0.4), 'scalar_broadcast_lhs', (True,)),
        ('max', (S, S, S), NO_ARGS),
        ('max', (S, S, S), (1,), 'dim', (), [0]),
        ('max', (S, S, S), (1, True,), 'keepdim_dim', (), [0]),
        ('max', (), NO_ARGS, 'scalar'),
        ('max', (), (0,), 'scalar_dim', (), [0]),
        ('max', (), (0, True,), 'scalar_keepdim_dim', (), [0]),
        ('max', (S, S, S), ((S, S, S),), 'elementwise', (True,)),
        ('max', (S, S, S), ((S,),), 'elementwise_broadcast_rhs', (True,)),
        ('max', (S,), ((S, S, S),), 'elementwise_broadcast_lhs', (True,)),
        ('max', (S, 1, S), ((S, S),), 'elementwise_broadcast_all', (True,)),
        ('max', (), ((),), 'scalar_elementwise', (True,)),
        ('max', (S, S, S), ((),), 'scalar_elementwise_broadcast_rhs', (True,)),
        ('max', (), ((S, S, S),), 'scalar_elementwise_broadcast_lhs', (True,)),
        ('min', (S, S, S), NO_ARGS, ),
        ('min', (S, S, S), (1,), 'dim', (), [0]),
        ('min', (S, S, S), (1, True,), 'keepdim_dim', (), [0]),
        ('min', (), NO_ARGS, 'scalar'),
        ('min', (), (0,), 'scalar_dim', (), [0]),
        ('min', (), (0, True,), 'scalar_keepdim_dim', (), [0]),
        ('min', (S, S, S), ((S, S, S),), 'elementwise', (True,)),
        ('min', (S, S, S), ((S,),), 'elementwise_broadcast_rhs', (True,)),
        ('min', (S,), ((S, S, S),), 'elementwise_broadcast_lhs', (True,)),
        ('min', (S, 1, S), ((S, S),), 'elementwise_broadcast_all', (True,)),
        ('min', (), ((),), 'scalar_elementwise', (True,)),
        ('min', (S, S, S), ((),), 'scalar_elementwise_broadcast_rhs', (True,)),
        ('min', (), ((S, S, S),), 'scalar_elementwise_broadcast_lhs', (True,)),
        ('amax', (S, S, S), NO_ARGS),
        ('amax', (S, S, S), (1,), 'dim'),
        ('amax', (S, S, S), ([1, 2],), 'multiple_dim'),
        ('amax', (S, S, S), (1, True,), 'keepdim_dim'),
        ('amax', (), NO_ARGS, 'scalar'),
        ('amax', (), (0,), 'scalar_dim'),
        ('amax', (), (0, True,), 'scalar_keepdim_dim'),
        ('amin', (S, S, S), NO_ARGS, ),
        ('amin', (S, S, S), (1,), 'dim',),
        ('amin', (S, S, S), ([1, 2],), 'multiple_dim'),
        ('amin', (S, S, S), (1, True,), 'keepdim_dim'),
        ('amin', (), NO_ARGS, 'scalar'),
        ('amin', (), (0,), 'scalar_dim'),
        ('amin', (), (0, True,), 'scalar_keepdim_dim'),
        ('mean', (S, S, S), NO_ARGS, '', (True,)),
        ('mean', (S, S, S), (1,), 'dim', (True,), [0]),
        ('mean', (S, S, S), (1, True,), 'keepdim_dim', (True,), [0]),
        ('mean', (), NO_ARGS, 'scalar', (True,)),
        ('mean', (), (0,), 'scalar_dim', (True,), [0]),
        ('mean', (), (0, True,), 'scalar_keepdim_dim', (True,), [0]),
        ('mean', (S, S, S), (), 'dtype', (True,), (), (), ident, {'dtype': torch.float64}),
        ('kthvalue', (S, S, S), (2,)),
        ('kthvalue', (S, S, S), (2, 1,), 'dim', (), [1]),
        ('kthvalue', (S, S, S), (2, 1, True,), 'keepdim_dim', (), [1]),
        ('kthvalue', (S,), (2, 0,), 'dim_1d', (), [1]),
        ('kthvalue', (S,), (2, 0, True,), 'keepdim_dim_1d', (), [1]),
        ('kthvalue', (), (1,), 'scalar', (), ()),
        ('kthvalue', (), (1, 0,), 'scalar_dim', (), [1]),
        ('kthvalue', (), (1, 0, True), 'scalar_keepdim_dim', (), [1]),
        ('quantile', (S, S, S), (0.5,)),
        ('quantile', (S, S, S), (0.5, 0), 'dim', (), [1]),
        ('quantile', (S, S, S), (0.5, None, True), 'keepdim'),
        ('quantile', (S, S, S), (0.5, 0, True), 'keepdim_dim', (), [1]),
        ('quantile', (), (0.5,), 'scalar'),
        ('nanquantile', (S, S, S), (0.5,)),
        ('nanquantile', (S, S, S), (0.5, 0), 'dim', (), [1]),
        ('nanquantile', (S, S, S), (0.5, None, True), 'keepdim'),
        ('nanquantile', (S, S, S), (0.5, 0, True), 'keepdim_dim', (), [1]),
        ('nanquantile', (), (0.5,), 'scalar'),
        ('median', (S, S, S), NO_ARGS),
        ('median', (S, S, S), (1,), 'dim', (), [0]),
        ('median', (S, S, S), (1, True,), 'keepdim_dim', (), [0]),
        ('median', (), NO_ARGS, 'scalar'),
        ('median', (), (0,), 'scalar_dim', (), [0]),
        ('median', (), (0, True,), 'scalar_keepdim_dim', (), [0]),
        ('nanmedian', (S, S, S), NO_ARGS),
        ('nanmedian', (S, S, S), (1,), 'dim', (), [0]),
        ('nanmedian', (S, S, S), (1, True,), 'keepdim_dim', (), [0]),
        ('nanmedian', (), NO_ARGS, 'scalar'),
        ('nanmedian', (), (0,), 'scalar_dim', (), [0]),
        ('nanmedian', (), (0, True,), 'scalar_keepdim_dim', (), [0]),
        ('mode', (S, S, S), NO_ARGS),
        ('mode', (S, S, S), (1,), 'dim', (), [0]),
        ('mode', (S, S, S), (1, True,), 'keepdim_dim', (), [0]),
        ('mode', (), NO_ARGS, 'scalar'),
        ('mode', (), (0,), 'scalar_dim', (), [0]),
        ('mode', (), (0, True,), 'scalar_keepdim_dim', (), [0]),
        ('sum', (S, S, S), NO_ARGS),
        ('sum', (S, S, S), (1,), 'dim', (), [0]),
        ('sum', (S, S, S), (1, True,), 'keepdim_dim', (), [0]),
        ('sum', (), NO_ARGS, 'scalar'),
        ('sum', (), (0,), 'scalar_dim', (), [0]),
        ('sum', (), (0, True,), 'scalar_keepdim_dim', (), [0]),
        ('sum', (S, S, S), ([1, 2],), 'multi_dim'),
        ('sum', (S, S, S), ([1, 2], True,), 'multi_dim_keepdim'),
        ('nansum', (S, S, S), NO_ARGS),
        ('nansum', (S, S, S), (1,), 'dim', (), [0]),
        ('nansum', (S, S, S), (1, True,), 'keepdim_dim', (), [0]),
        ('nansum', (), NO_ARGS, 'scalar'),
        ('nansum', (), (0,), 'scalar_dim', (), [0]),
        ('nansum', (), (0, True,), 'scalar_keepdim_dim', (), [0]),
        ('nansum', (S, S, S), ([1, 2],), 'multi_dim'),
        ('nansum', (S, S, S), ([1, 2], True,), 'multi_dim_keepdim'),
        ('prod', (S, S, S), NO_ARGS),
        ('prod', (S, S, S), (1,), 'dim', (), [0]),
        ('prod', (S, S, S), (1, True,), 'keepdim_dim', (), [0]),
        ('prod', (), NO_ARGS, 'scalar'),
        ('prod', (), (0,), 'scalar_dim', (), [0]),
        ('prod', (), (0, True,), 'scalar_keepdim_dim', (), [0]),
        ('prod', prod_zeros(S, [0, 1]), NO_ARGS, 'zerodims2'),
        ('prod', prod_zeros(S, [0, 2]), NO_ARGS, 'zerodims1'),
        ('prod', prod_zeros(S, [1, 2]), NO_ARGS, 'zerodims0'),
        ('prod', prod_zeros(S, [0, 1]), (1,), 'zeros_dims2', (), [0]),
        ('prod', prod_zeros(S, [0, 2]), (1,), 'zeros_dims1', (), [0]),
        ('prod', prod_zeros(S, [1, 2]), (1,), 'zeros_dims0', (), [0]),
        ('prod', prod_zeros(S, [0, 1]), (1, True), 'keepdim_zeros_dims2', (), [0]),
        ('prod', prod_zeros(S, [0, 2]), (1, True), 'keepdim_zeros_dims1', (), [0]),
        ('prod', prod_zeros(S, [1, 2]), (1, True), 'keepdim_zeros_dims0', (), [0]),
        ('prod', prod_single_zero(S), NO_ARGS, 'single_zero'),
        ('prod', (torch.tensor(0., requires_grad=True)), NO_ARGS, 'scalar_zero'),
        ('prod', (torch.tensor(0., requires_grad=True)), (0,), 'scalar_dim_zero', (), [0]),
        ('prod', (torch.tensor(0., requires_grad=True)), (0, True,), 'scalar_keepdim_dim_zero', (), [0]),
        ('var', (S, S, S), NO_ARGS, '', (True,)),
        ('var', (S, S, S), (1,), 'dim', (True,), [0]),
        ('var', (S, S, S), (1, True, True), 'keepdim_dim', (True,), [0]),
        ('var', (S,), (0,), 'dim_1d', (True,), [0]),
        ('var', (S,), (0, True, True), 'keepdim_dim_1d', (True,), [0]),
        ('std', (S, S, S), NO_ARGS, '', (True,)),
        ('std', (S, S, S), (1,), 'dim', (True,), [0]),
        ('std', (S, S, S), (1, True, True), 'keepdim_dim', (True,), [0]),
        ('std', (S,), (0,), 'dim_1d', (True,), [0]),
        ('std', (S,), (0, True, True), 'keepdim_dim_1d', (True,), [0]),
        ('var_mean', (S, S, S), NO_ARGS, ''),
        ('var_mean', (S, S, S), (1,), 'dim', [0]),
        ('var_mean', (S, S, S), (1, True, True), 'keepdim_dim', [0]),
        ('var_mean', (S,), (0,), 'dim_1d', [0]),
        ('var_mean', (S,), (0, True, True), 'keepdim_dim_1d', [0]),
        ('std_mean', (S, S, S), NO_ARGS, ''),
        ('std_mean', (S, S, S), (1,), 'dim', [0]),
        ('std_mean', (S, S, S), (1, True, True), 'keepdim_dim', [0]),
        ('std_mean', (S,), (0,), 'dim_1d', [0]),
        ('std_mean', (S,), (0, True, True), 'keepdim_dim_1d', [0]),
        ('renorm', (S, S, S), (2, 1, 0.5), 'dim', (), [1]),
        ('renorm', (S, S, S), (1, 2, 3), 'norm_1'),
        ('renorm', (S, S, S), (inf, 2, 0.5), 'norm_inf'),
        ('repeat', (S,), (2,), 'single_number'),
        ('repeat', (), (2, 3), 'scalar'),
        ('repeat', (2, 2), (3, 2)),
        ('repeat', (2, 2), (1, 3, 1, 2), 'unsqueeze'),
        ('repeat', (S, S), (1, 1), 'keepdim0'),
        ('repeat', (S, S), (3, 1, 1), 'keepdim1'),
        ('repeat', (S,), (0, ), 'zero_dim'),
        ('repeat', (S,), (0, 2), 'zero_dim_multi'),
        ('logcumsumexp', (S, S, S), (0,), 'dim0', (), [0]),
        ('logcumsumexp', (S, S, S), (1,), 'dim1', (), [0]),
        ('logcumsumexp', (), (0,), 'dim0_scalar', (), [0]),
        ('cummax', (S, S, S), (0,), 'dim0', (), [0]),
        ('cummax', (S, S, S), (1,), 'dim1', (), [0]),
        ('cummax', (), (0,), 'dim0_scalar', (), [0]),
        ('cummin', (S, S, S), (0,), 'dim0', (), [0]),
        ('cummin', (S, S, S), (1,), 'dim1', (), [0]),
        ('cummin', (), (0,), 'dim0_scalar', (), [0]),
        ('cumsum', (S, S, S), (0,), 'dim0', (), [0]),
        ('cumsum', (S, S, S), (1,), 'dim1', (), [0]),
        ('cumsum', (S, S, S), (1,), 'dim1_cast', (), [0], (), ident, {'dtype': torch.float64}),
        ('cumsum', (), (0,), 'dim0_scalar', (), [0]),
        ('cumprod', (S, S, S), (0,)),
        ('cumprod', (S, S, S), (1,), 'dim1', (), [0]),
        ('cumprod', (), (0,), 'scalar'),
        ('cumprod', (torch.tensor(0., requires_grad=True)), (0,), 'scalar_zeros'),
        ('cumprod', prod_zeros(S, [0, 1]), (1,), 'zeros_dim2', (), [0]),
        ('cumprod', prod_zeros(S, [0, 2]), (1,), 'zeros_dim1', (), [0]),
        ('cumprod', prod_zeros(S, [1, 2]), (1,), 'zeros_dim0', (), [0]),
        ('cumprod', prod_zeros(S, [1, 2]), (1,), 'zeros_dim0_cast', (), [0], (), ident, {'dtype': torch.float64}),
        ('log_softmax', (S, S, S), (1, torch.float64,), 'kwarg_dtype_would_break_jit_loader', (True,)),
        ('unfold', (), (0, 1, 1), 'scalar', (), [0]),
        ('unfold', (S, S, S, S), (0, 3, 1), '4d_dim0_step1', (), [0]),
        ('unfold', (S, S, S, S), (1, 3, 1), '4d_dim1_step1', (), [0]),
        ('unfold', (S, S, S, S), (2, 3, 1), '4d_dim2_step1', (), [0]),
        ('unfold', (S, S, S, S), (3, 3, 1), '4d_dim3_step1', (), [0]),
        ('unfold', (S, S, S, S), (0, 3, 2), '4d_dim0_step2', (), [0]),
        ('unfold', (S, S, S, S), (1, 3, 2), '4d_dim1_step2', (), [0]),
        ('unfold', (S, S, S, S), (2, 3, 2), '4d_dim2_step2', (), [0]),
        ('unfold', (S, S, S, S), (3, 3, 2), '4d_dim3_step2', (), [0]),
        ('unfold', (S, S, S, S), (0, 4, 1), '4d_dim0_size4', (), [0]),
        ('unfold', (S, S, S, S), (1, 4, 1), '4d_dim1_size4', (), [0]),
        ('unfold', (S, S, S, S), (2, 4, 1), '4d_dim2_size4', (), [0]),
        ('unfold', (S, S, S, S), (3, 4, 1), '4d_dim3_size4', (), [0]),
        ('unfold', (M,), (0, 3, 1), '1d_step1', (), [0]),
        ('unfold', (M,), (0, 3, 2), '1d_step2', (), [0]),
        ('unfold', (M,), (0, 3, 3), '1d_step3', (), [0]),
        ('unfold', (1000,), (0, 3, 11), '1d_step_gt_size', (), [0]),
        ('unfold', (1000,), (0, 2, 27), '1d_step_gt_size2', (), [0]),
        ('unfold', (10, 10), (0, 1, 2), '2d_step_gt_size', (), [0]),
        ('unfold', (10, 10), (1, 2, 3), '2d_step_gt_size2', (), [0]),
        ('unfold', (10, 10), (1, 2, 2), '2d_step_ge_size2', (), [0]),
        ('unfold', (S, S, S), (2, 3, 2), 'lastdim', (), [0]),
        ('addmm', (S, M), ((S, S), (S, M)), '', (True, ['aten::add', 'aten::mm'])),
        ('addmm', (1,), ((S, S), (S, M)), 'broadcast_lhs', (True, ['aten::add', 'aten::mm'])),
        ('addmm', (S, M), ((S, S), (S, M)), 'coef', (True,), (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('addmm', (1,), ((S, S), (S, M)), 'broadcast_lhs_coef', (True,), (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('addmm', (), ((S, S), (S, M)), 'scalar_broadcast_lhs', (True, ['aten::add', 'aten::mm'])),
        ('addmm', (), ((S, S), (S, M)), 'scalar_broadcast_lhs_coef', (True,), (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('addbmm', (S, M), ((S, S, S), (S, S, M)),),
        ('addbmm', (1,), ((S, S, S), (S, S, M)), 'broadcast_lhs'),
        ('addbmm', (S, M), ((S, S, S), (S, S, M)), 'coef', (), (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('addbmm', (1,), ((S, S, S), (S, S, M)), 'broadcast_lhs_coef', (),
         (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('addbmm', (), ((S, S, S), (S, S, M)), 'scalar_broadcast_lhs'),
        ('addbmm', (), ((S, S, S), (S, S, M)), 'scalar_broadcast_lhs_coef', (), (), (), ident,
         {'beta': 0.2, 'alpha': 0.6}),
        ('baddbmm', (S, S, M), ((S, S, S), (S, S, M)),),
        ('baddbmm', (1,), ((S, S, S), (S, S, M)), 'broadcast_lhs'),
        ('baddbmm', (S, S, M), ((S, S, S), (S, S, M)), 'coef', (), (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('baddbmm', (1,), ((S, S, S), (S, S, M)), 'broadcast_lhs_coef', (),
         (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('baddbmm', (), ((S, S, S), (S, S, M)), 'scalar_broadcast_lhs'),
        ('baddbmm', (), ((S, S, S), (S, S, M)), 'scalar_broadcast_lhs_coef', (), (), (), ident,
         {'beta': 0.2, 'alpha': 0.6}),
        ('addmv', (S,), ((S, M), (M,)),),
        ('addmv', (1,), ((S, M), (M,)), 'broadcast_lhs'),
        ('addmv', (S,), ((S, M), (M,)), 'coef', (), (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('addmv', (1,), ((S, M), (M,)), 'broadcast_lhs_coef', (), (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('addmv', (), ((S, M), (M,)), 'scalar_broadcast_lhs'),
        ('addmv', (), ((S, M), (M,)), 'scalar_broadcast_lhs_coef', (), (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('addr', (S, M), ((S,), (M,)),),
        ('addr', (), ((S,), (M,)), 'broadcast_lhs'),
        ('addr', (S, M), ((S,), (M,)), 'coef', (), (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('addr', (), ((S,), (M,)), 'broadcast_lhs_coef', (), (), (), ident, {'beta': 0.2, 'alpha': 0.6}),
        ('dot', (L,), ((L,),), '', (True,)),
        ('vdot', (L,), ((L,),),),
        ('mm', (S, M), ((M, S),), '', (True,)),
        ('bmm', (M, S, M), ((M, M, S),), '', (True,)),
        ('mv', (S, M), ((M,),), '', (True,)),
        ('ger', (S,), ((M,),)),
        ('matmul', (L,), ((L,),), '', (True,)),
        ('matmul', (S, M), ((M,),), "2d_1d", (True,)),
        ('matmul', (M,), ((M, S),), "1d_2d", (True,)),
        ('matmul', (S, M), ((M, S),), "2d_2d", (True,)),
        ('matmul', (S, S, M), ((M,),), "3d_1d", (True,)),
        ('matmul', (S, S, M), ((M, S),), "3d_2d", (True,)),
        ('matmul', (M,), ((S, M, S),), "1d_3d", (True,)),
        ('matmul', (S, M), ((S, M, S),), "2d_3d", (True,)),
        ('matmul', (S, S, M, M), ((S, S, M, S),), "4d_4d", (True,)),
        ('matmul', (S, S, M, M), ((M,),), "4d_1d", (True,)),
        ('matmul', (M,), ((S, S, M, S),), "1d_4d", (True,)),
        ('matrix_power', (S, S), [2], "n=2"),
        ('matrix_power', (S, S, S), [3], "n=3"),
        ('matrix_power', (S, S, S), [1], "n=1"),
        ('matrix_power', (S, S, S), [0], "n=0"),
        ('matrix_power', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S), [-1], "n=-1", (),
         NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('matrix_power', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S), [-3], "n=-3", (),
         NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('matrix_power', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S, S), [-2], "n=-2", (),
         NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('matrix_exp', (S, S), NO_ARGS, "single_matrix"),
        ('matrix_exp', (S, S, S), NO_ARGS, "batch_of_matrices"),
        ('mvlgamma', torch.empty(S,).uniform_(0.5, 1), [1], "p=1"),
        ('mvlgamma', torch.empty(S,).uniform_(1, 2), [2], "p=2"),
        ('mvlgamma', torch.empty(S, S).uniform_(1.5, 3), [3], "p=3"),
        ('mvlgamma', torch.empty(S, S).uniform_(2.5, 5), [5], "p=5"),
        ('addcmul', (S, S), ((S, S), (S, S)), '', (True,)),
        ('addcmul', (S, S), ((S, 1), (1, S)), 'broadcast_rhs', (True,)),
        ('addcmul', (1,), ((S, S, 1), (1, S)), 'broadcast_all', (True,)),
        ('addcmul', (S, S), ((S, S), (S, S)), 'scale', (True,), (), (), ident, {'value': 0.5}),
        ('addcmul', (S, S), ((S, 1), (1, S)), 'scale_broadcast_rhs', (True,), (), (), ident, {'value': 0.5}),
        ('addcmul', (1,), ((S, S, 1), (1, S)), 'scale_broadcast_all', (True,), (), (), ident, {'value': 0.5}),
        ('addcmul', (), ((), ()), 'scalar', (True,)),
        ('addcmul', (S, S), ((), ()), 'scalar_broadcast_rhs', (True,)),
        ('addcmul', (), ((S, S, 1), (1, S)), 'scalar_broadcast_lhs', (True,)),
        ('addcmul', (), ((), ()), 'scalar_scale', (True,), (), (), ident, {'value': 0.5}),
        ('addcmul', (S, S), ((), ()), 'scalar_scale_broadcast_rhs', (True,), (), (), ident, {'value': 0.5}),
        ('addcmul', (), ((S, S, 1), (1, S)), 'scalar_scale_broadcast_lhs', (True,), (), (), ident, {'value': 0.5}),
        ('addcdiv', (S, S), ((S, S), (S, S))),
        ('addcdiv', (S, S), ((S, 1), (1, S)), 'broadcast_rhs'),
        ('addcdiv', (1,), ((S, S, 1), (1, S)), 'broadcast_all'),
        ('addcdiv', (S, S), ((S, S), (S, S)), 'scale', (), (), (), ident, {'value': 0.5}),
        ('addcdiv', (S, S), ((S, 1), (1, S)), 'scale_broadcast_rhs', (), (), (), ident, {'value': 0.5}),
        ('addcdiv', (1,), ((S, S, 1), (1, S)), 'scale_broadcast_all', (), (), (), ident, {'value': 0.5}),
        ('addcdiv', (), ((), ()), 'scalar'),
        ('addcdiv', (S, S), ((), ()), 'scalar_broadcast_rhs'),
        ('addcdiv', (), ((S, S, 1), (1, S)), 'scalar_broadcast_lhs'),
        ('addcdiv', (), ((), ()), 'scalar_scale', (), (), (), ident, {'value': 0.5}),
        ('addcdiv', (S, S), ((), ()), 'scalar_scale_broadcast_rhs', (), (), (), ident, {'value': 0.5}),
        ('addcdiv', (), ((S, S, 1), (1, S)), 'scalar_scale_broadcast_lhs', (), (), (), ident, {'value': 0.5}),
        ('zero_', (S, S, S), NO_ARGS),
        ('zero_', (), NO_ARGS, 'scalar'),
        ('logaddexp', (S, S), ((S, S),)),
        ('logaddexp2', (S, S), ((S, S),)),
        ('logsumexp', (S, S), (1,), '', (True,)),
        ('logsumexp', (), (0,), 'scalar', (True,)),
        ('norm', (S, S), (), 'default'),
        ('norm', (S, S), (2,), '2'),
        ('norm', (S, S), (0,), '0'),
        ('norm', (S, S), (0.5,), '0_5'),
        ('norm', (S, S), (1,), '1'),
        ('norm', (S, S), (3,), '3'),
        ('norm', (S, S), (inf,), 'inf'),
        ('norm', (S, S), (-inf,), '-inf'),
        ('norm', (S, S), ('fro',), 'fro_default'),
        ('norm', (S, S), ('fro', [0, 1],), 'fro'),
        ('norm', (S, S), ('nuc',), 'nuc', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('norm', (S, S, S), ('nuc', [1, 2]), 'nuc_batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('norm', (S, S), (-1,), 'neg_1'),
        ('norm', (S, S), (-2,), 'neg_2'),
        ('norm', (S, S), (-0.5,), 'neg_0_5'),
        ('norm', (S, S), (-1.5,), 'neg_1_5'),
        ('norm', (S, S), (-2, 1,), 'neg_2_2_dim', (), [1]),
        ('norm', (S, S), (-1, 1,), 'neg_1_2_dim', (), [1]),
        ('norm', (S, S), (0, 1,), '0_2_dim', (), [1]),
        ('norm', (S, S), (1, 1,), '1_2_dim', (), [1]),
        ('norm', (S, S), (2, 1,), '2_2_dim', (), [1]),
        ('norm', (S, S), (3, 1,), '3_2_dim', (), [1]),
        ('norm', (S, S), (inf, 1,), 'inf_2_dim'),
        ('norm', torch.rand(S, S, S) + 5e-2, (1.5,), '1_5_default'),
        ('norm', (S, S, S), (2, 1), '2_dim', (), [1]),
        ('norm', (S, S, S), (3, 1), '3_dim', (), [1]),
        ('norm', torch.rand(S, S, S) + 5e-2, (1.5, 1), '1_5_dim', (), [1]),
        ('norm', (S, S, S), (2, 1, True), 'keepdim_2_dim', (), [1]),
        ('norm', (S, S, S), (3, 1, True), 'keepdim_3_dim', (), [1]),
        ('norm', torch.rand(S, S, S) + 5e-2, (1.5, 1, True), 'keepdim_1_5_dim', (), [1]),
        ('norm', (), (2, 0), '2_dim_scalar', (), [1]),
        ('norm', (), (3, 0), '3_dim_scalar', (), [1]),
        ('norm', (), (2, 0, True), 'keepdim_2_dim_scalar', (), [1]),
        ('norm', (), (3, 0, True), 'keepdim_3_dim_scalar', (), [1]),
        ('clone', (S, M, S), NO_ARGS),
        ('clone', (), NO_ARGS, 'scalar'),
        ('contiguous', (S, S), NO_ARGS, '', (True,)),
        ('contiguous', torch.randn(S, S).transpose(0, 1), NO_ARGS, 'not_contiguous', (True,)),
        ('dist', (S, S, S), ((S, S, S),)),
        ('dist', (S, S, S), ((S,),), 'broadcast_rhs'),
        ('dist', (S,), ((S, S, S),), 'broadcast_lhs'),
        ('dist', (S, 1, S), ((S, S),), 'broadcast_all'),
        ('dist', (), ((),), 'scalar'),
        ('dist', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('dist', (), ((S, S, S),), 'scalar_broadcast_lhs'),
        ('dist', (S, S, S), ((S, S, S), 4), '4'),
        ('dist', (S, S, S), ((S,), 4), '4_broadcast_rhs'),
        ('dist', (S,), ((S, S, S), 4), '4_broadcast_lhs'),
        ('dist', (S, 1, S), ((S, S), 4), '4_broadcast_all'),
        ('dist', (), ((), 4), 'scalar_4'),
        ('dist', (S, S, S), ((), 4), 'scalar_4_broadcast_rhs'),
        ('dist', (), ((S, S, S), 4), 'scalar_4_broadcast_lhs'),
        ('diag', (M, M), NO_ARGS, '2d'),
        ('diag', (3, 5), NO_ARGS, '2d_wide'),
        ('diag', (3, 5), (2,), '2d_wide_pos'),
        ('diag', (3, 5), (-2,), '2d_wide_neg'),
        ('diag', (5, 3), NO_ARGS, '2d_tall'),
        ('diag', (5, 3), (2,), '2d_tall_pos'),
        ('diag', (5, 3), (-2,), '2d_tall_neg'),
        ('diag', (M,), NO_ARGS, '1d'),
        ('diag', (M, M), (1,), '2d_1'),
        ('diag', (M, M), (2,), '2d_2'),
        ('diag_embed', (S, S), NO_ARGS),
        ('diagonal', (M, M), NO_ARGS, '2d'),
        ('diagonal', (3, 5), NO_ARGS, '2d_wide'),
        ('diagonal', (3, 5), (2,), '2d_wide_pos'),
        ('diagonal', (3, 5), (-2,), '2d_wide_neg'),
        ('diagonal', (5, 3), NO_ARGS, '2d_tall'),
        ('diagonal', (5, 3), (2,), '2d_tall_pos'),
        ('diagonal', (5, 3), (-2,), '2d_tall_neg'),
        ('diagonal', (M, M), (1,), '2d_1'),
        ('diagonal', (M, M), (2,), '2d_2'),
        ('diagonal', (M, M, M), (1, 1, 2), '3d_1'),
        ('diagonal', (M, M, M), (2, 0, 1), '3d_2'),
        ('diagonal', (M, M, M), (-2, 0, 1), '3d_3'),
        ('tril', (M, M), NO_ARGS),
        ('tril', (M, M), (2,), 'idx'),
        ('tril', (S, M, M), NO_ARGS, 'batched'),
        ('tril', (S, M, M), (2,), 'batched_idx'),
        ('tril', (3, 3, S, S), NO_ARGS, 'more_batched'),
        ('triu', (M, M), NO_ARGS),
        ('triu', (M, M), (2,), 'idx'),
        ('triu', (S, M, M), NO_ARGS, 'batched'),
        ('triu', (S, M, M), (2,), 'batched_idx'),
        ('triu', (3, 3, S, S), NO_ARGS, 'more_batched'),
        ('trace', (M, M), NO_ARGS),
        ('cross', (S, 3), ((S, 3),)),
        ('cross', (S, 3, S), ((S, 3, S), 1), 'dim'),
        ('index_select', (S, S, S), (0, index_variable(2, S)), 'dim', (), [0]),
        ('index_select', (), (0, torch.tensor([0], dtype=torch.int64)), 'scalar_mixed_dim', (), [0]),
        ('index_select', (), (0, torch.tensor(0, dtype=torch.int64)), 'scalar_dim', (), [0]),
        ('index_add', (S, S), (0, index_variable(2, S), (2, S)), 'dim', (), [0]),
        ('index_add', (), (0, torch.tensor([0], dtype=torch.int64), (1,)), 'scalar_input_dim', (), [0]),
        ('index_add', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim', (), [0]),
        ('index_add', (S, S), (0, index_variable(2, S), (2, S)), 'alert_nondeterministic', (), [0],
            [expectedAlertNondeterministic('index_add_cuda_', 'cuda')]),
        ('index_copy', (S, S), (0, index_perm_variable(2, S), (2, S)), 'dim', (), [0]),
        ('index_copy', (), (0, torch.tensor([0], dtype=torch.int64), (1,)), 'scalar_input_dim', (), [0]),
        ('index_copy', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim', (), [0]),
        ('index_fill', (S, S), (0, index_variable(2, S), 2), 'dim', (), [0]),
        ('index_fill', (S, S), (0, index_variable(2, S), ()), 'variable_dim', (), [0]),
        ('index_fill', (S, S), (0, torch.tensor(0, dtype=torch.int64), 2), 'scalar_index_dim', (), [0]),
        ('index_fill', (), (0, torch.tensor([0], dtype=torch.int64), 2), 'scalar_input_dim', (), [0]),
        ('index_fill', (), (0, torch.tensor(0, dtype=torch.int64), 2), 'scalar_both_dim', (), [0]),
        ('inverse', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),
            NO_ARGS, '', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('inverse', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S, 2, 3, dtype=dtype).to(device),
         NO_ARGS, 'batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', (S, S), NO_ARGS, '', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', (1, 1), NO_ARGS, '1x1', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_symmetric_matrix(S), NO_ARGS, 'symmetric', (),
            NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_symmetric_psd_matrix(S),
            NO_ARGS, 'symmetric_psd', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_symmetric_pd_matrix(S),
            NO_ARGS, 'symmetric_pd', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_square_matrix_of_rank(S, S - 2),
            NO_ARGS, 'dim2_null', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_square_matrix_of_rank(S, 1), NO_ARGS, 'rank1', (),
            NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_square_matrix_of_rank(S, 2), NO_ARGS, 'rank2', (),
            NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S), NO_ARGS,
         'distinct_singular_values', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', (3, 3, S, S), NO_ARGS, 'batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', (3, 3, 1, 1), NO_ARGS, 'batched_1x1', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_symmetric_matrix(S, 3),
            NO_ARGS, 'batched_symmetric', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_symmetric_psd_matrix(S, 3),
            NO_ARGS, 'batched_symmetric_psd', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_symmetric_pd_matrix(S, 3),
            NO_ARGS, 'batched_symmetric_pd', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('det', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S, 3, 3), NO_ARGS,
         'batched_distinct_singular_values', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        # For `logdet` and `slogdet`, the function at det=0 is not smooth.
        # We need to exclude tests with det=0 (e.g. dim2_null, rank1, rank2) and use
        # `make_nonzero_det` to make the random matrices have nonzero det. For
        # `logdet`, we also set `make_nonzero_det(matrix, sign=1)` to make the
        # matrix have positive det.
        ('logdet', lambda dtype, device: make_nonzero_det(torch.randn(S, S), 1),
            NO_ARGS, '', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('logdet', lambda dtype, device: make_nonzero_det(torch.randn(1, 1), 1),
            NO_ARGS, '1x1', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('logdet', lambda dtype, device: make_nonzero_det(random_symmetric_matrix(S), 1), NO_ARGS,
         'symmetric', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('logdet', lambda dtype, device: make_nonzero_det(random_symmetric_pd_matrix(S), 1), NO_ARGS,
         'symmetric_pd', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('logdet', lambda dtype, device: make_nonzero_det(random_fullrank_matrix_distinct_singular_value(S), 1, 0), NO_ARGS,
         'distinct_singular_values', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('logdet', lambda dtype, device: make_nonzero_det(torch.randn(3, 3, S, S), 1),
            NO_ARGS, 'batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('logdet', lambda dtype, device: make_nonzero_det(torch.randn(3, 3, 1, 1), 1),
            NO_ARGS, 'batched_1x1', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('logdet', lambda dtype, device: make_nonzero_det(random_symmetric_matrix(S, 3), 1), NO_ARGS,
         'batched_symmetric', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('logdet', lambda dtype, device: make_nonzero_det(random_symmetric_pd_matrix(S, 3), 1), NO_ARGS,
         'batched_symmetric_pd', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('logdet', lambda dtype, device: make_nonzero_det(random_fullrank_matrix_distinct_singular_value(S, 3), 1, 0), NO_ARGS,
         'batched_distinct_singular_values', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('slogdet', lambda dtype, device: make_nonzero_det(torch.randn(1, 1), 1), NO_ARGS,
         '1x1_pos_det', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: make_nonzero_det(torch.randn(1, 1), -1), NO_ARGS,
         '1x1_neg_det', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: make_nonzero_det(torch.randn(S, S), 1), NO_ARGS,
         'pos_det', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: make_nonzero_det(torch.randn(S, S), -1), NO_ARGS,
         'neg_det', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: make_nonzero_det(random_symmetric_matrix(S)), NO_ARGS,
         'symmetric', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: random_symmetric_pd_matrix(S), NO_ARGS,
         'symmetric_pd', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S), NO_ARGS,
         'distinct_singular_values', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: make_nonzero_det(torch.randn(3, 3, 1, 1), -1), NO_ARGS,
         'batched_1x1_neg_det', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: make_nonzero_det(torch.randn(3, 3, S, S), 1), NO_ARGS,
         'batched_pos_det', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: make_nonzero_det(random_symmetric_matrix(S, 3)), NO_ARGS,
         'batched_symmetric', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: random_symmetric_pd_matrix(S, 3), NO_ARGS,
         'batched_symmetric_pd', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('slogdet', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S, 3), NO_ARGS,
         'batched_distinct_singular_values', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], itemgetter(1)),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S),
            NO_ARGS, '', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S)[:(S - 2)], NO_ARGS,
         'wide', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S)[:, :(S - 2)], NO_ARGS,
         'tall', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S)[:(S - 2)], (False,),
         'wide_all', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], lambda usv: (usv[0], usv[1], usv[2][:, :(S - 2)])),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S)[:, :(S - 2)], (False,),
         'tall_all', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma], lambda usv: (usv[0][:, :(S - 2)], usv[1], usv[2])),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(M), NO_ARGS,
         'large', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S, 3), NO_ARGS,
         'batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S, 3)[..., :(S - 2), :], NO_ARGS,
         'wide_batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S, 3)[..., :, :(S - 2)], NO_ARGS,
         'tall_batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S, 3, 3)[..., :(S - 2), :], (False,),
         'wide_all_batched', (), NO_ARGS,
         [skipCPUIfNoLapack, skipCUDAIfNoMagma], lambda usv: (usv[0], usv[1], usv[2][..., :, :(S - 2)])),
        ('svd', lambda dtype, device: random_fullrank_matrix_distinct_singular_value(S, 3, 3)[..., :, :(S - 2)], (False,),
         'tall_all_batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma],
         lambda usv: (usv[0][..., :, :(S - 2)], usv[1], usv[2])),
        ('qr', (S, S), (False,), 'square_single', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('qr', (S, S - 2), (True,), 'tall_single' , (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('qr', (S - 2, S), (False,), 'wide_single' , (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('qr', (3, S, S), (False,), 'square_batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('qr', (3, S, S - 2), (True,), 'tall_batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('qr', (3, S - 2, S), (True,), 'wide_batched' , (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('qr', (3, 2, S, S), (False,), 'square_many_batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('qr', (3, 2, S, S - 2), (True,), 'tall_many_batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('qr', (3, 2, S - 2, S), (True,), 'wide_many_batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('lu', (S, S), (True, False), 'square_single_no_info', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('lu', (S, S), (True, True), 'square_single_with_info', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('lu', (3, S, S), (True, False), 'square_batch_no_info', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('lu', (3, S, S), (True, True), 'square_batch_with_info', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('lu', (3, 3, S, S), (True, False), 'square_many_batches_no_info', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('lu', (3, 3, S, S), (True, True), 'square_many_batches_with_info', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('solve', (S, S), (lambda dtype, device: random_fullrank_matrix_distinct_singular_value(
            S, silent=True, dtype=dtype, device=device),), '', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('solve', (S, S, S),
            (lambda dtype, device:
                random_fullrank_matrix_distinct_singular_value(S, S, silent=True, dtype=dtype, device=device),),
         'batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('solve', (2, 3, S, S),
            (lambda dtype, device:
                random_fullrank_matrix_distinct_singular_value(S, 2, 3, silent=True, dtype=dtype, device=device),),
         'batched_dims', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('solve', (2, 2, S, S),
            (lambda dtype, device:
                random_fullrank_matrix_distinct_singular_value(S, 1, silent=True, dtype=dtype, device=device),),
         'batched_broadcast_A', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('solve', (1, S, S),
            (lambda dtype, device:
                random_fullrank_matrix_distinct_singular_value(S, 2, 2, silent=True, dtype=dtype, device=device),),
         'batched_broadcast_b', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('fill_', (S, S, S), (1,), 'number'),
        ('fill_', (), (1,), 'number_scalar'),
        ('fill_', (S, S, S), ((),), 'variable'),
        ('eq_', (S, S, S), ((S, S, S),)),
        ('eq_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('eq_', (), ((),), 'scalar'),
        ('eq_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('ne_', (S, S, S), ((S, S, S),)),
        ('ne_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('ne_', (), ((),), 'scalar'),
        ('ne_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('gt_', (S, S, S), ((S, S, S),)),
        ('gt_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('gt_', (), ((),), 'scalar'),
        ('gt_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('ge_', (S, S, S), ((S, S, S),)),
        ('ge_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('ge_', (), ((),), 'scalar'),
        ('ge_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('lt_', (S, S, S), ((S, S, S),)),
        ('lt_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('lt_', (), ((),), 'scalar'),
        ('lt_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('le_', (S, S, S), ((S, S, S),)),
        ('le_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('le_', (), ((),), 'scalar'),
        ('le_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('eq_', (S, S, S), (0,), 'pyscalar'),
        ('ne_', (S, S, S), (0,), 'pyscalar'),
        ('gt_', (S, S, S), (0,), 'pyscalar'),
        ('ge_', (S, S, S), (0,), 'pyscalar'),
        ('le_', (S, S, S), (0,), 'pyscalar'),
        ('lt_', (), (0,), 'pyscalar'),
        ('eq_', (), (0,), 'pyscalar_scalar'),
        ('ne_', (), (0,), 'pyscalar_scalar'),
        ('gt_', (), (0,), 'pyscalar_scalar'),
        ('ge_', (), (0,), 'pyscalar_scalar'),
        ('lt_', (), (0,), 'pyscalar_scalar'),
        ('le_', (), (0,), 'pyscalar_scalar'),
        ('permute', (1, 2, 3, 4), (0, 2, 3, 1), '', (True,)),
        ('permute', (1, 2, 3, 4), (0, -2, -1, 1), 'neg_dim', (True,)),
        ('permute', (), (dont_convert(()),), 'scalar', (True,)),
        ('select', (S, S, S), (1, 2), 'dim', (), [0]),
        ('select', (S, S, S), (1, -1), 'wrap_dim', (), [0]),
        ('select', (S,), (0, 2), '1d'),
        ('narrow', (S, S, S), (1, 2, 2), 'dim', (), [0]),
        ('narrow', (S, S, S), (1, 0, 0), 'empty_dim', (), [0]),
        ('squeeze', (S, 1, S, 1), NO_ARGS, '', (True,)),
        ('squeeze', (1, 1, 1, 1), NO_ARGS, 'input_sizes_are_ones', (True,)),
        ('squeeze', (S, 1, S, 1), (1,), '1_dim', (True,), [0]),
        ('squeeze', (S, 1, S, 1), (2,), 'not_1_dim', (True,), [0]),
        ('squeeze', (), (0,), 'scalar', (True,), [0]),
        ('unsqueeze', (S, S, S), (0,), 'first', (True,), [0]),
        ('unsqueeze', (S, S, S), (1,), 'middle', (True,), [0]),
        ('unsqueeze', (S, S, S), (3,), 'last', (True,), [0]),
        ('unsqueeze', (), (0,), 'scalar', (True,), [0]),
        ('chunk', (S, S, S), (2,), '', (True, 'prim::ConstantChunk')),
        ('chunk', (S, S, S), (S, 1), 'dim', (True, 'prim::ConstantChunk'), [1]),
        ('split', (S, S, S), (2,), '', (True,)),
        ('split', (S, S, S), (S, 1), 'dim', (True,), [1]),
        ('split', (S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],), 'size_list',
            (True, 'aten::split_with_sizes')),
        ('split', (S, S, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)], 2), 'size_list_dim',
            (True, 'aten::split_with_sizes'), [1]),
        ('split_with_sizes', (S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],), '', (True,)),
        ('split_with_sizes', (S, S, S), ([int(S / 3), S - int(S / 3), 0],), 'size_0', (True, )),
        ('split_with_sizes', (S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],), 'dim', (True, ), [1]),
        ('tensor_split', (S, S, S), (3,), 'sections', (False,)),
        ('tensor_split', (S, S, S), (3, 1), 'sections_dim', (False,), [1]),
        ('tensor_split', (S, S, S), ([2, 4],), 'indices', (False,)),
        ('tensor_split', (S, S, S), ([2, 4], 1), 'indices_dim', (False,), [1]),
        ('gather', (M, S), (0, gather_variable((S, S), 1, M, True)), 'dim0', (), [0]),
        ('gather', (M, S), (1, gather_variable((M, S // 2), 0, S, True)), 'dim1', (), [0]),
        ('gather', (), (0, torch.tensor([0], dtype=torch.int64)), 'scalar_input', (), [0]),
        ('gather', (S,), (0, torch.tensor(0, dtype=torch.int64)), 'scalar_index', (), [0]),
        ('gather', (), (0, torch.tensor(0, dtype=torch.int64)), 'scalar_both', (), [0]),
        ('scatter', (M, S), (0, gather_variable((S, S), 1, M), (S, S)), 'dim0', (), [0]),
        ('scatter', (M, S), (1, gather_variable((M, S // 2), 0, S), (M, S // 2)), 'dim1', (), [0]),
        ('scatter', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalartensor_all_dim0', (), [0]),
        ('scatter', (), (0, torch.tensor(0, dtype=torch.int64), 2.5), 'scalar_all_dim0', (), [0]),
        ('scatter_add', (M, S), (0, gather_variable((S, S), 1, M), (S, S)), 'dim0', (), [0]),
        ('scatter_add', (M, S), (1, gather_variable((M, S // 2), 0, S), (M, S // 2)), 'dim1', (), [0]),
        ('scatter_add', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim0', (), [0]),
        ('scatter_add', (M, S), (0, gather_variable((S, S), 1, M), (S, S)), 'alert_nondeterministic', (), [0],
            [expectedAlertNondeterministic('scatter_add_cuda_kernel', 'cuda')]),
        ('masked_select', (M, M), (mask_not_all_zeros((M, M)),)),
        ('masked_select', (M, M), (mask_not_all_zeros((M,)),), 'broadcast_rhs'),
        ('masked_select', (M,), (mask_not_all_zeros((M, M)),), 'broadcast_lhs'),
        ('masked_select', (M, 1, M), (mask_not_all_zeros((M, M)),),
         'broadcast_all'),
        ('masked_select', (), (torch.tensor(1, dtype=torch.bool),), 'scalar'),
        ('masked_select', (M, M), (torch.tensor(1, dtype=torch.bool),), 'scalar_broadcast_rhs'),
        ('masked_select', (), (mask_not_all_zeros((M, M)),), 'scalar_broadcast_lhs'),
        ('masked_fill', (M, M), (torch.BoolTensor(M, M).bernoulli_(), 10)),
        ('masked_fill', (M, M), (torch.BoolTensor(M, M).bernoulli_(), ()), 'tensor'),
        ('masked_fill', (M,), (torch.BoolTensor(M, M).bernoulli_(), 10), 'broadcast_lhs'),
        ('masked_fill', (M, M), (torch.BoolTensor(M,).bernoulli_(), 10), 'broadcast_rhs'),
        ('masked_fill', (), (torch.tensor(0, dtype=torch.bool).bernoulli_(), 10), 'scalar'),
        ('masked_fill', (), (torch.tensor(0, dtype=torch.bool).bernoulli_(), ()),
         'scalar_variable'),
        ('masked_fill', (M, M), (torch.tensor(0, dtype=torch.bool).bernoulli_(), 10),
         'scalar_broadcast_rhs'),
        ('masked_scatter', (M, M), (torch.BoolTensor(M, M).bernoulli_(), (M, M))),
        ('masked_scatter', (M,), (torch.BoolTensor(M, M).bernoulli_(), (M, M)),
         'broadcast_lhs'),
        ('masked_scatter', (M, M), (torch.BoolTensor(M,).bernoulli_(), (M, M)),
         'broadcast_rhs'),
        ('masked_scatter', (M, M), (bernoulli_scalar(), (M, M)), 'scalar'),
        ('masked_scatter', (M, M), (bernoulli_scalar(), (M, M)),
         'scalar_broadcast_rhs'),
        ('maximum', (S, S), ((S, S),)),
        ('minimum', (S, S), ((S, S),)),
        ('resize_', (S, S, S), (torch.Size([S * S, S])), 'fewer_dims'),
        ('resize_', (), (dont_convert(()),), 'scalar'),
        ('resize_', (), (torch.Size([1, 1, 1])), 'scalar_to_dims'),
        ('resize_as_', (), (non_differentiable(torch.tensor(5.)),), 'scalar'),
        ('resize_as_', (), (non_differentiable(torch.randn((1, 1, 1))),), 'scalar_to_dims'),
        ('resize_as_', (S, S, S), (non_differentiable(torch.randn(S * S, S)),)),
        ('sort', (S, M, S), NO_ARGS),
        ('sort', (S, M, S), (1,), 'dim'),
        ('sort', (S, M, S), (1, True), 'dim_desc'),
        ('sort', (), NO_ARGS, 'scalar'),
        ('sort', (), (0,), 'dim_scalar'),
        ('sort', (), (0, True), 'dim_desc_scalar'),
        ('topk', (S, M, S), (3,)),
        ('topk', (S, M, S), (3, 1), 'dim', (), [1]),
        ('topk', (S, M, S), (3, 1, True), 'dim_desc', (), [1]),
        ('topk', (S, M, S), (3, 1, True, True), 'dim_desc_sort', (), [1]),
        ('topk', (), (1,), 'scalar'),
        ('topk', (), (1, 0), 'dim_scalar', (), [1]),
        ('topk', (), (1, 0, True), 'dim_desc_scalar', (), [1]),
        ('topk', (), (1, 0, True, True), 'dim_desc_sort_scalar', (), [1]),
        ('take', (S, S, S), (torch.LongTensor([[-3, 2], [20, 2]]),)),
        ('take', (S, S, S), (torch.tensor(0, dtype=torch.int64),), 'scalar_index'),
        ('take', (), (torch.LongTensor([0]),), 'scalar_data'),
        ('take', (), (torch.tensor(0, dtype=torch.int64),), 'scalar_both'),
        ('where', (M, M), (mask_not_all_zeros((M, M)), (M, M)), '', (True,)),
        ('where', (M, 1, M), (mask_not_all_zeros((M, M)), (M, M, 1)), 'broadcast_all', (True,)),
        ('where', (), (bernoulli_scalar(), ()), 'scalar', (True,)),
        ('where', (M, 1, M), (bernoulli_scalar(), (M, M, 1)), 'scalar_broadcast_mask', (True,)),
        ('where', (), (mask_not_all_zeros((M, M)), ()), 'scalar_broadcast_non_mask', (True,)),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([1, 2]),)),
        ('__getitem__', torch.randn(S, S, S), (slice(0, 3),), 'slice'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([slice(0, 3), 1]),), 'slice_index'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 2, 3], [1, 3, 3], [0, 0, 2]]),), 'adv_index'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 0, 3], [1, 1, 3], [0, 0, 2]]),), 'adv_index_dup'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([slice(None), slice(None), [0, 3]]),), 'adv_index_end'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([slice(None), [0, 3], slice(None)]),), 'adv_index_mid'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], slice(None), slice(None)]),), 'adv_index_beg'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], [1, 2], slice(None)]),), 'adv_index_comb'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], ]),), 'adv_index_sub'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], slice(None)]),), 'adv_index_sub_2'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], Ellipsis]),), 'adv_index_sub_3'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 2, 3], [1, 3, 3],
                                                             torch.LongTensor([0, 0, 2])]),), 'adv_index_var'),
        ('to_sparse', (S, S), (), '', (), (), [], lambda x: x.to_dense()),
        ('triangular_solve', (S, M), ((S, S), ), '', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('kron', (S, S), ((M, L),))
    ]

def create_input(call_args, requires_grad=True, non_contiguous=False, call_kwargs=None, dtype=torch.double, device=None):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)

    def map_arg(arg):
        def maybe_non_contig(tensor):
            return tensor if not non_contiguous else make_non_contiguous(tensor)

        if isinstance(arg, torch.Size) or isinstance(arg, dont_convert):
            return arg
        elif isinstance(arg, tuple) and len(arg) == 0:
            var = torch.randn((), dtype=dtype, device=device)
            var.requires_grad = requires_grad
            return var
        elif isinstance(arg, tuple) and not isinstance(arg[0], torch.Tensor):
            return Variable(maybe_non_contig(torch.randn(*arg, dtype=dtype, device=device)), requires_grad=requires_grad)
        # double check casting
        elif isinstance(arg, non_differentiable):
            if isinstance(arg.tensor, torch.Tensor):
                return maybe_non_contig(arg.tensor.to(device=device))
            return maybe_non_contig(arg.tensor.to(device=device))
        elif isinstance(arg, torch.Tensor):
            if arg.dtype == torch.float:
                arg = arg.double()
            if arg.dtype == torch.cfloat:
                arg = arg.to(torch.cdouble)
            if arg.is_complex() != dtype.is_complex:
                raise RuntimeError("User provided tensor is real for a test that runs with complex dtype, ",
                                   "which is not supported for now")
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of v afterwards
            v = maybe_non_contig(arg).detach().to(device=device).clone()
            v.requires_grad = requires_grad and (v.is_floating_point() or v.is_complex())
            return v
        elif callable(arg):
            return map_arg(arg(dtype=dtype, device=device))
        else:
            return arg
    args_out = tuple(map_arg(arg) for arg in call_args)
    kwargs_out = {k: map_arg(v) for k, v in call_kwargs.items()} if call_kwargs else {}
    return args_out, kwargs_out


def _compare_trilu_indices(
        self, row, col, offset=0, dtype=torch.long, device='cpu'):
    if row == 0 or col == 0:
        # have to handle this separately as tril and triu does not take
        # empty matrix as input
        self.assertEqual(
            torch.empty(0, 2, dtype=dtype, device=device).transpose(0, 1),
            torch.tril_indices(row, col, offset, dtype=dtype, device=device))

        self.assertEqual(
            torch.empty(0, 2, dtype=dtype, device=device).transpose(0, 1),
            torch.triu_indices(row, col, offset, dtype=dtype, device=device))

    else:
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(
            torch.ones(row, col, device='cpu')
                 .tril(offset).nonzero().to(dtype).transpose(0, 1),
            torch.tril_indices(row, col, offset, dtype=dtype, device=device))

        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(
            torch.ones(row, col, device='cpu')
                 .tril(offset).nonzero().to(dtype).transpose(0, 1),
            torch.tril_indices(row, col, offset, dtype=dtype, device=device))


def _compare_large_trilu_indices(
        self, row, col, offset=0, dtype=torch.long, device='cpu'):
    l = torch.ones(row, col, dtype=dtype, device='cpu').tril(offset) \
             .nonzero()[-100:-1, :].transpose(0, 1).to(device)
    torch.cuda.empty_cache()

    r = torch.tril_indices(
        row, col, offset, dtype=dtype, device=device)[:, -100:-1]
    self.assertEqual(l, r)
    torch.cuda.empty_cache()

    l = torch.ones(row, col, dtype=dtype, device='cpu').triu(offset) \
             .nonzero()[-100:-1, :].transpose(0, 1).to(device)
    torch.cuda.empty_cache()

    r = torch.triu_indices(
        row, col, offset, dtype=dtype, device=device)[:, -100:-1]
    self.assertEqual(l, r)
    torch.cuda.empty_cache()

# (
#   row
#   col
#   offset (optional)
#   dtype (optional)
# )
tri_tests_args = [
    (1, 1),
    (3, 3),
    (3, 3, 1),
    (3, 3, 2),
    (3, 3, 200),
    (3, 3, -1),
    (3, 3, -2),
    (3, 3, -200),
    (0, 3, 0),
    (0, 3, 1),
    (0, 3, -1),
    (3, 0, 0),
    (3, 0, 1),
    (3, 0, -1),
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
    (3, 6, 0),
    (3, 6, 1),
    (3, 6, 3),
    (3, 6, 9),
    (3, 6, -1),
    (3, 6, -3),
    (3, 6, -9),
    (6, 3, 0),
    (6, 3, 1),
    (6, 3, 3),
    (6, 3, 9),
    (6, 3, -1),
    (6, 3, -3),
    (6, 3, -9),
    (258, 253, 1, torch.float32),
    (257, 258, 1, torch.float64),
    (258, 258, 1, torch.short),
    (3, 513, 1, torch.long),
    (513, 3, 1, torch.int),
    (513, 0, 1, torch.double),
    (1024, 1024),
    (1024, 1024, 500, torch.float32),
    (1024, 1024, 1023),
    (1024, 1024, -500),
    (1023, 1025),
    (1025, 1023, 1022),
    (1024, 1024, -500),
    (3, 2028),
    (3, 2028, 1),
    (3, 2028, -1),
    (2028, 3),
    (2028, 1),
    (2028, 1, -1)
]

tri_large_tests_args: List[Tuple[int, ...]] = [
    # Large test cases below are deliberately commented out to speed up CI
    # tests and to avoid OOM error. When modifying implementations of
    # tril_indices and triu_indices, please enable these tests and make sure
    # they pass.
    #
    # (1, 268435455),
    # (5000, 5000),
    # (10000, 10000),
    # (268435455, 1),
    # (134217727, 2, 1),
    # (2, 134217727, 1),
    # (536870901, 1),
    # (1, 536870901),
    # (268435455, 2, 1),
    # (2, 268435455, 1)
]


def run_additional_tri_tests(self, device):
    x = torch.ones(
        3, 3, dtype=torch.long, device=device, layout=torch.strided)
    l = x.tril(0).nonzero().transpose(0, 1)
    u = x.triu(0).nonzero().transpose(0, 1)
    self.assertEqual(l, torch.tril_indices(3, 3, device=device))
    self.assertEqual(
        l, torch.tril_indices(3, 3, device=device, layout=torch.strided))

    self.assertEqual(u, torch.triu_indices(3, 3, device=device))
    self.assertEqual(
        u, torch.triu_indices(3, 3, device=device, layout=torch.strided))

    self.assertRaises(
        RuntimeError,
        lambda: torch.triu_indices(
            1, 1, device=device, layout=torch.sparse_coo))

    self.assertRaises(
        RuntimeError,
        lambda: torch.tril_indices(
            1, 1, device=device, layout=torch.sparse_coo))


def unpack_variables(args):
    if istuple(args):
        return tuple(unpack_variables(elem) for elem in args)
    else:
        return args


EXCLUDE_FUNCTIONAL = {
    'addmm',
    'addmm_',
    'addbmm',
    'baddbmm',
    'addmv',
    'addmv_',
    'addr',
    'addr_',
    'reshape',
    'where'  # argument order
}
EXCLUDE_GRADCHECK: Dict[str, Any] = {
}
EXCLUDE_GRADGRADCHECK: Dict[str, Any] = {
}
EXCLUDE_GRADGRADCHECK_BY_TEST_NAME = {
    # *det methods uses svd in backward when matrix is not invertible. However,
    # svd backward is unstable unless the matrix has positive distinct singular
    # values. Generated random matrices satisfy this with high probability, but
    # we can't rely on it. So only test gradgrad on invertible test cases and
    # _distinct_singular_values.
    'test_det',
    'test_det_1x1',
    'test_det_symmetric',
    'test_det_symmetric_psd',
    'test_det_dim2_null',
    'test_det_rank1',
    'test_det_rank2',
    'test_det_batched',
    'test_det_batched_1x1',
    'test_det_batched_symmetric',
    'test_det_batched_symmetric_psd',
    # `other` expand_as(self, other) is not used in autograd.
    'test_expand_as',
    'test_logdet',
    'test_logdet_1x1',
    'test_logdet_symmetric',
    'test_logdet_batched',
    'test_logdet_batched_1x1',
    'test_logdet_batched_symmetric',
    'test_slogdet_1x1_neg_det',
    'test_slogdet_neg_det',
    'test_slogdet_symmetric',
    'test_slogdet_batched_1x1_neg_det',
    'test_slogdet_batched_symmetric',
    'test_cdist',
}


def exclude_tensor_method(name, test_name):
    # there are no tensor equivalents for these (inplace or out)
    exclude_all_tensor_method_by_test_name = {
        'test_slice',
        'test_where',
        'test_where_broadcast_all',
        'test_where_scalar',
        'test_where_scalar_broadcast_mask',
        'test_where_scalar_broadcast_non_mask',
        'test_var_mean_keepdim_dim_1d',
        'test_var_mean_keepdim_dim',
        'test_var_mean_dim_1d',
        'test_var_mean_dim',
        'test_var_mean',
        'test_std_mean_keepdim_dim_1d',
        'test_std_mean_keepdim_dim',
        'test_std_mean_dim_1d',
        'test_std_mean_dim',
        'test_std_mean',
        'test_view_as_complex',
        'test_view_as_real_complex',
        'test_real_complex',
        'test_imag_complex',
        'test_complex'
    }
    # there are no out-of-place tensor equivalents for these
    exclude_outplace_tensor_method = {
        'index_add',
        'index_copy',
        'index_fill',
        'masked_fill',
        'masked_scatter',
        'scatter',
        'scatter_add',
        'det',
    }
    if test_name in exclude_all_tensor_method_by_test_name:
        return True
    is_magic_method = name[:2] == '__' and name[-2:] == '__'
    is_inplace = name[-1] == "_" and not is_magic_method
    if not is_inplace and name in exclude_outplace_tensor_method:
        return True
    if 'fft.' in name:
        return True
    return False
