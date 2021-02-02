from functools import reduce, wraps
from itertools import product
from operator import mul, itemgetter
import collections
import operator

import torch
import numpy as np
from torch._six import inf, istuple
from torch.autograd import Variable
import collections.abc

from typing import List, Tuple, Dict, Any

from torch.testing import \
    (make_non_contiguous, _dispatch_dtypes, floating_types, floating_types_and,
     floating_and_complex_types, floating_and_complex_types_and,
     all_types_and_complex_and, all_types_and, all_types_and_complex)
from torch.testing._internal.common_device_type import \
    (skipIf, skipCUDAIfNoMagma, skipCPUIfNoLapack, skipCPUIfNoMkl,
     skipCUDAIfRocm, expectedAlertNondeterministic, precisionOverride)
from torch.testing._internal.common_cuda import CUDA11OrLater
from torch.testing._internal.common_utils import \
    (prod_single_zero, random_square_matrix_of_rank,
     random_symmetric_matrix, random_symmetric_psd_matrix,
     random_symmetric_pd_matrix, make_nonzero_det,
     random_fullrank_matrix_distinct_singular_value, set_rng_seed,
     TEST_WITH_ROCM, IS_WINDOWS, IS_MACOS, make_tensor, TEST_SCIPY,
     torch_to_numpy_dtype_dict, slowTest)

from distutils.version import LooseVersion

if TEST_SCIPY:
    import scipy.special


class DecorateInfo(object):
    """Describes which test, or type of tests, should be wrapped in the given
       decorators when testing an operator. Any test that matches all provided
       arguments will be decorated. The decorators will only be applied if the
       active_if argument is True."""

    __slots__ = ['decorators', 'cls_name', 'test_name', 'device_type', 'dtypes', 'active_if']

    def __init__(self, decorators, cls_name=None, test_name=None, *,
                 device_type=None, dtypes=None, active_if=True):
        self.decorators = list(decorators) if isinstance(decorators, collections.abc.Sequence) else [decorators]
        self.cls_name = cls_name
        self.test_name = test_name
        self.device_type = device_type
        self.dtypes = dtypes
        self.active_if = active_if

    def is_active(self, cls_name, test_name, device_type, dtype):
        return (
            self.active_if and
            (self.cls_name is None or self.cls_name == cls_name) and
            (self.test_name is None or self.test_name == test_name) and
            (self.device_type is None or self.device_type == device_type) and
            (self.dtypes is None or dtype in self.dtypes)
        )


class SkipInfo(DecorateInfo):
    """Describes which test, or type of tests, should be skipped when testing
       an operator. Any test that matches all provided arguments will be skipped.
       The skip will only be checked if the active_if argument is True."""

    def __init__(self, cls_name=None, test_name=None, *,
                 device_type=None, dtypes=None, active_if=True):
        super().__init__(decorators=skipIf(True, "Skipped!"), cls_name=cls_name,
                         test_name=test_name, device_type=device_type, dtypes=dtypes,
                         active_if=active_if)

class SampleInput(object):
    """Represents sample inputs to a function."""

    # output_process_fn_grad is a function that modifies the output of op compatible with input
    __slots__ = ['input', 'args', 'kwargs', 'output_process_fn_grad']

    def __init__(self, input, *, args=tuple(), kwargs=None, output_process_fn_grad=None):
        # test_ops.py expects input to be a tuple
        self.input = input if isinstance(input, tuple) else (input,)
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
        self.output_process_fn_grad = output_process_fn_grad

    def __repr__(self):
        arguments = [
            f'input[{len(self.input)}]',
            f'args={self.args}' if len(self.args) > 0 else None,
            f'kwargs={self.kwargs}' if len(self.kwargs) > 0 else None,
            (f'output_process_fn_grad={self.output_process_fn_grad}'
             if self.output_process_fn_grad is not None else None)]

        return f'SampleInput({", ".join(a for a in arguments if a is not None)})'


_NOTHING = object()  # Unique value to distinguish default from anything else


# Extension of getattr to support qualified names
# e.g. _getattr_qual(torch, 'linalg.norm') -> torch.linalg.norm
def _getattr_qual(obj, name, default=_NOTHING):
    try:
        for path in name.split('.'):
            obj = getattr(obj, path)
        return obj
    except AttributeError:
        if default is not _NOTHING:
            return default
        else:
            raise


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
                 default_test_dtypes=None,  # dtypes to test with by default. Gets intersected
                                            # with the dtypes support on the tested device
                 test_inplace_grad=True,  # whether to gradcheck and gradgradcheck the inplace variant
                 test_complex_grad=True,  # whether to gradcheck and gradgradcheck for complex dtypes
                 skip_bfloat16_grad=False,  # whether to skip grad and gradgradcheck for bfloat16 dtype
                 assert_autodiffed=False,  # if a op's aten::node is expected to be symbolically autodiffed
                 autodiff_nonfusible_nodes=None,  # a list of strings with node names that are expected to be in a
                                                  # DifferentiableGraph when autodiffed. Ex: ['aten::add', 'aten::mm'],
                                                  # default is populated to be ['aten::(name of Python operator)']
                 autodiff_fusible_nodes=None,  # a list of strings with node names that are expected to be in FusionGroups
                                               # inside of DifferentiableGraphs when this operation is autodiffed.
                                               # Ex: ['aten::add', 'aten::mm'], defaults to an empty list
                                               # Note: currently no ops use fusible nodes
                 output_func=lambda x: x,  # fn mapping output to part that should be gradcheck'ed
                 supports_tensor_out=True,  # whether the op supports the out kwarg, returning a Tensor
                 skips=tuple(),  # information about which tests to skip
                 decorators=None,  # decorators to apply to generated tests
                 safe_casts_outputs=False,  # whether op allows safe casting when writing to out arguments
                 sample_inputs_func=None,  # function to generate sample inputs
                 aten_name=None,  # name of the corresponding aten:: operator
                 variant_test_name='',  # additional string to include in the test name
                 supports_sparse=False,  # supported for sparse
                 check_batched_grad=True,  # check batched grad when doing gradcheck
                 check_batched_gradgrad=True,  # check batched grad grad when doing gradgradcheck
                 ):

        # Validates the dtypes are generated from the dispatch-related functions
        for dtype_list in (dtypes, dtypesIfCPU, dtypesIfCUDA, dtypesIfROCM):
            assert isinstance(dtype_list, (_dispatch_dtypes, type(None)))

        self.name = name
        self.aten_name = aten_name if aten_name is not None else name
        self.variant_test_name = variant_test_name

        self.dtypes = set(dtypes)
        self.dtypesIfCPU = set(dtypesIfCPU) if dtypesIfCPU is not None else self.dtypes
        self.dtypesIfCUDA = set(dtypesIfCUDA) if dtypesIfCUDA is not None else self.dtypes
        self.dtypesIfROCM = set(dtypesIfROCM) if dtypesIfROCM is not None else self.dtypes
        self._default_test_dtypes = set(default_test_dtypes) if default_test_dtypes is not None else None

        # NOTE: if the op is unspecified it is assumed to be under the torch namespace
        self.op = op if op else _getattr_qual(torch, self.name)
        self.method_variant = getattr(torch.Tensor, name, None)
        inplace_name = name + "_"
        self.inplace_variant = getattr(torch.Tensor, inplace_name, None)
        self.operator_variant = getattr(operator, name, None)
        self.skip_bfloat16_grad = skip_bfloat16_grad

        self.test_inplace_grad = test_inplace_grad
        self.test_complex_grad = test_complex_grad
        self.supports_tensor_out = supports_tensor_out
        self.safe_casts_outputs = safe_casts_outputs

        self.skips = skips
        self.decorators = decorators
        self.output_func = output_func
        self.sample_inputs_func = sample_inputs_func

        self.assert_autodiffed = assert_autodiffed
        self.autodiff_fusible_nodes = autodiff_fusible_nodes if autodiff_fusible_nodes else []
        if autodiff_nonfusible_nodes is None:
            self.autodiff_nonfusible_nodes = ['aten::' + self.name]
        else:
            self.autodiff_nonfusible_nodes = autodiff_nonfusible_nodes
        self.supports_sparse = supports_sparse
        self.check_batched_grad = check_batched_grad
        self.check_batched_gradgrad = check_batched_gradgrad

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

    def get_operator_variant(self):
        """Returns operator variant of the operator, e.g. operator.neg
        Returns None if the operator has no operator variant.
        """
        return self.operator_variant

    def sample_inputs(self, device, dtype, requires_grad=False):
        """Returns an iterable of SampleInputs.

        These samples should be sufficient to test the function works correctly
        with autograd, TorchScript, etc.
        """
        return self.sample_inputs_func(self, device, dtype, requires_grad)

    # Returns True if the test should be skipped and False otherwise
    def should_skip(self, cls_name, test_name, device_type, dtype):
        return any(si.is_active(cls_name, test_name, device_type, dtype)
                   for si in self.skips)

    def supported_dtypes(self, device_type):
        if device_type == 'cpu':
            return self.dtypesIfCPU
        if device_type == 'cuda':
            return self.dtypesIfROCM if TEST_WITH_ROCM else self.dtypesIfCUDA
        else:
            return self.dtypes


    def supports_dtype(self, dtype, device_type):
        return dtype in self.supported_dtypes(device_type)

    def default_test_dtypes(self, device_type):
        """Returns the default dtypes used to test this operator on the device.

        Equal to the operator's default_test_dtypes filtered to remove dtypes
        not supported by the device.
        """
        supported = self.supported_dtypes(device_type)
        return (supported if self._default_test_dtypes is None
                else supported.intersection(self._default_test_dtypes))


L = 20
M = 10
S = 5


def sample_inputs_unary(op_info, device, dtype, requires_grad):
    low, high = op_info.domain
    low = low if low is None else low + op_info._domain_eps
    high = high if high is None else high - op_info._domain_eps

    return (SampleInput(make_tensor((L,), device, dtype,
                                    low=low, high=high,
                                    requires_grad=requires_grad)),
            SampleInput(make_tensor((), device, dtype,
                                    low=low, high=high,
                                    requires_grad=requires_grad)))

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
                 supports_complex_to_float=False,  # op supports casting from complex input to real output safely eg. angle
                 sample_inputs_func=sample_inputs_unary,
                 supports_sparse=False,
                 **kwargs):
        super(UnaryUfuncInfo, self).__init__(name,
                                             dtypes=dtypes,
                                             dtypesIfCPU=dtypesIfCPU,
                                             dtypesIfCUDA=dtypesIfCUDA,
                                             dtypesIfROCM=dtypesIfROCM,
                                             sample_inputs_func=sample_inputs_func,
                                             supports_sparse=supports_sparse,
                                             **kwargs)
        self.ref = ref
        self.domain = domain
        self.handles_large_floats = handles_large_floats
        self.handles_extremals = handles_extremals
        self.handles_complex_extremals = handles_complex_extremals
        self.supports_complex_to_float = supports_complex_to_float

        # Epsilon to ensure grad and gradgrad checks don't test values
        #   outside a function's domain.
        self._domain_eps = 1e-5

def sample_inputs_tensor_split(op_info, device, dtype, requires_grad):
    return (SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(torch.tensor([1, 2, 3]),),),
            SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(torch.tensor(1),),),
            SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(torch.tensor([1, 2, 3]),),
                        kwargs=dict(dim=1)),)

def sample_inputs_linalg_norm(op_info, device, dtype, requires_grad):
    test_sizes = [
        (S,),
        (0,),
        (S, S),
        (0, 0),
        (S, 0),
        (0, S),
        (S, S, S),
        (0, S, S),
        (S, 0, S),
        (0, 0, 0),
    ]

    vector_ords = (None, 0, 0.5, 1, 2, 3.5, inf, -0.5, -1, -2, -3.5, -inf)
    matrix_ords = (None, 'fro', 'nuc', 1, 2, inf, -1, -2, -inf)

    inputs = []

    is_dtype_half = dtype in [torch.float16, torch.bfloat16]

    for test_size in test_sizes:
        is_vector_norm = len(test_size) == 1
        is_matrix_norm = len(test_size) == 2

        for keepdim in [False, True]:
            inputs.append(SampleInput(
                make_tensor(
                    test_size, device, dtype, low=None, high=None,
                    requires_grad=requires_grad),
                kwargs=dict(
                    keepdim=keepdim)))

            if not (is_vector_norm or is_matrix_norm):
                continue

            ords = vector_ords if is_vector_norm else matrix_ords

            for ord in ords:
                # TODO: remove this check when `max` is implemented for
                #       float16 and bfloat16. Issue:
                #       https://github.com/pytorch/pytorch/issues/50790
                if is_vector_norm and is_dtype_half and ord in [inf, -inf]:
                    continue

                inputs.append(SampleInput(
                    make_tensor(
                        test_size, device, dtype,
                        low=None, high=None,
                        requires_grad=requires_grad),
                    args=(ord,),
                    kwargs=dict(
                        keepdim=keepdim)))

                if ord in ['nuc', 'fro']:
                    inputs.append(SampleInput(
                        make_tensor(
                            test_size, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                        kwargs=dict(
                            ord=ord,
                            keepdim=keepdim,
                            dim=(0, 1))))
        return inputs

def sample_inputs_slogdet(op_info, device, dtype, requires_grad):
    # original test cases from 'method_tests' have too many test_inputs
    # we don't actually need all of them to check the autograd and jit correctness
    # sample inputs with shapes 0x0, 0xSxS, 2x0x0 are added
    test_inputs = (
        torch.randn(0, 0, dtype=dtype, device=device),  # '0x0'
        torch.randn(S, S, dtype=dtype, device=device),  # 'SxS'
        torch.randn(0, S, S, dtype=dtype, device=device),  # 'zero_batched_SxS'
        torch.randn(2, 0, 0, dtype=dtype, device=device),  # 'batched_0x0'
        torch.randn(2, S, S, dtype=dtype, device=device),  # 'batched_SxS'
    )
    out = []
    for a in test_inputs:
        a.requires_grad = requires_grad
        out.append(SampleInput(a))
    return out

def sample_inputs_addmm(op_info, device, dtype, requires_grad):
    input = SampleInput((make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                         make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                         make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=False)))
    if dtype.is_complex:
        another_input = SampleInput((make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                                     make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                                     make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=False)),
                                    kwargs=dict(beta=1 + 2j, alpha=2 + 3j))
        return (input, another_input)
    else:
        return (input, )

def sample_inputs_addr(op_info, device, dtype, requires_grad):
    input1 = SampleInput((make_tensor((S, M), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad),
                          make_tensor((S, ), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad),
                          make_tensor((M, ), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad)))

    input2 = SampleInput((make_tensor((), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad),
                          make_tensor((S, ), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad),
                          make_tensor((M, ), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad)))
    if dtype.is_complex:
        alpha, beta = 0.1 + 0.3j, 0.4 + 0.6j
    elif dtype.is_floating_point:
        alpha, beta = 0.2, 0.6
    else:
        alpha, beta = 2, 3

    input3 = SampleInput((make_tensor((S, M), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad),
                          make_tensor((S, ), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad),
                          make_tensor((M, ), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad)),
                         kwargs=dict(beta=beta, alpha=alpha))

    input4 = SampleInput((make_tensor((), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad),
                          make_tensor((S, ), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad),
                          make_tensor((M, ), device, dtype,
                          low=None, high=None,
                          requires_grad=requires_grad)),
                         kwargs=dict(beta=beta, alpha=alpha))

    return (input1, input2, input3, input4)

def sample_inputs_xlogy(self, device, dtype, requires_grad):
    return (SampleInput((make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                         make_tensor((S, S), device, dtype,
                                     low=0, high=None,
                                     requires_grad=requires_grad))),)

def sample_inputs_trace(self, device, dtype, requires_grad):
    return (SampleInput((make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad))),)

def sample_inputs_linalg_inv(op_info, device, dtype, requires_grad=False):
    """
    This function generates always invertible input for torch.linalg.inv using
    random_fullrank_matrix_distinct_singular_value.
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (2, 3) - 2x3 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

    batches = [(), (0, ), (2, ), (2, 3)]
    ns = [0, 5]
    out = []
    for batch, n in product(batches, ns):
        a = random_fullrank_matrix_distinct_singular_value(n, *batch, dtype=dtype).to(device)
        a.requires_grad = requires_grad
        out.append(SampleInput(a))
    return out

def np_sinc_with_fp16_as_fp32(x):
    # Wraps numpy's sinc function so that fp16 values are promoted to fp32
    # before sinc is invoked. Context: numpy's sinc returns NaN when evaluated
    # at 0 for fp16.
    if x.dtype == np.float16:
        return np.sinc(x.astype(np.float32))
    else:
        return np.sinc(x)

def sample_inputs_broadcast_to(op_info, device, dtype, requires_grad):
    test_cases = (
        ((S, 1, 1), (S, S, S)),
        ((S, 1, S), (S, S, S)),
        ((S, 1), (S, S, S)),
        ((1,), (S, S, S)),
        ((1, S), (1, 1, S)),
        ((), ()),
        ((), (1, 3, 2)),
    )

    return tuple(SampleInput((make_tensor(size, device, dtype,
                                          low=None, high=None,
                                          requires_grad=requires_grad), shape))
                 for size, shape in test_cases)

def sample_inputs_stack(op_info, device, dtype, requires_grad):
    return (SampleInput((make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad)), kwargs=dict(idx=0)),)

def sample_inputs_hstack_dstack_vstack(op_info, device, dtype, requires_grad):
    return (SampleInput((make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad))),)

def sample_inputs_gather(op_info, device, dtype, requires_grad):
    return (SampleInput((make_tensor((M, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        0, gather_variable((S, S), 1, M, True, device=device))),
            SampleInput((make_tensor((M, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        1, gather_variable((M, S // 2), 0, S, True, device=device))),
            SampleInput((make_tensor((), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        0, torch.tensor([0], dtype=torch.int64, device=device))),
            SampleInput((make_tensor((S,), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        0, torch.tensor(0, dtype=torch.int64, device=device))),
            SampleInput((make_tensor((), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        0, torch.tensor(0, dtype=torch.int64, device=device))),
            )


def sample_inputs_index_select(op_info, device, dtype, requires_grad):
    return (SampleInput((make_tensor((S, S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        0, index_variable(2, S, device=device))),
            SampleInput((make_tensor((), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        0, torch.tensor([0], dtype=torch.int64, device=device))),
            SampleInput((make_tensor((), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        0, torch.tensor(0, dtype=torch.int64, device=device))),
            )

def sample_inputs_index_fill(op_info, device, dtype, requires_grad):
    samples = []
    t = make_tensor((S, S, S), device, dtype,
                    low=None, high=None,
                    requires_grad=requires_grad)
    fill_val = torch.tensor(-1 + 1j if t.is_complex() else -1)
    # non-contiguous input
    t01 = t.transpose(0, 1)
    t02 = t.transpose(0, 2)
    t12 = t.transpose(1, 2)
    idx = index_variable(1, S, device=device)
    # non-contiguous index
    idx_nonctg = torch.empty_strided((S,), (2,), device=device, dtype=torch.int64)
    idx_nonctg.copy_(idx)
    for d in range(t.dim()):
        for tensor in [t, t01, t02, t12]:
            samples.append(SampleInput((tensor, d, idx, fill_val)))
            samples.append(SampleInput((tensor, d, -idx - 1, fill_val)))
            samples.append(SampleInput((tensor, d, idx_nonctg, fill_val)))
    return samples

def sample_movedim_moveaxis(op_info, device, dtype, requires_grad):
    return (SampleInput((make_tensor((4, 3, 2, 1), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        (0, 1, 2, 3), (3, 2, 1, 0))),
            SampleInput((make_tensor((4, 3, 2, 1), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad),
                        (0, -1, -2, -3), (-3, -2, -1, -0))))


def sample_repeat_tile(op_info, device, dtype, requires_grad):
    rep_dims = ((), (0, ), (1, ), (0, 2), (1, 1), (2, 3), (2, 3, 2), (0, 2, 3), (2, 1, 1, 1),)
    shapes = ((), (0,), (2,), (3, 0), (3, 2), (3, 0, 1))

    if requires_grad:
        # Tests for variant_consistency_jit, grad, gradgrad
        # are slower. Use smaller bags of `rep_dims` and `shapes`
        # in this case.
        rep_dims = ((), (0, ), (0, 2), (1, 1), (2, 3), (1, 3, 2), (3, 1, 1))  # type: ignore
        shapes = ((), (0,), (2,), (3, 2))  # type: ignore

    tensors = [make_tensor(shape, device, dtype,
                           low=None, high=None,
                           requires_grad=requires_grad) for shape in shapes]

    samples = []
    for rep_dim, tensor in product(rep_dims, tensors):
        for t in (tensor, tensor.T):
            if op_info.name == 'repeat' and len(rep_dim) >= t.dim():
                # `torch.repeat` errors for `len(rep_dims) < t.dim()`,
                # so we filter such combinations.
                samples.append(SampleInput((t, rep_dim),))
            elif op_info.name == 'tile':
                samples.append(SampleInput((t, rep_dim),))

    return samples

def np_unary_ufunc_integer_promotion_wrapper(fn):
    # Wrapper that passes PyTorch's default scalar
    #   type as an argument to the wrapped NumPy
    #   unary ufunc when given an integer input.
    #   This mimicks PyTorch's integer->floating point
    #   type promotion.
    #
    # This is necessary when NumPy promotes
    #   integer types to double, since PyTorch promotes
    #   integer types to the default scalar type.

    # Helper to determine if promotion is needed
    def is_integral(dtype):
        return dtype in [np.bool, np.uint8, np.int8, np.int16, np.int32, np.int64]

    # NOTE: Promotion in PyTorch is from integer types to the default dtype
    np_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]

    @wraps(fn)
    def wrapped_fn(x):
        if is_integral(x.dtype):
            return fn(x, dtype=np_dtype)
        return fn(x)

    return wrapped_fn


# Metadata class for Fast Fourier Transforms in torch.fft.
class SpectralFuncInfo(OpInfo):
    """Operator information for torch.fft transforms. """

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref=None,  # Reference implementation (probably in np.fft namespace)
                 dtypes=floating_and_complex_types(),
                 ndimensional: bool,  # Whether dim argument can be a tuple
                 decorators=None,
                 **kwargs):
        decorators = list(decorators) if decorators is not None else []
        decorators += [
            skipCPUIfNoMkl,
            skipCUDAIfRocm,
            # gradgrad is quite slow
            DecorateInfo(slowTest, 'TestGradients', 'test_fn_gradgrad'),
        ]

        super().__init__(name=name,
                         dtypes=dtypes,
                         decorators=decorators,
                         **kwargs)
        self.ref = ref if ref is not None else _getattr_qual(np, name)
        self.ndimensional = ndimensional


    def sample_inputs(self, device, dtype, requires_grad=False):
        nd_tensor = make_tensor((S, S + 1, S + 2), device, dtype, low=None, high=None,
                                requires_grad=requires_grad)
        tensor = make_tensor((31,), device, dtype, low=None, high=None,
                             requires_grad=requires_grad)

        if self.ndimensional:
            return [
                SampleInput(nd_tensor, kwargs=dict(s=(3, 10), dim=(1, 2), norm='ortho')),
                SampleInput(nd_tensor, kwargs=dict(norm='ortho')),
                SampleInput(nd_tensor, kwargs=dict(s=(8,))),
                SampleInput(tensor),

                *(SampleInput(nd_tensor, kwargs=dict(dim=dim))
                  for dim in [-1, -2, -3, (0, -1)]),
            ]
        else:
            return [
                SampleInput(nd_tensor, kwargs=dict(n=10, dim=1, norm='ortho')),
                SampleInput(nd_tensor, kwargs=dict(norm='ortho')),
                SampleInput(nd_tensor, kwargs=dict(n=7)),
                SampleInput(tensor),

                *(SampleInput(nd_tensor, kwargs=dict(dim=dim))
                  for dim in [-1, -2, -3]),
            ]


class ShapeFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for Shape manipulating operations like tile and roll"""
    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref,  # a reference function
                 dtypes=floating_types(),
                 dtypesIfCPU=None,
                 dtypesIfCUDA=None,
                 dtypesIfROCM=None,
                 sample_inputs_func=None,
                 **kwargs):
        super(ShapeFuncInfo, self).__init__(name,
                                            dtypes=dtypes,
                                            dtypesIfCPU=dtypesIfCPU,
                                            dtypesIfCUDA=dtypesIfCUDA,
                                            dtypesIfROCM=dtypesIfROCM,
                                            sample_inputs_func=sample_inputs_func,
                                            **kwargs)
        self.ref = ref


class HermitianOpInfo(OpInfo):
    """Operator information for Hermitian functions
    These are functions that take Hermitian matrices as input.
    They require a modified function to be tested for gradcheck, because the finite-difference algorithm
    for calculating derivatives does not preserve the Hermitian property of the input and returning incorrect results.
    """

    def get_op(self):
        """
        Returns the function variant of the operator, torch.<op_name>,
        compatible with gradcheck for Hermitian functions.
        It works only for single input argument.
        """
        def hermitian_func(non_hermitian_input, **kwargs):
            hermitian_input = non_hermitian_input + non_hermitian_input.conj().transpose(-2, -1)
            return self.op(hermitian_input, **kwargs)

        return hermitian_func


class TriangularOpInfo(OpInfo):
    """Operator information for function that take lower or upper triangular matrices as input.
    They require a modified function to be tested for gradcheck, because the finite-difference algorithm
    for calculating derivatives does not preserve the triangular property of the input and returning incorrect results.
    """

    def get_op(self):
        """
        Returns the function variant of the operator, torch.<op_name>,
        compatible with gradcheck for triangular input functions.
        It works only for single input argument and upper kwarg
        """
        def triangular_func(non_triangular_input, upper=False):
            if upper:
                triangular_input = non_triangular_input.triu()
            else:
                triangular_input = non_triangular_input.tril()
            return self.op(triangular_input, upper=upper)

        return triangular_func

    def get_method(self):
        """
        Returns the method variant of the operator
        compatible with gradcheck for triangular input functions.
        It works only for single input argument and upper kwarg
        """
        def triangular_func(non_triangular_input, upper=False):
            if upper:
                triangular_input = non_triangular_input.triu()
            else:
                triangular_input = non_triangular_input.tril()
            return self.method_variant(triangular_input, upper=upper)

        return triangular_func

    def sample_inputs(self, device, dtype, requires_grad=False):
        """
        This function generates Cholesky factors of positive-definite (non-singular) Hermitian (symmetric) matrices
        for cholesky_inverse.
        """
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix
        inputs = (
            torch.zeros(0, 0, dtype=dtype, device=device),  # 0x0 matrix
            torch.zeros(0, 2, 2, dtype=dtype, device=device),  # zero batch of matrices
            random_hermitian_pd_matrix(S, dtype=dtype, device=device),  # single matrix
            random_hermitian_pd_matrix(S, 2, dtype=dtype, device=device),  # batch of matrices
        )
        test_cases = (torch.linalg.cholesky(a) for a in inputs)
        out = []
        for a in test_cases:
            a.requires_grad = requires_grad
            out.append(SampleInput(a))
            out.append(SampleInput(a, kwargs=dict(upper=True)))
        return out


def sample_inputs_linalg_pinv(op_info, device, dtype, requires_grad=False):
    """
    This function generates input for torch.linalg.pinv with distinct singular values so that autograd is always stable
    Implementation of torch.linalg.pinv depends on torch.svd and torch.linalg.eigh, therefore it's sufficient to
    check only square S x S matrix and the batched (3 x S x S) input.
    """
    from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

    test_cases = (
        random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),  # single matrix
        random_fullrank_matrix_distinct_singular_value(S, 3, dtype=dtype).to(device),  # batch of matrices
    )

    out = []
    for a in test_cases:
        a.requires_grad = requires_grad
        out.append(SampleInput(a))
    return out


def sample_inputs_linalg_pinv_hermitian(op_info, device, dtype, requires_grad=False):
    """
    This function generates input for torch.linalg.pinv with hermitian=True keyword argument.
    """
    out = sample_inputs_linalg_pinv(op_info, device, dtype, requires_grad)
    for o in out:
        o.kwargs = {"hermitian": True}
    return out

def sample_inputs_linalg_solve(op_info, device, dtype, requires_grad=False, vector_rhs_allowed=True):
    """
    This function generates always solvable input for torch.linalg.solve
    Using random_fullrank_matrix_distinct_singular_value gives a non-singular (=invertible, =solvable) matrices 'a'.
    The first input to torch.linalg.solve is generated as the itertools.product of 'batches' and 'ns'.
    The second input is generated as the product of 'batches', 'ns' and 'nrhs'.
    In total this function generates 18 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices.
    'ns' gives 0x0 and 5x5 matrices.
    and 'nrhs' controls the number of vectors to solve for:
        () - using 1 as the number of vectors implicitly
        (1,) - same as () but explicit
        (3,) - solve for 3 vectors.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    'vector_rhs_allowed' controls whether to include nrhs = () to the list of SampleInputs.
    torch.solve / triangular_solve / cholesky_solve (opposed to torch.linalg.solve) do not allow
    1D tensors (vectors) as the right-hand-side.
    Once torch.solve / triangular_solve / cholesky_solve and its testing are removed,
    'vector_rhs_allowed' may be removed here as well.
    """
    from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

    batches = [(), (0, ), (2, )]
    ns = [0, 5]
    if vector_rhs_allowed:
        nrhs = [(), (1,), (3,)]
    else:
        nrhs = [(1,), (3,)]
    out = []
    for n, batch, rhs in product(ns, batches, nrhs):
        a = random_fullrank_matrix_distinct_singular_value(n, *batch, dtype=dtype).to(device)
        a.requires_grad = requires_grad
        b = torch.randn(*batch, n, *rhs, dtype=dtype, device=device)
        b.requires_grad = requires_grad
        out.append(SampleInput((a, b)))
    return out


def sample_inputs_legacy_solve(op_info, device, dtype, requires_grad=False):
    """
    This function generates always solvable input for legacy solve functions
    (the ones that are not in torch.linalg module).
    The difference from sample_inputs_linalg_solve is that here the right-hand-side of A x = b equation
    should have b.ndim >= 2, vectors are not allowed.
    Also the arguments order is swapped.
    """
    out = sample_inputs_linalg_solve(
        op_info, device, dtype, requires_grad=requires_grad, vector_rhs_allowed=False
    )
    for sample in out:
        sample.input = tuple(reversed(sample.input))
    return out


def sample_inputs_std_var(op_info, device, dtype, requires_grad):
    tensor_nd = make_tensor((S, S, S), device=device, dtype=dtype,
                            low=None, high=None, requires_grad=requires_grad)
    tensor_1d = make_tensor((S,), device=device, dtype=dtype,
                            low=None, high=None, requires_grad=requires_grad)

    return [
        SampleInput(tensor_nd),
        SampleInput(tensor_nd, kwargs=dict(dim=1)),
        SampleInput(tensor_nd, kwargs=dict(dim=1, unbiased=True, keepdim=True)),
        SampleInput(tensor_1d, kwargs=dict(dim=0, unbiased=True, keepdim=True)),
        SampleInput(tensor_1d, kwargs=dict(dim=0, unbiased=False, keepdim=False)),

        SampleInput(tensor_nd, kwargs=dict(dim=(1,), correction=S // 2)),
        SampleInput(tensor_nd, kwargs=dict(correction=0, keepdim=True)),
    ]


def _sample_inputs_svd(op_info, device, dtype, requires_grad=False, is_linalg_svd=False):
    """
    This function generates input for torch.svd with distinct singular values so that autograd is always stable.
    Matrices of different size:
        square matrix - S x S size
        tall marix - S x (S-2)
        wide matrix - (S-2) x S
    and batched variants of above are generated.
    Each SampleInput has a function 'output_process_fn_grad' attached to it that is applied on the output of torch.svd
    It is needed for autograd checks, because backward of svd doesn't work for an arbitrary loss function.
    """
    from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

    # svd and linalg.svd returns V and V.conj().T, respectively. So we need to slice
    # along different dimensions when needed (this is used by
    # test_cases2:wide_all and wide_all_batched below)
    if is_linalg_svd:
        def slice_V(v):
            return v[..., :(S - 2), :]

        def uv_loss(usv):
            u00 = usv[0][0, 0]
            v00_conj = usv[2][0, 0]
            return u00 * v00_conj
    else:
        def slice_V(v):
            return v[..., :, :(S - 2)]

        def uv_loss(usv):
            u00 = usv[0][0, 0]
            v00_conj = usv[2][0, 0].conj()
            return u00 * v00_conj

    test_cases1 = (  # some=True (default)
        # loss functions for complex-valued svd have to be "gauge invariant",
        # i.e. loss functions shouldn't change when sigh of the singular vectors change.
        # the simplest choice to satisfy this requirement is to apply 'abs'.
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),
            lambda usv: usv[1]),  # 'check_grad_s'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),
            lambda usv: abs(usv[0])),  # 'check_grad_u'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),
            lambda usv: abs(usv[2])),  # 'check_grad_v'
        # this test is important as it checks the additional term that is non-zero only for complex-valued inputs
        # and when the loss function depends both on 'u' and 'v'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),
            uv_loss),  # 'check_grad_uv'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device)[:(S - 2)],
            lambda usv: (abs(usv[0]), usv[1], abs(usv[2][..., :, :(S - 2)]))),  # 'wide'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device)[:, :(S - 2)],
            lambda usv: (abs(usv[0]), usv[1], abs(usv[2]))),  # 'tall'
        (random_fullrank_matrix_distinct_singular_value(S, 2, dtype=dtype).to(device),
            lambda usv: (abs(usv[0]), usv[1], abs(usv[2]))),  # 'batched'
        (random_fullrank_matrix_distinct_singular_value(S, 2, dtype=dtype).to(device)[..., :(S - 2), :],
            lambda usv: (abs(usv[0]), usv[1], abs(usv[2]))),  # 'wide_batched'
        (random_fullrank_matrix_distinct_singular_value(S, 2, dtype=dtype).to(device)[..., :, :(S - 2)],
            lambda usv: (abs(usv[0]), usv[1], abs(usv[2]))),  # 'tall_batched'
    )
    test_cases2 = (  # some=False
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device)[:(S - 2)],
            lambda usv: (abs(usv[0]), usv[1], abs(slice_V(usv[2])))),  # 'wide_all'
        (random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device)[:, :(S - 2)],
            lambda usv: (abs(usv[0][:, :(S - 2)]), usv[1], abs(usv[2]))),  # 'tall_all'
        (random_fullrank_matrix_distinct_singular_value(S, 2, dtype=dtype).to(device)[..., :(S - 2), :],
            lambda usv: (abs(usv[0]), usv[1], abs(slice_V(usv[2])))),  # 'wide_all_batched'
        (random_fullrank_matrix_distinct_singular_value(S, 2, dtype=dtype).to(device)[..., :, :(S - 2)],
            lambda usv: (abs(usv[0][..., :, :(S - 2)]), usv[1], abs(usv[2]))),  # 'tall_all_batched'
    )

    out = []
    for a, out_fn in test_cases1:
        a.requires_grad = requires_grad
        if is_linalg_svd:
            kwargs = {'full_matrices': False}
        else:
            kwargs = {'some': True}
        out.append(SampleInput(a, kwargs=kwargs, output_process_fn_grad=out_fn))

    for a, out_fn in test_cases2:
        a.requires_grad = requires_grad
        if is_linalg_svd:
            kwargs = {'full_matrices': True}
        else:
            kwargs = {'some': False}
        out.append(SampleInput(a, kwargs=kwargs, output_process_fn_grad=out_fn))

    return out

def sample_inputs_svd(op_info, device, dtype, requires_grad=False):
    return _sample_inputs_svd(op_info, device, dtype, requires_grad, is_linalg_svd=False)

def sample_inputs_linalg_svd(op_info, device, dtype, requires_grad=False):
    return _sample_inputs_svd(op_info, device, dtype, requires_grad, is_linalg_svd=True)

def sample_inputs_pinverse(op_info, device, dtype, requires_grad=False):
    """
    This function generates input for torch.pinverse with distinct singular values so that autograd is always stable.
    Implementation of torch.pinverse depends on torch.svd, therefore it's sufficient to check only square S x S matrix
    and the batched (3 x S x S) input.
    """
    from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

    test_cases = (
        random_fullrank_matrix_distinct_singular_value(S, dtype=dtype).to(device),  # pinverse
        random_fullrank_matrix_distinct_singular_value(S, 3, dtype=dtype).to(device),  # pinverse 'batched'
    )

    out = []
    for a in test_cases:
        a.requires_grad = requires_grad
        out.append(SampleInput(a))
    return out


def sample_inputs_flip(op_info, device, dtype, requires_grad):
    tensors = (
        make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((S, 0, M), device, dtype, low=None, high=None, requires_grad=requires_grad)
    )

    dims = ((0, 1, 2), (0,), (0, 2), (-1,), ())

    samples = [SampleInput(tensor, kwargs={'dims': dim}) for tensor, dim in product(tensors, dims)]

    return samples

def sample_inputs_fliplr_flipud(op_info, device, dtype, requires_grad):
    tensors = (
        make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((S, 0, M), device, dtype, low=None, high=None, requires_grad=requires_grad)
    )
    return [SampleInput(tensor) for tensor in tensors]

def sample_inputs_logit(op_info, device, dtype, requires_grad):
    low, high = op_info.domain

    # Note: Operator is very sensitive at points near the
    # start and end of domain and leads to NaN for float16
    # if domain_eps is 1e-5.
    domain_eps = op_info._domain_eps if dtype != torch.float16 else 3e-2

    low = low + domain_eps
    high = high - domain_eps

    samples = (
        SampleInput(make_tensor((S, S, S), device, dtype, low=low, high=high, requires_grad=requires_grad)),
        SampleInput(make_tensor((S, S, S), device, dtype, low=low,
                                high=high, requires_grad=requires_grad), args=(0.2,)),
        SampleInput(make_tensor((), device, dtype, low=low, high=high, requires_grad=requires_grad)),
        SampleInput(make_tensor((), device, dtype, low=low,
                                high=high, requires_grad=requires_grad), args=(0.2,)),
    )

    return samples

def sample_inputs_masked_scatter(op_info, device, dtype, requires_grad):
    samples = (
        SampleInput(make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn(M, M, device=device) > 0,
                          make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad))),

        SampleInput(make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn((M,), device=device) > 0,
                          make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad))),

        SampleInput(make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(bernoulli_scalar().to(device),
                          make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad))),
    )

    return samples

def sample_inputs_masked_select(op_info, device, dtype, requires_grad):
    samples = (
        SampleInput(make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn(M, M, device=device) > 0,)),

        SampleInput(make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn((M,), device=device) > 0,)),

        SampleInput(make_tensor((M,), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn((M, M), device=device) > 0,)),

        SampleInput(make_tensor((M, 1, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn((M, M), device=device) > 0,)),

        SampleInput(make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.tensor(1, device=device, dtype=torch.bool),)),

        SampleInput(make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.tensor(1, device=device, dtype=torch.bool),)),

        SampleInput(make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn((M, M), device=device) > 0,)),
    )

    return samples

# Operator database (sorted alphabetically)
op_db: List[OpInfo] = [
    # NOTE: CPU complex acos produces incorrect outputs (https://github.com/pytorch/pytorch/issues/42952)
    UnaryUfuncInfo('acos',
                   ref=np.arccos,
                   domain=(-1, 1),
                   handles_complex_extremals=False,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   default_test_dtypes=[torch.long, torch.half, torch.bfloat16, torch.float32, torch.cfloat],
                   skip_bfloat16_grad=True,
                   assert_autodiffed=True,
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-1,
                                                  torch.complex64: 1e-2}),),
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
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
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   test_inplace_grad=False,
                   skips=(
                       # RuntimeError: "rsqrt_cuda" not implemented for 'BFloat16'
                       SkipInfo('TestCommon', 'test_variant_consistency_jit',
                                device_type='cuda', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                       # Reference: https://github.com/pytorch/pytorch/issues/50692
                       SkipInfo('TestGradients', 'test_fn_grad',
                                device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_method_grad',
                                device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                   )),
    OpInfo('addmm',
           dtypes=floating_types(),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           # BFloat16 support on CUDA requires CUDA 11 and SM53
           dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128,
                                           *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_types_and(torch.half),
           assert_autodiffed=True,
           autodiff_nonfusible_nodes=['aten::add', 'aten::mm'],
           skips=(
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        dtypes=[torch.bfloat16, torch.float16, torch.cfloat, torch.cdouble]),),
           sample_inputs_func=sample_inputs_addmm),
    OpInfo('addr',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           # Reference: https://github.com/pytorch/pytorch/issues/50747
           test_inplace_grad=False,
           skips=(
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        dtypes=[torch.float16, torch.cfloat, torch.cdouble, torch.bfloat16]),
               # Reference: https://github.com/pytorch/pytorch/issues/50747
               SkipInfo('TestCommon', 'test_variant_consistency_eager',
                        dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16)),),
           sample_inputs_func=sample_inputs_addr),

    UnaryUfuncInfo('asin',
                   ref=np.arcsin,
                   domain=(-1, 1),
                   supports_sparse=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS)
                   )),
    # NOTE: derivative for inplace asinh is not implemented
    UnaryUfuncInfo('asinh',
                   ref=np.arcsinh,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   test_inplace_grad=False,
                   skips=(
                       # RuntimeError: "rsqrt_cuda" not implemented for 'BFloat16'
                       SkipInfo('TestCommon', 'test_variant_consistency_jit',
                                device_type='cuda', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('atan',
                   ref=np.arctan,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
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
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   test_inplace_grad=False,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    OpInfo('broadcast_to',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_tensor_out=False,
           test_inplace_grad=False,
           sample_inputs_func=sample_inputs_broadcast_to),
    UnaryUfuncInfo('ceil',
                   ref=np.ceil,
                   dtypes=floating_types_and(torch.half),
                   dtypesIfCPU=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   assert_autodiffed=True),
    TriangularOpInfo('cholesky_inverse',
                     op=torch.cholesky_inverse,
                     dtypes=floating_and_complex_types(),
                     # TODO: RuntimeError: cholesky_inverse does not support automatic differentiation for outputs
                     # with complex dtype.
                     test_complex_grad=False,
                     test_inplace_grad=False,
                     check_batched_gradgrad=False,
                     supports_tensor_out=True,
                     decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
                     skips=(
                         # These tests do not take into account custom op.get_op()
                         # TODO: implement op.input_func instead of modifying op.get_op()
                         # See https://github.com/pytorch/pytorch/issues/50837
                         SkipInfo('TestCommon', 'test_variant_consistency_jit'),
                         SkipInfo('TestCommon', 'test_variant_consistency_eager',
                                  dtypes=[torch.complex64, torch.complex128]),)),
    UnaryUfuncInfo('cos',
                   ref=np.cos,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   handles_large_floats=False,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics', device_type='cpu',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                   )),
    UnaryUfuncInfo('cosh',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.cosh),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/48641
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.int8]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics', device_type='cpu',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                       SkipInfo('TestCommon', 'test_variant_consistency_jit',
                                device_type='cuda', dtypes=[torch.float16]),
                   )),
    UnaryUfuncInfo('exp',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.exp),
                   dtypes=all_types_and_complex_and(torch.bool, torch.half),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/50093#pullrequestreview-561791547
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics', dtypes=[torch.bfloat16]),
                       # Reference: https://github.com/pytorch/pytorch/issues/48010
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                   ),
                   assert_autodiffed=True,
                   safe_casts_outputs=True),
    SpectralFuncInfo('fft.fft',
                     aten_name='fft_fft',
                     ref=np.fft.fft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     supports_tensor_out=True,
                     test_inplace_grad=False,),
    SpectralFuncInfo('fft.fftn',
                     aten_name='fft_fftn',
                     ref=np.fft.fftn,
                     ndimensional=True,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     supports_tensor_out=True,
                     test_inplace_grad=False,
                     decorators=[precisionOverride(
                         {torch.float: 1e-4, torch.cfloat: 1e-4})],),
    SpectralFuncInfo('fft.hfft',
                     aten_name='fft_hfft',
                     ref=np.fft.hfft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     supports_tensor_out=True,
                     check_batched_gradgrad=False,
                     test_inplace_grad=False,),
    SpectralFuncInfo('fft.rfft',
                     aten_name='fft_rfft',
                     ref=np.fft.rfft,
                     ndimensional=False,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     supports_tensor_out=True,
                     check_batched_grad=False,
                     check_batched_gradgrad=False,
                     test_inplace_grad=False,),
    SpectralFuncInfo('fft.rfftn',
                     aten_name='fft_rfftn',
                     ref=np.fft.rfftn,
                     ndimensional=True,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     supports_tensor_out=True,
                     test_inplace_grad=False,
                     check_batched_grad=False,
                     check_batched_gradgrad=False,
                     decorators=[precisionOverride({torch.float: 1e-4})],),
    SpectralFuncInfo('fft.ifft',
                     aten_name='fft_ifft',
                     ref=np.fft.ifft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     supports_tensor_out=True,
                     test_inplace_grad=False,),
    SpectralFuncInfo('fft.ifftn',
                     aten_name='fft_ifftn',
                     ref=np.fft.ifftn,
                     ndimensional=True,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     supports_tensor_out=True,
                     test_inplace_grad=False,),
    SpectralFuncInfo('fft.ihfft',
                     aten_name='fft_ihfft',
                     ref=np.fft.ihfft,
                     ndimensional=False,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_types(),
                     supports_tensor_out=True,
                     check_batched_grad=False,
                     test_inplace_grad=False,),
    SpectralFuncInfo('fft.irfft',
                     aten_name='fft_irfft',
                     ref=np.fft.irfft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     supports_tensor_out=True,
                     check_batched_gradgrad=False,
                     test_inplace_grad=False,),
    SpectralFuncInfo('fft.irfftn',
                     aten_name='fft_irfftn',
                     ref=np.fft.irfftn,
                     ndimensional=True,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     supports_tensor_out=True,
                     check_batched_gradgrad=False,
                     test_inplace_grad=False,),
    UnaryUfuncInfo('floor',
                   ref=np.floor,
                   dtypes=floating_types_and(torch.half),
                   dtypesIfCPU=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   assert_autodiffed=True),
    OpInfo('flip',
           op=torch.flip,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_flip,
           test_inplace_grad=False,
           supports_tensor_out=False),
    OpInfo('fliplr',
           op=torch.fliplr,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_fliplr_flipud,
           test_inplace_grad=False,
           supports_tensor_out=False),
    OpInfo('flipud',
           op=torch.flipud,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_fliplr_flipud,
           test_inplace_grad=False,
           supports_tensor_out=False),
    OpInfo('linalg.norm',
           op=torch.linalg.norm,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           test_inplace_grad=False,
           supports_tensor_out=True,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           sample_inputs_func=sample_inputs_linalg_norm,
           aten_name='linalg_norm',
           skips=(
               # TODO: remove this once `pow` is implemented for float16
               #       and bfloat16 on CPU. Issue:
               #       https://github.com/pytorch/pytorch/issues/50789
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        device_type='cpu',
                        dtypes=[torch.float16, torch.bfloat16]),
           )),
    OpInfo('linalg.slogdet',
           aten_name='linalg_slogdet',
           op=torch.linalg.slogdet,
           dtypes=floating_and_complex_types(),
           test_inplace_grad=False,
           supports_tensor_out=False,
           sample_inputs_func=sample_inputs_slogdet,
           output_func=itemgetter(1),
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack]),
    UnaryUfuncInfo('log',
                   ref=np.log,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   safe_casts_outputs=True,
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
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
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
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True),
    UnaryUfuncInfo('log2',
                   ref=np.log2,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble]),
                   )),
    OpInfo('masked_scatter',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_masked_scatter,
           skips=(
               # _th_masked_fill_bool_ not supported for Complex Types.
               SkipInfo('TestGradients', 'test_fn_grad',
                        device_type='cuda', dtypes=[torch.complex128]),
               SkipInfo('TestGradients', 'test_fn_gradgrad',
                        device_type='cuda', dtypes=[torch.complex128]),
               SkipInfo('TestGradients', 'test_inplace_grad',
                        device_type='cuda', dtypes=[torch.complex128]),
               SkipInfo('TestGradients', 'test_inplace_gradgrad',
                        device_type='cuda', dtypes=[torch.complex128]),
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        dtypes=[torch.cfloat, torch.cdouble]),
           ),
           supports_tensor_out=False),
    OpInfo('masked_select',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_masked_select,
           test_inplace_grad=False,
           supports_tensor_out=True),
    UnaryUfuncInfo('neg',
                   ref=np.negative,
                   skip_bfloat16_grad=True,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16),
                   assert_autodiffed=True,),
    UnaryUfuncInfo('round',
                   ref=np.round,
                   dtypes=floating_types_and(torch.half),
                   dtypesIfCPU=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   assert_autodiffed=True,),
    UnaryUfuncInfo('sin',
                   ref=np.sin,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   handles_large_floats=False,
                   handles_complex_extremals=False,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('sinc',
                   ref=np_sinc_with_fp16_as_fp32,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   skip_bfloat16_grad=True,
                   handles_large_floats=False,
                   handles_complex_extremals=False,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2,
                                                  torch.float16: 1e-2}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/49133
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('sinh',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.sinh),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   decorators=(precisionOverride({torch.float16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                       # Reference: https://github.com/pytorch/pytorch/issues/48641
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.int8]),
                       SkipInfo('TestCommon', 'test_variant_consistency_jit',
                                device_type='cuda', dtypes=[torch.float16]),
                   )),
    OpInfo('std',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_std_var,
           test_complex_grad=False,
           test_inplace_grad=False,
           assert_autodiffed=True,
           ),
    UnaryUfuncInfo('tan',
                   ref=np.tan,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   safe_casts_outputs=True,
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
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                   )),
    OpInfo('tensor_split',
           dtypes=all_types_and_complex_and(torch.bool),
           dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_tensor_out=False,
           test_inplace_grad=False,
           sample_inputs_func=sample_inputs_tensor_split,),
    OpInfo('triangular_solve',
           op=torch.triangular_solve,
           dtypes=floating_and_complex_types(),
           test_inplace_grad=False,
           supports_tensor_out=False,
           sample_inputs_func=sample_inputs_legacy_solve,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           # CUDA gradchecks are slow and triangular solve backward is a composite operation
           # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
           skips=(SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    UnaryUfuncInfo('exp2',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.exp2),
                   dtypes=all_types_and(torch.bool, torch.half),
                   dtypesIfCPU=all_types_and(torch.bool, torch.half),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True),
    UnaryUfuncInfo('expm1',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.expm1),
                   dtypes=all_types_and(torch.bool, torch.half),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/48926#issuecomment-739734774
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                   )),
    UnaryUfuncInfo('nan_to_num',
                   ref=np.nan_to_num,
                   dtypes=all_types_and(torch.half, torch.bool),
                   dtypesIfCPU=None,
                   dtypesIfCUDA=None),
    UnaryUfuncInfo('reciprocal',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.reciprocal),
                   dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   dtypesIfCPU=None,
                   dtypesIfCUDA=None,
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   safe_casts_outputs=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/45690
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.cfloat, torch.cdouble]),
                       # Reference: https://github.com/pytorch/pytorch/pull/49102#issuecomment-744604601
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.bfloat16]),
                   )),
    UnaryUfuncInfo('rsqrt',
                   ref=lambda x: np.reciprocal(np.sqrt(x)),
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   decorators=(precisionOverride({torch.half: 5e-2}),),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   handles_complex_extremals=False),
    UnaryUfuncInfo('sqrt',
                   ref=np.sqrt,
                   supports_sparse=True,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   skip_bfloat16_grad=True,
                   decorators=(precisionOverride({torch.bfloat16: 7e-2}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/47358
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_MACOS),
                       # Reference: https://github.com/pytorch/pytorch/pull/47293#issuecomment-721774436
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                dtypes=[torch.bfloat16])),
                   safe_casts_outputs=True,
                   handles_complex_extremals=False),
    OpInfo('linalg.inv',
           aten_name='linalg_inv',
           op=torch.linalg.inv,
           dtypes=floating_and_complex_types(),
           test_inplace_grad=False,
           supports_tensor_out=True,
           sample_inputs_func=sample_inputs_linalg_inv,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack]),
    UnaryUfuncInfo('angle',
                   ref=np.angle,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool),
                   dtypesIfROCM=all_types_and_complex_and(torch.bool),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
                   supports_complex_to_float=True,
                   test_inplace_grad=False),
    OpInfo('linalg.solve',
           aten_name='linalg_solve',
           op=torch.linalg.solve,
           dtypes=floating_and_complex_types(),
           test_inplace_grad=False,
           supports_tensor_out=True,
           sample_inputs_func=sample_inputs_linalg_solve,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack]),
    OpInfo('linalg.pinv',
           aten_name='linalg_pinv',
           op=torch.linalg.pinv,
           dtypes=floating_and_complex_types(),
           test_inplace_grad=False,
           supports_tensor_out=False,
           sample_inputs_func=sample_inputs_linalg_pinv,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack]),
    HermitianOpInfo('linalg.pinv',
                    variant_test_name='hermitian',
                    aten_name='linalg_pinv',
                    op=torch.linalg.pinv,
                    dtypes=floating_and_complex_types(),
                    test_inplace_grad=False,
                    supports_tensor_out=False,
                    sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
                    decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
                    skips=(
                        # These tests do not take into account custom op.get_op()
                        SkipInfo('TestCommon', 'test_variant_consistency_jit'),)
                    ),
    OpInfo('svd',
           op=torch.svd,
           dtypes=floating_and_complex_types(),
           test_inplace_grad=False,
           supports_tensor_out=False,
           sample_inputs_func=sample_inputs_svd,
           decorators=[
               skipCUDAIfNoMagma,
               skipCPUIfNoLapack,
               # gradgrad checks are slow
               DecorateInfo(slowTest, 'TestGradients', 'test_fn_gradgrad'),
           ],
           skips=(
               # cuda gradchecks are very slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('linalg.svd',
           op=torch.linalg.svd,
           aten_name='linalg_svd',
           dtypes=floating_and_complex_types(),
           test_inplace_grad=False,
           supports_tensor_out=False,
           sample_inputs_func=sample_inputs_linalg_svd,
           decorators=[
               skipCUDAIfNoMagma,
               skipCPUIfNoLapack,
               # gradgrad checks are slow
               DecorateInfo(slowTest, 'TestGradients', 'test_fn_gradgrad'),
           ],
           skips=(
               # cuda gradchecks are very slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('pinverse',
           op=torch.pinverse,
           dtypes=floating_and_complex_types(),
           test_inplace_grad=False,
           supports_tensor_out=False,
           sample_inputs_func=sample_inputs_linalg_pinv,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack]),
    OpInfo('gather',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           test_inplace_grad=False,
           sample_inputs_func=sample_inputs_gather),
    OpInfo('index_fill',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           test_inplace_grad=False,
           supports_tensor_out=False,
           sample_inputs_func=sample_inputs_index_fill),
    OpInfo('index_select',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           test_inplace_grad=False,
           skips=(
               # https://github.com/pytorch/pytorch/issues/49707
               SkipInfo('TestCommon', 'test_variant_consistency_eager',
                        dtypes=[torch.float16, torch.bfloat16]),
               SkipInfo('TestCommon', 'test_variant_consistency_jit', dtypes=[torch.float16, torch.bfloat16]),
           ),
           sample_inputs_func=sample_inputs_index_select),
    OpInfo('stack',
           # gradcheck expects the input arguments as a flat list
           op=lambda *args, idx: torch.stack([*args], idx),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           test_inplace_grad=False,
           supports_tensor_out=False,
           skips=(
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
           ),
           sample_inputs_func=sample_inputs_stack),
    OpInfo('hstack',
           # gradcheck expects the input arguments as a flat list
           op=lambda *args: torch.hstack([*args]),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           test_inplace_grad=False,
           supports_tensor_out=False,
           skips=(
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
           ),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack),
    OpInfo('vstack',
           # gradcheck expects the input arguments as a flat list
           op=lambda *args: torch.vstack([*args]),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           test_inplace_grad=False,
           supports_tensor_out=False,
           skips=(
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
           ),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack),
    OpInfo('dstack',
           # gradcheck expects the input arguments as a flat list
           op=lambda *args: torch.dstack([*args]),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           test_inplace_grad=False,
           supports_tensor_out=False,
           skips=(
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
           ),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack),
    OpInfo('movedim',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           test_inplace_grad=False,
           supports_tensor_out=False,
           sample_inputs_func=sample_movedim_moveaxis),
    OpInfo('moveaxis',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           test_inplace_grad=False,
           supports_tensor_out=False,
           sample_inputs_func=sample_movedim_moveaxis),
    ShapeFuncInfo('repeat',
                  op=lambda x, dims: x.repeat(dims),
                  ref=np.tile,
                  dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                  supports_tensor_out=False,
                  test_inplace_grad=False,
                  skips=(
                      # torch.repeat does not exist so we get a RuntimeError.
                      SkipInfo('TestCommon', 'test_variant_consistency_jit',
                               dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
                  ),
                  sample_inputs_func=sample_repeat_tile),
    ShapeFuncInfo('tile',
                  ref=np.tile,
                  dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                  supports_tensor_out=False,
                  test_inplace_grad=False,
                  sample_inputs_func=sample_repeat_tile),
    OpInfo('var',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_std_var,
           test_complex_grad=False,
           test_inplace_grad=False,
           assert_autodiffed=True,
           ),
]

if TEST_SCIPY:
    def reference_sigmoid(x):
        # 'scipy.special.expit' not supported for the input types
        if x.dtype in [np.complex64, np.complex128]:
            return (1 / (1 + np.exp(-x)))
        return scipy.special.expit(x)

    def reference_lgamma(x):
        # scipy.special.gammaln returns `-inf` when input is `-inf`.
        # While Pytorch, C and C++, all return `inf` when input is `-inf`.
        # Reference:
        # https://en.cppreference.com/w/cpp/numeric/math/lgamma
        # https://en.cppreference.com/w/c/numeric/math/lgamma

        # To handle the above discrepancy,
        # we replace -inf with inf so values
        # that were originally -inf map to inf as expected
        if x.dtype.kind == 'f':
            x = np.where(x == float('-inf'), np.array(float('inf'), dtype=x.dtype), x)

        out = scipy.special.gammaln(x)

        if x.dtype == np.float16:
            # `scipy.special.gammaln` returns output of float32 when input is float16,
            # while `torch.lgamma` preserves `float16`. But due to smaller range of float16,
            # Pytorch version outputs `inf` while SciPy returns finite values.
            out = out.astype(np.float16)

        return out

    op_db_scipy_reference: List[OpInfo] = [
        UnaryUfuncInfo('sigmoid',
                       ref=reference_sigmoid,
                       decorators=(precisionOverride({torch.float16: 1e-2,
                                                      torch.bfloat16: 1e-2}),),
                       skips=(
                           SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                           # RuntimeError: sigmoid does not support automatic differentiation for outputs with complex dtype.
                           SkipInfo('TestCommon', 'test_variant_consistency_jit',
                                    dtypes=[torch.complex64, torch.complex128]),
                           SkipInfo('TestCommon', 'test_variant_consistency_eager',
                                    dtypes=[torch.complex64, torch.complex128]),),
                       dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                       dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                       dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                       safe_casts_outputs=True,
                       assert_autodiffed=True,
                       test_complex_grad=False),  # Reference: https://github.com/pytorch/pytorch/issues/48552
        UnaryUfuncInfo('digamma',
                       ref=scipy.special.digamma,
                       decorators=(precisionOverride({torch.float16: 5e-1}),),
                       dtypes=all_types_and(torch.bool),
                       dtypesIfCPU=all_types_and(torch.bool),
                       dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                       skips=(
                           # In some cases, output is NaN (for input close to
                           # negative integers) especially due to reduced precision
                           # in float16 and NaN's can't be tested for equality.
                           SkipInfo('TestCommon', 'test_variant_consistency_jit',
                                    device_type='cuda', dtypes=[torch.float16]),),
                       safe_casts_outputs=True),
        UnaryUfuncInfo('erf',
                       ref=scipy.special.erf,
                       decorators=(precisionOverride({torch.float16: 1e-2,
                                                      torch.bfloat16: 1e-2}),),
                       dtypes=all_types_and(torch.bool),
                       dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                       dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                       skips=(
                           # RuntimeError: "pow" not implemented for 'BFloat16'
                           SkipInfo('TestCommon', 'test_variant_consistency_jit',
                                    dtypes=[torch.bfloat16]),),
                       assert_autodiffed=True,
                       safe_casts_outputs=True),
        UnaryUfuncInfo('erfc',
                       ref=scipy.special.erfc,
                       decorators=(precisionOverride({torch.float16: 1e-2,
                                                      torch.bfloat16: 1e-2}),),
                       dtypes=all_types_and(torch.bool),
                       dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                       dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                       skips=(
                           # RuntimeError: "pow" not implemented for 'BFloat16'
                           SkipInfo('TestCommon', 'test_variant_consistency_jit',
                                    dtypes=[torch.bfloat16]),),
                       assert_autodiffed=True,
                       safe_casts_outputs=True),
        UnaryUfuncInfo('erfinv',
                       ref=scipy.special.erfinv,
                       decorators=(precisionOverride({torch.float16: 1e-2,
                                                      torch.bfloat16: 1e-2,
                                                      torch.float32: 1e-4}),),
                       dtypes=all_types_and(torch.bool),
                       dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                       dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                       safe_casts_outputs=True,
                       domain=(-1, 1),
                       skips=(
                           # Reference: https://github.com/pytorch/pytorch/pull/49155#issuecomment-742664611
                           SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                    active_if=LooseVersion(scipy.__version__) < "1.4.0"),
                           # RuntimeError: "pow" not implemented for 'BFloat16'
                           SkipInfo('TestCommon', 'test_variant_consistency_jit',
                                    dtypes=[torch.bfloat16]),
                       )
                       ),
        UnaryUfuncInfo('lgamma',
                       ref=reference_lgamma,
                       decorators=(precisionOverride({torch.float16: 7e-1}),),
                       dtypes=all_types_and(torch.bool),
                       dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                       dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                       skips=(
                           # Reference: https://github.com/pytorch/pytorch/pull/50140#discussion_r552615345
                           SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                    dtypes=[torch.bfloat16]),
                           # Reference: https://github.com/pytorch/pytorch/pull/50140#issuecomment-756150214
                           SkipInfo('TestUnaryUfuncs', 'test_reference_numerics',
                                    dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS),
                           # Backward of `lgamma` uses `digamma` but `digamma`
                           # is not implemented for `BFloat16`
                           # Error Raised:
                           #   RuntimeError: "digamma" not implemented for 'BFloat16'
                           SkipInfo('TestCommon', 'test_variant_consistency_jit',
                                    dtypes=[torch.bfloat16]),
                       ),
                       safe_casts_outputs=True),
        UnaryUfuncInfo('logit',
                       ref=scipy.special.logit,
                       domain=(0, 1),
                       decorators=(precisionOverride({torch.bfloat16: 5e-1,
                                                      torch.float16: 5e-1}),),
                       dtypes=floating_types_and(torch.half),
                       dtypesIfCPU=floating_types_and(torch.bfloat16),
                       dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                       sample_inputs_func=sample_inputs_logit),
        OpInfo('xlogy',
               dtypes=all_types_and(torch.bool),
               dtypesIfCPU=all_types_and(torch.bool, torch.half, torch.bfloat16),
               dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
               test_inplace_grad=True,
               supports_tensor_out=True,
               safe_casts_outputs=True,
               sample_inputs_func=sample_inputs_xlogy),
        OpInfo('trace',
               dtypes=all_types_and_complex(),
               dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
               test_inplace_grad=False,
               supports_tensor_out=False,
               # Reference: https://github.com/pytorch/pytorch/issues/50381
               test_complex_grad=False,
               sample_inputs_func=sample_inputs_trace,
               skips=(
                   SkipInfo('TestCommon', 'test_variant_consistency_jit',
                            dtypes=[torch.complex64, torch.complex128]),
                   SkipInfo('TestCommon', 'test_variant_consistency_eager',
                            dtypes=[torch.complex64, torch.complex128]))),
    ]
    op_db = op_db + op_db_scipy_reference

# Common operator groupings
unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo)]
spectral_funcs = [op for op in op_db if isinstance(op, SpectralFuncInfo)]
sparse_unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo) and op.supports_sparse is True]
shape_funcs = [op for op in op_db if isinstance(op, ShapeFuncInfo)]

def index_variable(shape, max_indices, device=torch.device('cpu')):
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = torch.rand(*shape, device=device).mul_(max_indices).floor_().long()
    return index


def index_perm_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)

    index = torch.randperm(max_indices).narrow(0, 0, reduce(mul, shape)).view(shape)
    return index


def gather_variable(shape, index_dim, max_indices, duplicate=False, device=torch.device('cpu')):
    assert len(shape) == 2
    assert index_dim < 2
    batch_dim = 1 - index_dim
    index = torch.zeros(*shape, dtype=torch.long, device=device)
    for i in range(shape[index_dim]):
        index.select(index_dim, i).copy_(
            torch.randperm(max_indices, device=device)[:shape[batch_dim]])
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
        ('float_power', torch.rand(S, S, S) + 1e-3, (torch.rand(S, S, S) + 0.1,), ''),
        ('float_power', torch.rand(S, S, S) + 1e-3, (torch.rand(1,) + 0.1,), 'broadcast_rhs'),
        ('float_power', torch.rand(1,) + 1e-3, (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs'),
        ('float_power', torch.rand(S, 1, S) + 1e-3, (torch.rand(1, S, 1) + 0.1,), 'broadcast_all'),
        ('float_power', uniform_scalar(1e-3, requires_grad=True), (uniform_scalar(0.1),), 'scalar'),
        ('float_power', torch.rand(S, S, S) + 1e-3, (uniform_scalar(0.1),), 'scalar_broadcast_rhs'),
        ('float_power', uniform_scalar(1e-3, requires_grad=True), (torch.rand(S, S, S) + 0.1,), 'scalar_broadcast_lhs'),
        ('float_power', torch.rand(S, S, S) + 1e-3, (3.14,), 'constant'),
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
        ('clamp', (S, S, S), (0, 1), '', (True,)),
        ('clamp', (S, S, S), (None, 0.5), 'min', (True,)),
        ('clamp', (S, S, S), (0.5, None), 'max', (True,)),
        ('clamp', (), (0, 1), 'scalar', (True,)),
        ('clamp', (), (None, 0.5), 'min_scalar', (True,)),
        ('clamp', (), (0.5, None), 'max_scalar', (True,)),
        ('clamp', (S, S), (), 'max_scalar_kwarg', (True,), (), (), ident, {'max': 1}),
        ('atan2', (S, S, S), ((S, S, S),)),
        ('atan2', (), ((),), 'scalar'),
        ('atan2', (S, S, S), ((S,),), 'broadcast_rhs'),
        ('atan2', (S,), ((S, S, S),), 'broadcast_lhs'),
        ('atan2', (S, 1, S), ((S, S),), 'broadcast_all'),
        ('sign', (S, S, S), NO_ARGS),
        ('sign', (), NO_ARGS, 'scalar'),
        ('sgn', (S, S, S), NO_ARGS),
        ('sgn', (), NO_ARGS, 'scalar'),
        ('trunc', (S, S, S), NO_ARGS, '', (True,)),
        ('trunc', (), NO_ARGS, 'scalar', (True,)),
        ('rad2deg', (S, S, S), NO_ARGS),
        ('deg2rad', (S, S, S), NO_ARGS),
        # Removing the 'rsqrt' entries leads to failure in
        # test_index_fill_variable_dim_*
        # TODO: Remove when fixed.
        # Reference: https://github.com/pytorch/pytorch/issues/48230
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
        ('kthvalue', (S, S, S), (2, 1,), 'dim_alert_nondeterministic', (), [1],
            [expectedAlertNondeterministic('kthvalue CUDA', 'cuda')]),
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
        ('median', (S, S, S), (1,), 'dim_alert_nondeterministic', (), [0],
            [expectedAlertNondeterministic('median CUDA with indices output', 'cuda')]),
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
        ('dot', (L,), ((L,),), '', (True,)),
        ('vdot', (L,), ((L,),),),
        ('mm', (S, M), ((M, S),), '', (True,)),
        ('bmm', (M, S, M), ((M, M, S),), '', (True,)),
        ('mv', (S, M), ((M,),), '', (True,)),
        ('ger', (S,), ((M,),)),
        ('inner', (S,), ((S,),), "1d_1d", (False,)),
        ('inner', (), ((S, S),), "scalar_2d", (False,)),
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
        ('cross', (S, 3), ((S, 3),)),
        ('cross', (S, 3, S), ((S, 3, S), 1), 'dim'),
        ('index_add', (S, S), (0, index_variable(2, S), (2, S)), 'dim', (), [0]),
        ('index_add', (), (0, torch.tensor([0], dtype=torch.int64), (1,)), 'scalar_input_dim', (), [0]),
        ('index_add', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim', (), [0]),
        ('index_add', (S, S), (0, index_variable(2, S), (2, S)), 'alert_nondeterministic', (), [0],
            [expectedAlertNondeterministic('index_add_cuda_', 'cuda')]),
        ('index_copy', (S, S), (0, index_perm_variable(2, S), (2, S)), 'dim', (), [0]),
        ('index_copy', (S, S), (0, index_perm_variable(2, S), (2, S)), 'dim_alert_nondeterministic', (), [0],
            [expectedAlertNondeterministic('index_copy')]),
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
        # For `logdet` the function at det=0 is not smooth.
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
        ('scatter', (M, S), (0, gather_variable((S, S), 1, M), (S, S)), 'dim0', (), [0]),
        ('scatter', (M, S), (1, gather_variable((M, S // 2), 0, S), (M, S // 2)), 'dim1', (), [0]),
        ('scatter', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalartensor_all_dim0', (), [0]),
        ('scatter', (), (0, torch.tensor(0, dtype=torch.int64), 2.5), 'scalar_all_dim0', (), [0]),
        ('scatter_add', (M, S), (0, gather_variable((S, S), 1, M), (S, S)), 'dim0', (), [0]),
        ('scatter_add', (M, S), (1, gather_variable((M, S // 2), 0, S), (M, S // 2)), 'dim1', (), [0]),
        ('scatter_add', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim0', (), [0]),
        ('scatter_add', (M, S), (0, gather_variable((S, S), 1, M), (S, S)), 'alert_nondeterministic', (), [0],
            [expectedAlertNondeterministic('scatter_add_cuda_kernel', 'cuda')]),
        ('masked_fill', (M, M), (torch.BoolTensor(M, M).bernoulli_(), 10)),
        ('masked_fill', (M, M), (torch.BoolTensor(M, M).bernoulli_(), ()), 'tensor'),
        ('masked_fill', (M,), (torch.BoolTensor(M, M).bernoulli_(), 10), 'broadcast_lhs'),
        ('masked_fill', (M, M), (torch.BoolTensor(M,).bernoulli_(), 10), 'broadcast_rhs'),
        ('masked_fill', (), (torch.tensor(0, dtype=torch.bool).bernoulli_(), 10), 'scalar'),
        ('masked_fill', (), (torch.tensor(0, dtype=torch.bool).bernoulli_(), ()),
         'scalar_variable'),
        ('masked_fill', (M, M), (torch.tensor(0, dtype=torch.bool).bernoulli_(), 10),
         'scalar_broadcast_rhs'),
        ('masked_scatter', (M,), (torch.BoolTensor(M, M).bernoulli_(), (M, M)),
         'broadcast_lhs'),
        ('maximum', (S, S), ((S, S),)),
        ('minimum', (S, S), ((S, S),)),
        ('fmax', (S, S), ((S, S),)),
        ('fmin', (S, S), ((S, S),)),
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
        ('msort', (S, M, S), NO_ARGS),
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
