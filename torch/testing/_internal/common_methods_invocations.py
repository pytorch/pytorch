from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import collections
import copy
from enum import Enum
import operator
import random
import unittest
import math

import torch
import numpy as np
from torch._six import inf
import collections.abc

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from torch.testing import make_non_contiguous, make_tensor
from torch.testing._internal.common_dtype import (
    _dispatch_dtypes, floating_types, floating_types_and, complex_types, floating_and_complex_types,
    floating_and_complex_types_and, all_types_and_complex_and, all_types_and, all_types_and_complex, integral_types_and,
    all_types, double_types, empty_types
)
from torch.testing._internal.common_device_type import \
    (onlyCPU, onlyCUDA, onlyNativeDeviceTypes, disablecuDNN, skipCUDAIfNoMagma, skipCUDAIfNoMagmaAndNoCusolver,
     skipCUDAIfNoCusolver, skipCPUIfNoLapack, skipCPUIfNoFFT, skipCUDAIfRocm, precisionOverride,
     toleranceOverride, tol, has_cusolver)
from torch.testing._internal.common_cuda import CUDA11OrLater, SM53OrLater, SM60OrLater
from torch.testing._internal.common_utils import \
    (is_iterable_of_tensors,
     random_symmetric_matrix, random_symmetric_psd_matrix,
     make_fullrank_matrices_with_distinct_singular_values,
     random_symmetric_pd_matrix, make_symmetric_matrices,
     make_symmetric_pd_matrices, random_square_matrix_of_rank,
     TEST_WITH_ROCM, IS_WINDOWS, IS_MACOS, TEST_SCIPY,
     torch_to_numpy_dtype_dict, TEST_WITH_ASAN,
     GRADCHECK_NONDET_TOL, slowTest, noncontiguous_like,
     freeze_rng_state)
import torch.testing._internal.opinfo_helper as opinfo_helper

from distutils.version import LooseVersion

has_scipy_fft = False
if TEST_SCIPY:
    import scipy.special
    try:
        import scipy.fft
        has_scipy_fft = True
    except ModuleNotFoundError:
        pass


# Reasonable testing sizes for dimensions
L = 20
M = 10
S = 5

# Unique value to distinguish default from anything else
_NOTHING = object()


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


class SampleInput(object):
    """Represents sample inputs to a function."""

    __slots__ = ['input', 'args', 'kwargs', 'output_process_fn_grad', 'broadcasts_input', 'name']

    def __init__(self, input, *, args=tuple(), kwargs=None, output_process_fn_grad=lambda x: x, broadcasts_input=False, name=""):
        # input is the first input to the op and must be either a Tensor or TensorList (Sequence[Tensor]).
        # This follows the typical pattern where for Tensor inputs op(t, ...) = t.op(...).
        # op with TensorList inputs do not support method or inplace variants.
        assert isinstance(input, torch.Tensor) or is_iterable_of_tensors(input)
        self.input: Union[torch.Tensor, Sequence[torch.Tensor]] = input
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
        self.output_process_fn_grad = output_process_fn_grad
        self.name = name

        # Specifies if `self.input` is broadcasted or not,
        # given that the operator supports broadcasting.
        # This field is used to verify the behavior for inplace variant.
        #
        # If a SampleInput is marked with `broadcasts_input=True`,
        # it is verified that we get a `RuntimerError` with this sample,
        # and inplace variant. Also inplace grad{grad} tests are skipped,
        # for such inputs (as they will error out otherwise).
        self.broadcasts_input = broadcasts_input

    def _repr_helper(self, formatter):
        # Helper function to return the details of the SampleInput as `str`
        # It consolidates all the fields of SampleInput and allows,
        # formatting the fields like `input`, `args`, etc with `formatter`
        # callable to customize the representation.
        # Look at `summary` method for example.
        arguments = [
            f'input={formatter(self.input)}',
            f'args={formatter(self.args)}',
            f'kwargs={formatter(self.kwargs)}',
            f'output_process_fn_grad={self.output_process_fn_grad}',
            f'broadcasts_input={self.broadcasts_input}',
            f'name={repr(self.name)}']

        return f'SampleInput({", ".join(a for a in arguments if a is not None)})'

    def __repr__(self):
        return self._repr_helper(lambda x: x)

    def summary(self):
        # Returns the SampleInput details in a more
        # friendly format.
        # It formats `Tensor` and `TensorList`
        # in a more condensed representation.
        def formatter(arg):
            # Format any instance of `Tensor` (standalone, in list, or in dict)
            # by Tensor[TensorShape]
            # Eg. Tensor with shape (3, 4) is formatted as Tensor[3, 4]
            if isinstance(arg, torch.Tensor):
                shape = str(tuple(arg.shape)).replace('(', '').replace(')', '')
                return f"Tensor[{shape}]"
            elif isinstance(arg, dict):
                return {k: formatter(v) for k, v in arg.items()}
            elif is_iterable_of_tensors(arg):
                return "TensorList[" + ", ".join(map(formatter, arg)) + "]"
            elif isinstance(arg, (list, tuple)):  # Handle list, tuple
                return "(" + ",".join(map(formatter, arg)) + ")"

            return repr(arg)

        return self._repr_helper(formatter)

    # Applies the transform f(t) -> t to each tensor and dtype in the SampleInput
    def transform(self, f):
        def tt(t):
            def _tt(t):
                return f(t)

            if isinstance(t, torch.Tensor):
                return _tt(t)
            elif isinstance(t, torch.dtype):
                return _tt(t)
            elif isinstance(t, list):
                return list(map(tt, t))
            elif isinstance(t, tuple):
                return tuple(map(tt, t))
            elif isinstance(t, dict):
                return {k: tt(v) for k, v in t.items()}
            else:
                return t

        sample_tt_input, tt_args, tt_kwargs = tt(self.input), tt(self.args), tt(self.kwargs)
        return (sample_tt_input, tt_args, tt_kwargs)

    # Returns the NumPy version of the sample input object in the form of a tuple: (input, args, kwargs)
    # Converts tensors to ndarrays by calling .detach().cpu().numpy() on them
    # Converts dtypes by remapping them using torch_to_numpy_dtype_dict
    def numpy(self):
        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            elif isinstance(t, torch.dtype):
                return torch_to_numpy_dtype_dict[t]

        return self.transform(to_numpy)

    def noncontiguous(self):
        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                return noncontiguous_like(t)
            if isinstance(t, torch.dtype):
                return t

        return self.transform(to_noncontiguous)


class ErrorInput(object):
    """
    A SampleInput that will cause the operation to throw an error plus information
    about the resulting error.
    """

    __slots__ = ['sample_input', 'error_type', 'error_regex']

    def __init__(self, sample_input, *, error_type, error_regex):
        self.sample_input = sample_input
        self.error_type = error_type
        self.error_regex = error_regex


class AliasInfo(object):
    """Class holds alias information. For example, torch.abs ->
    torch.absolute, torch.Tensor.absolute, torch.Tensor.absolute_
    """

    def __init__(self, alias_name):
        self.name = alias_name
        self.op = _getattr_qual(torch, alias_name)
        self.method_variant = getattr(torch.Tensor, alias_name, None)
        self.inplace_variant = getattr(torch.Tensor, alias_name + "_", None)

    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)


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


# test if a tensor is close to an integer
def close_to_int(x, eps=0.1):
    if x.is_complex():
        y = torch.abs(torch.view_as_complex(torch.frac(torch.view_as_real(x))))
    else:
        y = torch.abs(torch.frac(x))
    return (y < eps) | (y > (1 - eps))


NumericsFilter = collections.namedtuple('NumericsFilter', ['condition', 'safe_val'])


# Note [OpInfos]
# ~~~~~~~~~~~~~~
#
# The majority of this note was written shortly after the PyTorch 1.9 release.
# If you notice it's out-of-date or think it could be improved then please
# file an issue.
#
# See also: the OpInfo tracker (https://github.com/pytorch/pytorch/issues/54261)
# See also: "Writing Test Templates" in common_device_type.py to learn how to
#   parametrize a test template using OpInfos.
# See also: PyTorch's GitHub wiki on running and writing tests
#   https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
# See also: ModuleInfos, OpInfo's sister class, defined in common_modules.py
#
# An OpInfo is a collection of metadata related to a PyTorch operator. This
#   metadata is used to generate tests that validate properties of the operator,
#   like if it implements the correct gradient formula.
#
# WHY OPINFOS?
# ~~~~~~~~~~~~
#
# OpInfos are principally intended to do three things:
#
#   1) to allow systematic testing over all PyTorch's operators
#   2) to simplify operating testing by autogenerating many tests
#   3) to allow systems (like autograd, torchscript, fx, nnc...) to test
#        against every PyTorch operator
#
# All these goals are still a work in progress. Not every operator has an
#   OpInfo, and some operator tests that could be automatically generated
#   still have to be written manually.
#
# It's helpful to understand that OpInfos are both about test simplification and
#   modularity. PyTorch is a complicated framework with many interrelated systems,
#   too many for any one person to keep track of. An OpInfo can be thought of as the
#   interface between an operator implementer and those other systems. Instead of
#   requiring the implementer of torch.foo understand how to test its forward
#   mode AD or NNC support that's typically handled automatically just by
#   defining an OpInfo.
#
# It's often surprising to OpInfo writers that just implementing an OpInfo
#   typically can't verify an operator is actually implemented correctly:
#
# "If an OpInfo doesn't validate my op works as expected, what's the point
#     of it?"
#
# But the point of is the above. OpInfos are intended to let you focus on testing
#   the operator logic you're familiar with instead of having to write tests for
#   how the operator interacts with each of PyTorch's many systems.
#
# And, OK, it turns out that SOMETIMES just writing an OpInfo DOES
#   validate your op works as expected, but that's only in special
#   cases. See below for details.
#
# WHAT'S AN OPINFO?
# ~~~~~~~~~~~~~~~~~
#
# So what is an OpInfo? It's a Python class that describes an operator's properties,
#   like which dtypes it supports on the CPU and whether it has any aliases.
#   These properties can be divided into three categories:
#
#   1) Metadata describing the operator, like the operator's name and if it
#     "supports" the out kwarg.
#   2) Test directives, like "skips" that tell the test suite to skip some
#     tests.
#   3) A "sample inputs" function that generates valid inputs for the operator.
#
# OpInfo attributes are described in more detail below.
#
# THE SAMPLE INPUTS FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The "sample inputs" function merits special elaboration. This function is
#   crucial to testing with OpInfos. A typical OpInfo test has to treat the operator
#   as a black box. There's no structure for the test to understand or exploit.
#   Without "sample inputs" it wouldn't even know how to call the OpInfo's
#   operator. The sample input function saves the day by providing different
#   "SampleInputs" that can be used to call the operator. A sample input
#   function should have the following signature:
#
#   def sample_inputs_foo(op_info, device, dtype, requires_grad, **kwargs):
#
#   And should return an iterable of SampleInputs (see the class description
#   above). Each SampleInput defines an "input", "args", "kwargs", an
#   "output_process_fn_grad" function, the "broadcasts_input" bool and a
#   "name".
#
#   All the "sample_inputs" functions are invoked within a `torch.no_grad()`
#   environment for efficiency and correctness. As such remember to set the the
#   "requires_grad" flag on the inputs **after** performing any transformations
#   on them.
#
# The "input" is the first argument to the operator, or the tensor that
#   the method or inplace variants of the operator should be called on, and
#   should be on the requested device, of the requested dtype, and its
#   requires_grad attribute should be set to the requires_grad argument.
#
# "args" should contain positional arguments, and "kwargs" keyword arguments.
#
# "output_process_fn_grad" has an interesting name. It's a function that maps
#   the operator's output (when given the input, args, and kwargs) to the
#   portion of the output to gradcheck. For example, consider an operator
#   like torch.linalg.slogdet
#   (https://pytorch.org/docs/master/generated/torch.linalg.slogdet.html).
#   This operator returns a tuple of two tensors, but the first tensor
#   cannot be backwarded through. Its "output_process_fn_grad" filters
#   this output tuple to just the second argument, which we can call backward
#   on. Functions that produce a single tensor can ignore this argument.
#
# "broadcasts_input" is a bool indicated if the SampleInput causes the operator
#   to broadcast the "input" argument. This is important for tests to understand
#   because inplace variants of operations throw a runtime error if they
#   would broadcast their input arguments, so tests that work with inplace
#   variants filter SampleInputs that broadcast their input.
#
# "name" is a string that's just used for debugging. It appears when printing
#   the SampleInput.
#
# THE (OPTIONAL) ERROR INPUTS FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# OpInfos may optionally specify "error inputs" through an error function. If
#   specified test_errors in test_ops.py will call the op with these inputs
#   and validate that the desired error is thrown.
#
# Error inputs automate a common testing pattern where multiple inputs are
#   passed to an operation and the errors they thrown are reviewed. Tests
#   written in this style should be ported to the new OpInfo pattern.
#
# Error inputs are specified using the ErrorInputs class, which contains
#   a SampleInput (see above) and data about the expected error.
#
# OPINFO FILE ORGANIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# All OpInfos are currently defined in this file. Most OpInfo tests are defined
#   in test_ops.py, but some system-specific tests are defined in those
#   systems' test files, and subclass-specific tests are defined in the test
#   file that corresponds to that subclass (see the below).
#   Expect a reorganization in the future.
#
# WHAT'S TESTED?
# ~~~~~~~~~~~~~~
#
# Every OpInfo in the op_db sequence has the following properties validated in
# test_ops.py:
#
#   - that its supported dtypes are specified correctly
#   - that the operation produces the same results when called with noncontiguous inputs
#   - that it supports the out= argument properly (if it allows out=),
#       see https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
#   - that it works with the conjugate view bit properly
#   - that its function, method, and inplace variants perform the same operation
#       (that is, that torch.add, torch.Tensor.add, and torch.Tensor.add_ all
#       do the same thing).
#   - that its inplace variant preserves the input's storage
#   - that its gradient formula is implemented correctly, and that it supports
#       gradgrad and complex grad and gradgrad and forward mode AD properly for
#       the op's function and inplace variants (method variants are skipped
#       to reduce test time).
#   - that the operation performs the same operation when traced or scripted
#       using the jit
#   - that the operation is autodifferentiated by the jit as expected
#   - that the operator's aliases, if any, perform the same operation and that
#       the jit understands the alias
#
# Additional OpInfo tests are in test_jit_fuser_te.py, test_fx_experimental.py,
#   and test_fx.py. These tests validate that operators work with NNC and FX
#   as expected.
#
# For performance, some of the above tests may only run on the first
#   SampleInput returned by an OpInfo's sample input function.
#
# In addition to these tests, some subclasses (discussed in the next section)
#   define additional tests.
#
# Critically, as mentioned above, what's not tested is that the operator
#   works as expected. When implementing an OpInfo an engineer must still
#   typically write one or more tests validating the operator's behavior.
#
# OPINFO (SUB)CLASSES
# ~~~~~~~~~~~~~~~~~~~
#
# In addition to the OpInfo base class there are several specialized OpInfo
#   subclasses. For example, the UnaryUfuncInfo subclass is used for
#   unary elementwise operations. These operations have a common structure
#   that test_unary_ufuncs.py exploits with additional automated testing.
#   The automated testing in test_unary_ufuncs.py is so thorough, comparing
#   the operator to a NumPy reference function on a plethora of values, that
#   just implementing an OpInfo for a unary elementwise operation is often
#   sufficient testing.
#
# The ForeachFuncInfo is another OpInfo subclass that is hyper-specialized to a
#   very unique class of operations. These OpInfos aren't included in the
#   op_db sequence and have their own tests.
#
# Other OpInfo subclasses, like SpectralFuncInfo, are just for convenience
# when writing OpInfos.
#
# TESTING A NEW OPERATOR
# ~~~~~~~~~~~~~~~~~~~~~~
#
# If you're adding a new operator to any of the following namespaces:
#   - torch
#   - torch.fft
#   - torch.linalg,
#   - torch.special
#   - torch.nn.functional
# then you should typically add an OpInfo for it.
#
# As mentioned a couple times above, implementing an OpInfo is not
#   usually sufficient testing (unless the operator is a unary elementwise
#   operator). The OpInfo will only test the properties described in the
#   "WHAT'S TESTED" section. It DOES NOT verify that the operator is
#   implemented correctly.
#
# TIPS FOR WRITING AN OPINFO AND OPINFO TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Writing an OpInfo can be a little daunting. Since the point of an OpInfo is to
#   be consumed by a variety of systems it can be hard to understand how to
#   deal with test failures or how to set the OpInfo metadata properly.
#
# Before adding an OpInfo it helps to look at other OpInfos. A sample inputs
#   function must be defined, and the operator's dtypes must be specified.
#   Once that's done you should run the operator's tests in test_ops.py
#   (these can be filtered using the "-k" argument in pytest). Tests that
#   fail should provide an error message that describes what to change about
#   your OpInfo. You don't need to worry about changing an OpInfo's default
#   values unless a test yells at you.
#
# Similarly, if you're writing a test that consumes OpInfos then it's critical
#   your test provides a clear error message describing what to do when it
#   fails. You should not assume the OpInfo implementer is familiar with your
#   system.
#
# If you see a confusing error message while developing an OpInfo then please
#   file an issue describing what happened.
#
# This trial-and-error approach to writing an OpInfo can be frustrating,
#   but it's probably necessary as long as OpInfos don't require
#   learning about all the systems that consume them. One thing that can help
#   is the get_supported_dtypes() function defined in opinfo_helper.py. This
#   function can be used to programmatically specify the dtypes an operator
#   supports, and is especially useful if writing an OpInfo on a machine
#   without a CUDA device. See its documentation for more details.
#
# THE FUTURE OF OPINFOS AND OPINFO TESTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the future we expect OpInfo coverage to improve and cover
#   the great majority of PyTorch's (public) operators.
#

# Classes and methods for the operator database
class OpInfo(object):
    """Operator information and helper functions for acquiring it."""

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref=None,  # An optional reference function that accepts ndarrays (AKA "NumPy arrays").
                            # If given, the op will be compared with its reference on each of its sample inputs.
                 # the following metadata describes the operator, its variants,
                 #   and its aliases, if any
                 aliases=None,  # iterable of aliases, e.g. ("absolute",) for torch.abs
                 variant_test_name='',  # additional string to include in the test name
                                        # this is useful when an op needs multiple OpInfos,
                                        # like divide does, often because it's really several
                                        # different ops behind the scenes
                 op=None,  # the function variant of the operation, populated as torch.<name> if None
                 method_variant=_NOTHING,  # explicitly specifies the method variant of the operator
                                           # if _NOTHING (default), the method variant will be autopopulated
                                           # if None, then the OpInfo specifies no method variant
                 inplace_variant=_NOTHING,  # explicitly specifies the inplace variant of the operator
                                            # if _NOTHING (default), the method variant will be autopopulated
                                            # if None, then the OpInfo specifies no method variant

                 # the following metadata are test directives for skipping or
                 #   modifying tests
                 skips=tuple(),  # information about which tests to skip
                 decorators=tuple(),  # decorators to apply to generated tests

                 # the following are pointers to functions to generate certain classes
                 #   of inputs
                 sample_inputs_func=None,  # function to generate sample inputs
                 error_inputs_func=None,  # function to generate inputs that will throw errors

                 # the following metadata relates to dtype support and is tested for correctness in test_ops.py
                 dtypes,  # dtypes this function works with on the CPU,
                          # inherited by other device types that don't specify their own dtypes

                 # the following dtypesIf... options override the dtypes value
                 # on their respective device types
                 dtypesIfCPU=None,  # dtypes this function is expected to work with on the CPU,
                                    # typically unnecessary since it's (now) redundant with the dtypes kwarg above
                 dtypesIfCUDA=None,  # dtypes this function is expected to work with on CUDA
                 dtypesIfROCM=None,  # dtypes this function is expected to work with on ROCM
                 backward_dtypes=None,  # backward dtypes this function is expected to work with
                 backward_dtypesIfCPU=None,  # backward dtypes this function is expected to work with on CPU
                 backward_dtypesIfCUDA=None,  # backward dtypes this function is expected to work with on CUDA
                 backward_dtypesIfROCM=None,  # backward dtypes this function is expected to work with on ROCM
                 default_test_dtypes=None,  # dtypes to test with by default. Tests are instantiated with
                                            # these dtypes for the op unless otherwise specified.
                                            # This is helpful in reducing the test matrix.
                 # the following metadata describes the operators out= support
                 supports_out=True,  # whether the op supports the out kwarg
                                     # defaults to True, if the op does not allow the out kwarg or
                                     # supports it incorrectly then test_out in test_ops.py should fail
                 safe_casts_outputs=False,  # whether op allows safe casting when writing to out arguments

                 # the following metadata relates to autograd support
                 supports_autograd=True,  # whether the operation supports backward mode AD
                                          # if true, gradient correctness is tested in test_ops.py
                                          # using the op's sample inputs
                 supports_gradgrad=None,  # whether the op supports second order gradients
                                          # if true, gradgrad correctness is tested in test_ops.py
                                          # defaults to support_autograd's value
                                          # TODO: rename this to supports_bwgrad_bwgrad to be consistent with below
                 supports_fwgrad_bwgrad=False,  # whether the ops supports second order gradients via
                                                # forward-over-reverse. If True, forward-over-reverse gradgrad correctness
                                                # is tested. If False, test that forward grad is not implemented.
                                                # Defaults to False.
                 supports_inplace_autograd=None,  # whether the operation supports inplace autograd
                                                  # if true, tested in test_ops.py
                                                  # defaults to supports_autograd's value
                 supports_forward_ad=False,  # Whether the operation support forward mode AD
                                             # If the value is True, we check that the gradients are correct
                                             # If the value is False, we test that forward grad is not implemented
                 gradcheck_wrapper=lambda op, *args, **kwargs: op(*args, **kwargs),  # wrapper function for gradcheck
                 check_batched_grad=None,  # whether to check batched grad when doing gradcheck
                                           # defaults to support_autograd's value
                 check_batched_gradgrad=None,  # whether to check batched grad grad when doing gradgradcheck
                                               # default's to support_gradgrad's value
                 check_batched_forward_grad=None,  # whether to check batched forward grad when doing gradcheck
                                                   # defaults to the value of `supports_forward_ad`
                 check_inplace_batched_forward_grad=None,   # whether to check batched forward grad when doing gradcheck
                                                            # defaults to the value of `check_batched_forward_grad`
                 gradcheck_nondet_tol=0.0,  # tolerance for nondeterminism while performing gradcheck
                 gradcheck_fast_mode=None,  # Whether to use the fast implmentation for gradcheck/gradgradcheck.
                                            # When set to None, defers to the default value provided by the wrapper
                                            # function around gradcheck (testing._internal.common_utils.gradcheck)

                 # the following metadata relates to JIT support and is tested for correctness in test_ops.py
                 aten_name=None,  # name of the corresponding aten:: operator
                 assert_autodiffed=False,  # if a op's aten::node is expected to be symbolically autodiffed
                 autodiff_nonfusible_nodes=None,  # a list of strings with node names that are expected to be in a
                                                  # DifferentiableGraph when autodiffed. Ex: ['aten::add', 'aten::mm'],
                                                  # default is populated to be ['aten::(name of Python operator)']
                 autodiff_fusible_nodes=None,  # a list of strings with node names that are expected to be in FusionGroups
                                               # inside of DifferentiableGraphs when this operation is autodiffed.
                                               # Ex: ['aten::add', 'aten::mm'], defaults to an empty list
                                               # Note: currently no ops use fusible nodes

                 # the following metadata relates to sparse support and is used in test_sparse.py
                 supports_sparse=False,  # whether the op supports sparse inputs

                 supports_scripting=True,  # only run tracing tests
                 # the following metadata relates to sparse csr support and is used in test_sparse_csr.py
                 supports_sparse_csr=False,  # whether the op supports sparse csr inputs
                 # the following metadata relates to complex support and is checked in test_ops.py
                 test_conjugated_samples=True,
                 test_neg_view=True,
                 assert_jit_shape_analysis=False,  # assert that jit shape analysis fully propagates shape
                 ):

        dtypes_args = (dtypes, dtypesIfCPU, dtypesIfCUDA, dtypesIfROCM)
        # Validates the dtypes are generated from the dispatch-related functions
        for dtype_list in dtypes_args:
            assert isinstance(dtype_list, (_dispatch_dtypes, type(None)))

        self.name = name
        self.ref = ref
        self.aten_name = aten_name if aten_name is not None else name
        self.variant_test_name = variant_test_name

        # Attribute to verify dynamic_dtypes are used.
        self.dynamic_dtypes = any(map(lambda dtypes: isinstance(
            dtypes, opinfo_helper._dynamic_dispatch_dtypes), dtypes_args))

        if self.dynamic_dtypes:
            # Make sure `dtyesIfCUDA` is dynamic, if dynamic dispatch is used for CPU
            # This is because, below we set dtypesIfCUDA to dtypes if they are None.
            assert isinstance(dtypesIfCUDA, opinfo_helper._dynamic_dispatch_dtypes), \
                (f"To use dynamic dypes for operator {name}, "
                 "acquire the dtypes dynamically for argument `dtypesIfCUDA`."
                 "This is to ensure that CUDA dtypes are acquired correctly as they"
                 "differ from CPU dtypes occasionally")

        self.dtypes = set(dtypes)

        # NOTE: backward dtypes must be acquired before forward dtypes
        #   since they fallback to explicit (not implicit!) specifications of
        #   forward dtypes
        self.backward_dtypes = set(backward_dtypes) if backward_dtypes is not None else self.dtypes
        self.backward_dtypesIfCPU = set(backward_dtypesIfCPU) if backward_dtypesIfCPU is not None else (
            backward_dtypes if backward_dtypes is not None
            else dtypesIfCPU if dtypesIfCPU is not None
            else dtypes)
        self.backward_dtypesIfCUDA = set(backward_dtypesIfCUDA) if backward_dtypesIfCUDA is not None else (
            backward_dtypes if backward_dtypes is not None
            else dtypesIfCUDA if dtypesIfCUDA is not None
            else dtypes)
        self.backward_dtypesIfROCM = set(backward_dtypesIfROCM) if backward_dtypesIfROCM is not None else (
            backward_dtypesIfCUDA if backward_dtypesIfCUDA is not None
            else backward_dtypes if backward_dtypes is not None
            else dtypesIfROCM if dtypesIfROCM is not None
            else dtypesIfCUDA if dtypesIfCUDA is not None
            else dtypes)

        self.dtypesIfCPU = set(dtypesIfCPU) if dtypesIfCPU is not None else self.dtypes
        self.dtypesIfCUDA = set(dtypesIfCUDA) if dtypesIfCUDA is not None else self.dtypes
        self.dtypesIfROCM = set(dtypesIfROCM) if dtypesIfROCM is not None else self.dtypesIfCUDA

        self._default_test_dtypes = set(default_test_dtypes) if default_test_dtypes is not None else None

        # NOTE: if the op is unspecified it is assumed to be under the torch namespace
        self.op = op if op else _getattr_qual(torch, self.name)
        method_variant = getattr(torch.Tensor, name, None) if method_variant is _NOTHING else method_variant
        # attributes like real, imag are not callable
        self.method_variant = method_variant if callable(method_variant) else None
        inplace_name = name + "_"
        self.inplace_variant = getattr(torch.Tensor, inplace_name, None) \
            if inplace_variant is _NOTHING else inplace_variant
        self.operator_variant = getattr(operator, name, None)

        self.supports_out = supports_out
        self.safe_casts_outputs = safe_casts_outputs

        self.decorators = (*decorators, *skips)

        # We run the sampling functions without tracking the gradiends of the creation of inputs
        self.sample_inputs_func = torch.no_grad()(sample_inputs_func)
        self.error_inputs_func = error_inputs_func

        self.assert_autodiffed = assert_autodiffed
        self.autodiff_fusible_nodes = autodiff_fusible_nodes if autodiff_fusible_nodes else []
        if autodiff_nonfusible_nodes is None:
            self.autodiff_nonfusible_nodes = ['aten::' + self.name]
        else:
            self.autodiff_nonfusible_nodes = autodiff_nonfusible_nodes

        # Autograd support

        # Autograd flags that don't depend on backward AD
        self.supports_autograd = supports_autograd
        self.supports_forward_ad = supports_forward_ad
        self.gradcheck_fast_mode = gradcheck_fast_mode
        self.gradcheck_wrapper = gradcheck_wrapper
        self.gradcheck_nondet_tol = gradcheck_nondet_tol

        # Autograd flags that depend on backward AD only
        # - If setting has been explicitly set, raise error if inconsistent
        if supports_gradgrad is None:
            supports_gradgrad = supports_autograd
        else:
            assert not (supports_gradgrad and not supports_autograd), (
                "supports_gradgrad refines the part of autograd is supported, so it should "
                "not be set if supports_autograd is False")
        if check_batched_grad is None:
            check_batched_grad = supports_autograd or supports_forward_ad
        else:
            assert not (check_batched_grad and not (supports_autograd or supports_forward_ad)), (
                "check_batched_grad refines the part of autograd that will be checked (by gradcheck), so "
                "it should not be set if supports_autograd is False")
        if check_batched_gradgrad is None:
            check_batched_gradgrad = supports_gradgrad
        else:
            assert not (check_batched_gradgrad and not supports_gradgrad), (
                "check_batched_gradgrad refines the part of autograd that will be checked (by "
                "gradgradcheck), so it should not be set if either supports_gradgrad or supports_autograd "
                "is False.")
        if check_batched_forward_grad is None:
            check_batched_forward_grad = supports_forward_ad
        else:
            assert not (check_batched_forward_grad and not supports_forward_ad), (
                "check_batched_forward_grad should only be used when supports_forward_ad "
                "is True. It is used to disable the test in the specific cases "
                "where the op supports forward ad but fails to compute "
                "batched forward grad.")

        if check_inplace_batched_forward_grad is None:
            check_inplace_batched_forward_grad = check_batched_forward_grad
        else:
            assert not (check_inplace_batched_forward_grad and not check_batched_forward_grad), (
                "check_batched_forward_grad should only be used when check_batched_forward_grad "
                "is True. It is used to disable the test in the specific cases "
                "where the op supports batched forward grad but fails to compute batched forward "
                "grad for the inplace variant of the op.")

        assert not (supports_fwgrad_bwgrad and not supports_autograd), (
            "supports_fwgrad_bwgrad enables forward-over-backward gradgrad checks and should only be "
            "True if backward ad is also checked, i.e., supports_forward_ad should be True.", self.name)

        self.supports_fwgrad_bwgrad = supports_fwgrad_bwgrad

        self.supports_gradgrad = supports_gradgrad
        self.check_batched_grad = check_batched_grad
        self.check_batched_gradgrad = check_batched_gradgrad
        self.check_batched_forward_grad = check_batched_forward_grad
        self.check_inplace_batched_forward_grad = check_inplace_batched_forward_grad

        # Autograd flags that depend on both forward AD and backward AD
        if supports_inplace_autograd is None:
            supports_inplace_autograd = supports_autograd or supports_forward_ad
        else:
            assert not (supports_inplace_autograd and not supports_autograd and not supports_forward_ad), (
                "supports_inplace_autograd refines the part of autograd that is supported, so "
                "it should not be set if both supports_autograd and supports_forward_ad are False")
        self.supports_inplace_autograd = supports_inplace_autograd

        self.supports_sparse = supports_sparse
        self.supports_sparse_csr = supports_sparse_csr

        self.aliases = ()
        if aliases is not None:
            self.aliases = tuple(AliasInfo(a) for a in aliases)  # type: ignore[assignment]

        self.supports_scripting = supports_scripting
        self.assert_jit_shape_analysis = assert_jit_shape_analysis

        self.test_conjugated_samples = test_conjugated_samples
        self.test_neg_view = test_neg_view

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

    def conjugate_sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs but with the tensor input or first
        tensor in a sequence input conjugated.
        """

        # TODO: Remove the try/except once all operators have sample_inputs_func with
        #       **kwargs in their signature.
        try:
            samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)
        except TypeError:
            samples = self.sample_inputs_func(self, device, dtype, requires_grad)

        conj_samples = list(samples)

        def conjugate(tensor):
            _requires_grad = tensor.requires_grad
            tensor = tensor.conj()
            return tensor.requires_grad_(_requires_grad)

        for i, sample in enumerate(samples):
            sample = conj_samples[i]
            # Note: it is assumed that the input here is either a tensor or tensorlist
            if isinstance(sample.input, torch.Tensor):
                sample.input = conjugate(sample.input)
            else:
                sample.input[0] = conjugate(sample.input[0])

        return tuple(conj_samples)

    def sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs.

        These samples should be sufficient to test the function works correctly
        with autograd, TorchScript, etc.
        """

        # TODO: Remove the try/except once all operators have sample_inputs_func with
        #       **kwargs in their signature.
        try:
            samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)
        except TypeError:
            samples = self.sample_inputs_func(self, device, dtype, requires_grad)

        if 'include_conjugated_inputs' in kwargs and kwargs.get('include_conjugated_inputs'):
            conj_samples = self.conjugate_sample_inputs(device, dtype, requires_grad, **kwargs)
            samples_list = list(samples)
            samples_list.extend(conj_samples)
            samples = tuple(samples_list)

        return samples

    def error_inputs(self, device, **kwargs):
        """
        Returns an iterable of ErrorInputs.
        """
        return self.error_inputs_func(self, device, **kwargs)

    def get_decorators(self, test_class, test_name, device, dtype):
        '''Returns the decorators targeting the given test.'''
        result = []
        for decorator in self.decorators:
            if isinstance(decorator, DecorateInfo):
                if decorator.is_active(test_class, test_name, device, dtype):
                    result.extend(decorator.decorators)
            else:
                result.append(decorator)
        return result

    def supported_dtypes(self, device_type):
        if device_type == 'cpu':
            return self.dtypesIfCPU
        if device_type == 'cuda':
            return self.dtypesIfROCM if TEST_WITH_ROCM else self.dtypesIfCUDA
        else:
            return self.dtypes

    def supported_backward_dtypes(self, device_type):
        if not self.supports_autograd:
            return set()

        backward_dtypes = None
        if device_type == 'cpu':
            backward_dtypes = self.backward_dtypesIfCPU
        elif device_type == 'cuda':
            backward_dtypes = self.backward_dtypesIfROCM if TEST_WITH_ROCM else self.backward_dtypesIfCUDA
        else:
            backward_dtypes = self.backward_dtypes

        allowed_backward_dtypes = floating_and_complex_types_and(torch.bfloat16, torch.float16)
        return set(allowed_backward_dtypes).intersection(backward_dtypes)

    def supports_complex_autograd(self, device_type):
        if device_type == 'cpu':
            return any(dtype.is_complex for dtype in self.backward_dtypesIfCPU)
        if device_type == 'cuda':
            if TEST_WITH_ROCM:
                return any(dtype.is_complex for dtype in self.backward_dtypesIfROCM)
            else:
                return any(dtype.is_complex for dtype in self.backward_dtypesIfCUDA)
        else:
            return any(dtype.is_complex for dtype in self.backward_dtypes)

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

    @property
    def formatted_name(self):
        """Returns a formatted full name for this OpInfo that can be used in test names."""
        variant = '_' + self.variant_test_name.replace('.', '_') if self.variant_test_name else ''
        return '{}{}'.format(self.name.replace('.', '_'), variant)


def _generate_reduction_inputs(device, dtype, requires_grad, **kwargs):
    """Generates input tensors for testing reduction operators"""
    yield make_tensor([], device, dtype, requires_grad=requires_grad)
    yield make_tensor([2], device, dtype, requires_grad=requires_grad)
    yield make_tensor([3, 5], device, dtype, requires_grad=requires_grad)
    yield make_tensor([3, 2, 1, 2], device, dtype, requires_grad=requires_grad)


def _generate_reduction_kwargs(ndim, supports_multiple_dims=True):
    """Generates a subset of all valid dim and keepdim kwargs given ndim that
    is appropriate for testing reduction operators.
    """

    # Test default dim and keepdim
    yield {}

    # Test reducing inner and outer most dimensions
    yield {'dim': 0, 'keepdim': True}
    yield {'dim': -1, 'keepdim': False}

    # Test reducing middle dimension
    if ndim > 2:
        yield {'dim': ndim // 2, 'keepdim': True}

    if supports_multiple_dims:
        # Test reducing all dimensions
        yield {'dim': tuple(range(ndim)), 'keepdim': False}

        # Test reducing both first and last dimensions
        if ndim > 1:
            yield {'dim': (0, -1), 'keepdim': True}

        # Test reducing every other dimension starting with the second
        if ndim > 3:
            yield {'dim': tuple(range(1, ndim, 2)), 'keepdim': False}


def sample_inputs_reduction(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for reduction operators."""

    # TODO(@heitorschueroff) Once all reduction operators are using
    # ReductionOpInfo use op_info.supports_multiple_dims directly.
    supports_multiple_dims: bool = kwargs.get('supports_multiple_dims', True)

    # TODO(@heitorschueroff) Once all reduction operators are using ReductionOpInfo
    # use op_info.genearte_args_kwargs directly.
    generate_args_kwargs = kwargs.get('generate_args_kwargs', lambda *args, **kwargs: (yield tuple(), {}))

    inputs: List[SampleInput] = []
    for t in _generate_reduction_inputs(device, dtype, requires_grad):
        for reduction_kwargs in _generate_reduction_kwargs(t.ndim, supports_multiple_dims):
            for args, kwargs in generate_args_kwargs(t, **reduction_kwargs):
                kwargs.update(reduction_kwargs)
                inputs.append(SampleInput(
                    t.clone().requires_grad_(requires_grad),
                    args=args,
                    kwargs=kwargs))

    return inputs


def _generate_masked_op_mask(input_shape, device, **kwargs):
    yield None
    yield make_tensor(input_shape, device, torch.bool, requires_grad=False)
    if len(input_shape) > 2:
        # broadcast last mask dimension:
        yield make_tensor(input_shape[:-1] + (1,), device, torch.bool, requires_grad=False)
        # broadcast middle mask dimension:
        yield make_tensor(input_shape[:1] + (1,) + input_shape[2:], device, torch.bool, requires_grad=False)
        # broadcast first mask dimension:
        yield make_tensor((1,) + input_shape[1:], device, torch.bool, requires_grad=False)
        # mask.ndim < input.ndim
        yield make_tensor(input_shape[1:], device, torch.bool, requires_grad=False)
        # mask.ndim == 1
        yield make_tensor(input_shape[-1:], device, torch.bool, requires_grad=False)
        # masks that require broadcasting of inputs (mask.ndim >
        # input.ndim) will not be supported, however, we may
        # reconsider this if there will be demand on this kind of
        # degenerate cases.


def sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked reduction operators.

    Masked reduction operator is a reduction operator with trailing
    mask optional argument. A mask is a bool tensor with the same
    shape as input or a shape that is broadcastable to input shape.
    """
    inputs: List[SampleInput] = []
    kwargs['supports_multiple_dims'] = op_info.supports_multiple_dims
    for sample_input in sample_inputs_reduction(op_info, device, dtype, requires_grad, **kwargs):
        for mask in _generate_masked_op_mask(sample_input.input.shape, device, **kwargs):
            sample_input_args, sample_input_kwargs = sample_input.args, dict(mask=mask, **sample_input.kwargs)
            inputs.append(SampleInput(sample_input.input.clone().requires_grad_(requires_grad),
                                      args=sample_input_args, kwargs=sample_input_kwargs))
            if(not requires_grad and dtype.is_floating_point and
               sample_input.input.ndim == 2 and mask is not None and
               mask.shape == sample_input.input.shape):
                for v in [torch.inf, -torch.inf, torch.nan]:
                    t = sample_input.input.clone()
                    t.diagonal()[:] = v
                    inputs.append(SampleInput(t.detach().requires_grad_(requires_grad),
                                              args=sample_input_args,
                                              kwargs=sample_input_kwargs))

    return inputs


def sample_inputs_masked_norm(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked norm.
    """
    inputs: List[SampleInput] = []
    for ord in [2.0, 1, float('inf'), float('-inf'), 0]:
        for sample_input in sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
            sample_input_args, sample_input_kwargs = (ord,) + sample_input.args, sample_input.kwargs.copy()
            inputs.append(SampleInput(sample_input.input.clone().requires_grad_(requires_grad),
                                      args=sample_input_args, kwargs=sample_input_kwargs))
    return inputs


def sample_inputs_masked_var(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked var.
    """
    inputs: List[SampleInput] = []
    for unbiased in [False, True]:
        for sample_input in sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
            if sample_input.args:
                dim = sample_input.args[0]
                sample_input_args = sample_input.args[:1] + (unbiased,) + sample_input.args[1:]
                sample_input_kwargs = sample_input.kwargs.copy()
            else:
                dim = sample_input.kwargs.get('dim')
                sample_input_args = sample_input.args
                sample_input_kwargs = dict(sample_input.kwargs, unbiased=unbiased)
            if requires_grad:
                inmask = torch._masked._input_mask(sample_input.input, *sample_input_args, **sample_input_kwargs)
                orig_count = torch._masked.sum(inmask.new_ones(sample_input.input.shape, dtype=torch.int64),
                                               dim, keepdim=True, mask=inmask)
                if orig_count.min() <= int(unbiased):
                    # Skip samples that lead to singularities in var
                    # computation resulting nan values both in var and
                    # autograd output that test_grad_fn cannot handle
                    # correctly.
                    continue
            inputs.append(SampleInput(sample_input.input.clone().requires_grad_(requires_grad),
                                      args=sample_input_args, kwargs=sample_input_kwargs))
    return inputs


# NOTE [Reductions]:
#
# For testing purposes, we relax the definition of a reduction operator
# as defined in the docstring below. We do this to capture operators with
# a similar API so they can be tested automatically. However...
#
# Strictly speaking a reduction operator is an operator that can reduce an
# array to a single scalar value and that can be computed from the partial
# result of reducing subarrays. This usually means that the reduction operation
# should be commutative and associative. This definition is important when it
# comes to implementation as it determines how a reduction can be parallelized.
#
# For example, many summary statistics such as median, mode and quantile cannot
# be computed from partial results because these are sorting and counting based
# algorithms that need information that would be lost in the reduced value.
class ReductionOpInfo(OpInfo):
    """Reduction operator information.

    An operator is a reduction operator if it reduces one or more dimensions of
    the input tensor to a single value. Reduction operators must implement the
    following signature:

    - `op(input, *args, *, dim=None, keepdim=False, **kwargs) -> Tensor`

    ReductionOpInfo tests that reduction operators implement a consistent API.
    Optional features such as reducing over multiple dimensions are captured in
    the optional keyword parameters of the ReductionOpInfo constructor.

    If a reduction operator does not yet implement the full required API of
    reduction operators, this should be documented by skipping the failing
    tests rather than adding optional parameters to ReductionOpInfo.

    NOTE
    The API for reduction operators has not yet been finalized and some
    requirements may change.

    See tests in test/test_reductions.py
    """

    def __init__(
        self, name, *,

        # The identity value for the operator if it has one.
        identity: Optional[Any] = None,

        # The nan policy for the operator if it implements one.
        # - propagate: NaN values are propagated to the output
        # - omit: NaN values are discarded during the reduction
        nan_policy: Optional[str] = None,

        # Whether the operator supports reducing multiple dimensions.
        supports_multiple_dims: bool = True,

        # Whether the operator promotes integral to floating point dtypes.
        promotes_int_to_float: bool = False,

        # Whether the operator promotes all integral dtypes to int64.
        promotes_int_to_int64: bool = False,

        # If a specific dtype is given, then the operator always returns that
        # dtype irrespective of the input dtype. If None, the operator returns
        # the dtype according to the type promotion rules above.
        result_dtype: Optional[torch.dtype] = None,

        # ReductionOpInfo tests generate their own input, dim and keepdim
        # arguments and call this function to generate tuples of extra args and
        # kwargs to use when calling the op. This is required for operators that
        # have other required parameters besides the input tensor.
        generate_args_kwargs: Callable = lambda t, dim=None, keepdim=False: (yield tuple(), {}),

        # Options from the OpInfo base class
        **kwargs,
    ):
        assert nan_policy in (None, 'propagate', 'omit')

        # These are mutually exclusive options
        assert not (result_dtype and promotes_int_to_float)
        assert not (result_dtype and promotes_int_to_int64)
        assert not (promotes_int_to_float and promotes_int_to_int64)

        # Default sample_inputs_func for ReductionOpInfo which augments sample
        # inputs from sample_inputs_reduction with the args and kwargs from
        # generate_args_kwargs. This is only used if sample_inputs_func is None.
        def sample_inputs_func(*args, **kwargs):
            kwargs['supports_multiple_dims'] = supports_multiple_dims
            kwargs['generate_args_kwargs'] = generate_args_kwargs
            return sample_inputs_reduction(*args, **kwargs)

        # Override OpInfo defaults and call base class __init__
        kwargs.setdefault('inplace_variant', None)
        kwargs.setdefault('sample_inputs_func', sample_inputs_func)
        kwargs.setdefault('default_test_dtypes', (
            torch.uint8, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.complex64))
        super(ReductionOpInfo, self).__init__(name, **kwargs)

        self.identity = identity
        self.nan_policy = nan_policy
        self.supports_multiple_dims = supports_multiple_dims
        self.promotes_int_to_float = promotes_int_to_float
        self.promotes_int_to_int64 = promotes_int_to_int64
        self.result_dtype = result_dtype
        self.generate_args_kwargs = generate_args_kwargs


def sample_inputs_unary(op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs):
    if not op_kwargs:
        op_kwargs = {}

    low, high = op_info.domain
    low = low if low is None else low + op_info._domain_eps
    high = high if high is None else high - op_info._domain_eps

    if op_info.supports_sparse_csr:
        # Tensors with dim=2 for sparse CSR testing
        yield SampleInput(make_tensor((L, L), device=device, dtype=dtype,
                                      low=low, high=high,
                                      requires_grad=requires_grad), kwargs=op_kwargs)
    else:
        # Creates a 1D, empty, and scalar tensor
        for shape in ((L,), (1, 0, 3), ()):
            yield SampleInput(make_tensor(shape, device=device, dtype=dtype,
                                          low=low, high=high,
                                          requires_grad=requires_grad), kwargs=op_kwargs)


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
                 dtypesIfCUDA=None,
                 dtypesIfROCM=None,
                 default_test_dtypes=(
                     torch.uint8, torch.long, torch.half, torch.bfloat16,
                     torch.float32, torch.cfloat),  # dtypes which tests check by default
                 domain=(None, None),  # the [low, high) domain of the function
                 handles_large_floats=True,  # whether the op correctly handles large float values (like 1e20)
                 handles_extremals=True,  # whether the op correctly handles extremal values (like inf)
                 handles_complex_extremals=True,  # whether the op correct handles complex extremals (like inf -infj)
                 supports_complex_to_float=False,  # op supports casting from complex input to real output safely eg. angle
                 sample_inputs_func=sample_inputs_unary,
                 sample_kwargs=lambda device, dtype, input: ({}, {}),
                 supports_sparse=False,
                 reference_numerics_filter=None,  # Filter for singular input values for test_reference_numerics_normal
                 **kwargs):
        super(UnaryUfuncInfo, self).__init__(name,
                                             dtypes=dtypes,
                                             dtypesIfCUDA=dtypesIfCUDA,
                                             dtypesIfROCM=dtypesIfROCM,
                                             default_test_dtypes=default_test_dtypes,
                                             sample_inputs_func=sample_inputs_func,
                                             supports_sparse=supports_sparse,
                                             **kwargs)
        self.ref = ref
        self.domain = domain
        self.handles_large_floats = handles_large_floats
        self.handles_extremals = handles_extremals
        self.handles_complex_extremals = handles_complex_extremals
        self.supports_complex_to_float = supports_complex_to_float
        self.reference_numerics_filter = reference_numerics_filter

        # test_unary_ufuncs.py generates its own inputs to test the consistency
        # of the operator on sliced tensors, non-contig tensors, etc.
        # `sample_kwargs` is a utility function to provide kwargs
        # along with those inputs if required (eg. clamp).
        # It should return two dictionaries, first holding kwarg for
        # torch operator and second one for reference NumPy operator.
        self.sample_kwargs = sample_kwargs

        # Epsilon to ensure grad and gradgrad checks don't test values
        #   outside a function's domain.
        self._domain_eps = 1e-5

def sample_inputs_tensor_split(op_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype,
                         low=None, high=None, requires_grad=requires_grad)

    args_cases = (
        # Cases with tensor indices.
        (torch.tensor([1, 2, 3]),),
        (torch.tensor(1),),
        (torch.tensor([1, 2, 3]), 1),
        (torch.tensor([1, 4, 2, 5, 3, 6])[::2], 1),
        # Cases with list of indices.
        ((2, 4),),
        ((2, 4), 1),
        ((2, 4), -1),
        # Cases with integer section.
        (3,),
        (3, 1),
        (3, -1),
    )

    for args in args_cases:
        yield SampleInput(make_input((S, S, S)), args=args)


def sample_inputs_linalg_det(op_info, device, dtype, requires_grad, **kwargs):
    kw = dict(device=device, dtype=dtype)
    inputs = [
        make_tensor((S, S), **kw),
        make_tensor((1, 1), **kw),  # 1x1
        random_symmetric_matrix(S, **kw),  # symmetric
        random_symmetric_psd_matrix(S, **kw),  # symmetric_psd
        random_symmetric_pd_matrix(S, **kw),  # symmetric_pd

        random_square_matrix_of_rank(S, S - 2, **kw),  # dim2_null
        random_square_matrix_of_rank(S, 1, **kw),  # rank1
        random_square_matrix_of_rank(S, 2, **kw),  # rank2

        make_fullrank_matrices_with_distinct_singular_values(S, S, **kw),  # full rank
        make_tensor((3, 3, S, S), **kw),  # batched
        make_tensor((3, 3, 1, 1), **kw),  # batched_1x1
        random_symmetric_matrix(S, 3, **kw),  # batched_symmetric
        random_symmetric_psd_matrix(S, 3, **kw),  # batched_symmetric_psd
        random_symmetric_pd_matrix(S, 3, **kw),  # batched_symmetric_pd
        make_fullrank_matrices_with_distinct_singular_values(S, 3, 3, **kw),  # batched fullrank
        make_tensor((0, 0), **kw),
        make_tensor((0, S, S), **kw),
    ]
    for t in inputs:
        t.requires_grad = requires_grad
    return [SampleInput(t) for t in inputs]

def sample_inputs_linalg_det_singular(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_singular_matrix_batch_base(size, rank):
        assert size[-1] == size[-2]
        assert rank > 0 and rank < size[-1]

        n = size[-1]
        a = make_arg(size[:-2] + (n, rank)) / 10
        b = make_arg(size[:-2] + (rank, n)) / 10
        x = a @ b
        lu, pivs, _ = torch.linalg.lu_factor_ex(x)
        p, l, u = torch.lu_unpack(lu, pivs)
        u_diag_abs = u.diagonal(0, -2, -1).abs()
        u_diag_abs_largest = u_diag_abs.max(dim=-1, keepdim=True).values
        u_diag_abs_smallest_idxs = torch.topk(u_diag_abs, k=(n - rank), largest=False).indices
        u.diagonal(0, -2, -1).div_(u_diag_abs_largest)
        u[..., u_diag_abs_smallest_idxs] = torch.finfo(dtype).eps
        matrix = p @ l @ u

        matrix.requires_grad_(requires_grad)
        return matrix

    def sample_generator():
        for batch, size in product(((), (2,), (2, 2)), range(6)):
            shape = batch + (size, size)
            for rank in range(1, size):
                yield make_singular_matrix_batch_base(shape, rank)

    return [SampleInput(t) for t in sample_generator()]


def sample_inputs_linalg_matrix_power(op_info, device, dtype, requires_grad, **kwargs):
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    make_arg_fullrank = partial(make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad)
    # (<matrix_size>, (<batch_sizes, ...>))
    test_sizes = [
        (1, ()),
        (2, (0,)),
        (2, (2,)),
    ]

    for matrix_size, batch_sizes in test_sizes:
        size = batch_sizes + (matrix_size, matrix_size)
        for n in (0, 3, 5):
            yield SampleInput(make_arg(size), args=(n,))
        for n in [-4, -2, -1]:
            yield SampleInput(make_arg_fullrank(*size), args=(n,))

def sample_inputs_hsplit(op_info, device, dtype, requires_grad, **kwargs):
    return (SampleInput(make_tensor((6,), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(2,),),
            SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=([1, 2, 3],),),)

def sample_inputs_vsplit(op_info, device, dtype, requires_grad, **kwargs):
    return (SampleInput(make_tensor((6, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(2,),),
            SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=([1, 2, 3],),),)

def sample_inputs_dsplit(op_info, device, dtype, requires_grad, **kwargs):
    return (SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=([1, 2, 3],),),
            SampleInput(make_tensor((S, S, 6), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(2,),),)

def error_inputs_hsplit(op_info, device, **kwargs):
    err_msg1 = ("torch.hsplit requires a tensor with at least 1 dimension, "
                "but got a tensor with 0 dimensions!")
    si1 = SampleInput(make_tensor((),
                                  dtype=torch.float32,
                                  device=device),
                      args=(0,),)
    err_msg2 = (f"torch.hsplit attempted to split along dimension 1, "
                f"but the size of the dimension {S} "
                f"is not divisible by the split_size 0!")
    si2 = SampleInput(make_tensor((S, S, S),
                                  dtype=torch.float32,
                                  device=device),
                      args=(0,),)
    return (ErrorInput(si1, error_type=RuntimeError, error_regex=err_msg1),
            ErrorInput(si2, error_type=RuntimeError, error_regex=err_msg2),)

def error_inputs_vsplit(op_info, device, **kwargs):
    err_msg1 = ("torch.vsplit requires a tensor with at least 2 dimension, "
                "but got a tensor with 1 dimensions!")
    si1 = SampleInput(make_tensor((S,),
                                  dtype=torch.float32,
                                  device=device),
                      args=(0,),)
    err_msg2 = (f"torch.vsplit attempted to split along dimension 0, "
                f"but the size of the dimension {S} "
                f"is not divisible by the split_size 0!")
    si2 = SampleInput(make_tensor((S, S, S),
                                  dtype=torch.float32,
                                  device=device),
                      args=(0,),)
    return (ErrorInput(si1, error_type=RuntimeError, error_regex=err_msg1),
            ErrorInput(si2, error_type=RuntimeError, error_regex=err_msg2),)

def error_inputs_dsplit(op_info, device, **kwargs):
    err_msg1 = ("torch.dsplit requires a tensor with at least 3 dimension, "
                "but got a tensor with 1 dimensions!")
    si1 = SampleInput(make_tensor((S,),
                                  dtype=torch.float32,
                                  device=device),
                      args=(0,),)
    err_msg2 = (f"torch.dsplit attempted to split along dimension 2, "
                f"but the size of the dimension {S} "
                f"is not divisible by the split_size 0!")
    si2 = SampleInput(make_tensor((S, S, S),
                                  dtype=torch.float32,
                                  device=device),
                      args=(0,),)
    return (ErrorInput(si1, error_type=RuntimeError, error_regex=err_msg1),
            ErrorInput(si2, error_type=RuntimeError, error_regex=err_msg2),)

def sample_inputs_linalg_multi_dot(op_info, device, dtype, requires_grad, **kwargs):
    # Each test case consists of the sizes in the chain of multiplications
    # e.g. [2, 3, 4, 5] generates matrices (2, 3) @ (3, 4) @ (4, 5)
    test_cases = [
        [1, 2, 1],
        [2, 0, 2],
        [0, 2, 2],
        [2, 2, 2, 2],
        [2, 3, 4, 5],
        [5, 4, 0, 2],
        [2, 4, 3, 5, 3, 2]
    ]

    result = []
    for sizes in test_cases:
        tensors = []
        for size in zip(sizes[:-1], sizes[1:]):
            t = make_tensor(size, device, dtype, requires_grad=requires_grad)
            tensors.append(t)
        result.append(SampleInput(tensors))

    return result

def sample_inputs_linalg_matrix_norm(op_info, device, dtype, requires_grad, **kwargs):
    sizes = ((2, 2), (2, 3, 2))
    ords = ('fro', 'nuc', inf, -inf, 1, -1, 2, -2)
    dims = ((-2, -1), (-1, 0))

    inputs: List[SampleInput] = []
    for size, ord, dim, keepdim in product(sizes, ords, dims, [True, False]):
        t = make_tensor(size, device, dtype, requires_grad=requires_grad)
        inputs.append(SampleInput(t, args=(ord, dim, keepdim)))

    return inputs

def sample_inputs_linalg_norm(op_info, device, dtype, requires_grad, **kwargs):
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

def sample_inputs_as_strided(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input shape, output shape, output stride, output storage offset
    test_cases = [
        ((1,), (1,), (1,), 0),
        ((3, 3), (2, 2), (1, 2), 0),
        ((3, 3), (2, 2), (1, 2), 1),
        ((16,), (2, 2, 2, 2), (1, 1, 1, 1), 0),
        ((16,), (2, 1, 1, 2), (1, 7, 7, 1), 0),
    ]

    samples = []

    for input_shape, output_shape, stride, storage_offset in test_cases:
        input_t = make_arg(input_shape)
        kwargs = dict(storage_offset=storage_offset)
        samples.append(SampleInput(input_t, args=(output_shape, stride), kwargs=kwargs))

    return samples

def sample_inputs_combinations(op_info, device, dtype, requires_grad, **kwargs):
    inputs = (
        (0,),
        (0, 1),
        (0, 1, 2, 3),
    )

    rvals = [1, 2, 4]

    products = product(inputs, rvals, [False, True])

    samples = []

    for input_data, r, with_replacement in products:
        input_t = torch.tensor(input_data, device=device, dtype=dtype, requires_grad=requires_grad)
        kwargs = dict(r=r, with_replacement=with_replacement)

        samples.append(SampleInput(input_t, kwargs=kwargs))

    return tuple(samples)

def sample_inputs_cartesian_prod(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(torch.tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # constructs 1-D tensors with varying number of elements
    a = make_arg((0,))
    b = make_arg((0, 1))
    c = make_arg((0, 1, 2, 3))

    samples = []

    # sample with only 1 tensor
    samples.append(SampleInput(
        a
    ))

    # sample with 2 tensors
    samples.append(SampleInput(
        a,
        args=(b,)
    ))

    # sample with 3 tensors
    samples.append(SampleInput(
        a,
        args=(b, c)
    ))

    return tuple(samples)

def sample_inputs_cosine_similarity(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as input_shape, dict of dim and eps
    cases: Tuple[tuple, dict] = (  # type: ignore[assignment]
        ((S, S), {'dim': 1}),
        ((S, 2), {'dim': -1}),
        ((S,), {'dim': 0, 'eps': 0.5}),
        ((), {'dim': 0}),
        ((S, S, M), {'dim': 2}),
        ((S, S), {})
    )

    for input_shape, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(make_arg(input_shape),), kwargs=kwargs)
    # Test for Broadcasting
    yield SampleInput(make_arg((1, 2, 3)), args=(make_arg((2, 1, 3)),), kwargs={'dim': -1})
    yield SampleInput(make_arg((1, 2, 3)), args=(make_arg((2, 1, 3)),), kwargs={'dim': -2})
    yield SampleInput(make_arg((2, 3)), args=(make_arg((2, 1, 3)),), kwargs={'dim': -1})

def sample_inputs_batch_norm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_arg_without_requires_grad = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # Ordered as: input shape, kwargs for training, momentum, eps
    cases: Tuple[Tuple[int], dict] = (  # type: ignore[assignment]
        ((S, S, S), {'training': True, 'momentum': 0.5, 'eps': 0.6}),
        ((3, 2, 4), {'training': False, 'momentum': -1.2}),
        ((3, 1), {'training': True, 'momentum': 0.0}),
        ((0,), {'training': True}),
        ((0,), {'training': False}),
        ((3, 2, 3, 4), {'training': True, 'momentum': -1.0, 'eps': 0.5}),
        ((3, 2, 3, 4), {'training': False, 'momentum': -1.0, 'eps': 0.5}),
        ((2, 1), {}),
    )

    for input_shape, kwargs in cases:
        # args: running mean, running var, weight and bias should necessarily be of shape: (channels,)
        channels = input_shape[1] if len(input_shape) > 1 else 0
        weight = make_arg(channels) if channels > 0 else None
        bias = make_arg(channels) if channels > 0 else None
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)

        yield SampleInput(
            make_arg(input_shape),
            args=(
                running_mean,
                running_var,
                weight,
                bias
            ),
            kwargs=kwargs
        )

    # Checking for permutations of weights and biases as `None`
    weights = [channels, None, None]
    biases = [None, channels, None]
    is_training = [True, False, False]

    for weight, bias, training in zip(weights, biases, is_training):
        yield SampleInput(
            make_arg(input_shape),
            args=(
                running_mean,
                running_var,
                make_arg(channels),
                make_arg(channels)
            ),
            kwargs={'training': training}
        )

    # Test case for no optional kwargs
    # running_mean and running_var are required in evaluation mode (training: False) but not in training mode
    yield SampleInput(make_arg((1, 2, 3)), args=(None, None), kwargs={'training': True})

def sample_inputs_nn_activation_relu(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        (()),
        ((S, )),
        ((S, S)),
        ((S, M, S))
    )

    for shape in cases:
        yield SampleInput(make_arg(shape))

def sample_inputs_nn_functional_prelu(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        (()),
        ((S, )),
        ((S, S)),
        ((S, M, S))
    )

    for shape in cases:
        for weight in [-1., 0., 0.8, 1.]:
            weight_tensor = torch.tensor(weight, device=device, dtype=dtype, requires_grad=requires_grad)
            yield SampleInput(make_arg(shape), kwargs=dict(weight=weight_tensor))

        if len(shape) >= 2:
            channel_size = shape[1]
            yield SampleInput(make_arg(shape), kwargs=dict(weight=make_arg((channel_size,))))

def sample_inputs_norm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), (2,), '2'),
        ((S, S), (0,), '0'),
        ((S, S), (0.5,), '0_5'),
        ((S, S), (1,), '1'),
        ((S, S), (3,), '3'),
        ((S, S), (-1,), 'neg_1'),
        ((S, S), (-2,), 'neg_2'),
        ((S, S), (-0.5,), 'neg_0_5'),
        ((S, S), (-1.5,), 'neg_1_5'),
    )

    cases_nonzero_input = (
        ((S, S, S), (1.5,), '1_5_default'),
        ((S, S, S), (1.5, 1), '1_5_dim'),
        ((S, S, S), (1.5, -1), '1_5_neg_dim'),
        ((S, S, S), (1.5, 1, True), 'keepdim_1_5_dim'),
        ((S, S, S), (1.5, -1, True), 'keepdim_1_5_neg_dim'),
    )

    cases_negdim_base = (
        ((S, S), (-2, 1,), 'neg_2_2_dim'),
        ((S, S), (-1, 1,), 'neg_1_2_dim'),
        ((S, S), (0, 1,), '0_2_dim'),
        ((S, S), (1, 1,), '1_2_dim'),
        ((S, S), (2, 1,), '2_2_dim'),
        ((S, S), (3, 1,), '3_2_dim'),
        ((S, S, S), (2, 1), '2_dim'),
        ((S, S, S), (3, 1), '3_dim'),
        ((S, S, S), (2, 1, True), 'keepdim_2_dim'),
        ((S, S, S), (3, 1, True), 'keepdim_3_dim'),
        ((), (2, 0), '2_dim_scalar'),
        ((), (3, 0), '3_dim_scalar'),
        ((), (2, 0, True), 'keepdim_2_dim_scalar'),
        ((), (3, 0, True), 'keepdim_3_dim_scalar'),
    )

    cases_negdim = []
    for case in cases_negdim_base:
        cases_negdim.append(case)
        shape, args, name = case
        new_args = copy.deepcopy(list(args))
        new_args[1] *= -1
        cases_negdim.append((shape, tuple(new_args), name.replace("_dim", "_neg_dim")))

    for shape, args, name in itertools.chain(cases, cases_negdim):
        yield SampleInput(make_arg(shape), args=args, name=name)

    for shape, args, name in cases_nonzero_input:
        yield SampleInput(make_arg(shape, exclude_zero=True), args=args, name=name)


def sample_inputs_norm_fro(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), (), 'default'),
        ((S, S), ('fro',), 'fro_default'),
        ((S, S), ('fro', [0, 1],), 'fro'),
    )

    for shape, args, name in cases:
        yield SampleInput(make_arg(shape), args=args, name=name)


def sample_inputs_norm_nuc(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), ('nuc',), 'nuc'),
        ((S, S, S), ('nuc', [1, 2]), 'nuc_batched'),
    )

    for shape, args, name in cases:
        yield SampleInput(make_arg(shape), args=args, name=name)


def sample_inputs_norm_inf(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (
        ((S, S), (-inf,), '-inf'),
        ((S, S), (inf,), 'inf'),
        ((S, S), (inf, 1,), 'inf_2_dim'),
        ((S, S), (inf, -1,), 'inf_2_neg_dim'),
    )

    for shape, args, name in cases:
        yield SampleInput(make_arg(shape), args=args, name=name)


def sample_inputs_linalg_vector_norm(op_info, device, dtype, requires_grad, **kwargs):
    size_1D = (S,)
    size_2D = (2, 2)

    test_cases = [
        # input size, ord, dim args
        (size_1D, 2, None),
        (size_1D, 2, (0,)),
        (size_1D, 0, None),
        (size_1D, 0, (0,)),
        (size_1D, 0.9, None),
        (size_1D, 0.9, (0,)),
        (size_1D, 1, None),
        (size_1D, 1, (0,)),
        (size_1D, -2.1, None),
        (size_1D, -2.1, (0,)),
        (size_1D, inf, None),
        (size_1D, inf, (0,)),
        (size_1D, -inf, None),
        (size_1D, -inf, (0,)),

        (size_2D, 2, None),
        (size_2D, 2, (0,)),
        (size_2D, 2, (-1, 0)),
        (size_2D, 0, None),
        (size_2D, 0, (0,)),
        (size_2D, 0, (-1, 0)),
        (size_2D, 0.9, None),
        (size_2D, 0.9, (0,)),
        (size_2D, 0.9, (-1, 0)),
        (size_2D, 1, None),
        (size_2D, 1, (0,)),
        (size_2D, 1, (-1, 0)),
        (size_2D, -2.1, None),
        (size_2D, -2.1, (0,)),
        (size_2D, -2.1, (-1, 0)),
        (size_2D, inf, None),
        (size_2D, inf, (0,)),
        (size_2D, inf, (-1, 0)),
        (size_2D, -inf, None),
        (size_2D, -inf, (0,)),
        (size_2D, -inf, (-1, 0)),
    ]
    inputs = []

    for test_size, ord, dim in test_cases:
        for keepdim in [False, True]:
            inputs.append(SampleInput(
                make_tensor(
                    test_size, device, dtype,
                    low=None, high=None,
                    requires_grad=requires_grad),
                args=(ord,),
                kwargs=dict(
                    keepdim=keepdim,
                    dim=dim)))

    return inputs


# Metadata class for binary "universal functions (ufuncs)" that accept two
# tensor and have common properties
class BinaryUfuncInfo(OpInfo):
    """Operator information for 'universal binary functions (binary ufuncs).'
    These are functions of two tensors with common properties like:
      - they are elementwise functions
      - the output shape is determined by the input shape
      - they typically have method and inplace variants
      - they typically support the out kwarg
      - they typically have NumPy or SciPy references
    See NumPy's universal function documentation
    (https://numpy.org/doc/stable/reference/ufuncs.html) for more details
    about the concept of ufuncs.
    """
    def __init__(self, name, *,
                 lhs_make_tensor_kwargs=None,
                 rhs_make_tensor_kwargs=None,
                 promotes_int_to_float=False,  # Set to true if the op promotes integer inputs to float
                 always_returns_bool=False,  # Set to true if the op always returns bool tensors
                 **kwargs):
        super().__init__(name, **kwargs)

        # [lr]hs_make_tensor_kwargs are part of the OpInfo to be able to dynamically generate valid samples later on.
        if lhs_make_tensor_kwargs is None:
            lhs_make_tensor_kwargs = {}
        self.lhs_make_tensor_kwargs = lhs_make_tensor_kwargs

        if rhs_make_tensor_kwargs is None:
            rhs_make_tensor_kwargs = {}
        self.rhs_make_tensor_kwargs = rhs_make_tensor_kwargs

        self.promotes_int_to_float = promotes_int_to_float
        self.always_returns_bool = always_returns_bool

def _resolve_binary_pwise_kwargs(
        op_info, *, op_kwargs=None, lhs_make_tensor_kwargs=None, rhs_make_tensor_kwargs=None
):
    """Resolves default values for :func:`sample_inputs_binary_pwise`.

    By default :attr:`op_kwargs`, :attr:`lhs_make_tensor_kwargs`, and :attr:`rhs_make_tensor_kwargs` are just empty
    dictionaries. In case :attr:`op_info` is a :class:`BinaryUfuncInfo`, :attr:`BinaryUfuncInfo.lhs_make_tensor_kwargs`
    and :attr:`BinaryUfuncInfo.rhs_make_tensor_kwargs` will be used as defaults.
    """
    if op_kwargs is None:
        op_kwargs = {}
    if lhs_make_tensor_kwargs is None:
        lhs_make_tensor_kwargs = op_info.lhs_make_tensor_kwargs if isinstance(op_info, BinaryUfuncInfo) else {}
    if rhs_make_tensor_kwargs is None:
        rhs_make_tensor_kwargs = op_info.rhs_make_tensor_kwargs if isinstance(op_info, BinaryUfuncInfo) else {}

    return op_kwargs, lhs_make_tensor_kwargs, rhs_make_tensor_kwargs


def sample_inputs_binary_pwise(
    op_info,
    device,
    dtype,
    requires_grad,
    *,
    python_scalars=False,
    op_kwargs=None,
    lhs_make_tensor_kwargs=None,
    rhs_make_tensor_kwargs=None,
    **kwargs,
):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    op_kwargs, lhs_make_tensor_kwargs, rhs_make_tensor_kwargs = _resolve_binary_pwise_kwargs(
        op_info,
        op_kwargs=op_kwargs,
        lhs_make_tensor_kwargs=lhs_make_tensor_kwargs,
        rhs_make_tensor_kwargs=rhs_make_tensor_kwargs,
    )

    scalar = make_arg((), **rhs_make_tensor_kwargs)
    if python_scalars:
        scalar = scalar.item()  # type: ignore[assignment]

    shapes = [
        ((), scalar),
        ((S,), scalar),
        ((S, 1), (S,)),
        ((M, S), scalar),
        ((S, M, S), (M, S)),
        ((S, M, S), (S, M, S)),
        ((M, 1, S), (M, S)),
        ((M, 1, S), (1, M, S)),
        ((0, 1, 3), (0, 10, 3))
    ]

    for shape_lhs, shape_rhs_or_scalar in shapes:
        lhs = make_arg(shape_lhs, **lhs_make_tensor_kwargs)
        if isinstance(shape_rhs_or_scalar, tuple):
            # shape
            rhs = make_arg(shape_rhs_or_scalar, **rhs_make_tensor_kwargs)
            broadcasts_input = torch.broadcast_shapes(shape_lhs, shape_rhs_or_scalar) != shape_lhs
        else:
            # scalar
            rhs = shape_rhs_or_scalar  # type: ignore[assignment]
            broadcasts_input = False

        yield SampleInput(lhs, args=(rhs,), kwargs=op_kwargs, broadcasts_input=broadcasts_input)


def sample_inputs_add_sub(
    op_info,
    device,
    dtype,
    requires_grad,
    python_scalars=False,
    alpha=1,
    op_kwargs=None,
    lhs_make_tensor_kwargs=None,
    rhs_make_tensor_kwargs=None,
    **kwargs,
):
    op_kwargs, lhs_make_tensor_kwargs, rhs_make_tensor_kwargs = _resolve_binary_pwise_kwargs(
        op_info,
        op_kwargs=op_kwargs,
        lhs_make_tensor_kwargs=lhs_make_tensor_kwargs,
        rhs_make_tensor_kwargs=rhs_make_tensor_kwargs,
    )

    yield from sample_inputs_binary_pwise(
        op_info,
        device,
        dtype,
        requires_grad,
        python_scalars=python_scalars,
        op_kwargs=op_kwargs,
        lhs_make_tensor_kwargs=lhs_make_tensor_kwargs,
        rhs_make_tensor_kwargs=rhs_make_tensor_kwargs,
        **kwargs,
    )

    lhs = make_tensor((S, S), device=device, dtype=dtype, requires_grad=requires_grad, **lhs_make_tensor_kwargs)
    rhs = make_tensor((S, S), device=device, dtype=dtype, requires_grad=requires_grad, **rhs_make_tensor_kwargs)
    yield SampleInput(lhs, args=(rhs,), kwargs=dict(op_kwargs, alpha=alpha), broadcasts_input=False)

def sample_inputs_isclose(
    op_info,
    device,
    dtype,
    requires_grad,
    python_scalars=False,
    op_kwargs=None,
    lhs_make_tensor_kwargs=None,
    rhs_make_tensor_kwargs=None,
    **kwargs,
):
    op_kwargs, lhs_make_tensor_kwargs, rhs_make_tensor_kwargs = _resolve_binary_pwise_kwargs(
        op_info,
        op_kwargs=op_kwargs,
        lhs_make_tensor_kwargs=lhs_make_tensor_kwargs,
        rhs_make_tensor_kwargs=rhs_make_tensor_kwargs,
    )

    yield from sample_inputs_binary_pwise(
        op_info,
        device,
        dtype,
        requires_grad,
        python_scalars=python_scalars,
        op_kwargs=op_kwargs,
        lhs_make_tensor_kwargs=lhs_make_tensor_kwargs,
        rhs_make_tensor_kwargs=rhs_make_tensor_kwargs,
        **kwargs,
    )

    rtols = [0., 1e-7]
    atols = [0., 1e-7]
    equal_nans = [False, True]

    products = product(rtols, atols, equal_nans)

    for rtol, atol, equal_nan in products:
        lhs = make_tensor((S, S), device=device, dtype=dtype, requires_grad=requires_grad, **lhs_make_tensor_kwargs)
        rhs = make_tensor((S, S), device=device, dtype=dtype, requires_grad=requires_grad, **rhs_make_tensor_kwargs)

        yield SampleInput(lhs, args=(rhs,),
                          kwargs=dict(op_kwargs, rtol=rtol, atol=atol, equal_nan=equal_nan))

def sample_inputs_t(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return (SampleInput(make_arg((1, 2))),
            SampleInput(make_arg((2,))),
            SampleInput(make_arg(())))


def sample_inputs_mm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_arg_conj(size):
        return make_arg(size).conj().requires_grad_(requires_grad)

    first_shape, second_shape = (S, M), (M, S)

    yield SampleInput(make_arg(first_shape), args=(make_arg(second_shape),))

    if dtype.is_complex:
        yield SampleInput(make_arg(first_shape), args=(make_arg_conj(second_shape),))


def sample_inputs_addmm(op_info, device, dtype, requires_grad, **kwargs):
    alpha_val = kwargs.get('alpha', 2 + 3j if dtype.is_complex else 0.6)
    beta_val = kwargs.get('beta', 1 + 2j if dtype.is_complex else 0.2)
    tests_list = [
        ((2, 3), (2, 2), (2, 3), False)
    ]
    tests_with_lhs_broadcasting = [
        ((1,), (2, 2), (2, 3), True),
        ((), (2, 2), (2, 3), True)
    ]
    test_cases = tests_list + tests_with_lhs_broadcasting  # type: ignore[operator]

    sample_inputs = []

    for shape_a, shape_b, shape_c, broadcasts_input in test_cases:
        sample_inputs.append(
            SampleInput(
                make_tensor(shape_a, device, dtype, requires_grad=requires_grad),
                args=(
                    make_tensor(shape_b, device, dtype,
                                requires_grad=requires_grad),
                    make_tensor(shape_c, device, dtype,
                                requires_grad=requires_grad)),
                kwargs={'alpha': alpha_val, 'beta': beta_val},
                broadcasts_input=broadcasts_input))

    if dtype.is_complex:
        shape = (3, 3)
        sample_inputs.append(
            SampleInput(make_tensor(shape, device, dtype, requires_grad=requires_grad),
                        args=(
                            make_tensor(shape, device, dtype).mH.requires_grad_(requires_grad),
                            make_tensor(shape, device, dtype,
                                        requires_grad=requires_grad)),
                        kwargs={'alpha': alpha_val, 'beta': beta_val},))
        sample_inputs.append(
            SampleInput(make_tensor(shape, device, dtype, requires_grad=requires_grad),
                        args=(
                            make_tensor(shape, device, dtype,
                                        requires_grad=requires_grad),
                            make_tensor(shape, device, dtype).mH.requires_grad_(requires_grad)),
                        kwargs={'alpha': alpha_val, 'beta': beta_val},))
    return sample_inputs

def sample_inputs_mv(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, M, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            )
        ),
    )

def sample_inputs_bmm(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((M, S, M, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((M, M, S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            )
        ),
    )

def sample_inputs_dot_vdot(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_arg_conj(size):
        return make_arg(size).conj().requires_grad_(requires_grad)

    sample_inputs = []
    sample_inputs.append(SampleInput(make_arg((S, )), args=(make_arg((S, )),)))
    if dtype.is_complex:
        # dot/vdot for (conj(input), conj(arg_tensor)) and (conj(input), arg_tensor)
        # is tested in test_conj_view (which tests operations with only conjugated input tensor
        # -- not conjugated arg tensors)
        sample_inputs.append(SampleInput(make_arg((S, )), args=(make_arg_conj((S, )),)))
    return sample_inputs

def sample_inputs_addmv(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    test_cases = (((S,), (S, M), (M,), 1, 1, False),
                  ((S,), (S, M), (M,), 0.2, 0.6, False),
                  )

    test_cases_with_broadcast = (((1,), (S, M), (M,), 1, 1, True),
                                 ((1,), (S, M), (M,), 0.2, 0.6, True),
                                 ((), (S, M), (M,), 1, 1, True),
                                 ((), (S, M), (M,), 0.2, 0.6, True),
                                 )

    cases = test_cases + test_cases_with_broadcast

    # addmv performs: beta * M + alpha * (mat @ vec)
    for size, mat, vec, beta, alpha, broadcasts_input in cases:
        yield SampleInput(make_arg(size), args=(make_arg(mat), make_arg(vec)),
                          kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=broadcasts_input)

def sample_inputs_addbmm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # input_shape, batch1_shape, batch2_shape, beta_val, alpha_val, is_broadcasting
    test_cases = [((S, M), (S, S, S), (S, S, M), 1, 1, False),
                  ((1,), (S, S, S), (S, S, M), 1, 1, True),
                  ((S, M), (S, S, S), (S, S, M), 0.6, 0.2, False),
                  ((1,), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ((), (S, S, S), (S, S, M), 1, 1, True),
                  ((), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ]

    for input_shape, batch1_shape, batch2_shape, beta, alpha, is_broadcasting in test_cases:
        if dtype.is_complex:
            beta_complex, alpha_complex = beta * (1 + 2j), alpha * (2 + 3j)
            yield SampleInput(make_arg(input_shape), args=(make_arg(batch1_shape), make_arg(batch2_shape)),
                              kwargs=dict(beta=beta_complex, alpha=alpha_complex), broadcasts_input=is_broadcasting)
        yield SampleInput(make_arg(input_shape), args=(make_arg(batch1_shape), make_arg(batch2_shape)),
                          kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=is_broadcasting)

def sample_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = [(((S, S), (S, S), (S, S)), False),
                  (((S, S), (S, 1), (1, S)), False),
                  (((1,), (S, S, 1), (1, S)), True),
                  (((), (), ()), False),
                  (((S, S), (), ()), True),
                  (((), (S, S, 1), (1, S)), True)
                  ]

    sample_inputs = []
    for input_args, broadcasts_input in test_cases:
        args = tuple(make_tensor(arg, device, dtype, requires_grad=requires_grad) if isinstance(arg, tuple) else arg
                     for arg in input_args)
        sample_inputs.append(SampleInput(
            args[0],
            args=args[1:],
            broadcasts_input=broadcasts_input))

        args = tuple(make_tensor(arg, device, dtype, requires_grad=requires_grad) if isinstance(arg, tuple) else arg
                     for arg in input_args)
        sample_inputs.append(SampleInput(
            args[0],
            args=args[1:],
            kwargs=dict(value=3.14), broadcasts_input=broadcasts_input))

    return tuple(sample_inputs)

def sample_inputs_baddbmm(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = [((S, S, M), (S, S, S), (S, S, M), 1, 1, False),
                  ((1,), (S, S, S), (S, S, M), 1, 1, True),
                  ((S, S, M), (S, S, S), (S, S, M), 0.6, 0.2, False),
                  ((1,), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ((), (S, S, S), (S, S, M), 1, 1, True),
                  ((), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ]
    sample_inputs = []
    for (input_shape, batch1_shape, batch2_shape, alpha, beta, broadcasts_input) in test_cases:
        args = (make_tensor(input_shape, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(batch1_shape, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(batch2_shape, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad))

        sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]),
                             kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=broadcasts_input))
        if dtype.is_complex:
            sample_inputs.append(SampleInput(
                args[0].clone().requires_grad_(requires_grad),
                args=(args[1].clone().requires_grad_(requires_grad),
                      args[2].clone().requires_grad_(requires_grad)),
                kwargs=dict(beta=beta * (1 + 2j), alpha=alpha * (2 + 3j)),
                broadcasts_input=broadcasts_input))

    if dtype.is_complex:
        shapes = [(S, S, S), (S, M, S), (S, S, M)]
        args = (make_tensor(shapes[0], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(shapes[1], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(shapes[2], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad))
        sample_inputs.append(
            SampleInput(
                args[0].transpose_(-1, 1),
                args=(args[1].transpose(-1, 1).conj().requires_grad_(requires_grad),
                      args[2].transpose(-1, 1).conj().requires_grad_(requires_grad)),
                kwargs=dict(beta=beta * (1 + 2j), alpha=alpha * (2 + 3j)),))

    return tuple(sample_inputs)

def sample_inputs_addr(op_info, device, dtype, requires_grad, **kwargs):
    input1 = SampleInput(
        make_tensor((S, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad)))

    input2 = SampleInput(
        make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad)),
        broadcasts_input=True)

    if dtype.is_complex:
        alpha, beta = 0.1 + 0.3j, 0.4 + 0.6j
    elif dtype.is_floating_point:
        alpha, beta = 0.2, 0.6
    else:
        alpha, beta = 2, 3

    input3 = SampleInput(
        make_tensor((S, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad)),
        kwargs=dict(beta=beta, alpha=alpha))

    input4 = SampleInput(
        make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad)),
        kwargs=dict(beta=beta, alpha=alpha),
        broadcasts_input=True)

    return (input1, input2, input3, input4)

def sample_inputs_xlogy(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((S, S), device, dtype, low=0, high=None, requires_grad=requires_grad),
            )
        ),
    )


def sample_inputs_xlog1py(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # same shape
    yield SampleInput(make_arg((S, S)), args=(make_arg((S, S), low=-1),))
    # rhs broadcast
    yield SampleInput(make_arg((S, S)), args=(make_arg((S,), low=-1),))
    # all zero `x`
    x = make_arg((S, S))
    x.fill_(0)
    yield SampleInput(x, args=(make_arg((S, S), low=-1),))

    # randomly zero-masked `x`
    x = make_arg((S, S))
    y = make_arg((S, S), low=-1)
    x[torch.rand(x.shape) > 0.5] = 0
    yield SampleInput(x, args=(y,))

    # Scalar x
    # `input` has to be a tensor
    # yield SampleInput(0, args=(make_arg((S, S), low=-1),))
    # yield SampleInput(2.1, args=(make_arg((S, S), low=-1),))

    # Scalar y
    yield SampleInput(make_arg((S, S)), args=(-0.5,))
    yield SampleInput(make_arg((S, S)), args=(1.2,))

def sample_inputs_zero_(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = ((), (S, S, S), (S,))

    for shape in cases:
        yield(SampleInput(make_arg(shape)))


def sample_inputs_logsumexp(self, device, dtype, requires_grad, **kwargs):
    inputs = (
        ((), (0,), True),
        ((S, S), (1,), True),
        ((S, S), (1,), False)
    )
    samples = []

    for shape, dim, keepdim in inputs:
        t = make_tensor(shape, device, dtype,
                        low=None, high=None,
                        requires_grad=requires_grad)
        samples.append(SampleInput(t, args=(dim, keepdim)))

    return tuple(samples)

def sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
    inputs = [
        ((), {}),
        ((S, S), {}),
        ((0, S, 0), {}),
        ((S,), {'dtype': dtype, 'device': device}),
        # Hard-code some dtypes/devices. We want to test cases where the
        # (dtype, device) is different from the input's (dtype, device)
        ((S,), {'dtype': torch.double}),
        ((S,), {'device': 'cpu'}),
        ((S,), {'dtype': torch.double, 'device': 'cpu'}),
    ]
    if torch.cuda.is_available():
        inputs.append(((S,), {'device': 'cuda'}))

    samples = []
    for shape, kwargs in inputs:
        t = make_tensor(shape, device, dtype,
                        low=None, high=None,
                        requires_grad=requires_grad)
        samples.append(SampleInput(t, kwargs=kwargs))

    return tuple(samples)

def get_independent_tensor(tensor):
    return tensor.clone().requires_grad_(tensor.requires_grad)

def sample_inputs_randint_like(self, device, dtype, requires_grad, **kwargs):
    samples = []
    low = 2
    high = 10

    for sample in sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
        # With high
        samples.append(SampleInput(
            sample.input,
            args=(high,) + sample.args,
            kwargs=sample.kwargs))
        # With low and high
        samples.append(SampleInput(
            get_independent_tensor(sample.input),
            args=(low, high,) + sample.args,
            kwargs=sample.kwargs))
    return tuple(samples)

def sample_inputs_new_fns(self, device, dtype, requires_grad, **kwargs):
    inputs = [
        ((), (), {}),
        ((S, S), (2, 0), {}),
        ((0, S, 0), (3, 2, 2), {}),
        ((S,), (2, 3), {'dtype': dtype, 'device': device}),
        # Hard-code some dtypes/devices. We want to test cases where the
        # (dtype, device) is different from the input's (dtype, device)
        ((S,), (10,), {'dtype': torch.double}),
        ((S,), (1, 1, 12), {'device': 'cpu'}),
        ((S,), (2, 2, 2), {'dtype': torch.double, 'device': 'cpu'}),
    ]
    if torch.cuda.is_available():
        inputs.append(((S,), (7, 2), {'device': 'cuda'}))

    samples = []
    for input_shape, output_shape, kwargs in inputs:
        t = make_tensor(input_shape, device, dtype,
                        low=None, high=None,
                        requires_grad=requires_grad)
        samples.append(SampleInput(t, args=(output_shape,), kwargs=kwargs))

    return tuple(samples)

def sample_inputs_new_full(self, device, dtype, requires_grad, **kwargs):
    def get_val(dtype):
        return make_tensor([], 'cpu', dtype).item()

    samples = []
    for sample in sample_inputs_new_fns(self, device, dtype, requires_grad, **kwargs):
        # The scalar we are passing to new_full must be the same dtype
        # as the one of the resulting tensor
        use_dtype = sample.kwargs['dtype'] if 'dtype' in sample.kwargs else dtype
        samples.append(SampleInput(
            sample.input, args=sample.args + (get_val(use_dtype),), kwargs=sample.kwargs))
    return tuple(samples)

def sample_inputs_full_like(self, device, dtype, requires_grad, **kwargs):
    def get_val(dtype):
        return make_tensor([], 'cpu', dtype).item()

    inputs = [
        ((), get_val(dtype), {}),
        ((S, S), get_val(dtype), {}),
        ((0, S, 0), get_val(dtype), {}),
        ((S,), get_val(dtype), {'dtype': dtype, 'device': device}),
        # Hard-code some dtypes/devices. We want to test cases where the
        # (dtype, device) is different from the input's (dtype, device)
        ((S,), get_val(torch.double), {'dtype': torch.double}),
        ((S,), get_val(dtype), {'device': 'cpu'}),
        ((S,), get_val(torch.double), {'dtype': torch.double, 'device': 'cpu'}),
    ]
    if torch.cuda.is_available():
        inputs.append(((S,), get_val(dtype), {'device': 'cuda'}))

    samples = []
    for shape, fill_value, kwargs in inputs:
        t = make_tensor(shape, device, dtype,
                        low=None, high=None,
                        requires_grad=requires_grad)
        samples.append(SampleInput(t, args=(fill_value,), kwargs=kwargs))

    return tuple(samples)

def sample_inputs_multinomial(self, device, dtype, requires_grad, **kwargs):
    cases = [
        ([3], 3, dict()),
        ([10], 3, dict()),
        ([3, 10], 3, dict()),
        ([3], 3, dict(replacement=False)),
        ([3], 3, dict(replacement=True)),
        ([3, 4], 4, dict(replacement=True)),
        ([3, 4], 4, dict(replacement=False)),
    ]

    samples = []
    for shape, num_samples, kwargs in cases:
        t = make_tensor(shape, device, dtype,
                        low=0, high=None,
                        requires_grad=requires_grad)
        samples.append(SampleInput(t, args=(num_samples,), kwargs=kwargs))
    return tuple(samples)

def sample_inputs_normal_common(self, device, dtype, requires_grad, cases, **kwargs):
    def get_value_or_make_tensor(value_or_shape):
        if isinstance(value_or_shape, list):
            return make_tensor(value_or_shape, device, dtype,
                               low=0, high=None,
                               requires_grad=requires_grad)
        return value_or_shape

    samples = []
    for value_or_mean_shape, value_or_std_shape, kwargs in cases:
        mean = get_value_or_make_tensor(value_or_mean_shape)
        std = get_value_or_make_tensor(value_or_std_shape)
        samples.append(SampleInput(mean, args=(std,), kwargs=kwargs))
    return tuple(samples)

def sample_inputs_normal_tensor_first(self, device, dtype, requires_grad, **kwargs):
    # value_or_size, value_or_size, kwargs
    cases = [
        ([], [], {}),
        ([3], [3], {}),
        ([3, 4, 2], [3, 4, 2], {}),
        ([2, 3], 1.1, {}),
    ]

    return sample_inputs_normal_common(self, device, dtype, requires_grad, cases, **kwargs)

def sample_inputs_normal_tensor_second(self, device, dtype, requires_grad, **kwargs):
    cases = [
        ([3, 4], 0.3, {}),
    ]
    return sample_inputs_normal_common(self, device, dtype, requires_grad, cases, **kwargs)

def sample_inputs_bernoulli(self, device, dtype, requires_grad, **kwargs):
    shapes = [
        [3],
        [],
        [0, 3],
        [2, 3, 4],
    ]

    samples = []
    for shape in shapes:
        t = make_tensor(shape, device, dtype,
                        low=0, high=1,
                        requires_grad=requires_grad)
        samples.append(SampleInput(t))
    return tuple(samples)

def sample_inputs_logcumsumexp(self, device, dtype, requires_grad, **kwargs):
    inputs = (
        ((S, S, S), 0),
        ((S, S, S), 1),
        ((), 0),
    )
    samples = []

    for large_number in (True, False):
        for shape, dim in inputs:
            t = make_tensor(shape, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad)

            if large_number and t.dim() > 0:
                t[0] = 10000
            samples.append(SampleInput(t, args=(dim,)))

    return tuple(samples)

def sample_inputs_trace(self, device, dtype, requires_grad, **kwargs):
    return (SampleInput((make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad))),)


def sample_inputs_renorm(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((S, S, S), (2, 1, 0.5)),
             ((S, S, S), (2, -1, 0.5)),
             ((S, S, S), (1, 2, 3)),
             ((S, S, S), (float('inf'), 2, 0.5)),
             )

    for shape, args in cases:
        yield SampleInput(make_arg(shape), args=args)


def sample_inputs_transpose_swapdims(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((1, 2, 3), (-1, -2)),
             ((1, 2, 3), (-1, 2)),
             ((1, 2, 3), (1, -2)),
             ((1, 2, 3), (1, 2)),
             ((), (0, 0)),
             ((1, ), (0, 0)),
             ((M, M), (0, 1)),
             ((S, S, S), (2, 0)), )

    for shape, args in cases:
        yield SampleInput(make_arg(shape), args=args)

def sample_inputs_adjoint(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    shapes = ((1, 2, 3), (), (M, M), (S, S, S), (S, M, S), (M, S, M, S))
    return (SampleInput(make_arg(shape)) for shape in shapes)

def sample_inputs_T(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    shapes = ((), (M, M))
    return (SampleInput(make_arg(shape)) for shape in shapes)


def sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates invertible inputs for linear algebra ops
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    make_fn = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(make_fn, dtype=dtype, device=device, requires_grad=requires_grad)

    batches = [(), (0, ), (2, ), (1, 1)]
    ns = [5, 0]

    for batch, n in product(batches, ns):
        yield SampleInput(make_arg(*batch, n, n))

def sample_inputs_linalg_pinv_singular(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function produces factors `a` and `b` to generate inputs of the form `a @ b.t()` to
    test the backward method of `linalg_pinv`. That way we always preserve the rank of the
    input no matter the perturbations applied to it by the gradcheck.
    Note that `pinv` is Frechet-differentiable in a rank-preserving neighborhood.
    """
    batches = [(), (0, ), (2, ), (1, 1)]
    # the size of at least 30 is required to cause failures for the previous implicit implementation
    # of the pinv's backward method, albeit it is slow.
    size = [0, 3, 50]

    for batch, m, n in product(batches, size, size):
        for k in range(min(3, min(m, n))):
            # Note that by making the columns of `a` and `b` orthonormal we make sure that
            # the product matrix `a @ b.t()` has condition number 1 when restricted to its image
            a = torch.rand(*batch, m, k, device=device, dtype=dtype).qr().Q.requires_grad_(requires_grad)
            b = torch.rand(*batch, n, k, device=device, dtype=dtype).qr().Q.requires_grad_(requires_grad)
            yield SampleInput(a, args=(b,))


def sample_inputs_singular_matrix_factors(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function produces two tensors of shape (*, m, k) and (*, n, k) with k <= min(m, n).
    Their matrix product could be used to generate tensor of shape (*, m, n) of rank k.
    """

    batches = [(), (0, ), (2, ), (1, 1)]
    size = [1, 5, 10]

    for batch, m, n in product(batches, size, size):
        for k in range(min(3, min(m, n))):
            a = make_tensor((*batch, m, k), device, dtype, requires_grad=requires_grad)
            b = make_tensor((*batch, n, k), device, dtype, requires_grad=requires_grad)
            yield SampleInput(a, args=(b,), kwargs=kwargs)


def clone_sample(sample, **kwargs):
    """
    Given a SampleInput, this function analyzes its input, args and kwargs,
    and produces a copy with each non-Tensor entry being copied by reference,
    and with each Tensor entry cloned with `t.clone().requires_grad_(t.requires_grad)`
    """

    def clone_tensor(t):
        if isinstance(t, torch.Tensor):
            return t.clone().requires_grad_(t.requires_grad)
        else:
            return t

    sample_kwargs = kwargs if kwargs else sample.kwargs

    return SampleInput(
        clone_tensor(sample.input),
        args=tuple(map(clone_tensor, sample.args)),
        kwargs=dict(((k, clone_tensor(v)) for k, v in sample_kwargs.items()))
    )


def sample_inputs_svd_lowrank(op_info, device, dtype, requires_grad=False, **kwargs):
    for sample in sample_inputs_singular_matrix_factors(op_info, device, dtype, requires_grad, **kwargs):
        *batch, m, k = sample.input.shape
        *_, n, _ = sample.args[0].shape

        # NOTE: since svd_lowrank relies on non rank-revealing SVD,
        # it inherits the problem of unstable behavior with repeated
        # singular values including zeros.
        # Since we want to avoid (repeated) zeros as singular values,
        # we can only use k for q.
        # This issues could be resolved with using a rank-revealing SVD
        # which does not include "zero" singular values.
        op_kwargs = {
            'q': k,
            'M': None
        }

        # without M specified
        yield clone_sample(sample, **op_kwargs)

        # now with M
        # TODO: fix bug in the documentation for svd_lowrank:
        # M has to be (*, m, n), and not (*, 1, n) as written
        # in the documentation
        op_kwargs['M'] = make_tensor((*batch, m, n), device, dtype, requires_grad=requires_grad)
        yield clone_sample(sample, **op_kwargs)

def chunk_iter(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            break
        yield chunk

def sample_inputs_pca_lowrank(op_info, device, dtype, requires_grad=False, **kwargs):
    # we reuse samples from svd_lowrank which come in group of two with
    # kwarg['M'] = None and with kwarg['M'] = <some tensor>
    samples = sample_inputs_svd_lowrank(op_info, device, dtype, requires_grad, **kwargs)
    for s1, s2 in chunk_iter(samples, 2):
        del s1.kwargs['M']
        del s2.kwargs['M']
        s1.kwargs['center'] = False
        s2.kwargs['center'] = True
        yield s1
        yield s2

def sample_inputs_linalg_cond(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    # autograd is not supported for inputs with zero number of elements
    shapes = ((S, S),
              (2, S, S),
              (2, 1, S, S), )

    for shape in shapes:
        yield SampleInput(make_arg(shape))

def np_sinc_with_fp16_as_fp32(x):
    # Wraps numpy's sinc function so that fp16 values are promoted to fp32
    # before sinc is invoked. Context: numpy's sinc returns NaN when evaluated
    # at 0 for fp16.
    if x.dtype == np.float16:
        return np.sinc(x.astype(np.float32))
    else:
        return np.sinc(x)

def sample_inputs_broadcast_to(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((S, 1, 1), (S, S, S)),
        ((S, 1, S), (S, S, S)),
        ((S, 1), (S, S, S)),
        ((1,), (S, S, S)),
        ((1, S), (1, 1, S)),
        ((), ()),
        ((), (1, 3, 2)),
    )

    return tuple(
        SampleInput(
            make_tensor(size, device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(shape,)) for size, shape in test_cases)

def sample_inputs_broadcast_tensors(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases: Tuple[tuple] = (((3,), (1, 2, 1), (1, 1), (5, 1, 1),),)

    samples: List[SampleInput] = []
    for shape, *other_shapes in test_cases:
        samples.append(SampleInput(make_arg(shape), args=tuple(make_arg(s) for s in other_shapes)))

    return samples

def sample_inputs_block_diag(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases: Tuple[tuple] = (((1, S), (2, S), (3, S),),)

    samples: List[SampleInput] = []
    for shape, *other_shapes in test_cases:
        samples.append(SampleInput(make_arg(shape), args=tuple(make_arg(s) for s in other_shapes)))

    return samples

def sample_inputs_bitwise_shift(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        (S, S, S),
        (S,),
        (),
    )

    sample_inputs = []
    for size in test_cases:
        tensor1 = make_tensor(size, device, dtype, low=-32, high=32, requires_grad=requires_grad)
        tensor2 = make_tensor(size, device, dtype, low=0, high=5, requires_grad=requires_grad)
        sample_inputs.append(SampleInput(tensor1, args=(tensor2,)))
        sample_inputs.append(SampleInput(tensor1, args=(2,)))

    return tuple(sample_inputs)


def sample_inputs_cdist(op_info, device, dtype, requires_grad, **kwargs):
    small_S = 2
    test_cases = (
        ((S, S, 2), (S, S + 1, 2)),
        ((S, S), (S, S)),
        ((S, S, S), (S, S, S)),
        ((3, 5), (3, 5)),
        ((2, 3, 5), (2, 3, 5)),
        ((1, 2, 3), (1, 2, 3)),
        ((1, 1), (S, 1)),
        ((0, 5), (4, 5)),
        ((4, 5), (0, 5)),
        ((0, 4, 5), (3, 5)),
        ((4, 5), (0, 3, 5)),
        ((0, 4, 5), (1, 3, 5)),
        ((1, 4, 5), (0, 3, 5)),
        # Using S here would make this one test take 9s
        ((small_S, small_S, small_S + 1, 2), (small_S, small_S, small_S + 2, 2)),
        ((small_S, 1, 1, small_S), (1, small_S, small_S)),
        ((1, 1, small_S), (small_S, 1, small_S, small_S)),
    )

    samples = []
    for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
        # FIXME add an override for JIT and revert 0. back to 0
        # since it's accepted by eager
        for p in [0., 1., 2., 3., 0.5, 1.5, 2.5, float("inf")]:
            for t1_size, t2_size in test_cases:
                # The args should never be non-contiguous as this is not supported in the backward
                samples.append(SampleInput(
                    make_tensor(t1_size, device, dtype, requires_grad=requires_grad),
                    args=(make_tensor(t2_size, device, dtype, requires_grad=requires_grad), p, cm)))

    return samples


def sample_inputs_fill_(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype,
                       low=None, high=None, requires_grad=requires_grad)

    cases = (((S, S, S), (1,)),
             ((), (1,)),
             ((S, S, S), (make_arg(()),)))

    for shape, args in cases:
        yield SampleInput(make_arg(shape), args=args)


def sample_inputs_comparison_ops(self, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((S, S, S), (S, S, S), False),
        ((S, S, S), (), False),
        ((S, S, S), (1,), False),
        ((S,), (1,), False),
        ((), (), False),
    )
    test_cases_lhs_broadcasting = (
        ((S, 1, S), (S, S, S), True),
        ((1,), (S, S, S), True),
        ((1, S), (1, 1, S), True),
        ((), (0,), True),
        ((), (S, S, S), True),
    )
    cases = test_cases + test_cases_lhs_broadcasting
    sample_inputs = list(SampleInput(make_tensor(first_shape, device, dtype,
                                                 requires_grad=requires_grad),
                                     args=(make_tensor(second_shape, device, dtype,
                                                       requires_grad=requires_grad),),
                                     broadcasts_input=broadcasts_input)
                         for first_shape, second_shape, broadcasts_input in cases)
    equal_tensors_non_bool = (
        ([[[-8, 6], [9, 0]], [[0, 5], [5, 7]]]),
        ([[[6, 5]], [[1, -5]]]),
        ([[2], [-1]]),
        ([0, -6]),
        ([3],),
    )
    equal_tensors_bool = (
        ([[[1, 0], [0, 0]], [[0, 1], [1, 0]]]),
        ([[[1, 1]], [[1, 0]]]),
        ([[1], [0]]),
        ([0, 1]),
        ([1],),
    )
    more_cases = equal_tensors_bool if dtype is torch.bool else equal_tensors_non_bool
    more_inputs = list(SampleInput(torch.tensor(elements, device=device, dtype=dtype,
                                                requires_grad=requires_grad),
                                   args=(torch.tensor(elements, device=device, dtype=dtype,
                                                      requires_grad=requires_grad),))
                       for elements in more_cases)
    sample_inputs = [*sample_inputs, *more_inputs]
    return tuple(sample_inputs)


def sample_inputs_stack(op_info, device, dtype, requires_grad, **kwargs):
    tensors = [
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
    ]

    return (SampleInput(tensors, args=(0,)),)

def sample_inputs_cat_concat(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases: Tuple[tuple, tuple, dict] = (  # type: ignore[assignment]
        ((S, S), (S, S), {'dim': -1}),
        ((S, S), (S, S), {'dim': 1}),
        ((M, S), (S, S), {'dim': 0}),  # different shapes
        ((1, 2, 3), (1, 2, 3), {'dim': -2}),
        ((0,), (0,), {'dim': 0}),  # empty tensor
        ((0, S), (S, S), {'dim': 0}),
        ((1,), (1,), {})  # dim not passed, fallback to default
    )

    for input_shape1, input_shape2, kwargs in cases:
        yield SampleInput([make_arg(input_shape1), make_arg(input_shape2)], kwargs=kwargs)

def sample_inputs_hstack_dstack_vstack(op_info, device, dtype, requires_grad, **kwargs):
    tensors = [
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
    ]

    return (SampleInput(tensors),)

def sample_inputs_hypot(op_info, device, dtype, requires_grad, **kwargs):
    input = make_tensor((S, S), device, dtype, requires_grad=requires_grad)
    args = make_tensor((S, S), device, dtype, requires_grad=requires_grad)

    return (
        SampleInput(input, args=(args,)),
    )

def sample_inputs_gather(op_info, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, gather_variable((S, S), 1, M, True, device=device))),
        SampleInput(
            make_tensor((M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(1, gather_variable((M, S // 2), 0, S, True, device=device))),
        SampleInput(
            make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor([0], dtype=torch.int64, device=device))),
        # Empty index tensor case, see: https://github.com/pytorch/pytorch/pull/65006
        SampleInput(
            make_tensor((S,), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor([], dtype=torch.uint8, device=device))),
        SampleInput(
            make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor(0, dtype=torch.int64, device=device))),
    )

def _fill_indices(idx, dim, dim_size, elems_per_row, m, n, o):
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, idx.size(dim) + 1)
                idx[tuple(ii)] = torch.randperm(dim_size)[0:elems_per_row]

def error_inputs_gather(op_info, device, **kwargs):
    # src is [1, 2]
    #        [3, 4]
    src = torch.tensor(((1, 2), (3, 4)), device=device, dtype=torch.float32)

    # idx is [0, 0]
    #        [1, 0]
    idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)

    # Index should be smaller than self except on dimesion 1
    bad_src = make_tensor((1, 1), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(bad_src, args=(1, idx,)), error_type=RuntimeError,
                     error_regex="Size does not match at dimension 0")

    # Index must have long dtype
    bad_idx = idx.to(torch.int32)
    yield ErrorInput(SampleInput(src, args=(1, bad_idx)), error_type=RuntimeError,
                     error_regex="Expected dtype int64 for index")

    # TODO: FIXME
    # out.dtype must match src.dtype
    # Creates new src & idx since SampleInputs can't share tensors
    src = torch.tensor(((1, 2), (3, 4)), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)
    out = torch.empty((2, 2), device=device, dtype=torch.float64)
    yield ErrorInput(SampleInput(src, args=(1, idx), kwargs={'out': out}), error_type=RuntimeError,
                     error_regex="Expected out tensor to have dtype")

    # src and index tensors must have the same # of dimensions
    # idx too few dimensions
    src = torch.tensor(((1, 2), (3, 4)), device=device, dtype=torch.float32)
    idx = torch.tensor((0, 0), device=device, dtype=torch.long)
    yield ErrorInput(SampleInput(src, args=(1, idx)), error_type=RuntimeError,
                     error_regex="Index tensor must have the same number of dimensions")

    # src too few dimensions
    src = torch.tensor((1, 2), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)
    yield ErrorInput(SampleInput(src, args=(0, idx)), error_type=RuntimeError,
                     error_regex="Index tensor must have the same number of dimensions")

    # index out of bounds
    # NOTE: this ErrorInput is guarded because bounds checking does not occur on CUDA devices
    if torch.device(device).type == 'cpu':
        src = torch.tensor(((1, 2), (3, 4)), device=device, dtype=torch.float32)
        idx = torch.tensor(((0, 23), (1, 0)), device=device, dtype=torch.long)
        yield ErrorInput(SampleInput(src, args=(1, idx,)), error_type=RuntimeError,
                         error_regex="index 23 is out of bounds for dimension")

# Error inputs for scatter
def error_inputs_scatter_and_scatter_add(op_info, device, **kwargs):
    # Error when self.dtype != src.dtype (and src is not a scalar)
    src = make_tensor((2, 5), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 1), (1, 2)), device=device, dtype=torch.long)
    dst = torch.zeros((3, 5), device=device, dtype=torch.double)
    yield ErrorInput(SampleInput(dst, args=(0, idx, src)), error_type=RuntimeError,
                     error_regex="Expected self.dtype to be equal to src.dtype")

    # Index dtype must be long
    src = make_tensor((2, 5), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 1), (1, 2)), device=device, dtype=torch.int32)
    dst = torch.zeros((3, 5), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(dst, args=(0, idx, src)), error_type=RuntimeError,
                     error_regex="Expected dtype int64 for index")

    # Index and destination must have the same number of dimensions
    src = make_tensor((2, 5), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 1), (1, 2)), device=device, dtype=torch.long)
    dst = torch.zeros((3, 5, 3), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(dst, args=(0, idx, src)), error_type=RuntimeError,
                     error_regex="Index tensor must have the same number of dimensions as self tensor")

    # Index and src must have the same number of dimensions when src is not a scalar
    src = make_tensor((2, 5, 2), device=device, dtype=torch.float32)
    idx = torch.tensor(((34, 1), (1, 2)), device=device, dtype=torch.long)
    dst = torch.zeros((3, 5), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(dst, args=(0, idx, src)), error_type=RuntimeError,
                     error_regex="Index tensor must have the same number of dimensions as src tensor")

    # Index out of bounds
    # NOTE: this ErrorInput is guarded because bounds checking does not occur on CUDA devices
    if torch.device(device).type == 'cpu':
        src = make_tensor((2, 5), device=device, dtype=torch.float32)
        idx = torch.tensor(((34, 1), (1, 2)), device=device, dtype=torch.long)
        dst = torch.zeros((3, 5), device=device, dtype=torch.float32)
        yield ErrorInput(SampleInput(dst, args=(0, idx, src)), error_type=RuntimeError,
                         error_regex="index 34 is out of bounds for dimension 0 with size 3")

def sample_inputs_take_along_dim(op_info, device, dtype, requires_grad, **kwargs):
    return (SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S), 1, S, True, device=device), 0)),

            # `indices` broadcast
            SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((1, S // 2), 0, S, True, device=device), 1)),

            # `self` broadcast
            SampleInput(make_tensor((1, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S // 2), 0, S, True, device=device), 1)),

            # without `dim` arg
            SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S // 2), 0, S, True, device=device), )),
            SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S // 2), 0, S, True, device=device),)),
            )


def sample_inputs_aminmax(op_info, device, dtype, requires_grad, **kwargs):
    test_cases: Tuple[tuple, dict] = (  # type: ignore[assignment]
        ((S, S, S), {}),
        ((S, S, S), {'dim': 1}),
        ((S, S, S), {'dim': 1, 'keepdim': True}),
        ((), {'dim': 0}),
        ((), {}),
        ((), {'dim': 0, 'keepdim': True}),
    )

    samples: List[SampleInput] = []
    for shape, kwargs in test_cases:
        samples.append(SampleInput(
            make_tensor(shape, device, dtype, requires_grad=requires_grad),
            kwargs=kwargs))

    return samples

def sample_inputs_diff(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    test_cases = (
        ((1,), 0, None, None),
        ((S,), 0, None, None),
        ((S, 1), 0, None, None),
        ((S, 1), 1, None, None),
        ((S, S), 0, None, None),
        ((S, S), 1, None, None),
        ((S, S), 0, (1, S), (2, S)),
        ((S, S), 0, None, (2, S)),
        ((S, S, S), 1, None, None),
        ((S, S, S), 2, None, None),
        ((S, S, S), 1, (S, 1, S), (S, 1, S)),
        ((S, S, S), 2, (S, S, 1), (S, S, 1)),
        ((S, S, S), 2, (S, S, S), (S, S, S)),)

    sample_inputs = []
    for size, dim, size_prepend, size_append in test_cases:
        prepend_size = 0 if (size_prepend is None) else size_prepend[dim]
        append_size = 0 if (size_append is None) else size_append[dim]
        dim_size = size[dim] + prepend_size + append_size
        for n in range(dim_size):
            input_tensor = make_arg(size)
            prepend = make_arg(size_prepend) if size_prepend else None
            append = make_arg(size_append) if size_append else None
            sample_inputs.append(SampleInput(input_tensor, args=(n, dim, prepend, append,)))

    # add some samples with n > dim_size
    sample_inputs.append(SampleInput(make_arg((S, S, S)), args=(S + 1, 1,)))
    sample_inputs.append(SampleInput(make_arg((S, S, S)), args=(S * 3 + 2, 2, make_arg((S, S, S)), make_arg((S, S, S)),)))

    return sample_inputs

def sample_inputs_histogram(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))

    sample_inputs = []
    for size, bin_ct, weighted, density in product(sizes, range(1, 5), [False, True], [False, True]):
        input_tensor = make_arg(size)
        weight_tensor = make_arg(size) if weighted else None

        sample_inputs.append(SampleInput(input_tensor, args=(bin_ct,),
                                         kwargs=dict(weight=weight_tensor, density=density)))

        bins_tensor = make_arg((bin_ct + 1,))
        sample_inputs.append(SampleInput(input_tensor, args=(bins_tensor,),
                                         kwargs=dict(weight=weight_tensor, density=density)))

    return sample_inputs

def sample_inputs_histogramdd(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    sizes = ((S, S), (S, S, S), (S, 1, S), (S, 0, S))
    bin_ct_patterns = ((1, 1, 1, 1, 1), (2, 3, 2, 3, 2), (3, 2, 3, 2, 3))

    sample_inputs = []
    for size, bin_ct_pattern, weighted, density in product(sizes, bin_ct_patterns, [False, True], [False, True]):
        input_tensor = make_arg(size)
        bin_ct = bin_ct_pattern[:size[-1]]
        weight_tensor = make_arg(size[:-1]) if weighted else None

        sample_inputs.append(SampleInput(input_tensor, args=(bin_ct,),
                                         kwargs=dict(weight=weight_tensor, density=density)))

        bins_tensor = [make_arg(ct + 1) for ct in bin_ct]
        sample_inputs.append(SampleInput(input_tensor, args=(bins_tensor,),
                                         kwargs=dict(weight=weight_tensor, density=density)))

    return sample_inputs

def sample_inputs_histc(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))

    sample_inputs = []
    for size, min, max in product(sizes, [0, -10], [0, 10]):
        # construct sample input omitting bins arg
        sample_inputs.append(SampleInput(make_arg(size),
                                         kwargs=dict(min=min, max=max)))

        # construct sample inputs with a few different bins values
        for bins in [1, 3, 10]:
            sample_inputs.append(SampleInput(make_arg(size),
                                             kwargs=dict(bins=bins, min=min, max=max)))

    return sample_inputs

def sample_inputs_bincount(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    sample_inputs = []

    for size, weighted in product((S, M), [False, True]):
        input_tensor = torch.randint(0, size, (size,), dtype=dtype, device=device)
        weight_tensor = make_arg((size,)) if weighted else None

        max_val = int(input_tensor.max().item())

        for minlength in [0, max_val // 2, max_val, 2 * max_val]:
            sample_inputs.append(SampleInput(input_tensor,
                                             kwargs=dict(weights=weight_tensor, minlength=minlength)))

    return sample_inputs

def sample_inputs_bucketize(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))

    sample_inputs = []

    for size, out_int32, right in product(sizes, [False, True], [False, True]):
        input_tensor = make_arg(size)
        boundaries = make_arg((S,)).msort()

        sample_inputs.append(SampleInput(input_tensor, args=(boundaries, ),
                                         kwargs=dict(out_int32=out_int32, right=right)))

    return sample_inputs

def sample_inputs_searchsorted(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    sizes = ((0,), (M,), (0, 0), (M, M), (0, 0, 0), (M, M, M))
    inputs = []
    for size, noncontiguous, out_int32, right in product(sizes, [False, True], [False, True], [False, True]):
        unsorted_tensor = make_arg(size, noncontiguous=noncontiguous)
        input_tensor = make_arg(size, noncontiguous=noncontiguous)
        if np.product(size) == 0:
            boundary_tensor = unsorted_tensor
            sorter = make_tensor(size, dtype=torch.int64, device=device, noncontiguous=noncontiguous)
        else:
            boundary_tensor, sorter = torch.sort(unsorted_tensor)
        side = "right" if right else "left"

        inputs.append(SampleInput(boundary_tensor, args=(input_tensor,), kwargs=dict(out_int32=out_int32, right=right)))
        inputs.append(SampleInput(boundary_tensor, args=(input_tensor,), kwargs=dict(out_int32=out_int32, side=side)))

        inputs.append(
            SampleInput(unsorted_tensor, args=(input_tensor,), kwargs=dict(out_int32=out_int32, right=right, sorter=sorter)))
        inputs.append(
            SampleInput(unsorted_tensor, args=(input_tensor,), kwargs=dict(out_int32=out_int32, side=side, sorter=sorter)))
    return inputs

def sample_inputs_gradient(op_info, device, dtype, requires_grad, **kwargs):
    sample_inputs = []
    test_cases_float = (
        ((S,), None, None, 1),
        ((S,), 2., None, 1),
        ((S, S), None, None, 2),
        ((S, S), [2.0, 2.1], None, 1),
        ((S, S), [2.0, 2.1], (0, 1), 1),
        ((4, 4, 4), [2., 1.], (0, 1), 2),
    )
    for size, spacing, dim, edge_order in test_cases_float:
        t = make_tensor(size, device, dtype, low=None, high=None, requires_grad=requires_grad)
        sample_inputs.append(SampleInput(t, kwargs=dict(dim=dim, spacing=spacing, edge_order=edge_order)))

    test_cases_tensor = (
        ((3, 3, 3), ((1.1, 2.0, 3.5), (4.0, 2, 6.0)), (0, -1), 1),
        ((3, 3, 3), ((1.0, 3.0, 2.0), (8.0, 6.0, 1.0)), (0, 1), 2),
    )
    for size, coordinates, dim, edge_order in test_cases_tensor:
        t = make_tensor(size, device, dtype, low=None, high=None, requires_grad=requires_grad)
        coordinates_tensor_list = []
        for coords in coordinates:
            # `coords` will always contain floating point values and Python 3.10 does not support this
            # implicit conversion to an integer using `__int__`
            # TODO: this can be simplified after https://github.com/pytorch/pytorch/issues/69316 is fixed
            a = torch.tensor(coords, device=device)
            coordinates_tensor_list.append(a.to(dtype))
        sample_inputs.append(SampleInput(t, kwargs=dict(dim=dim, spacing=coordinates_tensor_list, edge_order=edge_order)))

    return tuple(sample_inputs)

def sample_inputs_getitem(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_args = [
        ([1, 2],),
        (slice(0, 3),),
        ([slice(0, 3), 1],),
        ([[0, 2, 3], [1, 3, 3], [0, 0, 2]],),
        ([[0, 0, 3], [1, 1, 3], [0, 0, 2]],),
        ([slice(None), slice(None), [0, 3]],),
        ([slice(None), [0, 3], slice(None)],),
        ([[0, 3], slice(None), slice(None)],),
        ([[0, 3], [1, 2], slice(None)],),
        ([[0, 3], ],),
        ([[0, 3], slice(None)],),
        ([[0, 3], Ellipsis],),
        ([[0, 2, 3], [1, 3, 3], torch.LongTensor([0, 0, 2])],),
        (index_variable(2, S, device=device),),
        (mask_not_all_zeros((S,)),),
    ]

    for args in test_args:
        yield SampleInput(make_arg((S, S, S)), args=args)

def sample_inputs_index_put(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    inputs = []
    for accumulate in [False, True]:
        # Test with indices arg
        inputs.append(SampleInput(
            make_arg((S, S,)),
            args=((index_variable(2, S, device=device),), make_arg((2, S))),
            kwargs=dict(accumulate=accumulate)))

        # Test with mask arg
        mask = torch.zeros(S, dtype=torch.bool) if accumulate else mask_not_all_zeros((S,))
        inputs.append(SampleInput(
            make_arg((S, S)),
            args=((mask, ), make_arg((S,))),
            kwargs=dict(accumulate=accumulate)))

    return inputs

def sample_inputs_sort(op_info, device, dtype, requires_grad, **kwargs):
    def small_3d_unique():
        res = torch.randperm(S * S * S, dtype=torch.int64, device=device).view(S, S, S)
        res = res.to(dtype).requires_grad_(requires_grad)
        return res

    def large_1d_unique():
        res = torch.randperm(L * L * L, dtype=torch.int64, device=device)
        res = res.to(dtype).requires_grad_(requires_grad)
        return res

    samples = []
    # Test case for large tensor.
    samples.append(SampleInput(large_1d_unique()))

    # Test cases for small 3d tensors.
    # Imitates legacy tests from test/test_torch.py
    dims = range(-3, 3)
    flag = [True, False]
    for dim, descending, stable in product(dims, flag, flag):
        # default schema without stable sort
        samples.append(SampleInput(small_3d_unique(),
                                   args=(dim, descending)))
        # schema with stable sort, no CUDA support yet
        if torch.device(device).type == 'cpu':
            samples.append(
                SampleInput(small_3d_unique(),
                            kwargs=dict(dim=dim, descending=descending, stable=stable))
            )

    # Test cases for scalar tensor
    samples.append(SampleInput(torch.tensor(1, dtype=dtype, device=device, requires_grad=requires_grad)))
    samples.append(SampleInput(torch.tensor(1, dtype=dtype, device=device, requires_grad=requires_grad),
                               args=(0,)))
    samples.append(SampleInput(torch.tensor(1, dtype=dtype, device=device, requires_grad=requires_grad),
                               args=(0, True)))

    # Test cases for stable sort
    samples.append(SampleInput(small_3d_unique(),
                   kwargs=dict(stable=True)))
    samples.append(SampleInput(small_3d_unique(),
                   kwargs=dict(dim=0, stable=True)))
    samples.append(SampleInput(small_3d_unique(),
                   kwargs=dict(dim=0, descending=True, stable=True)))
    return samples

def sample_inputs_threshold(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = ((), (S,), (S, S), (S, S, S))
    samples = []
    for x_size in sizes:
        # threshold and values args must be numbers
        samples.append(SampleInput(make_arg(x_size), args=(make_arg(()).item(), make_arg(()).item())))
    return samples

def sample_inputs_argsort(*args, **kwargs):
    return [sample_input for sample_input in sample_inputs_sort(*args, **kwargs) if "stable" not in sample_input.kwargs]

def sample_inputs_unique(op_info, device, dtype, requires_grad, **kwargs):
    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))

    sample_inputs = []
    for shape, sorted, return_inverse, return_counts, dim in \
            product(sizes, [False, True], [False, True], [False, True], [None, -2, -1, 0, 1, 2]):
        # torch.unique cannot be called if the input tensor has a zero dimension which isn't the selected dim
        if 0 in shape and shape.index(0) is not dim:
            continue

        # skip invalid dim args
        if dim is not None and (dim < -len(shape) or dim >= len(shape)):
            continue

        kwargs = dict(sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)

        # construct a test case with only one distinct value
        input_t = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        sample_inputs.append(SampleInput(input_t, kwargs=kwargs.copy()))

        # construct a test case with mixed 0s and 1s
        input_t = make_tensor(shape, dtype=torch.bool, device=device, requires_grad=False)\
            .to(dtype).requires_grad_(requires_grad)
        sample_inputs.append(SampleInput(input_t, kwargs=kwargs.copy()))

        # construct a test case with many different values
        input_t = make_tensor(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        sample_inputs.append(SampleInput(input_t, kwargs=kwargs.copy()))

    return sample_inputs

def sample_inputs_unique_consecutive(*args, **kwargs):
    for sample_input in sample_inputs_unique(*args, **kwargs):
        if not sample_input.kwargs["sorted"]:
            sample_input.kwargs.pop("sorted")
            yield sample_input

def sample_inputs_max_min_binary(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    args_for_binary_op = (
        ((S, S, S), (S, S, S),),
        ((S, S, S), (S,),),
        ((S,), (S, S, S),),
        ((S, 1, S), (S, S),),
        ((S, S), (S, S),),
        ((), (),),
        ((S, S, S), (),),
        ((), (S, S, S),),
    )
    inputs = list((SampleInput(make_tensor(input_tensor, device, dtype,
                                           low=None, high=None,
                                           requires_grad=requires_grad),
                               args=(make_tensor(other_tensor, device, dtype,
                                                 low=None, high=None,
                                                 requires_grad=requires_grad),),))
                  for input_tensor, other_tensor in args_for_binary_op)
    return inputs


def sample_inputs_adaptive_avg_pool1d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as (input shape, output size)
    cases = (
        ((0, 8, 8), (5,)),
        ((3, 8, 8), 5),
        ((3, 8, 8), 1)
    )

    for input_shape, output_size in cases:
        yield SampleInput(make_arg(input_shape), args=(output_size,))

def sample_inputs_adaptive_avg_pool2d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as (input shape, output size)
    cases = (
        ((1, 8, 8, 8), (5, 7)),
        ((2, 8, 8, 8), (None, 7)),
        ((1, 8, 4, 3), (5, None)),
        ((1, 8, 4, 3), (None, None)),
        ((1, 8, 4, 3), (5)),
    )

    for input_shape, output_size in cases:
        yield SampleInput(make_arg(input_shape), args=(output_size,))


def sample_inputs_adaptive_avg_pool3d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as (input shape, output size)
    cases = (
        ((0, 8, 8, 8, 8), (5, 7, 4)),
        ((1, 8, 4, 3, 7), (None, None, None)),
        ((1, 8, 4, 3, 7), (1, 1, 1)),
        ((3, 3, 8, 8, 6), (5, 7, None)),
        ((1, 3, 8, 8, 6), (5, None, 2)),
        ((3, 3, 8, 8, 6), (None, 3, 2)),
    )

    for input_shape, output_size in cases:
        yield SampleInput(make_arg(input_shape), args=(output_size,))

def sample_inputs_adaptive_max_pool1d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as (input shape, output size)
    cases = (
        # ((0, 8, 8), (5,)),
        # 0 batch size doesn't work,  cannot reshape tensor of 0 elements into shape [0, 8, -1]
        ((3, 4, 4), 3),
        ((3, 4, 4), 1)
    )

    for shapes, return_idx in product(cases, (True, False)):
        yield SampleInput(make_arg(shapes[0]), args=(shapes[1], return_idx))

def sample_inputs_adaptive_max_pool2d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as (input shape, output size)
    cases = (
        # ((0, 8, 8, 8), (5, 7)),
        # 0 batch size doesn't work,  cannot reshape tensor of 0 elements into shape [0, 8, -1]
        ((1, 4, 4, 4), (2, 3)),
        ((2, 4, 4, 4), (None, 3)),
        ((2, 4, 4, 4), (1, 1)),
        ((1, 4, 4, 3), (3, None)),
        ((1, 4, 4, 3), (None, None)),
        ((1, 4, 4, 3), (3)),
    )

    for shapes, return_idx in product(cases, (True, False)):
        yield SampleInput(make_arg(shapes[0]), args=(shapes[1], return_idx))


def sample_inputs_adaptive_max_pool3d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as (input shape, output size)
    cases = (
        # ((0, 8, 8, 8, 8), (5, 7, 4)),
        # 0 batch size doesn't work,  cannot reshape tensor of 0 elements into shape [0, 8, -1]
        ((1, 4, 4, 3, 5), (None, None, None)),
        ((1, 4, 4, 3, 5), (1, 1, 1)),
        ((3, 3, 4, 4, 6), (2, 3, None)),
        ((1, 3, 4, 4, 6), (3, None, 2)),
        ((3, 3, 4, 4, 6), (None, 3, 2)),
    )

    for shapes, return_idx in product(cases, (True, False)):
        yield SampleInput(make_arg(shapes[0]), args=(shapes[1], return_idx))

class _TestParamsMaxPoolBase(object):

    def __init__(self):
        self.kwargs = {
            'kernel_size': [3],
            'stride': [2, None],
            'ceil_mode': [True, False],
            'padding': [0, 1],
            'dilation': [1],
            'return_indices': [True, False]
        }

        self.shapes = [
            [1, 2, None],  # batch
            [2],  # channels
            [3, 6]  # signal
        ]

    def _gen_shape(self):
        for shape in product(*self.shapes):
            # shape[0] is None indicates missing batch dimension
            if shape[0] is None:
                shape = shape[1:]

            yield shape, torch.contiguous_format
            # only 2d (N, C, H, W) rank 4 tensors support channels_last memory format
            if len(self.shapes) == 4 and len(shape) == 4:
                yield shape, torch.channels_last

    def _gen_kwargs(self):
        keys = self.kwargs.keys()
        for values in product(*self.kwargs.values()):
            yield dict(zip(keys, values))

    def gen_input_params(self):
        yield from product(self._gen_shape(), self._gen_kwargs())

class _TestParamsMaxPool1d(_TestParamsMaxPoolBase):

    def __init__(self):
        super().__init__()
        self.kwargs['kernel_size'] += [(3,)]
        self.kwargs['stride'] += [(2,)]
        self.kwargs['padding'] += [(1,)]
        self.kwargs['dilation'] += [(1,)]

class _TestParamsMaxPool2d(_TestParamsMaxPoolBase):

    def __init__(self):
        super().__init__()
        self.kwargs['kernel_size'] += [(3, 2)]
        self.kwargs['stride'] += [(2, 1)]
        self.kwargs['padding'] += [(1, 1)]
        self.kwargs['dilation'] += [(1, 2)]

        self.shapes.append([6])

class _TestParamsMaxPool3d(_TestParamsMaxPoolBase):

    def __init__(self):
        super().__init__()
        self.kwargs['kernel_size'] += [(3, 2, 3)]
        self.kwargs['stride'] += [(2, 1, 2)]
        self.kwargs['dilation'] += [(1, 2, 1)]

        self.shapes.append([6])
        self.shapes.append([5])

def sample_inputs_max_pool(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    params_generator_type_dict = {
        'nn.functional.max_pool1d': _TestParamsMaxPool1d,
        'nn.functional.max_pool2d': _TestParamsMaxPool2d,
        'nn.functional.max_pool3d': _TestParamsMaxPool3d,
    }

    params_generator = params_generator_type_dict[op_info.name]()
    for (shape, memory_format), kwargs in params_generator.gen_input_params():
        arg = make_arg(shape).to(memory_format=memory_format).requires_grad_(requires_grad)
        yield SampleInput(arg, kwargs=kwargs)

def sample_inputs_normalize(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, low=-1, high=1, device=device, dtype=dtype, requires_grad=requires_grad)

    cases: Tuple[Tuple[int], dict] = (  # type: ignore[assignment]
                                     ((2, 1, 4, 5), {'p': 1., 'dim': 2}),
                                     ((2, 3, 4, 5), {'p': 2., 'dim': 1}),
                                     ((1, 2, 4, 5), {'p': 0.5, 'dim': 0}),
                                     ((1, 3, 4, 5), {'p': -1., 'dim': 1}),
                                     ((1, 3, 4, 5), {'p': 0., 'dim': -1}),
                                     ((), {'p': 1.2, 'dim': 0}),
                                     ((2, 3, 4, 5), {}),
                                     ((2, 3, 4, 5), {'eps': 1e-4}))

    for input_shape, kwargs in cases:
        yield SampleInput(make_arg(input_shape), kwargs=kwargs)

def sample_inputs_conv_transpose1d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as shapes for input, weight, bias
    # and a dict of values of (stride, padding, output_padding, groups, dilation)
    cases: Tuple[Tuple[int], Tuple[int], Tuple[int], dict] = (  # type: ignore[assignment]
        ((1, 3, 4), (3, 3, 3), (3,),
         {'stride': (2,), 'padding': 2, 'output_padding': (1,), 'groups': 1}),
        ((2, 2, 4), (2, 2, 4), (4,),
         {'stride': (3,), 'padding': (1,), 'output_padding': (2,), 'groups': 2, 'dilation': (4,)}),
        ((1, 1, 4), (1, 1, 4), (1,),
         {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1, 'dilation': (2,)}),
        ((1, 1, 4), (1, 2, 3), None,
         {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1}),
        ((1, 4, 5), (4, 8, 3), None,
         {})
    )

    for input_shape, weight, bias, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(
            make_arg(weight),
            make_arg(bias) if bias is not None else bias
        ), kwargs=kwargs)


def sample_inputs_conv_transpose2d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as shapes for input, weight, bias
    # and a dict of values of (stride, padding, output_padding, groups, dilation)
    cases: Tuple[Tuple[int], Tuple[int], Tuple[int], dict] = (  # type: ignore[assignment]
        ((1, 3, 4, 4), (3, 3, 3, 3), (3,),
         {'stride': (2, 2), 'padding': 2, 'output_padding': (1, 1), 'groups': 1}),
        ((2, 2, 4, 4), (2, 2, 4, 5), (4,),
         {'stride': (3, 2), 'padding': (1, 2), 'output_padding': (2, 3), 'groups': 2, 'dilation': (4, 4)}),
        ((1, 1, 4, 5), (1, 1, 4, 3), (1,),
         {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1, 'dilation': (2, 3)}),
        ((1, 1, 4, 3), (1, 2, 3, 4), None,
         {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1}),
        ((1, 4, 5, 5), (4, 8, 3, 3), None,
         {})
    )

    for input_shape, weight, bias, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(
            make_arg(weight),
            make_arg(bias) if bias is not None else bias
        ), kwargs=kwargs)

def sample_inputs_conv_transpose3d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as shapes for input, weight, bias
    # and a dict of values of (stride, padding, output_padding, groups, dilation)
    cases: Tuple[Tuple[int], Tuple[int], Tuple[int], dict] = (  # type: ignore[assignment]
        ((1, 3, 4, 4, 4), (3, 3, 3, 3, 3), (3,),
         {'stride': (2, 2, 2), 'padding': 2, 'output_padding': (1, 1, 1), 'groups': 1}),
        ((2, 2, 4, 4, 4), (2, 2, 4, 5, 6), (4,),
         {'stride': (3, 2, 1), 'padding': (1, 2, 3), 'output_padding': (2, 3, 1), 'groups': 2, 'dilation': (4, 4, 4)}),
        ((1, 1, 4, 5, 2), (1, 1, 4, 3, 1), (1,),
         {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1, 'dilation': (2, 3, 2)}),
        ((1, 1, 4, 3, 4), (1, 2, 3, 4, 5), None,
         {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1}),
        ((1, 4, 5, 5, 5), (4, 8, 3, 3, 3), None,
         {})
    )

    for input_shape, weight, bias, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(
            make_arg(weight),
            make_arg(bias) if bias is not None else bias
        ), kwargs=kwargs)


def sample_inputs_conv1d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as shapes for input, weight, bias,
    # and a dict of values of (stride, padding, dilation, groups)
    cases: Tuple = (
        ((1, 3, 4), (3, 3, 3), (3,), {'stride': (2,), 'padding': 2, 'groups': 1}),
        ((2, 4, 8), (2, 2, 3), (2,), {'stride': 3, 'padding': 1, 'groups': 2, 'dilation': 2}),
        ((1, 4, 5), (1, 4, 3), None, {'stride': (2,), 'padding': 'valid'}),
        ((2, 2, 4), (2, 1, 4), (2,), {'stride': (1,), 'padding': 'same', 'groups': 2, 'dilation': (2,)}),
        # With defaults
        ((1, 4, 5), (3, 4, 3), None, {}),
    )

    # TODO: (@krshrimali), add error_inputs_func once https://github.com/pytorch/pytorch/pull/67354 is merged
    # Should replace test_conv_modules_raise_error_on_incorrect_input_size and test_conv_shapecheck
    # in test/test_nn.py

    for input_shape, weight, bias, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(
            make_arg(weight),
            make_arg(bias) if bias is not None else bias
        ), kwargs=kwargs)


def sample_inputs_conv2d(op_info, device, dtype, requires_grad, jit_fail_sample=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as shapes for input, weight, bias
    # and a dict of values of (stride, padding, groups, dilation)
    cases: Tuple = (
        ((1, 3, 4, 4), (3, 3, 3, 3), (3,),
            {'stride': (2, 2), 'padding': 2, 'groups': 1}),
        ((2, 4, 8, 8), (2, 2, 3, 3), (2,),
            {'stride': (3, 2), 'padding': (2, 1), 'groups': 2, 'dilation': (4, 4)}),
        ((1, 4, 5, 5), (1, 4, 2, 3), (1,),
            {'stride': 2, 'padding': 1, 'groups': 1, 'dilation': (2, 3)}),
        ((1, 4, 5, 5), (1, 4, 2, 3), (1,),
            {'stride': 2, 'padding': 1, 'groups': 1, 'dilation': (2, 3)}),
        ((1, 2, 4, 3), (4, 2, 3, 4), None,
            {'stride': 2, 'padding': 1, 'groups': 1}),
        ((1, 4, 5, 5), (1, 4, 2, 3), (1,),
            {'stride': 2, 'padding': "valid"}),
        ((1, 4, 5, 5), (1, 4, 2, 3), (1,),
            {'stride': 1, 'padding': "same", 'dilation': 3}),
        # Below are the group related samples from common_nn.py
        ((2, 4, 6, 6), (4, 1, 3, 3), (4,), {'groups': 4}),
        ((2, 4, 6, 6), (8, 1, 3, 3), (8,), {'groups': 4}),
        ((2, 4, 6, 6), (8, 1, 3, 3), None, {'groups': 4}),
        ((2, 4, 6, 6), (4, 1, 3, 3), (4,), {'groups': 4, 'stride': (3, 2)}),
        ((2, 4, 6, 6), (4, 1, 3, 3), (4,), {'groups': 4, 'padding': (1, 1)}),
        ((2, 4, 5, 5), (4, 1, 2, 2), (4,), {'groups': 4, 'dilation': (2, 2)}),
        ((2, 4, 6, 5), (6, 2, 3, 2), (6,), {'groups': 2}),
        # With defaults
        ((1, 4, 5, 5), (3, 4, 3, 3), None, {}),
    )

    for input_shape, weight, bias, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(
            make_arg(weight),
            make_arg(bias) if bias is not None else bias
        ), kwargs=kwargs)


def sample_inputs_group_norm(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as input shape, num groups, and eps
    cases: Tuple[Tuple[int], int, float] = (  # type: ignore[assignment]
        ((1, 6, 3), 2, 0.5),
        ((2, 6, 3), 2, -0.5),
        ((1, 2), 1, None),
        ((0, 2), 1, None),
    )

    for input_shape, num_groups, eps in cases:
        # Shape of weight and bias should be the same as num_channels
        weight = make_arg(input_shape[1])
        bias = make_arg(input_shape[1])
        kwargs = {'weight': weight, 'bias': bias} if eps is None else {'weight': weight, 'bias': bias, 'eps': eps}
        yield SampleInput(
            make_arg(input_shape),
            args=(num_groups,),
            kwargs=kwargs
        )
    # Without any optional args
    yield SampleInput(make_arg((1, 2)), args=(1,))


def sample_inputs_instance_norm(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_arg_without_requires_grad = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # Ordered as: input shape, kwargs for momentum, eps
    cases: Tuple[Tuple[int], dict] = (  # type: ignore[assignment]
        ((S, S, S), {'momentum': 0.5, 'eps': 0.6}),
        ((S, S, S), {'momentum': 0.5, 'eps': 0.6, 'use_input_stats': True}),
        ((3, 2, 4), {'momentum': -1.2}),
        ((3, 2, 4), {'momentum': 0.0}),
        ((3, 2, 3, 4), {'momentum': -1.0, 'eps': 0.5}),
        ((3, 2, 3, 4), {'momentum': -1.0, 'eps': 0.5}),
    )

    for input_shape, kwargs in cases:
        # args: running mean, running var, weight and bias should necessarily be of shape: (channels,)
        channels = input_shape[1]
        weight = make_arg(channels)
        bias = make_arg(channels)
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)
        new_kwargs = {
            'running_mean': running_mean,
            'running_var': running_var,
            'weight': weight,
            'bias': bias,
            **kwargs
        }

        yield SampleInput(
            make_arg(input_shape),
            args=(),
            kwargs=new_kwargs
        )

    # Checking for permutations of weights and biases as `None`
    # instance_norm assumes that if there's a bias, there's a weight
    weights = [channels, None]
    biases = [None, None]

    for weight_channels, bias_channels in zip(weights, biases):
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)
        yield SampleInput(
            make_arg(input_shape),
            args=(),
            kwargs={
                'running_mean': running_mean,
                'running_var': running_var,
                'weight': make_arg(weight_channels) if weight_channels is not None else None,
                'bias': make_arg(bias_channels) if bias_channels is not None else None
            }
        )

    # Test case for no optional kwargs
    yield SampleInput(make_arg((1, 2, 3)), kwargs={})


def sample_inputs_layer_norm(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as input shape, normalized_shape and a kwarg dict for eps
    cases: Tuple[Tuple[int], Tuple[int], dict] = (  # type: ignore[assignment]
        ((1, 2, 3), (1, 2, 3), {'eps': 0.5}),
        ((2, 2, 3), (2, 3), {'eps': -0.5}),
        ((1,), (1,), {}),
        ((1, 2), (2,), {}),
        ((0, 1), (1,), {}),
    )

    for input_shape, normalized_shape, kwargs in cases:
        # Shape of weight and bias should be the same as normalized_shape
        weight = make_arg(normalized_shape)
        bias = make_arg(normalized_shape)
        yield SampleInput(
            make_arg(input_shape),
            args=(normalized_shape, weight, bias),
            kwargs=kwargs
        )
    # Without any optional args
    yield SampleInput(make_arg((1, 2)), args=((2,),))

    # TODO: @krshrimali, once to_numpy method in SampleInput class is modified to take None inputs,
    # enable these inputs; see https://github.com/pytorch/pytorch/pull/63276#discussion_r691950400

    # With weight and a `None` bias
    # yield SampleInput(make_arg((1, 2)), args=((2,), make_arg((2,)), None))

    # With `None` weight and bias (tests failing for this, see the link above)
    # yield SampleInput(make_arg((1, 2)), args=((2,), None, make_arg((2,))))


def sample_inputs_local_response_norm(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Ordered as input shape, size and a kwarg dict for alpha, beta, and k
    cases: Tuple[Tuple[int], Tuple[int], dict] = (  # type: ignore[assignment]
        ((1, 6, 3), 2, {'alpha': 3e-05, 'beta': 0.5, 'k': 1.25}),
        ((1, 6, 3), 2, {'beta': 0.5, 'k': 1.25}),
        ((1, 6, 3), 2, {'alpha': 3e-05, 'k': 1.25}),
        ((1, 6, 3), 2, {'alpha': 3e-05, 'beta': 0.5}),
        ((1, 6, 3), 2, {'alpha': 3e-05}),
        ((1, 6, 3), 2, {'beta': 0.5}),
        ((1, 6, 3), 2, {'k': 1.25}),
        ((1, 6, 3), 2, {}),
        ((2, 6, 3), 2, {'alpha': 3e-05, 'beta': 0.5, 'k': 1.25}),
        ((1, 1, 2), 1, {'alpha': 3e-05, 'beta': 0.5, 'k': 1.25}),
        ((0, 1, 2), 1, {'alpha': 3e-05, 'beta': 0.5, 'k': 1.25}),
    )

    for input_shape, size, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(size,), kwargs=kwargs)


def sample_inputs_hardswish(self, device, dtype, requires_grad, **kwargs):
    N = 5
    # make sure we are testing -3 -> 3 range. default is -10 -> 10 so maybe unnecessary ?
    tensors = [SampleInput(make_tensor((N * 2, N * 2), device=device, dtype=dtype,
               requires_grad=requires_grad, low=-5, high=5)) for _ in range(1, N)]
    return tensors

def sample_inputs_linear(self, device, dtype, requires_grad, **kwargs):
    features_options = [[3, 4], [8, 8]]
    batch_options: List[List[int]] = [
        [],  # no batch
        [0],
        [8],
        [2, 3],
    ]
    create_tensor = partial(make_tensor, device=device, dtype=dtype,
                            requires_grad=requires_grad, low=-2, high=2)

    sample_inputs = []
    for has_bias, (in_feat, out_feat), batch_shape in \
            itertools.product([True, False], features_options, batch_options):
        input_tensor = create_tensor(batch_shape + [in_feat])
        weight = create_tensor([out_feat, in_feat])
        if not has_bias:
            sample_inputs.append(SampleInput(input_tensor, args=(weight,)))
            continue

        bias = create_tensor([out_feat])
        sample_inputs.append(SampleInput(input_tensor, args=(weight, bias)))
    return sample_inputs

def sample_inputs_bilinear(self, device, dtype, requires_grad, **kwargs):
    features_options = [[3, 4, 5], [8, 8, 8]]
    batch_options: List[List[int]] = [
        [],  # no batch
        [0],
        [8],
        [2, 3],
    ]
    create_tensor = partial(make_tensor, device=device, dtype=dtype,
                            requires_grad=requires_grad, low=-2, high=2)

    sample_inputs = []
    for has_bias, (in_feat1, in_feat2, out_feat), batch_shape in \
            itertools.product([True, False], features_options, batch_options):
        input_tensor1 = create_tensor(batch_shape + [in_feat1])
        input_tensor2 = create_tensor(batch_shape + [in_feat2])
        weight = create_tensor([out_feat, in_feat1, in_feat2])
        if not has_bias:
            sample_inputs.append(SampleInput(input_tensor1, args=(input_tensor2, weight,)))
            continue
        bias = create_tensor([out_feat])
        sample_inputs.append(SampleInput(input_tensor1, args=(input_tensor2, weight, bias)))

    return sample_inputs

def sample_inputs_glu(self, device, dtype, requires_grad, **kwargs):
    features_options = [[2], [2, 4], [8, 8], [3, 6, 8], [1, 4, 6, 7]]
    batch_options: List[List[int]] = [
        [],  # no batch
        [0],
        [8],
        [2, 3],
    ]
    create_tensor = partial(make_tensor, device=device, dtype=dtype,
                            requires_grad=requires_grad, low=-2, high=2)

    sample_inputs = []
    for features, batch_shape in itertools.product(features_options, batch_options):
        ndim = len(features) + len(batch_shape)
        for dim in range(ndim):
            input_tensor = create_tensor(batch_shape + features)
            dim_size = input_tensor.size(dim)
            if dim_size > 0 and dim_size % 2 == 0:
                sample_inputs.append(SampleInput(input_tensor, args=(dim,)))

    return sample_inputs

def sample_inputs_interpolate(mode, self, device, dtype, requires_grad, **kwargs):
    N, C = 2, 3
    D = 4
    S = 3
    L = 5

    align_corners_options: Tuple[Any, ...] = (None,)
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        align_corners_options = (True, False, None)
    ranks_for_mode = {
        'nearest': [1, 2, 3],
        'linear': [1],
        'bilinear': [2],
        'bicubic': [2],
        'trilinear': [3],
        'area': [1, 2, 3]
    }

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = partial(make_tensor, device=device, dtype=dtype,
                       requires_grad=requires_grad, low=-1, high=1)

    sample_inputs = []
    for align_corners in align_corners_options:
        for rank in ranks_for_mode[mode]:
            sample_inputs.extend([
                SampleInput(make_arg(shape(D, rank)),
                            args=(shape(S, rank, False), None, mode, align_corners)),
                SampleInput(make_arg(shape(D, rank)),
                            args=(shape(L, rank, False), None, mode, align_corners)),
                SampleInput(make_arg(shape(D, rank)),
                            args=(None, 1.7, mode, align_corners)),
                SampleInput(make_arg(shape(D, rank)),
                            args=(None, 0.6, mode, align_corners)),
            ])

    return sample_inputs

def sample_inputs_upsample(mode, self, device, dtype, requires_grad, **kwargs):
    N, C = 2, 3
    D = 4
    S = 3
    L = 5

    ranks_for_mode = {
        'nearest': [1, 2, 3],
        'bilinear': [2],
    }

    def shape(size, rank, with_batch_channel=True):
        if with_batch_channel:
            return tuple([N, C] + ([size] * rank))
        return tuple([size] * rank)

    make_arg = partial(make_tensor, device=device, dtype=dtype,
                       requires_grad=requires_grad, low=-1, high=1)

    sample_inputs = []
    for rank in ranks_for_mode[mode]:
        sample_inputs.extend([
            SampleInput(make_arg(shape(D, rank)),
                        kwargs=dict(size=shape(S, rank, False))),
            SampleInput(make_arg(shape(D, rank)),
                        kwargs=dict(size=shape(L, rank, False))),
            SampleInput(make_arg(shape(D, rank)),
                        kwargs=dict(scale_factor=1.7)),
            SampleInput(make_arg(shape(D, rank)),
                        kwargs=dict(scale_factor=0.6)),
        ])

    return sample_inputs

def sample_inputs_gelu(self, device, dtype, requires_grad, **kwargs):
    N = 5
    tensors = [SampleInput(make_tensor((N * 2, N * 2), device=device, dtype=dtype,
               requires_grad=requires_grad, low=-3, high=3)) for _ in range(1, N)]
    return tensors

def sample_inputs_max_min_reduction_with_dim(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    args_for_reduction_with_dim = (
        ((S, S, S), (1,),),
        ((S, S, S), (1, True, ),),
        ((), (0,),),
        ((), (0, True,),),
    )
    inputs = list((SampleInput(make_tensor(input_tensor, device, dtype,
                                           low=None, high=None,
                                           requires_grad=requires_grad),
                               args=args,))
                  for input_tensor, args in args_for_reduction_with_dim)
    return inputs

def sample_inputs_max_min_reduction_no_dim(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    inputs.append(SampleInput(make_tensor((S, S, S), device, dtype,
                                          low=None, high=None,
                                          requires_grad=requires_grad),))
    inputs.append(SampleInput(make_tensor((), device, dtype,
                                          low=None, high=None,
                                          requires_grad=requires_grad),))
    return inputs

def _generate_nan_reduction_inputs(device, dtype, requires_grad, **kwargs):
    yield from _generate_reduction_inputs(device, dtype, requires_grad)
    yield torch.tensor([2, torch.nan, -1], device=device, dtype=dtype, requires_grad=requires_grad)
    yield torch.tensor([[torch.nan, 2], [0, 1]], device=device, dtype=dtype, requires_grad=requires_grad)

def sample_inputs_nan_reduction(supports_multiple_dims):
    # Generates sample inputs for reduction ops that contain the input tensor
    # and dim and keepdim kwargs. If a reduction op needs to test additional
    # args/kwargs then create a separate sample_inputs function
    def fn(op_info, device, dtype, requires_grad):
        inputs = []

        for t in _generate_nan_reduction_inputs(device, dtype, requires_grad):
            # Add case without dim and keepdim kwargs
            inputs.append(SampleInput(t.clone().requires_grad_(requires_grad)))
            for kwargs in _generate_reduction_kwargs(t.ndim, supports_multiple_dims):
                inputs.append(SampleInput(t.clone().requires_grad_(requires_grad),
                                          kwargs=kwargs))

        return inputs

    return fn

def sample_inputs_reduction_quantile(op_info, device, dtype, requires_grad, **kwargs):
    test_quantiles = (0.5, make_tensor((2,), device, dtype, low=0, high=1, requires_grad=requires_grad))
    test_interpolations = ['linear', 'midpoint']

    inputs = []
    for quantiles in test_quantiles:
        for t in _generate_reduction_inputs(device, dtype, requires_grad):
            # Add case without dim and keepdim kwargs
            inputs.append(SampleInput(t.clone().requires_grad_(requires_grad),
                                      args=(quantiles,)))
            for kwargs in _generate_reduction_kwargs(t.ndim, supports_multiple_dims=False):
                # Interpolation kwarg for now is only supported when providing both dim and keepdim
                kwargs.setdefault('dim', 0)
                kwargs.setdefault('keepdim', False)
                for interpolation in test_interpolations:
                    kwargs['interpolation'] = interpolation
                    inputs.append(SampleInput(t.clone().requires_grad_(requires_grad),
                                              args=(quantiles,), kwargs=kwargs))

    return inputs

def sample_inputs_reduction_count_nonzero(*args, **kwargs):
    """Sample inputs for count_nonzero"""
    samples: List[SampleInput] = sample_inputs_reduction(*args, **kwargs)
    # count_nonzero does not support keepdim yet
    for sample in samples:
        sample.kwargs.pop('keepdim', None)
    return samples

def sample_inputs_leaky_relu(op_info, device, dtype, requires_grad, **kwargs):
    N = 10
    tensors = [SampleInput(make_tensor((N, N), device=device, dtype=dtype,
               requires_grad=requires_grad)) for _ in range(1, N)]
    return tensors

def sample_inputs_fractional_max_pool2d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Order: input_shape, kernel_size
    cases = (((1, 3, 9, 9), 3),
             ((1, 3, 9, 9), (4, 4)),
             ((1, 3, 9, 9), (6, 6)),
             ((2, 3, 9, 9), (3, 3)),
             ((1, 1, 4, 4), (2, 2)),
             ((1, 2, 6, 6), (4, 4)))

    samples = []

    for input_shape, kernel_size in cases:
        for return_indices in [False, True]:
            # test case passing a single output size
            samples.append(SampleInput(
                make_arg(input_shape),
                args=(kernel_size,),
                kwargs=dict(output_size=(2), return_indices=return_indices)
            ))

            # test case passing a tuple output size
            samples.append(SampleInput(
                make_arg(input_shape),
                args=(kernel_size,),
                kwargs=dict(output_size=(2, 3), return_indices=return_indices)
            ))

            # test case passing an output ratio
            samples.append(SampleInput(
                make_arg(input_shape),
                args=(kernel_size,),
                kwargs=dict(output_ratio=(0.5, 0.5), return_indices=return_indices)
            ))

    return samples

def sample_inputs_fractional_max_pool3d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Order: input_shape, kernel_size
    cases = (((2, 3, 5, 5, 5), (2, 2, 2)),
             ((1, 2, 6, 5, 4), 2),
             ((1, 2, 5, 6, 5), (2, 3, 2)),
             ((1, 2, 6, 6, 6), (2, 3, 2)),
             ((1, 1, 7, 6, 7), (2, 3, 4)),
             ((1, 1, 4, 5, 4), (2, 2, 1)),
             ((1, 1, 8, 7, 6), (4, 3, 2)),
             ((0, 1, 4, 5, 4), (2, 2, 1)))

    samples = []

    for input_shape, kernel_size in cases:
        for return_indices in [False, True]:
            # test case passing a single output size
            samples.append(SampleInput(
                make_arg(input_shape),
                args=(kernel_size,),
                kwargs=dict(output_size=(2), return_indices=return_indices)
            ))

            # test case passing a tuple output size
            samples.append(SampleInput(
                make_arg(input_shape),
                args=(kernel_size,),
                kwargs=dict(output_size=(2, 3, 2), return_indices=return_indices)
            ))

            # test case passing an output ratio
            samples.append(SampleInput(
                make_arg(input_shape),
                args=(kernel_size,),
                kwargs=dict(output_ratio=(0.5, 0.5, 0.5), return_indices=return_indices)
            ))

    return samples

def sample_inputs_avgpool2d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Order: input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    cases = (((1, 3, 9, 9), 3, 1, 1, True, False, 2),
             ((1, 3, 9, 9), (4, 4), (2, 3), 1, True, False, 2),
             ((1, 3, 9, 9), (6, 6), (3, 3), (2, 3), True, True, 2),
             ((2, 3, 9, 9), (3, 3), (1, 1), (1, ), True, False, 2),
             ((1, 1, 4, 4), (2, 2), (), (0, ), False, True, -2),
             ((1, 2, 6, 6), (4, 4), (2, 2), (2, ), True, True, None))

    for input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override in cases:
        yield SampleInput(make_arg(input_shape),
                          args=(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override))
    # Case with just input_shape and kernel_size
    yield SampleInput(make_arg((1, 3, 9, 9)), args=((3, 3)))

def sample_inputs_avgpool1d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Order: input_shape, kernel_size, kwargs
    cases: List[Tuple[Tuple[int, ...], Union[int, Tuple[int, ...]], Dict]] = [
        ((2, 3, 9), (3,), dict()),
        ((1, 3, 9), 3, dict(stride=1, padding=1, ceil_mode=True, count_include_pad=False)),
        ((1, 3, 9), (6,), dict(stride=(3,), padding=(2,), ceil_mode=True, count_include_pad=True)),
        ((2, 3, 9), (3,), dict(stride=(1,), padding=(1,), ceil_mode=False, count_include_pad=True)),
        ((0, 3, 9), (6,), dict(stride=(3,), padding=(2,), ceil_mode=False, count_include_pad=True)),
        ((1, 2, 9), (7,), dict(stride=(3,), padding=(2,), ceil_mode=False)),
        ((1, 2, 9), (7,), dict(stride=(3,), padding=(3,), ceil_mode=True)),
        ((1, 2, 9), (7,), dict(stride=(3,), ceil_mode=False)),
        ((1, 2, 9), (7,), dict(stride=(3,), ceil_mode=True)),
    ]

    for input_shape, kernel_size, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(kernel_size,), kwargs=kwargs)

def sample_inputs_avgpool3d(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # Order: input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
    cases: List[Tuple[Tuple[int, ...], Union[int, Tuple[int, ...]], Dict]] = [
        ((2, 3, 3, 4, 4), (2, 2, 2), dict()),
        ((1, 2, 4, 4, 4), 2, dict(stride=1, padding=1, ceil_mode=True,
                                  count_include_pad=False, divisor_override=2)),
        ((1, 2, 5, 5, 5), (2, 3, 4), dict(stride=(1, 2, 2), padding=(0, 1, 2), ceil_mode=True,
                                          count_include_pad=True, divisor_override=2)),
        ((1, 2, 5, 5, 5), (2, 3, 4), dict(stride=(1, 2, 2), padding=(0, 1, 2), ceil_mode=False)),
        ((1, 1, 7, 5, 7), (6, 3, 4), dict(stride=(2, 3, 2), padding=(3, 1, 0), ceil_mode=False,
                                          count_include_pad=False, divisor_override=2)),
        ((1, 1, 4, 5, 4), (2, 2, 3), dict(stride=(2, 2, 1), padding=0, ceil_mode=False,
                                          count_include_pad=True, divisor_override=-2)),
        ((1, 1, 6, 5, 6), (4, 5, 6), dict(stride=(2, 3, 2), padding=2, ceil_mode=True,
                                          count_include_pad=True, divisor_override=None)),
        ((0, 1, 4, 5, 4), (2, 3, 1), dict(stride=(2, 1, 2), padding=0, ceil_mode=False,
                                          count_include_pad=True, divisor_override=None)),
    ]

    for input_shape, kernel_size, kwargs in cases:
        yield SampleInput(make_arg(input_shape), args=(kernel_size,), kwargs=kwargs)

def sample_inputs_topk(op_info, device, dtype, requires_grad, **kwargs):
    def get_tensor_input(size):
        return make_tensor(size, device, dtype, requires_grad=requires_grad)

    inputs = []
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3,)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, 1)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, -2)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, 1, True)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, -2, True)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, 1, True, True)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, -2, True, True)))

    inputs.append(SampleInput(get_tensor_input(()), args=(1,)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, 0)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, -1)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, 0, True)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, -1, True)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, 0, True, True)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, -1, True, True)))

    return inputs

def sample_inputs_outer(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    arg_a = make_tensor((S,), device, dtype, requires_grad=requires_grad)
    arg_b = make_tensor((M,), device, dtype, requires_grad=requires_grad)
    inputs.append(SampleInput(arg_a, args=(arg_b,)))
    return inputs


def sample_inputs_igamma_igammac(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, low=1e-3)
    cases = (((S, S), (S, S), False),
             ((S, S), (S, ), False),
             ((S, ), (S, S), True),
             ((), (), False))

    for shape, other_shape, broadcasts_input in cases:
        yield SampleInput(make_arg(shape, requires_grad=requires_grad),
                          args=(make_arg(other_shape, requires_grad=False),),
                          broadcasts_input=broadcasts_input)


def sample_inputs_dist(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    sizes = ((S, S, S), (S,), (S, 1, S), (), (S, S))
    ps = (2, 4)

    for size_x, size_y, p in product(sizes, sizes, ps):
        yield SampleInput(make_arg(size_x), args=(make_arg(size_y), p))

# Missing to test the nondeterminism of the operation
# https://github.com/pytorch/pytorch/issues/53352
def sample_inputs_index(op_info, device, dtype, requires_grad, **kwargs):
    # target.index_select(dim, idx)
    select = op_info.name == "index_select"
    # target.index_add(dim, idx, source, *, alpha=1)
    add = op_info.name == "index_add"
    # target.index_copy(dim, idx, source)
    copy = op_info.name == "index_copy"
    # target.index_fill(dim, idx, value)
    fill = op_info.name == "index_fill"

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_permutation = partial(torch.randperm, device=device, dtype=torch.int64)

    def make_idx(n):
        return make_tensor((n,), device=device, dtype=torch.int64, low=0, high=n)

    shapes = [(), (1,), (S, S)]
    # extra parameter for add
    alphas = (-1, 0, 2) if add else (None,)

    for shape, alpha in product(shapes, alphas):
        t = make_arg(shape)
        args = []

        # dim. We handle the scalar case
        dim = 1 if t.ndim == 2 else 0
        args.append(dim)

        # idx They need to be different for copy and add to be deterministic
        make_idx_fn = make_permutation if copy or add else make_idx
        idx = make_idx_fn(t.shape[dim] if t.ndim != 0 else 1)
        args.append(idx)

        # source
        if copy or add:
            args.append(make_arg(shape))
        elif fill:
            # A weird number to catch errors
            args.append(make_arg((1,)).item())

        args = tuple(args)
        kwargs = {} if alpha is None else {"alpha": alpha}

        yield SampleInput(t, args=args, kwargs=kwargs)

def sample_inputs_mode(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    args = (
        ((S, S, S), (),),
        ((S, S, S), (1, ),),
        ((S, S, S), (1, True, ),),
        ((), (),),
        ((), (0,),),
        ((), (0, True,),),
    )
    inputs = list((SampleInput(make_tensor(input_tensor, device, dtype,
                                           low=None, high=None,
                                           requires_grad=requires_grad),
                               args=args,))
                  for input_tensor, args in args)
    return inputs

# Missing to test the nondeterminism of the operation
# https://github.com/pytorch/pytorch/issues/53352
def sample_inputs_put(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    make_idx = partial(make_tensor, low=0, dtype=torch.int64, device=device, requires_grad=False)

    S = 3

    # Generic inputs
    idx = torch.randperm(S * S, device=device, dtype=torch.int64)[:S]
    idx_list = [idx, -idx - 1]
    for idx, acc in product(idx_list, (True, False)):
        yield SampleInput(input=make_arg((S, S)),
                          args=(idx.clone(),
                                make_arg((S,)),
                                acc))

    # Scalar cases
    scalar_sizes = [(), (1,)]
    tgt_gen = (make_arg(size) for size in scalar_sizes)
    idx_gen = (make_idx(size, high=1) for size in scalar_sizes)
    src_gen = (make_arg(size) for size in scalar_sizes)
    for tgt, idx, src, acc in product(tgt_gen, idx_gen, src_gen, (True, False)):
        yield SampleInput(input=tgt.clone().requires_grad_(requires_grad),
                          args=(idx.clone(),
                                src.clone().requires_grad_(requires_grad),
                                acc))

    # Empty cases
    tgt_sizes = [(0,), (), (1,), (3, 2)]
    tgt_gen = (make_arg(size) for size in tgt_sizes)
    idx = make_idx((0,), high=1)
    src = make_arg((0,))
    for tgt, acc in product(tgt, (True, False)):
        yield SampleInput(input=tgt.clone().requires_grad_(requires_grad),
                          args=(idx.clone(),
                                src.clone().requires_grad_(requires_grad),
                                acc))

def sample_inputs_take(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    make_idx = partial(make_tensor, low=0, dtype=torch.int64, device=device, requires_grad=False)

    S = 3

    # Generic inputs: take S elements out of S * S
    index = make_idx((S,), high=(S * S))
    for idx in (index, -index - 1):
        yield SampleInput(input=make_arg((S, S)), args=(idx,))

    # Scalar cases
    scalar_sizes = [(), (1,)]
    src_gen = (make_arg(size) for size in scalar_sizes)
    idx_gen = (make_idx(size, high=1) for size in scalar_sizes)
    for src, idx in product(src_gen, idx_gen):
        yield SampleInput(input=src.clone().requires_grad_(requires_grad),
                          args=(idx.clone(),))

    # Empty cases
    src_sizes = [(0,), (), (1,), (3, 2)]
    src_gen = (make_arg(size) for size in src_sizes)

    idx = make_idx((0,), high=1)
    for src in src_gen:
        yield SampleInput(input=src.clone().requires_grad_(requires_grad),
                          args=(idx.clone(),))

def sample_movedim_moveaxis(op_info, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((4, 3, 2, 1), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=([0, 1, 2, 3], [3, 2, 1, 0])),
        SampleInput(
            make_tensor((4, 3, 2, 1), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=([0, -1, -2, -3], [-3, -2, -1, -0]))
    )


def sample_repeat_tile(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    rep_dims = ((), (0, ), (1, ), (0, 2), (1, 1), (2, 3), (2, 3, 2), (0, 2, 3), (2, 1, 1, 1),)
    shapes = ((), (0,), (2,), (3, 0), (3, 2), (3, 0, 1))

    if requires_grad:
        # Tests for variant_consistency_jit, grad, gradgrad
        # are slower. Use smaller bags of `rep_dims` and `shapes`
        # in this case.
        rep_dims = ((), (0, ), (0, 2), (1, 1), (2, 3), (1, 3, 2), (3, 1, 1))  # type: ignore[assignment]
        shapes = ((), (0,), (2,), (3, 2))  # type: ignore[assignment]

    samples = []
    for rep_dim, shape in product(rep_dims, shapes):
        # `torch.repeat` errors for `len(rep_dims) < t.dim()`,
        # so we filter such combinations.
        if op_info.name == 'repeat' and len(rep_dim) < len(shape):
            continue
        samples.append(SampleInput(make_arg(shape), args=(rep_dim,),))

    return samples


def sample_inputs_narrow(op_info, device, dtype, requires_grad, **kwargs):
    shapes_and_args = (
        ((S, S, S), (1, 2, 2)),
        ((S, S, S), (-1, 2, 2)),
        ((S, S, S), (1, 0, 0)),
        ((S, S, S), (-1, 0, 0)),
    )

    for shape, args in shapes_and_args:
        tensor = make_tensor(shape, device, dtype, low=None, high=None,
                             requires_grad=requires_grad)
        yield SampleInput(tensor, args=args)

def sample_trapezoid(op_info, device, dtype, requires_grad, **kwargs):
    y_shape_x_shape_and_kwargs = [
        ((2, 3), (2, 3), {}),
        ((2, 3), (2, 3), {'dim': 1}),
        ((6,), (6,), {}),
        ((6,), None, {}),
        # When 'trapezoid' is called with an empty input, it does not produce an output with requires_grad
        # See Issue #{61619}
        # ((6,0), (6,0), {}),
        ((2, 3), (1, 3), {}),
        ((3, 3), (3, 3), {}),
        ((3, 3), (3, 3), {'dim': -2}),
        ((5,), None, {'dx': 2.0}),
        ((2, 2), None, {'dx': 3.0})
    ]
    samples = []
    for y_shape, x_shape, kwarg in y_shape_x_shape_and_kwargs:
        y_tensor = make_tensor(y_shape, device, dtype, low=None, high=None,
                               requires_grad=requires_grad)
        if x_shape is not None:
            x_tensor = make_tensor(x_shape, device, dtype, low=None, high=None,
                                   requires_grad=requires_grad)
            samples.append(SampleInput(y_tensor, args=(x_tensor,), kwargs=kwarg))
        else:
            samples.append(SampleInput(y_tensor, kwargs=kwarg))
    return samples

def sample_cumulative_trapezoid(op_info, device, dtype, requires_grad, **kwargs):

    y_shape_x_shape_and_kwargs = [
        ((2, 3), (2, 3), {}),
        ((2, 3), (2, 3), {'dim': 1}),
        ((6,), (6,), {}),
        ((6,), None, {}),
        # When 'cumulative_trapezoid' is called with an empty input, it does not produce an output with requires_grad
        # See Issue #{61619}
        # ((6,0), (6,0), {}),
        ((2, 3), (1, 3), {}),
        ((3, 3), (3, 3), {}),
        ((3, 3), (3, 3), {'dim': -2}),
        ((5,), None, {'dx': 2.0}),
        ((2, 2), None, {'dx': 3.0})
    ]
    samples = []
    for y_shape, x_shape, kwarg in y_shape_x_shape_and_kwargs:
        y_tensor = make_tensor(y_shape, device, dtype, low=None, high=None,
                               requires_grad=requires_grad)
        if x_shape is not None:
            x_tensor = make_tensor(x_shape, device, dtype, low=None, high=None,
                                   requires_grad=requires_grad)
            samples.append(SampleInput(y_tensor, args=(x_tensor,), kwargs=kwarg))
        else:
            samples.append(SampleInput(y_tensor, kwargs=kwarg))
    return samples

def sample_unsqueeze(op_info, device, dtype, requires_grad, **kwargs):
    shapes_and_axes = [
        ((3, 4, 5), 0),
        ((3, 4, 5), 1),
        ((3, 4, 5), 3),
        ((3, 4, 5), -1),
        ((3, 4, 5), -3),
        ((), 0)
    ]

    samples = []
    for shape, axis in shapes_and_axes:
        tensor = make_tensor(shape, device, dtype, low=None, high=None,
                             requires_grad=requires_grad)
        samples.append(SampleInput(tensor, args=(axis,),))

    return samples


def sample_inputs_nn_unfold(op_info, device, dtype, requires_grad, **kwargs):
    shapes = ((0, 1, 5, 5), (1, 1, 5, 5), (2, 3, 5, 5))
    kernel_sizes = (2, (2, 2), (3, 3))
    dilations = (1, 2, (1, 2))
    paddings = (0, 1, (1, 1))
    strides = (1, 2, (1, 2))

    cases = product(shapes, kernel_sizes, dilations, paddings, strides)
    for shape, kernel_size, dilation, padding, stride in cases:
        tensor = make_tensor(shape, device, dtype, requires_grad=requires_grad)
        yield SampleInput(tensor, args=(kernel_size, dilation, padding, stride))

    # With default args
    yield SampleInput(make_tensor((1, 1, 5, 5), device, dtype, requires_grad=requires_grad),
                      args=((3, 3),))


def sample_inputs_squeeze(op_info, device, dtype, requires_grad, **kwargs):
    shapes_and_args = (
        ((S, 1, S, 1), ()),
        ((1, 1, 1, 1), ()),
        ((S, 1, S, 1), (1,)),
        ((S, 1, S, 1), (-1,)),
        ((S, 1, S, 1), (2,)),
        ((S, 1, S, 1), (-2,)),
        ((), (0, )),
    )

    for shape, args in shapes_and_args:
        tensor = make_tensor(shape, device, dtype, low=None, high=None,
                             requires_grad=requires_grad)

        yield SampleInput(tensor, args=args)


def sample_inputs_nn_pad(op_info, device, dtype, requires_grad, mode, **kwargs):
    assert mode in ('constant', 'reflect', 'replicate', 'circular')
    if mode in ['reflect', 'replicate']:
        cases: tuple = (  # ignore
            ((1, 3), (1, 2)),
            ((1, 3), (0, 1)),
            ((0, 3, 3), (1, 2)),
            ((0, 3, 3), (0, 1)),
            ((1, 3, 3), (1, 2)),
            ((1, 3, 3), (0, 1)),
            ((1, 3, 3), (0, 2, 0, 1)),
            ((0, 3, 3, 3), (0, 2, 0, 1)),
            ((3, 3, 5, 5), (0, 2, 0, 1)),
            ((3, 3, 5, 5), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 4, 4), (-1, 1, -2, 1)),
        )
    elif mode == 'constant':
        cases = (
            ((1, 3), (1, 2)),
            ((1, 3), (0, 1)),
            ((1, 3), (0, 2, 0, 1)),
            ((0, 3, 3), (1, 2)),
            ((0, 3, 3), (0, 1)),
            ((0, 3, 3), (0, 2, 0, 1)),
            ((0, 3, 3), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 3), (1, 2)),
            ((1, 3, 3), (0, 1)),
            ((1, 3, 3), (0, 2, 0, 1)),
            ((1, 3, 3), (1, 1, 1, 1, 1, 1)),
            ((0, 3, 3, 3), (1, 2)),
            ((0, 3, 3, 3), (0, 1)),
            ((0, 3, 3, 3), (0, 2, 0, 1)),
            ((0, 3, 3, 3), (1, 1, 1, 1, 1, 1)),
            ((3, 3, 5, 5), (1, 2)),
            ((3, 3, 5, 5), (0, 1)),
            ((3, 3, 5, 5), (0, 2, 0, 1)),
            ((3, 3, 5, 5), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 3, 3, 3), (1, 2)),
            ((1, 3, 3, 3, 3), (0, 1)),
            ((1, 3, 3, 3, 3), (0, 2, 0, 1)),
            ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)),
            ((1, 3, 4, 4), (-1, 1, -2, 1)),
        )
    else:  # mode == 'circular'
        if dtype == torch.bool:
            # test_dtypes fails on ASAN with for the case ab
            # runtime error: load of value 190, which is not a valid value for type 'bool'
            # Reference: https://github.com/pytorch/pytorch/pull/62814#issuecomment-894156562
            # Reference Issue: https://github.com/pytorch/pytorch/issues/63034
            cases = (
                ((2, 3, 3), (1, 2)),
                ((1, 3, 3), (1, 2)),
            )
        else:
            cases = (
                ((0, 3, 3), (1, 2)),
                ((0, 3, 3), (0, 1)),
                ((1, 3, 3), (1, 2)),
                ((1, 3, 3), (0, 1)),
                ((0, 3, 3, 3), (0, 2, 0, 1)),
                ((3, 3, 5, 5), (0, 2, 0, 1)),
                ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)),
                ((1, 3, 4, 4), (-1, 1, -2, 1)),
            )

    make_inp = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    if mode == 'constant':
        # Default args
        yield SampleInput(make_inp((1, 3, 3)), args=((2, 2),))

    if mode in ['reflect', 'replicate', 'circular']:
        for shape, pad in cases:
            yield SampleInput(make_inp(shape), args=(pad, mode))
    else:  # mode == 'constant'
        for pad_value in (1., 2.):
            for shape, pad in cases:
                yield SampleInput(make_inp(shape), args=(pad, mode, pad_value))


# TODO: reconcile with torch.linalg.det and torch.linalg.slogdet
# Creates matrices with a positive nonzero determinant
def sample_inputs_logdet(op_info, device, dtype, requires_grad, **kwargs):
    def make_nonzero_det(A, *, sign=1, min_singular_value=0.1, **kwargs):
        u, s, vh = torch.linalg.svd(A, full_matrices=False)
        s.clamp_(min=min_singular_value)
        A = (u * s.unsqueeze(-2)) @ vh
        det = A.det()
        if sign is not None:
            if A.dim() == 2:
                if (det < 0) ^ (sign < 0):
                    A[0, :].neg_()
            else:
                cond = ((det < 0) ^ (sign < 0)).nonzero()
                if cond.size(0) > 0:
                    for i in range(cond.size(0)):
                        A[list(cond[i])][0, :].neg_()
        return A

    # cases constructed using make_tensor()
    tensor_shapes = (
        (S, S),
        (1, 1),
        (3, 3, S, S),
        (3, 3, 1, 1)
    )

    for shape in tensor_shapes:
        t = make_tensor(shape, device=device, dtype=dtype)
        d = make_nonzero_det(t).requires_grad_(requires_grad)
        yield SampleInput(d)

    # cases constructed using:
    #  1) make_symmetric_matrices
    #  2) make_symmetric_pd_matrices
    #  3) make_fullrank_matrices_with_distinct_singular_values
    symmetric_shapes = (
        (S, S),
        (3, S, S),
    )


    def _helper(constructor, *shape, **kwargs):
        t = constructor(*shape, device=device, dtype=dtype)
        d = make_nonzero_det(t, **kwargs).requires_grad_(requires_grad)
        yield SampleInput(d)

    for shape in symmetric_shapes:
        _helper(make_symmetric_matrices, *shape)
        _helper(make_symmetric_pd_matrices, *shape)
        _helper(make_fullrank_matrices_with_distinct_singular_values, *shape, min_singular_value=0)


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
        return dtype in [np.bool_, bool, np.uint8, np.int8, np.int16, np.int32, np.int64]

    @wraps(fn)
    def wrapped_fn(x):
        # As the default dtype can change, acquire it when function is called.
        # NOTE: Promotion in PyTorch is from integer types to the default dtype
        np_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]

        if is_integral(x.dtype):
            return fn(x.astype(np_dtype))
        return fn(x)

    return wrapped_fn

def sample_inputs_spectral_ops(self, device, dtype, requires_grad=False, **kwargs):
    nd_tensor = partial(make_tensor, (S, S + 1, S + 2), device=device,
                        dtype=dtype, requires_grad=requires_grad)
    oned_tensor = partial(make_tensor, (31,), device=device,
                          dtype=dtype, requires_grad=requires_grad)

    if self.ndimensional == SpectralFuncType.ND:
        return [
            SampleInput(nd_tensor(),
                        kwargs=dict(s=(3, 10), dim=(1, 2), norm='ortho')),
            SampleInput(nd_tensor(),
                        kwargs=dict(norm='ortho')),
            SampleInput(nd_tensor(),
                        kwargs=dict(s=(8,))),
            SampleInput(oned_tensor()),

            *(SampleInput(nd_tensor(),
                          kwargs=dict(dim=dim))
                for dim in [-1, -2, -3, (0, -1)]),
        ]
    elif self.ndimensional == SpectralFuncType.TwoD:
        return [
            SampleInput(nd_tensor(),
                        kwargs=dict(s=(3, 10), dim=(1, 2), norm='ortho')),
            SampleInput(nd_tensor(),
                        kwargs=dict(norm='ortho')),
            SampleInput(nd_tensor(),
                        kwargs=dict(s=(6, 8))),
            SampleInput(nd_tensor(),
                        kwargs=dict(dim=0)),
            SampleInput(nd_tensor(),
                        kwargs=dict(dim=(0, -1))),
            SampleInput(nd_tensor(),
                        kwargs=dict(dim=(-3, -2, -1))),
        ]
    else:
        return [
            SampleInput(nd_tensor(),
                        kwargs=dict(n=10, dim=1, norm='ortho')),
            SampleInput(nd_tensor(),
                        kwargs=dict(norm='ortho')),
            SampleInput(nd_tensor(),
                        kwargs=dict(n=7)),
            SampleInput(oned_tensor()),

            *(SampleInput(nd_tensor(),
                          kwargs=dict(dim=dim))
                for dim in [-1, -2, -3]),
        ]

def sample_inputs_repeat_interleave(op_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        SampleInput(make_input(()), kwargs=dict(repeats=2)),
        SampleInput(make_input((2, 3, 4)), kwargs=dict(repeats=2)),
        SampleInput(make_input((2, 3, 4)), kwargs=dict(repeats=2, dim=1)),
        SampleInput(make_input((2, 3, 4)), kwargs=dict(repeats=torch.arange(3, device=device), dim=1))
    ]

SpectralFuncType = Enum('SpectralFuncType', ('OneD', 'TwoD', 'ND'))

# Metadata class for Fast Fourier Transforms in torch.fft.
class SpectralFuncInfo(OpInfo):
    """Operator information for torch.fft transforms. """

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref=None,  # Reference implementation (probably in np.fft namespace)
                 dtypes=floating_and_complex_types(),
                 ndimensional: SpectralFuncType,
                 sample_inputs_func=sample_inputs_spectral_ops,
                 decorators=None,
                 **kwargs):
        decorators = list(decorators) if decorators is not None else []
        decorators += [
            skipCPUIfNoFFT,
            skipCUDAIfRocm,
        ]

        super().__init__(name=name,
                         dtypes=dtypes,
                         decorators=decorators,
                         sample_inputs_func=sample_inputs_func,
                         **kwargs)
        self.ref = ref
        self.ndimensional = ndimensional


def sample_inputs_stft(op_info, device, dtype, requires_grad, **kwargs):
    def mt(shape, **kwargs):
        return make_tensor(shape, device=device, dtype=dtype,
                           requires_grad=requires_grad, **kwargs)
    yield SampleInput(mt(100), kwargs=dict(n_fft=10))

    for center in [False, True]:
        yield SampleInput(mt(10), kwargs=dict(n_fft=7, center=center))
        yield SampleInput(mt((10, 100)), kwargs=dict(n_fft=16, hop_length=4, center=center))

    window = make_tensor(16, low=.5, high=2.0, dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(
        mt((2, 100)), kwargs=dict(n_fft=16, window=window, return_complex=True, center=center))
    yield SampleInput(
        mt((3, 100)), kwargs=dict(n_fft=16, window=window, return_complex=True, center=center))
    if not dtype.is_complex:
        yield SampleInput(
            mt((10, 100)), kwargs=dict(n_fft=16, window=window, onesided=False))


def sample_inputs_istft(op_info, device, dtype, requires_grad, **kwargs):
    def mt(shape, **kwargs):
        real_shape = shape if dtype.is_complex else shape + (2,)
        return make_tensor(real_shape, device=device, dtype=dtype,
                           requires_grad=requires_grad, **kwargs)

    yield SampleInput(mt((10, 2)), kwargs=dict(n_fft=10))
    yield SampleInput(mt((6, 3)), kwargs=dict(n_fft=6, onesided=False))
    yield SampleInput(mt((6, 4)), kwargs=dict(n_fft=10, onesided=True))

    for center in [False, True]:
        yield SampleInput(mt((10, 10, 6)), kwargs=dict(n_fft=10, center=center))
        yield SampleInput(mt((1, 9, 10)), kwargs=dict(n_fft=16, hop_length=4, center=center))

    window = make_tensor(10, low=.5, high=2.0, dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(mt((10, 10, 6)), kwargs=dict(
        n_fft=10, window=window, center=center, return_complex=dtype.is_complex))
    yield SampleInput(mt((10, 10, 10)), kwargs=dict(
        n_fft=10, window=window[:8], win_length=8, center=center, return_complex=True))

    real_window = window if not dtype.is_complex else window.real
    yield SampleInput(mt((10, 5, 6)), kwargs=dict(n_fft=8, window=real_window[:8], center=center))


def sample_inputs_fftshift(op_info, device, dtype, requires_grad, **kwargs):
    def mt(shape, **kwargs):
        return make_tensor(shape, device=device, dtype=dtype,
                           requires_grad=requires_grad, **kwargs)

    yield SampleInput(mt((9, 10)))
    yield SampleInput(mt((50,)), kwargs=dict(dim=0))
    yield SampleInput(mt((5, 11)), kwargs=dict(dim=(1,)))
    yield SampleInput(mt((5, 6)), kwargs=dict(dim=(0, 1)))
    yield SampleInput(mt((5, 6, 2)), kwargs=dict(dim=(0, 2)))


class ShapeFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for Shape manipulating operations like tile and roll"""
    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref,  # a reference function
                 dtypes=floating_types(),
                 dtypesIfCUDA=None,
                 dtypesIfROCM=None,
                 sample_inputs_func=None,
                 **kwargs):
        super(ShapeFuncInfo, self).__init__(name,
                                            dtypes=dtypes,
                                            dtypesIfCUDA=dtypesIfCUDA,
                                            dtypesIfROCM=dtypesIfROCM,
                                            sample_inputs_func=sample_inputs_func,
                                            **kwargs)
        self.ref = ref

def sample_inputs_foreach(self, device, dtype, N, *, noncontiguous=False, same_size=False):
    if same_size:
        return [make_tensor((N, N), device, dtype, noncontiguous=noncontiguous) for _ in range(N)]
    else:
        return [make_tensor((N - i, N - i), device, dtype, noncontiguous=noncontiguous) for i in range(N)]


def get_foreach_method_names(name):
    # get torch inplace reference function
    op_name = "_foreach_" + name
    inplace_op_name = "_foreach_" + name + "_"

    op = getattr(torch, op_name, None)
    inplace_op = getattr(torch, inplace_op_name, None)

    ref = getattr(torch, name, None)
    ref_inplace = getattr(torch.Tensor, name + "_", None)
    return op, inplace_op, ref, ref_inplace

class ForeachFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for foreach functions"""
    def __init__(self,
                 name,
                 dtypes=floating_and_complex_types(),
                 dtypesIfCUDA=floating_and_complex_types_and(torch.half),
                 dtypesIfROCM=None,
                 safe_casts_outputs=True,
                 supports_alpha_param=False,
                 sample_inputs_func=sample_inputs_foreach,
                 **kwargs):
        super().__init__(
            "_foreach_" + name,
            dtypes=dtypes,
            dtypesIfCUDA=dtypesIfCUDA,
            dtypesIfROCM=dtypesIfROCM,
            safe_casts_outputs=safe_casts_outputs,
            sample_inputs_func=sample_inputs_func,
            **kwargs
        )

        foreach_method, foreach_method_inplace, torch_ref_method, torch_ref_inplace = get_foreach_method_names(name)
        self.method_variant = foreach_method
        self.inplace_variant = foreach_method_inplace
        self.ref = torch_ref_method
        self.ref_inplace = torch_ref_inplace
        self.supports_alpha_param = supports_alpha_param

        if name == "norm":
            self.ref = torch.linalg.vector_norm


def sample_inputs_linalg_cholesky_inverse(op_info, device, dtype, requires_grad=False, **kwargs):
    # Generate Cholesky factors of positive-definite (non-singular) Hermitian (symmetric) matrices
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
        out.append(SampleInput(a.clone().requires_grad_(requires_grad), kwargs=dict(upper=True)))
    return out

def sample_inputs_linalg_lstsq(op_info, device, dtype, requires_grad=False, **kwargs):
    from torch.testing._internal.common_utils import random_well_conditioned_matrix

    device = torch.device(device)

    drivers: Tuple[str, ...]
    if device.type == 'cuda':
        drivers = ('gels',)
    else:
        drivers = ('gels', 'gelsy', 'gelss', 'gelsd')

    # we generate matrices of shape (..., n + delta, n)
    deltas: Tuple[int, ...]
    if device.type == 'cpu' or has_cusolver():
        deltas = (-1, 0, +1)
    # only square systems if Cusolver is not available
    # becase we solve a lstsq problem with a transposed matrix in the backward
    else:
        deltas = (0,)

    out = []
    for batch, driver, delta in product(((), (3,), (3, 3)), drivers, deltas):
        shape = batch + (3 + delta, 3)
        a = random_well_conditioned_matrix(*shape, dtype=dtype, device=device)
        a.requires_grad_(requires_grad)
        b = make_tensor(shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        out.append(SampleInput(a, args=(b,), kwargs=dict(driver=driver)))
    return out

def sample_inputs_householder_product(op_info, device, dtype, requires_grad, **kwargs):
    """
    This function generates input for torch.linalg.householder_product (torch.orgqr).
    The first argument should be a square matrix or batch of square matrices, the second argument is a vector or batch of vectors.
    Empty, square, rectangular, batched square and batched rectangular input is generated.
    """
    # Each column of the matrix is getting multiplied many times leading to very large values for
    # the Jacobian matrix entries and making the finite-difference result of grad check less accurate.
    # That's why gradcheck with the default range [-9, 9] fails and [-2, 2] is used here.
    samples = (
        SampleInput(make_tensor((S, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((S,), device, dtype, low=-2, high=2, requires_grad=requires_grad),)),

        SampleInput(make_tensor((S + 1, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((S,), device, dtype, low=-2, high=2, requires_grad=requires_grad),)),

        SampleInput(make_tensor((2, 1, S, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((2, 1, S,), device, dtype, low=-2, high=2, requires_grad=requires_grad),)),

        SampleInput(make_tensor((2, 1, S + 1, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((2, 1, S,), device, dtype, low=-2, high=2, requires_grad=requires_grad),)),

        SampleInput(make_tensor((0, 0), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(make_tensor((0,), device, dtype, low=None, high=None, requires_grad=requires_grad),)),

        SampleInput(make_tensor((S, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((0,), device, dtype, low=None, high=None, requires_grad=requires_grad),)),

        # m = n = S, k = S - 2
        SampleInput(make_tensor((S, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((S - 2,), device, dtype, low=None, high=None, requires_grad=requires_grad),)),

        # m = S, n = S -1, k = S - 2
        SampleInput(make_tensor((S, S - 1), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((S - 2,), device, dtype, low=None, high=None, requires_grad=requires_grad),)),
    )

    return samples

def sample_inputs_ormqr(op_info, device, dtype, requires_grad, **kwargs):
    # create a helper function wrapping `make_tensor`
    make_input = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    def gen_inputs():
        batches = [(), (0, ), (2, ), (2, 1)]
        ns = [5, 2, 0]
        tf = [True, False]
        for batch, (m, n), left, transpose in product(batches, product(ns, ns), tf, tf):
            reflectors = make_input((*batch, m, n))
            tau = make_input((*batch, min(m, n)))
            other_matrix_shape = (m, n) if left else (n, m)
            other = make_input((*batch, *other_matrix_shape))
            kwargs = {"left": left, "transpose": transpose}
            yield SampleInput(reflectors, args=(tau, other,), kwargs=kwargs)

    return tuple(gen_inputs())

def sample_inputs_linalg_cholesky(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates always positive-definite input for torch.linalg.cholesky using
    random_hermitian_pd_matrix.
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    from torch.testing._internal.common_utils import random_hermitian_pd_matrix

    batches = [(), (0, ), (2, ), (1, 1)]
    ns = [5, 0]
    out = []
    for batch, n, upper in product(batches, ns, [True, False]):
        a = random_hermitian_pd_matrix(n, *batch, dtype=dtype, device=device)
        a.requires_grad = requires_grad
        out.append(SampleInput(a, kwargs={"upper": upper}))
    return out

def sample_inputs_symeig(op_info, device, dtype, requires_grad=False, **kwargs):
    out = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)

    for o in out:
        o.kwargs = {"upper": bool(np.random.choice([True, False])),
                    "eigenvectors": True}
        # A gauge-invariant function
        o.output_process_fn_grad = lambda output: (output[0], abs(output[1]))
        yield o

def sample_inputs_linalg_eig(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.eig
    """
    def out_fn(output):
        return output[0], abs(output[1])

    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    for sample in samples:
        sample.output_process_fn_grad = out_fn
        yield sample

def sample_inputs_linalg_eigh(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.eigh/eigvalsh with UPLO="U" or "L" keyword argument.
    """
    def out_fn(output):
        if isinstance(output, tuple):
            # eigh function
            return output[0], abs(output[1])
        else:
            # eigvalsh function
            return output

    # Samples do not need to be Hermitian, as we're using gradcheck_wrapper_hermitian_input
    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    for sample in samples:
        sample.kwargs = {"UPLO": np.random.choice(["L", "U"])}
        sample.output_process_fn_grad = out_fn
        yield sample


def sample_inputs_linalg_slogdet(op_info, device, dtype, requires_grad=False, **kwargs):
    def out_fn(output):
        return output[1]

    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    for sample in samples:
        sample.output_process_fn_grad = out_fn
        yield sample


def sample_inputs_linalg_pinv(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.pinv with hermitian=False keyword argument.
    """
    for o in sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad, **kwargs):
        real_dtype = o.input.real.dtype if dtype.is_complex else dtype
        # requires_grad path for rtol tensor is not implemented
        for rtol in (None, 1.0, torch.tensor(1.0, dtype=real_dtype, device=device)):
            o = clone_sample(o)
            o.kwargs = {"rtol": rtol}
            yield o


def sample_inputs_linalg_pinv_hermitian(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.pinv with hermitian=True keyword argument.
    """
    for o in sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad, **kwargs):
        o.kwargs = {"hermitian": True}
        yield o

def sample_inputs_linalg_solve(op_info, device, dtype, requires_grad=False, vector_rhs_allowed=True, **kwargs):
    """
    This function generates always solvable input for torch.linalg.solve
    We sample a fullrank square matrix (i.e. invertible) A
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
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_a = partial(make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad)
    make_b = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    batches = [(), (0, ), (2, )]
    ns = [5, 0]
    if vector_rhs_allowed:
        nrhs = [(), (1,), (3,)]
    else:
        nrhs = [(1,), (3,)]

    for n, batch, rhs in product(ns, batches, nrhs):
        yield SampleInput(make_a(*batch, n, n), args=(make_b((batch + (n,) + rhs)),))


def sample_inputs_linalg_solve_triangular(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device)
    bs = (1, 2, 0)
    ns = (3, 0)
    ks = (1, 3, 0)

    for b, n, k, (left, upper, uni) in product(bs, ns, ks, product((True, False), repeat=3)):
        with torch.no_grad():
            if b == 1:
                A = make_arg((n, n)) if left else make_arg((k, k))
                B = make_arg((n, k))
            else:
                A = make_arg((b, n, n)) if left else make_arg((b, k, k))
                B = make_arg((b, n, k))
            if uni:
                # Not really necessary, but writing it for consistency
                A.diagonal(0, -2, -1).fill_(1.)
            else:
                d = A.diagonal(0, -2, -1)
                d[d.abs() < 1e-6] = 1.
            if upper:
                A.triu_()
            else:
                A.tril_()
        kwargs = {"upper": upper, "left": left, "unitriangular": uni}
        if requires_grad:
            for grad_A, grad_B in product((True, False), repeat=2):
                # Either A or B needs to have a gradient
                if not grad_A and not grad_B:
                    continue
                yield SampleInput(
                    A.clone().requires_grad_(grad_A),
                    args=(B.clone().requires_grad_(grad_B),),
                    kwargs=kwargs)
        else:
            yield SampleInput(A, args=(B,), kwargs=kwargs)

def sample_inputs_legacy_solve(op_info, device, dtype, requires_grad=False, **kwargs):
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

    # Reverses tensor order
    for sample in out:
        sample.input, sample.args = sample.args[0], (sample.input,)
        yield sample


def sample_inputs_cholesky_solve(op_info, device, dtype, requires_grad=False, **kwargs):
    out = sample_inputs_linalg_cholesky_inverse(
        op_info, device, dtype, requires_grad=False
    )

    for sample in out:
        psd_matrix = sample.input
        sample.input = make_tensor(psd_matrix.shape, device, dtype, requires_grad=requires_grad, low=None, high=None)
        sample.args = (psd_matrix.requires_grad_(requires_grad),)

    return out


def sample_inputs_lu(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_fullrank_matrices_with_distinct_singular_values,
                       dtype=dtype, device=device, requires_grad=requires_grad)

    # not needed once OpInfo tests support Iterables
    batch_shapes = ((), (3,), (3, 3))
    for batch_shape, get_infos, size_delta in product(batch_shapes, (True, False), (-2, -1, 0, +1, +2)):
        shape = batch_shape + (S + size_delta, S)
        input = make_arg(*shape)
        yield SampleInput(input, args=(True, get_infos))

def sample_inputs_linalg_lu_factor(op_info, device, dtype, requires_grad=False, **kwargs):
    # When calling `lu_factor` we need to assure that the matrix is invertible
    make_fn = make_tensor if "ex" in op_info.name else make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(make_fn, dtype=dtype, device=device, requires_grad=requires_grad)

    # not needed once OpInfo tests support Iterables
    batch_shapes = ((), (3,), (3, 3))
    # pivot=False only supported in CUDA
    pivots = (True, False) if torch.device(device).type == "cuda" else (True,)
    deltas = (-2, -1, 0, +1, +2)
    for batch_shape, pivot, delta in product(batch_shapes, pivots, deltas):
        shape = batch_shape + (S + delta, S)
        # Insanely annoying that make_fullrank_blablabla accepts a *shape and not a tuple!
        A = make_arg(shape) if "ex" in op_info.name else make_arg(*shape)
        yield SampleInput(A, kwargs={"pivot": pivot})


def sample_inputs_lu_solve(op_info, device, dtype, requires_grad=False, **kwargs):
    make_fn = make_fullrank_matrices_with_distinct_singular_values
    make_a = partial(make_fn, dtype=dtype, device=device)
    make_b = partial(make_tensor, dtype=dtype, device=device)

    batches = ((), (0, ), (2, ))
    ns = (5, 3, 0)
    nrhs = (0, 1, 6)

    for n, batch, rhs in product(ns, batches, nrhs):
        shape_a = batch + (n, n)
        a = make_a(*shape_a)
        lu, pivs = a.lu()
        lu = lu.contiguous()

        shape_b = batch + (n, rhs)
        b = make_b(shape_b)

        grads = (False,) if not requires_grad else (True, False)
        # we try all possible combinations of requires_grad for each input
        for lu_grad, b_grad in product(grads, grads):
            # when requires_grad == True, at least one input has to have requires_grad enabled
            if requires_grad and not lu_grad and not b_grad:
                continue

            lu_ = lu.clone()
            lu_.requires_grad_(lu_grad)
            b_ = b.clone()
            b_.requires_grad_(b_grad)
            yield SampleInput(b_, args=(lu_, pivs))

def sample_inputs_lu_unpack(op_info, device, dtype, requires_grad=False, **kwargs):
    for lu_sample in sample_inputs_lu(op_info, device, dtype, requires_grad, **kwargs):
        lu_data, pivots = torch.linalg.lu_factor(lu_sample.input)
        lu_data.requires_grad_(requires_grad)
        yield SampleInput(lu_data, args=(pivots,))


def sample_inputs_roll(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    args = ((0, 0), (1, 2), (0, 2), (2, 0), (-1, 0), (10000, 1), (2,), ((1, 2, -1), (0, 1, 2)))

    for arg in args:
        yield SampleInput(make_arg((S, S, S)), args=arg)


def sample_inputs_rot90(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    args = ((1, (0, 1),),
            (1, (1, 2),),
            (1, (1, -1),),
            ())

    for arg in args:
        yield SampleInput(make_arg((S, S, S)), args=arg)


def sample_inputs_std_var(op_info, device, dtype, requires_grad, **kwargs):
    tensor_nd = partial(make_tensor, (S, S, S), device=device, dtype=dtype,
                        requires_grad=requires_grad)
    tensor_1d = partial(make_tensor, (S,), device=device, dtype=dtype,
                        requires_grad=requires_grad)

    return [
        SampleInput(tensor_nd()),
        SampleInput(tensor_nd(), kwargs=dict(dim=1)),
        SampleInput(tensor_nd(), kwargs=dict(dim=1, unbiased=True, keepdim=True)),
        SampleInput(tensor_1d(), kwargs=dict(dim=0, unbiased=True, keepdim=True)),
        SampleInput(tensor_1d(), kwargs=dict(dim=0, unbiased=False, keepdim=False)),

        SampleInput(tensor_nd(), kwargs=dict(dim=(1,), correction=S // 2)),
        SampleInput(tensor_nd(), kwargs=dict(dim=None, correction=0, keepdim=True)),
    ]


def _generate_correlation_inputs(device, dtype, requires_grad, **kwargs):
    shapes = [(2,), (1, 2), (3, 2), (2, 3)]
    for shape in shapes:
        yield make_tensor(shape, device, dtype, requires_grad=requires_grad)


def sample_inputs_corrcoef(op_info, device, dtype, requires_grad, **kwargs):
    return [SampleInput(t) for t in _generate_correlation_inputs(device, dtype, requires_grad)]


def sample_inputs_cov(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    for t in _generate_correlation_inputs(device, dtype, requires_grad):
        inputs.append(SampleInput(t))
        num_observations = t.numel() if t.ndimension() < 2 else t.size(1)
        fweights = make_tensor((num_observations,), device, torch.int, low=1, high=10)
        aweights = make_tensor((num_observations,), device, torch.float, low=0, high=1, requires_grad=requires_grad)
        for correction, fw, aw in product(range(num_observations), [None, fweights], [None, aweights]):
            inputs.append(SampleInput(t.clone().requires_grad_(requires_grad),
                                      kwargs={'correction': correction, 'fweights': fw, 'aweights': aw}))
    return inputs


def sample_inputs_svd(op_info, device, dtype, requires_grad=False, **kwargs):
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad)

    is_linalg_svd = (op_info.name == "linalg.svd")
    batches = [(), (0, ), (3, )]
    ns = [0, 3, 5]

    def uniformize(usv):
        S = usv[1]
        k = S.shape[-1]
        U = usv[0][..., :k]
        Vh = usv[2] if is_linalg_svd else usv[2].mH
        Vh = Vh[..., :k, :]
        return U, S, Vh

    def fn_U(usv):
        U, _, _ = uniformize(usv)
        return U.abs()


    def fn_S(usv):
        return uniformize(usv)[1]

    def fn_Vh(usv):
        # We also return S to test
        _, S, Vh = uniformize(usv)
        return S, Vh.abs()

    def fn_UVh(usv):
        U, S, Vh = uniformize(usv)
        return U @ Vh, S

    fns = (fn_U, fn_S, fn_Vh, fn_UVh)

    fullmat = 'full_matrices' if is_linalg_svd else 'some'

    for batch, n, k, fullmat_val, fn in product(batches, ns, ns, (True, False), fns):
        shape = batch + (n, k)
        yield SampleInput(make_arg(*shape), kwargs={fullmat: fullmat_val}, output_process_fn_grad=fn)


def sample_inputs_permute(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = [((1, 2, 3, 4), (0, 2, 3, 1)),
             ((1, 2, 3, 4), (0, -2, -1, 1)),
             ((), ()),
             ((1, 2, 3, 4), (2, 1, 3, 0))]

    for shape, args in cases:
        yield SampleInput(make_arg(shape), args=(args,))


# Based on erstwhile method_tests tests & some tensor_op_tests for pow
def sample_inputs_pow(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype)

    samples = []

    if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        test_cases = (
            ((2, 2), 0, 5, 1e-3, requires_grad, (2, 2), 0, 1, 0.1, requires_grad, False),
            ((2, 2), 0, 5, 1e-3, requires_grad, (1,), 0, 1, 0.1, requires_grad, False),
            ((), 1e-3, 1e-3 + 1, 0, requires_grad, (), 0.1, 1.1, 0, requires_grad, False),
            ((2, 2), 0, 5, 1e-3, requires_grad, (), 0.1, 1.1, 1, requires_grad, False),
        )
        tests_require_resizing = (
            ((1,), 0, 5, 1e-3, requires_grad, (2, 2), 0, 1, 0.1, requires_grad, requires_grad),
            ((2, 1, 2), 0, 5, 1e-3, requires_grad, (1, 2, 1), 0, 1, 0.1, requires_grad, requires_grad),
            ((), 1e-3, 1e-3 + 1, 0, requires_grad, (1, S, 1), 0, 1, 0.1, requires_grad, requires_grad),
        )
        cases = test_cases + tests_require_resizing

        samples = []
        for (shape_b, low_b, high_b, additive_b, b_grad, shape_e, low_e,
             high_e, additive_e, e_grad, broadcasts_input) in cases:
            si = SampleInput((make_arg(shape_b, low=low_b, high=high_b) + additive_b).requires_grad_(b_grad),
                             args=((make_arg(shape_e, low=low_e, high=high_e) + additive_e).requires_grad_(e_grad),),
                             broadcasts_input=broadcasts_input)
            samples.append(si)

        tensor_scalar_inputs = (
            ((2, 2), 0, 5, 1e-3, requires_grad, (3.14,)),
            ((), 1e-3, 1e-3 + 1, 0, requires_grad, (3.14,))
        )
        more_samples = list(SampleInput(
            (make_arg(shape, high=high, low=low) + additive).requires_grad_(b_grad),
            args=exp)
            for shape, low, high, additive, b_grad, exp in tensor_scalar_inputs)

        samples = [*samples, *more_samples]
    elif dtype in [torch.complex64, torch.complex128]:
        args_tuple = (
            ((2, 2), 0, 5, requires_grad, (3.14,)),
            ((), 0, 1, requires_grad, (3.14,)),
            ((), 0, 1, requires_grad, (3.14j,))
        )
        samples = list(SampleInput(
            (make_arg(shape, high=high, low=low) + 1e-3 * (1 + 1j)).requires_grad_(b_grad),
            args=arg)
            for shape, low, high, b_grad, arg in args_tuple)
    else:  # integral dtype
        exp_tuple = (1, 2, 3)
        samples = list(SampleInput(
            make_arg((2, 2), requires_grad=requires_grad),
            args=(arg,))
            for arg in exp_tuple)
        samples.append(SampleInput(
            make_arg((2, 2), requires_grad=requires_grad),
            args=(make_arg((2, 2), requires_grad=requires_grad),)))
    return tuple(samples)

def sample_inputs_linalg_svdvals(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    batches = [(), (0, ), (2, ), (1, 1)]
    ns = [5, 2, 0]

    for batch, m, n in product(batches, ns, ns):
        yield SampleInput(make_arg(batch + (m, n)))


def sample_inputs_softshrink_hardshrink_hardtanh(op_info, device, dtype, requires_grad=False, **kwargs):
    N = 10
    tensors = [SampleInput(make_tensor((N, N), device=device, dtype=dtype,
               requires_grad=requires_grad)) for _ in range(1, N)]
    return tensors

def sample_inputs_eig(op_info, device, dtype, requires_grad=False, **kwargs):
    eigvecs = make_tensor((S, S), device=device, dtype=dtype,
                          low=None, high=None)
    eigvals = make_tensor((S,), device=device, dtype=dtype,
                          low=None, high=None)
    # we produce only diagonazible inputs which do not have
    # complex eigenvalues for real inputs, as there is no
    # backward implementation for real inputs with complex
    # eigenvalues yet.
    input = (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.inverse()
    input.requires_grad_(requires_grad)

    def process_output(eigpair):
        eigvals, eigvecs = eigpair
        if dtype.is_complex:
            # eig produces eigenvectors which are normalized to 1 norm.
            # Note that if v is an eigenvector, so is v * e^{i \phi},
            # and |v| = |v * e^{i \phi}| = 1.
            # This, however, makes the eigenvector backward computation process
            # rather unstable unless the objective function is gauge-invariant,
            # that is if f(z) == f(|z|), for example.
            # Hence for complex inputs we ignore the phases and return only
            # the absolute values.
            return eigvals, eigvecs.abs()
        else:
            return eigvals, eigvecs

    return [
        SampleInput(
            input,
            kwargs=dict(eigenvectors=True),
            output_process_fn_grad=process_output
        ),
    ]


def sample_inputs_einsum(op_info, device, dtype, requires_grad=False, **kwargs):
    def c(t):
        return t.clone().requires_grad_(requires_grad)

    x = make_tensor((3,), device, dtype, requires_grad=requires_grad)
    y = make_tensor((4,), device, dtype, requires_grad=requires_grad)
    A = make_tensor((2, 3,), device, dtype, requires_grad=requires_grad)
    B = make_tensor((1, 3,), device, dtype, requires_grad=requires_grad)
    C = make_tensor((1, 2, 3,), device, dtype, requires_grad=requires_grad)
    D = make_tensor((1, 3, 4,), device, dtype, requires_grad=requires_grad)
    E = make_tensor((4, 4,), device, dtype, requires_grad=requires_grad)
    H = make_tensor((3, 3,), device, dtype, requires_grad=requires_grad)
    I = make_tensor((1, 3, 1,), device, dtype, requires_grad=requires_grad)

    inputs = []

    # Vector operations
    inputs.append(SampleInput([c(x)], args=('i->',)))                      # sum
    inputs.append(SampleInput([c(x), c(y)], args=('i,j->ij',)))               # outer

    # Matrix operations
    inputs.append(SampleInput([c(A)], args=("ij->i",)))                    # col sum
    inputs.append(SampleInput([c(A), c(B)], args=("ij,kj->ik",)))             # matmul
    inputs.append(SampleInput([c(A), c(E)], args=("ij,Ab->ijAb",)))           # matrix outer product

    # Tensor operations
    inputs.append(SampleInput([c(C), c(D)], args=("aij,ajk->aik",)))          # batch matmul
    inputs.append(SampleInput([c(D), c(E)], args=("aij,jk->aik",)))           # tensor matrix contraction
    inputs.append(SampleInput([c(C), c(B)], args=("ijk,ik->j",)))             # non contiguous

    # Test diagonals
    inputs.append(SampleInput([c(I)], args=('iji->j',)))                   # non-contiguous trace

    # Test ellipsis
    inputs.append(SampleInput([c(H)], args=("i...->...",)))
    inputs.append(SampleInput([c(C), c(x)], args=('...ik, ...j -> ij',)))

    return inputs


def sample_inputs_linalg_qr_geqrf(op_info, device, dtype, requires_grad=False, **kwargs):
    # QR is just well defined when the matrix is full rank
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad)

    batches = [(), (0,), (2, ), (1, 1)]
    ns = [5, 2, 0]

    for batch, (m, n) in product(batches, product(ns, ns)):
        shape = batch + (m, n)
        yield SampleInput(make_arg(*shape))

def sample_inputs_flip(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = ((S, M, S), (S, 0, M))
    all_dims = ((0, 1, 2), (0,), (0, 2), (-1,), ())

    for size, dims in product(sizes, all_dims):
        yield SampleInput(make_arg(size), kwargs={"dims": dims})

def sample_inputs_fliplr_flipud(op_info, device, dtype, requires_grad, **kwargs):
    tensors = (
        make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((S, 0, M), device, dtype, low=None, high=None, requires_grad=requires_grad)
    )
    return [SampleInput(tensor) for tensor in tensors]

def sample_inputs_fmod_remainder(op_info, device, dtype, requires_grad, *, autodiffed=False, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    if autodiffed:
        samples = (
            ((S, S, S), 1.5, False),
            ((), 1.5, False),
        )
    else:
        cases = (
            ((S, S, S), (), False),
            ((S, S, S), (S, S, S), False),
            ((S, S, S), (S,), False),
        )

        # Sample inputs with scalars as torch tensors
        # FIXME It does not work for mak make_arg((1,), exclude_zero=True)
        cases_with_tensor_scalar = (
            ((), make_arg((), exclude_zero=True), False),
        )

        # Sample inputs with broadcasting
        cases_with_broadcasting = (
            ((S,), (S, S, S), True),
            ((S, 1, S), (S, S, S), True),
            ((), (S, S, S), True),
        )

        samples = cases + cases_with_tensor_scalar + cases_with_broadcasting  # type: ignore[assignment]

    for shape, arg_other, broadcasts_input in samples:
        if isinstance(arg_other, tuple):
            arg = make_arg(arg_other, exclude_zero=True)
        else:
            # shape_other is scalar or torch.tensor
            arg = arg_other
        yield(SampleInput(make_arg(shape), args=(arg,), broadcasts_input=broadcasts_input))

# TODO: clamp shares tensors among its sample inputs --- we should prohibit this!
def sample_inputs_clamp(op_info, device, dtype, requires_grad, **kwargs):
    x = make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad)
    lb = make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad)
    ub = make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad)

    def detach(tensor):
        return tensor.clone().detach_().requires_grad_(requires_grad)

    return [
        SampleInput(detach(x), args=(lb, ub)),
        SampleInput(detach(x), args=(detach(lb[0]), detach(ub[0]))),
        SampleInput(detach(x), args=(detach(lb[:, :1]),)),
    ]

def sample_inputs_clamp_scalar(op_info, device, dtype, requires_grad, **kwargs):
    tensors = (
        make_tensor((2, 3, 2), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((2, 0, 3), device, dtype, low=None, high=None, requires_grad=requires_grad),
    )

    if dtype is torch.uint8:
        min_max_vals = ((2, 5), (3, 7))
    else:
        min_max_vals = ((0, 1), (-1, 1))

    output = [SampleInput(
        tensor.clone().requires_grad_(requires_grad),
        args=vals) for tensor, vals in product(tensors, min_max_vals)]
    output += [
        SampleInput(tensors[0].clone().requires_grad_(requires_grad),
                    args=(0.5, None)),
        SampleInput(tensors[0].clone().requires_grad_(requires_grad),
                    args=(None, 0.5))]
    empty_tensor = make_tensor((), device=device, dtype=dtype, low=None, high=None, requires_grad=requires_grad)
    output.append(SampleInput(empty_tensor, args=(0.0, 1.0)))
    return output

def sample_kwargs_clamp_scalar(device, dtype, input):
    if dtype is torch.uint8:
        min_val, max_val = (random.randint(1, 3), random.randint(4, 8))
    elif dtype.is_floating_point:
        min_val, max_val = (random.uniform(-8, 0), random.uniform(1, 8))  # type: ignore[assignment]
    else:
        min_val, max_val = (random.randint(-8, 0), random.randint(1, 8))
    return {'min': min_val, 'max': max_val}, {'a_min': min_val, 'a_max': max_val}

def sample_inputs_cross(op_info, device, dtype, requires_grad, **kwargs):
    sample0 = SampleInput(make_tensor((S, 3), device=device, dtype=dtype, requires_grad=requires_grad),
                          args=(make_tensor((S, 3), device=device, dtype=dtype, requires_grad=requires_grad),))
    sample1 = SampleInput(make_tensor((S, 3, S), device=device, dtype=dtype, requires_grad=requires_grad),
                          args=(make_tensor((S, 3, S), device=device, dtype=dtype, requires_grad=requires_grad),),
                          kwargs={'dim': 1})
    sample2 = SampleInput(make_tensor((S, 3), device=device, dtype=dtype, requires_grad=requires_grad),
                          args=(make_tensor((S, 3), device=device, dtype=dtype, requires_grad=requires_grad),),
                          kwargs={'dim': -1})

    return (sample0, sample1, sample2)

def sample_inputs_cumprod(op_info, device, dtype, requires_grad, **kwargs):
    def make_arg(shape):
        # shrink values to be in the interval [-1, +1] for better precision in gradgradcheck
        return make_tensor(shape, device, dtype, low=-1, high=+1, requires_grad=requires_grad)

    def prod_zeros(dim_select):
        assert len(dim_select) == 2
        result = make_arg(3 * (S,))
        result.narrow(dim_select[0], 0, 1).narrow(dim_select[1], 1, 1).zero_()
        result.narrow(dim_select[0], 2, 1).narrow(dim_select[1], 3, 1).zero_()
        result.narrow(dim_select[0], 4, 1).narrow(dim_select[1], 3, 1).zero_()
        return result

    for dim in range(3):
        yield SampleInput(make_arg((S, S, S)), args=(dim,))
    # Scalar tensors and empty tensor
    for size in [(), (1,), (0,)]:
        yield SampleInput(make_arg(size), args=(0,))

    yield SampleInput(prod_zeros([0, 1]), args=(1,))
    yield SampleInput(prod_zeros([0, 2]), args=(1,))
    yield SampleInput(prod_zeros([1, 2]), args=(1,))

    # test dtype kwarg
    yield SampleInput(prod_zeros([1, 2]), args=(1,), kwargs={'dtype': dtype})

def sample_inputs_view_as_complex(op_info, device, dtype, requires_grad, **kwargs):
    return [SampleInput(make_tensor((S, 2), device, dtype, requires_grad=requires_grad),)]

def sample_inputs_view_as_real(op_info, device, dtype, requires_grad, **kwargs):
    tensors = (
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((), device, dtype, requires_grad=requires_grad)
    )
    return [SampleInput(tensor) for tensor in tensors]

def sample_inputs_copysign(op_info, device, dtype, requires_grad, **kwargs):
    def _make_tensor(*shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    cases = [
        # no broadcast
        ((S, S, S), (S, S, S), False),
        # broadcast rhs
        ((S, S, S), (S, S), False),

        # scalar
        ((S, S), 3.14, False),
        # scalar positive zero
        ((S, S), 0.0, False),
        # scalar negative zero
        ((S, S), -0.0, False),
    ]

    # broadcast lhs
    cases.append(((S, S), (S, S, S), True))
    # broadcast all
    cases.append(((S, 1, S), (M, S), True))

    for input_shape, arg_val, broadcasts_input in cases:
        if isinstance(arg_val, tuple):
            arg = _make_tensor(*arg_val)
        else:
            # arg_val is scalar
            arg = arg_val

        yield SampleInput(_make_tensor(*input_shape), args=(arg, ), broadcasts_input=broadcasts_input)

def sample_inputs_prod(op_info, device, dtype, requires_grad, **kwargs):
    def make_arg(shape):
        # shrink values to be in the interval [-1, +1] for better precision in gradgradcheck
        return make_tensor(shape, device, dtype, low=-1, high=+1, requires_grad=requires_grad)

    def prod_single_zero():
        result = make_arg(2 * (S,))
        result[0, 1] = 0
        return result

    for sample in sample_inputs_cumprod(op_info, device, dtype, requires_grad):
        # only Tensor, ignore other inputs
        yield SampleInput(sample.input.clone().requires_grad_(requires_grad))
        yield sample

    # Generates samples with keepdim = True
    for sample in sample_inputs_cumprod(op_info, device, dtype, requires_grad):
        sample.kwargs['keepdim'] = True
        yield sample

    yield SampleInput(prod_single_zero())
    yield SampleInput(make_arg((3, 3, 3)), args=(1,))
    yield SampleInput(make_arg((3, 3, 3)), args=(1,), kwargs={'keepdim': True})

    # test zero scalar tensor
    zero = make_arg(())
    zero.zero_()
    yield SampleInput(zero.clone().requires_grad_(requires_grad))
    yield SampleInput(zero.clone().requires_grad_(requires_grad), args=(0,))
    yield SampleInput(zero.clone().requires_grad_(requires_grad),
                      args=(0,),
                      kwargs={'keepdim': True})

def error_inputs_neg(op_info, device, **kwargs):
    si = SampleInput(torch.tensor((False, True), device=device))
    msg = ("Negation, the `\\-` operator, on a bool tensor is not supported."
           " If you are trying to invert a mask, use the `\\~` or"
           " `logical_not\\(\\)` operator instead.")
    return (ErrorInput(si, error_type=RuntimeError, error_regex=msg),)

def sample_inputs_nextafter(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (
        ((S, S), (S, S), False),
        ((S, S), (S,), False),
        ((S, ), (S, S), True)
    )

    for shape, other_shape, broadcasts_input in cases:
        yield SampleInput(make_arg(shape), args=(make_arg(other_shape),), broadcasts_input=broadcasts_input)


def sample_inputs_diag(op_info, device, dtype, requires_grad, **kwargs):
    vec_sample = SampleInput(make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad))

    tensors = (
        make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((3, 5), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((5, 3), device, dtype, low=None, high=None, requires_grad=requires_grad),
    )

    args = ((), (2,), (-2,), (1,), (2,))

    samples = []
    for tensor, arg in product(tensors, args):
        samples.append(SampleInput(tensor.clone().requires_grad_(requires_grad), args=arg))

    return samples + [vec_sample]

def sample_inputs_diagonal_diag_embed(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    # Shapes for 2D Tensors
    shapes_2d = ((M, M), (3, 5), (5, 3))

    # Shapes for 3D Tensors
    shapes_3d = ((M, M, M),)

    kwargs_2d = (dict(), dict(offset=2), dict(offset=2), dict(offset=1))
    kwargs_3d = (dict(offset=1, dim1=1, dim2=2),
                 dict(offset=2, dim1=0, dim2=1),
                 dict(offset=-2, dim1=0, dim2=1))

    for shape, kwarg in chain(product(shapes_2d, kwargs_2d), product(shapes_3d, kwargs_3d)):
        yield SampleInput(make_arg(shape), kwargs=kwarg)


def sample_inputs_diagonal_scatter(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    # Shapes for 2D Tensors
    shapes_2d = ((M, M), (3, 5), (5, 3))

    # Shapes for 3D Tensors
    shapes_3d = ((M, M, M),)

    args_2d = ((), (2,), (-2,), (1,))
    args_3d = ((1, 1, 2), (2, 0, 1), (-2, 0, 1))

    for input_shape, arg in chain(product(shapes_2d, args_2d), product(shapes_3d, args_3d)):
        input_ = make_arg(input_shape)
        # We can programatically figure out the right shape for src:
        # It should be the same size as input.diagonal(other_args...)
        if not isinstance(arg, tuple):
            arg_tuple = (arg,)
        else:
            arg_tuple = arg
        src_shape = input_.diagonal(*arg_tuple).size()
        src = make_arg(src_shape)
        yield SampleInput(input_, args=(src, *arg_tuple))


def sample_inputs_to_sparse(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return (SampleInput(make_arg((S, S)), args=(), output_process_fn_grad=lambda x: x.to_dense()),
            SampleInput(make_arg((S, S)), args=(1,), output_process_fn_grad=lambda x: x.to_dense()),)

def sample_inputs_cross_entropy(op_info, device, dtype, requires_grad, **kwargs):
    batch_size, num_classes = shape = (2, 3)
    reductions = ("mean", "sum", "none")

    input_shape_and_kwargs: List[Tuple[Tuple[int, ...], Dict[str, Any]]] = [
        (shape, dict()),
        ((*shape, 1), dict()),
        ((*shape, 1, 2), dict()),
        ((*shape, 1, 2, 3), dict()),
        *[(shape, dict(reduction=reduction)) for reduction in reductions],
        *[
            (
                shape,
                dict(
                    weight=make_tensor((num_classes,), device=device, dtype=dtype),
                    reduction=reduction,
                ),
            )
            for reduction in reductions
        ],
        (shape, dict(ignore_index=1)),
    ]

    sample_inputs = []
    for (input_shape, kwargs), probabilities_target in itertools.product(input_shape_and_kwargs, (False, True)):
        input = make_tensor(input_shape, device=device, dtype=dtype, requires_grad=requires_grad)

        if probabilities_target:
            # ignore_index is not supported for probabilities target
            if "ignore_index" in kwargs:
                continue

            target = make_tensor(
                input_shape,
                low=0,
                high=1,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )
        else:
            target = make_tensor(
                (batch_size, *input_shape[2:]),
                low=0,
                high=num_classes,
                device=device,
                dtype=torch.long,
            )

            if "ignore_index" in kwargs and torch.all(target == kwargs["ignore_index"]):
                # make sure at least one item in target is not ignored
                target[0] = random.sample(set(range(num_classes)) - {kwargs["ignore_index"]}, 1)[0]

        sample_inputs.append(SampleInput(input, args=(target,), kwargs=kwargs))

    return sample_inputs

# Used for log_softmax, softmax, softmin
def sample_inputs_softmax_variant(op_info, device, dtype, requires_grad, with_dtype=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = [
        ((S, ), (0, )),
        ((S, S), (0, )),
        ((S, S), (1, )),
        ((S, S), (-1, )),
        ((S, M, S), (2, )),
    ]

    # PyTorch on XLA throws an error when passed with dim argument for 0d tensor.
    # See https://github.com/pytorch/xla/issues/3061 for more details.
    if torch.device(device).type != 'xla':
        cases.append(((), (0, )))

    return [
        SampleInput(make_arg(shape), args=dim, kwargs=dict(dtype=torch.float64) if with_dtype else None)
        for shape, dim in cases
    ]


def sample_inputs_masked_softmax(op_info, device, dtype, requires_grad, with_dtype=False, **kwargs):
    """Sample inputs for masked softmax, log_softmax, and softmin.

    Masked normalization operator is a reduction operator with
    trailing mask optional argument. A mask is a bool tensor with the
    same shape as input or a shape that is broadcastable to input
    shape.
    """
    inputs: List[SampleInput] = []
    for sample_input in sample_inputs_softmax_variant(op_info, device, dtype, requires_grad, with_dtype=with_dtype, **kwargs):
        for mask in _generate_masked_op_mask(sample_input.input.shape, device, **kwargs):
            sample_input_args, sample_input_kwargs = sample_input.args, dict(mask=mask, **sample_input.kwargs)
            inputs.append(SampleInput(sample_input.input.clone().requires_grad_(requires_grad),
                                      args=sample_input_args, kwargs=sample_input_kwargs))
    return inputs


def sample_inputs_masked_normalize(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked normalize.
    """
    inputs: List[SampleInput] = []
    for ord in [2.0, 1, float('inf'), float('-inf'), 0]:
        for sample_input in sample_inputs_softmax_variant(op_info, device, dtype, requires_grad, **kwargs):
            sample_input_args, sample_input_kwargs = (ord,) + sample_input.args, sample_input.kwargs.copy()
            inputs.append(SampleInput(sample_input.input.clone().requires_grad_(requires_grad),
                                      args=sample_input_args, kwargs=sample_input_kwargs))
    return inputs

def sample_inputs_logit(op_info, device, dtype, requires_grad, **kwargs):
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

def sample_inputs_isin(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # isin has two paths based on the size of elements and test_elements.
    # if elements.numel() < 10 * pow(test_elements.numel(), 0.145):
    yield SampleInput(make_arg((L,)), args=(make_arg((S,)),))
    # else:
    yield SampleInput(make_arg((S,)), args=(make_arg((L,)),))

def sample_inputs_masked_scatter(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, make_arg((S, S))))
    yield SampleInput(make_arg((S, S)), args=(torch.randn((S,), device=device) > 0, make_arg((S, S))))
    yield SampleInput(make_arg((S, S)), args=(bernoulli_scalar().to(device), make_arg((S, S))))
    yield SampleInput(make_arg((S,)),
                      args=(torch.randn(S, S, device=device) > 0, make_arg((S, S))),
                      broadcasts_input=True)


def sample_inputs_masked_fill(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, 10))
    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, make_arg(())))
    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, device=device) > 0, 10))
    yield SampleInput(make_arg(()), args=(torch.randn((), device=device) > 0, 10))
    yield SampleInput(make_arg(()), args=(torch.randn((), device=device) > 0, make_arg(())))
    yield SampleInput(make_arg((S, S)), args=(torch.randn((), device=device) > 0, 10))

    yield SampleInput(make_arg((S,)),
                      args=(torch.randn(S, S, device=device) > 0, make_arg(())),
                      broadcasts_input=True)
    yield SampleInput(make_arg((S,)),
                      args=(torch.randn(S, S, device=device) > 0, 10),
                      broadcasts_input=True)


def sample_inputs_masked_select(op_info, device, dtype, requires_grad, **kwargs):
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

def sample_inputs_matrix_exp(op_info, device, dtype, requires_grad, **kwargs):
    samples = (
        SampleInput(make_tensor((S, S), device, dtype, requires_grad=requires_grad)),
        SampleInput(make_tensor((S, S, S), device, dtype, requires_grad=requires_grad)),
    )

    return samples

def sample_inputs_matmul(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (((L,), (L,)),
                  ((S, M), (M,)),
                  ((M,), (M, S)),
                  ((S, M), (M, S)),
                  ((S, 0), (0, M)),
                  ((S, S, M), (M,)),
                  ((S, S, M), (M, S)),
                  ((S, S, 0), (0, S)),
                  ((M,), (S, M, S)),
                  ((S, M), (S, M, S)),
                  ((0, 0), (S, 0, 0)),
                  ((S, S, M, M), (S, S, M, S)),
                  ((S, S, M, M), (M,)),
                  ((M,), (S, S, M, S)))
    sample_inputs = []
    for lhs_shape, rhs_shape in test_cases:
        lhs = make_tensor(lhs_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        rhs = make_tensor(rhs_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        if op_info.name == 'matmul':
            sample_inputs.append(SampleInput(lhs, args=(rhs,)))
        elif op_info.name == '__rmatmul__':
            sample_inputs.append(SampleInput(rhs, args=(lhs,)))
        else:
            raise RuntimeError("`op_info.name` must be 'matmul' or '__rmatmul__'")
    return tuple(sample_inputs)


def sample_inputs_meshgrid(op_info: OpInfo, device: torch.device, dtype: torch.dtype,
                           requires_grad: bool,
                           *, variant: str) -> List[SampleInput]:
    if variant == 'variadic':
        def make_inputs(
                tensors: List[torch.Tensor]) -> Tuple[Union[torch.Tensor,
                                                            List[torch.Tensor]],
                                                      Tuple[torch.Tensor, ...]]:
            return tensors[0], tuple(tensors[1:])
    elif variant == 'list':
        def make_inputs(
                tensors: List[torch.Tensor]) -> Tuple[Union[torch.Tensor,
                                                            List[torch.Tensor]],
                                                      Tuple[torch.Tensor, ...]]:
            return tensors, ()
    else:
        raise ValueError(
            'Unsupported variant, must be one of {"variadic", "list"}. '
            f'Got "{variant}".')

    SCALAR = torch.Size([])
    VECTOR = torch.Size([3])
    test_cases: List[List[torch.Size]] = [
        [SCALAR],
        [VECTOR],
        [VECTOR, SCALAR],
        [VECTOR, SCALAR, VECTOR],
        [VECTOR, SCALAR, VECTOR, SCALAR],
    ]

    sample_inputs = []
    for shapes, indexing in itertools.product(test_cases, {'xy', 'ij'}):
        input, args = make_inputs(
            [make_tensor(shape, device, dtype, requires_grad=requires_grad)
             for shape in shapes])
        sample_inputs.append(SampleInput(input=input, args=args,
                                         kwargs=dict(indexing=indexing)))
    return sample_inputs


def sample_inputs_polar(op_info, device, dtype, requires_grad, **kwargs):
    def _make_tensor_helper(shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    samples = (
        SampleInput(_make_tensor_helper((S, S), low=0), args=(_make_tensor_helper((S, S)),)),
        SampleInput(_make_tensor_helper((), low=0), args=(_make_tensor_helper(()),)),
    )

    return samples

def sample_inputs_complex(op_info, device, dtype, requires_grad, **kwargs):
    def _make_tensor_helper(shape):
        return make_tensor(shape, device, dtype, requires_grad=requires_grad)

    samples = (
        SampleInput(_make_tensor_helper((S, S)), args=(_make_tensor_helper((S, S)),)),
        SampleInput(_make_tensor_helper(()), args=(_make_tensor_helper(()),)),
    )

    return samples


def sample_inputs_polygamma(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    tensor_shapes = ((S, S), ())
    ns = (1, 2, 3, 4, 5)

    for shape, n in product(tensor_shapes, ns):
        yield SampleInput(make_arg(shape), args=(n,))


def sample_inputs_mvlgamma(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    tensor_shapes = ((S, S), ())
    ns = (1, 2, 3, 4, 5)

    # Since the accepted lower bound for input
    # to mvlgamma depends on `p` argument,
    # the following function computes the lower bound
    # which we pass to `make_tensor`.
    def compute_min_val(p):
        return (p - 1.) / 2

    for shape, n in product(tensor_shapes, ns):
        min_val = compute_min_val(n)
        if not dtype.is_floating_point:
            # Round-up minimum value for integral dtypes
            min_val += 1
        yield SampleInput(make_arg(shape, low=min_val), args=(n,))


# Since `mvlgamma` has multiple entries,
# there are multiple common skips for the additional
# entries. Following function is a helper to that end.
def skips_mvlgamma(skip_redundant=False):
    skips = (
        # outside domain values are hard error for mvlgamma op.
        DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_float_domains'),
    )
    if skip_redundant:
        # Redundant tests
        skips = skips + (  # type: ignore[assignment]
            DecorateInfo(unittest.skip("Skipped!"), 'TestGradients'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestCommon'),
        )
    return skips


# To test reference numerics against multiple values of argument `p`,
# we make multiple OpInfo entries with each entry corresponding to different value of p.
# We run the op tests from test_ops.py only for `p=1` to avoid redundancy in testing.
# Class `MvlGammaInfo` already contains the basic information related to the operator,
# it only takes arguments like `domain`, `skips` and `sample_kwargs`, which
# differ between the entries.
class MvlGammaInfo(UnaryUfuncInfo):
    def __init__(self, variant_test_name, domain, skips, sample_kwargs):
        super(MvlGammaInfo, self).__init__(
            'mvlgamma',
            ref=reference_mvlgamma if TEST_SCIPY else _NOTHING,
            aliases=('special.multigammaln',),
            variant_test_name=variant_test_name,
            domain=domain,
            decorators=(precisionOverride({torch.float16: 5e-2}),),
            dtypes=all_types_and(torch.bfloat16),
            dtypesIfCUDA=all_types_and(torch.half),
            sample_inputs_func=sample_inputs_mvlgamma,
            safe_casts_outputs=True,
            supports_forward_ad=True,
            supports_fwgrad_bwgrad=True,
            skips=skips,
            sample_kwargs=sample_kwargs)


def sample_inputs_entr(op_info, device, dtype, requires_grad, **kwargs):
    low, _ = op_info.domain

    if requires_grad:
        low = 0 + op_info._domain_eps

    return (SampleInput(make_tensor((L,), device, dtype,
                                    low=low,
                                    requires_grad=requires_grad)),
            SampleInput(make_tensor((), device, dtype,
                                    low=low,
                                    requires_grad=requires_grad)))


def sample_inputs_zeta(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = (SampleInput(make_arg((S,), low=1, requires_grad=requires_grad),
                           args=(make_arg((S,), low=2, requires_grad=False),)),
               SampleInput(make_arg((S,), low=1, requires_grad=requires_grad),
                           args=(3.,)),
               )

    return samples


# TODO: Consolidate `i0e` with sample_inputs_unary when `make_tensor`,
#       supports `exclude` argument.
#       For more context: https://github.com/pytorch/pytorch/pull/56352#discussion_r633277617
def sample_inputs_i0_i1(op_info, device, dtype, requires_grad, **kwargs):

    samples = (SampleInput(make_tensor((S,), device, dtype,
                                       requires_grad=requires_grad)),
               SampleInput(make_tensor((), device, dtype,
                                       requires_grad=requires_grad)))

    if requires_grad and op_info.op == torch.special.i0e:
        # NOTE: `i0e`'s first-order gradient is not continous
        # at `0`, hence we don't test `i0e` with any input being `0`.
        # TODO: Remove this when `make_tensor` supports excluding `0`.
        for sample in samples:
            t = sample.input
            t[t == 0] = torch.finfo(dtype).eps  # type: ignore[index]
    elif requires_grad and op_info.op != torch.special.i0e:
        # Special Case for gradient
        # Sample with `0` in the input
        t = make_tensor((S,), device, dtype,
                        requires_grad=requires_grad)
        t[0] = 0

        samples += (SampleInput(t),)  # type: ignore[assignment]

    return samples


def sample_inputs_rsub(op_info, device, dtype, requires_grad, other_scalar, **kwargs):
    make_arg = partial(make_tensor, device=device)

    shapes = ((S, S), (S,), ()) if not other_scalar else ((),)
    # We are doing y - a*x, where y may be a scalar or a tensor
    # If y is a scalar, y may be of any dtype that can be cast to the dtype of x
    # a may always be of any dtype that can be cast to the dtype of x
    if dtype.is_complex:
        dtypes_a = (torch.int32, torch.float32, dtype)
    elif dtype.is_floating_point:
        dtypes_a = (torch.int32, dtype)
    else:
        dtypes_a = (dtype, )
    dtypes_y = dtypes_a if other_scalar else (dtype,)

    for shape_x, shape_y, dtype_y, dtype_a in product(shapes, shapes, dtypes_y, dtypes_a):
        requires_grad_y = (requires_grad and
                           not other_scalar and
                           (dtype_y.is_floating_point or dtype_y.is_complex))

        x = make_arg(shape_x, dtype=dtype, requires_grad=requires_grad)
        y = make_arg(shape_y, dtype=dtype_y, requires_grad=requires_grad_y)
        if other_scalar:
            y = y.item()
        a = make_arg((), dtype=dtype_a).item()
        yield SampleInput(x, args=(y,), kwargs={"alpha": a})

def sample_inputs_cumulative_ops(op_info, device, dtype, requires_grad, supports_dtype_kwargs=True, **kwargs):
    def _make_tensor_helper(shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    samples = [
        SampleInput(_make_tensor_helper((S, S, S)), args=(0,)),
        SampleInput(_make_tensor_helper((S, S, S)), args=(1,)),
        SampleInput(_make_tensor_helper(()), args=(0,)),
    ]

    if supports_dtype_kwargs:
        # NOTE: if `dtype` is not same as input, then inplace variants fail with
        # `provided dtype must match the dtype of self tensor in cumsum`
        samples.append(SampleInput(_make_tensor_helper((S, S, S)), args=(1,), kwargs={'dtype': dtype}))

    return samples


def sample_inputs_unfold(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((), (0, 1, 1)),
        ((S, S, S, S), (0, 3, 1)),
        ((S, S, S, S), (1, 3, 1)),
        ((S, S, S, S), (2, 3, 1)),
        ((S, S, S, S), (3, 3, 1)),
        ((S, S, S, S), (0, 3, 2)),
        ((S, S, S, S), (1, 3, 2)),
        ((S, S, S, S), (2, 3, 2)),
        ((S, S, S, S), (3, 3, 2)),
        ((S, S, S, S), (0, 4, 1)),
        ((S, S, S, S), (1, 4, 1)),
        ((S, S, S, S), (2, 4, 1)),
        ((S, S, S, S), (3, 4, 1)),
        ((M,), (0, 3, 1)),
        ((M,), (0, 3, 2)),
        ((M,), (0, 3, 3)),
        ((1000,), (0, 3, 11)),
        ((1000,), (0, 2, 27)),
        ((10, 10), (0, 1, 2)),
        ((10, 10), (1, 2, 3)),
        ((10, 10), (1, 2, 2)),
        ((S, S, S), (2, 3, 2)),
    )

    sample_inputs = []
    for shape, arguments in test_cases:
        sample_inputs += [SampleInput(make_tensor(shape, device, dtype,
                                      low=None, high=None,
                                      requires_grad=requires_grad),
                                      args=arguments)]
    return sample_inputs


def sample_inputs_atan2(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (
        ((S, S, S), (S, S, S), False),
        ((), (), False),
        ((S, S, S), (S,), False),
        ((S,), (S, S, S), True),
        ((S, 1, S), (S, S), True),
    )

    for x_shape, y_shape, broadcasts_input in cases:
        yield SampleInput(make_arg(x_shape), args=(make_arg(y_shape),),
                          broadcasts_input=broadcasts_input)


def sample_inputs_split(op_info, device, dtype, requires_grad, *, list_args=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    if list_args:
        cases = (
            ((S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],)),
            ((S, S, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)], 2),),
            ((S, S, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)], -2),)
        )
    else:
        cases = (  # type: ignore[assignment]
            ((S, S, S), (2,)),
            ((S, S, S), (S, 1)),
        )

    for shape, args in cases:
        yield SampleInput(make_arg(shape), args=args)


def sample_inputs_split_with_sizes(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = (((S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],)),
             ((S, S, S), ([int(S / 3), S - int(S / 3), 0],)),
             ((S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)], 2)),
             ((S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)], -2)),
             )

    for shape, args in cases:
        yield SampleInput(make_arg(shape), args=args)


def sample_inputs_msort(op_info, device, dtype, requires_grad, **kwargs):
    def apply_grad(t):
        if dtype in floating_types_and(torch.float16, torch.bfloat16):
            t.requires_grad_(requires_grad)

    def large_1d_unique(dtype, device):
        res = torch.randperm(L * L * L, dtype=torch.int64, device=device)
        res = res.to(dtype)
        apply_grad(res)
        return res

    samples = []
    # Test case for large tensor.
    largesample = SampleInput(large_1d_unique(dtype, device))

    sample = SampleInput(make_tensor((S, M, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad))

    return [largesample, sample]

def sample_inputs_lerp(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    samples = (
        # no broadcast
        SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 0.4)),
        # broadcast rhs
        SampleInput(make_arg((S, S)), args=(make_arg((S,)), 0.4)),
        # scalar tensor
        SampleInput(make_arg(()), args=(make_arg(()), 0.4)),
        # broadcast rhs scalar-tensor
        SampleInput(make_arg((S, S)), args=(make_arg(()), 0.4)),
        # broadcast rhs with weight tensor
        SampleInput(make_arg((S, S)), args=(make_arg((S,)), make_arg((S, S)))),
        # broadcast rhs and weight tensor
        SampleInput(make_arg((S, S)), args=(make_arg((S, 1)), make_arg((S,)))),
        # broadcast lhs
        SampleInput(make_arg((S,)), args=(make_arg((S, S)), 0.4), broadcasts_input=True),
        # scalar broadcast_lhs
        SampleInput(make_arg(()), args=(make_arg((S, S)), 0.4), broadcasts_input=True),
        # broadcast all
        SampleInput(make_arg((S, 1)), args=(make_arg((S, S)), 0.4), broadcasts_input=True),
        # tensor broadcast all
        SampleInput(make_arg((S, 1)), args=(make_arg((S, S)), make_arg((S, 1))),
                    broadcasts_input=True),
        # no broadcast with weight tensor
        SampleInput(make_arg((S, S)), args=(make_arg((S, S)), make_arg((S, S)))),
        # broadcast lhs with weight tensor
        SampleInput(make_arg((S,)), args=(make_arg((S, S)), make_arg((S, S))), broadcasts_input=True),
        # broadcast lhs and weight tensor
        SampleInput(make_arg((S,)), args=(make_arg((S, S, S)), make_arg((S, S))), broadcasts_input=True),
        # broadcast lhs and weight tensor variant
        SampleInput(make_arg((S, S)), args=(make_arg((S, S, S)), make_arg((S,))), broadcasts_input=True),
    )

    if dtype.is_complex:
        samples = samples + (  # type: ignore[assignment]
            # no broadcast
            SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 0.4j)),
            SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 1.2 + 0.1j)),
            # broadcast rhs
            SampleInput(make_arg((S, S)), args=(make_arg((S,)), 0.4j)),
            SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 5.4 + 9j)),
            # scalar tensor
            SampleInput(make_arg(()), args=(make_arg(()), 0.4j)),
            SampleInput(make_arg(()), args=(make_arg(()), 6.1 + 0.004j)),
            # broadcast rhs scalar-tensor
            SampleInput(make_arg((S, S)), args=(make_arg(()), 0.4j)),
            SampleInput(make_arg((S, S)), args=(make_arg(()), 1 + 2j)),
        )

    return samples

def sample_inputs_tensordot(self, device, dtype, requires_grad, **kwargs):
    cases = (
        ((2, 2, 2), (2, 2, 2), (2)),
        ((2, 2, 1), (2, 1, 2), ([0, 1], [2, 0])),
    )
    samples = []
    for first_shape, second_shape, dims in cases:
        samples.append(SampleInput(make_tensor(first_shape, device, dtype,
                                   requires_grad=requires_grad),
                       args=(make_tensor(second_shape, device, dtype,
                             requires_grad=requires_grad),),
                       kwargs=dict(dims=dims,)))
    return tuple(samples)

def sample_inputs_kron(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((S, S), (M, L)),
    )

    sample_inputs = []
    for input_shape, other_shape in test_cases:
        input = make_tensor(input_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        other = make_tensor(other_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        sample = SampleInput(input, args=(other,))
        sample_inputs.append(sample)
    return tuple(sample_inputs)

def sample_inputs_inner(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, ), device, dtype, requires_grad=requires_grad),
            args=(
                make_tensor((S, ), device, dtype, requires_grad=requires_grad),
            )
        ),
        SampleInput(
            make_tensor((), device, dtype, requires_grad=requires_grad),
            args=(
                make_tensor((S, S), device, dtype, requires_grad=requires_grad),
            )
        ),
    )

def sample_inputs_scatter(op_info, device, dtype, requires_grad, **kwargs):
    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    def _gather(shape, index_dim, max_indices):
        return gather_variable(shape, index_dim, max_indices, device=device)

    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = (
        (_tensor((M, S)), (0, _gather((S, S), 1, M), _tensor((S, S)))),
        (_tensor((M, S)), (1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (-1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))),
        (_tensor((M, S)), (1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor((M, S)), (-1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor(()), (0, zero.clone().detach(), _tensor(()))),
        (_tensor(()), (0, zero.clone().detach(), 2.5)),
    )

    samples = []
    for tensor, args in test_cases:
        samples.append(SampleInput(tensor, args=args))

        if not requires_grad:
            samples.append(SampleInput(
                tensor.clone().detach(),
                args=args, kwargs={'reduce': 'add'}
            ))

            if dtype.is_floating_point:
                samples.append(SampleInput(
                    tensor.clone().detach(),
                    args=args, kwargs={'reduce': 'multiply'}
                ))

    return samples

def sample_inputs_scatter_add(op_info, device, dtype, requires_grad, **kwargs):
    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    def _gather(shape, index_dim, max_indices):
        return gather_variable(shape, index_dim, max_indices, device=device)

    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = (
        (_tensor((M, S)), (0, _gather((S, S), 1, M), _tensor((S, S)))),
        (_tensor((M, S)), (1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (-1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))),
        (_tensor((M, S)), (1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor((M, S)), (-1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor(()), (0, zero.clone().detach(), _tensor(()))),
    )

    return [SampleInput(tensor, args=args) for tensor, args in test_cases]

def sample_inputs_scatter_reduce(op_info, device, dtype, requires_grad):
    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    def _index(shape, max_index):
        return torch.from_numpy(np.random.choice(max_index, size=shape)).to(dtype=torch.int64, device=device)

    reduces = ["sum", "prod", "mean", "amax", "amin"]
    shapes_and_dims = [((M,), 1), ((M, S), 2), ((M, M, S), 3), ((1, M, M, S), 4)]

    sample_inputs = []

    for ((shape, dim), reduce) in itertools.product(shapes_and_dims, reduces):
        for d in range(dim):
            # Generate a random maximum integer that can appear in index array
            max_index = np.random.randint(1, shape[d] * 2)
            index = _index(shape, max_index)
            sample_inputs.append(
                SampleInput(
                    _tensor(shape),
                    args=(d, index, reduce),
                )
            )

    return sample_inputs

def sample_inputs_ravel(op_info, device, dtype, requires_grad, **kwargs):
    samples = (SampleInput(make_tensor((S, S, S), device, dtype,
                                       low=None, high=None,
                                       requires_grad=requires_grad)),
               SampleInput(make_tensor((), device, dtype,
                                       low=None, high=None,
                                       requires_grad=requires_grad)),)

    return samples


def sample_inputs_tril_triu(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((M, M), ()),
             ((M, M), (2,),),
             ((S, M, M), ()),
             ((S, M, M), (2,)),
             ((3, 3, S, S), ()),)

    for shape, args in cases:
        yield SampleInput(make_arg(shape), args=args)


def sample_inputs_clone(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    yield SampleInput(make_arg((S, M, S)))
    yield SampleInput(make_arg(()))


def sample_inputs_contiguous(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S)))


def sample_inputs_sum_to_size(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    # list of tuples (shape, shape) defining the shapes of the input and output tensors
    sample_shapes = [
        ((), ()),
        ((S), (1)),
        ((S, S), (1, 1)),
        ((S, S), (1, S)),
        ((S, S), (S, S)),
        ((S, S, S), (S, 1, S)),
    ]

    samples = []

    for input_shape, output_shape in sample_shapes:
        input_t = make_arg(input_shape)
        samples.append(SampleInput(input_t, args=(output_shape,)))

    return samples

def sample_inputs_resize_ops(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device)
    cases = (((S, S, S), (S * S, S)),
             ((), ()),
             ((), (1, 1, 1)),
             )

    for shape, args_or_shape in cases:
        # Update `args` based on operator
        if op_info.name == 'resize_':
            # resize_ takes shape/tuple of ints,
            args = (args_or_shape, )
        elif op_info.name == 'resize_as_':
            # resize_as_ takes another tensor
            args = (make_arg(shape, requires_grad=False), )  # type:ignore[assignment]
        else:
            raise ValueError("sample_inputs_resize_ops is being used with incorrect operator")

        yield(SampleInput(make_arg(shape, requires_grad=requires_grad), args=args))

def sample_inputs_view_reshape(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((S, S, S), (S * S, S)),
             ((S * S, S), (S, S, S)),
             ((S * S, S), (S, -1, S)),
             ((S * S * 2, S), (S, -1)),
             ((S,), (S,)),
             ((), ()),
             ((), (1,)))

    for case in cases:
        shape, args = case
        inp = make_arg(shape, requires_grad=requires_grad)
        yield(SampleInput(inp, args=(args, )))

        if op_info.name != "view" and len(shape) >= 2:
            yield(SampleInput(
                inp.clone().transpose(0, 1).requires_grad_(requires_grad),
                args=(args, )))

def sample_inputs_view_as_reshape_as(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device)

    cases = (((S, S, S), (S * S, S)),
             ((), ()),
             ((), (1, 1)),
             )

    for case in cases:
        shape, shape_other = case
        inp = make_arg(shape, requires_grad=requires_grad)
        yield(SampleInput(inp, args=(make_arg(shape_other, requires_grad=False),)))

        if op_info.name != "view_as" and len(shape) >= 2:
            yield(SampleInput(
                inp.clone().transpose(0, 1).requires_grad_(requires_grad),
                args=(make_arg(shape_other, requires_grad=False),)))

def sample_inputs_atleast1d2d3d(op_info, device, dtype, requires_grad, **kwargs):
    input_list = []
    shapes = ((S, S, S, S), (S, S, S), (S, S), (S, ), (),)
    make_tensor_partial = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    samples = []
    for shape in shapes:
        input_list.append(make_tensor_partial(shape))
        samples.append(SampleInput(make_tensor_partial(shape)))
    samples.append(SampleInput(input_list, ))
    return samples

def sample_inputs_column_stack(op_info, device, dtype, requires_grad, **kwargs):
    input_list = []
    cases: Tuple[tuple, tuple] = (  # type: ignore[assignment]
        ((S, 2, 1), (S, 3, 1)),
        ((S), (S, 5)), ((), (1, S))
    )
    make_tensor_partial = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape1, shape2 in cases:
        input_list.append(SampleInput([make_tensor_partial(shape1), make_tensor_partial(shape2)]))

    return input_list

def sample_inputs_flatten(op_info, device, dtype, requires_grad, **kwargs):
    samples = []
    shapes = ((S, S, S), (S, S), (S, ), (),)
    make_tensor_partial = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape in shapes:
        samples.append(SampleInput(make_tensor_partial(shape)))
        if len(shape) > 1:
            samples.append(SampleInput(make_tensor_partial(shape), kwargs=dict(start_dim=1, end_dim=-1)))
    return samples

def sample_inputs_select(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((S, S, S), (1, 2)),
             ((S, S, S), (-1, 2)),
             ((S, S, S), (-1, -1)),
             ((S, S, S), (1, -1)),
             ((S,), (0, 2))
             )

    for shape, args in cases:
        yield SampleInput(make_arg(shape), args=args)


def sample_inputs_select_scatter(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((S, S, S), (S, S), (1, 2)),
             ((S, S, S), (S, S), (-1, 2)),
             ((S, S, S), (S, S), (-1, -1)),
             ((S, S, S), (S, S), (1, -1)),
             ((S,), (), (0, 2))
             )

    for input_shape, src_shape, args in cases:
        input_ = make_arg(input_shape)
        src = make_arg(src_shape)
        yield SampleInput(input_, args=(src, *args))


def sample_inputs_slice_scatter(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((L, L, L), (L, L, L,), (0, 0, L, 1)),
             ((L, L, L), (L // 2, L, L,), (0, L // 2, L, 1)),
             ((L, L, L), (L // 4, L, L,), (0, L // 2, L, 2)),
             ((L, L, L), (L, L, L,), (1, 0, L, 1)),
             ((L, L, L), (L, L // 2, L,), (1, L // 2, L, 1)),
             ((L, L, L), (L, L // 4, L,), (1, L // 2, L, 2)),
             ((L, L, L), (L, L, L,), (2, 0, L, 1)),
             ((L, L, L), (L, L, L // 2,), (2, L // 2, L, 1)),
             ((L, L, L), (L, L, L // 4,), (2, L // 2, L, 2)),
             )

    for input_shape, src_shape, args in cases:
        input_ = make_arg(input_shape)
        src = make_arg(src_shape)
        yield SampleInput(input_, args=(src, *args))


def sample_inputs_rbinops(op_info, device, dtype, requires_grad, supports_dtype_kwargs=True, **kwargs):
    def _make_tensor_helper(shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    scalar: Union[int, float, complex] = 3

    if dtype.is_floating_point:
        scalar = 3.14
    elif dtype.is_complex:
        scalar = 3.14j

    samples = [
        SampleInput(_make_tensor_helper((S, S, S)), args=(scalar,)),
        SampleInput(_make_tensor_helper(()), args=(scalar,)),
    ]

    return samples


def sample_inputs_expand(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((S, 1, 1), (S, S, S)),
             ((S, 1, S), (S, S, S)),
             ((S, 1, S), (-1, S, -1)),
             ((S, 1, S), (-1, S, S)),
             ((S, 1), (S, S, S)),
             ((1,), (S, S, S)),
             ((1, S), (1, 1, S)),
             ((), ()),
             ((), (1, 3, 2)),
             )

    for case in cases:
        shape, args = case
        yield(SampleInput(make_arg(shape), args=(args, )))

def sample_inputs_conversion(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    shapes = ((),
              (2, 3))
    memory_format_options = [None, torch.contiguous_format]

    for shape, memory_format in itertools.product(shapes, memory_format_options):
        yield SampleInput(make_arg(shape),
                          kwargs={'memory_format': memory_format} if memory_format else {})

def sample_inputs_conversion_channels_last(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    return [
        # Channels last case: input must be 4d
        SampleInput(make_arg((2, 3, 2, 3)), kwargs={'memory_format': torch.channels_last})

    ]

def sample_inputs_expand_as(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device)

    cases = (((S, 1, 1), (S, S, S)),
             ((), ()),
             ((), (1, 1)),
             )

    for shape, shape_other in cases:
        yield(SampleInput(make_arg(shape, requires_grad=requires_grad),
                          args=(make_arg(shape_other, requires_grad=False), )))


def sample_inputs_where(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    def make_bool_mask(shape):
        # Make sure atleast one element is nonzero,
        # except for empty tensor
        mask_t = make_tensor(shape, dtype=torch.bool, device=device, requires_grad=False)

        if mask_t.numel() == 0:
            return mask_t
        elif mask_t.numel() == 1:
            mask_t.fill_(True)
            return mask_t

        if mask_t.sum() == 0:
            def random_index(shape):
                return tuple(map(lambda max_idx: random.randint(0, max_idx), shape))

            mask_t[random_index(mask_t.shape)] = True
            return mask_t

        return mask_t

    cases = (((M, M), (M, M), (M, M), False),
             ((M, 1, M), (M, M), (M, M, 1), True),
             ((), (), (), False),
             ((M, 1, M), (), (M, M, 1), True),
             ((), (M, M), (), True),)

    for shape, mask_shape, other_shape, broadcasts_input in cases:
        yield SampleInput(make_arg(shape),
                          args=(make_bool_mask(mask_shape), make_arg(other_shape)),
                          broadcasts_input=broadcasts_input)

def sample_inputs_nonzero(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))

    inputs = []
    for shape in sizes:
        # construct input without any non-zero elements
        zeros = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        inputs.append(zeros)

        # construct input with mixed zero and non-zero elements
        mixed = make_arg(shape).requires_grad_(False)
        mask_t = make_tensor(shape, dtype=torch.bool, device=device, requires_grad=False)
        mixed[mask_t] = 0
        inputs.append(mixed)

    for input_t, as_tuple in product(inputs, [False, True]):
        yield(SampleInput(input_t.clone().requires_grad_(requires_grad),
                          kwargs=dict(as_tuple=as_tuple)))

def sample_inputs_chunk(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device)

    cases = (((S, S, S), (2,)),
             ((S, S, S), (S, 1)),
             ((S, S, S), (S, -1)))

    for case in cases:
        shape, args = case
        yield(SampleInput(make_arg(shape, requires_grad=requires_grad), args=args))

def sample_inputs_kthvalue(op_info, device, dtype, requires_grad, **kwargs):
    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    test_cases = [
        (_tensor((S, S, S)), (2,)),
        (_tensor((S, S, S)), (2, 1,)),
        (_tensor((S, S, S)), (2, -1,)),
        (_tensor((S, S, S)), (2, 1, True,)),
        (_tensor((S, S, S)), (2, -1, True,)),
        (_tensor((S,)), (2, 0,)),
        (_tensor((S,)), (2, 0, True,)),
        (_tensor(()), (1,)),
        (_tensor(()), (1, 0,)),
        (_tensor(()), (1, 0, True))
    ]

    return [SampleInput(tensor, args=args) for tensor, args in test_cases]

def error_inputs_kthvalue(op_info, device, **kwargs):
    # tests overlapping output fails
    t = make_tensor(10, dtype=torch.float32, device=device)
    indices = torch.empty((), device=device, dtype=torch.long)
    si = SampleInput(t, args=(5,), kwargs={'out': (t, indices)})

    k_out_of_range_err = "selected number k out of range for dimension"
    return (ErrorInput(si, error_type=RuntimeError, error_regex="unsupported operation"),
            ErrorInput(SampleInput(torch.randn(2, 2, device=device), args=(3, 0)),
                       error_type=RuntimeError, error_regex=k_out_of_range_err),
            ErrorInput(SampleInput(torch.randn(2, 2, device=device), args=(3,)),
                       error_type=RuntimeError, error_regex=k_out_of_range_err),
            ErrorInput(SampleInput(torch.tensor(2, device=device), args=(3,)),
                       error_type=RuntimeError, error_regex=k_out_of_range_err),)

def sample_inputs_dropout(op_info, device, dtype, requires_grad, *,
                          train=None, valid_input_dim=None, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    if valid_input_dim:
        cases = ((S,) * i for i in valid_input_dim)
    else:
        cases = ((S, S), (S,), ())
    p_vals = [0.0, 0.5, 1.0]
    # This is to handle special case for feature_alpha_dropout which has different
    # supported dtypes depending on `train` parameter
    training_vals = [train] if train is not None else [True, False]

    for case, p, training in product(cases, p_vals, training_vals):
        yield SampleInput(make_arg(case), kwargs=dict(p=p, training=training))
    yield SampleInput(make_arg(case), kwargs=dict())


def sample_inputs_embedding_bag(op_info, device, dtype, requires_grad, **kwargs):
    def make_input(shape):
        return make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_long_input(shape, *, low, high, noncontiguous=False):
        return make_tensor(shape, device=device, dtype=torch.long, low=low, high=high,
                           noncontiguous=noncontiguous)

    def make_per_sample_weight(flag, idx):
        # a tensor of float / double weights, or None
        # to indicate all weights should be taken to be 1
        if flag:
            return make_input(idx.shape)
        return None

    offsets = torch.tensor([0, 3], device=device, dtype=torch.long)
    for generate_per_sample_weight in (True, False):
        for mode in ('sum', 'mean', 'max'):
            # per_sample_weights is only supported for mode='sum' (got mode='****')
            if generate_per_sample_weight and mode in ('mean', 'max'):
                continue

            # 1-D index tensor
            idx = make_long_input((S,), low=0, high=M)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((M, S)), args=(idx,),
                              kwargs={'offsets': offsets, 'mode': mode,
                                      'per_sample_weights': per_sample_weights})

            idx = make_long_input((S,), low=0, high=M, noncontiguous=True)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((M, S)), args=(idx,),
                              kwargs={'offsets': offsets, 'mode': mode,
                                      'per_sample_weights': per_sample_weights})

            # bag with zero length
            idx = make_long_input((S,), low=0, high=M, noncontiguous=True)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((M, S)), args=(idx,),
                              kwargs={'offsets': torch.tensor([0, 0, 3], device=device, dtype=torch.long),
                                      'mode': mode,
                                      'per_sample_weights': per_sample_weights})

            # 2-D index tensor
            idx = make_long_input((S, S), low=0, high=M)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((M, S)), args=(idx,),
                              kwargs={'mode': mode, 'per_sample_weights': per_sample_weights})

            idx = make_long_input((S, S), low=0, high=M, noncontiguous=True)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((M, S)), args=(idx,),
                              kwargs={'mode': mode, 'per_sample_weights': per_sample_weights})

            # The gradient vector at `padding_idx` is not updated.
            # Negative padding_idx
            idx = make_long_input((6,), low=0, high=S)
            idx[0] = 4
            idx[4] = 4
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((S, S)), args=(idx,),
                              kwargs={'padding_idx': -1, 'offsets': offsets,
                                      'mode': mode, 'per_sample_weights': per_sample_weights},)

            idx = make_long_input((3, 3), low=0, high=S)
            # Positive padding_idx
            idx[0, 0] = 2
            idx[1, 1] = 2
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((S, S)), args=(idx,),
                              kwargs={'padding_idx': 2, 'mode': mode,
                                      'per_sample_weights': per_sample_weights},)

            idx = make_long_input((6, ), low=0, high=S)
            weights = make_input((S, S))
            offsets_ = torch.tensor([0, 3, 6], device=device, dtype=torch.long)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(weights, args=(idx,),
                              kwargs={'mode': mode, 'offsets': offsets_, 'include_last_offset': True},)

            if not requires_grad:
                # Following inputs return different gradient from the numerical gradient.
                # This is expected and relevant tests are present in `test_nn.py`.

                # Due to inplace renorming of weight, the numerical gradient doesn't match the
                # analytical gradient.
                idx = make_long_input((2, 2), low=0, high=S)
                weights = make_input((S, S)) * 2
                per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
                yield SampleInput(weights, args=(idx,),
                                  kwargs={'max_norm': 1., 'mode': mode,
                                          'per_sample_weights': per_sample_weights},)

                idx = make_long_input((6, ), low=0, high=S)
                weights = make_input((S, S)) * 2
                per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
                yield SampleInput(weights, args=(idx,),
                                  kwargs={'max_norm': 1., 'norm_type': 1.0,
                                          'mode': mode, 'offsets': offsets,
                                          'per_sample_weights': per_sample_weights},)

                if mode != 'max':
                    # Scale the gradient based on the inverse frequency of a particular index.
                    # Note : smax mode does not support sparse weights
                    idx = make_long_input((2, 2), low=0, high=S)
                    idx[0, 0] = 1
                    idx[0, 1] = 1
                    weights = make_input((S, S))
                    per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
                    yield SampleInput(weights, args=(idx,),
                                      kwargs={'scale_grad_by_freq': True, 'mode': mode,
                                              'per_sample_weights': per_sample_weights},)

                    # gradcheck not implemented for sparse tensors.
                    # Note : max mode does not support sparse weights
                    idx = make_long_input((6, ), low=0, high=S)
                    weights = make_input((S, S))
                    per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
                    yield SampleInput(weights, args=(idx,),
                                      kwargs={'sparse': True, 'offsets': offsets,
                                              'mode': mode, 'per_sample_weights': per_sample_weights})

                    idx = make_long_input((6, ), low=0, high=S)
                    idx[0] = 1  # freq more than 1
                    idx[1] = 1  # freq more than 1
                    idx[3] = 0  # padding_idx
                    weights = make_input((S, S)) * 2
                    per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
                    yield SampleInput(weights, args=(idx,),
                                      kwargs={'sparse': True, 'scale_grad_by_freq': True, 'padding_idx': 0,
                                              'max_norm': 1., 'offsets': offsets,
                                              'mode': mode, 'per_sample_weights': per_sample_weights})


def sample_inputs_embedding(op_info, device, dtype, requires_grad, **kwargs):
    def make_input(shape):
        return make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_long_input(shape, *, low, high):
        return make_tensor(shape, device=device, dtype=torch.long, low=low, high=high)

    # 0-D index tensor
    idx = make_long_input((), low=0, high=M)
    yield SampleInput(make_input((M, S)), args=(idx,),)

    # 1-D index tensor
    idx = make_long_input((S,), low=0, high=M)
    yield SampleInput(make_input((M, S)), args=(idx,),)

    # 2-D index tensor
    idx = make_long_input((S, S), low=0, high=M)
    yield SampleInput(make_input((M, S)), args=(idx,),)

    if not requires_grad:
        # Following inputs return different gradient from the numerical gradient.
        # This is expected and relevant tests are present in `test_nn.py`.

        # The gradient vector at `padding_idx` is not updated.
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 2
        idx[1, 1] = 2
        yield SampleInput(make_input((S, S)), args=(idx,), kwargs={'padding_idx': 2},)

        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 4
        idx[1, 1] = 4
        yield SampleInput(make_input((S, S)), args=(idx,), kwargs={'padding_idx': -1},)

        # Due to inplace renorming of weight, the numerical gradient doesn't match the
        # analytical gradient.
        idx = make_long_input((2, 2), low=0, high=S)
        weights = make_input((S, S)) * 2
        yield SampleInput(weights, args=(idx,), kwargs={'max_norm': 1.},)

        idx = make_long_input((2, 2), low=0, high=S)
        weights = make_input((S, S)) * 2
        yield SampleInput(weights, args=(idx,), kwargs={'max_norm': 1., 'norm_type': 1.0},)

        # Scale the gradient based on the inverse frequency of a particular index.
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 1
        idx[0, 1] = 1
        weights = make_input((S, S))
        yield SampleInput(weights, args=(idx,), kwargs={'scale_grad_by_freq': True},)

        # gradcheck not implemented for sparse tensors.
        idx = make_long_input((2, 2), low=0, high=S)
        weights = make_input((S, S))
        yield SampleInput(weights, args=(idx,), kwargs={'sparse': True})

        idx = make_long_input((3, 3), low=0, high=S)
        idx[0, 0] = 1  # freq more than 1
        idx[0, 1] = 1  # freq more than 1
        idx[1, 0] = 0  # padding_idx
        weights = make_input((S, S)) * 2
        yield SampleInput(weights, args=(idx,),
                          kwargs={'sparse': True, 'scale_grad_by_freq': True,
                                  'padding_idx': 0, 'max_norm': 1.})


def sample_inputs_one_hot(op_info, device, dtype, requires_grad, **kwargs):
    def make_input(shape, *, low, high):
        return make_tensor(shape, device=device, dtype=dtype, low=low, high=high, requires_grad=requires_grad)

    shapes = ((), (S,), (L, M, S))
    num_classess = (-1, 10)

    return [
        SampleInput(
            make_input(
                shape,
                low=0,
                high=10 if num_classes == -1 else num_classes // 2,
            ),
            kwargs=dict(num_classes=num_classes),
        )
        for shape, num_classes in itertools.product(shapes, num_classess)
    ]

def sample_inputs_softplus(op_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, (S,), device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        SampleInput(make_input()),
        SampleInput(make_input(), kwargs=dict(beta=3)),
        SampleInput(make_input(low=1), kwargs=dict(threshold=1)),
    ]

def sample_inputs_tensorinv(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = make_fullrank_matrices_with_distinct_singular_values

    def make_input():
        return make_arg(12, 12, device=device, dtype=dtype, requires_grad=requires_grad)

    # lhs / rhs shape can have any number of dimensions as long as their product equals 12
    shapes = [
        ((2, 2, 3), (12, 1)),
        ((4, 3), (6, 1, 2)),
    ]

    samples = []
    for shape_lhs, shape_rhs in shapes:
        inp = make_input().reshape(*shape_lhs, *shape_rhs).detach()
        inp.requires_grad_(requires_grad)
        samples.append(SampleInput(inp, kwargs=dict(ind=len(shape_lhs))))

    return samples

def sample_inputs_tensorsolve(op_info, device, dtype, requires_grad, **kwargs):
    a_shapes = [(2, 3, 6), (3, 4, 4, 3)]
    # Zero-dim tensors are not supported in NumPy, so we skip them for now.
    # NumPy is used in reference check tests.
    # See https://github.com/numpy/numpy/pull/20482 for tracking NumPy bugfix.
    # a_shapes += [(0, 0, 1, 2, 3, 0)]
    dimss = [None, (0, 2)]

    for a_shape, dims in itertools.product(a_shapes, dimss):
        a = make_tensor(a_shape, dtype=dtype, device=device, requires_grad=requires_grad)
        b = make_tensor(a_shape[:2], dtype=dtype, device=device, requires_grad=requires_grad)
        yield SampleInput(a, args=(b,), kwargs=dict(dims=dims))

def sample_inputs_mse_loss(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    shapes_and_kwargs = [
        ((), None),
        ((S,), dict(reduction="mean")),
        ((S,), dict(reduction="sum")),
        ((S,), dict(reduction="none")),
        ((S, S), None),
        ((S, S, S), None),
    ]

    return [
        SampleInput(_make_tensor(shape), args=(_make_tensor(shape),), kwargs=kwargs)
        for shape, kwargs in shapes_and_kwargs
    ]

def sample_inputs_grid_sample(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    batch_size = 2
    num_channels = 3
    modes = ("bilinear", "nearest")
    align_cornerss = (False, True)
    padding_modes = ("zeros", "border", "reflection")

    sample_inputs = []
    for dim in (2, 3):

        modes_ = (*modes, "bicubic") if dim == 2 else modes

        for mode, padding_mode, align_corners in itertools.product(modes_, padding_modes, align_cornerss):
            sample_inputs.append(
                SampleInput(
                    _make_tensor((batch_size, num_channels, *[S] * dim)),
                    args=(_make_tensor((batch_size, *[S] * dim, dim)),),
                    kwargs=dict(
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    )
                )
            )

    return sample_inputs

def sample_inputs_cosine_embedding_loss(op_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_target(shape):
        shape = () if len(shape) == 1 else (shape[0], )
        t = torch.randint(0, 2, shape, device=device, dtype=torch.long)
        # Label with -1 or 1
        t = t * 2 - 1
        target = t.to(dtype=dtype).detach_().requires_grad_(requires_grad)
        return target

    shapes = ((S, S), (S,))
    reductions = ('none', 'mean', 'sum')
    for s, r in product(shapes, reductions):
        yield SampleInput(
            make_input(s),
            args=(make_input(s), make_target(s)),
            kwargs=dict(reduction=r, margin=random.uniform(-1, 1))
        )

def sample_inputs_ctc_loss(op_info, device, dtype, requires_grad, **kwargs):
    input_length = 50
    batch = 16
    num_char = 20
    target_length = 30

    def make_log_probs(s):
        t = make_tensor(s, device=device, dtype=dtype)
        log_probs = t.log_softmax(2).to(device=device, dtype=dtype).detach().requires_grad_(requires_grad=requires_grad)
        return log_probs

    reductions = ('none', 'mean', 'sum')
    zero_inf = (True, False)
    for r, z in product(reductions, zero_inf):
        log_probs = make_log_probs((input_length, batch, num_char))
        targets = torch.randint(1, num_char, (batch, target_length), dtype=torch.long, device=device)
        input_lengths = torch.full((batch, ), input_length, dtype=torch.long, device=device)
        target_lengths = torch.randint(10, target_length, (batch, ), dtype=torch.long, device=device)

        yield SampleInput(log_probs, args=(targets, input_lengths, target_lengths,), kwargs=dict(reduction=r, zero_infinity=z))

def sample_inputs_nll_loss(op_info, device, dtype, requires_grad, **kwargs):
    shape = (2, 3)
    num_classes = shape[1]
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # FIXME: Derivative wrt. weight not implemented
    make_weight = partial(make_tensor, shape=(num_classes,), device=device, dtype=dtype, requires_grad=False)

    def make_target(shape, zeros=False):
        s = (shape[0], *shape[2:]) if len(shape) > 1 else ()
        if zeros:
            return torch.zeros(s, device=device, dtype=torch.long)
        else:
            return make_tensor(s,
                               low=0,
                               high=shape[1] if len(shape) > 1 else shape[0],
                               device=device,
                               dtype=torch.long)


    def gen_shape_kwargs():
        # Batched, non-batched and 2d
        shapes = (shape, (num_classes,), shape + (2, 2))
        reductions = ('none', 'mean', 'sum')
        for reduction, s in product(reductions, shapes):
            yield make_input(s), make_target(s), dict(reduction=reduction)
            yield make_input(s), make_target(s), dict(weight=make_weight(), reduction=reduction)
            yield make_input(s), make_target(s), dict(weight=make_weight(low=0), reduction=reduction)
            yield make_input(s), make_target(s), dict(weight=make_weight(high=0), reduction=reduction)
            t = make_target(s)
            ignore = num_classes // 2
            # If "mean", nll returns NaN, so it's not differentiable at those points
            if t.eq(ignore).all() and reduction == "mean":
                t.fill_(0)
            yield make_input(s), t, dict(ignore_index=num_classes // 2, reduction=reduction)
            # Test ignoring all the targets
            # If "mean", nll returns NaN, so it's not differentiable at those points
            if reduction != "mean":
                yield make_input(s), make_target(s, zeros=True), dict(ignore_index=0, reduction=reduction)

    for input, target, kwargs in gen_shape_kwargs():
        yield SampleInput(input, args=(target,), kwargs=kwargs)

def sample_inputs_argwhere(op_info, device, dtype, requires_grad, **kwargs):
    yield SampleInput(torch.tensor([1, 0, 2, 0], dtype=dtype, device=device, requires_grad=requires_grad))
    mask = torch.tensor([[0, 1, 0, 1, 0],
                         [1, 1, 1, 1, 0],
                         [0, 0, 0, 1, 0],
                         [1, 0, 1, 1, 0],
                         [1, 0, 0, 1, 0]], dtype=torch.bool, device=device)
    t = make_tensor((S, S), dtype=dtype, device=device, requires_grad=requires_grad)
    t[mask] = 0
    yield SampleInput(t)

    t = make_tensor((S, S), dtype=dtype, device=device, requires_grad=requires_grad, noncontiguous=True)
    t[mask] = 0
    yield SampleInput(t)

    t = make_tensor((S, 0), dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(t)

    yield SampleInput(torch.zeros((S,), dtype=dtype, device=device, requires_grad=requires_grad))
    yield SampleInput(make_tensor((), dtype=dtype, device=device, requires_grad=requires_grad))

def _generate_sample_shape_reduction():
    shapes = ((S,), (S, S), (S, S, S))
    reductions = ('none', 'mean', 'sum')
    for s, r in product(shapes, reductions):
        yield s, r

def sample_inputs_gaussian_nll_loss(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # Set low slightly above 0 so gradcheck doesn't accidentally dip below 0
    make_var = partial(make_tensor, low=0.1, device=device, dtype=dtype, requires_grad=requires_grad)

    def gen_shape(shape):
        yield shape
        # Broadcast
        yield (*shape[:-1], 1)
        yield shape[:-1]

    def gen_shape_kwargs():
        for s, r in _generate_sample_shape_reduction():
            for t_s, v_s in product(gen_shape(s), gen_shape(s)):
                yield _make_tensor(s), _make_tensor(t_s), make_var(v_s), dict(reduction=r)
                yield (
                    _make_tensor(s), _make_tensor(t_s), make_var(v_s),
                    dict(full=True, reduction=r)
                )
                yield (
                    _make_tensor(s), _make_tensor(t_s), make_var(v_s),
                    dict(eps=random.uniform(1e-6, 1e-3), reduction=r)
                )
                yield (
                    _make_tensor(s), _make_tensor(t_s), make_var(v_s),
                    dict(full=True, eps=random.uniform(1e-6, 1e-3), reduction=r)
                )

    for input, target, var, kwargs in gen_shape_kwargs():
        yield SampleInput(input, args=(target, var, ), kwargs=kwargs)

def _generate_sample_inputs_nn_loss(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    for s, r in _generate_sample_shape_reduction():
        yield _make_tensor(s), _make_tensor(s), dict(reduction=r)

def sample_inputs_hinge_embedding_loss(op_info, device, dtype, requires_grad, **kwargs):
    for input, target, d in _generate_sample_inputs_nn_loss(op_info, device, dtype, requires_grad, **kwargs):
        d['margin'] = random.uniform(-9, 9)
        yield SampleInput(input, args=(target, ), kwargs=d)

def sample_inputs_huber_loss(op_info, device, dtype, requires_grad, **kwargs):
    for input, target, d in _generate_sample_inputs_nn_loss(op_info, device, dtype, requires_grad, **kwargs):
        d['delta'] = random.uniform(1e-3, 9)
        yield SampleInput(input, args=(target, ), kwargs=d)

def sample_inputs_poisson_nll_loss(op_info, device, dtype, requires_grad, **kwargs):
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def gen_shape_kwargs():
        for s, r in _generate_sample_shape_reduction():
            for li in (True, False):
                for f in (True, False):
                    i1 = _make_tensor(s)
                    i2 = _make_tensor(s)
                    # For Poisson NLL Loss,
                    # target is assumed to be from
                    # Poisson Distribution which
                    # always has positive samples
                    t1 = _make_tensor(s, low=0)
                    t2 = _make_tensor(s, low=0)

                    with torch.no_grad():
                        if not li:
                            i1.abs_()
                            i2.abs_()
                        t1.abs_()
                        t2.abs_()

                    yield (
                        i1, t1,
                        dict(log_input=li, full=f, reduction=r)
                    )
                    yield (
                        i2, t2,
                        dict(log_input=li, full=f,
                             eps=random.uniform(1e-8, 1e-3),
                             reduction=r)
                    )

    for input, target, kwargs in gen_shape_kwargs():
        yield SampleInput(input, args=(target, ), kwargs=kwargs)

def sample_inputs_pairwise_distance(op_info, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    shape = (3,)
    batched_shape = (2, *shape)
    shapes_and_kwargs = [
        (shape, None),
        (batched_shape, None),
        (shape, dict(keepdim=True)),
        (batched_shape, dict(keepdim=True)),
        (shape, dict(p=5.0)),
        (shape, dict(p=-1.0)),
        (shape, dict(eps=1.0)),
    ]

    return [
        SampleInput(make(shape), args=(make(shape),), kwargs=kwargs) for shape, kwargs in shapes_and_kwargs
    ]

def sample_inputs_pixel_shuffle(op_info, device, dtype, requires_grad, **kwargs):
    return [
        SampleInput(
            make_tensor((1, 9, 2, 2), device=device, dtype=dtype, requires_grad=requires_grad),
            kwargs=dict(upscale_factor=upscale_factor),
        )
        for upscale_factor in (1, 3)
    ]

def sample_inputs_pixel_unshuffle(op_info, device, dtype, requires_grad, **kwargs):
    return [
        SampleInput(
            make_tensor((1, 1, 6, 6), device=device, dtype=dtype, requires_grad=requires_grad),
            kwargs=dict(downscale_factor=downscale_factor),
        )
        for downscale_factor in (1, 3)
    ]

def sample_inputs_allclose(op_info, device, dtype, requires_grad, **kwargs):
    samples = []
    sample_shapes = [(), (S), (S, S, S)]
    atols = [1e-2, 1e-16]
    rtols = [1e-1, 0.5]
    eps = 1e-8
    for s, rtol, atol in product(sample_shapes, rtols, atols):
        # close sample
        t = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)
        close = (t + atol).detach().requires_grad_(requires_grad)
        close_sample = SampleInput(t, args=(close,), kwargs=dict(rtol=rtol, atol=atol))
        samples.append(close_sample)

        # random sample
        a = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)
        b = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)
        r_sample = SampleInput(a, args=(b,), kwargs=dict(rtol=rtol, atol=atol))
        samples.append(r_sample)

    return samples


def sample_inputs_kl_div(op_info, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    shapes_and_reduction = [
        ((2,), "mean"),
        ((2, 3), "mean"),
        ((2, 3, 4), "mean"),
        ((2,), "none"),
        ((2,), "batchmean"),
        ((2,), "sum"),
    ]

    sample_inputs = []
    for (shape, reduction), log_target in itertools.product(shapes_and_reduction, (True, False)):
        # input should be log-probability, i.e. lie in (-inf, 0]
        input = make(shape, low=None, high=0)
        # target should be a probability by default, i.e. lie in [0, 1], and a log-probability if log_target is set,
        # i.e. lie in (-inf, 0]
        target = make(shape, low=None, high=0) if log_target else make(shape, low=0, high=1)
        sample_inputs.append(
            SampleInput(input, args=(target,), kwargs=dict(reduction=reduction, log_target=log_target))
        )
    return sample_inputs

def sample_inputs_diagflat(op_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        SampleInput(make_input(())),
        SampleInput(make_input((2,))),
        SampleInput(make_input((2, 2))),
        SampleInput(make_input((2,)), kwargs=dict(offset=1)),
        SampleInput(make_input((2,)), kwargs=dict(offset=-1)),
    ]


foreach_unary_op_db: List[OpInfo] = [
    ForeachFuncInfo('exp'),
    ForeachFuncInfo('acos'),
    ForeachFuncInfo('asin'),
    ForeachFuncInfo('atan'),
    ForeachFuncInfo('cos'),
    ForeachFuncInfo('cosh'),
    ForeachFuncInfo('log'),
    ForeachFuncInfo('log10'),
    ForeachFuncInfo('log2'),
    ForeachFuncInfo('tan'),
    ForeachFuncInfo('tanh'),
    ForeachFuncInfo('sin'),
    ForeachFuncInfo('sinh'),

    ForeachFuncInfo(
        'neg',
        dtypes=all_types_and_complex(),
        dtypesIfCUDA=all_types_and_complex(),
        sample_inputs_func=sample_inputs_foreach,
        safe_casts_outputs=False,
    ),

    ForeachFuncInfo(
        'sqrt',
        dtypes=floating_and_complex_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_and_complex_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'ceil',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'erf',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'erfc',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'expm1',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'floor',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'log1p',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'round',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'frac',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'reciprocal',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'sigmoid',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'trunc',
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'abs',
        dtypes=all_types_and_complex_and(torch.bfloat16, torch.half),
        dtypesIfCUDA=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
        safe_casts_outputs=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
    ),
]

foreach_binary_op_db: List[OpInfo] = [
    ForeachFuncInfo(
        "add",
        dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
        dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
        supports_alpha_param=True,
    ),
    ForeachFuncInfo(
        "sub",
        dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
        dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
        supports_alpha_param=True,
    ),
    ForeachFuncInfo(
        "mul",
        dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
        dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
    ),
    ForeachFuncInfo(
        "div",
        dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
        dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
    ),
]

foreach_pointwise_op_db: List[ForeachFuncInfo] = [
    ForeachFuncInfo(
        "addcmul",
        dtypes=all_types_and_complex(),
        dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16),
    ),
    ForeachFuncInfo(
        "addcdiv",
        dtypes=all_types_and_complex(),
        dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16),
    ),
]

foreach_minmax_op_db: List[ForeachFuncInfo] = [
    ForeachFuncInfo(
        "maximum",
        dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
        dtypesIfCUDA=all_types_and(torch.float16, torch.bool),
    ),
    ForeachFuncInfo(
        "minimum",
        dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
        dtypesIfCUDA=all_types_and(torch.float16, torch.bool),
    ),
]

foreach_reduce_op_db: List[ForeachFuncInfo] = [
    ForeachFuncInfo(
        "norm",
        dtypesIfCPU=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16),
    ),
]

def reference_sign(x):
    if x.dtype == np.bool_:
        # `np.sign` doesn't support `bool`.
        # >>> np.sign(True)
        # ufunc 'sign' did not contain a loop
        # with signature matching types dtype('bool') -> dtype('bool')
        return np.sign(x, dtype=np.uint8).astype(np.bool_)
    return np.sign(x)


def reference_sgn(x):
    # NumPy doesn't have an equivalent to `torch.sgn` when the dtype is complex.
    # For complex inputs, `np.sign` returns sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j.
    # while `torch.sgn` returns, 0 if abs(input) == 0 else input/abs(input)
    if x.dtype not in [np.complex64, np.complex128]:
        return reference_sign(x)

    out = (x / np.abs(x))
    if out.ndim == 0:
        # Handle x == 0 case
        if (x == 0):
            # Can't assign to np.complex object
            # So make a new one.
            return np.array(complex(0, 0), dtype=x.dtype)
        return out

    # Handle x == 0 case
    mask = (x == 0)
    out[mask] = complex(0, 0)
    return out


def reference_sigmoid(x):
    # 'scipy.special.expit' not supported for the input types
    if x.dtype in [np.complex64, np.complex128]:
        return (1 / (1 + np.exp(-x)))
    return scipy.special.expit(x)


def reference_logsigmoid(x):
    return np.where(
        x < 0,
        x - np.log1p(np.exp(x)),
        -np.log1p(np.exp(-x)))


def reference_hardsigmoid(x):
    intermediate = x / 6 + 0.5
    y = np.clip(intermediate, 0, None)
    return np.where(y > 1, 1, y).astype(x.dtype)


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

def reference_polygamma(x, n):
    # WEIRD `scipy.special.polygamma` behavior
    # >>> scipy.special.polygamma(0, np.array(501, dtype=np.float32)).dtype
    # dtype('float64')
    # >>> scipy.special.polygamma(0, np.array([501], dtype=np.float32)).dtype
    # dtype('float32')
    #
    # Thus we cast output to the default torch dtype.
    np_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]
    return scipy.special.polygamma(n, x).astype(np_dtype)


def reference_mvlgamma(x, d):
    if x.dtype == np.float16:
        return scipy.special.multigammaln(x, d).astype(np.float16)

    return scipy.special.multigammaln(x, d)

def reference_softplus(input, beta=1, threshold=20):
    non_linear = input * beta <= threshold
    output = input.copy()
    output[non_linear] = np.log(1 + np.exp(beta * input[non_linear])) / beta
    return output


def reference_one_hot(a: np.ndarray, num_classes: int = -1) -> np.ndarray:
    if num_classes == -1:
        num_classes = int(np.amax(a) + 1)

    idcs = a.reshape(-1) + np.arange(0, a.size, dtype=np.int64) * num_classes
    one_hot = np.zeros((a.size, num_classes), dtype=a.dtype)
    np.put(one_hot, idcs, 1)
    return one_hot.reshape(*a.shape, -1)


def reference_mse_loss(input, target, reduction="mean"):
    se = (input - target) ** 2
    if reduction == "mean":
        return np.mean(se)
    elif reduction == "sum":
        return np.sum(se)
    else:  # reduction == "none"
        return se


def wrapper_set_seed(op, *args, **kwargs):
    """Wrapper to set seed manually for some functions like dropout
    See: https://github.com/pytorch/pytorch/pull/62315#issuecomment-896143189 for more details.
    """
    with freeze_rng_state():
        torch.manual_seed(42)
        return op(*args, **kwargs)


def reference_layer_norm(inp: np.ndarray, normalized_shape: Tuple[int], weight=None, bias=None, eps=1e-5):
    feature_size = np.prod(normalized_shape)
    inp_view = inp.reshape(-1, feature_size)  # type: ignore[call-overload]
    mean = inp_view.mean(axis=-1, keepdims=True)
    var = inp_view.var(axis=-1, ddof=0, keepdims=True)
    Y = (inp_view - mean) / np.sqrt(var + eps)
    if weight is None and bias is not None:
        Y = Y + bias.reshape(-1)
    elif weight is not None and bias is None:
        Y = Y * weight.reshape(-1)
    elif weight is not None and bias is not None:
        Y = Y * weight.reshape(-1) + bias.reshape(-1)
    return Y.reshape(*inp.shape)

def reference_group_norm(inp: np.ndarray, num_groups: int, weight=None, bias=None, eps=1e-5):
    inp_view = inp
    if np.prod(inp.shape) != 0:
        inp_view = inp.reshape((inp.shape[0], num_groups, -1))
    mean = inp_view.mean(axis=-1, keepdims=True)
    var = inp_view.var(axis=-1, ddof=0, keepdims=True)
    Y = (inp_view - mean) / np.sqrt(var + eps)
    Y = Y.reshape(inp.shape)
    if weight is not None:
        # weight is a vector of length equal to the channel
        if len(Y.shape) > 2:
            weight = np.tile(np.expand_dims(weight, 1), [1] + list(inp.shape[2:]))
        Y = Y * weight
    if bias is not None:
        # bias is a vector of length equal to the channel
        if len(Y.shape) > 2:
            bias = np.tile(np.expand_dims(bias, 1), [1] + list(inp.shape[2:]))
        Y = Y + bias
    return Y


# using a custom reference function since numpy only has a string side arg (instead of right and side) and doesn't
# have an out_int32 arg. Additionally, numpy doesn't support searchsorted with ND arrays, so this splits those into
# stacked 1D cases
def reference_searchsorted(sorted_sequence, boundary, out_int32=False, right=False, side='left', sorter=None):
    side = 'right' if (right or side == 'right') else 'left'
    if len(sorted_sequence.shape) == 1 :
        ret = np.searchsorted(sorted_sequence, boundary, side=side, sorter=sorter)
        return ret.astype(np.int32) if out_int32 else ret
    elif sorted_sequence.shape[0] == 0:
        if sorter is not None:
            sorter = sorter.flatten()
        ret = np.searchsorted(sorted_sequence.flatten(), boundary.flatten(), side=side, sorter=sorter)
        ret = ret.astype(np.int32) if out_int32 else ret
        return ret.reshape(boundary.shape)
    else:
        # numpy searchsorted only supports 1D inputs so we split up ND inputs
        orig_shape = boundary.shape
        num_splits = np.prod(sorted_sequence.shape[:-1])
        splits = range(0, num_splits)
        sorted_sequence, boundary = sorted_sequence.reshape(num_splits, -1), boundary.reshape(num_splits, -1)
        if sorter is not None:
            sorter = sorter.reshape(num_splits, -1)

        split_sequence = [sorted_sequence[i] for i in splits]
        split_boundary = [boundary[i] for i in splits]
        split_sorter = [sorter[i] if (sorter is not None) else None for i in splits]

        split_ret = [np.searchsorted(s_seq, b, side=side, sorter=s_sort)
                     for (s_seq, b, s_sort) in zip(split_sequence, split_boundary, split_sorter)]
        split_ret = [i.astype(np.int32) for i in split_ret] if out_int32 else split_ret
        return np.stack(split_ret).reshape(orig_shape)


def gradcheck_wrapper_hermitian_input(op, input, *args, **kwargs):
    """Gradcheck wrapper for functions that take Hermitian matrices as input.

    They require a modified function because the finite-difference algorithm
    for calculating derivatives does not preserve the Hermitian property of the input.
    """
    return op(input + input.mH, *args, **kwargs)


def gradcheck_wrapper_triangular_input(op, *args, upper=False, idx=0, **kwargs):
    """Gradcheck wrpper for functions that take lower or upper triangular matrices as input.

    They require a modified function because the finite-difference algorithm
    for calculating derivatives does not preserve the triangular property of the input.
    `idx` is used to specific which `args[idx]` is to be triangularized.
    """
    triangular_arg = args[idx].triu() if upper else args[idx].tril()
    return op(*args[:idx], triangular_arg, *args[idx + 1:], upper, **kwargs)


def gradcheck_wrapper_masked_operation(op, input, *args, **kwargs):
    """Gradcheck wrapper for masked operations.

    When mask is specified, replaces masked-out elements with zeros.

    Use for operations that produce non-finite masked-out elements,
    for instance, for minimum and maximum reductions.
    """
    output = op(input, *args, **kwargs)
    mask = kwargs.get('mask')
    if mask is not None:
        output_mask = torch._masked._output_mask(op, input, *args, **kwargs)
        output = torch.where(output_mask, output, output.new_zeros([]))
    return output


def reference_reduction_numpy(f, supports_keepdims=True):
    """Wraps a NumPy reduction operator.

    The wrapper function will forward dim, keepdim, mask, and identity
    kwargs to the wrapped function as the NumPy equivalent axis,
    keepdims, where, and initiak kwargs, respectively.

    Args:
        f: NumPy reduction operator to wrap
        supports_keepdims (bool, optional): Whether the NumPy operator accepts
            keepdims parameter. If it does not, the wrapper will manually unsqueeze
            the reduced dimensions if it was called with keepdim=True. Defaults to True.

    Returns:
        Wrapped function

    """
    @wraps(f)
    def wrapper(x: np.ndarray, *args, **kwargs):
        # Copy keys into a set
        keys = set(kwargs.keys())

        dim = kwargs.pop('dim', None)
        keepdim = kwargs.pop('keepdim', False)

        if 'dim' in keys:
            dim = tuple(dim) if isinstance(dim, Sequence) else dim

            # NumPy reductions don't accept dim=0 for scalar inputs
            # so we convert it to None if and only if dim is equivalent
            if x.ndim == 0 and dim in {0, -1, (0,), (-1,)}:
                kwargs['axis'] = None
            else:
                kwargs['axis'] = dim

        if 'keepdim' in keys and supports_keepdims:
            kwargs['keepdims'] = keepdim

        if 'mask' in keys:
            mask = kwargs.pop('mask')
            if mask is not None:
                kwargs['where'] = mask.cpu().numpy()

        if 'identity' in keys:
            identity = kwargs.pop('identity')
            if identity is not None:
                if identity.dtype is torch.bfloat16:
                    identity = identity.cpu().to(torch.float32)
                else:
                    identity = identity.cpu()
                kwargs['initial'] = identity.numpy()

        if 'unbiased' in keys:
            unbiased = kwargs.pop('unbiased')
            if unbiased is not None:
                kwargs['ddof'] = int(unbiased)

        result = f(x, *args, **kwargs)

        # Unsqueeze reduced dimensions if NumPy does not support keepdims
        if keepdim and not supports_keepdims and x.ndim > 0:
            dim = list(range(x.ndim)) if dim is None else dim
            result = np.expand_dims(result, dim)

        return result

    return wrapper


def reference_std_var(f):
    """Forwards unbiased/correction kwargs as NumPy's equivalent ddof"""
    g = reference_reduction_numpy(f)

    @wraps(g)
    def wrapper(x: np.ndarray, *args, **kwargs):
        assert not ('unbiased' in kwargs and 'correction' in kwargs)

        if 'unbiased' in kwargs:
            kwargs['ddof'] = int(kwargs.pop('unbiased'))
        elif 'correction' in kwargs:
            kwargs['ddof'] = kwargs.pop('correction')

        return g(x, *args, **kwargs)

    return wrapper


def generate_std_var_kwargs(t: torch.Tensor, **kwargs):
    """Generates unbiased/correction kwargs for std/var operators"""
    yield ((), {'unbiased': True})
    yield ((), {'unbiased': False})

    # Currently, calling std with correction is only enabled when
    # both dim and keepdim are provided.
    if 'dim' in kwargs and 'keepdim' in kwargs:
        yield ((), {'correction': 0})
        yield ((), {'correction': 1})

        numel = torch.tensor(t.shape)[kwargs.get('dim')].prod()
        yield ((), {'correction': numel // 2})

def ref_pairwise_distance(input1, input2):
    pass


# Operator database (sorted alphabetically)
op_db: List[OpInfo] = [
    UnaryUfuncInfo('abs',
                   aliases=('absolute', ),
                   ref=np.abs,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat]),
                       # Reference: https://github.com/pytorch/pytorch/issues/49224
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    dtypes=[torch.int8], active_if=TEST_WITH_ASAN),
                       # TODO: Fix test_out_arg_all_dtypes as torch.empty_like(expected_output) where expected_output=op(input)
                       # We can break the logic of the loop over all possible types but it is OK.
                       # https://github.com/pytorch/pytorch/blob/master/test/test_unary_ufuncs.py#L440-L449
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_out_arg_all_dtypes',
                                    dtypes=[torch.cfloat, torch.cdouble]),
                       # The complex formula might be wrong
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD',
                                    dtypes=complex_types()),
                       # Forward-over-reverse gradgrad might be wrong for complex (see above):
                       DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad',
                                    dtypes=complex_types()),
                   ),
                   supports_inplace_autograd=False,
                   supports_fwgrad_bwgrad=True,
                   assert_autodiffed=True,
                   supports_sparse_csr=True,
                   supports_forward_ad=True),
    # NOTE: CPU complex acos produces incorrect outputs (https://github.com/pytorch/pytorch/issues/42952)
    UnaryUfuncInfo('acos',
                   aliases=('arccos', ),
                   ref=np.arccos,
                   domain=(-1, 1),
                   handles_complex_extremals=False,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # "rsqrt_cpu" not implemented for 'BFloat16'
                   backward_dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-1,
                                                  torch.complex64: 1e-2}),),
                   safe_casts_outputs=True,
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_grad',
                                    dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_method_grad',
                                    dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_inplace_grad',
                                    dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD',
                                    dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_inplace_forward_mode_AD',
                                    dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                   )),
    # NOTE: the derivative for inplace acosh is not implemented
    UnaryUfuncInfo('acosh',
                   aliases=('arccosh', ),
                   ref=np.arccosh,
                   domain=(1, None),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # "rsqrt_cuda" not implemented for 'BFloat16'
                   backward_dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   supports_inplace_autograd=False,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cuda', dtypes=[torch.cdouble],
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cuda', dtypes=[torch.cdouble],
                                    active_if=IS_WINDOWS),
                       # Reference: https://github.com/pytorch/pytorch/issues/50692
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_grad',
                                    device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_method_grad',
                                    device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD',
                                    device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                   ),
                   # acosh is not defined at x < 1 (real) or |z| < 1 (complex)
                   reference_numerics_filter=NumericsFilter(
                       condition=lambda x: (torch.abs(x) < 1 if x.is_complex() else x < 1),
                       safe_val=2)),
    BinaryUfuncInfo('add',
                    # NumPy has no builtin reference for the alpha kwarg, but it is easy enough to emulate
                    ref=lambda input, other, *, alpha=1: np.add(input, other) if alpha == 1 \
                    else np.add(input, np.multiply(alpha, other)),
                    dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                    assert_autodiffed=True,
                    sample_inputs_func=partial(sample_inputs_add_sub, alpha=2),
                    supports_inplace_autograd=False,
                    supports_fwgrad_bwgrad=True,
                    supports_forward_ad=True,
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"),
                                     'TestBinaryUfuncs',
                                     'test_reference_numerics_extremal_values',
                                     dtypes=(torch.complex64, torch.complex128)),
                    )),
    BinaryUfuncInfo('mul',
                    aliases=('multiply',),
                    dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16, torch.bool),
                    assert_autodiffed=True,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    sample_inputs_func=partial(sample_inputs_binary_pwise, python_scalars=True)),
    BinaryUfuncInfo('sub',
                    # NumPy has no builtin reference for the alpha kwarg, but it is easy enough to emulate
                    ref=lambda input, other, *, alpha=1: np.subtract(input, np.multiply(alpha, other)),
                    aliases=('subtract',),
                    dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16),
                    assert_autodiffed=True,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    sample_inputs_func=partial(sample_inputs_add_sub, alpha=2, python_scalars=True),
                    supports_inplace_autograd=False,
                    decorators=(
                        DecorateInfo(
                            toleranceOverride({torch.float16: tol(atol=1e-2, rtol=0)}),
                            'TestBinaryUfuncs', 'test_reference_numerics'),
                    ),
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"),
                                     'TestBinaryUfuncs',
                                     'test_reference_numerics',
                                     dtypes=(torch.uint8,)),
                        DecorateInfo(unittest.skip("Skipped!"),
                                     'TestBinaryUfuncs',
                                     'test_reference_numerics_small_values',
                                     dtypes=(torch.uint8,)),
                    )),
    OpInfo('addmm',
           # This addmm OpInfo is for when alpha and beta are not both equal to 1.
           # alpha=beta=1 is tested in the following opinfo, because that special case will
           # trigger addmm being decomposed by a jit pass.
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfROCM=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_addmm),
    OpInfo('addmm',
           # When alpha=beta=1 as compile-time constants, JIT will decompose addmm into mm and add.
           variant_test_name='decomposed',
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfROCM=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           autodiff_nonfusible_nodes=['aten::add', 'aten::mm'],
           sample_inputs_func=partial(sample_inputs_addmm, alpha=1, beta=1)),
    OpInfo('addmv',
           dtypes=all_types_and_complex_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128,
                                           *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_types_and(torch.half),
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_addmv),
    OpInfo('addbmm',
           ref=lambda M, batch1, batch2, beta=1, alpha=1: np.add(np.multiply(np.asarray(beta, dtype=M.dtype), M),
                                                                 np.multiply(np.asarray(alpha, dtype=batch1.dtype),
                                                                             np.sum(np.matmul(batch1, batch2), axis=0))),
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if SM53OrLater else []),
           dtypesIfROCM=floating_types_and(torch.half),
           backward_dtypesIfROCM=floating_types_and(torch.half),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           decorators=[
               DecorateInfo(
                   toleranceOverride({torch.float32: tol(atol=1.3e-05, rtol=1.3e-05),
                                      torch.complex64: tol(atol=1e-05, rtol=1.2e-03)}),
                   'TestCommon', 'test_reference_testing')],
           skips=(
               # FIXME: bfloat16 backward support likely depends on CUDA11+
               #   and SM53+
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_dtypes', active_if=IS_WINDOWS),
               # addbmm does not correctly warn when resizing out= inputs
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),
               # https://github.com/pytorch/pytorch/issues/55907
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_variant_consistency_eager'),
           ),
           sample_inputs_func=sample_inputs_addbmm),
    OpInfo('baddbmm',
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128,
                                           *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=floating_types_and(torch.float16,
                                                    *[torch.bfloat16] if SM53OrLater else [],
                                                    torch.complex64, torch.complex128),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           decorators=[
               DecorateInfo(
                   toleranceOverride({torch.complex64: tol(atol=1e-05, rtol=1.2e-03)}),
                   'TestCommon', 'test_variant_consistency_eager', device_type='cuda'),
               DecorateInfo(
                   toleranceOverride({torch.complex64: tol(atol=1e-05, rtol=1.2e-03)}),
                   'TestMathBits', 'test_conj_view', device_type='cuda')],
           sample_inputs_func=sample_inputs_baddbmm),
    OpInfo('dot',
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_dot_vdot,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           ),
    OpInfo('vdot',
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           sample_inputs_func=sample_inputs_dot_vdot,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           ),
    OpInfo('bmm',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if SM53OrLater else []),
           assert_autodiffed=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # FIXME: bfloat16 backward support likely depends on CUDA11+
               #   and SM53+
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_dtypes', active_if=IS_WINDOWS),
           ),
           sample_inputs_func=sample_inputs_bmm),
    OpInfo('mv',
           dtypes=all_types_and_complex_and(torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_mv),
    OpInfo('addr',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           backward_dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
           backward_dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           # Reference: https://github.com/pytorch/pytorch/issues/50747
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # Reference: https://github.com/pytorch/pytorch/issues/50747
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_variant_consistency_eager',
                            dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16)),
           ),
           sample_inputs_func=sample_inputs_addr,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('addcmul',
           dtypes=all_types_and_complex_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.float16, torch.bfloat16),
           assert_autodiffed=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_inplace_autograd=False,
           skips=(
               # TODO: update sample inputs with for_inplace_variant kwarg to support this test
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_variant_consistency_eager'),),
           sample_inputs_func=sample_inputs_addcmul_addcdiv),
    OpInfo('addcdiv',
           dtypes=floating_and_complex_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # TODO: update sample inputs with for_inplace_variant kwarg to support this test
               DecorateInfo(unittest.skip("Skipped!"),
                            'TestCommon',
                            'test_variant_consistency_eager'),),
           sample_inputs_func=sample_inputs_addcmul_addcdiv),
    UnaryUfuncInfo('asin',
                   aliases=('arcsin', ),
                   ref=np.arcsin,
                   domain=(-1, 1),
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   decorators=[
                       DecorateInfo(
                           toleranceOverride({torch.float16: tol(atol=1e-05, rtol=1e-03)}),
                           'TestUnaryUfuncs', device_type='cuda'),
                       precisionOverride({torch.bfloat16: 1e-2}),
                   ],
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cuda', dtypes=[torch.cdouble],
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cuda', dtypes=[torch.cdouble],
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   )),
    # NOTE: derivative for inplace asinh is not implemented
    UnaryUfuncInfo('asinh',
                   aliases=('arcsinh', ),
                   ref=np.arcsinh,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   supports_inplace_autograd=False,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cuda', dtypes=[torch.cdouble],
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cuda', dtypes=[torch.cdouble],
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   )),
    UnaryUfuncInfo('atan',
                   aliases=('arctan', ),
                   ref=np.arctan,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   )),
    BinaryUfuncInfo('atan2',
                    aliases=('arctan2',),
                    dtypes=all_types_and(torch.bool),
                    dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                    promotes_int_to_float=True,
                    sample_inputs_func=sample_inputs_atan2,
                    skips=(
                        # TypeError: atan2(): argument 'other' (position 2) must be Tensor, not float
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs'),
                    )),
    UnaryUfuncInfo('atanh',
                   aliases=('arctanh', ),
                   ref=np.arctanh,
                   domain=(-1, 1),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   supports_inplace_autograd=False,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    device_type='cpu', dtypes=[torch.cfloat]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cuda', dtypes=[torch.cfloat],
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   )),
    OpInfo('allclose',
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           ref=np.allclose,
           supports_autograd=False,
           supports_forward_ad=False,
           sample_inputs_func=sample_inputs_allclose,
           skips=(
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('broadcast_to',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_broadcast_to),
    OpInfo('broadcast_tensors',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # JIT does not support variadic tensors.
               # RuntimeError: input->type()->kind() == TypeKind::OptionalType
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":252,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=[torch.float32]),
           ),
           sample_inputs_func=sample_inputs_broadcast_tensors),
    OpInfo('block_diag',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # JIT does not support variadic tensors.
               # RuntimeError: input->type()->kind() == TypeKind::OptionalType
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":252,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=[torch.float32]),
           ),
           sample_inputs_func=sample_inputs_block_diag),
    BinaryUfuncInfo('bitwise_and',
                    dtypes=integral_types_and(torch.bool),
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_binary_pwise,
                    skips=(
                        # RuntimeError: "bitwise_and_cuda" not implemented for 'Half'
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_type_promotion'),
                    )),
    UnaryUfuncInfo('bitwise_not',
                   ref=np.bitwise_not,
                   dtypes=integral_types_and(torch.bool),
                   supports_autograd=False),
    BinaryUfuncInfo('bitwise_left_shift',
                    op=torch.bitwise_left_shift,
                    dtypes=all_types(),
                    dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_bitwise_shift,
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_type_promotion'),
                        # FIXME: Undefined behavior sanitizer: shift exponent -9 is negative
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', device_type='cpu'),
                    )),
    BinaryUfuncInfo('bitwise_right_shift',
                    op=torch.bitwise_right_shift,
                    dtypes=all_types(),
                    dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_bitwise_shift,
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_type_promotion'),
                        # FIXME: Undefined behavior sanitizer
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', device_type='cpu'),
                    )),
    OpInfo('combinations',
           op=torch.combinations,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_autograd=False,
           supports_out=False,
           sample_inputs_func=sample_inputs_combinations),
    OpInfo('cartesian_prod',
           op=torch.cartesian_prod,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_autograd=False,
           supports_out=False,
           sample_inputs_func=sample_inputs_cartesian_prod,
           skips=(
               # RuntimeError: input->type()->kind() == TypeKind::OptionalType
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":270
               DecorateInfo(unittest.skip("Skipped!"),
                            'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),
           )),
    OpInfo('cdist',
           dtypes=floating_types(),
           supports_out=False,
           supports_gradgrad=False,
           assert_autodiffed=False,
           sample_inputs_func=sample_inputs_cdist),
    UnaryUfuncInfo('ceil',
                   ref=np.ceil,
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   assert_autodiffed=True),
    OpInfo('cholesky',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_cholesky,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],),
    OpInfo('cholesky_inverse',
           dtypes=floating_and_complex_types(),
           backward_dtypes=floating_types(),
           # TODO: RuntimeError: cholesky_inverse does not support automatic differentiation for outputs
           # with complex dtype.
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_cholesky_inverse,
           gradcheck_wrapper=gradcheck_wrapper_triangular_input,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           skips=(
               # TODO: FIXME: cholesky_inverse throws an error in forward when requires_grad=True
               #   for complex tensors
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_dtypes'),
               # cholesky_inverse does not correctly warn when resizing out= inputs
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),)),
    OpInfo('cholesky_solve',
           op=torch.cholesky_solve,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_cholesky_solve,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_wrapper=lambda *args, **kwargs: gradcheck_wrapper_triangular_input(*args, idx=1, **kwargs),
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cholesky_solve does not correctly warn when resizing out= inputs
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),),),
    OpInfo('chunk',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           sample_inputs_func=sample_inputs_chunk,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False),
    OpInfo('clone',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           sample_inputs_func=sample_inputs_clone,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False),
    OpInfo('contiguous',
           op=lambda x, *args, **kwargs: x.contiguous(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           sample_inputs_func=sample_inputs_contiguous,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           autodiff_fusible_nodes=['aten::contiguous'],
           assert_jit_shape_analysis=True,
           supports_out=False),
    OpInfo('sum_to_size',
           op=lambda x, *args, **kwargs: x.sum_to_size(*args, **kwargs),
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_sum_to_size,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False,
           skips=(
               # RuntimeError: inputSet && outputSet
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":118
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),),),
    OpInfo('symeig',
           dtypes=floating_and_complex_types(),
           check_batched_grad=False,
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_symeig,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    # NOTE: clamp has seperate opinfos for scalar min/max (unary op) vs. tensors
    OpInfo('clamp',
           aliases=('clip',),
           dtypes=all_types_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_clamp),
    UnaryUfuncInfo('clamp',
                   variant_test_name='scalar',
                   aliases=('clip', ),
                   decorators=(precisionOverride({torch.bfloat16: 7e-2, torch.float16: 1e-2}),),
                   ref=np.clip,
                   dtypes=all_types_and(torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/54841
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.bfloat16]),
                   ),
                   sample_kwargs=sample_kwargs_clamp_scalar,
                   sample_inputs_func=sample_inputs_clamp_scalar),
    UnaryUfuncInfo('positive',
                   ref=np.positive,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   supports_out=False,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   ),
    UnaryUfuncInfo('conj',
                   ref=np.conj,
                   dtypes=all_types_and_complex_and(torch.bool,
                                                    torch.bfloat16, torch.half),
                   supports_sparse=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_out=False),
    UnaryUfuncInfo('conj_physical',
                   ref=np.conj,
                   dtypes=all_types_and_complex_and(torch.bool,
                                                    torch.bfloat16, torch.half),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   skips=(
                       # RuntimeError: inputSet && outputSet
                       # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":118,
                       # please report a bug to PyTorch.
                       DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32, )),
                       DecorateInfo(unittest.skip("Skipped! conj_physical_ not implemented for sparse"),
                                    'TestSparseUnaryUfuncs', 'test_inplace'),
                   )),
    OpInfo('resolve_conj',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_view_as_real,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False,
           ),
    OpInfo('resolve_neg',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_view_as_real,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False,
           ),
    OpInfo('view_as_real',
           dtypes=complex_types(),
           supports_forward_ad=True,
           supports_out=False,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_view_as_real,
           test_conjugated_samples=False,
           ),
    OpInfo('view_as_complex',
           dtypes=floating_types(),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           test_neg_view=False,
           sample_inputs_func=sample_inputs_view_as_complex,
           skips=(
               # RuntimeError: Tensor must have a last dimension with stride 1
               DecorateInfo(unittest.expectedFailure, "TestCommon", "test_noncontiguous_samples"),
           )),
    BinaryUfuncInfo('complex',
                    dtypes=floating_types(),
                    sample_inputs_func=sample_inputs_complex,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    skips=(
                        # RuntimeError: Expected object of scalar type Float but got scalar type Double for second argument
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_type_promotion'),
                        # TypeError: complex(): argument 'imag' (position 2) must be Tensor, not float
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs'),
                    )),
    BinaryUfuncInfo('copysign',
                    dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
                    promotes_int_to_float=True,
                    sample_inputs_func=sample_inputs_copysign,
                    supports_inplace_autograd=False,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True),
    OpInfo('corrcoef',
           dtypes=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.half, *[torch.bfloat16] if CUDA11OrLater else []),
           sample_inputs_func=sample_inputs_corrcoef,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False),
    UnaryUfuncInfo('cos',
                   ref=np.cos,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   handles_large_floats=False,
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu',
                                    dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                   )),
    UnaryUfuncInfo('cosh',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.cosh),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/48641
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.int8]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu',
                                    dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu',
                                    dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                   )),
    OpInfo('cov',
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.half, *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=all_types_and_complex_and(torch.half, *[torch.bfloat16] if CUDA11OrLater else []),
           sample_inputs_func=sample_inputs_cov,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # JIT test not working for tensor kwargs (https://github.com/pytorch/pytorch/issues/58507)
               # RuntimeError:
               # undefined value tensor:
               #   File "<string>", line 3
               # def the_method(i0):
               #     return torch.cov(i0, correction=0, fweights=None, aweights=tensor([0.0518, 0.4681], dtype=torch.float32, requires_grad=True)) # noqa: B950
               #                                                                ~~~~~~ <--- HERE
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('cross',
           dtypes=all_types_and_complex_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.half),
           sample_inputs_func=sample_inputs_cross,
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True),
    OpInfo('linalg.cross',
           ref=lambda x, y, dim=-1: np.cross(x, y, axis=dim),
           op=torch.linalg.cross,
           dtypes=all_types_and_complex(),
           dtypesIfCPU=all_types_and_complex_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.half),
           aten_name='linalg_cross',
           sample_inputs_func=sample_inputs_cross,
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True),
    OpInfo('cumsum',
           dtypes=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # cumsum does not handle correctly out= dtypes
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),
           ),
           sample_inputs_func=sample_inputs_cumulative_ops),
    OpInfo('cumprod',
           dtypes=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.float16, torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # cumprod does not handle correctly out= dtypes
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),
           ),
           # gradgradcheck fails in fast_mode=True: #56275
           sample_inputs_func=sample_inputs_cumprod,
           gradcheck_fast_mode=False),
    OpInfo('cummax',
           dtypes=all_types_and(torch.bool, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_cumulative_ops, supports_dtype_kwargs=False),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('cummin',
           dtypes=all_types_and(torch.bool, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_cumulative_ops, supports_dtype_kwargs=False),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    UnaryUfuncInfo('deg2rad',
                   ref=np.radians,
                   decorators=(precisionOverride({torch.bfloat16: 7e-1,
                                                  torch.float16: 7e-1}),),
                   dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/51283#issuecomment-770614273
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    dtypes=[torch.bfloat16]),
                   ),
                   safe_casts_outputs=True),
    OpInfo('diff',
           op=torch.diff,
           # np.diff has np._NoValue as default values for prepend and append, compare_with_reference breaks if prepend/append
           # are set as None when converting to numpy
           ref=lambda input, n=1, dim=-1, prepend=np._NoValue, append=np._NoValue: (
               np.diff(input, n, dim, np._NoValue if prepend is None else prepend, np._NoValue if append is None else append)
           ),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_diff),
    BinaryUfuncInfo('div',
                    aliases=('divide',),
                    variant_test_name='no_rounding_mode',
                    dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                    sample_inputs_func=partial(sample_inputs_binary_pwise, python_scalars=True),
                    supports_forward_ad=True,
                    promotes_int_to_float=True,
                    supports_fwgrad_bwgrad=True,
                    assert_autodiffed=True,
                    rhs_make_tensor_kwargs=dict(exclude_zero=True),
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD',
                                     device_type='cuda', dtypes=[torch.double, torch.cdouble]),
                        DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_inplace_forward_mode_AD',
                                     device_type='cuda', dtypes=[torch.double, torch.cdouble]),
                    ),),
    BinaryUfuncInfo('div',
                    aliases=('divide',),
                    variant_test_name='trunc_rounding',
                    dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                    sample_inputs_func=partial(sample_inputs_binary_pwise, rounding_mode="trunc", python_scalars=True),
                    supports_forward_ad=True,
                    promotes_int_to_float=True,
                    supports_fwgrad_bwgrad=True,
                    assert_autodiffed=True,
                    rhs_make_tensor_kwargs=dict(exclude_zero=True),
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD',
                                     device_type='cuda', dtypes=[torch.double, torch.cdouble]),
                        DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_inplace_forward_mode_AD',
                                     device_type='cuda', dtypes=[torch.double, torch.cdouble]),
                    ),),
    BinaryUfuncInfo('div',
                    aliases=('divide',),
                    variant_test_name='floor_rounding',
                    dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                    sample_inputs_func=partial(sample_inputs_binary_pwise, rounding_mode="floor", python_scalars=True),
                    supports_forward_ad=True,
                    promotes_int_to_float=True,
                    supports_fwgrad_bwgrad=True,
                    assert_autodiffed=True,
                    rhs_make_tensor_kwargs=dict(exclude_zero=True),
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD',
                                     device_type='cuda', dtypes=[torch.double, torch.cdouble]),
                        DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_inplace_forward_mode_AD',
                                     device_type='cuda', dtypes=[torch.double, torch.cdouble]),
                    ),),
    BinaryUfuncInfo('true_divide',
                    dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                    supports_forward_ad=True,
                    promotes_int_to_float=True,
                    supports_fwgrad_bwgrad=True,
                    sample_inputs_func=sample_inputs_binary_pwise,
                    rhs_make_tensor_kwargs=dict(exclude_zero=True)),
    UnaryUfuncInfo('exp',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.exp),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/50093#pullrequestreview-561791547
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    dtypes=[torch.bfloat16]),
                       # Reference: https://github.com/pytorch/pytorch/issues/48010
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                   ),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True),
    OpInfo('expand',
           op=lambda self, shape: self.expand(shape),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_expand,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           assert_jit_shape_analysis=True,
           supports_out=False),
    OpInfo('expand_as',
           op=lambda self, other: self.expand_as(other),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_expand_as,
           supports_out=False,
           skips=(
               DecorateInfo(unittest.skip("Second argument does not need gradient"),
                            "TestCommon", "test_floating_inputs_are_differentiable"),),
           ),
    OpInfo('diag',
           dtypes=all_types_and_complex_and(torch.bool),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_diag),
    OpInfo('diag_embed',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_diagonal_diag_embed),
    OpInfo('diagonal',
           # They are not strictly aliases as they have diverging defaults, but we can see them as aliases for testing purposes
           # If we add tests that test the function against the alias, make linalg.diagonal into its own OpInfo
           aliases=('linalg.diagonal',),
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_diagonal_diag_embed),
    OpInfo('diagonal_scatter',
           dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_diagonal_scatter),
    BinaryUfuncInfo('eq',
                    dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                    always_returns_bool=True,
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_comparison_ops),
    BinaryUfuncInfo('fmax',
                    op=torch.fmax,
                    dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    sample_inputs_func=sample_inputs_max_min_binary,
                    skips=(
                        # RuntimeError: "max_elementwise_cuda" not implemented for 'ComplexFloat'
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_type_promotion'),
                        # TypeError: fmax(): argument 'other' (position 2) must be Tensor, not float
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs'),
                    )),
    BinaryUfuncInfo('fmin',
                    op=torch.fmin,
                    dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    sample_inputs_func=sample_inputs_max_min_binary,
                    skips=(
                        # RuntimeError: "min_elementwise_cuda" not implemented for 'ComplexFloat'
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_type_promotion'),
                        # TypeError: fmin(): argument 'other' (position 2) must be Tensor, not float
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs'),
                    )),
    BinaryUfuncInfo('fmod',
                    ref=np.fmod,
                    dtypes=all_types_and(torch.float16),
                    rhs_make_tensor_kwargs={'exclude_zero': True},
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    sample_inputs_func=sample_inputs_fmod_remainder,
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs',
                                     dtypes=(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)),
                    )),
    BinaryUfuncInfo('fmod',
                    ref=np.fmod,
                    variant_test_name='autodiffed',
                    dtypes=all_types_and(torch.float16, torch.bool),
                    assert_autodiffed=True,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    rhs_make_tensor_kwargs={'exclude_zero': True},
                    sample_inputs_func=partial(sample_inputs_fmod_remainder, autodiffed=True),
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs',
                                     dtypes=(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)),
                    )),
    BinaryUfuncInfo('remainder',
                    ref=np.remainder,
                    dtypes=all_types_and(torch.float16, torch.bfloat16),
                    dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    rhs_make_tensor_kwargs={'exclude_zero': True},
                    sample_inputs_func=sample_inputs_fmod_remainder,
                    skips=(
                        # AssertionError: False is not true : Tensors failed to compare as equal!
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs',
                                     dtypes=(torch.bfloat16,)),
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs',
                                     dtypes=(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)),
                    )),
    BinaryUfuncInfo('remainder',
                    ref=np.remainder,
                    variant_test_name='autodiffed',
                    dtypes=all_types_and(torch.float16, torch.bool, torch.bfloat16),
                    dtypesIfCUDA=all_types_and(torch.float16, torch.bool, torch.bfloat16),
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    assert_autodiffed=True,
                    rhs_make_tensor_kwargs={'exclude_zero': True},
                    sample_inputs_func=partial(sample_inputs_fmod_remainder, autodiffed=True),
                    decorators=(
                        # Fails on XLA
                        # False is not true : Tensors failed to compare as equal!
                        # Attempted to compare equality of tensors with different dtypes
                        DecorateInfo(unittest.expectedFailure, 'TestOpInfo', device_type='xla', dtypes=(torch.long,)),
                    ),
                    skips=(
                        # AssertionError: False is not true : Tensors failed to compare as equal!
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs',
                                     dtypes=(torch.bfloat16,)),
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs',
                                     dtypes=(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)),
                    )),
    UnaryUfuncInfo('frac',
                   ref=lambda x: np.modf(x)[0],
                   dtypes=floating_types_and(torch.bfloat16, torch.float16),
                   dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   # Reference for disabling extremals
                   # https://github.com/pytorch/pytorch/issues/51948
                   handles_extremals=False),
    SpectralFuncInfo('fft.fft',
                     aten_name='fft_fft',
                     ref=np.fft.fft,
                     ndimensional=SpectralFuncType.OneD,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     ),
    SpectralFuncInfo('fft.fft2',
                     aten_name='fft_fft2',
                     ref=np.fft.fft2,
                     ndimensional=SpectralFuncType.TwoD,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     decorators=[precisionOverride(
                         {torch.float: 1e-4, torch.cfloat: 1e-4})],
                     ),
    SpectralFuncInfo('fft.fftn',
                     aten_name='fft_fftn',
                     ref=np.fft.fftn,
                     ndimensional=SpectralFuncType.ND,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     decorators=[precisionOverride(
                         {torch.float: 1e-4, torch.cfloat: 1e-4})],
                     ),
    SpectralFuncInfo('fft.hfft',
                     aten_name='fft_hfft',
                     ref=np.fft.hfft,
                     ndimensional=SpectralFuncType.OneD,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False),
    SpectralFuncInfo('fft.hfft2',
                     aten_name='fft_hfft2',
                     ref=scipy.fft.hfft2 if has_scipy_fft else None,
                     ndimensional=SpectralFuncType.TwoD,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False,
                     decorators=[
                         DecorateInfo(
                             precisionOverride({torch.float: 2e-4, torch.cfloat: 2e-4}),
                             'TestFFT', 'test_reference_nd')],
                     ),
    SpectralFuncInfo('fft.hfftn',
                     aten_name='fft_hfftn',
                     ref=scipy.fft.hfftn if has_scipy_fft else None,
                     ndimensional=SpectralFuncType.ND,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False,
                     decorators=[
                         DecorateInfo(
                             precisionOverride({torch.float: 2e-4, torch.cfloat: 2e-4}),
                             'TestFFT', 'test_reference_nd')],
                     ),
    SpectralFuncInfo('fft.rfft',
                     aten_name='fft_rfft',
                     ref=np.fft.rfft,
                     ndimensional=SpectralFuncType.OneD,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_grad=False,
                     check_batched_gradgrad=False),
    SpectralFuncInfo('fft.rfft2',
                     aten_name='fft_rfft2',
                     ref=np.fft.rfft2,
                     ndimensional=SpectralFuncType.TwoD,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_grad=False,
                     check_batched_gradgrad=False,
                     decorators=[precisionOverride({torch.float: 1e-4})],),
    SpectralFuncInfo('fft.rfftn',
                     aten_name='fft_rfftn',
                     ref=np.fft.rfftn,
                     ndimensional=SpectralFuncType.ND,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_grad=False,
                     check_batched_gradgrad=False,
                     decorators=[precisionOverride({torch.float: 1e-4})],),
    SpectralFuncInfo('fft.ifft',
                     aten_name='fft_ifft',
                     ref=np.fft.ifft,
                     ndimensional=SpectralFuncType.OneD,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types()),
    SpectralFuncInfo('fft.ifft2',
                     aten_name='fft_ifft2',
                     ref=np.fft.ifft2,
                     ndimensional=SpectralFuncType.TwoD,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     decorators=[
                         DecorateInfo(
                             precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                             'TestFFT', 'test_reference_nd')],
                     ),
    SpectralFuncInfo('fft.ifftn',
                     aten_name='fft_ifftn',
                     ref=np.fft.ifftn,
                     ndimensional=SpectralFuncType.ND,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     decorators=[
                         DecorateInfo(
                             precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                             'TestFFT', 'test_reference_nd')],
                     ),
    SpectralFuncInfo('fft.ihfft',
                     aten_name='fft_ihfft',
                     ref=np.fft.ihfft,
                     ndimensional=SpectralFuncType.OneD,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_types(),
                     check_batched_grad=False),
    SpectralFuncInfo('fft.ihfft2',
                     aten_name='fft_ihfft2',
                     ref=scipy.fft.ihfftn if has_scipy_fft else None,
                     ndimensional=SpectralFuncType.TwoD,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_types(),
                     check_batched_grad=False,
                     check_batched_gradgrad=False,
                     decorators=[
                         DecorateInfo(
                             precisionOverride({torch.float: 2e-4}),
                             'TestFFT', 'test_reference_nd')],
                     ),
    SpectralFuncInfo('fft.ihfftn',
                     aten_name='fft_ihfftn',
                     ref=scipy.fft.ihfftn if has_scipy_fft else None,
                     ndimensional=SpectralFuncType.ND,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_types(),
                     check_batched_grad=False,
                     check_batched_gradgrad=False,
                     decorators=[
                         DecorateInfo(
                             precisionOverride({torch.float: 2e-4}),
                             'TestFFT', 'test_reference_nd')],
                     ),
    SpectralFuncInfo('fft.irfft',
                     aten_name='fft_irfft',
                     ref=np.fft.irfft,
                     ndimensional=SpectralFuncType.OneD,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False),
    SpectralFuncInfo('fft.irfft2',
                     aten_name='fft_irfft2',
                     ref=np.fft.irfft2,
                     ndimensional=SpectralFuncType.TwoD,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False,
                     decorators=[
                         DecorateInfo(
                             precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                             'TestFFT', 'test_reference_nd')],
                     ),
    SpectralFuncInfo('fft.irfftn',
                     aten_name='fft_irfftn',
                     ref=np.fft.irfftn,
                     ndimensional=SpectralFuncType.ND,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False,
                     decorators=[
                         DecorateInfo(
                             precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                             'TestFFT', 'test_reference_nd')],
                     ),
    OpInfo('fft.fftshift',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           sample_inputs_func=sample_inputs_fftshift,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           ),
    OpInfo('fft.ifftshift',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           sample_inputs_func=sample_inputs_fftshift,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           ),
    OpInfo('stft',
           decorators=[
               skipCPUIfNoFFT,
               DecorateInfo(unittest.skip("Skipped! stft does not match the native function"),
                            'TestJit', 'test_variant_consistency_jit'),
           ],
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_stft,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_out=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           ),
    OpInfo('istft',
           decorators=[
               skipCPUIfNoFFT,
               DecorateInfo(unittest.skip("Skipped! istft does not match the native function"),
                            'TestJit', 'test_variant_consistency_jit'),
               # gradcheck fails on ROCm (gh-68429)
               DecorateInfo(skipCUDAIfRocm, 'TestGradients', 'test_fn_grad'),
           ],
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_istft,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_out=False,
           ),
    UnaryUfuncInfo('floor',
                   ref=np.floor,
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   assert_autodiffed=True),
    OpInfo('flip',
           op=torch.flip,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_flip,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False),
    OpInfo('fliplr',
           op=torch.fliplr,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_fliplr_flipud,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False),
    OpInfo('flipud',
           op=torch.flipud,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_fliplr_flipud,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False),
    UnaryUfuncInfo('i0',
                   ref=np_unary_ufunc_integer_promotion_wrapper(
                       scipy.special.i0) if TEST_SCIPY else _NOTHING,
                   aliases=('special.i0',),
                   decorators=(precisionOverride({torch.bfloat16: 3e-1,
                                                  torch.float16: 5e-1}),),
                   backward_dtypesIfCPU=floating_types(),
                   backward_dtypesIfCUDA=floating_types(),
                   backward_dtypesIfROCM=floating_types(),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   sample_inputs_func=sample_inputs_i0_i1),
    UnaryUfuncInfo('special.i0e',
                   aten_name='special_i0e',
                   ref=scipy.special.i0e if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.bfloat16: 3e-1,
                                                  torch.float16: 3e-1}),),
                   backward_dtypesIfCPU=floating_types(),
                   backward_dtypesIfCUDA=floating_types(),
                   backward_dtypesIfROCM=floating_types(),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   sample_inputs_func=sample_inputs_i0_i1,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('special.i1',
                   aten_name='special_i1',
                   ref=np_unary_ufunc_integer_promotion_wrapper(scipy.special.i1) if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool),
                   sample_inputs_func=sample_inputs_i0_i1,
                   safe_casts_outputs=True,
                   decorators=(
                       DecorateInfo(toleranceOverride({
                           torch.float32: tol(atol=1e-4, rtol=0),
                           torch.bool: tol(atol=1e-4, rtol=0)})),
                   ),
                   skips=(
                       # TODO: FIXME: jiterator does not support casting to complex outs
                       DecorateInfo(unittest.skip("FIXME: Jiterator does not support complex outs!"),
                                    "TestUnaryUfuncs",
                                    "test_out_arg_all_dtypes",
                                    device_type='cuda'),
                   ),
                   supports_fwgrad_bwgrad=True,
                   supports_forward_ad=True),
    UnaryUfuncInfo('special.i1e',
                   aten_name='special_i1e',
                   ref=scipy.special.i1e if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool),
                   sample_inputs_func=sample_inputs_i0_i1,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('special.ndtr',
                   aten_name='special_ndtr',
                   decorators=(precisionOverride({torch.bfloat16: 5e-3,
                                                  torch.float16: 5e-4}),),
                   ref=scipy.special.ndtr if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.bfloat16, torch.float16),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True,
                   skips=(
                       # Dispatch stub: unsupported device typemeta
                       DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad', device_type='meta'),
                       # (ROCm) Memory access fault by GPU node-4 (Agent handle: 0x55cebc9e8430) on address 0x7fa17b757000
                       # CUDA memory exception, see https://github.com/pytorch/pytorch/issues/72177
                       DecorateInfo(unittest.skip("Skipped! CUDA/ROCm memory exception"), 'TestGradients', 'test_fn_fwgrad_bwgrad',
                                    device_type='cuda', dtypes=[torch.float64]),
                   )),
    BinaryUfuncInfo('floor_divide',
                    dtypes=all_types_and(torch.half, torch.bfloat16),
                    sample_inputs_func=sample_inputs_binary_pwise,
                    supports_autograd=False,
                    rhs_make_tensor_kwargs=dict(exclude_zero=True),
                    ),
    UnaryUfuncInfo('frexp',
                   op=torch.frexp,
                   ref=np.frexp,
                   dtypes=floating_types_and(torch.half, torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   # skip testing torch.frexp as it is not supported by ROCm platform yet
                   decorators=[],
                   supports_out=False,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   skips=(
                       # skips below tests as torch.frexp returns tuple-like (mantissa, exponent) as outputs,
                       # while theses tests currently requires output to a single tensor.
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_batch_vs_slicing'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_contig_vs_every_other'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_contig_vs_transposed'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_non_contig_expand'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_variant_consistency'),

                       # skips test_reference_numerics due to error in Windows CI.
                       # The np.frexp returns exponent as np.intc dtype on Windows platform,
                       # and np.intc does not have the correspond torch dtype
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    active_if=IS_WINDOWS),
                   )),
    BinaryUfuncInfo('ge',
                    aliases=('greater_equal',),
                    dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
                    always_returns_bool=True,
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('geqrf',
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_linalg_qr_geqrf,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],),
    BinaryUfuncInfo('gt',
                    aliases=('greater',),
                    dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
                    always_returns_bool=True,
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_comparison_ops),
    UnaryUfuncInfo('imag',
                   ref=np.imag,
                   dtypes=complex_types(),
                   supports_out=False,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   # See https://github.com/pytorch/pytorch/issues/66357
                   # RuntimeError: view_as_real doesn't work on unresolved conjugated tensors.
                   check_batched_forward_grad=False,
                   skips=(
                       # Skip since real and imag don't have out variants.
                       DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_out_arg_all_dtypes'),
                   )),
    OpInfo('gradient',
           dtypes=floating_and_complex_types_and(torch.int8, torch.int16,
                                                 torch.int32, torch.int64,
                                                 torch.bfloat16, torch.half),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # following tests give a runtime error with undefined value tensor
               # see discussion : https://github.com/pytorch/pytorch/issues/56660
               # RuntimeError:
               # Arguments for call are not valid.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32, torch.complex64)),  # noqa: B950
           ),
           supports_inplace_autograd=False,
           sample_inputs_func=sample_inputs_gradient),
    OpInfo('inverse',
           op=torch.inverse,
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('isin',
           dtypes=all_types(),
           dtypesIfCUDA=all_types_and(torch.half),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_isin),
    OpInfo('kthvalue',
           dtypes=all_types_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_kthvalue,
           error_inputs_func=error_inputs_kthvalue),
    BinaryUfuncInfo('le',
                    aliases=('less_equal',),
                    dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
                    always_returns_bool=True,
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('linalg.det',
           op=torch.linalg.det,
           aliases=('det', ),
           dtypes=floating_and_complex_types(),
           backward_dtypes=floating_and_complex_types(),
           aten_name='linalg_det',
           sample_inputs_func=sample_inputs_linalg_det,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack, skipCUDAIfRocm,
                       DecorateInfo(toleranceOverride({torch.complex64: tol(atol=1e-3, rtol=1e-3)}))],
           check_batched_gradgrad=False,
           supports_inplace_autograd=False),
    OpInfo('linalg.det',
           op=torch.linalg.det,
           variant_test_name='singular',
           aliases=('det', ),
           dtypes=double_types(),
           backward_dtypes=double_types(),
           aten_name='linalg_det',
           sample_inputs_func=sample_inputs_linalg_det_singular,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack, skipCUDAIfRocm,
                       DecorateInfo(toleranceOverride({torch.complex64: tol(atol=1e-3, rtol=1e-3)}))],
           check_batched_gradgrad=False,
           supports_inplace_autograd=False,
           skips=(
               # These tests started breaking after touching the SVD.
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_grad', device_type='cpu',
                            dtypes=(torch.complex128,), active_if=IS_WINDOWS),
               # For complex dtypes: Will be removed once https://github.com/pytorch/pytorch/issues/62328 is fixed
               # Probable fix (open PR): https://github.com/pytorch/pytorch/pull/62570
               # Illegal Memory Access failure: https://github.com/pytorch/pytorch/issues/72203
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_grad', device_type='cuda'),
               # Illegal Memory Access failure: https://github.com/pytorch/pytorch/issues/72204
               DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_view', device_type='cuda'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_dtypes'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_gradgrad'),
           )),
    OpInfo('linalg.cholesky',
           aten_name='linalg_cholesky',
           dtypes=floating_and_complex_types(),
           # TODO: RuntimeError: While computing batched gradients,
           # got: vmap: Calling Tensor.as_strided is not supported
           # unless the batch dims being vmapped over are at the front of the tensor (in memory layout).
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_linalg_cholesky,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           ),
    OpInfo('linalg.cholesky_ex',
           aten_name='linalg_cholesky_ex',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_linalg_cholesky,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           ),
    OpInfo('linalg.cond',
           aten_name='linalg_cond',
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_cond,
           check_batched_gradgrad=False,
           check_batched_forward_grad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],),
    OpInfo('linalg.eig',
           aten_name='linalg_eig',
           op=torch.linalg.eig,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_eig,
           check_batched_forward_grad=False,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # Forward-over-reverse gradgrad might be incorrect
               DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad'),),),
    OpInfo('linalg.eigvals',
           aten_name='linalg_eigvals',
           op=torch.linalg.eigvals,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_invertible,
           check_batched_forward_grad=False,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # Pre-existing condition; Needs to be fixed
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
           )),
    OpInfo('linalg.eigh',
           aten_name='linalg_eigh',
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_eigh,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           check_batched_forward_grad=False,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # Forward-over-reverse gradgrad might be incorrect
               DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad',
                            dtypes=complex_types()),),
           ),
    OpInfo('linalg.eigvalsh',
           aten_name='linalg_eigvalsh',
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_eigh,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           check_batched_forward_grad=False,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # Pre-existing condition; Needs to be fixed
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
           )),
    OpInfo('linalg.householder_product',
           aten_name='linalg_householder_product',
           op=torch.linalg.householder_product,
           aliases=('orgqr', ),
           dtypes=floating_and_complex_types(),
           # TODO: backward uses in-place operations that vmap doesn't like
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           check_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_householder_product,
           decorators=[
               skipCUDAIfNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack,
               DecorateInfo(toleranceOverride({torch.complex64: tol(atol=1e-3, rtol=1e-3)})),
           ]),
    OpInfo('linalg.lstsq',
           aten_name='linalg_lstsq',
           dtypes=floating_and_complex_types(),
           supports_out=True,
           sample_inputs_func=sample_inputs_linalg_lstsq,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           skips=(
               # we skip gradient checks for this suite as they are tested in
               # variant_test_name='grad_oriented'
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients'),
           )),
    OpInfo('linalg.lstsq',
           aten_name='linalg_lstsq',
           variant_test_name='grad_oriented',
           # gradchecks for forward AD fails with multi-Tensor outputs
           op=lambda a, b, driver: torch.linalg.lstsq(a, b, driver=driver)[0],
           supports_out=False,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_lstsq,
           supports_autograd=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           skips=(
               # tests do not work with passing lambda for op
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
               DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),
               DecorateInfo(unittest.expectedFailure, 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )),
    OpInfo('linalg.matrix_power',
           aliases=('matrix_power',),
           aten_name='linalg_matrix_power',
           dtypes=floating_and_complex_types(),
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           check_batched_grad=False,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, skipCUDAIfRocm],
           sample_inputs_func=sample_inputs_linalg_matrix_power,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           ),
    OpInfo('linalg.multi_dot',
           # Need this lambda because gradcheck does not work with TensorList inputs
           aten_name='linalg_multi_dot',
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, *[torch.bfloat16] if CUDA11OrLater else []),
           supports_inplace_autograd=False,
           # Batched grad checks fail for empty input tensors (see https://github.com/pytorch/pytorch/issues/53407)
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_linalg_multi_dot,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           skips=(
               # https://github.com/pytorch/pytorch/issues/67470
               DecorateInfo(unittest.skip("67470!"), 'TestCommon', 'test_noncontiguous_samples'),
               # Fails on XLA.
               # AssertionError: False is not true : Tensors failed to compare as equal!
               DecorateInfo(unittest.expectedFailure, 'TestOpInfo', device_type='xla', dtypes=(torch.long,)),
           )),
    OpInfo('linalg.norm',
           op=torch.linalg.norm,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
           sample_inputs_func=sample_inputs_linalg_norm,
           aten_name='linalg_norm',
           skips=(
               # Pre-existing condition; Needs to be fixed
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
               # Expected RuntimeError when calling with input.device=cpu and out.device=cuda
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),
           )),
    OpInfo('linalg.matrix_norm',
           aten_name='linalg_matrix_norm',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
           sample_inputs_func=sample_inputs_linalg_matrix_norm,
           skips=(
               # Pre-existing condition; Needs to be fixed
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
               # Expected RuntimeError when calling with input.device=cpu and out.device=cuda
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),
           )),
    OpInfo('linalg.qr',
           aten_name='linalg_qr',
           op=torch.linalg.qr,
           dtypes=floating_and_complex_types(),
           # batched gradients do not work for empty inputs
           # https://github.com/pytorch/pytorch/issues/50743#issuecomment-767376085
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # See https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_linalg_qr_geqrf,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.slogdet',
           aten_name='linalg_slogdet',
           op=torch.linalg.slogdet,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_slogdet,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],),
    OpInfo('linalg.vector_norm',
           op=torch.linalg.vector_norm,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           sample_inputs_func=sample_inputs_linalg_vector_norm,
           aten_name='linalg_vector_norm'),
    UnaryUfuncInfo('log',
                   ref=np.log,
                   domain=(0, None),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=IS_WINDOWS),
                   ),
                   # log(z)->-inf for |z|->0
                   reference_numerics_filter=NumericsFilter(condition=lambda x: torch.abs(x) < 0.1, safe_val=1)),
    UnaryUfuncInfo('log10',
                   ref=np.log10,
                   domain=(0, None),
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   assert_autodiffed=True,
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=IS_WINDOWS),
                   ),
                   # log10(z)->-inf for |z|->0
                   reference_numerics_filter=NumericsFilter(condition=lambda x: torch.abs(x) < 0.1, safe_val=1)),
    UnaryUfuncInfo('log1p',
                   ref=np.log1p,
                   aliases=('special.log1p',),
                   domain=(-1, None),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
                   skips=(
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   ),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   assert_autodiffed=True),
    UnaryUfuncInfo('log2',
                   ref=np.log2,
                   domain=(0, None),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.cfloat, torch.cdouble]),
                   ),
                   # log2(z)->-inf for |z|->0
                   reference_numerics_filter=NumericsFilter(condition=lambda x: torch.abs(x) < 0.1, safe_val=1)),
    BinaryUfuncInfo('ldexp',
                    dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    supports_inplace_autograd=False,
                    sample_inputs_func=sample_inputs_binary_pwise,
                    promotes_int_to_float=True,
                    supports_out=True,
                    skips=(
                        # RuntimeError: mul(): functions with out=... arguments don't support
                        # automatic differentiation, but one of the arguments requires grad
                        # https://github.com/pytorch/pytorch/issues/68966
                        DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'),
                        DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'),
                        DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'),
                        DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'),
                        # FIXME: ldexp does not accept scalar inputs
                        DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_broadcast_python_scalar'),
                    ),
                    decorators=[
                        DecorateInfo(
                            toleranceOverride({
                                torch.complex64: tol(atol=1e-05, rtol=1e-05)
                            }),
                            'TestCommon', device_type='cpu',
                        ),
                    ], ),
    OpInfo('logaddexp',
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.bfloat16),
           dtypesIfROCM=floating_types_and(torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=lambda op_info, device, dtype, requires_grad=False, **kwargs:
           (SampleInput(make_tensor((S, S), device, dtype, requires_grad=requires_grad),
                        args=(make_tensor((S, S), device, dtype, requires_grad=requires_grad),)),)),
    OpInfo('logaddexp2',
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.bfloat16),
           dtypesIfROCM=floating_types_and(torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=lambda op_info, device, dtype, requires_grad=False, **kwargs:
           (SampleInput(make_tensor((S, S), device, dtype, requires_grad=requires_grad),
                        args=(make_tensor((S, S), device, dtype, requires_grad=requires_grad),)),)),
    UnaryUfuncInfo('logical_not',
                   ref=np.logical_not,
                   decorators=(precisionOverride({torch.bfloat16: 7e-1,
                                                  torch.float16: 5e-1}),),
                   dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   supports_autograd=False,
                   skips=(
                       # The function variant always returns BoolTensor
                       # while the inplace variant preserves the input dtype.
                       # >>> t = torch.randn(3)
                       # >>> torch.logical_not(t)
                       # tensor([False, False, False])
                       # >>> torch.logical_not(t).dtype
                       # torch.bool
                       # >>> t.logical_not_().dtype
                       # torch.float32
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_variant_consistency',
                                    dtypes=all_types_and_complex_and(torch.half, torch.bfloat16)),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_variant_consistency_eager',
                                    dtypes=all_types_and_complex_and(torch.half, torch.bfloat16)),
                   )),
    BinaryUfuncInfo('lt',
                    aliases=('less',),
                    dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
                    always_returns_bool=True,
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('linalg.lu_factor',
           aten_name='linalg_lu_factor',
           op=torch.linalg.lu_factor,
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_linalg_lu_factor,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack]),
    OpInfo('linalg.lu_factor_ex',
           aten_name='linalg_lu_factor_ex',
           op=torch.linalg.lu_factor_ex,
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_linalg_lu_factor,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack]),
    OpInfo('lu',
           op=torch.lu,
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=False,  # need: lu_unpack
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           supports_out=False,
           sample_inputs_func=sample_inputs_lu,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
           skips=(
               # we skip jit tests because `lu` is a torch function
               # RuntimeError:
               # 'Tensor (inferred)' object has no attribute or method 'lu'.:
               # File "<string>", line 3
               # def the_method(i0):
               #     return i0.lu(True, True)
               #            ~~~~~ <--- HERE
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('lu_solve',
           op=torch.lu_solve,
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=False,  # need: lu_unpack
           # See https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_lu_solve,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # RuntimeError: lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_noncontiguous_samples'),  # noqa: B950
               DecorateInfo(unittest.skip("Tests different backward implementations"),
                            "TestCommon", "test_floating_inputs_are_differentiable"),),
           ),
    OpInfo('lu_unpack',
           op=torch.lu_unpack,
           dtypes=floating_and_complex_types(),
           supports_inplace_autograd=False,
           # we use in-place operations which cannot be avoided.
           # This causes vmap failures, hence we skip batched gradient checks
           check_batched_grad=False,
           supports_out=True,
           sample_inputs_func=sample_inputs_lu_unpack,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # LU_pivots is expected to be a contiguous tensor
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_noncontiguous_samples'),  # noqa: B950
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_gradgrad', device_type='cuda'),
           )),
    OpInfo('masked_fill',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_masked_fill,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           check_batched_forward_grad=False,
           supports_out=False),
    OpInfo('masked_scatter',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_masked_scatter,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           supports_out=False),
    OpInfo('masked_select',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_masked_select),
    OpInfo('matrix_exp',
           dtypes=floating_and_complex_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           aliases=('linalg.matrix_exp',),
           sample_inputs_func=sample_inputs_matrix_exp,
           # Needs to construct a 2nx2n matrix by copy_ ing into it
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           supports_out=False,
           ),
    OpInfo('matmul',
           aliases=('linalg.matmul',),
           dtypes=all_types_and_complex_and(torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_types_and(torch.half, torch.bfloat16),
           backward_dtypesIfCUDA=floating_and_complex_types_and(torch.float16,
                                                                *[torch.bfloat16] if (SM60OrLater and CUDA11OrLater) else []),
           assert_autodiffed=True,
           assert_jit_shape_analysis=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           check_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_matmul,
           decorators=[
               # ROCm intermittently fails the test with standard atol/rtol
               DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-4, rtol=0)}),
                            'TestCommon', 'test_noncontiguous_samples',
                            active_if=TEST_WITH_ROCM), ],
           skips=(
               # https://github.com/pytorch/pytorch/issues/67470
               DecorateInfo(unittest.skip("67470!"),
                            'TestCommon', 'test_noncontiguous_samples',
                            device_type='cpu', dtypes=(torch.long,)),
               # AssertionError: False is not true : Tensors failed to compare as equal!
               DecorateInfo(unittest.expectedFailure, 'TestOpInfo',
                            device_type='xla', dtypes=(torch.long,)),
           )),
    OpInfo('max',
           variant_test_name='reduction_with_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_reduction_with_dim,
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True),
    OpInfo('max',
           variant_test_name='reduction_no_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_max_min_reduction_no_dim),
    OpInfo('median',
           dtypes=all_types_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16),
           # TODO: some signatures of median do support out
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=partial(sample_inputs_reduction, supports_multiple_dims=False)),
    OpInfo('nanmedian',
           dtypes=all_types_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16),
           # TODO: some signatures of nanmedian do support out
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=partial(sample_inputs_reduction, supports_multiple_dims=False)),
    OpInfo('var_mean',
           dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_reduction, supports_multiple_dims=False),
           backward_dtypes=floating_types_and(torch.half),
           backward_dtypesIfCPU=floating_types_and(torch.half, torch.bfloat16),
           backward_dtypesIfCUDA=floating_types_and(torch.half),
           # TODO: some signatures of var_mean do support out
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=False,  # Need: var_mean
           skips=(
               # https://github.com/pytorch/pytorch/issues/67539
               DecorateInfo(unittest.skip("67539"), 'TestCommon', 'test_noncontiguous_samples',
                            active_if=TEST_WITH_ASAN, device_type='cpu'),
               # TODO: FIXME: complex inputs requiring grad error in forward
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_dtypes'),
               # TODO: review with var_mean tests in test_autograd.py
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_grad'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_gradgrad'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD'),
               # Division by zero, may be related to above?
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_fwgrad_bwgrad'))),
    OpInfo('std_mean',
           dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_reduction, supports_multiple_dims=False),
           backward_dtypes=floating_types_and(torch.half),
           backward_dtypesIfCPU=floating_types_and(torch.half, torch.bfloat16),
           backward_dtypesIfCUDA=floating_types_and(torch.half),
           # TODO: some signatures of std_mean do support out
           supports_out=False,
           supports_forward_ad=True,  # Supports only certain variants?
           supports_fwgrad_bwgrad=False,  # Need: std_mean
           skips=(
               # https://github.com/pytorch/pytorch/issues/67539
               DecorateInfo(unittest.skip("67539"), 'TestCommon', 'test_noncontiguous_samples',
                            active_if=TEST_WITH_ASAN, device_type='cpu'),
               # TODO: FIXME: complex inputs requiring grad error in forward
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_dtypes'),
               # TODO: fix along with var_mean autograd tests
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_grad'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_gradgrad'),
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD'),
               # Division by zero, may be related to above?
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_fwgrad_bwgrad'))),
    OpInfo('meshgrid',
           variant_test_name='variadic_tensors',
           ref=np.meshgrid,
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.bool, torch.float16),
           sample_inputs_func=partial(sample_inputs_meshgrid, variant='variadic'),
           skips=[
               # JIT does not support variadic tensors.
               # RuntimeError: input->type()->kind() == TypeKind::OptionalType
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":252,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # meshgrid is defined in torch.functional to take a
               # variadic list of tensors. Variadic parameters are not
               # compatible with the normalize operator tests.
               DecorateInfo(unittest.skip("Skipped!"), 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),
               # Skip operator schema test because this is a functional and not an operator
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           ],
           supports_out=False,
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True),
    OpInfo('meshgrid',
           variant_test_name='list_of_tensors',
           # Unlike the variant above, we do not use np.meshgrid as a
           # ref since it does not officially support list of numpy
           # arrays.
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.bool, torch.float16),
           sample_inputs_func=partial(sample_inputs_meshgrid, variant='list'),
           skips=[
               # meshgrid is defined in torch.functional to take a
               # variadic list of tensors. Variadic parameters are not
               # compatible with the normalize operator tests.
               DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),
           ],
           assert_autodiffed=True,
           supports_out=False,
           autodiff_nonfusible_nodes=[],
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True),
    OpInfo('min',
           variant_test_name='reduction_with_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_reduction_with_dim,
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True),
    OpInfo('min',
           variant_test_name='reduction_no_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_max_min_reduction_no_dim),
    OpInfo('quantile',
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_reduction_quantile,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # See https://github.com/pytorch/pytorch/issues/66357
           # Relies on copy_ to broadcast, but the forward AD path calls broadcast_to which
           # does not have a batching rule in core
           check_batched_forward_grad=False),
    OpInfo('nanquantile',
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_reduction_quantile,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # See https://github.com/pytorch/pytorch/issues/66357
           # Relies on copy_ to broadcast, but the forward AD path calls broadcast_to which
           # does not have a batching rule in core
           check_batched_forward_grad=False),
    BinaryUfuncInfo(
        'max',
        aliases=('maximum',),
        variant_test_name='binary',
        dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
        sample_inputs_func=sample_inputs_max_min_binary,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        assert_autodiffed=True,
        ref=np.maximum,
        skips=(
            # FIXME: maximum does not accept scalar inputs
            DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_broadcast_python_scalar'),
            # TODO: FIXME: RuntimeError: "max_elementwise_cuda" not implemented for 'ComplexFloat'
            DecorateInfo(unittest.expectedFailure,
                         'TestBinaryUfuncs',
                         'test_type_promotion',
                         device_type='cuda'),
        ),
    ),
    BinaryUfuncInfo(
        'maximum',
        dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_max_min_binary,
        ref=np.maximum,
        skips=(
            # FIXME: maximum does not accept scalar inputs
            DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_broadcast_python_scalar'),
            # TODO: FIXME: RuntimeError: "max_elementwise_cuda" not implemented for 'ComplexFloat'
            DecorateInfo(unittest.expectedFailure,
                         'TestBinaryUfuncs',
                         'test_type_promotion',
                         device_type='cuda'),
        ),
    ),
    BinaryUfuncInfo(
        'min',
        aliases=('minimum',),
        variant_test_name='binary',
        dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
        sample_inputs_func=sample_inputs_max_min_binary,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        assert_autodiffed=True,
        ref=np.minimum,
        skips=(
            # FIXME: min does not accept scalar inputs
            DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_broadcast_python_scalar'),
            # TODO: FIXME: RuntimeError: "min_elementwise_cuda" not implemented for 'ComplexFloat'
            DecorateInfo(unittest.expectedFailure,
                         'TestBinaryUfuncs',
                         'test_type_promotion',
                         device_type='cuda'),
        ),
    ),
    BinaryUfuncInfo(
        'minimum',
        dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_max_min_binary,
        ref=np.minimum,
        skips=(
            # FIXME: minimum does not accept scalar inputs
            DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_broadcast_python_scalar'),
            # TODO: FIXME: RuntimeError: "min_elementwise_cuda" not implemented for 'ComplexFloat'
            DecorateInfo(unittest.expectedFailure,
                         'TestBinaryUfuncs',
                         'test_type_promotion',
                         device_type='cuda'),
        ),
    ),
    BinaryUfuncInfo('logical_and',
                    ref=np.logical_and,
                    dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                    sample_inputs_func=sample_inputs_binary_pwise,
                    supports_autograd=False,
                    always_returns_bool=True,
                    skips=(
                        # FIXME: logical_and does not accept scalar inputs
                        DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_broadcast_python_scalar'),
                    )),
    BinaryUfuncInfo('logical_or',
                    ref=np.logical_or,
                    dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                    sample_inputs_func=sample_inputs_binary_pwise,
                    supports_autograd=False,
                    always_returns_bool=True,
                    skips=(
                        # FIXME: logical_or does not accept scalar inputs
                        DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_broadcast_python_scalar'),
                    )),
    BinaryUfuncInfo('logical_xor',
                    ref=np.logical_xor,
                    dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                    sample_inputs_func=sample_inputs_binary_pwise,
                    supports_autograd=False,
                    always_returns_bool=True,
                    skips=(
                        # FIXME: logical_xor does not accept scalar inputs
                        DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_broadcast_python_scalar'),
                    )),
    BinaryUfuncInfo('bitwise_or',
                    ref=np.bitwise_or,
                    dtypes=integral_types_and(torch.bool),
                    sample_inputs_func=sample_inputs_binary_pwise,
                    supports_autograd=False,
                    skips=(
                        # TODO: FIXME: RuntimeError: "bitwise_or_cuda" not implemented for 'Half'
                        DecorateInfo(unittest.expectedFailure,
                                     'TestBinaryUfuncs',
                                     'test_type_promotion',
                                     device_type='cuda'),
                    )),
    BinaryUfuncInfo('bitwise_xor',
                    ref=np.bitwise_xor,
                    dtypes=integral_types_and(torch.bool),
                    sample_inputs_func=sample_inputs_binary_pwise,
                    supports_autograd=False,
                    skips=(
                        # TODO: FIXME: RuntimeError: "bitwise_xor_cuda" not implemented for 'Half'
                        DecorateInfo(unittest.expectedFailure,
                                     'TestBinaryUfuncs',
                                     'test_type_promotion',
                                     device_type='cuda'),
                    )),
    BinaryUfuncInfo('heaviside',
                    ref=lambda a, b: (
                        # necessary because np.heaviside incorrectly returns float64 when passed args of dtype int64
                        np.int64(np.heaviside(a, b)) if a.dtype == np.int64 and b.dtype == np.int64 else np.heaviside(a, b)
                    ),
                    dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16),
                    sample_inputs_func=sample_inputs_binary_pwise,
                    supports_autograd=False,
                    # FIXME: heaviside does not accept scalar inputs
                    skips=(
                        # RuntimeError: heaviside is not yet implemented for tensors with different dtypes.
                        DecorateInfo(unittest.expectedFailure,
                                     'TestBinaryUfuncs',
                                     'test_type_promotion'),
                        # PyTorch's heaviside does not appear to propagate NaNs
                        DecorateInfo(unittest.skip("Skipped!"),
                                     'TestBinaryUfuncs',
                                     'test_reference_numerics_extremal_values'),
                        DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_broadcast_python_scalar'),
                    )),
    BinaryUfuncInfo('lcm',
                    ref=np.lcm,
                    dtypes=integral_types_and(),
                    sample_inputs_func=sample_inputs_binary_pwise,
                    supports_autograd=False,
                    skips=(
                        # TODO: FIXME: lcm doesn't support scalars
                        DecorateInfo(unittest.expectedFailure,
                                     'TestBinaryUfuncs',
                                     'test_broadcast_python_scalar'),
                    )),
    BinaryUfuncInfo('gcd',
                    ref=np.gcd,
                    dtypes=integral_types_and(),
                    sample_inputs_func=sample_inputs_binary_pwise,
                    supports_autograd=False,
                    skips=(
                        DecorateInfo(unittest.expectedFailure,
                                     'TestBinaryUfuncs',
                                     'test_reference_numerics_small_values',
                                     dtypes=(torch.int8,)),
                        # TODO: FIXME: jiterator doesn't support non-tensor inputs
                        DecorateInfo(unittest.expectedFailure,
                                     'TestBinaryUfuncs',
                                     'test_broadcast_python_scalar'),)),
    BinaryUfuncInfo('isclose',
                    ref=np.isclose,
                    dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                    sample_inputs_func=sample_inputs_isclose,
                    supports_autograd=False,
                    supports_out=False,
                    skips=(
                        # RuntimeError: Short did not match Int
                        DecorateInfo(unittest.expectedFailure,
                                     'TestBinaryUfuncs',
                                     'test_type_promotion'),
                        DecorateInfo(unittest.skip("Skipped!"),
                                     'TestBinaryUfuncs',
                                     'test_reference_numerics_extremal_values'),
                        # FIXME: isclose does not accept scalar inputs
                        DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_broadcast_python_scalar'),
                    )),
    # `softmax` supports different dtypes based on whether `dtype` argument,
    # is passed or not. Hence two OpInfo entries, one with dtype and other without.
    # https://github.com/pytorch/pytorch/issues/68752
    OpInfo('softmax',
           aliases=('special.softmax', 'nn.functional.softmax',),
           aten_name='softmax',
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_softmax_variant,
           assert_jit_shape_analysis=True,
           assert_autodiffed=True,
           supports_out=False),
    OpInfo('softmax',
           aliases=('special.softmax', 'nn.functional.softmax',),
           variant_test_name="with_dtype",
           aten_name='softmax',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_softmax_variant, with_dtype=True),
           assert_autodiffed=True,
           supports_out=False),
    # `softmin` supports different dtypes based on whether `dtype` argument,
    # is passed or not. Hence two OpInfo entries, one with dtype and other without.
    # https://github.com/pytorch/pytorch/issues/68752
    OpInfo('nn.functional.softmin',
           aten_name='softmin',
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_softmax_variant,
           assert_jit_shape_analysis=False,
           assert_autodiffed=False,
           supports_out=False),
    OpInfo('nn.functional.softmin',
           variant_test_name="with_dtype",
           aten_name='softmin',
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_softmax_variant, with_dtype=True),
           assert_autodiffed=False,
           supports_out=False),
    OpInfo(
        "nn.functional.cross_entropy",
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        sample_inputs_func=sample_inputs_cross_entropy,
        supports_out=False,
        decorators=(
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-5, rtol=1e-3)}),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="cpu",
            ),
            # FIXME Derivative wrt weights is not implemented
            DecorateInfo(unittest.expectedFailure, "TestCommon",
                         "test_floating_inputs_are_differentiable")
        ),
        skips=(
            # AssertionError: False is not true : Scalars failed to compare as equal! 0 != 1536
            # test_ops.TestJitCUDA.test_variant_consistency_jit_nn_functional_cross_entropy_cuda_float32 leaked
            # 1536 bytes CUDA memory on device 0
            DecorateInfo(
                unittest.expectedFailure,
                "TestJit",
                "test_variant_consistency_jit",
                device_type="cuda",
            ),
        )
    ),
    OpInfo('nn.functional.normalize',
           dtypes=floating_and_complex_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_normalize),
    OpInfo('aminmax',
           ref=lambda x, dim=None, keepdim=False: (np.amin(x, axis=dim, keepdims=keepdim), np.amax(x, axis=dim, keepdims=keepdim)),
           dtypes=all_types_and(torch.bool),
           dtypesIfCUDA=all_types_and(torch.bool, torch.float16, torch.bfloat16),
           decorators=(onlyNativeDeviceTypes,),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_aminmax),
    OpInfo('as_strided',
           op=lambda x, size, stride, storage_offset=0:
               torch.as_strided(x, size, stride, storage_offset=storage_offset),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # vmap does not support inplace views
           check_inplace_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_as_strided,
           skips=(
               # AssertionError: False is not true : Tensors failed to compare as equal!
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_noncontiguous_samples'),
               # AssertionError: False is not true : Scalars failed to compare as equal!
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'),)),
    OpInfo('nn.functional.cosine_similarity',
           aten_name="cosine_similarity",
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_cosine_similarity),
    OpInfo('nn.functional.adaptive_avg_pool1d',
           dtypes=floating_types(),
           dtypesIfCPU=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_adaptive_avg_pool1d),
    OpInfo('nn.functional.adaptive_avg_pool2d',
           dtypes=floating_types(),
           dtypesIfCPU=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           decorators=(
               # RuntimeError:
               # adaptive_avg_pool2d(Tensor input, int[2] output_size) -> (Tensor):
               # Expected a value of type 'List[int]' for argument 'output_size' but
               # instead found type 'Tuple[NoneType, int]'. :
               #   File "<string>", line 3
               # def the_method(i0):
               #     return torch.nn.functional.adaptive_avg_pool2d(i0, (None, 7))
               #            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_adaptive_avg_pool2d),
    OpInfo('nn.functional.adaptive_avg_pool3d',
           dtypes=floating_types_and(torch.half),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           decorators=(
               # RuntimeError:
               # adaptive_avg_pool3d(Tensor input, int[3] output_size) -> (Tensor):
               # Expected a value of type 'List[int]' for argument 'output_size' but
               # instead found type 'Tuple[NoneType, NoneType, NoneType]'. :
               #   File "<string>", line 3
               #
               # def the_method(i0):
               #     return torch.nn.functional.adaptive_avg_pool3d(i0, (None, None, None))
               #            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
               #
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_adaptive_avg_pool3d),
    OpInfo('nn.functional.adaptive_max_pool1d',
           dtypes=floating_types(),
           dtypesIfCPU=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # got: Batching rule not implemented for aten::flatten.using_ints
           check_batched_forward_grad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_adaptive_max_pool1d),
    OpInfo('nn.functional.adaptive_max_pool2d',
           dtypes=floating_types(),
           dtypesIfCPU=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           decorators=(
               # RuntimeError:
               # adaptive_max_pool2d(Tensor input, int[2] output_size) -> (Tensor):
               # Expected a value of type 'List[int]' for argument 'output_size' but
               # instead found type 'Tuple[NoneType, int]'. :
               #   File "<string>", line 3
               # def the_method(i0):
               #     return torch.nn.functional.adaptive_max_pool2d(i0, (None, 7))
               #            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # got: Batching rule not implemented for aten::flatten.using_ints
           check_batched_forward_grad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_adaptive_max_pool2d),
    OpInfo('nn.functional.adaptive_max_pool3d',
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           decorators=(
               # RuntimeError:
               # adaptive_max_pool3d(Tensor input, int[3] output_size) -> (Tensor):
               # Expected a value of type 'List[int]' for argument 'output_size' but
               # instead found type 'Tuple[NoneType, NoneType, NoneType]'. :
               #   File "<string>", line 3
               #
               # def the_method(i0):
               #     return torch.nn.functional.adaptive_max_pool3d(i0, (None, None, None))
               #            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
               #
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # got: Batching rule not implemented for aten::flatten.using_ints
           check_batched_forward_grad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_adaptive_max_pool3d),
    OpInfo('nn.functional.avg_pool1d',
           aten_name='avg_pool1d',
           supports_autograd=True,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           dtypes=floating_types_and(torch.int64),
           dtypesIfCPU=floating_types_and(torch.int64, torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_avgpool1d),
    OpInfo('nn.functional.avg_pool3d',
           aten_name='avg_pool3d',
           supports_autograd=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False,
           dtypes=floating_types_and(torch.int64),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_avgpool3d),
    OpInfo('nn.functional.relu',
           aten_name="relu",
           supports_autograd=True,
           dtypes=all_types_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_nn_activation_relu,
           supports_out=False,
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True),
    OpInfo('nn.functional.conv_transpose1d',
           aten_name='conv_transpose1d',
           aliases=('conv_transpose1d',),
           dtypes=floating_types_and(torch.int64),
           dtypesIfCUDA=floating_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           sample_inputs_func=sample_inputs_conv_transpose1d,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           decorators=[
               DecorateInfo(
                   toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1.3e-06), }),
                   'TestCommon', 'test_variant_consistency_eager', device_type='cuda')],
           skips=(
               # RuntimeError: !lhs.isAliasOf(rhs)INTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":104, please report a bug to PyTorch.
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False,),
    OpInfo('nn.functional.conv_transpose2d',
           aten_name='conv_transpose2d',
           aliases=('conv_transpose2d',),
           dtypes=floating_types_and(torch.int64),
           dtypesIfCUDA=floating_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           sample_inputs_func=sample_inputs_conv_transpose2d,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           decorators=[
               DecorateInfo(
                   toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1.3e-06), }),
                   'TestCommon', 'test_variant_consistency_eager', device_type='cuda')],
           skips=(
               # RuntimeError: !lhs.isAliasOf(rhs)INTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":104, please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False,),
    OpInfo('nn.functional.conv_transpose3d',
           aten_name='conv_transpose3d',
           aliases=('conv_transpose3d',),
           dtypes=floating_types_and(torch.int64),
           dtypesIfCUDA=floating_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           sample_inputs_func=sample_inputs_conv_transpose3d,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           decorators=[
               DecorateInfo(
                   toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1.3e-06), }),
                   'TestCommon', 'test_variant_consistency_eager', device_type='cuda'),
               DecorateInfo(
                   toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1.3e-06), }),
                   'TestCommon', 'test_noncontiguous_samples', device_type='cuda')],
           skips=(
               # RuntimeError: !lhs.isAliasOf(rhs)INTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":104, please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               DecorateInfo(unittest.skip("Skipped! RuntimeError: bias tensor has to be contiguous"), 'TestGradients',
                            'test_forward_mode_AD', device_type='cuda', active_if=TEST_WITH_ROCM),
           ),
           supports_out=False,),
    OpInfo('nn.functional.conv1d',
           aliases=('conv1d',),
           aten_name='conv1d',
           dtypes=floating_types_and(torch.int64),
           dtypesIfCUDA=floating_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           sample_inputs_func=sample_inputs_conv1d,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           skips=(
               # RuntimeError: !lhs.isAliasOf(rhs)INTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":103, please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False,),
    OpInfo('nn.functional.conv2d',
           aliases=('conv2d',),
           aten_name='conv2d',
           dtypes=floating_types_and(torch.int64),
           dtypesIfCUDA=floating_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           sample_inputs_func=partial(sample_inputs_conv2d),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # RuntimeError: !lhs.isAliasOf(rhs)INTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":103, please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False,),
    OpInfo('nn.functional.group_norm',
           aten_name='group_norm',
           aliases=('group_norm',),
           ref=reference_group_norm,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           decorators=[
               # RuntimeError: Cannot insert a Tensor that requires grad as a constant.
               # Consider making it a parameter or input, or detaching the gradient
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,))
           ],
           sample_inputs_func=sample_inputs_group_norm,),
    OpInfo('nn.functional.instance_norm',
           # no ref because instance_norm will often have numerical instability (large numbers or nan)
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           decorators=[
               # RuntimeError: Cannot insert a Tensor that requires grad as a constant.
               # Consider making it a parameter or input, or detaching the gradient
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,))
           ],
           skips=(
               DecorateInfo(unittest.skip("We don't want to differentiate wrt running mean / std"),
                            "TestCommon", "test_floating_inputs_are_differentiable"),),
           sample_inputs_func=sample_inputs_instance_norm,),
    OpInfo('nn.functional.layer_norm',
           aten_name='layer_norm',
           aliases=('layer_norm',),
           ref=reference_layer_norm,
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           decorators=[
               DecorateInfo(
                   toleranceOverride({torch.float32: tol(atol=1e-05, rtol=1e-03)}),
                   'TestCommon', 'test_reference_testing'
               )
           ],
           sample_inputs_func=sample_inputs_layer_norm,),
    OpInfo('nn.functional.local_response_norm',
           dtypes=floating_types_and(torch.int64),
           dtypesIfCPU=floating_types_and(torch.int64, torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           decorators=[
               # RuntimeError: falseINTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185, please report a bug to PyTorch.
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,))
           ],
           sample_inputs_func=sample_inputs_local_response_norm,),
    OpInfo('nn.functional.pad',
           variant_test_name='constant',
           aten_name='constant_pad_nd',
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           sample_inputs_func=partial(sample_inputs_nn_pad, mode='constant'),
           supports_out=False),
    OpInfo('nn.functional.pad',
           variant_test_name='reflect',
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           dtypes=floating_and_complex_types(),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half),
           sample_inputs_func=partial(sample_inputs_nn_pad, mode='reflect'),
           skips=(
               # Doesn't have a corresponding aten operator.
               # RuntimeError: falseINTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185, please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),
           ),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           supports_out=False),
    OpInfo('nn.functional.pad',
           variant_test_name='replicate',
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           dtypes=floating_and_complex_types(),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half),
           sample_inputs_func=partial(sample_inputs_nn_pad, mode='replicate'),
           skips=(
               # Doesn't have a corresponding aten operator.
               # RuntimeError: falseINTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185, please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),
           ),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           supports_out=False),
    OpInfo('nn.functional.pad',
           variant_test_name='circular',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           sample_inputs_func=partial(sample_inputs_nn_pad, mode='circular'),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           check_batched_grad=False,
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           skips=(
               # Doesn't have a corresponding aten operator.
               # RuntimeError: falseINTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185, please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),
           ),
           supports_out=False),
    OpInfo('nn.functional.hardswish',
           aten_name="hardswish",
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_hardswish,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=False,  # Need: hardswish_backward
           supports_out=False,
           autodiff_nonfusible_nodes=["aten::hardswish"]),
    OpInfo('nn.functional.unfold',
           aten_name='im2col',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCPU=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_nn_unfold,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # RuntimeError: false
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='nearest',
           supports_autograd=True,
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True,
           dtypes=floating_types_and(torch.uint8),
           dtypesIfCUDA=floating_types_and(torch.half, torch.uint8),
           sample_inputs_func=partial(sample_inputs_interpolate, 'nearest'),
           skips=(
               # RuntimeError: false
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='linear',
           supports_autograd=True,
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.half),
           sample_inputs_func=partial(sample_inputs_interpolate, 'linear'),
           skips=(
               # RuntimeError: false
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='bilinear',
           supports_fwgrad_bwgrad=True,
           supports_autograd=True,
           supports_forward_ad=True,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.half),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=partial(sample_inputs_interpolate, 'bilinear'),
           skips=(
               # RuntimeError: false
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='bicubic',
           supports_autograd=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.half),
           sample_inputs_func=partial(sample_inputs_interpolate, 'bicubic'),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           skips=(
               # RuntimeError: false
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='trilinear',
           supports_autograd=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.half),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=partial(sample_inputs_interpolate, 'trilinear'),
           skips=(
               # RuntimeError: false
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.interpolate',
           aten_name="interpolate",
           variant_test_name='area',
           supports_autograd=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_interpolate, 'area'),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           skips=(
               # RuntimeError: false
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.upsample_bilinear',
           supports_autograd=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.half),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=partial(sample_inputs_upsample, 'bilinear'),
           skips=(
               # RuntimeError: false
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.upsample_nearest',
           supports_autograd=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           dtypes=floating_types_and(torch.uint8),
           dtypesIfCUDA=floating_types_and(torch.half, torch.uint8),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=partial(sample_inputs_upsample, 'nearest'),
           skips=(
               # RuntimeError: false
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":185,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           supports_out=False),
    OpInfo('nn.functional.leaky_relu',
           aliases=None,
           aten_name="leaky_relu",
           sample_inputs_func=sample_inputs_leaky_relu,
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_autograd=True,
           assert_autodiffed=True,
           supports_gradgrad=True,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=False,  # Need: leaky_relu_backward
           autodiff_nonfusible_nodes=["aten::leaky_relu"]),
    OpInfo('nn.functional.avg_pool2d',
           aten_name='avg_pool2d',
           supports_autograd=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False,
           dtypes=floating_types_and(torch.int64),
           dtypesIfCPU=floating_types_and(torch.int64, torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_avgpool2d),
    OpInfo('nn.functional.fractional_max_pool2d',
           supports_autograd=True,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           op=lambda input, *args, **kwargs:
               wrapper_set_seed(torch.nn.functional.fractional_max_pool2d, input, *args, **kwargs),
           # vmap does not support random operations
           check_batched_forward_grad=False,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.float16),
           test_neg_view=False,
           sample_inputs_func=sample_inputs_fractional_max_pool2d,
           decorators=(
               # FIXME: AssertionError: False is not true : Tensors failed to compare as equal!
               DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),
               # RuntimeError: input->type()->kind() == TypeKind::OptionalType
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":270
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'))),
    OpInfo('nn.functional.fractional_max_pool3d',
           supports_autograd=True,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           op=lambda input, *args, **kwargs:
               wrapper_set_seed(torch.nn.functional.fractional_max_pool3d, input, *args, **kwargs),
           # vmap does not support random operations
           check_batched_forward_grad=False,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.float16),
           test_neg_view=False,
           sample_inputs_func=sample_inputs_fractional_max_pool3d,
           decorators=(
               # FIXME: both derivatives are implemented incorrectly
               # https://github.com/pytorch/pytorch/issues/69322
               # RuntimeError: cannot reshape tensor of 0 elements into shape [0, 1, -1] because the
               # unspecified dimension size -1 can be any value and is ambiguous
               DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_gradgrad'),
               # FIXME: AssertionError: False is not true : Tensors failed to compare as equal!
               DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),
               # RuntimeError: input->type()->kind() == TypeKind::OptionalType
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":270
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),)),
    OpInfo('nn.functional.max_pool1d',
           aten_name='max_pool1d',
           supports_autograd=True,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # got: Batching rule not implemented for aten::flatten.using_ints
           check_batched_forward_grad=False,
           # TODO: add shape checks
           assert_jit_shape_analysis=False,
           dtypes=floating_types(),
           dtypesIfCPU=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           skips=(
               # Pre-existing condition; Needs to be fixed
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
           ),
           sample_inputs_func=sample_inputs_max_pool),
    OpInfo('nn.functional.max_pool2d',
           aten_name='max_pool2d',
           supports_autograd=True,
           # Vmap is not happy with non-contiguous (channels_last) inputs
           check_batched_gradgrad=False,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # got: Batching rule not implemented for aten::flatten.using_ints
           check_batched_forward_grad=False,
           assert_jit_shape_analysis=True,
           dtypes=floating_types(),
           dtypesIfCPU=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_max_pool),
    OpInfo('nn.functional.max_pool3d',
           aten_name='max_pool3d',
           supports_autograd=True,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # got: Batching rule not implemented for aten::flatten.using_ints
           check_batched_forward_grad=False,
           # TODO: add shape checks
           assert_jit_shape_analysis=False,
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           # TODO: investigate nondeterminism
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_max_pool),
    OpInfo('nn.functional.linear',
           aten_name='linear',
           supports_autograd=True,
           sample_inputs_func=sample_inputs_linear,
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfROCM=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=floating_and_complex_types_and(torch.float16,
                                                                *[torch.bfloat16] if CUDA11OrLater else []),
           # linear calls mm under the hood which is nondeterministic on CUDA
           # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # See https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           supports_out=False),
    OpInfo('nn.functional.bilinear',
           aten_name='bilinear',
           supports_autograd=True,
           sample_inputs_func=sample_inputs_bilinear,
           dtypes=all_types_and(torch.half, torch.bfloat16),
           dtypesIfROCM=floating_types_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=floating_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           skips=(
               # FIXME: bfloat16 backward support likely depends on CUDA11+ and SM53+
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_dtypes'),
           ),
           supports_forward_ad=False,
           supports_out=False),
    OpInfo('nn.functional.glu',
           aten_name='glu',
           supports_autograd=True,
           sample_inputs_func=sample_inputs_glu,
           dtypes=floating_types(),
           dtypesIfROCM=floating_types_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           backward_dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_forward_ad=False,
           supports_out=False),
    UnaryUfuncInfo(
        'nn.functional.elu',
        ref=lambda x, alpha=1.0, inplace=False:
            np.maximum(0., x) + np.minimum(0., alpha * (np.exp(x) - 1)),
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=False,  # Need: elu_backward
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        sample_kwargs=lambda device, dtype, input:
            ({'alpha': 0.8}, {'alpha': 0.8}),
        inplace_variant=lambda x, alpha=1.0:
            torch.nn.functional.elu(x, alpha, inplace=True),
        decorators=[
            # Not implemented yet
            DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_inplace_forward_mode_AD'),
            DecorateInfo(
                toleranceOverride({
                    torch.float16: tol(atol=1e-03, rtol=1.2e-03),
                    torch.bfloat16: tol(atol=1e-03, rtol=1.2e-03)
                }),
                'TestUnaryUfuncs', device_type='cuda',
            ), ],
    ),
    OpInfo(
        'nn.functional.prelu',
        ref=lambda x, weight:
            np.maximum(0., x) + np.minimum(0., x) *
            (weight if x.ndim == 1 else weight.reshape([weight.size if i == 1 else 1 for i in range(0, x.ndim)])),
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.float16),
        supports_forward_ad=False,
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        sample_inputs_func=sample_inputs_nn_functional_prelu,
        decorators=[
            # FIXME: second derivative is implemented but seems to be incorrect
            # https://github.com/pytorch/pytorch/issues/68760
            DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_gradgrad'),
            # RuntimeError: Cannot insert a Tensor that requires grad as a constant.
            # Consider making it a parameter or input, or detaching the gradient
            # https://github.com/pytorch/pytorch/issues/68752
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'), ],
    ),
    UnaryUfuncInfo(
        'nn.functional.celu',
        ref=lambda x, alpha=1.0, inplace=False:
            np.maximum(0., x) + np.minimum(0., alpha * (np.exp(x / alpha) - 1)),
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=False,  # Need: elu_backward
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        sample_kwargs=lambda device, dtype, input:
            ({'alpha': 0.8}, {'alpha': 0.8}),
        inplace_variant=lambda x, alpha=1.0:
            torch.nn.functional.celu(x, alpha, inplace=True),
        decorators=[
            # Not implemented yet
            DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_inplace_forward_mode_AD'),
            DecorateInfo(
                toleranceOverride({
                    torch.float16: tol(atol=1e-03, rtol=1.2e-03),
                    torch.bfloat16: tol(atol=1e-03, rtol=1.2e-03)
                }),
                'TestUnaryUfuncs', device_type='cuda',
            ), ],
    ),
    UnaryUfuncInfo(
        'nn.functional.rrelu',
        op=lambda input, *args, **kwargs:
            wrapper_set_seed(torch.nn.functional.rrelu, input, *args, **kwargs),
        ref=_NOTHING,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        gradcheck_wrapper=wrapper_set_seed,
        supports_forward_ad=False,
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        sample_kwargs=lambda device, dtype, input:
            ({'lower': 0., 'upper': 1.}, {'lower': 0., 'upper': 1.}),
        inplace_variant=lambda input, *args, **kwargs:
            wrapper_set_seed(partial(torch.nn.functional.rrelu, inplace=True), input, *args, **kwargs),
        decorators=[
            DecorateInfo(
                toleranceOverride({
                    torch.float16: tol(atol=1e-03, rtol=1.2e-03),
                    torch.bfloat16: tol(atol=1e-03, rtol=1.2e-03)
                }),
                'TestUnaryUfuncs', device_type='cuda',
            ),
            # Probably because we have used lambda for the op here
            # AssertionError: JIT Test does not execute any logic
            DecorateInfo(
                unittest.skip("Skipped!"),
                'TestJit', 'test_variant_consistency_jit'
            ), ],
    ),
    UnaryUfuncInfo(
        'nn.functional.selu',
        ref=lambda x, inplace=False:
            1.0507009873554804934193349852946 * (
                np.maximum(0., x) + np.minimum(0., 1.6732632423543772848170429916717 * (np.exp(x) - 1))
            ),
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,  # depends on 'elu'
        supports_fwgrad_bwgrad=False,  # Needs: elu_backward
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        inplace_variant=lambda x: torch.nn.functional.selu(x, inplace=True),
        decorators=[
            # Not implemented yet (depends on 'elu_')
            DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_inplace_forward_mode_AD'),
            DecorateInfo(
                toleranceOverride({
                    torch.float16: tol(atol=1e-2, rtol=1.8e-2),
                    torch.bfloat16: tol(atol=1e-2, rtol=1.8e-2)
                }),
                'TestUnaryUfuncs', device_type='cuda',
            ), ],
    ),
    UnaryUfuncInfo(
        'nn.functional.silu',
        ref=lambda x, inplace=False:
            x / (1 + np.exp(-x)),
        dtypes=floating_and_complex_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=False,
        supports_autograd=False,
        assert_autodiffed=False,
        supports_out=False,
        inplace_variant=lambda x: torch.nn.functional.silu(x, inplace=True),
        decorators=[
            DecorateInfo(
                toleranceOverride({
                    torch.float16: tol(atol=1e-3, rtol=1e-3),
                    torch.bfloat16: tol(atol=1e-4, rtol=1e-4)
                }),
                'TestUnaryUfuncs', device_type='cuda',
            ), ],
        skips=[
            # FIXME: numpy reference diverges: Comparing (nan+nanj) and (-0+0j)
            DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', dtypes=(torch.complex64,)), ],
    ),
    UnaryUfuncInfo(
        'nn.functional.hardsigmoid',
        ref=reference_hardsigmoid,
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=False,
        supports_forward_ad=True,
        supports_out=False,
        inplace_variant=partial(torch.nn.functional.hardsigmoid, inplace=True),
        decorators=[
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-04, rtol=0.001)}), 'TestUnaryUfuncs', device_type='cuda',), ],
        skips=[
            # still want to test that first derivative works though second derivative isn't supported
            DecorateInfo(unittest.expectedFailure, 'TestGradients', "test_inplace_gradgrad"),
            # produces 0 instead of nan on ROCM
            DecorateInfo(unittest.expectedFailure,
                         'TestUnaryUfuncs', "test_reference_numerics_extremal",
                         dtypes=(torch.bfloat16, torch.float16, torch.float32,), device_type='cuda',
                         active_if=(TEST_WITH_ROCM)), ]
    ),
    UnaryUfuncInfo(
        'nn.functional.logsigmoid',
        aten_name="log_sigmoid",
        ref=reference_logsigmoid,
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.float16),
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
    ),
    UnaryUfuncInfo(
        'nn.functional.mish',
        ref=lambda x: x * np.tanh(reference_softplus(x)),
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        inplace_variant=partial(torch.nn.functional.mish, inplace=True),
        decorators=[
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-03)}), 'TestUnaryUfuncs', device_type='cuda',), ],
    ),
    UnaryUfuncInfo(
        'nn.functional.softsign',
        ref=lambda x: x / (np.abs(x) + 1),
        dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
        dtypesIfCUDA=all_types_and_complex_and(torch.float16, torch.bfloat16, torch.bool),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        decorators=[
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1.3e-04)}), 'TestUnaryUfuncs',), ],
        skips=(
            DecorateInfo(unittest.expectedFailure, 'TestGradients',
                         "test_fn_fwgrad_bwgrad", dtypes=(torch.complex128,)),
            # pytorch computes (0+nanj), numpy computes (-5e-18-1j) for input (-501.-1.0000e+20j)
            DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs',
                         "test_reference_numerics_hard", dtypes=(torch.complex64,)),),
    ),
    UnaryUfuncInfo(
        'nn.functional.tanhshrink',
        ref=lambda x: x - np.tanh(x),
        dtypes=all_types_and_complex_and(torch.bfloat16),
        dtypesIfCUDA=all_types_and_complex_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_autograd=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        decorators=[
            DecorateInfo(
                toleranceOverride({torch.bfloat16: tol(atol=1e-02, rtol=1.6e-02)}), 'TestUnaryUfuncs',), ],
        skips=(
            # in each case, pytorch will produce a nan while numpy will not
            DecorateInfo(unittest.expectedFailure,
                         'TestUnaryUfuncs', "test_reference_numerics_normal",
                         dtypes=(torch.complex64,), active_if=(IS_MACOS)),
            DecorateInfo(unittest.expectedFailure,
                         'TestUnaryUfuncs', "test_reference_numerics_hard",
                         dtypes=(torch.complex64,), active_if=(IS_MACOS)),
            DecorateInfo(unittest.expectedFailure,
                         'TestUnaryUfuncs', "test_reference_numerics_extremal",
                         dtypes=(torch.complex64,), device_type='cpu',
                         active_if=(IS_MACOS or IS_WINDOWS)),)
    ),
    OpInfo(
        'nn.functional.threshold',
        ref=lambda x, threshold, value: np.where(x > threshold, x, value).astype(x.dtype),
        dtypes=all_types_and(torch.bfloat16),
        dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
        supports_autograd=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        assert_autodiffed=False,
        supports_gradgrad=True,
        supports_out=False,
        sample_inputs_func=sample_inputs_threshold,
    ),
    BinaryUfuncInfo('nextafter',
                    dtypes=floating_types_and(torch.bfloat16),
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_nextafter,
                    skips=(
                        # TypeError: nextafter(): argument 'other' (position 2) must be Tensor, not float
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs'),
                    )),
    OpInfo('topk',
           dtypes=all_types_and(torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.bfloat16, torch.float16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_topk),
    # Multiple variants for batch_norm to test with and without cuDNN disabled
    # See https://github.com/pytorch/pytorch/pull/63218#discussion_r688549391 for more details
    OpInfo('nn.functional.batch_norm',
           aten_name='batch_norm',
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_batch_norm,
           skips=(
               DecorateInfo(unittest.skip("We don't want to differentiate wrt running mean / std"),
                            "TestCommon", "test_floating_inputs_are_differentiable"),)
           ),
    # This variant tests batch_norm with cuDNN disabled only on CUDA devices
    OpInfo('nn.functional.batch_norm',
           variant_test_name='without_cudnn',
           aten_name='batch_norm',
           dtypes=empty_types(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           decorators=[onlyCUDA, disablecuDNN],
           skips=(
               DecorateInfo(unittest.skip("We don't want to differentiate wrt running mean / std"),
                            "TestCommon", "test_floating_inputs_are_differentiable"),),
           sample_inputs_func=sample_inputs_batch_norm),
    # We have to add 2 OpInfo entry for `igamma` and `igammac`.First is the
    # standard entry, second is to run gradcheck tests on the second argument.
    BinaryUfuncInfo('igamma',
                    dtypes=floating_types_and(torch.bfloat16, torch.float16),
                    aliases=('torch.special.gammainc',),
                    dtypesIfCUDA=floating_types(),
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_igamma_igammac,
                    skips=(
                        # TypeError: igamma(): argument 'input' (position 1) must be Tensor, not float
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs'),
                    )),
    BinaryUfuncInfo('igamma',
                    variant_test_name='grad_other',
                    # Since autograd formula is implemented only for other and
                    # gradcheck test verifies the formula for input in SampleInput,
                    # we permute the arguments.
                    op=lambda self, other, **kwargs: torch.igamma(other, self, **kwargs),
                    inplace_variant=None,
                    method_variant=None,
                    dtypes=floating_types_and(torch.bfloat16, torch.float16),
                    backward_dtypesIfCPU=floating_types_and(torch.bfloat16),
                    dtypesIfCUDA=floating_types(),
                    backward_dtypesIfCUDA=floating_types(),
                    supports_inplace_autograd=False,
                    decorators=[
                        # Derivative wrt first tensor not implemented
                        DecorateInfo(unittest.expectedFailure, "TestCommon",
                                     "test_floating_inputs_are_differentiable")
                    ],
                    skips=(
                        # test does not work with passing lambda for op
                        # AssertionError: False is not true : Tensors failed to compare as equal!
                        DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
                        # test fails are we permute the arguments function variant
                        # but not for inplace or method.
                        DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_variant_consistency_eager'),
                        # TypeError: igamma(): argument 'input' (position 1) must be Tensor, not float
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs'),
                    ),
                    sample_inputs_func=sample_inputs_igamma_igammac),
    BinaryUfuncInfo('igammac',
                    dtypes=floating_types_and(torch.bfloat16, torch.float16),
                    aliases=('torch.special.gammaincc',),
                    dtypesIfCUDA=floating_types(),
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_igamma_igammac,
                    skips=(
                        # TypeError: igammac(): argument 'input' (position 1) must be Tensor, not float
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs'),
                    )),
    BinaryUfuncInfo('igammac',
                    variant_test_name='grad_other',
                    # Since autograd formula is implemented only for other and
                    # gradcheck test verifies the formula for input in SampleInput,
                    # we permute the arguments
                    op=lambda self, other, **kwargs: torch.igammac(other, self, **kwargs),
                    inplace_variant=None,
                    method_variant=None,
                    dtypes=floating_types_and(torch.bfloat16, torch.float16),
                    backward_dtypesIfCPU=floating_types_and(torch.bfloat16),
                    dtypesIfCUDA=floating_types(),
                    backward_dtypesIfCUDA=floating_types(),
                    supports_inplace_autograd=False,
                    decorators=[
                        # Derivative wrt first tensor not implemented
                        DecorateInfo(unittest.expectedFailure, "TestCommon",
                                     "test_floating_inputs_are_differentiable"),
                    ],
                    skips=(
                        # test does not work with passing lambda for op
                        # AssertionError: False is not true : Tensors failed to compare as equal!
                        DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
                        # test fails are we permute the arguments function variant
                        # but not for inplace or method.
                        DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_variant_consistency_eager'),
                        # TypeError: igammac(): argument 'input' (position 1) must be Tensor, not float
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs'),
                    ),
                    sample_inputs_func=sample_inputs_igamma_igammac),
    OpInfo('nn.functional.softshrink',
           aten_name="softshrink",
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_autograd=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           assert_autodiffed=False,
           sample_inputs_func=sample_inputs_softshrink_hardshrink_hardtanh,
           supports_gradgrad=True,
           supports_out=False,
           ),
    OpInfo('nn.functional.hardshrink',
           aten_name="hardshrink",
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_softshrink_hardshrink_hardtanh,
           supports_gradgrad=True,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           autodiff_nonfusible_nodes=["aten::hardshrink"]),
    OpInfo('nn.functional.hardtanh',
           aten_name="hardtanh",
           dtypes=floating_types_and(torch.int8, torch.int16, torch.int32, torch.int64, torch.bfloat16),
           backward_dtypesIfCPU=all_types(),
           dtypesIfCUDA=floating_types_and(torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16),
           backward_dtypesIfCUDA=floating_types_and(torch.float16),
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_softshrink_hardshrink_hardtanh,
           supports_gradgrad=True,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           autodiff_nonfusible_nodes=["aten::hardtanh"],
           ),
    OpInfo('nn.functional.gelu',
           aten_name="gelu",
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_gelu,
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_gradgrad=True,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           autodiff_nonfusible_nodes=["aten::gelu"]),
    OpInfo('nn.functional.relu6',
           aten_name="relu6",
           dtypes=all_types_and(torch.bfloat16),
           backward_dtypesIfCPU=floating_types(),
           dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
           backward_dtypesIfCUDA=floating_types_and(torch.float16),
           supports_autograd=True,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_softshrink_hardshrink_hardtanh,
           supports_gradgrad=True,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           autodiff_nonfusible_nodes=["aten::relu6"]),
    OpInfo('mm',
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_mm),
    OpInfo('mode',
           op=torch.mode,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_mode,),
    MvlGammaInfo(variant_test_name='mvlgamma_p_1',
                 domain=(1, None),
                 skips=skips_mvlgamma(),
                 sample_kwargs=lambda device, dtype, input: ({'p': 1}, {'d': 1})),
    MvlGammaInfo(variant_test_name='mvlgamma_p_3',
                 domain=(2, None),
                 skips=skips_mvlgamma(skip_redundant=True) + (
                     DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                  dtypes=(torch.float16,)),
                 ),
                 sample_kwargs=lambda device, dtype, input: ({'p': 3}, {'d': 3})),
    MvlGammaInfo(variant_test_name='mvlgamma_p_5',
                 domain=(3, None),
                 skips=skips_mvlgamma(skip_redundant=True) + (
                     DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                  dtypes=(torch.float16,)),
                 ),
                 sample_kwargs=lambda device, dtype, input: ({'p': 5}, {'d': 5})),
    BinaryUfuncInfo('ne',
                    aliases=('not_equal',),
                    dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                    always_returns_bool=True,
                    supports_autograd=False,
                    sample_inputs_func=sample_inputs_comparison_ops,
                    skips=(
                        # AssertionError: False is not true : Tensors failed to compare as equal!
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_type_promotion'),
                    )),
    OpInfo('narrow',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_narrow),
    UnaryUfuncInfo('neg',
                   aliases=('negative', ),
                   ref=np.negative,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   error_inputs_func=error_inputs_neg,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   assert_autodiffed=True,),
    OpInfo('dist',
           op=torch.dist,
           dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_dist),
    OpInfo('outer',
           op=torch.outer,
           aliases=('ger', ),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_outer,),
    OpInfo('ormqr',
           op=torch.ormqr,
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_ormqr,
           decorators=[skipCUDAIfNoCusolver, skipCPUIfNoLapack]),
    OpInfo('permute',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           assert_autodiffed=True,
           autodiff_fusible_nodes=[],  # aliases inputs, shouldn't be fused
           autodiff_nonfusible_nodes=[],  # aliases inputs, shouldn't be fused
           assert_jit_shape_analysis=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_permute),
    BinaryUfuncInfo('pow',
                    dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                    # Due to AVX2 curently not being fully supported for Float16, log_vml_cpu can't be enabled
                    # for Float16, causing this test to fail. pow's autograd for Float16 is thus currently
                    # unsupported on CPU.
                    backward_dtypes=floating_and_complex_types_and(torch.bfloat16),
                    backward_dtypesIfCUDA=floating_and_complex_types_and(torch.bfloat16, torch.half),
                    sample_inputs_func=sample_inputs_pow,
                    supports_inplace_autograd=False,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    assert_autodiffed=True,
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_type_promotion'),
                        # RuntimeError: Integers to negative integer powers are not allowed.
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs',
                                     dtypes=(torch.int8, torch.int16, torch.int32, torch.int64)),
                    )),
    BinaryUfuncInfo('float_power',
                    dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
                    promotes_int_to_float=True,
                    sample_inputs_func=sample_inputs_pow,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    skips=(
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs', 'test_type_promotion'),
                    )),
    OpInfo('qr',
           op=torch.qr,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_qr_geqrf,
           # batched gradients do not work for empty inputs
           # https://github.com/pytorch/pytorch/issues/50743#issuecomment-767376085
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # See https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    UnaryUfuncInfo('rad2deg',
                   ref=np.degrees,
                   decorators=(precisionOverride({torch.bfloat16: 7e-1,
                                                  torch.float16: 7e-1}),),
                   dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/51283#issuecomment-770614273
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.bfloat16]),
                   ),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('real',
                   ref=np.real,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
                   supports_out=False,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   # See https://github.com/pytorch/pytorch/issues/66357
                   check_batched_forward_grad=False,
                   skips=(
                       # Skip since real and imag don't have out variants.
                       DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_out_arg_all_dtypes'),
                   )),
    OpInfo('roll',
           ref=np.roll,
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_roll),
    OpInfo('rot90',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_rot90),
    # To test reference numerics against multiple values of argument `decimals`,
    # we make multiple OpInfo entries with each entry corresponding to different value of decimals.
    UnaryUfuncInfo('round',
                   ref=np.round,
                   aliases=('special.round',),
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   assert_autodiffed=True,),
    UnaryUfuncInfo('round',
                   ref=np.round,
                   variant_test_name='decimals_0',
                   aliases=('special.round',),
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   sample_kwargs=lambda device, dtype, input: ({'decimals': 0}, {'decimals': 0}),
                   sample_inputs_func=partial(sample_inputs_unary, op_kwargs={'decimals': 0}),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   assert_autodiffed=False,
                   supports_sparse_csr=False),
    UnaryUfuncInfo('round',
                   ref=np.round,
                   variant_test_name='decimals_3',
                   aliases=('special.round',),
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   sample_kwargs=lambda device, dtype, input: ({'decimals': 3}, {'decimals': 3}),
                   sample_inputs_func=partial(sample_inputs_unary, op_kwargs={'decimals': 3}),
                   skips=(
                       # test_ops already tested for this overload with `decimals_0` opinfo entry
                       DecorateInfo(unittest.skip("Skipped!"), 'TestCommon'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestJit'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits'),
                   ),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   assert_autodiffed=False,
                   supports_sparse_csr=False),
    UnaryUfuncInfo('round',
                   ref=np.round,
                   variant_test_name='decimals_neg_3',
                   aliases=('special.round',),
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   sample_kwargs=lambda device, dtype, input: ({'decimals': -3}, {'decimals': -3}),
                   sample_inputs_func=partial(sample_inputs_unary, op_kwargs={'decimals': -3}),
                   skips=(
                       # test_ops already tested for this overload with `decimals_0` opinfo entry
                       DecorateInfo(unittest.skip("Skipped!"), 'TestCommon'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestJit'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits'),
                   ),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   assert_autodiffed=False,
                   supports_sparse_csr=False),
    UnaryUfuncInfo('sin',
                   ref=np.sin,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   handles_large_floats=False,
                   handles_complex_extremals=False,
                   safe_casts_outputs=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   skips=(
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   ),
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),)),
    UnaryUfuncInfo('sinc',
                   ref=np_sinc_with_fp16_as_fp32,
                   aliases=('special.sinc',),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   handles_large_floats=False,
                   handles_complex_extremals=False,
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2,
                                                  torch.float16: 1e-2}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/49133
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    dtypes=[torch.cfloat]),
                   )),
    UnaryUfuncInfo('sinh',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.sinh),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   decorators=(precisionOverride({torch.float16: 1e-2}),),
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=(IS_MACOS or IS_WINDOWS)),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=(IS_MACOS or IS_WINDOWS)),
                       # Reference: https://github.com/pytorch/pytorch/issues/48641
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.int8]),
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   )),
    UnaryUfuncInfo('sign',
                   ref=reference_sign,
                   dtypes=all_types_and(torch.bool, torch.bfloat16, torch.half),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.bfloat16, torch.half),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/41245
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64]),
                   )),
    UnaryUfuncInfo('sgn',
                   ref=reference_sgn,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/41245
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64]),
                       # Reference: https://github.com/pytorch/pytorch/issues/53958
                       # Test fails in comparison on Nan as the `equal_nan` is True for
                       # comparing the CPU tensors.
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.complex64, torch.complex128]),
                       # Reference: https://github.com/pytorch/pytorch/issues/48486
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.complex64]),
                       # The complex formula might be wrong
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD',
                                    dtypes=complex_types()),
                       # Passes for float, but for complex - Need: _s_where
                       DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad',
                                    dtypes=complex_types()),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_inplace_forward_mode_AD',
                                    dtypes=complex_types()),
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   )),
    OpInfo('split',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=partial(sample_inputs_split, list_args=False),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False,
           autodiff_fusible_nodes=[],  # aliases inputs, shouldn't be fused
           autodiff_nonfusible_nodes=[],  # aliases inputs, shouldn't be fused
           assert_autodiffed=True),
    OpInfo('split',
           variant_test_name='list_args',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=partial(sample_inputs_split, list_args=True),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False),
    OpInfo('split_with_sizes',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=sample_inputs_split_with_sizes,
           autodiff_fusible_nodes=[],  # aliases inputs, shouldn't be fused
           autodiff_nonfusible_nodes=[],  # aliases inputs, shouldn't be fused
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           assert_autodiffed=True),
    BinaryUfuncInfo('__radd__',
                    op=torch.Tensor.__radd__,
                    dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
                    sample_inputs_func=sample_inputs_rbinops,
                    supports_out=False,
                    skips=(
                        # RuntimeError:
                        # object has no attribute __radd__:
                        #   File "<string>", line 3
                        # def the_method(i0):
                        #     return torch.__radd__(i0, 3.14j)
                        #            ~~~~~~~~~~~~~~ <--- HERE
                        DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit',),
                    ),
                    assert_autodiffed=True,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    autodiff_nonfusible_nodes=['aten::add'],),
    BinaryUfuncInfo('__rdiv__',
                    op=torch.Tensor.__rdiv__,
                    dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
                    promotes_int_to_float=True,
                    lhs_make_tensor_kwargs={'exclude_zero': True},
                    sample_inputs_func=sample_inputs_rbinops,
                    supports_out=False,
                    skips=(
                        # RuntimeError:
                        # object has no attribute __rdiv__:
                        #   File "<string>", line 3
                        # def the_method(i0):
                        #     return torch.__rdiv__(i0, 3.14j)
                        #            ~~~~~~~~~~~~~~ <--- HERE
                        DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit',),
                    ),
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    assert_autodiffed=True,
                    autodiff_nonfusible_nodes=['aten::mul', 'aten::reciprocal'],),
    BinaryUfuncInfo('__rmul__',
                    op=torch.Tensor.__rmul__,
                    dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
                    sample_inputs_func=sample_inputs_rbinops,
                    supports_out=False,
                    skips=(
                        # RuntimeError:
                        # object has no attribute __rmul__:
                        #   File "<string>", line 3
                        # def the_method(i0):
                        #     return torch.__rmul__(i0, 3.14j)
                        #            ~~~~~~~~~~~~~~ <--- HERE
                        DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit',),
                    ),
                    assert_autodiffed=True,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    autodiff_nonfusible_nodes=['aten::mul'],),
    OpInfo('__rand__',
           op=torch.Tensor.__rand__,
           dtypes=integral_types_and(torch.bool),
           sample_inputs_func=sample_inputs_rbinops,
           supports_out=False,
           supports_autograd=False,
           supports_forward_ad=True,),
    BinaryUfuncInfo('__ror__',
                    op=torch.Tensor.__ror__,
                    dtypes=integral_types_and(torch.bool),
                    sample_inputs_func=sample_inputs_rbinops,
                    supports_out=False,
                    supports_autograd=False,
                    supports_forward_ad=True,),
    BinaryUfuncInfo('__rxor__',
                    op=torch.Tensor.__rxor__,
                    dtypes=integral_types_and(torch.bool),
                    sample_inputs_func=sample_inputs_rbinops,
                    supports_out=False,
                    supports_autograd=False,
                    supports_forward_ad=True,),
    OpInfo('__rmatmul__',
           op=torch.Tensor.__rmatmul__,
           dtypes=all_types_and_complex_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else [],
                                           torch.complex64, torch.complex128),
           backward_dtypesIfCUDA=floating_types_and(torch.float16,
                                                    *[torch.bfloat16] if (SM60OrLater and CUDA11OrLater) else [],
                                                    torch.complex64, torch.complex128),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_matmul,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           check_batched_forward_grad=False,
           decorators=(
               # https://github.com/pytorch/pytorch/issues/67470
               DecorateInfo(unittest.skip("67470!"),
                            'TestCommon', 'test_noncontiguous_samples',
                            device_type='cpu', dtypes=(torch.long,)),
               DecorateInfo(toleranceOverride({torch.complex64: tol(atol=1e-05, rtol=1.2e-03)}),
                            'TestMathBits', 'test_conj_view'),
               # Fails on XLA.
               # AssertionError: False is not true : Tensors failed to compare as equal
               DecorateInfo(unittest.expectedFailure, 'TestOpInfo', device_type='xla', dtypes=(torch.long,)),
               DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-05, rtol=1.2e-03)}),
                            'TestCommon', 'test_noncontiguous_samples',
                            device_type='cuda', active_if=TEST_WITH_ROCM)),
           skips=(
               # RuntimeError:
               # object has no attribute __rmatmul__:
               #   File "<string>", line 3
               # def the_method(i0, i1):
               #     return torch.__rmatmul__(i0, i1)
               #            ~~~~~~~~~~~~~~ <--- HERE
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit',),
           )),
    BinaryUfuncInfo('__rmod__',
                    op=torch.Tensor.__rmod__,
                    dtypes=floating_types_and(torch.bfloat16, torch.half,),
                    dtypesIfCUDA=all_types_and(torch.bfloat16, torch.half, torch.bool),
                    sample_inputs_func=sample_inputs_rbinops,
                    supports_out=False,
                    skips=(
                        # RuntimeError:
                        # object has no attribute __rmod__:
                        #   File "<string>", line 3
                        # def the_method(i0):
                        #     return torch.__rmod__(i0, 3.14)
                        #            ~~~~~~~~~~~~~~ <--- HERE
                        DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit',),
                        # RuntimeError: "remainder_cuda" not implemented for 'Bool'
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs',
                                     dtypes=(torch.bool,))
                    ),
                    # Support autograd after torch.remainder(Tensor, Tensor) supports
                    # autograd of the second argument.
                    # https://github.com/pytorch/pytorch/pull/58476/files#r637167630
                    supports_autograd=False,
                    assert_autodiffed=True,
                    autodiff_nonfusible_nodes=['aten::remainder'],),
    BinaryUfuncInfo('__rpow__',
                    op=torch.Tensor.__rpow__,
                    dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
                    # Reference: https://github.com/pytorch/pytorch/issues/54774
                    # "log2" "_vml_cpu" not implemented for Half
                    backward_dtypesIfCPU=all_types_and_complex_and(torch.bfloat16, torch.bool),
                    sample_inputs_func=sample_inputs_rbinops,
                    supports_out=False,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    skips=(
                        # RuntimeError:
                        # object has no attribute __rpow__:
                        #   File "<string>", line 3
                        # def the_method(i0):
                        #     return torch.__rpow__(i0, 3.14j)
                        #            ~~~~~~~~~~~~~~ <--- HERE
                        DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit',),
                        # RuntimeError: "pow_cuda" not implemented for 'Bool'
                        DecorateInfo(unittest.skip("Skipped!"), 'TestBinaryUfuncs',
                                     dtypes=(torch.bool,)),
                    ),
                    assert_autodiffed=True,
                    autodiff_nonfusible_nodes=['aten::pow'],),
    BinaryUfuncInfo('__rsub__',
                    op=torch.Tensor.__rsub__,
                    dtypes=all_types_and_complex_and(torch.bfloat16, torch.half),
                    sample_inputs_func=sample_inputs_rbinops,
                    supports_out=False,
                    skips=(
                        # RuntimeError:
                        # object has no attribute __rsub__:
                        #   File "<string>", line 3
                        # def the_method(i0):
                        #     return torch.__rsub__(i0, 3.14j)
                        #            ~~~~~~~~~~~~~~ <--- HERE
                        DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit',),
                    ),
                    assert_autodiffed=True,
                    autodiff_nonfusible_nodes=['aten::rsub'],),
    BinaryUfuncInfo('rsub',
                    dtypes=all_types_and_complex_and(torch.bfloat16, torch.half),
                    variant_test_name='rsub_tensor',
                    supports_out=False,
                    supports_inplace_autograd=False,
                    sample_inputs_func=partial(sample_inputs_rsub, other_scalar=False),),
    BinaryUfuncInfo('rsub',
                    dtypes=all_types_and_complex_and(torch.bfloat16, torch.half),
                    variant_test_name='rsub_scalar',
                    supports_out=False,
                    supports_inplace_autograd=False,
                    sample_inputs_func=partial(sample_inputs_rsub, other_scalar=True),
                    assert_autodiffed=True,),
    OpInfo('select',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=sample_inputs_select,
           assert_jit_shape_analysis=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False),
    OpInfo('select_scatter',
           dtypes=all_types_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=sample_inputs_select_scatter,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False),
    OpInfo('slice_scatter',
           dtypes=all_types_and(torch.bfloat16, torch.half, torch.bool),
           sample_inputs_func=sample_inputs_slice_scatter,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False),
    UnaryUfuncInfo('signbit',
                   ref=np.signbit,
                   dtypes=all_types_and(torch.bool, torch.bfloat16, torch.half),
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   supports_autograd=False,),
    OpInfo('solve',
           op=torch.solve,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_legacy_solve,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    UnaryUfuncInfo('tan',
                   ref=np.tan,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    device_type='cpu', dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=(IS_MACOS or IS_WINDOWS)),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=(IS_MACOS or IS_WINDOWS)),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cuda', dtypes=[torch.float64],
                                    active_if=TEST_WITH_ROCM),
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   ),
                   # tan(pi/2 * odd_number) is nan
                   reference_numerics_filter=NumericsFilter(
                       condition=lambda x: close_to_int(x / (math.pi * 0.5)), safe_val=math.pi)),
    UnaryUfuncInfo('tanh',
                   ref=np.tanh,
                   aliases=('nn.functional.tanh',),
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # "tanh_backward_cpu" not implemented for 'BFloat16'
                   backward_dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   assert_jit_shape_analysis=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=(IS_MACOS or IS_WINDOWS)),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=(IS_MACOS or IS_WINDOWS)),
                       # alias, nn.functional.tanh, will produce (because of warning string saved):
                       # "RuntimeError: Expected to not find "tanh" but found it"
                       DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping'),
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   ),
                   # tan(j * pi/2 * odd_number) is nan
                   reference_numerics_filter=NumericsFilter(
                       condition=lambda x: (close_to_int(x / (math.pi * 0.5j))
                                            if x.is_complex() else x.new_tensor(False, dtype=torch.bool)),
                       safe_val=0)),
    OpInfo('tensor_split',
           ref=np.array_split,
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # Pre-existing condition; Needs to be fixed
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
           ),
           sample_inputs_func=sample_inputs_tensor_split,),
    OpInfo('hsplit',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_hsplit,
           error_inputs_func=error_inputs_hsplit,),
    OpInfo('vsplit',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_vsplit,
           error_inputs_func=error_inputs_vsplit,),
    OpInfo('dsplit',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_dsplit,
           error_inputs_func=error_inputs_dsplit,),
    OpInfo('triangular_solve',
           op=torch.triangular_solve,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_legacy_solve,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_wrapper=lambda *args, **kwargs: gradcheck_wrapper_triangular_input(*args, idx=1, **kwargs),
           decorators=[skipCUDAIfNoMagma],
           skips=(
               # Gradcheck fails
               DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad',
                            dtypes=floating_and_complex_types()),
           )),
    UnaryUfuncInfo('trunc',
                   aliases=('fix', ),
                   ref=np.trunc,
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   assert_autodiffed=True),
    UnaryUfuncInfo('exp2',
                   aliases=('special.exp2', ),
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.exp2),
                   dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('expm1',
                   aliases=('special.expm1', ),
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.expm1),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/48926#issuecomment-739734774
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    device_type='cpu', dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   )),
    UnaryUfuncInfo('nan_to_num',
                   ref=np.nan_to_num,
                   dtypes=all_types_and(torch.half, torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.half, torch.bool, torch.bfloat16),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse=True,
                   skips=(
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   ),
                   # Passing numpy_kwargs via sample_kwargs, as numpy does comparison
                   # with BFloat16 in float, since it currently doesn't support BFloat16.
                   # Ref: https://github.com/pytorch/pytorch/issues/57982#issuecomment-839150556
                   sample_kwargs=lambda device, dtype, input: ({},
                                                               {'posinf': torch.finfo(torch.bfloat16).max,
                                                                'neginf': torch.finfo(torch.bfloat16).min})
                   if dtype is torch.bfloat16 else ({}, {})),
    UnaryUfuncInfo('reciprocal',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.reciprocal),
                   dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/45690
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.cfloat, torch.cdouble]),
                       # Reference: https://github.com/pytorch/pytorch/pull/49102#issuecomment-744604601
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    dtypes=[torch.bfloat16]),
                   )),
    UnaryUfuncInfo('rsqrt',
                   ref=lambda x: np.reciprocal(np.sqrt(x)),
                   domain=(0, None),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   decorators=(precisionOverride({torch.half: 5e-2}),),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   handles_complex_extremals=False),
    UnaryUfuncInfo('sqrt',
                   ref=np.sqrt,
                   supports_sparse=True,
                   domain=(0, None),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_sparse_csr=True,
                   supports_fwgrad_bwgrad=True,
                   decorators=(precisionOverride({torch.bfloat16: 7e-2}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/47358
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                    active_if=IS_MACOS),
                       # Reference: https://github.com/pytorch/pytorch/pull/47293#issuecomment-721774436
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),
                   ),
                   safe_casts_outputs=True,
                   handles_complex_extremals=False),
    UnaryUfuncInfo('square',
                   ref=np.square,
                   dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                   decorators=(precisionOverride({torch.complex64: 3e-4, torch.bfloat16: 3e-1}),),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/52549
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    dtypes=[torch.cfloat, torch.cdouble]),
                       # >>> t = torch.tensor(complex(-0.01, float("inf")))
                       # >>> np.square(t.numpy())
                       # (-inf-infj)
                       # >>> t.square()
                       # tensor(-inf-infj)
                       # >>> t.cuda().square()
                       # tensor(inf+nanj, device='cuda:0')
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cuda', dtypes=[torch.cfloat, torch.cdouble]),
                       # Reference: https://github.com/pytorch/pytorch/pull/52551#issuecomment-782596181
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    dtypes=[torch.bfloat16]),
                   ),),
    OpInfo('lerp',
           dtypes=floating_and_complex_types(),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_lerp,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           assert_autodiffed=True),
    OpInfo('linalg.inv',
           aten_name='linalg_inv',
           op=torch.linalg.inv,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_invertible,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           ),
    OpInfo('linalg.inv_ex',
           aten_name='linalg_inv_ex',
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_invertible,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           ),
    UnaryUfuncInfo('angle',
                   ref=np.angle,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   supports_sparse_csr=True,
                   supports_complex_to_float=True,
                   skips=(
                       # The complex formula might be wrong
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD',
                                    dtypes=complex_types()),
                   )),
    UnaryUfuncInfo('isfinite',
                   ref=np.isfinite,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                   supports_out=False,
                   supports_autograd=False,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/66402
                       DecorateInfo(unittest.expectedFailure, "TestUnaryUfuncs", "test_reference_numerics_hard",
                                    device_type='cpu', dtypes=(torch.complex64,), active_if=not (IS_MACOS or IS_WINDOWS)),
                   )),
    UnaryUfuncInfo('isinf',
                   ref=np.isinf,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                   supports_out=False,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   supports_autograd=False),
    UnaryUfuncInfo('isposinf',
                   ref=np.isposinf,
                   dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   supports_autograd=False),
    UnaryUfuncInfo('isneginf',
                   ref=np.isneginf,
                   dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   supports_autograd=False),
    UnaryUfuncInfo('isreal',
                   ref=np.isreal,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                   supports_out=False,
                   supports_autograd=False),
    UnaryUfuncInfo('isnan',
                   ref=np.isnan,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                   supports_out=False,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   supports_autograd=False),
    OpInfo('linalg.solve',
           aten_name='linalg_solve',
           op=torch.linalg.solve,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_solve,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.solve_triangular',
           aten_name='linalg_solve_triangular',
           op=torch.linalg.solve_triangular,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_solve_triangular,
           supports_fwgrad_bwgrad=True,
           # linalg.solve_triangular cannot be batched over because of a call to out.copy_(result);
           supports_forward_ad=True,
           skips=(
               DecorateInfo(unittest.skip("Tests different backward implementations"),
                            "TestCommon", "test_floating_inputs_are_differentiable"),),
           ),
    OpInfo('linalg.matrix_rank',
           aten_name='linalg_matrix_rank',
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
           skips=(
               # Pre-existing condition; Needs to be fixed
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
           ),
           ),
    OpInfo('linalg.matrix_rank',
           aten_name='linalg_matrix_rank',
           variant_test_name='hermitian',
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
           skips=(
               # Pre-existing condition; Needs to be fixed
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
           ),
           ),
    OpInfo('linalg.pinv',
           aten_name='linalg_pinv',
           op=torch.linalg.pinv,
           dtypes=floating_and_complex_types(),
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_linalg_pinv,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack,
                       # Derivative wrt rcond tensor not implemented
                       DecorateInfo(unittest.expectedFailure,
                                    "TestCommon", "test_floating_inputs_are_differentiable")],
           skips=(
               # errors with "leaked XXXX bytes CUDA memory on device 0"
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', device_type='cuda'),)
           ),
    OpInfo('linalg.pinv',
           aten_name='linalg_pinv',
           variant_test_name='singular',
           # pinv is Frechet-differentiable in a rank-preserving neighborhood,
           # so we feed inputs that are the products of two full-rank factors,
           # to avoid any rank changes caused by the perturbations in the gradcheck
           op=lambda a, b: torch.linalg.pinv(a @ b.mT),
           dtypes=floating_and_complex_types(),
           supports_out=False,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_linalg_pinv_singular,
           # Only large tensors show issues with implicit backward used prior to
           # explicit backward implementation.
           decorators=[slowTest, skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
           skips=(
               # test does not work with passing lambda for op
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # CUDA runs out of memory
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_fwgrad_bwgrad',
                            device_type='cuda', dtypes=[torch.cdouble]),
           )),
    OpInfo('linalg.pinv',
           aten_name='linalg_pinv',
           variant_test_name='hermitian',
           dtypes=floating_and_complex_types(),
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           ),
    OpInfo('eig',
           op=torch.eig,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_eig,
           decorators=[
               skipCUDAIfNoMagma,
               skipCPUIfNoLapack,
               skipCUDAIfRocm
           ],),
    OpInfo('einsum',
           # we need this lambda because SampleInput expects tensor input as the first argument
           # TODO(@heitorschueroff) update SampleInput to handle such cases
           op=lambda tensors, equation: torch.einsum(equation, tensors),
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, *[torch.bfloat16] if CUDA11OrLater else []),
           backward_dtypesIfCUDA=floating_and_complex_types_and(torch.half,
                                                                *[torch.bfloat16] if (SM60OrLater and CUDA11OrLater) else []),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           check_batched_forward_grad=False,
           # See https://github.com/pytorch/pytorch/issues/66357
           sample_inputs_func=sample_inputs_einsum,
           skips=(
               # test does not work with passing lambda for op
               # there's a test `test_einsum` in `test_jit.py` to handle this case
               # AssertionError: JIT Test does not execute any logic
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('svd',
           op=torch.svd,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_svd,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           check_batched_forward_grad=False,
           # We're using at::allclose, which does not have a batching rule
           check_batched_grad=False,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
           skips=(
               # Fixme, forward over backward gives a numerical error
               DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad', dtypes=(torch.complex128,)),
           )),
    OpInfo('linalg.svd',
           op=torch.linalg.svd,
           aten_name='linalg_svd',
           dtypes=floating_and_complex_types(),
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True,
           check_batched_forward_grad=False,
           # We're using at::allclose, which does not have a batching rule
           check_batched_grad=False,
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_svd,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
           skips=(
               # FIXME forward over backward gives a numerical error
               DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad', dtypes=(torch.complex128,)),
           )),
    OpInfo('linalg.svdvals',
           op=torch.linalg.svdvals,
           aten_name='linalg_svdvals',
           dtypes=floating_and_complex_types(),
           check_batched_forward_grad=False,
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True,
           # We're using at::allclose, which does not have a batching rule
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_svdvals,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack]),
    OpInfo('svd_lowrank',
           op=lambda *args, **kwargs: wrapper_set_seed(
               lambda a, b, **kwargs: torch.svd_lowrank(a @ b.mT, **kwargs),
               *args, **kwargs
           ),
           dtypes=floating_types(),
           supports_out=False,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           check_batched_forward_grad=False,
           supports_fwgrad_bwgrad=True,
           supports_forward_ad=True,
           sample_inputs_func=sample_inputs_svd_lowrank,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack,
                       DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-03, rtol=1e-03)}),
                                    'TestCommon', 'test_noncontiguous_samples',
                                    device_type='cuda')],
           skips=(
               # test does not work with passing lambda for op
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('pca_lowrank',
           op=lambda *args, **kwargs: wrapper_set_seed(
               lambda a, b, **kwargs: torch.pca_lowrank(a @ b.mT, **kwargs),
               *args, **kwargs
           ),
           dtypes=floating_types(),
           supports_out=False,
           check_batched_forward_grad=False,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_pca_lowrank,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
           skips=(
               # test does not work with passing lambda for op
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    BinaryUfuncInfo('polar',
                    dtypes=floating_types(),
                    sample_inputs_func=sample_inputs_polar,
                    skips=(
                        # RuntimeError: Expected object of scalar type Float but got scalar type Double for second argument
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_type_promotion'),
                        # TypeError: polar(): argument 'angle' (position 2) must be Tensor, not float
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs'),
                    )),
    # TODO(@kshitij12345): Refactor similar to `mvlgamma` entries.
    # To test reference numerics against multiple values of argument `n`,
    # we make multiple OpInfo entries with each entry corresponding to different value of n (currently 0 to 4).
    # We run the op tests from test_ops.py only for `n=0` to avoid redundancy in testing.
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_0',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # AssertionError: JIT Test does not execute any logic
                       DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
                   ),
                   sample_kwargs=lambda device, dtype, input: ({'n': 0}, {'n': 0})),
    # A separate OpInfo entry for special.polygamma is needed to reorder the arguments
    # for the alias. See the discussion here: https://github.com/pytorch/pytorch/pull/59691#discussion_r650261939
    UnaryUfuncInfo('special.polygamma',
                   op=lambda x, n, **kwargs: torch.special.polygamma(n, x, **kwargs),
                   variant_test_name='special_polygamma_n_0',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # AssertionError: JIT Test does not execute any logic
                       DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
                   ),
                   sample_kwargs=lambda device, dtype, input: ({'n': 0}, {'n': 0}),
                   # polygamma functions have multiple singularities at x <= 0
                   reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_1',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestJit'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard'),
                   ),
                   sample_kwargs=lambda device, dtype, input: ({'n': 1}, {'n': 1}),
                   # polygamma functions have multiple singularities at x <= 0
                   reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_2',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestJit'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    active_if=TEST_WITH_ROCM),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    active_if=TEST_WITH_ROCM),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 2}, {'n': 2}),
                   # polygamma functions have multiple singularities at x <= 0
                   reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_3',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestJit'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    active_if=TEST_WITH_ROCM),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    active_if=TEST_WITH_ROCM),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 3}, {'n': 3}),
                   # polygamma functions have multiple singularities at x <= 0
                   reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_4',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.float16: 5e-4, torch.float32: 5e-4}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       DecorateInfo(unittest.skip("Skipped!"), 'TestGradients'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestJit'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    active_if=TEST_WITH_ROCM),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    active_if=TEST_WITH_ROCM),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 4}, {'n': 4}),
                   # polygamma functions have multiple singularities at x <= 0
                   reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)),
    OpInfo('ravel',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_ravel,
           ),
    OpInfo('reshape',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_view_reshape,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           ),
    OpInfo('reshape_as',
           op=lambda x, other: x.reshape_as(other),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_view_as_reshape_as,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               DecorateInfo(unittest.skip("Second argument does not need gradient"),
                            "TestCommon", "test_floating_inputs_are_differentiable"),),
           ),
    OpInfo('view',
           op=lambda x, shape: x.view(shape),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           assert_jit_shape_analysis=True,
           sample_inputs_func=sample_inputs_view_reshape,
           ),
    OpInfo('view_as',
           op=lambda x, other: x.view_as(other),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_view_as_reshape_as,
           skips=(
               DecorateInfo(unittest.skip("Second argument does not need gradient"),
                            "TestCommon", "test_floating_inputs_are_differentiable"),),
           ),
    OpInfo('atleast_1d',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_atleast1d2d3d,
           skips=(
               # JIT does not support variadic tensors.
               # RuntimeError: input->type()->kind() == TypeKind::OptionalType
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":252,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=[torch.float32]),
           ),
           ),
    OpInfo('atleast_2d',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # JIT does not support variadic tensors.
               # RuntimeError: input->type()->kind() == TypeKind::OptionalType
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":252,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=[torch.float32]),
           ),
           sample_inputs_func=sample_inputs_atleast1d2d3d,
           ),
    OpInfo('atleast_3d',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # JIT does not support variadic tensors.
               # RuntimeError: input->type()->kind() == TypeKind::OptionalType
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":252,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=[torch.float32]),
           ),
           sample_inputs_func=sample_inputs_atleast1d2d3d,
           ),
    OpInfo('flatten',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_flatten,
           ),
    OpInfo('column_stack',
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # JIT does not support variadic tensors.
               # RuntimeError: input->type()->kind() == TypeKind::OptionalType
               # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":252,
               # please report a bug to PyTorch.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),
               # AssertionError: UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_out_warning'),),
           sample_inputs_func=sample_inputs_column_stack,),
    OpInfo('pinverse',
           op=torch.pinverse,
           dtypes=floating_and_complex_types(),
           check_batched_grad=False,
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           supports_out=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack]),
    OpInfo('gather',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_gather,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           error_inputs_func=error_inputs_gather
           ),
    OpInfo('index_fill',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_index),
    OpInfo('index_copy',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_index,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('index_select',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_index,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           assert_jit_shape_analysis=True,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('index_add',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           # An `out=` variant exists but is not exposed to the Python API
           # see: https://github.com/pytorch/pytorch/pull/65993#discussion_r737760723
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_index,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('__getitem__',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_inplace_autograd=False,
           supports_scripting=False,
           op=torch.Tensor.__getitem__,
           skips=(
               # AssertionError: False is not true : Scalars failed to compare as equal! 0 != 104448
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit', device_type='cuda'),),
           assert_jit_shape_analysis=False,  # TODO: support index.Tensor()
           sample_inputs_func=sample_inputs_getitem),
    OpInfo('index_put',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_inplace_autograd=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           test_neg_view=False,
           sample_inputs_func=sample_inputs_index_put,
           skips=(
               # RuntimeError: The following operation failed in the TorchScript interpreter.
               # Traceback of TorchScript (most recent call last):
               #   File "<string>", line 3, in forward
               # def the_method(i0, i1: List[torch.Tensor], i2):
               #     return torch.index_put(i0, i1, i2, accumulate=False)
               #            ~~~~~~~~~~~~~~~ <--- HERE
               # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('sort',
           dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
           dtypesIfROCM=all_types_and(torch.float16),
           sample_inputs_func=sample_inputs_sort,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # sort does not correctly warn when resizing out= inputs
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),
               # RuntimeError not raised
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cpu'),
           )),
    OpInfo('unique',
           dtypes=all_types_and(torch.bool, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.bool, torch.float16),
           sample_inputs_func=sample_inputs_unique,
           supports_out=False,
           supports_autograd=False,
           skips=(
               # RuntimeError:
               # 'Tensor (inferred)' object has no attribute or method 'unique'.:
               #   File "<string>", line 3
               #
               #  def the_method(i0):
               #      return i0.unique(sorted=False, return_inverse=False, return_counts=False, dim=None)
               #                 ~~~~~~~~~ <--- HERE
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('unique_consecutive',
           dtypes=all_types_and(torch.bool, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.bool, torch.float16),
           sample_inputs_func=sample_inputs_unique_consecutive,
           supports_out=False,
           supports_autograd=False,
           skips=(
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('put',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           check_batched_gradgrad=False,  # vmap complains of the sizes
           sample_inputs_func=sample_inputs_put),
    OpInfo('take',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           check_batched_grad=False,  # vmap complains of the sizes
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=False,  # Need: put_
           sample_inputs_func=sample_inputs_take),
    OpInfo('scatter',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_scatter,
           error_inputs_func=error_inputs_scatter_and_scatter_add),
    OpInfo('bfloat16',
           op=lambda x, *args, **kwargs: x.bfloat16(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_conversion,
           # The autograd test runner cannot handle functions that change dtype
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('bfloat16',
           op=lambda x, *args, **kwargs: x.bfloat16(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           variant_test_name='channels_last',
           sample_inputs_func=sample_inputs_conversion_channels_last,
           # The autograd test runner cannot handle functions that change dtype
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('bool',
           op=lambda x, *args, **kwargs: x.bool(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_conversion,
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('bool',
           op=lambda x, *args, **kwargs: x.bool(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           variant_test_name='channels_last',
           sample_inputs_func=sample_inputs_conversion_channels_last,
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('byte',
           op=lambda x, *args, **kwargs: x.byte(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_conversion,
           # The autograd test runner cannot handle functions that change dtype
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('byte',
           op=lambda x, *args, **kwargs: x.byte(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           variant_test_name='channels_last',
           sample_inputs_func=sample_inputs_conversion_channels_last,
           # The autograd test runner cannot handle functions that change dtype
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('char',
           op=lambda x, *args, **kwargs: x.char(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_conversion,
           # The autograd test runner cannot handle functions that change dtype
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('char',
           op=lambda x, *args, **kwargs: x.char(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           variant_test_name='channels_last',
           sample_inputs_func=sample_inputs_conversion_channels_last,
           # The autograd test runner cannot handle functions that change dtype
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('double',
           op=lambda x, *args, **kwargs: x.double(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_conversion,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('double',
           op=lambda x, *args, **kwargs: x.double(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           variant_test_name='channels_last',
           sample_inputs_func=sample_inputs_conversion_channels_last,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('float',
           op=lambda x, *args, **kwargs: x.float(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_conversion,
           # The autograd test runner cannot handle functions that change dtype
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('float',
           op=lambda x, *args, **kwargs: x.float(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           variant_test_name='channels_last',
           sample_inputs_func=sample_inputs_conversion_channels_last,
           # The autograd test runner cannot handle functions that change dtype
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('half',
           op=lambda x, *args, **kwargs: x.half(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_conversion,
           # The autograd test runner cannot handle functions that change dtype
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('half',
           op=lambda x, *args, **kwargs: x.half(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           variant_test_name='channels_last',
           sample_inputs_func=sample_inputs_conversion_channels_last,
           # The autograd test runner cannot handle functions that change dtype
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('int',
           op=lambda x, *args, **kwargs: x.int(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_conversion,
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('int',
           op=lambda x, *args, **kwargs: x.int(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           variant_test_name='channels_last',
           sample_inputs_func=sample_inputs_conversion_channels_last,
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('long',
           op=lambda x, *args, **kwargs: x.long(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_conversion,
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('long',
           op=lambda x, *args, **kwargs: x.long(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           variant_test_name='channels_last',
           sample_inputs_func=sample_inputs_conversion_channels_last,
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('short',
           op=lambda x, *args, **kwargs: x.short(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_conversion,
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('short',
           op=lambda x, *args, **kwargs: x.short(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           variant_test_name='channels_last',
           sample_inputs_func=sample_inputs_conversion_channels_last,
           supports_autograd=False,
           skips=(
               # RuntimeError: attribute lookup is not defined on builtin
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('empty_like',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_like_fns,
           supports_autograd=False,
           skips=(
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_noncontiguous_samples'),
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_conj_view'),
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_view'),
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_conj_view'),
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )),
    OpInfo('zeros_like',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_like_fns,
           supports_autograd=False,
           skips=(
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )),
    OpInfo('ones_like',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_like_fns,
           supports_autograd=False,
           skips=(
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )),
    OpInfo('randn_like',
           dtypes=floating_types_and(torch.half, torch.bfloat16, torch.complex64, torch.complex128),
           op=lambda inp, *args, **kwargs:
               wrapper_set_seed(torch.randn_like, inp, *args, **kwargs),
           supports_out=False,
           sample_inputs_func=sample_inputs_like_fns,
           supports_autograd=False,
           supports_sparse_csr=True,
           skips=(
               # AssertionError: JIT Test does not execute any logic
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('rand_like',
           dtypes=floating_types_and(torch.half, torch.bfloat16, torch.complex64, torch.complex128),
           op=lambda inp, *args, **kwargs:
               wrapper_set_seed(torch.randn_like, inp, *args, **kwargs),
           supports_out=False,
           sample_inputs_func=sample_inputs_like_fns,
           supports_autograd=False,
           skips=(
               # AssertionError: JIT Test does not execute any logic
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )),
    OpInfo('randint_like',
           dtypes=all_types_and(torch.half, torch.bfloat16),
           op=lambda inp, *args, **kwargs:
               wrapper_set_seed(torch.randint_like, inp, *args, **kwargs),
           supports_out=False,
           sample_inputs_func=sample_inputs_randint_like,
           supports_autograd=False,
           skips=(
               # AssertionError: JIT Test does not execute any logic
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )),
    OpInfo('full_like',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_full_like,
           supports_autograd=False,
           skips=(
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )),
    OpInfo('new_zeros',
           op=lambda x, *args, **kwargs: x.new_zeros(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_new_fns,
           skips=(
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           ),
           supports_autograd=False),
    OpInfo('new_ones',
           op=lambda x, *args, **kwargs: x.new_ones(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_new_fns,
           skips=(
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           ),
           supports_autograd=False),
    OpInfo('new_empty',
           op=lambda x, *args, **kwargs: x.new_empty(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_new_fns,
           skips=(
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_variant_consistency_eager'),
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_noncontiguous_samples'),
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_conj_view'),
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_view'),
               # Empty tensor data is garbage so it's hard to make comparisons with it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_conj_view'),
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           ),
           supports_autograd=False),
    OpInfo('new_full',
           op=lambda x, *args, **kwargs: x.new_full(*args, **kwargs),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_new_full,
           skips=(
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           ),
           supports_autograd=False),
    OpInfo('multinomial',
           op=lambda inp, *args, **kwargs:
               wrapper_set_seed(torch.multinomial, inp, *args, **kwargs),
           method_variant=lambda inp, *args, **kwargs:
               wrapper_set_seed(torch.Tensor.multinomial, inp, *args, **kwargs),
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.half),
           supports_out=True,
           sample_inputs_func=sample_inputs_multinomial,
           skips=(
               # AssertionError: JIT Test does not execute any logic
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_out_warning')),
           supports_autograd=False),
    OpInfo('normal',
           op=lambda inp, *args, **kwargs:
               wrapper_set_seed(torch.normal, inp, *args, **kwargs),
           # The inplace variant (Tensor.normal_) is different from torch.normal
           inplace_variant=None,
           dtypes=floating_types_and(torch.bfloat16, torch.half),
           dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.half),
           supports_out=True,
           sample_inputs_func=sample_inputs_normal_tensor_first,
           skips=(
               # AssertionError: JIT Test does not execute any logic
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_out_warning'),),
           supports_autograd=False),
    OpInfo('normal',
           # This has its own variant b/c OpInfos assume the first arg is a Tensor but it is not here
           variant_test_name='number_mean',
           op=lambda std, mean, *args, **kwargs:
               wrapper_set_seed(torch.normal, mean, std, *args, **kwargs),
           # The inplace variant (Tensor.normal_) is different from torch.normal
           inplace_variant=None,
           dtypes=floating_types_and(torch.bfloat16, torch.half),
           dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.half),
           supports_out=True,
           sample_inputs_func=sample_inputs_normal_tensor_second,
           skips=(
               # AssertionError: JIT Test does not execute any logic
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # Seems like a bug:
               # The size of tensor a (0) must match the size of tensor b (4) at non-singleton dimension 1
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_out'),
               # UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_out_warning'),),
           supports_autograd=False),
    OpInfo('bernoulli',
           op=lambda inp, *args, **kwargs:
               wrapper_set_seed(torch.bernoulli, inp, *args, **kwargs),
           # The inplace variant (Tensor.bernoulli_) is different from torch.bernoulli
           inplace_variant=None,
           method_variant=lambda inp, *args, **kwargs:
               wrapper_set_seed(torch.Tensor.bernoulli, inp, *args, **kwargs),
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.half),
           supports_out=True,
           sample_inputs_func=sample_inputs_bernoulli,
           skips=(
               # AssertionError: JIT Test does not execute any logic
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # Expected RuntimeError when doing an unsafe cast from a result of
               # dtype torch.float32 into an out= with dtype torch.lon
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_out'),
               # UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_out_warning')),
           supports_autograd=False),
    OpInfo('scatter_add',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_scatter_add,
           error_inputs_func=error_inputs_scatter_and_scatter_add,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           ),
    OpInfo('stack',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_stack,
           assert_autodiffed=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           ),
    OpInfo('hstack',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # TODO: see https://github.com/pytorch/pytorch/issues/64709
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),
           )),
    BinaryUfuncInfo('hypot',
                    dtypes=floating_types_and(torch.bfloat16),
                    dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    sample_inputs_func=sample_inputs_hypot,
                    skips=(
                        # TypeError: hypot(): argument 'other' (position 2) must be Tensor, not float
                        DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs'),
                    )),
    OpInfo('histogram',
           dtypes=floating_types(),
           dtypesIfCUDA=_dispatch_dtypes(),  # histogram is only implemented on CPU
           sample_inputs_func=sample_inputs_histogram,
           supports_autograd=False,
           skips=(
               # JIT tests don't work with Tensor keyword arguments
               # https://github.com/pytorch/pytorch/issues/58507
               # RuntimeError:
               # undefined value tensor:
               #   File "<string>", line 3
               # def the_method(i0):
               #     return torch.histogram(i0, 1, weight=tensor(-0.5735, dtype=torch.float32), density=False)
               #                                          ~~~~~~ <--- HERE
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
               # Not Implemented on XLA.
               DecorateInfo(unittest.expectedFailure, 'TestOpInfo', device_type='xla'),
           )),
    OpInfo('histogramdd',
           dtypes=floating_types(),
           dtypesIfCUDA=_dispatch_dtypes(),  # histogramdd is only implemented on CPU
           sample_inputs_func=sample_inputs_histogramdd,
           supports_autograd=False,
           skips=(
               # JIT tests don't work with Tensor keyword arguments
               # https://github.com/pytorch/pytorch/issues/58507
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('histc',
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.int8, torch.int16, torch.int32, torch.int64),
           sample_inputs_func=sample_inputs_histc,
           supports_out=True,
           supports_autograd=False,
           skips=(
               # CUDA histc returns a float tensor but does not correctly warn when passed an integral out tensor
               # "AssertionError: RuntimeError not raised : Expected RuntimeError when doing an unsafe cast
               # from a result of dtype torch.float32 into an out= with dtype torch.long"
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cuda'),
           )),
    OpInfo('bincount',
           dtypes=integral_types_and(),
           sample_inputs_func=sample_inputs_bincount,
           supports_out=False,
           supports_autograd=False,
           skips=(
               # JIT tests don't work with Tensor keyword arguments
               # https://github.com/pytorch/pytorch/issues/58507
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('bucketize',
           dtypes=all_types_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16),
           sample_inputs_func=sample_inputs_bucketize,
           supports_autograd=False,
           skips=(
               # JIT tests don't work with Tensor keyword arguments
               DecorateInfo(unittest.skip("Expected failure!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('searchsorted',
           dtypes=all_types(),
           dtypesIfCPU=all_types_and(torch.bfloat16, torch.float16),
           dtypesIfCUDA=all_types_and(torch.float16),
           sample_inputs_func=sample_inputs_searchsorted,
           supports_autograd=False,
           ref=reference_searchsorted,
           skips=(
               # JIT tests don't work with Tensor keyword arguments
               # https://github.com/pytorch/pytorch/issues/58507
               DecorateInfo(unittest.skip("Expected failure!"), 'TestJit', 'test_variant_consistency_jit'),
           )),
    OpInfo('cat',
           ref=lambda input_seq, dim=0, **kwargs: np.concatenate(input_seq, axis=dim, **kwargs),
           aliases=('concat',),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_cat_concat,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           assert_autodiffed=True,
           skips=(
               # TODO: see https://github.com/pytorch/pytorch/issues/64709
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),
               # RuntimeError: Arguments for call not valid.
               #               Expected a value of type 'List[Tensor]' for argument
               #               'tensors' but instead found type 'Tensor (inferred)'.
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping'),)),
    OpInfo('vstack',
           aliases=('row_stack',),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # TODO: see https://github.com/pytorch/pytorch/issues/64709
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),
               # RuntimeError: _fn() Expected a value of type
               #   'Tensor (inferred)' for argument 't0' but instead found type 'tuple'.
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping'),)),
    OpInfo('dstack',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # TODO: see https://github.com/pytorch/pytorch/issues/64709
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),
           )),
    OpInfo('unfold',
           op=lambda x, *args: x.unfold(*args),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           check_batched_gradgrad=False,
           # See https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           skips=(
               # Skip operator schema test because this is a functional and not an operator
               DecorateInfo(unittest.expectedFailure, 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           ),
           sample_inputs_func=sample_inputs_unfold),
    OpInfo('msort',
           dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
           dtypesIfROCM=all_types_and(torch.float16),
           check_batched_gradgrad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # msort does not correctly warn when resizing out= inputs.
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),
               # Expected RuntimeError when doing an unsafe cast from a result of dtype
               #   torch.float32 into an out= with dtype torch.long
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cpu'),
           ),
           sample_inputs_func=sample_inputs_msort),
    OpInfo('movedim',
           aliases=('moveaxis',),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_movedim_moveaxis),
    OpInfo('renorm',
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_renorm),
    ShapeFuncInfo('repeat',
                  op=lambda x, dims: x.repeat(dims),
                  ref=np.tile,
                  dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                  supports_out=False,
                  supports_forward_ad=True,
                  supports_fwgrad_bwgrad=True,
                  sample_inputs_func=sample_repeat_tile),
    OpInfo('squeeze',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           assert_autodiffed=True,
           autodiff_fusible_nodes=[],  # aliases inputs, shouldn't be fused
           autodiff_nonfusible_nodes=[],  # aliases inputs, shouldn't be fused
           assert_jit_shape_analysis=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # vmap does not support inplace views
           check_inplace_batched_forward_grad=False,
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_squeeze),
    OpInfo('fill_',
           op=lambda x, scalar: torch.fill_(x.clone(), scalar),
           method_variant=None,
           inplace_variant=torch.Tensor.fill_,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # https://github.com/pytorch/pytorch/issues/66357
           check_batched_forward_grad=False,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           skips=(
               # JIT has issue when op is passed as lambda
               # AssertionError: JIT Test does not execute any logic
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
               # Fails due to a limitation of gradgradcheck
               # https://github.com/pytorch/pytorch/issues/59137
               DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_gradgrad'),
               DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_inplace_gradgrad'),
           ),
           sample_inputs_func=sample_inputs_fill_),
    OpInfo('resize_',
           op=lambda x, shape: x.clone().resize_(shape),
           method_variant=None,
           inplace_variant=torch.Tensor.resize_,
           # the test fails because resize_ doesn't work with imag views as expected by the test
           # https://github.com/pytorch/pytorch/issues/65945
           test_neg_view=False,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_autograd=False,
           skips=(
               # resize_ is raising an error on input that requires grad on purpose
               DecorateInfo(
                   unittest.skip('Skipped! Resizing of variables that require grad is not supported.'),
                   'TestGradients',
                   'test_nondifferentiable',
               ),
               DecorateInfo(unittest.skip("Allowed exception"), 'TestCommon', 'test_composite_compliance'),
           ),
           sample_inputs_func=sample_inputs_resize_ops),
    OpInfo('resize_as_',
           op=lambda x, other: torch.resize_as_(x.clone(), other),
           method_variant=None,
           inplace_variant=torch.Tensor.resize_as_,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_autograd=False,
           skips=(
               # resize_ is raising an error on input that requires grad on purpose
               DecorateInfo(
                   unittest.skip('Skipped! Resizing of variables that require grad is not supported.'),
                   'TestGradients',
                   'test_nondifferentiable',
               ),
           ),
           sample_inputs_func=sample_inputs_resize_ops),
    OpInfo('take_along_dim',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_take_along_dim,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    ShapeFuncInfo('tile',
                  ref=np.tile,
                  dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                  supports_out=False,
                  supports_forward_ad=True,
                  supports_fwgrad_bwgrad=True,
                  sample_inputs_func=sample_repeat_tile),
    OpInfo('trapz',  # TODO: in the future, 'trapz' should be made a proper alias of 'trapezoid'
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_trapezoid,
           skips=(
               # Dispatch stub: unsupported device typemeta
               DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad', device_type='meta'),
               # (ROCm) Memory access fault by GPU node-4 (Agent handle: 0x55860348e690) on address 0x7f0f4ddcb000
               # CUDA memory access fault  https://github.com/pytorch/pytorch/issues/72189
               DecorateInfo(unittest.skip("Skipped! CUDA/ROCm memory exception"), 'TestGradients', 'test_fn_fwgrad_bwgrad',
                            device_type='cuda', dtypes=[torch.float64, torch.complex128]),
           )),
    OpInfo('trapezoid',
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_trapezoid,
           skips=(
               # Dispatch stub: unsupported device typemeta
               DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad', device_type='meta'),
               # (ROCm) Memory access fault by GPU node-4 (Agent handle: 0x55bbf53d5500) on address 0x7fe536eb5000
               # CUDA memory access fault  https://github.com/pytorch/pytorch/issues/72189
               DecorateInfo(unittest.skip("Skipped! CUDA/ROCm memory exception"), 'TestGradients', 'test_fn_fwgrad_bwgrad',
                            device_type='cuda', dtypes=[torch.float64, torch.complex128]),
           )),
    OpInfo('cumulative_trapezoid',
           dtypes=all_types_and_complex_and(),
           dtypesIfCUDA=all_types_and_complex_and(torch.bfloat16, torch.float16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           supports_out=False,
           sample_inputs_func=sample_cumulative_trapezoid,),
    OpInfo('unsqueeze',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # vmap does not support inplace views
           check_inplace_batched_forward_grad=False,
           assert_jit_shape_analysis=True,
           assert_autodiffed=True,
           autodiff_fusible_nodes=[],  # aliases inputs, shouldn't be fused
           autodiff_nonfusible_nodes=[],  # aliases inputs, shouldn't be fused
           sample_inputs_func=sample_unsqueeze),
    BinaryUfuncInfo('xlogy',
                    aliases=('special.xlogy',),
                    dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
                    promotes_int_to_float=True,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    safe_casts_outputs=True,
                    sample_inputs_func=sample_inputs_xlogy),
    OpInfo('zero_',
           op=lambda x: torch.zero_(x.clone()),
           method_variant=None,
           inplace_variant=torch.Tensor.zero_,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # JIT has issue when op is passed as lambda
               # AssertionError: JIT Test does not execute any logic
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           sample_inputs_func=sample_inputs_zero_),
    BinaryUfuncInfo('special.xlog1py',
                    aten_name='special_xlog1py',
                    dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
                    backward_dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                    promotes_int_to_float=True,
                    safe_casts_outputs=True,
                    supports_forward_ad=True,
                    supports_fwgrad_bwgrad=True,
                    sample_inputs_func=sample_inputs_xlog1py),
    BinaryUfuncInfo('special.zeta',
                    aten_name='special_zeta',
                    dtypes=all_types_and(torch.bool),
                    promotes_int_to_float=True,
                    supports_autograd=False,
                    safe_casts_outputs=True,
                    sample_inputs_func=sample_inputs_binary_pwise),
    # OpInfo entry to verify the gradient formula of `other`/`q`
    BinaryUfuncInfo('special.zeta',
                    op=lambda q, x, **kwargs: torch.special.zeta(x, q, **kwargs),
                    aten_name='special_zeta',
                    variant_test_name='grad',
                    dtypes=all_types_and(torch.bool),
                    promotes_int_to_float=True,
                    supports_autograd=True,
                    safe_casts_outputs=True,
                    decorators=[
                        # Derivative wrt first tensor not implemented
                        DecorateInfo(unittest.expectedFailure, "TestCommon",
                                     "test_floating_inputs_are_differentiable")
                    ],
                    skips=(
                        # Lambda doesn't work in JIT test
                        # AssertionError: JIT Test does not execute any logic
                        DecorateInfo(unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"),
                    ),
                    sample_inputs_func=sample_inputs_zeta),
    OpInfo('logsumexp',
           aliases=('special.logsumexp',),
           dtypes=all_types_and(torch.bool, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.bool, torch.bfloat16, torch.half),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_logsumexp),
    OpInfo('trace',
           dtypes=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_trace),
    OpInfo('transpose',
           aliases=('swapdims', 'swapaxes'),
           assert_jit_shape_analysis=True,
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # vmap does not support inplace views
           check_inplace_batched_forward_grad=False,
           sample_inputs_func=sample_inputs_transpose_swapdims),
    OpInfo('T',
           op=lambda x: x.T,
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(  # Lambda doesn't work in JIT test
               DecorateInfo(unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"),),
           sample_inputs_func=sample_inputs_T),
    OpInfo('H',
           op=lambda x: x.H,
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(  # Lambda doesn't work in JIT test
               DecorateInfo(unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"),),
           sample_inputs_func=sample_inputs_T),
    OpInfo('mT',
           op=lambda x: x.mT,
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(  # Lambda doesn't work in JIT test
               DecorateInfo(unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"),),
           sample_inputs_func=sample_inputs_adjoint),
    OpInfo('mH',
           op=lambda x: x.mH,
           aliases=('adjoint',),
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(  # Lambda doesn't work in JIT test
               DecorateInfo(unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"),),
           sample_inputs_func=sample_inputs_adjoint),
    OpInfo('tril',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_tril_triu),
    OpInfo('triu',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_tril_triu),
    OpInfo('kron',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_kron),
    OpInfo('inner',
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16),
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_inner,
           ),
    OpInfo('tensordot',
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16),
           safe_casts_outputs=True,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           sample_inputs_func=sample_inputs_tensordot,
           skips=(
               # Skip operator schema test because this is a functional and not an operator.
               # Reference: https://github.com/pytorch/pytorch/issues/54574
               DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )
           ),
    OpInfo('to_sparse',
           op=lambda x, *args: x.to_sparse(*args),
           sample_inputs_func=sample_inputs_to_sparse,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           backward_dtypes=floating_types(),
           backward_dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           skips=(
               # NotImplementedError: Could not run 'aten::normal_' with arguments from the 'SparseCPU' backend
               DecorateInfo(unittest.skip(""), 'TestCommon', 'test_noncontiguous_samples'),
               # TODO: FIXME: complex inputs requiring grad error in forward
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_dtypes'),
               # JIT has issue when op is passed as lambda
               # NotImplementedError: Cannot access storage of SparseTensorImpl
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
               # Allowed exception: sparse tensors don't have strides
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
           )
           ),
    OpInfo('logcumsumexp',
           dtypes=floating_types_and(),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           backward_dtypesIfCUDA=floating_types_and(),
           skips=(
               # AssertionError: UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning', device_type='cuda'),
           ),
           sample_inputs_func=sample_inputs_logcumsumexp),
    UnaryUfuncInfo('sigmoid',
                   aliases=('special.expit', 'nn.functional.sigmoid'),
                   ref=reference_sigmoid if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.complex64: 1e-1,
                                                  torch.bfloat16: 1e-2}),),
                   skips=(
                       # TODO: FIXME: sigmoid fails on complex inputs that require grad
                       DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_dtypes'),
                       # Reference: https://github.com/pytorch/pytorch/issues/56012
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cuda', dtypes=[torch.complex64]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cuda', dtypes=[torch.complex64]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       # alias, nn.functional.sigmoid, will produce (because of warning string saved):
                       # "RuntimeError: Expected to not find "sigmoid" but found it"
                       DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   assert_autodiffed=True,
                   # sigmoid(z) = 1 / (1 + exp(-z)), at z = j * pi * odd_number, the denominator is zero
                   reference_numerics_filter=NumericsFilter(
                       condition=lambda x: (close_to_int(x / (math.pi * 1j))
                                            if x.is_complex() else x.new_tensor(False, dtype=torch.bool)),
                       safe_val=0)),
    UnaryUfuncInfo('digamma',
                   ref=scipy.special.digamma if TEST_SCIPY else _NOTHING,
                   aliases=('special.psi', 'special.digamma',),
                   decorators=(precisionOverride({torch.float16: 5e-1}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('special.entr',
                   ref=scipy.special.entr if TEST_SCIPY else _NOTHING,
                   aten_name='special_entr',
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   decorators=(precisionOverride({torch.float16: 1e-1,
                                                  torch.bfloat16: 1e-1}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    dtypes=[torch.bfloat16, torch.float16]),
                   ),
                   supports_inplace_autograd=False,
                   safe_casts_outputs=True,
                   sample_inputs_func=sample_inputs_entr),
    UnaryUfuncInfo('special.ndtri',
                   ref=scipy.special.ndtri if TEST_SCIPY else _NOTHING,
                   domain=(0, 1),
                   aten_name='special_ndtri',
                   dtypes=all_types_and(torch.bool),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('erf',
                   ref=scipy.special.erf if TEST_SCIPY else _NOTHING,
                   aliases=('special.erf', ),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2}),),
                   skips=(
                       DecorateInfo(unittest.skip("Skipped! sparse backward not supported"),
                                    'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),

                   ),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   assert_jit_shape_analysis=True,
                   supports_sparse=True,
                   supports_sparse_csr=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('erfc',
                   ref=scipy.special.erfc if TEST_SCIPY else _NOTHING,
                   aliases=('special.erfc', ),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('erfinv',
                   ref=scipy.special.erfinv if TEST_SCIPY else _NOTHING,
                   aliases=('special.erfinv', ),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2,
                                                  torch.float32: 1e-4}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   supports_sparse_csr=True,
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   domain=(-1, 1),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/49155#issuecomment-742664611
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    active_if=TEST_SCIPY and LooseVersion(scipy.__version__) < "1.4.0"),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    active_if=TEST_SCIPY and LooseVersion(scipy.__version__) < "1.4.0"),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    active_if=TEST_SCIPY and LooseVersion(scipy.__version__) < "1.4.0"),
                   )),
    UnaryUfuncInfo('lgamma',
                   ref=reference_lgamma if TEST_SCIPY else _NOTHING,
                   aliases=('special.gammaln', ),
                   decorators=(precisionOverride({torch.float16: 7e-1}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/50140#discussion_r552615345
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    device_type='cpu', dtypes=[torch.bfloat16]),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_normal',
                                    device_type='cpu', dtypes=[torch.bfloat16]),
                       # Reference: https://github.com/pytorch/pytorch/pull/50140#issuecomment-756150214
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                    dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS),
                       DecorateInfo(unittest.skip("Skipped!"), 'TestUnaryUfuncs', 'test_reference_numerics_hard',
                                    dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS),
                   ),
                   safe_casts_outputs=True,
                   # lgamma have multiple singularities at x <= 0
                   reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)),
    OpInfo(
        'logdet',
        dtypes=floating_types(),
        supports_out=False,
        sample_inputs_func=sample_inputs_logdet,
        decorators=(skipCPUIfNoLapack, skipCUDAIfNoMagma, skipCUDAIfRocm)),
    # `log_softmax` supports different dtypes based on whether `dtype` argument,
    # is passed or not. Hence two OpInfo entries, one with dtype and other without.
    OpInfo(
        'log_softmax',
        aliases=('special.log_softmax', 'nn.functional.log_softmax'),
        supports_out=False,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        sample_inputs_func=sample_inputs_softmax_variant,
        assert_autodiffed=True),
    OpInfo(
        'log_softmax',
        variant_test_name='dtype',
        aliases=('special.log_softmax', 'nn.functional.log_softmax'),
        supports_out=False,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        sample_inputs_func=partial(sample_inputs_softmax_variant, with_dtype=True),
        assert_autodiffed=True),
    UnaryUfuncInfo('logit',
                   ref=scipy.special.logit if TEST_SCIPY else _NOTHING,
                   domain=(0, 1),
                   aliases=('special.logit', ),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-1,
                                                  torch.float16: 5e-1}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   sample_inputs_func=sample_inputs_logit,
                   safe_casts_outputs=True),
    OpInfo('where',
           # Currently only the `input` is tested in gradcheck.
           # If we pass `condition` first, none of the input which supports
           # autograd will be tested. Hence the following lambda.
           op=lambda self, condition, other: torch.where(condition, self, other),
           ref=lambda self, condition, other: np.where(condition, self, other),
           sample_inputs_func=sample_inputs_where,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           skips=(
               # test does not work with passing lambda for op
               # AssertionError: False is not true :
               # Failure in testing nodes' autodifferentiation.
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16)),
    OpInfo('nonzero',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           sample_inputs_func=sample_inputs_nonzero,
           supports_autograd=False,
           skips=(
               # https://github.com/pytorch/pytorch/issues/67458
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
               # nonzero is not raising a warning when the out is resized
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),
               # Can't find schemas for this operator for some reason
               DecorateInfo(unittest.expectedFailure, 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )),
    # `torch.norm` has multiple code paths depending on the value of `p`.
    # These paths have different dtype support. Also JIT supports,
    # most variants but not all of them. So we split the OpInfo entries,
    # for `norm` based on the code-paths and JIT support.
    OpInfo('norm',
           sample_inputs_func=sample_inputs_norm,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16)),
    OpInfo('norm',
           variant_test_name='nuc',
           sample_inputs_func=sample_inputs_norm_nuc,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
           check_batched_gradgrad=False,
           check_batched_forward_grad=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           dtypes=floating_and_complex_types(),
           dtypesIfCUDA=floating_and_complex_types(),
           skips=(
               # Pre-existing condition; Needs to be fixed
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
               # RuntimeError not raised :
               # Expected RuntimeError when calling with input.device=cpu and out.device=cuda
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),
               # RuntimeError:
               # Arguments for call are not valid.
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.complex64, torch.float32,)),  # noqa: B950
           )
           ),
    OpInfo('norm',
           variant_test_name='fro',
           sample_inputs_func=sample_inputs_norm_fro,
           dtypes=floating_and_complex_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           skips=(
               # Pre-existing condition; Needs to be fixed
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
               # Expected RuntimeError when calling with input.device=cpu and out.device=cuda
               DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),
               # Arguments for call are not valid.
               DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.complex64, torch.float32,)),  # noqa: B950
           )),
    OpInfo('norm',
           variant_test_name='inf',
           sample_inputs_func=sample_inputs_norm_inf,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           backward_dtypesIfCPU=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           skips=(
               # https://github.com/pytorch/pytorch/issues/67517
               DecorateInfo(unittest.skip("Skipped!"), 'TestCommon', 'test_noncontiguous_samples'),
               # following 2 tests failed intermittenly
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_grad', device_type='cpu', dtypes=(torch.complex128,)),  # noqa: B950
               DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_fn_gradgrad', device_type='cpu', dtypes=(torch.complex128,)),  # noqa: B950
           )
           ),
    OpInfo('t',
           sample_inputs_func=sample_inputs_t,
           supports_out=False,
           supports_forward_ad=True,
           supports_fwgrad_bwgrad=True,
           # vmap does not support inplace views
           check_inplace_batched_forward_grad=False,
           autodiff_fusible_nodes=[],  # aliases inputs, shouldn't be fused
           autodiff_nonfusible_nodes=[],  # aliases inputs, shouldn't be fused
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           assert_autodiffed=True,),
    UnaryUfuncInfo('special.erfcx',
                   ref=scipy.special.erfcx if TEST_SCIPY else _NOTHING,
                   aten_name='special_erfcx',
                   decorators=(toleranceOverride({torch.float32: tol(atol=0, rtol=4e-6), }),),
                   dtypes=all_types_and(torch.bool),
                   supports_forward_ad=True,
                   supports_fwgrad_bwgrad=True,
                   safe_casts_outputs=True),
    OpInfo(
        "nn.functional.dropout",
        op=lambda input, *args, **kwargs:
            wrapper_set_seed(torch.nn.functional.dropout, input, *args, **kwargs),
        ref=_NOTHING,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        skips=(
            # Probably because we have used lambda for the op here
            # AssertionError: JIT Test does not execute any logic
            DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
            # inplace variant dispatches to dropout kernel, while on CUDA
            # the op dispatches to _fused_dropout (with a few more conditions)
            # hence, different values and this skip here
            DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_view', device_type='cuda'),
            # On CUDA, the op is dispatched (and a few more conditions) to
            # _fused_dropout, which doesn't support forward AD
            DecorateInfo(unittest.skip("Skipped!"), 'TestGradients', 'test_forward_mode_AD', device_type='cuda'),
            # NotImplementedError: Trying to use forward AD with native_dropout that does not support it
            DecorateInfo(unittest.expectedFailure, 'TestGradients', 'test_fn_fwgrad_bwgrad',
                         device_type='cuda', dtypes=[torch.float64]),),
        gradcheck_wrapper=wrapper_set_seed,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # https://github.com/pytorch/pytorch/issues/66357
        check_batched_forward_grad=False,
        supports_out=False,
        sample_inputs_func=sample_inputs_dropout,
        inplace_variant=lambda input, *args, **kwargs:
            wrapper_set_seed(torch.nn.functional.dropout, input, *args, **kwargs, inplace=True)),
    OpInfo(
        "nn.functional.dropout2d",
        op=lambda input, *args, **kwargs:
            wrapper_set_seed(torch.nn.functional.dropout2d, input, *args, **kwargs),
        ref=_NOTHING,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        skips=(
            # torch.autograd.gradcheck.GradcheckError: While computing batched gradients, got:
            # vmap: We do not yet support calling random operations inside of vmap.
            # Please perform random operations outside of vmap as a workaround
            DecorateInfo(unittest.expectedFailure, 'TestGradients', "test_forward_mode_AD"),
            DecorateInfo(unittest.expectedFailure, 'TestGradients', "test_inplace_forward_mode_AD"),
            # Probably because we have used lambda for the op here
            # AssertionError: JIT Test does not execute any logic
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),),
        gradcheck_wrapper=wrapper_set_seed,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
        # As per the docs, valid input dims are (3, 4)
        sample_inputs_func=partial(sample_inputs_dropout, valid_input_dim=(3, 4)),
        inplace_variant=lambda input, *args, **kwargs:
            wrapper_set_seed(torch.nn.functional.dropout2d, input, *args, **kwargs, inplace=True)),
    # In training mode, feature_alpha_dropout currently doesn't support inputs of complex dtype
    # unlike when `train=False`, it supports complex inputs, hence 2 OpInfos to cover all cases
    OpInfo(
        "nn.functional.feature_alpha_dropout",
        op=lambda input, *args, **kwargs:
            wrapper_set_seed(torch.nn.functional.feature_alpha_dropout, input, *args, **kwargs),
        variant_test_name="with_train",
        ref=_NOTHING,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        skips=(
            # torch.autograd.gradcheck.GradcheckError: While computing batched gradients, got:
            # vmap: We do not yet support calling random operations inside of vmap.
            # Please perform random operations outside of vmap as a workaround
            DecorateInfo(unittest.expectedFailure, 'TestGradients', "test_forward_mode_AD"),
            DecorateInfo(unittest.expectedFailure, 'TestGradients', "test_inplace_forward_mode_AD"),
            # Probably because we have used lambda for the op here
            # AssertionError: JIT Test does not execute any logic
            DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),),
        gradcheck_wrapper=wrapper_set_seed,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
        # As per the docs, valid input dims are (4, 5)
        sample_inputs_func=partial(sample_inputs_dropout, train=True, valid_input_dim=(4, 5)),
        inplace_variant=lambda input, *args, **kwargs:
            wrapper_set_seed(torch.nn.functional.feature_alpha_dropout, input, *args, **kwargs, inplace=True)),
    OpInfo(
        "nn.functional.feature_alpha_dropout",
        op=lambda input, *args, **kwargs:
            wrapper_set_seed(torch.nn.functional.feature_alpha_dropout, input, *args, **kwargs),
        variant_test_name="without_train",
        ref=_NOTHING,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        skips=(
            # Probably because we have used lambda for the op here
            # AssertionError: JIT Test does not execute any logic
            DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),),
        gradcheck_wrapper=wrapper_set_seed,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
        sample_inputs_func=partial(sample_inputs_dropout, train=False),
        inplace_variant=lambda input, *args, **kwargs:
            wrapper_set_seed(torch.nn.functional.feature_alpha_dropout, input, *args, **kwargs, inplace=True)),
    OpInfo(
        "nn.functional.one_hot",
        ref=reference_one_hot,
        supports_out=False,
        dtypes=_dispatch_dtypes((torch.int64,)),
        sample_inputs_func=sample_inputs_one_hot,
    ),
    OpInfo(
        "nn.functional.embedding",
        # We use lambda to reshuffle the positional arguments.
        # This is because currently only the `input` field of SampleInput
        # is tested in gradient tests.
        op=lambda weight, idx, **kwargs: torch.nn.functional.embedding(idx, weight, **kwargs),
        dtypes=floating_types_and(torch.bfloat16, torch.float16),
        sample_inputs_func=sample_inputs_embedding,
        skips=(
            # Does not work with lambda
            # Raises : JIT Test does not execute any logic
            DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
            # Reference: https://github.com/pytorch/pytorch/issues/67084
            DecorateInfo(unittest.skip("Skipped!"), 'TestMathBits', 'test_neg_view', device_type='cuda'),
        ),
        supports_out=False,
    ),
    OpInfo(
        "nn.functional.embedding_bag",
        # We use lambda to reshuffle the positional arguments.
        # This is because currently only the `input` field of SampleInput
        # is tested in gradient tests.
        op=lambda weight, idx, **kwargs: torch.nn.functional.embedding_bag(idx, weight, **kwargs),
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16),
        # backward is not supported for mode `max` and dtype `bfloat16`
        backward_dtypesIfCUDA=floating_types_and(torch.float16),
        sample_inputs_func=sample_inputs_embedding_bag,
        skips=(
            # Does not work with lambda
            # Raises : JIT Test does not execute any logic
            DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),
        ),
        gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
        supports_out=False,
        supports_gradgrad=False,
    ),
    OpInfo(
        "nn.functional.softplus",
        ref=reference_softplus,
        sample_inputs_func=sample_inputs_softplus,
        supports_forward_ad=True,
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16),
        supports_out=False,
    ),
    OpInfo(
        "linalg.tensorinv",
        ref=np.linalg.tensorinv,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_tensorinv,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
    ),
    OpInfo(
        "linalg.tensorsolve",
        ref=lambda a, b, dims=None: np.linalg.tensorsolve(a, b, axes=dims),
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_tensorsolve,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
    ),
    OpInfo(
        "nn.functional.mse_loss",
        ref=reference_mse_loss,
        sample_inputs_func=sample_inputs_mse_loss,
        supports_out=False,
        supports_forward_ad=True,
        dtypes=floating_types_and(torch.float16),
        backward_dtypesIfCPU=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16),
        skips=(
            # RuntimeError: input->type()->kind() == TypeKind::OptionalType
            # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":252,
            # please report a bug to PyTorch.
            DecorateInfo(unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit", dtypes=(torch.float32,),),
        ),
    ),
    OpInfo(
        "nn.functional.grid_sample",
        ref=_NOTHING,
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types_and(torch.float16),
        supports_out=False,
        sample_inputs_func=sample_inputs_grid_sample,
        supports_gradgrad=False,
        gradcheck_nondet_tol=1e-15),
    OpInfo(
        "argwhere",
        ref=np.argwhere,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        supports_autograd=False,
        sample_inputs_func=sample_inputs_argwhere,
    ),
    ReductionOpInfo(
        'all',
        identity=True,
        supports_multiple_dims=False,
        supports_out=False,
        supports_autograd=False,
        result_dtype=torch.bool,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.all),
        skips=(
            # FIXME: does not support passing keepdim without dim
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_default_keepdim'),
            # FIXME: does not support dim=None
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_none'),
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_none_keepdim'),
            # FIXME: uint8 input returns uint8 instead of bool
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_result_dtype', dtypes=[torch.uint8]),
        ),
    ),
    ReductionOpInfo(
        'any',
        identity=False,
        supports_multiple_dims=False,
        supports_out=False,
        supports_autograd=False,
        result_dtype=torch.bool,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.any),
        skips=(
            # FIXME: does not support passing keepdim without dim
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_default_keepdim'),
            # FIXME: does not support dim=None
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_none'),
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_none_keepdim'),
            # FIXME: uint8 input returns uint8 instead of bool
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_result_dtype', dtypes=[torch.uint8]),
        ),
    ),
    ReductionOpInfo(
        'amax',
        nan_policy='propagate',
        dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
        ref=reference_reduction_numpy(np.amax),
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'),
        ),
    ),
    ReductionOpInfo(
        'amin',
        nan_policy='propagate',
        dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
        ref=reference_reduction_numpy(np.amin),
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'),
        ),
    ),
    ReductionOpInfo(
        'argmax',
        supports_multiple_dims=False,
        supports_autograd=False,
        result_dtype=torch.int64,
        dtypes=all_types_and(torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.argmax, supports_keepdims=False),
        skips=(
            # FIXME: keepdim parameter is ignored when dim=None
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_default_keepdim'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none_keepdim'),
        ),
    ),
    ReductionOpInfo(
        'argmin',
        supports_multiple_dims=False,
        supports_autograd=False,
        result_dtype=torch.int64,
        dtypes=all_types_and(torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.argmin, supports_keepdims=False),
        skips=(
            # FIXME: keepdim parameter is ignored when dim=None
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_default_keepdim'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none_keepdim'),
        ),
    ),
    ReductionOpInfo(
        'count_nonzero',
        identity=0,
        supports_out=False,
        supports_autograd=False,
        result_dtype=torch.int64,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        sample_inputs_func=sample_inputs_reduction_count_nonzero,
        ref=reference_reduction_numpy(np.count_nonzero),
        skips=(
            # FIXME: count_nonzero does not accept keepdim kwarg
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_default_keepdim'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none_keepdim'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_single_keepdim'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_multi_keepdim'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_multi_unsorted_keepdim'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_offbounds_keepdim'),
            # FIXME: dim=[] reduces all dimensions
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
        ),
    ),
    ReductionOpInfo(
        'mean',
        nan_policy='propagate',
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        assert_autodiffed=True,
        assert_jit_shape_analysis=True,
        promotes_int_to_float=True,
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.mean),
        skips=(
            # FIXME: mean does not support passing keepdim without passing dim
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_default_keepdim'),
            # FIXME: mean reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # FIXME: mean does not support passing None to dim
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none_keepdim'),
            # FIXME: improve precision
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_small_input',
                         dtypes=[torch.float16]),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_extremal_values',
                         device_type='cuda', dtypes=[torch.complex64]),
        ),
    ),
    ReductionOpInfo(
        'nanmean',
        nan_policy='omit',
        assert_autodiffed=True,
        promotes_int_to_float=True,
        dtypes=floating_types_and(torch.float16, torch.bfloat16),
        sample_inputs_func=sample_inputs_nan_reduction(supports_multiple_dims=True),
        ref=reference_reduction_numpy(np.nanmean),
        skips=(
            # AssertionError: False is not true :
            # Failure in testing nodes' autodifferentiation.
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
            # FIXME: prod reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # FIXME: improve precision
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_small_input',
                         dtypes=[torch.float16]),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_duplicate_values',
                         device_type='cuda', dtypes=[torch.float16]),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_extremal_values',
                         device_type='cuda', dtypes=[torch.complex64]),
        ),
    ),
    ReductionOpInfo(
        'std',
        nan_policy='propagate',
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        assert_autodiffed=True,
        promotes_int_to_float=True,
        dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
        dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_std_var,
        ref=reference_std_var(np.std),
        generate_args_kwargs=generate_std_var_kwargs,
        skips=(
            # FIXME: cannot specify keepdim without dim
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_default_keepdim'),
            # FIXME: dim=None not supported
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none_keepdim'),
            # FIXME: dim=[] reduces all dimensions
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # TODO(@heitorschueroff) std return float for complex types
            # need to find a better way to model result dtype
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_result_dtype'),
            # FIXME: improve precision
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_small_input'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_duplicate_values'),
            # NumPy is giving NaN for this
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_large_input'),
        ),
    ),
    ReductionOpInfo(
        'var',
        nan_policy='propagate',
        supports_out=False,
        assert_autodiffed=True,
        promotes_int_to_float=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
        dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_std_var,
        ref=reference_std_var(np.var),
        generate_args_kwargs=generate_std_var_kwargs,
        skips=(
            # FIXME: cannot specify keepdim without dim
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_default_keepdim'),
            # FIXME: dim=None not supported
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none_keepdim'),
            # FIXME: dim=[] reduces all dimensions
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # TODO(@heitorschueroff) std return float for complex types
            # need to find a better way to model result dtype
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_result_dtype'),
            # FIXME: improve precision
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_small_input'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_duplicate_values'),
            # NumPy is giving NaN for this
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_large_input'),
        ),
    ),
    ReductionOpInfo(
        'prod',
        identity=1,
        nan_policy='propagate',
        supports_multiple_dims=False,
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        promotes_int_to_int64=True,
        gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
        dtypes=all_types_and_complex_and(torch.bool),
        dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        sample_inputs_func=sample_inputs_prod,
        ref=reference_reduction_numpy(np.prod),
        skips=(
            # FIXME: prod does not support passing keepdim without passing dim
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_default_keepdim'),
            # FIXME: prod reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # FIXME: prod does not support passing None to dim
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none_keepdim'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_small_input',
                         dtypes=[torch.float16, torch.complex64]),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_duplicate_values',
                         dtypes=[torch.uint8, torch.float16, torch.complex64]),
        ),
    ),
    ReductionOpInfo(
        'sum',
        identity=0,
        nan_policy='propagate',
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        promotes_int_to_int64=True,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.sum),
        skips=(
            # FIXME: sum does not support passing keepdim without passing dim
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_default_keepdim'),
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # FIXME: sum does not support passing None to dim
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none_keepdim'),
            # FIXME: improve precision
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_small_input',
                         dtypes=[torch.float16]),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_duplicate_values',
                         dtypes=[torch.float16]),
        ),
    ),
    ReductionOpInfo(
        'nansum',
        identity=0,
        nan_policy='omit',
        supports_out=False,
        promotes_int_to_int64=True,
        dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.nansum),
        skips=(
            # FIXME: nansum does not support passing keepdim without passing dim
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_default_keepdim'),
            # FIXME: nansum reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # FIXME: nansum does not support passing None to dim
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_none_keepdim'),
            # FIXME: improve precision
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_small_input',
                         dtypes=[torch.float16]),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_ref_duplicate_values',
                         dtypes=[torch.float16]),
        ),
    ),
    ReductionOpInfo(
        '_masked.sum',
        ref=reference_reduction_numpy(np.sum),
        method_variant=None,
        identity=0,
        nan_policy='propagate',
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        promotes_int_to_int64=False,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # RuntimeError: undefined value tensor
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
        ),
        decorators=[
            DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=1e-03, rtol=1e-03)}),
                         'TestReductions', 'test_reference_masked'),
            DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-03)}),
                         'TestReductions', 'test_reference_masked'),
            DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-03)}),
                         'TestReductions', 'test_ref_small_input'),
        ],
        sample_inputs_func=sample_inputs_masked_reduction
    ),
    ReductionOpInfo(
        '_masked.prod',
        ref=reference_reduction_numpy(np.prod),
        method_variant=None,
        identity=1,
        nan_policy='propagate',
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        promotes_int_to_int64=True,
        # FIXME: "prod_cpu" not implemented for 'BFloat16'
        # FIXME: "prod_cpu" not implemented for 'Half'
        dtypes=all_types_and_complex_and(torch.bool),
        dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        skips=(
            # NotSupportedError: Compiled functions can't ... use keyword-only arguments with defaults
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
        ),
        decorators=[
            DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-02)}),
                         'TestReductions', 'test_reference_masked'),
            DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-03)}),
                         'TestReductions', 'test_ref_duplicate_values'),
        ],
        sample_inputs_func=sample_inputs_masked_reduction
    ),
    ReductionOpInfo(
        '_masked.amax',
        nan_policy='propagate',
        supports_out=False,
        dtypes=all_types_and(torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.amax),
        skips=(
            # FIXME: amax reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # RuntimeError: Unknown builtin op: aten::iinfo
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation
    ),
    ReductionOpInfo(
        '_masked.amin',
        nan_policy='propagate',
        supports_out=False,
        dtypes=all_types_and(torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.amin),
        skips=(
            # FIXME: amax reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # RuntimeError: Unknown builtin op: aten::iinfo
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation
    ),
    ReductionOpInfo(
        '_masked.mean',
        ref=reference_reduction_numpy(np.mean) if np.lib.NumpyVersion(np.__version__) >= '1.20.2' else None,
        method_variant=None,
        nan_policy='propagate',
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        promotes_int_to_float=True,
        dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16, torch.bool),
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # RuntimeError: undefined value tensor
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
        ),
        decorators=[
            DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-03)}),
                         'TestReductions', 'test_reference_masked'),
        ],
        sample_inputs_func=sample_inputs_masked_reduction,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation
    ),
    ReductionOpInfo(
        '_masked.norm',
        identity=0,
        method_variant=None,
        nan_policy='propagate',
        supports_out=False,
        promotes_int_to_float=True,
        dtypes=floating_types_and(torch.float16, torch.bfloat16),
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # torch.jit.frontend.NotSupportedError: Compiled functions
            # can't take variable number of arguments or use
            # keyword-only arguments with defaults
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
        ),
        sample_inputs_func=sample_inputs_masked_norm,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation
    ),
    ReductionOpInfo(
        '_masked.var',
        ref=reference_reduction_numpy(np.var) if np.lib.NumpyVersion(np.__version__) >= '1.20.2' else None,
        method_variant=None,
        nan_policy='propagate',
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        promotes_int_to_float=True,
        dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty'),
            DecorateInfo(unittest.skip("Skipped!"), 'TestReductions', 'test_dim_empty_keepdim'),
            # RuntimeError: undefined value tensor
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
        ),
        decorators=[
            DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-02)}),
                         'TestReductions', 'test_reference_masked'),
            DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-02)}),
                         'TestReductions', 'test_ref_small_input'),
            DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-02)}),
                         'TestMasked', 'test_reference_masked'),
        ],
        sample_inputs_func=sample_inputs_masked_var,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation
    ),
    OpInfo(
        '_masked.softmax',
        method_variant=None,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_masked_softmax,
        skips=(
            # torch.jit.frontend.NotSupportedError: Compiled
            # functions can't take variable number of arguments or
            # use keyword-only arguments with defaults
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
        ),
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        supports_out=False),
    OpInfo(
        '_masked.log_softmax',
        method_variant=None,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_masked_softmax,
        skips=(
            # torch.jit.frontend.NotSupportedError: Compiled
            # functions can't take variable number of arguments or
            # use keyword-only arguments with defaults
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
        ),
        decorators=[
            DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=1e-02, rtol=1e-02)}),
                         'TestMasked', 'test_reference_masked'),
        ],
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        supports_out=False),
    OpInfo(
        '_masked.softmin',
        method_variant=None,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_masked_softmax,
        skips=(
            # torch.jit.frontend.NotSupportedError: Compiled
            # functions can't take variable number of arguments or
            # use keyword-only arguments with defaults
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
        ),
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        supports_out=False),
    OpInfo(
        '_masked.normalize',
        method_variant=None,
        dtypes=floating_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_masked_normalize,
        skips=(
            # torch.jit.frontend.NotSupportedError: Compiled
            # functions can't take variable number of arguments or
            # use keyword-only arguments with defaults
            DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
            # RuntimeError: "clamp_min_cpu" not implemented for 'Half'
            DecorateInfo(unittest.skip("Skipped!"), 'TestMasked', 'test_reference_masked',
                         device_type='cpu', dtypes=[torch.half]),
        ),
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        supports_out=False),
    OpInfo(
        "nn.functional.ctc_loss",
        ref=_NOTHING,
        dtypes=floating_types(),
        supports_out=False,
        sample_inputs_func=sample_inputs_ctc_loss,
        skips=(
            # https://github.com/pytorch/pytorch/issues/67462
            # torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0
            DecorateInfo(
                unittest.expectedFailure,
                "TestGradients",
                "test_fn_grad",
                dtypes=(torch.float64,),
            ),
            # RuntimeError: derivative for aten::_ctc_loss_backward is not implemented
            DecorateInfo(
                unittest.expectedFailure,
                "TestGradients",
                "test_fn_gradgrad",
                dtypes=(torch.float64,),
            ),
            # RuntimeError: derivative for aten::_ctc_loss_backward is not implemented
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                dtypes=(torch.float32,),
            ),
            # Operation calls data_ptr() somewhere; needs to be fixed
            DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_composite_compliance'),
        ),
    ),
    OpInfo(
        "nn.functional.cosine_embedding_loss",
        ref=_NOTHING,
        dtypes=all_types_and(torch.bfloat16, torch.bool),
        dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16, torch.bool),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_cosine_embedding_loss,
    ),
    OpInfo(
        "nn.functional.nll_loss",
        ref=_NOTHING,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        supports_out=False,
        sample_inputs_func=sample_inputs_nll_loss,
        decorators=[
            # FIXME: Derivative wrt. weight not implemented
            DecorateInfo(unittest.expectedFailure, "TestCommon",
                         "test_floating_inputs_are_differentiable")],
        skips=(
            # RuntimeError:
            # undefined value tensor:
            #   File "<string>", line 3
            # def the_method(i0, i1):
            #     return torch.nn.functional.nll_loss(i0, i1, weight=tensor([8.4784, 1.7658, 4.3228], dtype=torch.float32))
            #                                                        ~~~~~~ <--- HERE
            DecorateInfo(unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit", dtypes=(torch.float32,),),
        ),
    ),
    OpInfo(
        "nn.functional.gaussian_nll_loss",
        ref=_NOTHING,
        dtypes=all_types_and(torch.bfloat16),
        dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_gaussian_nll_loss,
        skips=(
            # JIT does not support variadic tensors.
            # RuntimeError: input->type()->kind() == TypeKind::OptionalType
            # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":270,
            # please report a bug to PyTorch.
            DecorateInfo(unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit", dtypes=(torch.float32,),),
        ),
    ),
    OpInfo(
        "nn.functional.hinge_embedding_loss",
        ref=_NOTHING,
        dtypes=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_hinge_embedding_loss,
    ),
    OpInfo(
        "nn.functional.huber_loss",
        ref=_NOTHING,
        dtypes=floating_types_and(torch.float16, torch.bfloat16),
        supports_out=False,
        supports_forward_ad=True,
        sample_inputs_func=sample_inputs_huber_loss,
        skips=(
            # JIT does not support variadic tensors.
            # RuntimeError: input->type()->kind() == TypeKind::OptionalType
            # INTERNAL ASSERT FAILED at "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":270,
            # please report a bug to PyTorch.
            DecorateInfo(unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit", dtypes=(torch.float32,),),
        )
    ),
    OpInfo(
        "nn.functional.poisson_nll_loss",
        ref=_NOTHING,
        dtypes=all_types_and(torch.bfloat16),
        dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_poisson_nll_loss,
    ),
    OpInfo(
        "argsort",
        dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16),
        dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16),
        sample_inputs_func=sample_inputs_argsort,
        supports_out=False,
        supports_autograd=False,
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                dtypes=(torch.float32,),
            ),
        ),
    ),
    OpInfo(
        "repeat_interleave",
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        sample_inputs_func=sample_inputs_repeat_interleave,
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                dtypes=(torch.float32, torch.complex64),
            ),
        ),
    ),
    OpInfo(
        "nn.functional.pairwise_distance",
        ref=lambda a, b, p=2.0, eps=1e-6, keepdim=False: (
            np.sum(np.abs(a - b + eps) ** p, axis=-1, keepdims=keepdim) ** (1 / p)
        ),
        sample_inputs_func=sample_inputs_pairwise_distance,
        dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
        supports_out=False,
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                dtypes=(torch.float32, torch.complex64),
            ),
        ),
    ),
    OpInfo(
        "nn.functional.pixel_shuffle",
        sample_inputs_func=sample_inputs_pixel_shuffle,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                dtypes=(torch.float32, torch.complex64),
            ),
        ),
    ),
    OpInfo(
        "nn.functional.pixel_unshuffle",
        sample_inputs_func=sample_inputs_pixel_unshuffle,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                dtypes=(torch.float32, torch.complex64),
            ),
        ),
    ),
    OpInfo(
        "nn.functional.kl_div",
        sample_inputs_func=sample_inputs_kl_div,
        dtypes=floating_types_and(torch.bfloat16, torch.int8, torch.int16, torch.int32, torch.int64),
        backward_dtypesIfCPU=floating_types_and(torch.int8, torch.int16, torch.int32, torch.int64),
        dtypesIfCUDA=floating_types_and(
            torch.float16, torch.bfloat16, torch.int8, torch.int16, torch.int32, torch.int64
        ),
        backward_dtypesIfCUDA=floating_types_and(torch.float16, torch.int8, torch.int16, torch.int32, torch.int64),
        supports_out=False,
        check_batched_grad=False,
        supports_forward_ad=True,
        skips=(
            # See https://github.com/pytorch/pytorch/issues/65466
            DecorateInfo(
                unittest.expectedFailure,
                "TestGradients",
                "test_fn_gradgrad",
            ),
            # (ROCm) Memory access fault by GPU node-4 (Agent handle: 0x5642a3aa7b60) on address 0x5642bab40000
            # CUDA memory exception as well
            DecorateInfo(unittest.skip("Skipped! CUDA/ROCm memory exception"), 'TestGradients', 'test_forward_mode_AD',
                         device_type='cuda', dtypes=[torch.float64, torch.complex128]),
        ),
    ),
    OpInfo(
        "diagflat",
        ref=lambda input, offset=0: np.diagflat(input, k=offset),
        sample_inputs_func=sample_inputs_diagflat,
        dtypes=all_types_and_complex_and(torch.bool),
        dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
    ),
    OpInfo(
        'scatter_reduce',
        dtypes=all_types_and(torch.float16, torch.bfloat16),
        sample_inputs_func=sample_inputs_scatter_reduce,
        supports_out=False,
        decorators=(onlyCPU,),
        skips=(
            DecorateInfo(unittest.skip("Skipped!"), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive',
                         active_if=IS_WINDOWS),
            DecorateInfo(unittest.skip("Skipped!"), 'TestNormalizeOperators', 'test_normalize_operator_exhaustive',
                         active_if=IS_WINDOWS),
        ),
    ),
]

# Common operator groupings
unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo)]
binary_ufuncs = [op for op in op_db if isinstance(op, BinaryUfuncInfo)]
spectral_funcs = [op for op in op_db if isinstance(op, SpectralFuncInfo)]
sparse_unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo) and op.supports_sparse]
sparse_csr_unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo) and op.supports_sparse_csr]
shape_funcs = [op for op in op_db if isinstance(op, ShapeFuncInfo)]
reduction_ops = [op for op in op_db if isinstance(op, ReductionOpInfo)]
reference_filtered_ops = [op for op in reduction_ops if op.ref not in (_NOTHING, None)]
reference_masked_ops = [op for op in reference_filtered_ops if op.name.startswith('_masked.')]

# TODO: review porting these to make_tensor
def index_variable(shape, max_indices, device=torch.device('cpu')):
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = torch.rand(*shape, dtype=torch.double, device=device).mul_(max_indices).floor_().long()
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


# TODO: move all tri/tril/triu testing to tensor creation op test suite and remove
#   these from here
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
                 .triu(offset).nonzero().to(dtype).transpose(0, 1),
            torch.triu_indices(row, col, offset, dtype=dtype, device=device))


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

# TODO: move into common_utils.py or the test suite(s) that use this
def unpack_variables(args):
    if isinstance(args, tuple):
        return tuple(unpack_variables(elem) for elem in args)
    else:
        return args


class dont_convert(tuple):
    pass


non_differentiable = collections.namedtuple('non_differentiable', ['tensor'])


# TODO: move into common_utils.py or the test suite(s) that use this
def create_input(call_args, requires_grad=True, non_contiguous=False, call_kwargs=None, dtype=torch.double, device=None):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)

    def map_arg(arg):
        def maybe_non_contig(tensor):
            return tensor if not non_contiguous else make_non_contiguous(tensor)

        def conjugate(tensor):
            return tensor.conj()

        if isinstance(arg, torch.Size) or isinstance(arg, dont_convert):
            return arg
        elif isinstance(arg, tuple) and len(arg) == 0:
            var = conjugate(torch.randn((), dtype=dtype, device=device))
            var.requires_grad = requires_grad
            return var
        elif isinstance(arg, tuple) and not isinstance(arg[0], torch.Tensor):
            return conjugate(maybe_non_contig(torch.randn(*arg, dtype=dtype, device=device))).requires_grad_(requires_grad)
        # double check casting
        elif isinstance(arg, non_differentiable):
            if isinstance(arg.tensor, torch.Tensor):
                if arg.tensor.dtype == torch.float:
                    return maybe_non_contig(arg.tensor.to(dtype=torch.double, device=device))
                if arg.tensor.dtype == torch.cfloat:
                    return conjugate(maybe_non_contig(arg.tensor.to(dtype=torch.cdouble, device=device)))
                return conjugate(maybe_non_contig(arg.tensor.to(device=device)))
            return conjugate(maybe_non_contig(arg.tensor.to(device=device)))
        elif isinstance(arg, torch.Tensor):
            if arg.dtype == torch.float:
                arg = arg.double()
            if arg.dtype == torch.cfloat:
                arg = arg.to(torch.cdouble)
            if arg.is_complex() != dtype.is_complex:
                raise RuntimeError("User provided tensor is real for a test that runs with complex dtype, ",
                                   "which is not supported for now")
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of v afterwards
            v = conjugate(maybe_non_contig(arg)).detach().to(device=device).clone()
            v.requires_grad = requires_grad and (v.is_floating_point() or v.is_complex())
            return v
        elif callable(arg):
            return map_arg(arg(dtype=dtype, device=device))
        else:
            return arg
    args_out = tuple(map_arg(arg) for arg in call_args)
    kwargs_out = {k: map_arg(v) for k, v in call_kwargs.items()} if call_kwargs else {}
    return args_out, kwargs_out
