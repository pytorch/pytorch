# mypy: ignore-errors

import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    skipCPUIfNoFFT,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_dtype import (
    _dispatch_dtypes,
    floating_and_complex_types,
    floating_and_complex_types_and,
    floating_types,
    get_all_dtypes,
)
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    is_iterable_of_tensors,
    noncontiguous_like,
    OPINFO_SAMPLE_INPUT_INDEX,
    TEST_WITH_ROCM,
    torch_to_numpy_dtype_dict,
    TrackedInputIter,
)
from torch.testing._internal.opinfo import utils
from torchgen.utils import dataclass_repr


# Reasonable testing sizes for dimensions
L = 20
M = 10
S = 5
XS = 3

# Unique value to distinguish default from anything else
_NOTHING = object()


# Extension of getattr to support qualified names
# e.g. _getattr_qual(torch, 'linalg.norm') -> torch.linalg.norm
def _getattr_qual(obj, name, default=_NOTHING):
    try:
        for path in name.split("."):
            obj = getattr(obj, path)
        return obj
    except AttributeError:
        if default is not _NOTHING:
            return default
        else:
            raise


class DecorateInfo:
    """Describes which test, or type of tests, should be wrapped in the given
    decorators when testing an operator. Any test that matches all provided
    arguments will be decorated. The decorators will only be applied if the
    active_if argument is True."""

    __slots__ = [
        "decorators",
        "cls_name",
        "test_name",
        "device_type",
        "dtypes",
        "active_if",
    ]

    def __init__(
        self,
        decorators,
        cls_name=None,
        test_name=None,
        *,
        device_type=None,
        dtypes=None,
        active_if=True,
    ):
        self.decorators = (
            list(decorators)
            if isinstance(decorators, collections.abc.Sequence)
            else [decorators]
        )
        self.cls_name = cls_name
        self.test_name = test_name
        self.device_type = device_type
        self.dtypes = dtypes
        self.active_if = active_if

        # Validate dtypes
        if self.dtypes is not None:
            for dtype in self.dtypes:
                assert isinstance(dtype, torch.dtype)

    def is_active(self, cls_name, test_name, device_type, dtype, param_kwargs):
        return (
            self.active_if
            and (self.cls_name is None or self.cls_name == cls_name)
            and (self.test_name is None or self.test_name == test_name)
            and (self.device_type is None or self.device_type == device_type)
            and (self.dtypes is None or dtype in self.dtypes)
            # Support callables over kwargs to determine if the decorator is active.
            and (
                self.active_if(param_kwargs)
                if isinstance(self.active_if, Callable)
                else self.active_if
            )
        )


# FIXME
# Note: historically the 'input' kwarg had to be a Tensor or TensorList, but we are trying
#   to support scalar inputs, too. Some tests still depend on 'input' being a Tensor
#   or TensorList, however.
class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "input",
        "args",
        "kwargs",
        "output_process_fn_grad",
        "broadcasts_input",
        "name",
    ]

    def __init__(
        self,
        input,
        *var_args,
        args=None,
        kwargs=None,
        output_process_fn_grad=None,
        broadcasts_input=None,
        name=None,
        **var_kwargs,
    ):
        # input is the first input to the op and is typically either a Tensor or TensorList (Sequence[Tensor]).
        # This follows the typical pattern where for Tensor inputs op(t, ...) = t.op(...).
        self.input = input

        # Allow calling either as SampleInput(input, args=args, kwargs=kwargs), or as
        # SampleInput(input, *args, **kwargs) but not to mix the two forms
        if args is not None or kwargs is not None:
            assert (
                not var_args and not var_kwargs
            ), """
A SampleInput can be constructed "naturally" with *args and **kwargs or by
explicitly setting the "args" and "kwargs" parameters, but the two
methods of construction cannot be mixed!"""
        elif len(var_args) or len(var_kwargs):
            assert (
                output_process_fn_grad is None
                and broadcasts_input is None
                and name is None
            ), """
A SampleInput constructed "naturally" with *args and **kwargs
cannot specify additional metadata in keyword arguments"""

        self.args = args if args is not None else var_args
        assert isinstance(self.args, tuple)
        self.kwargs = kwargs if kwargs is not None else var_kwargs
        assert isinstance(self.kwargs, dict)

        self.output_process_fn_grad = (
            output_process_fn_grad
            if output_process_fn_grad is not None
            else lambda x: x
        )
        self.name = name if name is not None else ""

        # Specifies if `self.input` is broadcasted or not,
        # given that the operator supports broadcasting.
        # This field is used to verify the behavior for inplace variant.
        #
        # If a SampleInput is marked with `broadcasts_input=True`,
        # it is verified that we get a `RuntimeError` with this sample,
        # and inplace variant. Also inplace grad{grad} tests are skipped,
        # for such inputs (as they will error out otherwise).
        self.broadcasts_input = (
            broadcasts_input if broadcasts_input is not None else False
        )

    def with_metadata(
        self, *, output_process_fn_grad=None, broadcasts_input=None, name=None
    ):
        if output_process_fn_grad is not None:
            self.output_process_fn_grad = output_process_fn_grad
        if broadcasts_input is not None:
            self.broadcasts_input = broadcasts_input
        if name is not None:
            self.name = name
        return self

    def _repr_helper(self, formatter):
        # Helper function to return the details of the SampleInput as `str`
        # It consolidates all the fields of SampleInput and allows,
        # formatting the fields like `input`, `args`, etc with `formatter`
        # callable to customize the representation.
        # Look at `summary` method for example.
        arguments = [
            f"input={formatter(self.input)}",
            f"args={formatter(self.args)}",
            f"kwargs={formatter(self.kwargs)}",
            f"broadcasts_input={self.broadcasts_input}",
            f"name={repr(self.name)}",
        ]

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
                shape = str(tuple(arg.shape))
                dtype = str(arg.dtype)
                device = str(arg.device)
                contiguity_suffix = ""
                # NB: sparse CSR tensors annoyingly return is_sparse=False
                is_sparse = arg.is_sparse or arg.layout == torch.sparse_csr
                if not is_sparse and not arg.is_contiguous():
                    contiguity_suffix = ", contiguous=False"
                return f'Tensor[size={shape}, device="{device}", dtype={dtype}{contiguity_suffix}]'
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
                with torch.no_grad():
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

        sample_tt_input, tt_args, tt_kwargs = (
            tt(self.input),
            tt(self.args),
            tt(self.kwargs),
        )

        # Note the transformed SampleInput assumes metadata like output_process_fn_grad is still valid!
        return SampleInput(
            sample_tt_input,
            args=tt_args,
            kwargs=tt_kwargs,
            output_process_fn_grad=self.output_process_fn_grad,
            broadcasts_input=self.broadcasts_input,
            name=self.name + "_transformed",
        )

    # Returns the NumPy version of the sample input object in the form of a tuple: (input, args, kwargs)
    # Converts tensors to ndarrays by calling .detach().cpu().numpy() on them
    # Converts dtypes by remapping them using torch_to_numpy_dtype_dict
    def numpy(self):
        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                if t.dtype is torch.bfloat16:
                    return t.detach().cpu().to(torch.float32).numpy()
                if t.dtype is torch.chalf:
                    return t.detach().cpu().to(torch.cfloat).numpy()
                return t.detach().cpu().numpy()
            elif isinstance(t, torch.dtype):
                return torch_to_numpy_dtype_dict[t]

            return t

        return self.transform(to_numpy)

    def noncontiguous(self):
        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                return noncontiguous_like(t)
            elif isinstance(t, torch.dtype):
                return t

            return t

        return self.transform(to_noncontiguous)


NumericsFilter = collections.namedtuple("NumericsFilter", ["condition", "safe_val"])


class ErrorInput:
    """
    A SampleInput that will cause the operation to throw an error plus information
    about the resulting error.
    """

    __slots__ = ["sample_input", "error_type", "error_regex"]

    def __init__(self, sample_input, *, error_type=RuntimeError, error_regex):
        self.sample_input = sample_input
        self.error_type = error_type
        self.error_regex = error_regex


class AliasInfo:
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
#   environment for efficiency and correctness. As such remember to set the
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
#   (https://pytorch.org/docs/main/generated/torch.linalg.slogdet.html).
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
# Sample inputs are designed to be used with many tests, some
#   that are very time consuming, so they should be a small
#   set with small tensors. An elaborated set of sample inputs
#   can be specified using the "reference_inputs_func" attribute.
#   The "reference inputs" for an operation are an extended
#   set of sample inputs that can more exhausively test an
#   operator. They are used by only a few tests that are careful
#   not to take too long to run. Adding reference inputs
#   is highly encouraged!
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
#   - that the operator throws the correct errors (if error_inputs is defined)
#   - that the operator produces the same results as a NumPy reference (if ref is defined)
#   - that the operator produces the same results as a NumPy reference on an extended
#       set of "reference inputs" (if both ref and reference_inputs_func are defined)
#       (NOTE: elementwise unary and elementwise binary OpInfos do this even if only
#         ref is defined, because they effectively autogenerate reference inputs)
#   - that the operator works on different CUDA devices
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
# Critically, as mentioned above, what's not necessarily tested is that the operator
#   works as expected. When implementing an OpInfo an engineer must still
#   typically write one or more tests validating the operator's behavior.
#   The exception to this is if reference testing is sufficient, or if
#   the operation belongs to an OpInfo subclass that has more exhaustive
#   operator testing. Elementwise unary and elementwise binary operators,
#   in particular, usually don't require additional testing beyond
#   writing an Opinfo.
#
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
#   usually sufficient testing (unless the operator is a unary or binary elementwise
#   operator). The OpInfo will only test the properties described in the
#   "WHAT'S TESTED" section. It DOES NOT necessarily verify that the operator is
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
#   is the get_supported_dtypes() function defined in utils.py. This
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
@dataclass
class OpInfo:
    """Operator information and helper functions for acquiring it."""

    # the string name of the function
    name: str

    # An optional reference function that accepts ndarrays (AKA "NumPy arrays").
    # If given, the op will be compared with its reference on each of its sample inputs.
    ref: Optional[Callable] = None

    # the following metadata describes the operator, its variants, and its aliases, if any

    # iterable of aliases, e.g. ("absolute",) for torch.abs
    aliases: Iterable = None

    # additional string to include in the test name
    # this is useful when an op needs multiple OpInfos,
    # like divide does, often because it's really several
    # different ops behind the scenes
    variant_test_name: str = ""

    # the function variant of the operation, populated as torch.<name> if None
    op: Callable = None

    # allows the method variant of this operation to be specified as follows:
    # - if _NOTHING (default), then the OpInfo attempts to discover the variant using its name
    # - if None, then the OpInfo explicitly specifies is has no associated method
    # - if a Callable, then that callable should be the method associated with this operation
    method_variant: Callable = _NOTHING

    # allows the inplace variant of this operation to be specified as follows:
    # - if _NOTHING (default), then the OpInfo attempts to discover the variant using its name
    # - if None, then the OpInfo explicitly specifies is has no associated inplace variant
    # - if a Callable, then that callable should be the inplace variant associated with this operation
    inplace_variant: Callable = _NOTHING

    # allows the operator variant of this operation to be specified as follows:
    # - if _NOTHING (default), then the OpInfo attempts to discover the variant using its name
    # - if None, then the OpInfo explicitly specifies is has no associated operator
    # - if a Callable, then that callable should be the operator associated with this operation
    operator_variant: Callable = _NOTHING

    # allows the inplace operator variant of this operation to be specified as follows:
    # - if _NOTHING (default), then the OpInfo attempts to discover the variant using its name
    # - if None, then the OpInfo explicitly specifies is has no associated inplace operator
    # - if a Callable, then that callable should be the inplace operator associated with this operation
    inplace_operator_variant: Callable = _NOTHING

    # the following metadata are test directives for skipping or modifying tests

    # information about which tests to skip
    skips: Tuple = ()

    # decorators to apply to generated tests
    decorators: Tuple = ()

    # the following are pointers to functions to generate certain classes of inputs

    # function to generate sample inputs with strided layouts
    sample_inputs_func: Callable = None

    # function to generate a more thorough set of samples inputs with strided layouts
    reference_inputs_func: Callable = None

    # function to generate inputs that will throw errors
    error_inputs_func: Callable = None

    # function to generate sparse (coo, csr, csc, bsr, bsc) inputs that will throw errors
    error_inputs_sparse_func: Callable = None

    # function to generate sample inputs with sparse coo layouts
    sample_inputs_sparse_coo_func: Callable = None

    # function to generate sample inputs with sparse csr layouts
    sample_inputs_sparse_csr_func: Callable = None

    # function to generate sample inputs with sparse csc layouts
    sample_inputs_sparse_csc_func: Callable = None

    # function to generate sample inputs with sparse bsr layouts
    sample_inputs_sparse_bsr_func: Callable = None

    # function to generate sample inputs with sparse bsc layouts
    sample_inputs_sparse_bsc_func: Callable = None

    # the following metadata relates to dtype support and is tested for correctness in test_ops.py

    # dtypes this function works with on the CPU,
    # inherited by other device types that don't specify their own dtypes
    dtypes: _dispatch_dtypes = None

    # the following dtypesIf... options override the dtypes value on their respective device types

    # dtypes this function is expected to work with on CUDA
    dtypesIfCUDA: _dispatch_dtypes = None

    # dtypes this function is expected to work with on ROCM
    dtypesIfROCM: _dispatch_dtypes = None

    dtypesIfHpu: _dispatch_dtypes = None

    # dtypes this function is expected to work with on XPU
    dtypesIfXPU: _dispatch_dtypes = None

    # backward dtypes this function is expected to work with
    backward_dtypes: _dispatch_dtypes = None

    # backward dtypes this function is expected to work with on CUDA
    backward_dtypesIfCUDA: _dispatch_dtypes = None

    # backward dtypes this function is expected to work with on ROCM
    backward_dtypesIfROCM: _dispatch_dtypes = None

    backward_dtypesIfHpu: _dispatch_dtypes = None

    # the following metadata describes the operators out= support

    # whether the op supports the out kwarg
    # defaults to True, if the op does not allow the out kwarg or
    # supports it incorrectly then test_out in test_ops.py should fail
    supports_out: bool = True

    # the following metadata relates to autograd support
    # whether the operation supports backward mode AD
    # if true, gradient correctness is tested in test_ops.py
    # using the op's sample inputs
    supports_autograd: bool = True

    # whether the op supports second order gradients
    # if true, gradgrad correctness is tested in test_ops.py
    # defaults to support_autograd's value
    # TODO: rename this to supports_bwgrad_bwgrad to be consistent with below
    supports_gradgrad: bool = None

    # whether the ops supports second order gradients via
    # forward-over-reverse. If True, forward-over-reverse gradgrad correctness
    # is tested. If False, test that forward grad is not implemented.
    # Defaults to False.
    supports_fwgrad_bwgrad: bool = False

    # whether the operation supports inplace autograd
    # if true, tested in test_ops.py
    # defaults to supports_autograd's value
    supports_inplace_autograd: bool = None

    # Whether the operation support forward mode AD
    # If the value is True, we check that the gradients are correct
    # If the value is False, we test that forward grad is not implemented
    supports_forward_ad: bool = False

    # Whether the operation has a varargs variant
    # (e.g. functions like ones, zeros, methods like view, permute)
    supports_varargs: bool = False

    # Whether the forward operation avoids materializing COW tensor inputs
    supports_cow_input_no_materialize_forward: bool = True

    # Whether the backward operation avoids materializing COW tensor inputs
    supports_cow_input_no_materialize_backward: bool = True

    # Whether to skip the backward part of the COW tensor input test
    skip_cow_input_backward: bool = False

    # If `supports_cow_input_no_materialize_forward == True`, this list contains
    # the arg indices or kwarg names of inputs that are expected to materialize
    allow_cow_input_materialize_forward: List[Union[int, str]] = None

    # If `supports_cow_input_no_materialize_backward == True`, this list contains
    # the arg indices or kwarg names of inputs that are expected to materialize
    allow_cow_input_materialize_backward: List[Union[int, str]] = None

    # wrapper function for gradcheck
    gradcheck_wrapper: Callable = lambda op, *args, **kwargs: op(*args, **kwargs)

    # whether to check batched grad when doing gradcheck
    # defaults to support_autograd's value
    check_batched_grad: bool = None

    # whether to check batched grad grad when doing gradgradcheck
    # default's to support_gradgrad's value
    check_batched_gradgrad: bool = None

    # whether to check batched forward grad when doing gradcheck
    # defaults to the value of `supports_forward_ad`
    check_batched_forward_grad: bool = None

    # whether to check batched forward grad when doing gradcheck
    # defaults to the value of `check_batched_forward_grad`
    check_inplace_batched_forward_grad: bool = None

    # tolerance for nondeterminism while performing gradcheck
    gradcheck_nondet_tol: float = 0.0

    # Whether to use the fast implmentation for gradcheck/gradgradcheck.
    # When set to None, defers to the default value provided by the wrapper
    # function around gradcheck (testing._internal.common_utils.gradcheck)
    gradcheck_fast_mode: bool = None

    # the following metadata relates to JIT support and is tested for correctness in test_ops.py

    # name of the corresponding aten:: operator
    aten_name: str = None

    # if this is a composite implicit autograd op, the decomposed op
    decomp_aten_name: Optional[str] = None

    # name of the corresponding aten:: operator for backwards
    aten_backward_name: Optional[str] = None

    # if a op's aten::node is expected to be symbolically autodiffed
    assert_autodiffed: bool = False

    # a list of strings with node names that are expected to be in a
    # DifferentiableGraph when autodiffed. Ex: ['aten::add', 'aten::mm'],
    # default is populated to be ['aten::(name of Python operator)']
    autodiff_nonfusible_nodes: List[str] = None

    # a list of strings with node names that are expected to be in FusionGroups
    # inside of DifferentiableGraphs when this operation is autodiffed.
    # Ex: ['aten::add', 'aten::mm'], defaults to an empty list
    # Note: currently no ops use fusible nodes
    autodiff_fusible_nodes: List[str] = None

    # the following metadata relates to sparse support and is used in test_sparse.py

    # whether the op supports sparse coo inputs, defaults to False
    # TODO: rename supports_sparse to supports_sparse_coo
    supports_sparse: bool = None

    # only run tracing tests
    supports_scripting: bool = True

    # if the operator can be traced
    supports_tracing: bool = True

    # the following metadata relates to sparse compressed support and
    # is used in test_sparse_csr.py and test_sparse.py

    # whether the op supports sparse csr inputs, defaults to False
    supports_sparse_csr: bool = None
    # whether the op supports sparse csc inputs, defaults to False
    supports_sparse_csc: bool = None
    # whether the op supports sparse bsr inputs, defaults to False
    supports_sparse_bsr: bool = None
    # whether the op supports sparse bsc inputs, defaults to False
    supports_sparse_bsc: bool = None
    # whether the op supports nested jagged inputs, defaults to False
    supports_njt: bool = None

    # whether the op promotes integer inputs to float
    promotes_int_to_float: bool = False

    # the following metadata relates to complex support and is checked in test_ops.py

    test_conjugated_samples: bool = True

    test_neg_view: bool = True

    # assert that jit shape analysis fully propagates shape
    assert_jit_shape_analysis: bool = False

    # the following metadata relates to ExpandedWeights support and is checked in test_expanded_weights.py

    supports_expanded_weight: bool = False

    is_factory_function: bool = False

    skip_correctness_check_compile_vs_eager: bool = False

    def __post_init__(self):
        self._original_opinfo_args = asdict(self).copy()

        assert self.dtypes is not None, f"OpInfo for {self.name} has no dtypes!"

        dtypes_args = (
            self.dtypes,
            self.dtypesIfCUDA,
            self.dtypesIfROCM,
            self.dtypesIfXPU,
        )

        # Validates the dtypes are generated from the dispatch-related functions
        for dtype_list in dtypes_args:
            assert isinstance(dtype_list, (_dispatch_dtypes, type(None)))

        if self.aten_name is None:
            self.aten_name = self.name

        # Attribute to verify dynamic_dtypes are used.
        self.dynamic_dtypes = any(
            isinstance(dtypes, utils._dynamic_dispatch_dtypes) for dtypes in dtypes_args
        )

        if self.dynamic_dtypes:
            # Make sure `dtyesIfCUDA` is dynamic, if dynamic dispatch is used for CPU
            # This is because, below we set dtypesIfCUDA to dtypes if they are None.
            assert isinstance(self.dtypesIfCUDA, utils._dynamic_dispatch_dtypes), (
                f"To use dynamic dypes for operator {self.name}, "
                "acquire the dtypes dynamically for argument `dtypesIfCUDA`."
                "This is to ensure that CUDA dtypes are acquired correctly as they"
                "differ from CPU dtypes occasionally"
            )

        self.dtypes = set(self.dtypes)

        # NOTE: backward dtypes must be acquired before forward dtypes
        #   since they fallback to explicit (not implicit!) specifications of
        #   forward dtypes
        self.backward_dtypesIfROCM = (
            set(self.backward_dtypesIfROCM)
            if self.backward_dtypesIfROCM is not None
            else (
                self.backward_dtypesIfCUDA
                if self.backward_dtypesIfCUDA is not None
                else self.backward_dtypes
                if self.backward_dtypes is not None
                else self.dtypesIfROCM
                if self.dtypesIfROCM is not None
                else self.dtypesIfCUDA
                if self.dtypesIfCUDA is not None
                else self.dtypes
            )
        )
        self.backward_dtypesIfCUDA = (
            set(self.backward_dtypesIfCUDA)
            if self.backward_dtypesIfCUDA is not None
            else (
                self.backward_dtypes
                if self.backward_dtypes is not None
                else self.dtypesIfCUDA
                if self.dtypesIfCUDA is not None
                else self.dtypes
            )
        )
        self.backward_dtypesIfHpu = (
            set(self.backward_dtypesIfHpu)
            if self.backward_dtypesIfHpu is not None
            else (
                self.backward_dtypes
                if self.backward_dtypes is not None
                else self.dtypes
            )
        )

        self.backward_dtypes = (
            set(self.backward_dtypes)
            if self.backward_dtypes is not None
            else self.dtypes
        )

        self.dtypesIfCUDA = (
            set(self.dtypesIfCUDA) if self.dtypesIfCUDA is not None else self.dtypes
        )
        self.dtypesIfROCM = (
            set(self.dtypesIfROCM)
            if self.dtypesIfROCM is not None
            else self.dtypesIfCUDA
        )
        self.dtypesIfXPU = (
            set(self.dtypesIfXPU) if self.dtypesIfXPU is not None else self.dtypesIfCUDA
        )

        self.dtypesIfHpu = (
            set(self.dtypesIfHpu) if self.dtypesIfHpu is not None else self.dtypes
        )

        # NOTE: if the op is unspecified it is assumed to be under the torch namespace
        if not self.op:
            self.op = _getattr_qual(torch, self.name)

        if self.method_variant is _NOTHING:
            self.method_variant = getattr(torch.Tensor, self.name, None)

        # attributes like real, imag are not callable
        if not callable(self.method_variant):
            self.method_variant = None

        if self.inplace_variant is _NOTHING:
            inplace_name = self.name + "_"
            self.inplace_variant = getattr(torch.Tensor, inplace_name, None)

        if self.operator_variant is _NOTHING:
            self.operator_variant = getattr(operator, self.name, None)

        if self.inplace_operator_variant is _NOTHING:
            # Note: operator.i<op> will use operator.<op> and assign the result to the lhs when no
            # __i<op>__ method is found. This results in the appearance of an inplace operator variant which
            # does not have the correct inplace behavior. To avoid this, we guard automatic detection of the inplace
            # operator with a check that an inplace variant exists.
            if self.inplace_variant is not None:
                inplace_operator_name = "i" + self.name
                self.inplace_operator_variant = getattr(
                    operator, inplace_operator_name, None
                )
            else:
                self.inplace_operator_variant = None

        self.decorators = (*self.decorators, *self.skips)

        # Specifying sample inputs function without specifying the
        # corresponding layout support implies the layout support:
        if self.supports_sparse is None:
            self.supports_sparse = self.sample_inputs_sparse_coo_func is not None
        if self.sample_inputs_sparse_coo_func is None:
            self.sample_inputs_sparse_coo_func = self._sample_inputs_unspecified

        if self.supports_sparse_csr is None:
            self.supports_sparse_csr = self.sample_inputs_sparse_csr_func is not None
        if self.sample_inputs_sparse_csr_func is None:
            self.sample_inputs_sparse_csr_func = self._sample_inputs_unspecified

        if self.supports_sparse_csc is None:
            self.supports_sparse_csc = self.sample_inputs_sparse_csc_func is not None
        if self.sample_inputs_sparse_csc_func is None:
            self.sample_inputs_sparse_csc_func = self._sample_inputs_unspecified

        if self.supports_sparse_bsr is None:
            self.supports_sparse_bsr = self.sample_inputs_sparse_bsr_func is not None
        if self.sample_inputs_sparse_bsr_func is None:
            self.sample_inputs_sparse_bsr_func = self._sample_inputs_unspecified

        if self.supports_sparse_bsc is None:
            self.supports_sparse_bsc = self.sample_inputs_sparse_bsc_func is not None
        if self.sample_inputs_sparse_bsc_func is None:
            self.sample_inputs_sparse_bsc_func = self._sample_inputs_unspecified

        if self.supports_njt is None:
            self.supports_njt = False

        # We run the sampling functions without tracking the gradiends of the creation of inputs
        self.sample_inputs_func = torch.no_grad()(self.sample_inputs_func)
        self.sample_inputs_sparse_coo_func = torch.no_grad()(
            self.sample_inputs_sparse_coo_func
        )
        self.sample_inputs_sparse_csr_func = torch.no_grad()(
            self.sample_inputs_sparse_csr_func
        )
        self.sample_inputs_sparse_csc_func = torch.no_grad()(
            self.sample_inputs_sparse_csc_func
        )
        self.sample_inputs_sparse_bsr_func = torch.no_grad()(
            self.sample_inputs_sparse_bsr_func
        )
        self.sample_inputs_sparse_bsc_func = torch.no_grad()(
            self.sample_inputs_sparse_bsc_func
        )
        if self.reference_inputs_func is not None:
            self.reference_inputs_func = torch.no_grad()(self.reference_inputs_func)

        if not self.autodiff_fusible_nodes:
            self.autodiff_fusible_nodes = []

        if self.autodiff_nonfusible_nodes is None:
            self.autodiff_nonfusible_nodes = ["aten::" + self.name]

        # Autograd support

        # Autograd flags that depend on backward AD only
        # - If setting has been explicitly set, raise error if inconsistent
        if self.supports_gradgrad is None:
            self.supports_gradgrad = self.supports_autograd
        else:
            assert not (self.supports_gradgrad and not self.supports_autograd), (
                "supports_gradgrad refines the part of autograd is supported, so it should "
                "not be set if supports_autograd is False"
            )
        if self.check_batched_grad is None:
            self.check_batched_grad = self.supports_autograd or self.supports_forward_ad
        else:
            assert not (
                self.check_batched_grad
                and not (self.supports_autograd or self.supports_forward_ad)
            ), (
                "check_batched_grad refines the part of autograd that will be checked (by gradcheck), so "
                "it should not be set if supports_autograd is False"
            )
        if self.check_batched_gradgrad is None:
            self.check_batched_gradgrad = self.supports_gradgrad
        else:
            assert not (self.check_batched_gradgrad and not self.supports_gradgrad), (
                "check_batched_gradgrad refines the part of autograd that will be checked (by "
                "gradgradcheck), so it should not be set if either supports_gradgrad or supports_autograd "
                "is False."
            )
        if self.check_batched_forward_grad is None:
            self.check_batched_forward_grad = self.supports_forward_ad
        else:
            assert not (
                self.check_batched_forward_grad and not self.supports_forward_ad
            ), (
                "check_batched_forward_grad should only be used when supports_forward_ad "
                "is True. It is used to disable the test in the specific cases "
                "where the op supports forward ad but fails to compute "
                "batched forward grad."
            )

        if self.check_inplace_batched_forward_grad is None:
            self.check_inplace_batched_forward_grad = self.check_batched_forward_grad
        else:
            assert not (
                self.check_inplace_batched_forward_grad
                and not self.check_batched_forward_grad
            ), (
                "check_batched_forward_grad should only be used when check_batched_forward_grad "
                "is True. It is used to disable the test in the specific cases "
                "where the op supports batched forward grad but fails to compute batched forward "
                "grad for the inplace variant of the op."
            )

        assert not (self.supports_fwgrad_bwgrad and not self.supports_autograd), (
            "supports_fwgrad_bwgrad enables forward-over-backward gradgrad checks and should only be "
            "True if backward ad is also checked, i.e., supports_forward_ad should be True.",
            self.name,
        )

        # Autograd flags that depend on both forward AD and backward AD
        if self.supports_inplace_autograd is None:
            self.supports_inplace_autograd = (
                self.supports_autograd or self.supports_forward_ad
            )
        else:
            assert not (
                self.supports_inplace_autograd
                and not self.supports_autograd
                and not self.supports_forward_ad
            ), (
                "supports_inplace_autograd refines the part of autograd that is supported, so "
                "it should not be set if both supports_autograd and supports_forward_ad are False"
            )

        if self.aliases is not None:
            self.aliases = tuple(AliasInfo(a) for a in self.aliases)  # type: ignore[assignment]
        else:
            self.aliases = ()

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    def __str__(self):
        return dataclass_repr(self)

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

    def get_operator(self):
        """Returns operator variant of the operator, e.g. operator.neg
        Returns None if the operator has no operator variant.
        """
        return self.operator_variant

    def get_inplace_operator(self):
        """Returns the inplace operator variant of the operator, e.g operator.iadd
        Returns None if the operator has no inplace operator variant"""
        return self.inplace_operator_variant

    def conjugate_sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs but with the tensor input or first
        tensor in a sequence input conjugated.
        """

        set_seed = kwargs.pop("set_seed", True)
        samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)
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

        return TrackedInputIter(
            iter(conj_samples),
            "conjugate sample input",
            set_seed=set_seed,
            restrict_to_index=OPINFO_SAMPLE_INPUT_INDEX,
        )

    def sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
        """
        Returns an iterable of SampleInputs.

        These samples should be sufficient to test the function works correctly
        with autograd, TorchScript, etc.
        """
        set_seed = kwargs.pop("set_seed", True)
        samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)

        if kwargs.get("include_conjugated_inputs", False):
            conj_samples = self.conjugate_sample_inputs(
                device, dtype, requires_grad, **kwargs
            )
            samples_list = list(samples)
            samples_list.extend(conj_samples)
            samples = tuple(samples_list)

        return TrackedInputIter(
            iter(samples),
            "sample input",
            set_seed=set_seed,
            restrict_to_index=OPINFO_SAMPLE_INPUT_INDEX,
        )

    def reference_inputs(self, device, dtype, requires_grad=False, **kwargs):
        """
        Returns an iterable of SampleInputs.

        Distinct from sample_inputs() above because this returns an expanded set
        of inputs when reference_inputs_func is defined. If undefined this returns
        the sample inputs.
        """
        set_seed = kwargs.pop("set_seed", True)
        if self.reference_inputs_func is None:
            samples = self.sample_inputs_func(
                self, device, dtype, requires_grad, **kwargs
            )
            return TrackedInputIter(
                iter(samples),
                "reference input",
                set_seed=set_seed,
                restrict_to_index=OPINFO_SAMPLE_INPUT_INDEX,
            )

        if kwargs.get("include_conjugated_inputs", False):
            raise NotImplementedError

        references = self.reference_inputs_func(
            self, device, dtype, requires_grad, **kwargs
        )
        return TrackedInputIter(
            iter(references),
            "reference input",
            set_seed=set_seed,
            restrict_to_index=OPINFO_SAMPLE_INPUT_INDEX,
        )

    def error_inputs(self, device, **kwargs):
        """
        Returns an iterable of ErrorInputs.
        """
        set_seed = kwargs.pop("set_seed", True)
        errs = self.error_inputs_func(self, device, **kwargs)
        return TrackedInputIter(
            iter(errs),
            "error input",
            callback=lambda e: e.sample_input,
            set_seed=set_seed,
            restrict_to_index=OPINFO_SAMPLE_INPUT_INDEX,
        )

    def error_inputs_sparse(self, device, layout, **kwargs):
        """
        Returns an iterable of ErrorInputs that contain sparse sample
        inputs with a specified layout.
        """
        if not self.supports_sparse_layout(layout):
            raise unittest.SkipTest("unsupported sparse layout")
        return self.error_inputs_sparse_func(self, device, layout, **kwargs)

    def supports_sparse_layout(self, layout):
        """Return True if OpInfo supports the specified sparse layout."""
        layout_name = str(layout).split(".")[-1]
        # map torch.sparse_coo to OpInfo.supports_sparse:
        layout_name = layout_name.replace("_coo", "")
        return getattr(self, f"supports_{layout_name}")

    def sample_inputs_sparse(
        self, layout, device, dtype, requires_grad=False, **kwargs
    ):
        """Returns an iterable of SampleInputs that contain inputs with a
        specified sparse layout.
        """
        layout_name = str(layout).split(".")[-1]
        sample_inputs_mth = getattr(self, "sample_inputs_" + layout_name)

        def non_empty_sampler(op, generator):
            found_sample = False
            for sample in generator:
                found_sample = True
                yield sample
            if not found_sample:
                raise unittest.SkipTest("NO SAMPLES!")

        return non_empty_sampler(
            self,
            sample_inputs_mth(device, dtype, requires_grad=requires_grad, **kwargs),
        )

    def _sample_inputs_unspecified(self, *args, **kwargs):
        """Raises an NotImplemented exception in a OpInfo instance creation
        that specifies supports_sparse(|_csr|_csc|_bsr|_bsc)=True
        without specifying the corresponding sample function as
        sample_inputs_sparse_(coo|csr|csc|bsr|bsc)_func.

        To avoid this, either define the corresponding sample function,
        or re-map unsupported samples to error inputs in an appropiate

          opinfo/definitions/sparse.py:_validate_sample_input_sparse_<op>

        function.
        """
        raise NotImplementedError("no sample function specified")

    def sample_inputs_sparse_coo(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs that contain inputs with sparse
        coo layout.
        """
        return self.sample_inputs_sparse_coo_func(
            self, device, dtype, requires_grad, **kwargs
        )

    def sample_inputs_sparse_csr(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs that contain inputs with sparse
        csr layout.
        """
        return self.sample_inputs_sparse_csr_func(
            self, device, dtype, requires_grad, **kwargs
        )

    def sample_inputs_sparse_csc(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs that contain inputs with sparse
        csc layout.
        """
        return self.sample_inputs_sparse_csc_func(
            self, device, dtype, requires_grad, **kwargs
        )

    def sample_inputs_sparse_bsr(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs that contain inputs with sparse
        bsr layout.
        """
        return self.sample_inputs_sparse_bsr_func(
            self, device, dtype, requires_grad, **kwargs
        )

    def sample_inputs_sparse_bsc(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs that contain inputs with sparse
        bsc layout.
        """
        return self.sample_inputs_sparse_bsc_func(
            self, device, dtype, requires_grad, **kwargs
        )

    def get_decorators(self, test_class, test_name, device, dtype, param_kwargs):
        """Returns the decorators targeting the given test."""
        result = []
        for decorator in self.decorators:
            if isinstance(decorator, DecorateInfo):
                if decorator.is_active(
                    test_class, test_name, device, dtype, param_kwargs
                ):
                    result.extend(decorator.decorators)
            else:
                result.append(decorator)
        return result

    def supported_dtypes(self, device_type):
        if device_type == "privateuse1":
            device_type = torch._C._get_privateuse1_backend_name()
        device_type = torch.device(device_type).type
        if device_type == "cuda":
            return self.dtypesIfROCM if TEST_WITH_ROCM else self.dtypesIfCUDA
        if device_type == "xpu":
            return self.dtypesIfXPU
        if device_type == "hpu":
            return self.dtypesIfHpu
        return self.dtypes

    def supported_backward_dtypes(self, device_type):
        if not self.supports_autograd:
            return set()

        if device_type == "privateuse1":
            device_type = torch._C._get_privateuse1_backend_name()
        device_type = torch.device(device_type).type
        backward_dtypes = None
        if device_type == "cuda":
            backward_dtypes = (
                self.backward_dtypesIfROCM
                if TEST_WITH_ROCM
                else self.backward_dtypesIfCUDA
            )
        elif device_type == "hpu":
            backward_dtypes = self.backward_dtypesIfHpu
        else:
            backward_dtypes = self.backward_dtypes

        allowed_backward_dtypes = floating_and_complex_types_and(
            torch.bfloat16, torch.float16, torch.complex32
        )
        return set(allowed_backward_dtypes).intersection(backward_dtypes)

    def supports_dtype(self, dtype, device_type) -> bool:
        return dtype in self.supported_dtypes(device_type)

    @property
    def full_name(self):
        """Returns a full name that helps to uniquely identify this OpInfo."""
        variant = "." + self.variant_test_name if self.variant_test_name else ""
        # example: "normal.in_place" where "normal" is the name and "in_place" is the variant
        return f"{self.name}{variant}"

    @property
    def formatted_name(self):
        """Returns a formatted full name for this OpInfo that can be used in test names."""
        return self.full_name.replace(".", "_")


def _generate_reduction_inputs(device, dtype, requires_grad, **kwargs):
    """Generates input tensors for testing reduction operators"""
    yield make_tensor([], dtype=dtype, device=device, requires_grad=requires_grad)
    yield make_tensor([2], dtype=dtype, device=device, requires_grad=requires_grad)
    yield make_tensor([3, 5], dtype=dtype, device=device, requires_grad=requires_grad)
    yield make_tensor(
        [3, 2, 1, 2], dtype=dtype, device=device, requires_grad=requires_grad
    )


def _generate_reduction_kwargs(ndim, supports_multiple_dims=True):
    """Generates a subset of all valid dim and keepdim kwargs given ndim that
    is appropriate for testing reduction operators.
    """

    # Test default dim and keepdim
    yield {}

    # Test reducing inner and outer most dimensions
    yield {"dim": 0, "keepdim": True}
    yield {"dim": -1, "keepdim": False}

    # Test reducing middle dimension
    if ndim > 2:
        yield {"dim": ndim // 2, "keepdim": True}

    if supports_multiple_dims:
        # Test reducing all dimensions
        yield {"dim": tuple(range(ndim)), "keepdim": False}

        # Test reducing both first and last dimensions
        if ndim > 1:
            yield {"dim": (0, -1), "keepdim": True}

        # Test reducing every other dimension starting with the second
        if ndim > 3:
            yield {"dim": tuple(range(1, ndim, 2)), "keepdim": False}


def sample_inputs_reduction(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for reduction operators."""

    # TODO(@heitorschueroff) Once all reduction operators are using
    # ReductionOpInfo use op_info.supports_multiple_dims directly.
    supports_multiple_dims: bool = kwargs.get("supports_multiple_dims", True)

    # TODO(@heitorschueroff) Once all reduction operators are using ReductionOpInfo
    # use op_info.generate_args_kwargs directly.
    generate_args_kwargs = kwargs.get(
        "generate_args_kwargs", lambda *args, **kwargs: (yield (), {})
    )

    for t in _generate_reduction_inputs(device, dtype, requires_grad):
        for reduction_kwargs in _generate_reduction_kwargs(
            t.ndim, supports_multiple_dims
        ):
            for args, kwargs in generate_args_kwargs(t, **reduction_kwargs):
                kwargs.update(reduction_kwargs)
                yield SampleInput(
                    t.detach().requires_grad_(requires_grad), args=args, kwargs=kwargs
                )


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
    reduction operators, this should be documented by xfailing the failing
    tests rather than adding optional parameters to ReductionOpInfo.

    NOTE
    The API for reduction operators has not yet been finalized and some
    requirements may change.

    See tests in test/test_reductions.py
    """

    def __init__(
        self,
        name,
        *,
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
        # Casts complex results to real (e.g. linalg.norm or torch.var)
        complex_to_real: bool = False,
        # ReductionOpInfo tests generate their own input, dim and keepdim
        # arguments and call this function to generate tuples of extra args and
        # kwargs to use when calling the op. This is required for operators that
        # have other required parameters besides the input tensor.
        generate_args_kwargs: Callable = lambda t, dim=None, keepdim=False: (
            yield (),
            {},
        ),
        # Options from the OpInfo base class
        **kwargs,
    ):
        self._original_reduction_args = locals().copy()
        assert nan_policy in (None, "propagate", "omit")

        # These are mutually exclusive options
        assert not (result_dtype and promotes_int_to_float)
        assert not (result_dtype and promotes_int_to_int64)
        assert not (result_dtype and complex_to_real)
        assert not (promotes_int_to_float and promotes_int_to_int64)

        # Default sample_inputs_func for ReductionOpInfo which augments sample
        # inputs from sample_inputs_reduction with the args and kwargs from
        # generate_args_kwargs. This is only used if sample_inputs_func is None.
        def sample_inputs_func(*args, **kwargs):
            kwargs["supports_multiple_dims"] = supports_multiple_dims
            kwargs["generate_args_kwargs"] = generate_args_kwargs
            yield from sample_inputs_reduction(*args, **kwargs)

        # Override OpInfo defaults and call base class __init__
        kwargs.setdefault("inplace_variant", None)
        kwargs.setdefault("sample_inputs_func", sample_inputs_func)
        super().__init__(name, promotes_int_to_float=promotes_int_to_float, **kwargs)

        self.identity = identity
        self.nan_policy = nan_policy
        self.supports_multiple_dims = supports_multiple_dims
        self.promotes_int_to_int64 = promotes_int_to_int64
        self.complex_to_real = complex_to_real
        self.result_dtype = result_dtype
        self.generate_args_kwargs = generate_args_kwargs


# The base reference input generation for elementwise binary operations
def _reference_inputs_elementwise_binary(
    op, device, dtype, requires_grad, exclude_zero, **kwargs
):
    yield from op.sample_inputs_func(op, device, dtype, requires_grad, **kwargs)
    yield from generate_elementwise_binary_tensors(
        op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    if dtype is not torch.bool:
        yield from generate_elementwise_binary_small_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad
        )
    if dtype not in (torch.bool, torch.uint8, torch.int8):
        yield from generate_elementwise_binary_large_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad
        )
    yield from generate_elementwise_binary_broadcasting_tensors(
        op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    yield from generate_elementwise_binary_with_scalar_samples(
        op, device=device, dtype=dtype, requires_grad=requires_grad
    )

    yield from generate_elementwise_binary_with_scalar_and_type_promotion_samples(
        op, device=device, dtype=dtype, requires_grad=requires_grad
    )

    if dtype.is_floating_point or dtype.is_complex:
        yield from generate_elementwise_binary_extremal_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad
        )


# Note that these references inputs use scalars for the SampleInput.input value,
#   and many tests require SampleInput.input be a tensor or a list of tensors
def reference_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs):
    if hasattr(op, "rhs_make_tensor_kwargs"):
        exclude_zero = op.rhs_make_tensor_kwargs.get("exclude_zero", False)

    gen = partial(
        _reference_inputs_elementwise_binary,
        op,
        device,
        dtype,
        requires_grad,
        exclude_zero,
        **kwargs,
    )

    # yields "normal" samples
    yield from gen()

    # yields noncontiguous samples
    for sample in gen():
        yield sample.noncontiguous()

    yield from generate_elementwise_binary_noncontiguous_tensors(
        op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )

    yield from generate_elementwise_binary_arbitrarily_strided_tensors(
        op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )


# A functional that extends an elementwise binary operator's bespoke error inputs
#   with generic error inputs for the class of elementwise binary operations
def make_error_inputs_elementwise_binary(error_inputs_func):
    def error_inputs_func_wrapper(op, device, **kwargs):
        if error_inputs_func is not None:
            yield from error_inputs_func(op, device, **kwargs)

        if not op.supports_rhs_python_scalar:
            si = SampleInput(torch.tensor((1, 2, 3), device=device), args=(2,))
            yield ErrorInput(si, error_type=Exception, error_regex="")

        if not op.supports_one_python_scalar:
            si = SampleInput(2, args=(torch.tensor((1, 2, 3), device=device),))
            yield ErrorInput(si, error_type=Exception, error_regex="")

        if (
            not kwargs.get("skip_two_python_scalars", False)
            and not op.supports_two_python_scalars
        ):
            si = SampleInput(2, args=(3,))
            yield ErrorInput(si, error_type=Exception, error_regex="")

    return error_inputs_func_wrapper


# The following functions and classes are for testing elementwise binary operators.


# Returns a generator of pairs of contiguous tensors on the requested device
#   and with the requested dtype.
#
# This function is intended to test the non-vectorized and vectorized code
#   paths of elementwise binary functions, as well as their handling of odd tensor
#   sizes (like zero-dim tensors and tensors with zero elements).
#
# Each iterable will include an a tensor with no elements,
#   zero dim (scalar) tensors, small 1D tensors, a medium 1D tensor, and
#   a large 2D tensor.
def generate_elementwise_binary_tensors(
    op, *, device, dtype, requires_grad=False, exclude_zero=False
):
    shapes = (
        # tensors with no elements
        (0,),
        (1, 0, 3),
        # zero dim (scalar) tensor
        (),
        # small 1D tensor
        (20,),
        # medium 1D tensor
        (812,),
        # large 2D tensor
        (1029, 917),
    )

    make_arg = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    for shape in shapes:
        lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
        yield SampleInput(lhs, args=(rhs,))


def generate_elementwise_binary_arbitrarily_strided_tensors(
    op, *, device, dtype, requires_grad=False, exclude_zero=False
):
    # shape, strides, offset
    strided_cases = (
        ((5, 6, 2), (1, 1, 7), 2),
        ((5, 5, 4), (1, 1, 7), 2),
        ((5, 5, 2), (4, 5, 7), 3),
        ((5, 5, 2), (5, 5, 7), 3),
        ((5, 5, 2), (5, 5, 5), 3),
        ((9, 5, 2), (0, 1, 7), 3),
    )

    make_arg = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    for shape, strides, offset in strided_cases:
        a = make_arg(
            500,
        ).as_strided(shape, strides, offset)
        b = make_arg(shape)
        yield SampleInput(a, args=(b,))


# Returns a generator of pairs of contiguous tensors on the requested device and with
#   the requested dtype.
#
# Unlike the previous function, the values in these tensors are specified manually.
def generate_elementwise_binary_small_value_tensors(
    op, *, device, dtype, requires_grad=False, exclude_zero=None
):
    if exclude_zero is None:
        if hasattr(op, "rhs_make_tensor_kwargs"):
            exclude_zero = op.rhs_make_tensor_kwargs.get("exclude_zero", False)

    # defines interesting values
    _unsigned_int_vals = (0, 1, 55, 127, 128, 190, 210, 220, 254)
    _int_vals = (0, -1, 1, -55, 55, -127, 127, -128)
    _float_vals = (
        0.0,
        -0.0,
        -0.001,
        0.001,
        -0.25,
        0.25,
        -1.0,
        1.0,
        -math.pi / 2,
        math.pi / 2,
        -math.pi + 0.00001,
        math.pi - 0.00001,
        -math.pi,
        math.pi,
        -math.pi - 0.00001,
        math.pi + 0.00001,
    )

    l_vals = []
    r_vals = []

    if dtype.is_floating_point:
        prod = product(_float_vals, _float_vals)
    elif dtype.is_complex:
        complex_vals = product(_float_vals, _float_vals)
        # Note the use of list is required here or the map generator will be
        #  emptied by the following product and it won't produce the desired cross-product
        complex_vals = [complex(*x) for x in complex_vals]
        prod = product(complex_vals, complex_vals)
    elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        prod = product(_int_vals, _int_vals)
    elif dtype is torch.uint8:
        prod = product(_unsigned_int_vals, _unsigned_int_vals)
    else:
        raise ValueError("Unsupported dtype!")

    for l, r in prod:
        l_vals.append(l)
        if r == 0 and exclude_zero:
            r_vals.append(1)
        else:
            r_vals.append(r)

    lhs = torch.tensor(l_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    rhs = torch.tensor(r_vals, device=device, dtype=dtype, requires_grad=requires_grad)

    yield SampleInput(lhs, args=(rhs,))


def generate_elementwise_binary_large_value_tensors(
    op, *, device, dtype, requires_grad=False
):
    _large_int_vals = (-1113, 1113, -10701, 10701)
    _large_float16_vals = (-501, 501, -1001.2, 1001.2, -13437.7, 13437.7)
    _large_float_vals = _large_float16_vals + (-4988429.2, 4988429.2, -1e20, 1e20)

    l_vals = []
    r_vals = []

    if dtype == torch.float16:
        prod = product(_large_float16_vals, _large_float16_vals)
    elif dtype.is_floating_point:
        prod = product(_large_float_vals, _large_float_vals)
    elif dtype.is_complex:
        complex_vals = product(_large_float_vals, _large_float_vals)
        # Note the use of list is required here or the map generator will be
        #  emptied by the following product and it won't produce the desired cross-product
        complex_vals = [complex(*x) for x in complex_vals]
        prod = product(complex_vals, complex_vals)
    elif dtype in (torch.int16, torch.int32, torch.int64):
        prod = product(_large_int_vals, _large_int_vals)
    else:
        raise ValueError("Unsupported dtype!")

    for l, r in prod:
        l_vals.append(l)
        r_vals.append(r)

    lhs = torch.tensor(l_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    rhs = torch.tensor(r_vals, device=device, dtype=dtype, requires_grad=requires_grad)

    yield SampleInput(lhs, args=(rhs,))


def generate_elementwise_binary_extremal_value_tensors(
    op, *, device, dtype, requires_grad=False
):
    _float_extremals = (float("inf"), float("-inf"), float("nan"))

    l_vals = []
    r_vals = []

    if dtype.is_floating_point:
        prod = product(_float_extremals, _float_extremals)
    elif dtype.is_complex:
        complex_vals = product(_float_extremals, _float_extremals)
        # Note the use of list is required here or the map generator will be
        #  emptied by the following product and it won't produce the desired cross-product
        complex_vals = [complex(*x) for x in complex_vals]
        prod = product(complex_vals, complex_vals)
    else:
        raise ValueError("Unsupported dtype!")

    for l, r in prod:
        l_vals.append(l)
        r_vals.append(r)

    lhs = torch.tensor(l_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    rhs = torch.tensor(r_vals, device=device, dtype=dtype, requires_grad=requires_grad)

    yield SampleInput(lhs, args=(rhs,))

    # Test case for NaN propagation
    nan = (
        float("nan") if dtype.is_floating_point else complex(float("nan"), float("nan"))
    )
    lhs = make_tensor(
        (128, 128), device=device, dtype=dtype, requires_grad=requires_grad
    )
    lhs.view(-1)[::3] = nan
    rhs = make_tensor(
        (128, 128), device=device, dtype=dtype, requires_grad=requires_grad
    )
    rhs.view(-1)[::3] = nan

    yield SampleInput(lhs, args=(rhs,))


# Returns a generator of pairs of contiguous and noncontiguous tensors that
#   require broadcasting
def generate_elementwise_binary_broadcasting_tensors(
    op, *, device, dtype, requires_grad=False, exclude_zero=False
):
    shapes = (
        ((1,), ()),
        ((2,), ()),
        ((1,), (2,)),
        ((2, 1), (2,)),
        ((1, 2), (2,)),
        ((3, 2), (2,)),
        ((1, 3, 2), (2,)),
        ((1, 3, 2), (3, 2)),
        ((3, 1, 2), (3, 2)),
        ((2, 3, 2), ()),
        ((3, 1, 2), (1, 3, 2)),
    )

    make_arg = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    for shape, noncontiguous in product(shapes, [True, False]):
        shape_lhs, shape_rhs = shape
        lhs = make_arg(
            shape_lhs, noncontiguous=noncontiguous, **op.lhs_make_tensor_kwargs
        )
        rhs = make_arg(
            shape_rhs, noncontiguous=noncontiguous, **op.rhs_make_tensor_kwargs
        )

        yield SampleInput(lhs, args=(rhs,), broadcasts_input=True)


# Returns a generator of pairs of contiguous tensors and scalars
def generate_elementwise_binary_with_scalar_samples(
    op, *, device, dtype, requires_grad=False
):
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    shapes = ((), (3,), (5, 3), (0, 1, 3), (1, 5))
    if op.supports_rhs_python_scalar:
        for shape in shapes:
            lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
            rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
            lhs_scalar = make_arg((), **op.lhs_make_tensor_kwargs).item()
            rhs_scalar = make_arg((), **op.rhs_make_tensor_kwargs).item()

            yield SampleInput(lhs, args=(rhs_scalar,))

        # Extends with scalar lhs
        if op.supports_one_python_scalar:
            yield SampleInput(lhs_scalar, args=(rhs,))

    if op.supports_two_python_scalars:
        lhs_scalar = make_arg((), **op.lhs_make_tensor_kwargs).item()
        rhs_scalar = make_arg((), **op.rhs_make_tensor_kwargs).item()

        yield SampleInput(lhs_scalar, args=(rhs_scalar,))


# Returns a generator of pairs of contiguous tensors and 0d tensors and scalars and type promotion
def generate_elementwise_binary_with_scalar_and_type_promotion_samples(
    op, *, device, dtype, requires_grad=False
):
    # add these samples only for logical and comparison ops, arithmetic ops are not happy about extremal scalars
    if op.name in (
        "eq",
        "ne",
        "gt",
        "ge",
        "lt",
        "le",
        "logical_and",
        "logical_or",
        "logical_xor",
    ):
        make_arg = partial(
            make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
        )
        shape = (
            23,
        )  # this shape is big enough to trigger vectorization, and has non-vectorized tail
        values = (float("nan"), float("inf"), -float("inf"))
        scalar_tensors = tuple(torch.tensor(val) for val in values)
        if op.supports_rhs_python_scalar:
            lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
            rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
            for scalar in values + scalar_tensors:
                yield SampleInput(lhs, args=(scalar,))
                # Extends with scalar lhs
                if op.supports_one_python_scalar:
                    yield SampleInput(scalar, args=(rhs,))


# Returns a generator of pairs of noncontiguous tensors
def generate_elementwise_binary_noncontiguous_tensors(
    op, *, device, dtype, requires_grad=False, exclude_zero=False
):
    make_arg = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )

    # Generic noncontiguity
    lhs = make_arg((1026,), noncontiguous=True, **op.lhs_make_tensor_kwargs)
    rhs = make_arg((1026,), noncontiguous=True, **op.rhs_make_tensor_kwargs)

    yield SampleInput(lhs.clone(), args=(rhs.clone(),))
    yield SampleInput(lhs.contiguous(), args=(rhs,))

    # Transposed
    lhs = make_arg((789, 357), **op.lhs_make_tensor_kwargs)
    rhs = make_arg((789, 357), **op.rhs_make_tensor_kwargs)

    yield SampleInput(lhs.T, args=(rhs.T,))

    # More noncontiguity
    shapes = ((5, 7), (1024,))

    for shape in shapes:
        lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)

        lhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
        lhs_non_contig.copy_(lhs)

        rhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
        rhs_non_contig.copy_(rhs)

        yield SampleInput(lhs_non_contig.clone(), args=(rhs_non_contig.clone(),))
        yield SampleInput(lhs_non_contig.contiguous(), args=(rhs_non_contig,))

    # Noncontiguous indices
    shape = (2, 2, 1, 2)
    lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
    rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)

    lhs_non_contig = lhs[:, 1, ...]
    rhs_non_contig = rhs[:, 1, ...]

    yield SampleInput(lhs_non_contig.clone(), args=(rhs_non_contig.clone(),))
    yield SampleInput(lhs_non_contig.contiguous(), args=(rhs_non_contig,))

    # Expanded tensors
    shapes = ((1, 3), (1, 7), (5, 7))

    for shape in shapes:
        lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)

        lhs_non_contig = lhs.expand(3, -1, -1)
        rhs_non_contig = rhs.expand(3, -1, -1)

        yield SampleInput(lhs_non_contig, args=(rhs_non_contig,))


# Sample inputs for elementwise binary operators, like add
def sample_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs):
    _M = S if kwargs.get("small_inputs_only", False) else M
    _S = XS if kwargs.get("small_inputs_only", False) else S

    if hasattr(op, "rhs_make_tensor_kwargs"):
        exclude_zero = op.rhs_make_tensor_kwargs.get("exclude_zero", False)

    make_arg = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )

    shapes = (
        ((), ()),
        ((_S,), ()),
        ((_S, 1), (_S,)),
        ((_M, _S), ()),
        ((_S, _M, _S), (_M, _S)),
        ((_S, _M, _S), (_S, _M, _S)),
        ((_M, 1, _S), (_M, _S)),
        ((_M, 1, _S), (1, _M, _S)),
        ((0, 1, XS), (0, _M, XS)),
    )

    sample_kwargs = kwargs.get("sample_kwargs", {})

    for shape_lhs, shape_rhs in shapes:
        lhs = make_arg(shape_lhs, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape_rhs, **op.rhs_make_tensor_kwargs)
        broadcasts_input = shape_lhs != torch.broadcast_shapes(shape_lhs, shape_rhs)

        yield SampleInput(
            lhs, args=(rhs,), kwargs=sample_kwargs, broadcasts_input=broadcasts_input
        )


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

    def __init__(
        self,
        name,
        *,
        sample_inputs_func=sample_inputs_elementwise_binary,
        reference_inputs_func=reference_inputs_elementwise_binary,
        error_inputs_func=None,
        lhs_make_tensor_kwargs=None,
        rhs_make_tensor_kwargs=None,
        always_returns_bool=False,  # Set to true if the op always returns bool tensors
        supports_rhs_python_scalar=True,  # Whether the operator allows Tensor x scalar inputs
        supports_one_python_scalar=False,  # Whether the operator allows scalar x tensor and tensor x scalar inputs
        supports_two_python_scalars=False,  # Whether the operator allows scalar x scalar inputs
        **kwargs,
    ):
        self._original_binary_ufunc_args = locals().copy()

        # Elementwise binary operations perform the equivalent of test_numpy_refs
        #   in test_binary_ufuncs, but with additional test granularity. So the
        #   generic test_ops.py test is skipped because it's redundant.
        common_skips = (
            DecorateInfo(
                unittest.skip("Skipping redundant test."),
                "TestCommon",
                "test_numpy_refs",
            ),
        )
        kwargs["skips"] = kwargs.get("skips", ()) + common_skips
        super().__init__(
            name,
            sample_inputs_func=sample_inputs_func,
            reference_inputs_func=reference_inputs_func,
            error_inputs_func=make_error_inputs_elementwise_binary(error_inputs_func),
            **kwargs,
        )

        # [lr]hs_make_tensor_kwargs are part of the OpInfo to be able to dynamically generate valid samples later on.
        if lhs_make_tensor_kwargs is None:
            lhs_make_tensor_kwargs = {}
        self.lhs_make_tensor_kwargs = lhs_make_tensor_kwargs

        if rhs_make_tensor_kwargs is None:
            rhs_make_tensor_kwargs = {}
        self.rhs_make_tensor_kwargs = rhs_make_tensor_kwargs

        self.always_returns_bool = always_returns_bool
        self.supports_rhs_python_scalar = supports_rhs_python_scalar
        self.supports_one_python_scalar = supports_one_python_scalar
        self.supports_two_python_scalars = supports_two_python_scalars

        if self.supports_two_python_scalars:
            self.supports_one_python_scalar = True

        if self.supports_one_python_scalar:
            assert (
                supports_rhs_python_scalar
            ), "Can't support lhs and rhs Python scalars but not rhs scalars!"


# The following functions and classes are for testing elementwise unary operators.
def sample_inputs_elementwise_unary(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    if not op_kwargs:
        op_kwargs = {}

    _L = S if kwargs.get("small_inputs_only", False) else L

    low, high = op_info.domain
    is_floating = dtype.is_floating_point or dtype.is_complex
    low = low if low is None or not is_floating else low + op_info._domain_eps
    high = high if high is None or not is_floating else high - op_info._domain_eps
    if (
        op_info.supports_sparse_csr
        or op_info.supports_sparse_csc
        or op_info.supports_sparse_bsr
        or op_info.supports_sparse_bsc
    ):
        # Tensors with dim=2 for sparse compressed testing
        yield SampleInput(
            make_tensor(
                (_L, _L),
                device=device,
                dtype=dtype,
                low=low,
                high=high,
                requires_grad=requires_grad,
            ),
            kwargs=op_kwargs,
        )
    else:
        # Creates a 1D, empty, and scalar tensor
        for shape in ((_L,), (1, 0, 3), ()):
            yield SampleInput(
                make_tensor(
                    shape,
                    device=device,
                    dtype=dtype,
                    low=low,
                    high=high,
                    requires_grad=requires_grad,
                ),
                kwargs=op_kwargs,
            )


# Replace values satisfying condition with a safe value. This is used to block
# out values the could cause singularity like tan(pi/2)
def _replace_values_in_tensor(tensor, condition, safe_value):
    mask = condition(tensor)
    tensor.masked_fill_(mask, safe_value)


# Helper to create a unary elementwise tensor with valid inputs
def _make_unary_elementwise_tensor(shape, *, op, dtype, **kwargs):
    low, high = op.domain
    is_floating = dtype.is_floating_point or dtype.is_complex
    low = low if low is None or not is_floating else low + op._domain_eps
    high = high if high is None or not is_floating else high - op._domain_eps

    a = make_tensor(shape, low=low, high=high, dtype=dtype, **kwargs)

    if op.reference_numerics_filter is not None and dtype is not torch.bool:
        condition, safe_value = op.reference_numerics_filter
        _replace_values_in_tensor(a, condition, safe_value)

    return a


# Restricts the values in the tensor to the domain of the
# given elementwise unary operator
def _filter_unary_elementwise_tensor(a, *, op):
    # short-circuits for boolean tensors
    if a.dtype is torch.bool:
        return a

    low, high = op.domain
    is_floating = a.dtype.is_floating_point or a.dtype.is_complex
    low = low if low is None or not is_floating else low + op._domain_eps
    high = high if high is None or not is_floating else high - op._domain_eps

    if a.dtype is torch.uint8 and low is not None:
        low = max(low, 0)

    if not a.dtype.is_floating_point and not a.dtype.is_complex:
        low = math.ceil(low) if low is not None else None
        high = math.floor(high) if high is not None else None

    if op.reference_numerics_filter is not None:
        condition, safe_value = op.reference_numerics_filter
        _replace_values_in_tensor(a, condition, safe_value)

    if low is not None or high is not None:
        if a.dtype.is_complex:
            a.real.clamp_(low, high)
            a.imag.clamp_(low, high)
        else:
            a.clamp_(min=low, max=high)

    return a


def generate_elementwise_unary_tensors(op, *, device, dtype, requires_grad, **kwargs):
    # Special-cases bool
    if dtype is torch.bool:
        tensors = (
            torch.empty(0, device=device, dtype=torch.bool),
            torch.tensor(True, device=device),
            torch.tensor(False, device=device),
            torch.tensor((True, False), device=device),
            make_tensor((812,), device=device, dtype=dtype),
            make_tensor((1029, 917), device=device, dtype=dtype),
        )
        for a in tensors:
            yield SampleInput(a, kwargs=op.sample_kwargs(device, dtype, a)[0])

    shapes = (
        (1029, 917),
        (812,),
        # Empty sizes
        (0,),
        (0, 3, 3),
        (1, 0, 5),
        (6, 0, 0, 0),
        (3, 0, 1, 0),
    )

    make_arg = partial(
        _make_unary_elementwise_tensor,
        op=op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    for shape in shapes:
        a = make_arg(shape)
        yield SampleInput(a, kwargs=op.sample_kwargs(device, dtype, a)[0])


def generate_elementwise_unary_small_value_tensors(
    op, *, device, dtype, requires_grad=False
):
    for sample in generate_elementwise_binary_small_value_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad
    ):
        a = _filter_unary_elementwise_tensor(sample.input, op=op)
        yield SampleInput(a, kwargs=op.sample_kwargs(device, dtype, a)[0])


def generate_elementwise_unary_large_value_tensors(
    op, *, device, dtype, requires_grad=False
):
    for sample in generate_elementwise_binary_large_value_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad
    ):
        a = _filter_unary_elementwise_tensor(sample.input, op=op)
        yield SampleInput(sample.input, kwargs=op.sample_kwargs(device, dtype, a)[0])


def generate_elementwise_unary_extremal_value_tensors(
    op, *, device, dtype, requires_grad=False
):
    for sample in generate_elementwise_binary_extremal_value_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad
    ):
        yield SampleInput(
            sample.input, kwargs=op.sample_kwargs(device, dtype, sample.input)[0]
        )


def generate_elementwise_unary_noncontiguous_tensors(
    op, *, device, dtype, requires_grad=False
):
    make_arg = partial(
        _make_unary_elementwise_tensor,
        op=op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    # Generic noncontiguity
    t = make_arg((1026,), noncontiguous=True)
    yield SampleInput(t, kwargs=op.sample_kwargs(device, dtype, t)[0])

    # Transposed
    t = make_arg((1024, 1024)).T
    yield SampleInput(t, kwargs=op.sample_kwargs(device, dtype, t)[0])

    # Expanded tensors
    shapes = ((1, 3), (1, 7), (5, 7))

    for shape in shapes:
        t = make_arg(shape)
        t_non_contig = t.expand(3, -1, -1)
        yield SampleInput(
            t_non_contig, kwargs=op.sample_kwargs(device, dtype, t_non_contig)[0]
        )


def generate_elementwise_unary_arbitrarily_strided_tensors(
    op, *, device, dtype, requires_grad=False
):
    # shape, strides, offset
    strided_cases = (
        ((5, 6, 2), (1, 1, 7), 2),
        ((5, 5, 4), (1, 1, 7), 2),
        ((5, 5, 2), (4, 5, 7), 3),
        ((5, 5, 2), (5, 5, 7), 3),
        ((5, 5, 2), (5, 5, 5), 3),
        ((9, 5, 2), (0, 1, 7), 3),
    )

    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    for shape, strides, offset in strided_cases:
        a = make_arg(
            500,
        ).as_strided(shape, strides, offset)
        yield SampleInput(a, kwargs=op.sample_kwargs(device, dtype, a)[0])


# Reuses the elementwise binary generators for consistency
# TODO: in the future generalize the reference generators to handle n-ary elementwise operations
def _reference_inputs_elementwise_unary(op, device, dtype, requires_grad, **kwargs):
    yield from op.sample_inputs_func(op, device, dtype, requires_grad, **kwargs)

    yield from generate_elementwise_unary_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
    )

    if dtype is not torch.bool:
        yield from generate_elementwise_unary_small_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
        )
    if dtype not in (torch.bool, torch.uint8, torch.int8) and (
        op.handles_large_floats
        or (not dtype.is_floating_point and not dtype.is_complex)
    ):
        yield from generate_elementwise_unary_large_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
        )

    if dtype.is_floating_point or (
        op.handles_complex_extremal_values and dtype.is_complex
    ):
        yield from generate_elementwise_unary_extremal_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
        )


def reference_inputs_elementwise_unary(op, device, dtype, requires_grad, **kwargs):
    gen = partial(
        _reference_inputs_elementwise_unary, op, device, dtype, requires_grad, **kwargs
    )

    # yields "normal" samples
    yield from gen()

    # yields noncontiguous samples
    for sample in gen():
        yield sample.noncontiguous()

    yield from generate_elementwise_unary_noncontiguous_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
    )

    yield from generate_elementwise_unary_arbitrarily_strided_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
    )


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

    def __init__(
        self,
        name,  # the string name of the function
        *,
        dtypes=floating_types(),
        domain=(None, None),  # the [low, high) domain of the function
        handles_complex_extremal_values=True,  # whether the op correctly handles extremal values (like nan/inf)
        handles_large_floats=True,  # whether the op correctly handles large float values (like 1e20)
        supports_complex_to_float=False,  # op supports casting from complex input to real output safely eg. angle
        sample_inputs_func=sample_inputs_elementwise_unary,
        reference_inputs_func=reference_inputs_elementwise_unary,
        sample_kwargs=lambda device, dtype, input: ({}, {}),
        reference_numerics_filter=None,  # Filters values in the range of the domain specified above but that should not be tested
        **kwargs,
    ):
        self._original_unary_ufunc_args = locals().copy()

        super().__init__(
            name,
            dtypes=dtypes,
            sample_inputs_func=sample_inputs_func,
            reference_inputs_func=reference_inputs_func,
            **kwargs,
        )
        self.domain = domain
        self.handles_complex_extremal_values = handles_complex_extremal_values
        self.handles_large_floats = handles_large_floats
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


def sample_inputs_spectral_ops(self, device, dtype, requires_grad=False, **kwargs):
    is_fp16_or_chalf = dtype == torch.complex32 or dtype == torch.half
    if not is_fp16_or_chalf:
        nd_tensor = partial(
            make_tensor,
            (S, S + 1, S + 2),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        oned_tensor = partial(
            make_tensor, (31,), device=device, dtype=dtype, requires_grad=requires_grad
        )
    else:
        # cuFFT supports powers of 2 for half and complex half precision
        # NOTE: For hfft, hfft2, hfftn, irfft, irfft2, irfftn with default args
        # where output_size n=2*(input_size - 1), we make sure that logical fft size is a power of two
        low = None
        high = None
        if self.name in ["fft.hfft", "fft.irfft", "_refs.fft.hfft", "_refs.fft.irfft"]:
            shapes = ((2, 9, 9), (33,))
        elif self.name in [
            "fft.hfft2",
            "fft.irfft2",
            "_refs.fft.hfft2",
            "_refs.fft.irfft2",
        ]:
            shapes = ((2, 8, 9), (33,))
        elif self.name in [
            "fft.hfftn",
            "fft.irfftn",
            "_refs.fft.hfftn",
            "_refs.fft.irfftn",
        ]:
            shapes = ((2, 2, 33), (33,))
            # Adjusting the limits because the test would be flaky due to over-saturation of float16
            # See: https://github.com/pytorch/pytorch/pull/81416
            low = -1.0
            high = 1.0
        else:
            shapes = ((2, 8, 16), (32,))
        nd_tensor = partial(
            make_tensor,
            shapes[0],
            device=device,
            low=low,
            high=high,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        oned_tensor = partial(
            make_tensor,
            shapes[1],
            device=device,
            low=low,
            high=high,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    if self.ndimensional == SpectralFuncType.ND:
        yield SampleInput(
            nd_tensor(),
            s=(3, 10) if not is_fp16_or_chalf else (4, 8),
            dim=(1, 2),
            norm="ortho",
        )
        yield SampleInput(nd_tensor(), norm="ortho")
        yield SampleInput(nd_tensor(), s=(8,))
        yield SampleInput(oned_tensor())
        yield from (SampleInput(nd_tensor(), dim=dim) for dim in [-1, -2, -3, (0, -1)])
    elif self.ndimensional == SpectralFuncType.TwoD:
        yield SampleInput(
            nd_tensor(),
            s=(3, 10) if not is_fp16_or_chalf else (4, 8),
            dim=(1, 2),
            norm="ortho",
        )
        yield SampleInput(nd_tensor(), norm="ortho")
        yield SampleInput(nd_tensor(), s=(6, 8) if not is_fp16_or_chalf else (4, 8))
        yield SampleInput(nd_tensor(), dim=0)
        yield SampleInput(nd_tensor(), dim=(0, -1))
        yield SampleInput(nd_tensor(), dim=(-3, -2, -1))
    else:
        yield SampleInput(
            nd_tensor(),
            n=10 if not is_fp16_or_chalf else 8,
            dim=1,
            norm="ortho",
        )
        yield SampleInput(nd_tensor(), norm="ortho")
        yield SampleInput(nd_tensor(), n=7 if not is_fp16_or_chalf else 8)
        yield SampleInput(oned_tensor())
        yield from (SampleInput(nd_tensor(), dim=dim) for dim in [-1, -2, -3])


SpectralFuncType = Enum("SpectralFuncType", ("OneD", "TwoD", "ND"))


# Metadata class for Fast Fourier Transforms in torch.fft.
class SpectralFuncInfo(OpInfo):
    """Operator information for torch.fft transforms."""

    def __init__(
        self,
        name,  # the string name of the function
        *,
        ref=None,  # Reference implementation (probably in np.fft namespace)
        dtypes=floating_and_complex_types(),
        ndimensional: SpectralFuncType,
        sample_inputs_func=sample_inputs_spectral_ops,
        decorators=None,
        **kwargs,
    ):
        self._original_spectral_func_args = dict(locals()).copy()
        self._original_spectral_func_args.update(kwargs)

        decorators = list(decorators) if decorators is not None else []
        decorators += [
            skipCPUIfNoFFT,
            DecorateInfo(
                toleranceOverride({torch.chalf: tol(4e-2, 4e-2)}),
                "TestCommon",
                "test_complex_half_reference_testing",
            ),
        ]

        super().__init__(
            name=name,
            dtypes=dtypes,
            decorators=decorators,
            sample_inputs_func=sample_inputs_func,
            **kwargs,
        )
        self.ref = ref
        self.ndimensional = ndimensional


class ShapeFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for Shape manipulating operations like tile and roll"""

    def __init__(
        self,
        name,  # the string name of the function
        *,
        ref,  # a reference function
        dtypes=floating_types(),
        dtypesIfCUDA=None,
        dtypesIfROCM=None,
        dtypesIfXPU=None,
        sample_inputs_func=None,
        **kwargs,
    ):
        super().__init__(
            name,
            dtypes=dtypes,
            dtypesIfCUDA=dtypesIfCUDA,
            dtypesIfROCM=dtypesIfROCM,
            dtypesIfXPU=dtypesIfXPU,
            sample_inputs_func=sample_inputs_func,
            **kwargs,
        )
        self.ref = ref


def sample_inputs_foreach(
    self,
    device,
    dtype,
    N,
    *,
    noncontiguous=False,
    same_size=False,
    low=None,
    high=None,
    zero_size: bool,
    requires_grad: bool,
    # mutually exclusive from same_size and zero_size, which are all or nothing
    intersperse_empty_tensors: bool = False,
):
    if zero_size:
        return [torch.empty(0, dtype=dtype, device=device) for _ in range(N)]
    if same_size:
        return [
            make_tensor(
                (N, N),
                dtype=dtype,
                device=device,
                noncontiguous=noncontiguous,
                low=low,
                high=high,
                requires_grad=requires_grad,
            )
            for _ in range(N)
        ]
    else:
        # interweave some empty tensors + have the last 2 tensors be empty (see #100701)
        return [
            torch.empty(0, dtype=dtype, device=device, requires_grad=requires_grad)
            if (i % 3 == 0 or i >= N - 2) and intersperse_empty_tensors
            else make_tensor(
                (N - i, N - i),
                dtype=dtype,
                device=device,
                noncontiguous=noncontiguous,
                low=low,
                high=high,
                requires_grad=requires_grad,
            )
            for i in range(N)
        ]


def get_foreach_method_names(name):
    # get torch inplace reference function
    op_name = "_foreach_" + name
    inplace_op_name = op_name + "_"

    op = getattr(torch, op_name, None)
    inplace_op = getattr(torch, inplace_op_name, None)

    ref = getattr(torch, name, None)
    ref_inplace = getattr(torch.Tensor, name + "_", None)
    return op, inplace_op, ref, ref_inplace


@dataclass
class ForeachFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for foreach functions

    The main differences from the parent class are (a) `dtypes`, `dtypesIfCUDA`, and `dtypesIfROCM`
    are set to `get_all_dtypes(include_qint=False)`, and (b) the following arguments.

    ``supports_alpha_param=True`` means that the function supports a python scalar (``numbers.Number``)
    as the last keyword argument such as `_foreach_add`.
    ``supports_scalar_self_arg=True`` means that the function can take a python scalar as its first argument.
    Currently only `_foreach_pow` supports this.
    ``backward_requires_result=True``, which could sound self-explanatory, means that the function uses
    the forward result for its backward computation.
    """

    supports_alpha_param: bool = False
    supports_scalar_self_arg: bool = False
    backward_requires_result: bool = False

    def __post_init__(self):
        (
            foreach_method,
            foreach_method_inplace,
            torch_ref_method,
            torch_ref_inplace,
        ) = get_foreach_method_names(self.name)
        if not self.supports_out:
            # note(crcrpar): `foreach_method` for `"zero"` is `None` but `None` would call
            # `_getattr_qual` in `OpInfo.__post_init__` which should fail since `_foreach_zero`
            # is not defined at the moment. Thus to skip the qualification, set a similar torch
            # function.
            assert foreach_method is None
            assert torch_ref_method is None
            foreach_method = foreach_method_inplace
            torch_ref_method = torch_ref_inplace

        # We disable all complex128 tests internally for foreach due to reported flakiness
        # tracked in #139648
        supported_dtypes = get_all_dtypes(include_qint=False)
        if IS_FBCODE:
            supported_dtypes = [
                x for x in supported_dtypes if x is not torch.complex128
            ]
        self.dtypes = _dispatch_dtypes(supported_dtypes)

        self.op = foreach_method
        self.method_variant = foreach_method
        self.ref = torch_ref_method
        self.inplace_variant = foreach_method_inplace
        self.ref_inplace = torch_ref_inplace
        self.has_no_in_place = self.inplace_variant is None

        name = self.name
        self.name = f"_foreach_{name}"
        if name == "norm":
            self.ref = torch.linalg.vector_norm
        elif name == "minimum":
            # because minimum ref does not support inplace or scalar
            self.ref = torch.clamp_max
            self.ref_inplace = torch.Tensor.clamp_max_
        elif name == "maximum":
            # because maximum ref does not support inplace or scalar
            self.ref = torch.clamp_min
            self.ref_inplace = torch.Tensor.clamp_min_

        # The following sets `dtypesIfCUDA` and `dtypesIfROCM` accordingly.
        super().__post_init__()

    def sample_zero_size_inputs(self, device, dtype, requires_grad=False, **kwargs):
        if not hasattr(self.sample_inputs_func, "sample_zero_size_tensor_inputs"):
            return []
        return self.sample_inputs_func.sample_zero_size_tensor_inputs(
            self, device, dtype, requires_grad, **kwargs
        )


def gradcheck_wrapper_hermitian_input(op, input, *args, **kwargs):
    """Gradcheck wrapper for functions that take Hermitian matrices as input.

    They require a modified function because the finite-difference algorithm
    for calculating derivatives does not preserve the Hermitian property of the input.
    """
    return op(input + input.mH, *args, **kwargs)


def gradcheck_wrapper_triangular_input(op, *args, upper=False, idx=0, **kwargs):
    """Gradcheck wrapper for functions that take lower or upper triangular matrices as input.

    They require a modified function because the finite-difference algorithm
    for calculating derivatives does not preserve the triangular property of the input.
    `idx` is used to specific which `args[idx]` is to be triangularized.
    """
    triangular_arg = args[idx].triu() if upper else args[idx].tril()
    return op(*args[:idx], triangular_arg, *args[idx + 1 :], upper, **kwargs)


def gradcheck_wrapper_triangular_input_real_positive_diagonal(
    op, *args, upper=False, idx=0, **kwargs
):
    """Gradcheck wrapper for functions that take lower/upper triangular matrices
    with real and positive diagonals, for example, cholesky-like operations.
    """
    arg = args[idx]
    arg_diag = arg.diagonal(0, -2, -1)
    arg_diag_embed = torch.diag_embed(arg_diag)
    id_diag_tensor = torch.ones_like(arg_diag)
    id_tensor = torch.diag_embed(id_diag_tensor)
    # new_arg = arg - diag(arg) + I
    new_arg = arg - arg_diag_embed + id_tensor
    return gradcheck_wrapper_triangular_input(
        op, *args[:idx], new_arg, *args[idx + 1 :], upper=upper, idx=idx, **kwargs
    )


def gradcheck_wrapper_masked_operation(op, input, *args, **kwargs):
    """Gradcheck wrapper for masked operations.

    When mask is specified, replaces masked-out elements with zeros.

    Use for operations that produce non-finite masked-out elements,
    for instance, for minimum and maximum reductions.
    """
    output = op(input, *args, **kwargs)
    mask = kwargs.get("mask")
    if mask is not None:
        output_mask = torch.masked._output_mask(op, input, *args, **kwargs)
        output = torch.where(output_mask, output, output.new_zeros([]))
    return output


def gradcheck_wrapper_masked_pointwise_operation(op, input, *args, **kwargs):
    """Gradcheck wrapper for masked pointwise operations. Assumes that the result
    will be masked iff both tensors are masked at a specific index

    When mask is specified, replaces masked-out elements with zeros.

    Use for operations that produce non-finite masked-out elements,
    for instance, for minimum and maximum reductions.
    """
    output = op(input, *args, **kwargs)
    input_mask = kwargs.get("input_mask")
    other_mask = kwargs.get("other_mask")
    if input_mask is not None and other_mask is not None:
        combined_mask = torch.logical_and(input_mask, other_mask)
        new_kwargs = dict(mask=combined_mask, **kwargs)
        output_mask = torch.masked._input_mask(input, *args, **new_kwargs)
        output = torch.where(output_mask, output, output.new_zeros([]))
    return output


def clone_sample(sample, **kwargs):
    """
    Given a SampleInput, this function analyzes its input, args and kwargs,
    and produces a copy with each non-Tensor entry being copied by reference,
    and with each Tensor entry cloned with `t.clone().requires_grad_(t.requires_grad)`
    """

    def clone_tensor(t):
        if isinstance(t, torch.Tensor):
            return t.detach().clone().requires_grad_(t.requires_grad)
        else:
            return t

    sample_kwargs = kwargs if kwargs else sample.kwargs

    return SampleInput(
        clone_tensor(sample.input),
        args=tuple(map(clone_tensor, sample.args)),
        kwargs={k: clone_tensor(v) for k, v in sample_kwargs.items()},
    )
