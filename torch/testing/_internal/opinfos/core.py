import collections
import operator
import numbers

import torch
import collections.abc

from typing import Sequence, Union

from torch.testing import floating_types, floating_and_complex_types_and

from torch.testing._core import _dispatch_dtypes
from torch.testing._internal.common_device_type import skipIf
from torch.testing._internal.common_utils import is_iterable_of_tensors, TEST_WITH_ROCM
import torch.testing._internal.opinfo_helper as opinfo_helper

# default sizes of Large, Medium and Small tensors used by tests
L = 20
M = 10
S = 5


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

    # Returns the NumPy version of the sample input object in the form of a tuple: (input, args, kwargs)
    def numpy(self):
        # Converts tensors to ndarrays by calling .detach().cpu().numpy() on them
        # Numbers, strings, and bool are preserved as is
        # Lists, tuples and dicts are handled by calling this function recursively
        def to_numpy(x):
            def _np(t):
                return t.detach().cpu().numpy()

            if isinstance(x, torch.Tensor):
                return _np(x)
            elif isinstance(x, list):
                return list(map(to_numpy, x))
            elif isinstance(x, tuple):
                return tuple(map(to_numpy, x))
            elif isinstance(x, dict):
                return {k: to_numpy(v) for k, v in x.items()}
            elif isinstance(x, (numbers.Number, bool, str)):
                return x

            raise ValueError("Unknown type {0}!".format(type(x)))

        sample_np_input, np_args, np_kwargs = to_numpy(self.input), to_numpy(self.args), to_numpy(self.kwargs)
        return (sample_np_input, np_args, np_kwargs)

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

# Note [OpInfos]
# ~~~~~~~~~~~~~~
#
# This note was written shortly after the PyTorch 1.9 release.
# If you notice it's out-of-date or think it could be improved then please
# file an issue.
#
# See also: the OpInfo tracker (https://github.com/pytorch/pytorch/issues/54261)
# See also: "Writing Test Templates" in common_device_type.py to learn how to
#   parametrize a test template using OpInfos.
#
# An OpInfo is a collection of metadata related to a PyTorch operator. This
#   metadata is used to generate tests that validate properties of the operator,
#   like if it implements the correct gradient formula.
#
# WHY OPINFOS?
# ~~~~~~~~~~~~
#
# OpInfos are principally intended to do two things:
#
#   1) to simplify testing an operator
#   2) to allow systems (like autograd, torchscript, fx, nnc...) to test
#        against every PyTorch operator
#
# Both these goals are still a work in progress. Not every operator has an
#   OpInfo, and some operator tests still have to be written manually.
#
# The utility of OpInfos can also be motivated from a different perspective.
#   PyTorch is a complicated framework with many interrelated systems, too
#   many for any one person to keep track of. An OpInfo can be thought of as the
#   interface between an operator implementer and those other systems. Instead of
#   requiring the implementer of torch.foo understand how to test its forward
#   mode AD or NNC support that's typically handled automatically just by
#   defining an OpInfo. This is a helpful perspective to have, because it's often
#   surprising to OpInfo writers that just implementing an OpInfo typically can't
#   verify an operator is actually implemented correctly. "If an OpInfo doesn't
#   validate my op works as expected, what's the point of it?" But the point of
#   it is that it lets engineers focus on testing their operator logic instead
#   of having to write tests for how the operator interacts with each of
#   PyTorch's many systems. And, OK, sometimes it validates your op works
#   the way you want and all you have to do is write an OpInfo and you're done
#   testing... more on that below.
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
#   And should return a list of SampleInputs (see the class description above).
#   Each SampleInput defines an "input", "args", "kwargs",
#   an "output_process_fn_grad" function, the "broadcasts_input" bool and
#   a "name".
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
# If you're adding a new operator to the torch, torch.fft, torch.linalg,
#   or torch.special namespaces then you should add an OpInfo for it. As
#   mentioned a couple times above, implementing an OpInfo is not usually
#   sufficient testing (unless the operator is a unary elementwise operator).
#   The OpInfo will only test the properties described in the "WHAT'S TESTED"
#   section. It DOES NOT verify that the operator is implemented correctly.
#
# We are currently reviewing if operators in the torch.nn.functional namespace
#   will be added as OpInfos, but you are encouraged to add an OpInfo for
#   such operators, too.
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
# This trial-and-error approach can be frustrating to writing an OpInfo can
#   be frustrating, but it's probably necessary as long as OpInfos don't require
#   learning about all the systems that consume them. One thing that can help
#   is the get_supported_dtypes() function defined in opinfo_helper.py. This
#   function can be used to programmatically specify the dtypes an operator
#   supports, and is especially useful if writing an OpInfo on a machine
#   without a CUDA device. See its documentation for more details.
#
# THE FUTURE OF OPINFOS AND OPINFO TESTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the future we expect OpInfo coverage to improve, particularly for the
#   torch, torch.fft, torch.linalg, and torch.special namespaces, and possibly
#   for the torch.nn.functional namespace, too. In addition an analogous class,
#   ModuleInfo, will be developed to improve module testing.
#
# We also expect at least two new OpInfo subclasses: BinaryUfuncInfo and
#   ReductionInfo. Both will have new automated tests for correctness, too,
#   which might make testing binary elementwise operations and reductions as
#   simple as testing unary elementwise operations today.

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
                 # modifying tests and a pointer to the op's sample inputs function
                 # this function lets the OpInfo generate valid inputs
                 skips=tuple(),  # information about which tests to skip
                 decorators=None,  # decorators to apply to generated tests
                 sample_inputs_func=None,  # function to generate sample inputs

                 # the following metadata relates to dtype support and is tested for correctness in test_ops.py
                 dtypes=floating_types(),  # dtypes this function is expected to work with
                 # the following dtypesIf... options override the dtypes value
                 # on their respective device types
                 dtypesIfCPU=None,  # dtypes this function is expected to work with on CPU
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
                 supports_autograd=True,  # whether the operation supports gradient computations
                                          # if true, gradient correctness is tested in test_ops.py
                                          # using the op's sample inputs
                 supports_gradgrad=True,  # whether the op supports second order gradients
                                          # if true, gradgrad correctness is tested in test_ops.py
                                          # (this value is ignored if supports_autograd=False)
                 supports_inplace_autograd=None,  # whether the operation supports inplace autograd
                                                  # if true, tested in test_ops.py
                                                  # defaults to supports_autograd's value
                 supports_forward_ad=False,  # Whether the operation support forward mode AD
                                             # If the value is True, we check that the gradients are correct
                                             # If the value is False, we test that forward grad is not implemented
                 gradcheck_wrapper=lambda op, *args, **kwargs: op(*args, **kwargs),  # wrapper function for gradcheck
                 check_batched_grad=True,  # whether to check batched grad when doing gradcheck
                 check_batched_gradgrad=True,  # whether to check batched grad grad when doing gradgradcheck
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

                 # the following metadata relates to complex support and is checked in test_ops.py
                 test_conjugated_samples=True,
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

        self.skips = skips
        self.decorators = decorators
        self.sample_inputs_func = sample_inputs_func

        self.assert_autodiffed = assert_autodiffed
        self.autodiff_fusible_nodes = autodiff_fusible_nodes if autodiff_fusible_nodes else []
        if autodiff_nonfusible_nodes is None:
            self.autodiff_nonfusible_nodes = ['aten::' + self.name]
        else:
            self.autodiff_nonfusible_nodes = autodiff_nonfusible_nodes

        # autograd support
        self.supports_autograd = supports_autograd
        self.supports_inplace_autograd = supports_inplace_autograd
        if self.supports_inplace_autograd is None:
            self.supports_inplace_autograd = supports_autograd

        self.gradcheck_wrapper = gradcheck_wrapper
        self.supports_gradgrad = supports_gradgrad
        self.supports_forward_ad = supports_forward_ad
        self.check_batched_grad = check_batched_grad
        self.check_batched_gradgrad = check_batched_gradgrad
        self.gradcheck_nondet_tol = gradcheck_nondet_tol
        self.gradcheck_fast_mode = gradcheck_fast_mode

        self.supports_sparse = supports_sparse

        self.aliases = ()
        if aliases is not None:
            self.aliases = tuple(AliasInfo(a) for a in aliases)  # type: ignore[assignment]

        self.test_conjugated_samples = test_conjugated_samples

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
            with torch.no_grad():
                tensor = tensor.conj()
            return tensor.requires_grad_(_requires_grad)

        for i in range(len(samples)):
            sample = conj_samples[i]
            # Note: it is assumed that the input here is either a tensor or tensorlist
            if isinstance(sample.input, torch.Tensor):
                sample.input = conjugate(sample.input)
            else:
                with torch.no_grad():
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
