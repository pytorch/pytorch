import copy
import gc
import inspect
import runpy
import threading
from collections import namedtuple
from enum import Enum
from functools import wraps
from typing import List, Any, ClassVar, Optional, Sequence, Tuple
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
    skipCUDANonDefaultStreamIf, TEST_WITH_ASAN, TEST_WITH_UBSAN, TEST_WITH_TSAN, \
    IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, DeterministicGuard, TEST_SKIP_NOARCH, \
    _TestParametrizer, dtype_name, TEST_WITH_MIOPEN_SUGGEST_NHWC
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_dtype import get_all_dtypes

try:
    import psutil  # type: ignore[import]
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Note [Writing Test Templates]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This note was written shortly after the PyTorch 1.9 release.
# If you notice it's out-of-date or think it could be improved then please
# file an issue.
#
# PyTorch has its own framework for instantiating test templates. That is, for
#   taking test classes that look similar to unittest or pytest
#   compatible test classes and optionally doing the following:
#
#     - instantiating a version of the test class for each available device type
#         (often the CPU, CUDA, and META device types)
#     - further instantiating a version of each test that's always specialized
#         on the test class's device type, and optionally specialized further
#         on datatypes or operators
#
# This functionality is similar to pytest's parametrize functionality
#   (see https://docs.pytest.org/en/6.2.x/parametrize.html), but with considerable
#   additional logic that specializes the instantiated test classes for their
#   device types (see CPUTestBase and CUDATestBase below), supports a variety
#   of composable decorators that allow for test filtering and setting
#   tolerances, and allows tests parametrized by operators to instantiate
#   only the subset of device type x dtype that operator supports.
#
# This framework was built to make it easier to write tests that run on
#   multiple device types, multiple datatypes (dtypes), and for multiple
#   operators. It's also useful for controlling which tests are fun. For example,
#   only tests that use a CUDA device can be run on platforms with CUDA.
#   Let's dive in with an example to get an idea for how it works:
#
# --------------------------------------------------------
# A template class (looks like a regular unittest TestCase)
# class TestClassFoo(TestCase):
#
#   # A template test that can be specialized with a device
#   # NOTE: this test case is not runnably by unittest or pytest because it
#   #   accepts an extra positional argument, "device", they do not understand
#   def test_bar(self, device):
#     pass
#
# # Function that instantiates a template class and its tests
# instantiate_device_type_tests(TestCommon, globals())
# --------------------------------------------------------
#
# In the above code example we see a template class and a single test template
#   that can be instantiated with a device. The function
#   instantiate_device_type_tests(), called at file scope, instantiates
#   new test classes, one per available device type, and new tests in those
#   classes from these templates. It actually does this by removing
#   the class TestClassFoo and replacing it with classes like TestClassFooCPU
#   and TestClassFooCUDA, instantiated test classes that inherit from CPUTestBase
#   and CUDATestBase respectively. Additional device types, like XLA,
#   (see https://github.com/pytorch/xla) can further extend the set of
#   instantiated test classes to create classes like TestClassFooXLA.
#
# The test template, test_bar(), is also instantiated. In this case the template
#   is only specialized on a device, so (depending on the available device
#   types) it might become test_bar_cpu() in TestClassFooCPU and test_bar_cuda()
#   in TestClassFooCUDA. We can think of the instantiated test classes as
#   looking like this:
#
# --------------------------------------------------------
# # An instantiated test class for the CPU device type
# class TestClassFooCPU(CPUTestBase):
#
#   # An instantiated test that calls the template with the string representation
#   #   of a device from the test class's device type
#   def test_bar_cpu(self):
#     test_bar(self, 'cpu')
#
# # An instantiated test class for the CUDA device type
# class TestClassFooCUDA(CUDATestBase):
#
#   # An instantiated test that calls the template with the string representation
#   #   of a device from the test class's device type
#   def test_bar_cuda(self):
#     test_bar(self, 'cuda:0')
# --------------------------------------------------------
#
# These instantiated test classes are discoverable and runnable by both
#   unittest and pytest. One thing that may be confusing, however, is that
#   attempting to run "test_bar" will not work, despite it appearing in the
#   original template code. This is because "test_bar" is no longer discoverable
#   after instantiate_device_type_tests() runs, as the above snippet shows.
#   Instead "test_bar_cpu" and "test_bar_cuda" may be run directly, or both
#   can be run with the option "-k test_bar".
#
# Removing the template class and adding the instantiated classes requires
#   passing "globals()" to instantiate_device_type_tests(), because it
#   edits the file's Python objects.
#
# As mentioned, tests can be additionally parametrized on dtypes or
#   operators. Datatype parametrization uses the @dtypes decorator and
#   require a test template like this:
#
# --------------------------------------------------------
# # A template test that can be specialized with a device and a datatype (dtype)
# @dtypes(torch.float32, torch.int64)
# def test_car(self, device, dtype)
#   pass
# --------------------------------------------------------
#
# If the CPU and CUDA device types are available this test would be
#   instantiated as 4 tests that cover the cross-product of the two dtypes
#   and two device types:
#
#     - test_car_cpu_float32
#     - test_car_cpu_int64
#     - test_car_cuda_float32
#     - test_car_cuda_int64
#
# The dtype is passed as a torch.dtype object.
#
# Tests parametrized on operators (actually on OpInfos, more on that in a
#   moment...) use the @ops decorator and require a test template like this:
# --------------------------------------------------------
# # A template test that can be specialized with a device, dtype, and OpInfo
# @ops(op_db)
# def test_car(self, device, dtype, op)
#   pass
# --------------------------------------------------------
#
# See the documentation for the @ops decorator below for additional details
#   on how to use it and see the note [OpInfos] in
#   common_methods_invocations.py for more details on OpInfos.
#
# A test parametrized over the entire "op_db", which contains hundreds of
#   OpInfos, will likely have hundreds or thousands of instantiations. The
#   test will be instantiated on the cross-product of device types, operators,
#   and the dtypes the operator supports on that device type. The instantiated
#   tests will have names like:
#
#     - test_car_add_cpu_float32
#     - test_car_sub_cuda_int64
#
# The first instantiated test calls the original test_car() with the OpInfo
#   for torch.add as its "op" argument, the string 'cpu' for its "device" argument,
#   and the dtype torch.float32 for is "dtype" argument. The second instantiated
#   test calls the test_car() with the OpInfo for torch.sub, a CUDA device string
#   like 'cuda:0' or 'cuda:1' for its "device" argument, and the dtype
#   torch.int64 for its "dtype argument."
#
# Clever test filtering can be very useful when working with parametrized
#   tests. "-k test_car" would run every instantiated variant of the test_car()
#   test template, and "-k test_car_add" runs every variant instantiated with
#   torch.add.
#
# It is important to use the passed device and dtype as appropriate. Use
#   helper functions like make_tensor() that require explicitly specifying
#   the device and dtype so they're not forgotten.
#
# Test templates can use a variety of composable decorators to specify
#   additional options and requirements, some are listed here:
#
#     - @deviceCountAtLeast(<minimum number of devices to run test with>)
#         Passes a list of strings representing all available devices of
#         the test class's device type as the test template's "device" argument.
#         If there are a fewer devices than the value passed to the decorator
#         the test is skipped.
#     - @dtypes(<list of tuples of dtypes>)
#         In addition to accepting multiple dtypes, the @dtypes decorator
#         can accept a sequence of tuple pairs of dtypes. The test template
#         will be called with each tuple for its "dtype" argument.
#     - @onlyOnCPUAndCUDA
#         Skips the test if the device is not a CPU or CUDA device
#     - @onlyCPU
#         Skips the test if the device is not a CPU device
#     - @onlyCUDA
#         Skips the test if the device is not a CUDA device
#     - @skipCPUIfNoLapack
#         Skips the test if the device is a CPU device and LAPACK is not installed
#     - @skipCPUIfNoMkl
#         Skips the test if the device is a CPU device and MKL is not installed
#     - @skipCUDAIfNoMagma
#         Skips the test if the device is a CUDA device and MAGMA is not installed
#     - @skipCUDAIfRocm
#         Skips the test if the device is a CUDA device and ROCm is being used


# Note [Adding a Device Type]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To add a device type:
#
#   (1) Create a new "TestBase" extending DeviceTypeTestBase.
#       See CPUTestBase and CUDATestBase below.
#   (2) Define the "device_type" attribute of the base to be the
#       appropriate string.
#   (3) Add logic to this file that appends your base class to
#       device_type_test_bases when your device type is available.
#   (4) (Optional) Write setUpClass/tearDownClass class methods that
#       instantiate dependencies (see MAGMA in CUDATestBase).
#   (5) (Optional) Override the "instantiate_test" method for total
#       control over how your class creates tests.
#
# setUpClass is called AFTER tests have been created and BEFORE and ONLY IF
# they are run. This makes it useful for initializing devices and dependencies.


# Note [Overriding methods in generic tests]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Device generic tests look a lot like normal test classes, but they differ
# from ordinary classes in some important ways.  In particular, overriding
# methods in generic tests doesn't work quite the way you expect.
#
#     class TestFooDeviceType(TestCase):
#         # Intention is to override
#         def assertEqual(self, x, y):
#             # This DOESN'T WORK!
#             super(TestFooDeviceType, self).assertEqual(x, y)
#
# If you try to run this code, you'll get an error saying that TestFooDeviceType
# is not in scope.  This is because after instantiating our classes, we delete
# it from the parent scope.  Instead, you need to hardcode a direct invocation
# of the desired subclass call, e.g.,
#
#     class TestFooDeviceType(TestCase):
#         # Intention is to override
#         def assertEqual(self, x, y):
#             TestCase.assertEqual(x, y)
#
# However, a less error-prone way of customizing the behavior of TestCase
# is to either (1) add your functionality to TestCase and make it toggled
# by a class attribute, or (2) create your own subclass of TestCase, and
# then inherit from it for your generic test.


def _dtype_test_suffix(dtypes):
    """ Returns the test suffix for a dtype, sequence of dtypes, or None. """
    if isinstance(dtypes, list) or isinstance(dtypes, tuple):
        if len(dtypes) == 0:
            return ''
        return '_' + '_'.join((dtype_name(d) for d in dtypes))
    elif dtypes:
        return '_{}'.format(dtype_name(dtypes))
    else:
        return ''


def _update_param_kwargs(param_kwargs, name, value):
    """ Adds a kwarg with the specified name and value to the param_kwargs dict. """
    if isinstance(value, list) or isinstance(value, tuple):
        # Make name plural (e.g. devices / dtypes) if the value is composite.
        param_kwargs['{}s'.format(name)] = value
    elif value:
        param_kwargs[name] = value

    # Leave param_kwargs as-is when value is None.


class DeviceTypeTestBase(TestCase):
    device_type: str = 'generic_device_type'

    # Flag to disable test suite early due to unrecoverable error such as CUDA error.
    _stop_test_suite = False

    # Precision is a thread-local setting since it may be overridden per test
    _tls = threading.local()
    _tls.precision = TestCase._precision
    _tls.rel_tol = TestCase._rel_tol

    @property
    def precision(self):
        return self._tls.precision

    @precision.setter
    def precision(self, prec):
        self._tls.precision = prec

    @property
    def rel_tol(self):
        return self._tls.rel_tol

    @rel_tol.setter
    def rel_tol(self, prec):
        self._tls.rel_tol = prec

    # Returns a string representing the device that single device tests should use.
    # Note: single device tests use this device exclusively.
    @classmethod
    def get_primary_device(cls):
        return cls.device_type

    # Returns a list of strings representing all available devices of this
    # device type. The primary device must be the first string in the list
    # and the list must contain no duplicates.
    # Note: UNSTABLE API. Will be replaced once PyTorch has a device generic
    #   mechanism of acquiring all available devices.
    @classmethod
    def get_all_devices(cls):
        return [cls.get_primary_device()]

    # Returns the dtypes the test has requested.
    # Prefers device-specific dtype specifications over generic ones.
    @classmethod
    def _get_dtypes(cls, test):
        if not hasattr(test, 'dtypes'):
            return None
        return test.dtypes.get(cls.device_type, test.dtypes.get('all', None))

    def _get_precision_override(self, test, dtype):
        if not hasattr(test, 'precision_overrides'):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)

    def _get_tolerance_override(self, test, dtype):
        if not hasattr(test, 'tolerance_overrides'):
            return self.precision, self.rel_tol
        return test.tolerance_overrides.get(dtype, tol(self.precision, self.rel_tol))

    def _apply_precision_override_for_test(self, test, param_kwargs):
        dtype = param_kwargs['dtype'] if 'dtype' in param_kwargs else None
        dtype = param_kwargs['dtypes'] if 'dtypes' in param_kwargs else dtype
        if dtype:
            self.precision = self._get_precision_override(test, dtype)
            self.precision, self.rel_tol = self._get_tolerance_override(test, dtype)

    # Creates device-specific tests.
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):

        def instantiate_test_helper(cls, name, *, test, param_kwargs=None):
            # Constructs the test
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                # Add the device param kwarg if the test needs device or devices.
                param_kwargs = {} if param_kwargs is None else param_kwargs
                test_sig_params = inspect.signature(test).parameters
                if 'device' in test_sig_params or 'devices' in test_sig_params:
                    device_arg: str = cls.get_primary_device()
                    if hasattr(test, 'num_required_devices'):
                        device_arg = cls.get_all_devices()
                    _update_param_kwargs(param_kwargs, 'device', device_arg)

                # Sets precision and runs test
                # Note: precision is reset after the test is run
                guard_precision = self.precision
                guard_rel_tol = self.rel_tol
                try:
                    self._apply_precision_override_for_test(test, param_kwargs)
                    result = test(self, **param_kwargs)
                except RuntimeError as rte:
                    # check if rte should stop entire test suite.
                    self._stop_test_suite = self._should_stop_test_suite()
                    # raise the runtime error as is for the test suite to record.
                    raise rte
                finally:
                    self.precision = guard_precision
                    self.rel_tol = guard_rel_tol

                return result

            assert not hasattr(cls, name), "Redefinition of test {0}".format(name)
            setattr(cls, name, instantiated_test)

        # Handles tests that need parametrization (e.g. those that run across a set of
        # ops / modules using the @ops or @modules decorators).

        def default_parametrize_fn(test, generic_cls, cls):
            # By default, parametrize only over device.
            test_suffix = cls.device_type
            yield (test, test_suffix, {})

        parametrize_fn = test.parametrize_fn if hasattr(test, 'parametrize_fn') else default_parametrize_fn
        for (test, test_suffix, param_kwargs) in parametrize_fn(test, generic_cls, cls):
            if hasattr(test, 'handles_dtypes') and test.handles_dtypes:
                full_name = '{}_{}'.format(name, test_suffix)
                instantiate_test_helper(cls=cls, name=full_name, test=test, param_kwargs=param_kwargs)
            else:
                # The parametrize_fn doesn't handle dtypes internally; handle them here instead by generating
                # a test per dtype.
                dtypes = cls._get_dtypes(test)
                dtypes = tuple(dtypes) if dtypes is not None else (None,)
                for dtype in dtypes:
                    all_param_kwargs = dict(param_kwargs)
                    _update_param_kwargs(all_param_kwargs, 'dtype', dtype)
                    full_name = '{}_{}{}'.format(name, test_suffix, _dtype_test_suffix(dtype))
                    instantiate_test_helper(cls=cls, name=full_name, test=test, param_kwargs=all_param_kwargs)

    def run(self, result=None):
        super().run(result=result)
        # Early terminate test if _stop_test_suite is set.
        if self._stop_test_suite:
            result.stop()


class CPUTestBase(DeviceTypeTestBase):
    device_type = 'cpu'

    # No critical error should stop CPU test suite
    def _should_stop_test_suite(self):
        return False

# The meta device represents tensors that don't have any storage; they have
# all metadata (size, dtype, strides) but they don't actually do any compute
class MetaTestBase(DeviceTypeTestBase):
    device_type = 'meta'
    _ignore_not_implemented_error = True

    def _should_stop_test_suite(self):
        return False

class CUDATestBase(DeviceTypeTestBase):
    device_type = 'cuda'
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    primary_device: ClassVar[str]
    cudnn_version: ClassVar[Any]
    no_magma: ClassVar[bool]
    no_cudnn: ClassVar[bool]

    def has_cudnn(self):
        return not self.no_cudnn

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = torch.cuda.device_count()

        prim_device = cls.get_primary_device()
        cuda_str = 'cuda:{0}'
        non_primary_devices = [cuda_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        # has_magma shows up after cuda is initialized
        t = torch.ones(1).cuda()
        cls.no_magma = not torch.cuda.has_magma

        # Determines if cuDNN is available and its version
        cls.no_cudnn = not torch.backends.cudnn.is_acceptable(t)
        cls.cudnn_version = None if cls.no_cudnn else torch.backends.cudnn.version()

        # Acquires the current device as the primary (test) device
        cls.primary_device = 'cuda:{0}'.format(torch.cuda.current_device())


# Adds available device-type-specific test base classes
def get_device_type_test_bases():
    # set type to List[Any] due to mypy list-of-union issue:
    # https://github.com/python/mypy/issues/3351
    test_bases: List[Any] = list()

    if IS_SANDCASTLE or IS_FBCODE:
        if IS_REMOTE_GPU:
            # Skip if sanitizer is enabled
            if not TEST_WITH_ASAN and not TEST_WITH_TSAN and not TEST_WITH_UBSAN:
                test_bases.append(CUDATestBase)
        else:
            test_bases.append(CPUTestBase)
            test_bases.append(MetaTestBase)
    else:
        test_bases.append(CPUTestBase)
        if not TEST_SKIP_NOARCH:
            test_bases.append(MetaTestBase)
        if torch.cuda.is_available():
            test_bases.append(CUDATestBase)

    return test_bases


device_type_test_bases = get_device_type_test_bases()


def filter_desired_device_types(device_type_test_bases, except_for=None, only_for=None):
    # device type cannot appear in both except_for and only_for
    intersect = set(except_for if except_for else []) & set(only_for if only_for else [])
    assert not intersect, f"device ({intersect}) appeared in both except_for and only_for"

    if except_for:
        device_type_test_bases = filter(
            lambda x: x.device_type not in except_for, device_type_test_bases)
    if only_for:
        device_type_test_bases = filter(
            lambda x: x.device_type in only_for, device_type_test_bases)

    return list(device_type_test_bases)


# Note [How to extend DeviceTypeTestBase to add new test device]
# The following logic optionally allows downstream projects like pytorch/xla to
# add more test devices.
# Instructions:
#  - Add a python file (e.g. pytorch/xla/test/pytorch_test_base.py) in downstream project.
#    - Inside the file, one should inherit from `DeviceTypeTestBase` class and define
#      a new DeviceTypeTest class (e.g. `XLATestBase`) with proper implementation of
#      `instantiate_test` method.
#    - DO NOT import common_device_type inside the file.
#      `runpy.run_path` with `globals()` already properly setup the context so that
#      `DeviceTypeTestBase` is already available.
#    - Set a top-level variable `TEST_CLASS` equal to your new class.
#      E.g. TEST_CLASS = XLATensorBase
#  - To run tests with new device type, set `TORCH_TEST_DEVICE` env variable to path
#    to this file. Multiple paths can be separated by `:`.
# See pytorch/xla/test/pytorch_test_base.py for a more detailed example.
_TORCH_TEST_DEVICES = os.environ.get('TORCH_TEST_DEVICES', None)
if _TORCH_TEST_DEVICES:
    for path in _TORCH_TEST_DEVICES.split(':'):
        # runpy (a stdlib module) lacks annotations
        mod = runpy.run_path(path, init_globals=globals())  # type: ignore[func-returns-value]
        device_type_test_bases.append(mod['TEST_CLASS'])


PYTORCH_CUDA_MEMCHECK = os.getenv('PYTORCH_CUDA_MEMCHECK', '0') == '1'

PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY = 'PYTORCH_TESTING_DEVICE_ONLY_FOR'
PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY = 'PYTORCH_TESTING_DEVICE_EXCEPT_FOR'


# Adds 'instantiated' device-specific test cases to the given scope.
# The tests in these test cases are derived from the generic tests in
# generic_test_class.
# See note "Generic Device Type Testing."
def instantiate_device_type_tests(generic_test_class, scope, except_for=None, only_for=None):
    # Removes the generic test class from its enclosing scope so its tests
    # are not discoverable.
    del scope[generic_test_class.__name__]

    # Creates an 'empty' version of the generic_test_class
    # Note: we don't inherit from the generic_test_class directly because
    #   that would add its tests to our test classes and they would be
    #   discovered (despite not being runnable). Inherited methods also
    #   can't be removed later, and we can't rely on load_tests because
    #   pytest doesn't support it (as of this writing).
    empty_name = generic_test_class.__name__ + "_base"
    empty_class = type(empty_name, generic_test_class.__bases__, {})

    # Acquires members names
    # See Note [Overriding methods in generic tests]
    generic_members = set(generic_test_class.__dict__.keys()) - set(empty_class.__dict__.keys())
    generic_tests = [x for x in generic_members if x.startswith('test')]

    # Filter out the device types based on user inputs
    desired_device_type_test_bases = filter_desired_device_types(device_type_test_bases,
                                                                 except_for, only_for)

    def split_if_not_empty(x: str):
        return x.split(",") if len(x) != 0 else []

    # Filter out the device types based on environment variables if available
    # Usage:
    # export PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda,cpu
    # export PYTORCH_TESTING_DEVICE_EXCEPT_FOR=xla
    env_only_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, ''))
    env_except_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, ''))

    desired_device_type_test_bases = filter_desired_device_types(desired_device_type_test_bases,
                                                                 env_except_for, env_only_for)


    # Creates device-specific test cases
    for base in desired_device_type_test_bases:
        # Special-case for ROCm testing -- only test for 'cuda' i.e. ROCm device by default
        # The except_for and only_for cases were already checked above. At this point we only need to check 'cuda'.
        if TEST_WITH_ROCM and base.device_type != 'cuda':
            continue

        class_name = generic_test_class.__name__ + base.device_type.upper()

        # type set to Any and suppressed due to unsupport runtime class:
        # https://github.com/python/mypy/wiki/Unsupported-Python-Features
        device_type_test_class: Any = type(class_name, (base, empty_class), {})

        for name in generic_members:
            if name in generic_tests:  # Instantiates test member
                test = getattr(generic_test_class, name)
                # XLA-compat shim (XLA's instantiate_test takes doesn't take generic_cls)
                sig = inspect.signature(device_type_test_class.instantiate_test)
                if len(sig.parameters) == 3:
                    # Instantiates the device-specific tests
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test), generic_cls=generic_test_class)
                else:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test))
            else:  # Ports non-test member
                assert name not in device_type_test_class.__dict__, "Redefinition of directly defined member {0}".format(name)
                nontest = getattr(generic_test_class, name)
                setattr(device_type_test_class, name, nontest)

        # Mimics defining the instantiated class in the caller's file
        # by setting its module to the given class's and adding
        # the module to the given scope.
        # This lets the instantiated class be discovered by unittest.
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class


# Category of dtypes to run an OpInfo-based test for
# Example use: @ops(dtype=OpDTypes.supported)
#
# There are 6 categories:
# - basic: The dtypes the operator wants to be tested on by default. This will be
#          a subset of the types supported by the operator.
# - supported: Every dtype supported by the operator. Use for exhaustive
#              testing of all dtypes.
# - unsupported: Run tests on dtypes not supported by the operator. e.g. for
#                testing the operator raises an error and doesn't crash.
# - supported_backward: Every dtype supported by the operator's backward pass.
# - unsupported_backward: Run tests on dtypes not supported by the operator's backward pass.
# - none: Useful for tests that are not dtype-specific. No dtype will be passed to the test
#         when this is selected.
class OpDTypes(Enum):
    basic = 0  # Test the basic set of dtypes (default)
    supported = 1  # Test all supported dtypes
    unsupported = 2  # Test only unsupported dtypes
    supported_backward = 3  # Test all supported backward dtypes
    unsupported_backward = 4  # Test only unsupported backward dtypes
    none = 5  # Instantiate no dtype variants (no dtype kwarg needed)


# Decorator that defines the OpInfos a test template should be instantiated for.
#
# Example usage:
#
# @ops(unary_ufuncs)
# def test_numerics(self, device, dtype, op):
#   <test_code>
#
# This will instantiate variants of test_numerics for each given OpInfo,
# on each device the OpInfo's operator supports, and for every dtype supported by
# that operator. There are a few caveats to the dtype rule, explained below.
#
# First, if the OpInfo defines "default_test_dtypes" then the test
# is instantiated for the intersection of default_test_dtypes and the
# dtypes the operator supports. Second, the @ops decorator can accept two
# additional arguments, "dtypes" and "allowed_dtypes". If "dtypes" is specified
# then the test variants are instantiated for those dtypes, regardless of
# what the operator supports. If given "allowed_dtypes" then test variants
# are instantiated only for the intersection of allowed_dtypes and the dtypes
# they would otherwise be instantiated with. That is, allowed_dtypes composes
# with the options listed above and below.
#
# The "dtypes" argument can also accept additional values (see OpDTypes above):
#   OpDTypes.supported - the test is instantiated for all dtypes the operator
#     supports
#   OpDTypes.unsupported - the test is instantiated for all dtypes the operator
#     doesn't support
#   OpDTypes.supported_backward - the test is instantiated for all dtypes the
#     operator's gradient formula supports
#   OpDTypes.unsupported_backward - the test is instantiated for all dtypes the
#     operator's gradient formula doesn't support
#   OpDTypes.none - the test is instantied without any dtype. The test signature
#     should not include a dtype kwarg in this case.
#
# These options allow tests to have considerable control over the dtypes
#   they're instantiated for. Finally, the @dtypes decorator composes with the
#   @ops decorator, and works the same as the "dtypes" argument to @ops.

class ops(_TestParametrizer):
    def __init__(self, op_list, *, dtypes: OpDTypes = OpDTypes.basic,
                 allowed_dtypes: Optional[Sequence[torch.dtype]] = None):
        super().__init__(handles_dtypes=True)
        self.op_list = op_list
        self.opinfo_dtypes = dtypes
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None

    def _parametrize_test(self, test, generic_cls, device_cls):
        """ Parameterizes the given test function across each op and its associated dtypes. """
        for op in self.op_list:
            # Acquires dtypes, using the op data if unspecified
            dtypes = device_cls._get_dtypes(test)
            if dtypes is None:
                if self.opinfo_dtypes == OpDTypes.unsupported_backward:
                    dtypes = set(get_all_dtypes()).difference(op.supported_backward_dtypes(device_cls.device_type))
                elif self.opinfo_dtypes == OpDTypes.supported_backward:
                    dtypes = op.supported_backward_dtypes(device_cls.device_type)
                elif self.opinfo_dtypes == OpDTypes.unsupported:
                    dtypes = set(get_all_dtypes()).difference(op.supported_dtypes(device_cls.device_type))
                elif self.opinfo_dtypes == OpDTypes.supported:
                    dtypes = op.supported_dtypes(device_cls.device_type)
                elif self.opinfo_dtypes == OpDTypes.basic:
                    dtypes = op.default_test_dtypes(device_cls.device_type)
                elif self.opinfo_dtypes == OpDTypes.none:
                    dtypes = [None]
                else:
                    raise RuntimeError(f"Unknown OpDType: {self.opinfo_dtypes}")

                if self.allowed_dtypes is not None:
                    dtypes = dtypes.intersection(self.allowed_dtypes)
            else:
                assert self.allowed_dtypes is None, "ops(allowed_dtypes=[...]) and the dtypes decorator are incompatible"
                assert self.opinfo_dtypes == OpDTypes.basic, "ops(dtypes=...) and the dtypes decorator are incompatible"

            for dtype in dtypes:
                # Construct the test name.
                test_name = '{}{}_{}{}'.format(op.name.replace('.', '_'),
                                               '_' + op.variant_test_name if op.variant_test_name else '',
                                               device_cls.device_type,
                                               _dtype_test_suffix(dtype))

                # Construct parameter kwargs to pass to the test.
                param_kwargs = {'op': op}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)

                # Wraps instantiated test with op decorators
                # NOTE: test_wrapper exists because we don't want to apply
                #   op-specific decorators to the original test.
                #   Test-specific decorators are applied to the original test,
                #   however.
                try:
                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    for decorator in op.get_decorators(
                            generic_cls.__name__, test.__name__, device_cls.device_type, dtype):
                        test_wrapper = decorator(test_wrapper)

                    yield (test_wrapper, test_name, param_kwargs)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    print("Failed to instantiate {0} for op {1}!".format(test_name, op.name))
                    raise ex

# Decorator that skips a test if the given condition is true.
# Notes:
#   (1) Skip conditions stack.
#   (2) Skip conditions can be bools or strings. If a string the
#       test base must have defined the corresponding attribute to be False
#       for the test to run. If you want to use a string argument you should
#       probably define a new decorator instead (see below).
#   (3) Prefer the existing decorators to defining the 'device_type' kwarg.
class skipIf(object):

    def __init__(self, dep, reason, device_type=None):
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def dep_fn(slf, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                if (isinstance(self.dep, str) and getattr(slf, self.dep, True)) or (isinstance(self.dep, bool) and self.dep):
                    raise unittest.SkipTest(self.reason)

            return fn(slf, *args, **kwargs)
        return dep_fn


# Skips a test on CPU if the condition is true.
class skipCPUIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='cpu')


# Skips a test on CUDA if the condition is true.
class skipCUDAIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='cuda')

# Skips a test on Meta if the condition is true.
class skipMetaIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='meta')

def _has_sufficient_memory(device, size):
    if torch.device(device).type == 'cuda':
        if not torch.cuda.is_available():
            return False
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device) >= size

    if device == 'xla':
        raise unittest.SkipTest('TODO: Memory availability checks for XLA?')

    if device != 'cpu':
        raise unittest.SkipTest('Unknown device type')

    # CPU
    if not HAS_PSUTIL:
        raise unittest.SkipTest('Need psutil to determine if memory is sufficient')

    # The sanitizers have significant memory overheads
    if TEST_WITH_ASAN or TEST_WITH_TSAN or TEST_WITH_UBSAN:
        effective_size = size * 10
    else:
        effective_size = size

    if psutil.virtual_memory().available < effective_size:
        gc.collect()
    return psutil.virtual_memory().available >= effective_size


def largeTensorTest(size, device=None):
    """Skip test if the device has insufficient memory to run the test

    size may be a number of bytes, a string of the form "N GB", or a callable

    If the test is a device generic test, available memory on the primary device will be checked.
    It can also be overriden by the optional `device=` argument.
    In other tests, the `device=` argument needs to be specified.
    """
    if isinstance(size, str):
        assert size.endswith("GB") or size.endswith("gb"), "only bytes or GB supported"
        size = 1024 ** 3 * int(size[:-2])

    def inner(fn):
        @wraps(fn)
        def dep_fn(self, *args, **kwargs):
            size_bytes = size(self, *args, **kwargs) if callable(size) else size
            _device = device if device is not None else self.get_primary_device()
            if not _has_sufficient_memory(_device, size_bytes):
                raise unittest.SkipTest('Insufficient {} memory'.format(_device))

            return fn(self, *args, **kwargs)
        return dep_fn
    return inner


class expectedFailure(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def efail_fn(slf, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                try:
                    fn(slf, *args, **kwargs)
                except Exception:
                    return
                else:
                    slf.fail('expected test to fail, but it passed')

            return fn(slf, *args, **kwargs)
        return efail_fn


class onlyOn(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def only_fn(slf, *args, **kwargs):
            if self.device_type != slf.device_type:
                reason = "Only runs on {0}".format(self.device_type)
                raise unittest.SkipTest(reason)

            return fn(slf, *args, **kwargs)

        return only_fn


# Decorator that provides all available devices of the device type to the test
# as a list of strings instead of providing a single device string.
# Skips the test if the number of available devices of the variant's device
# type is less than the 'num_required_devices' arg.
class deviceCountAtLeast(object):

    def __init__(self, num_required_devices):
        self.num_required_devices = num_required_devices

    def __call__(self, fn):
        assert not hasattr(fn, 'num_required_devices'), "deviceCountAtLeast redefinition for {0}".format(fn.__name__)
        fn.num_required_devices = self.num_required_devices

        @wraps(fn)
        def multi_fn(slf, devices, *args, **kwargs):
            if len(devices) < self.num_required_devices:
                reason = "fewer than {0} devices detected".format(self.num_required_devices)
                raise unittest.SkipTest(reason)

            return fn(slf, devices, *args, **kwargs)

        return multi_fn

# Only runs the test on the CPU and CUDA (the native device types)
def onlyOnCPUAndCUDA(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type != 'cpu' and self.device_type != 'cuda':
            reason = "onlyOnCPUAndCUDA: doesn't run on {0}".format(self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn

# Specifies per-dtype precision overrides.
# Ex.
#
# @precisionOverride({torch.half : 1e-2, torch.float : 1e-4})
# @dtypes(torch.half, torch.float, torch.double)
# def test_X(self, device, dtype):
#   ...
#
# When the test is instantiated its class's precision will be set to the
# corresponding override, if it exists.
# self.precision can be accessed directly, and it also controls the behavior of
# functions like self.assertEqual().
#
# Note that self.precision is a scalar value, so if you require multiple
# precisions (or are working with multiple dtypes) they should be specified
# explicitly and computed using self.precision (e.g.
# self.precision *2, max(1, self.precision)).
class precisionOverride(object):

    def __init__(self, d):
        assert isinstance(d, dict), "precisionOverride not given a dtype : precision dict!"
        for dtype, prec in d.items():
            assert isinstance(dtype, torch.dtype), "precisionOverride given unknown dtype {0}".format(dtype)

        self.d = d

    def __call__(self, fn):
        fn.precision_overrides = self.d
        return fn

# Specifies per-dtype tolerance overrides tol(atol, rtol). It has priority over
# precisionOverride.
# Ex.
#
# @toleranceOverride({torch.float : tol(atol=1e-2, rtol=1e-3},
#                     torch.double : tol{atol=1e-4, rtol = 0})
# @dtypes(torch.half, torch.float, torch.double)
# def test_X(self, device, dtype):
#   ...
#
# When the test is instantiated its class's tolerance will be set to the
# corresponding override, if it exists.
# self.rtol and self.precision can be accessed directly, and they also control
# the behavior of functions like self.assertEqual().
#
# The above example sets atol = 1e-2 and rtol = 1e-3 for torch.float and
# atol = 1e-4 and rtol = 0 for torch.double.
tol = namedtuple('tol', ['atol', 'rtol'])

class toleranceOverride(object):
    def __init__(self, d):
        assert isinstance(d, dict), "toleranceOverride not given a dtype : tol dict!"
        for dtype, prec in d.items():
            assert isinstance(dtype, torch.dtype), "toleranceOverride given unknown dtype {0}".format(dtype)
            assert isinstance(prec, tol), "toleranceOverride not given a dtype : tol dict!"

        self.d = d

    def __call__(self, fn):
        fn.tolerance_overrides = self.d
        return fn

# Decorator that instantiates a variant of the test for each given dtype.
# Notes:
#   (1) Tests that accept the dtype argument MUST use this decorator.
#   (2) Can be overridden for the CPU or CUDA, respectively, using dtypesIfCPU
#       or dtypesIfCUDA.
#   (3) Can accept an iterable of dtypes or an iterable of tuples
#       of dtypes.
# Examples:
# @dtypes(torch.float32, torch.float64)
# @dtypes((torch.long, torch.float32), (torch.int, torch.float64))
class dtypes(object):

    def __init__(self, *args, device_type="all"):
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for arg in args:
                assert isinstance(arg, (list, tuple)), \
                    "When one dtype variant is a tuple or list, " \
                    "all dtype variants must be. " \
                    "Received non-list non-tuple dtype {0}".format(str(arg))
                assert all(isinstance(dtype, torch.dtype) for dtype in arg), "Unknown dtype in {0}".format(str(arg))
        else:
            assert all(isinstance(arg, torch.dtype) for arg in args), "Unknown dtype in {0}".format(str(args))

        self.args = args
        self.device_type = device_type

    def __call__(self, fn):
        d = getattr(fn, 'dtypes', {})
        assert self.device_type not in d, "dtypes redefinition for {0}".format(self.device_type)
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn


# Overrides specified dtypes on the CPU.
class dtypesIfCPU(dtypes):

    def __init__(self, *args):
        super().__init__(*args, device_type='cpu')


# Overrides specified dtypes on CUDA.
class dtypesIfCUDA(dtypes):

    def __init__(self, *args):
        super().__init__(*args, device_type='cuda')


def onlyCPU(fn):
    return onlyOn('cpu')(fn)


def onlyCUDA(fn):
    return onlyOn('cuda')(fn)


def expectedFailureCUDA(fn):
    return expectedFailure('cuda')(fn)

def expectedFailureMeta(fn):
    return expectedFailure('meta')(fn)

class expectedAlertNondeterministic:
    def __init__(self, caller_name, device_type=None, fn_has_device_arg=True):
        self.device_type = device_type
        self.error_message = caller_name + ' does not have a deterministic implementation, but you set'
        self.fn_has_device_arg = fn_has_device_arg

    def __call__(self, fn):
        @wraps(fn)
        def efail_fn(slf, device, *args, **kwargs):
            with DeterministicGuard(True):
                # If a nondeterministic error is expected for this case,
                # check that it is raised
                if self.device_type is None or self.device_type == slf.device_type:
                    try:
                        if self.fn_has_device_arg:
                            fn(slf, device, *args, **kwargs)
                        else:
                            fn(slf, *args, **kwargs)
                    except RuntimeError as e:
                        if self.error_message not in str(e):
                            slf.fail(
                                'expected non-deterministic error message to start with "'
                                + self.error_message
                                + '" but got this instead: "' + str(e) + '"')
                        return
                    else:
                        slf.fail('expected a non-deterministic error, but it was not raised')

                # If a nondeterministic error is not expected for this case,
                # make sure that it is not raised
                try:
                    if self.fn_has_device_arg:
                        return fn(slf, device, *args, **kwargs)
                    else:
                        return fn(slf, *args, **kwargs)
                except RuntimeError as e:
                    if 'does not have a deterministic implementation' in str(e):
                        slf.fail(
                            'did not expect non-deterministic error message, '
                            + 'but got this: "' + str(e) + '"')
                    # Reraise exceptions unrelated to nondeterminism
                    raise

        @wraps(fn)
        def efail_fn_no_device(slf, *args, **kwargs):
            return efail_fn(slf, None, *args, **kwargs)

        if self.fn_has_device_arg:
            return efail_fn
        else:
            return efail_fn_no_device

# Skips a test on CPU if LAPACK is not available.
def skipCPUIfNoLapack(fn):
    return skipCPUIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)


# Skips a test on CPU if FFT is not available.
def skipCPUIfNoFFT(fn):
    return skipCPUIf(not torch._C.has_spectral, "PyTorch is built without FFT support")(fn)


# Skips a test on CPU if MKL is not available.
def skipCPUIfNoMkl(fn):
    return skipCPUIf(not TEST_MKL, "PyTorch is built without MKL support")(fn)


# Skips a test on CUDA if MAGMA is not available.
def skipCUDAIfNoMagma(fn):
    return skipCUDAIf('no_magma', "no MAGMA library detected")(skipCUDANonDefaultStreamIf(True)(fn))

# Skips a test on CUDA if cuSOLVER is not available
def skipCUDAIfNoCusolver(fn):
    version = _get_torch_cuda_version()
    return skipCUDAIf(version < (10, 2), "cuSOLVER not available")(fn)

# Skips a test if both cuSOLVER and MAGMA are not available
def skipCUDAIfNoMagmaAndNoCusolver(fn):
    version = _get_torch_cuda_version()
    if version >= (10, 2):
        return fn
    else:
        # cuSolver is disabled on cuda < 10.1.243, tests depend on MAGMA
        return skipCUDAIfNoMagma(fn)

# Skips a test on CUDA when using ROCm.
def skipCUDAIfRocm(fn):
    return skipCUDAIf(TEST_WITH_ROCM, "test doesn't currently work on the ROCm stack")(fn)

# Skips a test on CUDA when not using ROCm.
def skipCUDAIfNotRocm(fn):
    return skipCUDAIf(not TEST_WITH_ROCM, "test doesn't currently work on the CUDA stack")(fn)

# Skips a test on CUDA if ROCm is unavailable or its version is lower than requested.
def skipCUDAIfRocmVersionLessThan(version=None):

    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if self.device_type == 'cuda':
                if not TEST_WITH_ROCM:
                    reason = "ROCm not available"
                    raise unittest.SkipTest(reason)
                rocm_version = str(torch.version.hip)
                rocm_version = rocm_version.split("-")[0]    # ignore git sha
                rocm_version_tuple = tuple(int(x) for x in rocm_version.split("."))
                if rocm_version_tuple is None or version is None or rocm_version_tuple < tuple(version):
                    reason = "ROCm {0} is available but {1} required".format(rocm_version_tuple, version)
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn
    return dec_fn

# Skips a test on CUDA when using ROCm.
def skipCUDAIfNotMiopenSuggestNHWC(fn):
    return skipCUDAIf(not TEST_WITH_MIOPEN_SUGGEST_NHWC, "test doesn't currently work without MIOpen NHWC activation")(fn)

# Skips a test for specified CUDA versions, given in the form of a list of [major, minor]s.
def skipCUDAVersionIn(versions : List[Tuple[int, int]] = None):
    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            version = _get_torch_cuda_version()
            if version == (0, 0):  # cpu
                return fn(self, *args, **kwargs)
            if version in (versions or []):
                reason = "test skipped for CUDA version {0}".format(version)
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)

        return wrap_fn
    return dec_fn

# Skips a test on CUDA if cuDNN is unavailable or its version is lower than requested.
def skipCUDAIfCudnnVersionLessThan(version=0):

    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if self.device_type == 'cuda':
                if self.no_cudnn:
                    reason = "cuDNN not available"
                    raise unittest.SkipTest(reason)
                if self.cudnn_version is None or self.cudnn_version < version:
                    reason = "cuDNN version {0} is available but {1} required".format(self.cudnn_version, version)
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn
    return dec_fn


def skipCUDAIfNoCudnn(fn):
    return skipCUDAIfCudnnVersionLessThan(0)(fn)

def skipMeta(fn):
    return skipMetaIf(True, "test doesn't work with meta tensors")(fn)
