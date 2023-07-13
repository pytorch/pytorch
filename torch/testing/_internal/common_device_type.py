import copy
import gc
import inspect
import runpy
import sys
import threading
from collections import namedtuple
from enum import Enum
from functools import wraps, partial
from typing import List, Any, ClassVar, Optional, Sequence, Tuple, Union, Dict, Set
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
    skipCUDANonDefaultStreamIf, TEST_WITH_ASAN, TEST_WITH_UBSAN, TEST_WITH_TSAN, \
    IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, IS_WINDOWS, TEST_MPS, \
    _TestParametrizer, compose_parametrize_fns, dtype_name, \
    NATIVE_DEVICES, skipIfTorchDynamo
from torch.testing._internal.common_cuda import _get_torch_cuda_version, \
    TEST_CUSPARSE_GENERIC, TEST_HIPSPARSE_GENERIC, _get_torch_rocm_version
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
#   operators. It's also useful for controlling which tests are run. For example,
#   only tests that use a CUDA device can be run on platforms with CUDA.
#   Let's dive in with an example to get an idea for how it works:
#
# --------------------------------------------------------
# A template class (looks like a regular unittest TestCase)
# class TestClassFoo(TestCase):
#
#   # A template test that can be specialized with a device
#   # NOTE: this test case is not runnable by unittest or pytest because it
#   #   accepts an extra positional argument, "device", that they do not understand
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
# These instantiated test classes ARE discoverable and runnable by both
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
# In addition to parametrizing over device, dtype, and ops via OpInfos, the
#   @parametrize decorator is supported for arbitrary parametrizations:
# --------------------------------------------------------
# # A template test that can be specialized with a device, dtype, and value for x
# @parametrize("x", range(5))
# def test_car(self, device, dtype, x)
#   pass
# --------------------------------------------------------
#
# See the documentation for @parametrize in common_utils.py for additional details
#   on this. Note that the instantiate_device_type_tests() function will handle
#   such parametrizations; there is no need to additionally call
#   instantiate_parametrized_tests().
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
#         If there are fewer devices than the value passed to the decorator
#         the test is skipped.
#     - @dtypes(<list of tuples of dtypes>)
#         In addition to accepting multiple dtypes, the @dtypes decorator
#         can accept a sequence of tuple pairs of dtypes. The test template
#         will be called with each tuple for its "dtype" argument.
#     - @onlyNativeDeviceTypes
#         Skips the test if the device is not a native device type (currently CPU, CUDA, Meta)
#     - @onlyCPU
#         Skips the test if the device is not a CPU device
#     - @onlyCUDA
#         Skips the test if the device is not a CUDA device
#     - @onlyMPS
#         Skips the test if the device is not a MPS device
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
#             super().assertEqual(x, y)
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
    if isinstance(dtypes, (list, tuple)):
        if len(dtypes) == 0:
            return ''
        return '_' + '_'.join((dtype_name(d) for d in dtypes))
    elif dtypes:
        return '_{}'.format(dtype_name(dtypes))
    else:
        return ''


def _update_param_kwargs(param_kwargs, name, value):
    """ Adds a kwarg with the specified name and value to the param_kwargs dict. """
    # Make name plural (e.g. devices / dtypes) if the value is composite.
    plural_name = '{}s'.format(name)

    # Clear out old entries of the arg if any.
    if name in param_kwargs:
        del param_kwargs[name]
    if plural_name in param_kwargs:
        del param_kwargs[plural_name]

    if isinstance(value, (list, tuple)):
        param_kwargs[plural_name] = value
    elif value is not None:
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

    @classmethod
    def _init_and_get_primary_device(cls):
        try:
            return cls.get_primary_device()
        except Exception:
            # For CUDATestBase, XLATestBase, and possibly others, the primary device won't be available
            # until setUpClass() sets it. Call that manually here if needed.
            if hasattr(cls, 'setUpClass'):
                cls.setUpClass()
            return cls.get_primary_device()

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

        default_dtypes = test.dtypes.get('all')
        msg = f"@dtypes is mandatory when using @dtypesIf however '{test.__name__}' didn't specify it"
        assert default_dtypes is not None, msg

        return test.dtypes.get(cls.device_type, default_dtypes)

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

        def instantiate_test_helper(cls, name, *, test, param_kwargs=None, decorator_fn=lambda _: []):
            # Add the device param kwarg if the test needs device or devices.
            param_kwargs = {} if param_kwargs is None else param_kwargs
            test_sig_params = inspect.signature(test).parameters
            if 'device' in test_sig_params or 'devices' in test_sig_params:
                device_arg: str = cls._init_and_get_primary_device()
                if hasattr(test, 'num_required_devices'):
                    device_arg = cls.get_all_devices()
                _update_param_kwargs(param_kwargs, 'device', device_arg)

            # Apply decorators based on param kwargs.
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            # Constructs the test
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
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
                    # Check if test has been decorated with `@expectedFailure`
                    # Using `__unittest_expecting_failure__` attribute, see
                    # https://github.com/python/cpython/blob/ffa505b580464/Lib/unittest/case.py#L164
                    # In that case, make it fail with "unexpected success" by suppressing exception
                    if getattr(test, "__unittest_expecting_failure__", False) and self._stop_test_suite:
                        import sys
                        print("Suppressing fatal exception to trigger unexpected success", file=sys.stderr)
                        return
                    # raise the runtime error as is for the test suite to record.
                    raise rte
                finally:
                    self.precision = guard_precision
                    self.rel_tol = guard_rel_tol

                return result

            assert not hasattr(cls, name), "Redefinition of test {0}".format(name)
            setattr(cls, name, instantiated_test)

        def default_parametrize_fn(test, generic_cls, device_cls):
            # By default, no parametrization is needed.
            yield (test, '', {}, lambda _: [])

        # Parametrization decorators set the parametrize_fn attribute on the test.
        parametrize_fn = getattr(test, "parametrize_fn", default_parametrize_fn)

        # If one of the @dtypes* decorators is present, also parametrize over the dtypes set by it.
        dtypes = cls._get_dtypes(test)
        if dtypes is not None:

            def dtype_parametrize_fn(test, generic_cls, device_cls, dtypes=dtypes):
                for dtype in dtypes:
                    param_kwargs: Dict[str, Any] = {}
                    _update_param_kwargs(param_kwargs, "dtype", dtype)

                    # Note that an empty test suffix is set here so that the dtype can be appended
                    # later after the device.
                    yield (test, '', param_kwargs, lambda _: [])

            parametrize_fn = compose_parametrize_fns(dtype_parametrize_fn, parametrize_fn)

        # Instantiate the parametrized tests.
        for (test, test_suffix, param_kwargs, decorator_fn) in parametrize_fn(test, generic_cls, cls):
            test_suffix = '' if test_suffix == '' else '_' + test_suffix
            device_suffix = '_' + cls.device_type

            # Note: device and dtype suffix placement
            # Special handling here to place dtype(s) after device according to test name convention.
            dtype_kwarg = None
            if 'dtype' in param_kwargs or 'dtypes' in param_kwargs:
                dtype_kwarg = param_kwargs['dtypes'] if 'dtypes' in param_kwargs else param_kwargs['dtype']
            test_name = '{}{}{}{}'.format(name, test_suffix, device_suffix, _dtype_test_suffix(dtype_kwarg))

            instantiate_test_helper(cls=cls, name=test_name, test=test, param_kwargs=param_kwargs,
                                    decorator_fn=decorator_fn)

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

# See Note [Lazy Tensor tests in device agnostic testing]
lazy_ts_backend_init = False
class LazyTestBase(DeviceTypeTestBase):
    device_type = 'lazy'

    def _should_stop_test_suite(self):
        return False

    @classmethod
    def setUpClass(cls):
        import torch._lazy
        import torch._lazy.metrics
        import torch._lazy.ts_backend
        global lazy_ts_backend_init
        if not lazy_ts_backend_init:
            # Need to connect the TS backend to lazy key before running tests
            torch._lazy.ts_backend.init()
            lazy_ts_backend_init = True

class MPSTestBase(DeviceTypeTestBase):
    device_type = 'mps'
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        # currently only one device is supported on MPS backend
        prim_device = cls.get_primary_device()
        return [prim_device]

    @classmethod
    def setUpClass(cls):
        cls.primary_device = 'mps:0'

    def _should_stop_test_suite(self):
        return False

class PrivateUse1TestBase(DeviceTypeTestBase):
    primary_device: ClassVar[str]
    device_mod = None
    device_type = 'privateuse1'

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = cls.device_mod.device_count()
        prim_device = cls.get_primary_device()
        device_str = f'{cls.device_type}:{{0}}'
        non_primary_devices = [device_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        cls.device_type = torch._C._get_privateuse1_backend_name()
        cls.device_mod = getattr(torch, cls.device_type, None)
        assert cls.device_mod is not None, f'''torch has no module of `{cls.device_type}`, you should register
                                            a module by `torch._register_device_module`.'''
        cls.primary_device = '{device_type}:{id}'.format(device_type=cls.device_type, id=cls.device_mod.current_device())

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
    else:
        test_bases.append(CPUTestBase)
        if torch.cuda.is_available():
            test_bases.append(CUDATestBase)
        device_type = torch._C._get_privateuse1_backend_name()
        device_mod = getattr(torch, device_type, None)
        if hasattr(device_mod, "is_available") and device_mod.is_available():
            test_bases.append(PrivateUse1TestBase)
        # Disable MPS testing in generic device testing temporarily while we're
        # ramping up support.
        # elif torch.backends.mps.is_available():
        #   test_bases.append(MPSTestBase)

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
# generic_test_class. This function should be used instead of
# instantiate_parametrized_tests() if the test class contains
# device-specific tests (NB: this supports additional @parametrize usage).
#
# See note "Writing Test Templates"
def instantiate_device_type_tests(generic_test_class, scope, except_for=None, only_for=None, include_lazy=False, allow_mps=False):
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

    # allow callers to specifically opt tests into being tested on MPS, similar to `include_lazy`
    test_bases = device_type_test_bases.copy()
    if allow_mps and TEST_MPS and MPSTestBase not in test_bases:
        test_bases.append(MPSTestBase)
    # Filter out the device types based on user inputs
    desired_device_type_test_bases = filter_desired_device_types(test_bases, except_for, only_for)
    if include_lazy:
        # Note [Lazy Tensor tests in device agnostic testing]
        # Right now, test_view_ops.py runs with LazyTensor.
        # We don't want to opt every device-agnostic test into using the lazy device,
        # because many of them will fail.
        # So instead, the only way to opt a specific device-agnostic test file into
        # lazy tensor testing is with include_lazy=True
        if IS_FBCODE:
            print("TorchScript backend not yet supported in FBCODE/OVRSOURCE builds", file=sys.stderr)
        else:
            desired_device_type_test_bases.append(LazyTestBase)

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
# There are 5 categories:
# - supported: Every dtype supported by the operator. Use for exhaustive
#              testing of all dtypes.
# - unsupported: Run tests on dtypes not supported by the operator. e.g. for
#                testing the operator raises an error and doesn't crash.
# - supported_backward: Every dtype supported by the operator's backward pass.
# - unsupported_backward: Run tests on dtypes not supported by the operator's backward pass.
# - any_one: Runs a test for one dtype the operator supports. Prioritizes dtypes the
#     operator supports in both forward and backward.
# - none: Useful for tests that are not dtype-specific. No dtype will be passed to the test
#         when this is selected.
class OpDTypes(Enum):
    supported = 0  # Test all supported dtypes (default)
    unsupported = 1  # Test only unsupported dtypes
    supported_backward = 2  # Test all supported backward dtypes
    unsupported_backward = 3  # Test only unsupported backward dtypes
    any_one = 4  # Test precisely one supported dtype
    none = 5  # Instantiate no dtype variants (no dtype kwarg needed)
    any_common_cpu_cuda_one = 6  # Test precisely one supported dtype that is common to both cuda and cpu


# Arbitrary order
ANY_DTYPE_ORDER = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.long,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool
)

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
# The @ops decorator can accept two
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
#   OpDTypes.any_one - the test is instantiated for one dtype the
#     operator supports. The dtype supports forward and backward if possible.
#   OpDTypes.none - the test is instantiated without any dtype. The test signature
#     should not include a dtype kwarg in this case.
#
# These options allow tests to have considerable control over the dtypes
#   they're instantiated for.

class ops(_TestParametrizer):
    def __init__(self, op_list, *, dtypes: Union[OpDTypes, Sequence[torch.dtype]] = OpDTypes.supported,
                 allowed_dtypes: Optional[Sequence[torch.dtype]] = None):
        self.op_list = list(op_list)
        self.opinfo_dtypes = dtypes
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None

    def _parametrize_test(self, test, generic_cls, device_cls):
        """ Parameterizes the given test function across each op and its associated dtypes. """
        if device_cls is None:
            raise RuntimeError('The @ops decorator is only intended to be used in a device-specific '
                               'context; use it with instantiate_device_type_tests() instead of '
                               'instantiate_parametrized_tests()')

        op = check_exhausted_iterator = object()
        for op in self.op_list:
            # Determine the set of dtypes to use.
            dtypes: Union[Set[torch.dtype], Set[None]]
            if isinstance(self.opinfo_dtypes, Sequence):
                dtypes = set(self.opinfo_dtypes)
            elif self.opinfo_dtypes == OpDTypes.unsupported_backward:
                dtypes = set(get_all_dtypes()).difference(op.supported_backward_dtypes(device_cls.device_type))
            elif self.opinfo_dtypes == OpDTypes.supported_backward:
                dtypes = op.supported_backward_dtypes(device_cls.device_type)
            elif self.opinfo_dtypes == OpDTypes.unsupported:
                dtypes = set(get_all_dtypes()).difference(op.supported_dtypes(device_cls.device_type))
            elif self.opinfo_dtypes == OpDTypes.supported:
                dtypes = op.supported_dtypes(device_cls.device_type)
            elif self.opinfo_dtypes == OpDTypes.any_one:
                # Tries to pick a dtype that supports both forward or backward
                supported = op.supported_dtypes(device_cls.device_type)
                supported_backward = op.supported_backward_dtypes(device_cls.device_type)
                supported_both = supported.intersection(supported_backward)
                dtype_set = supported_both if len(supported_both) > 0 else supported
                for dtype in ANY_DTYPE_ORDER:
                    if dtype in dtype_set:
                        dtypes = {dtype}
                        break
                else:
                    dtypes = {}
            elif self.opinfo_dtypes == OpDTypes.any_common_cpu_cuda_one:
                # Tries to pick a dtype that supports both CPU and CUDA
                supported = op.dtypes.intersection(op.dtypesIfCUDA)
                if supported:
                    dtypes = {next(dtype for dtype in ANY_DTYPE_ORDER if dtype in supported)}
                else:
                    dtypes = {}

            elif self.opinfo_dtypes == OpDTypes.none:
                dtypes = {None}
            else:
                raise RuntimeError(f"Unknown OpDType: {self.opinfo_dtypes}")

            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)

            # Construct the test name; device / dtype parts are handled outside.
            # See [Note: device and dtype suffix placement]
            test_name = op.formatted_name

            for dtype in dtypes:
                # Construct parameter kwargs to pass to the test.
                param_kwargs = {'op': op}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)

                # NOTE: test_wrapper exists because we don't want to apply
                #   op-specific decorators to the original test.
                #   Test-specific decorators are applied to the original test,
                #   however.
                try:
                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    decorator_fn = partial(op.get_decorators, generic_cls.__name__,
                                           test.__name__, device_cls.device_type, dtype)

                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    print("Failed to instantiate {0} for op {1}!".format(test_name, op.name))
                    raise ex
        if op is check_exhausted_iterator:
            raise ValueError('An empty op_list was passed to @ops. '
                             'Note that this may result from reuse of a generator.')

# Decorator that skips a test if the given condition is true.
# Notes:
#   (1) Skip conditions stack.
#   (2) Skip conditions can be bools or strings. If a string the
#       test base must have defined the corresponding attribute to be False
#       for the test to run. If you want to use a string argument you should
#       probably define a new decorator instead (see below).
#   (3) Prefer the existing decorators to defining the 'device_type' kwarg.
class skipIf:

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

# Skips a test on MPS if the condition is true.
class skipMPSIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='mps')

# Skips a test on XLA if the condition is true.
class skipXLAIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='xla')

class skipPRIVATEUSE1If(skipIf):

    def __init__(self, dep, reason):
        device_type = torch._C._get_privateuse1_backend_name()
        super().__init__(dep, reason, device_type=device_type)

def _has_sufficient_memory(device, size):
    if torch.device(device).type == 'cuda':
        if not torch.cuda.is_available():
            return False
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.mem_get_info, aka cudaMemGetInfo, returns a tuple of (free memory, total memory) of a GPU
        if device == 'cuda':
            device = 'cuda:0'
        return torch.cuda.memory.mem_get_info(device)[0] >= size

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
        assert size.endswith(('GB', 'gb')), "only bytes or GB supported"
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


class expectedFailure:

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


class onlyOn:

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
class deviceCountAtLeast:

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

# Only runs the test on the native device type (currently CPU, CUDA, Meta)
def onlyNativeDeviceTypes(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type not in NATIVE_DEVICES:
            reason = "onlyNativeDeviceTypes: doesn't run on {0}".format(self.device_type)
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
class precisionOverride:

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

class toleranceOverride:
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
#   (2) Can be overridden for CPU or CUDA, respectively, using dtypesIfCPU
#       or dtypesIfCUDA.
#   (3) Can accept an iterable of dtypes or an iterable of tuples
#       of dtypes.
# Examples:
# @dtypes(torch.float32, torch.float64)
# @dtypes((torch.long, torch.float32), (torch.int, torch.float64))
class dtypes:

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

class dtypesIfMPS(dtypes):

    def __init__(self, *args):
        super().__init__(*args, device_type='mps')

def onlyCPU(fn):
    return onlyOn('cpu')(fn)


def onlyCUDA(fn):
    return onlyOn('cuda')(fn)


def onlyMPS(fn):
    return onlyOn('mps')(fn)

def onlyPRIVATEUSE1(fn):
    device_type = torch._C._get_privateuse1_backend_name()
    device_mod = getattr(torch, device_type, None)
    if device_mod is None:
        reason = "Skip as torch has no module of {0}".format(device_type)
        return unittest.skip(reason)(fn)
    return onlyOn(device_type)(fn)

def disablecuDNN(fn):

    @wraps(fn)
    def disable_cudnn(self, *args, **kwargs):
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                return fn(self, *args, **kwargs)
        return fn(self, *args, **kwargs)

    return disable_cudnn

def disableMkldnn(fn):

    @wraps(fn)
    def disable_mkldnn(self, *args, **kwargs):
        if torch.backends.mkldnn.is_available():
            with torch.backends.mkldnn.flags(enabled=False):
                return fn(self, *args, **kwargs)
        return fn(self, *args, **kwargs)

    return disable_mkldnn


def expectedFailureCUDA(fn):
    return expectedFailure('cuda')(fn)

def expectedFailureMeta(fn):
    return skipIfTorchDynamo()(expectedFailure('meta')(fn))

def expectedFailureXLA(fn):
    return expectedFailure('xla')(fn)

# Skips a test on CPU if LAPACK is not available.
def skipCPUIfNoLapack(fn):
    return skipCPUIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)


# Skips a test on CPU if FFT is not available.
def skipCPUIfNoFFT(fn):
    return skipCPUIf(not torch._C.has_spectral, "PyTorch is built without FFT support")(fn)


# Skips a test on CPU if MKL is not available.
def skipCPUIfNoMkl(fn):
    return skipCPUIf(not TEST_MKL, "PyTorch is built without MKL support")(fn)


# Skips a test on CPU if MKL Sparse is not available (it's not linked on Windows).
def skipCPUIfNoMklSparse(fn):
    return skipCPUIf(IS_WINDOWS or not TEST_MKL, "PyTorch is built without MKL support")(fn)


# Skips a test on CPU if mkldnn is not available.
def skipCPUIfNoMkldnn(fn):
    return skipCPUIf(not torch.backends.mkldnn.is_available(), "PyTorch is built without mkldnn support")(fn)


# Skips a test on CUDA if MAGMA is not available.
def skipCUDAIfNoMagma(fn):
    return skipCUDAIf('no_magma', "no MAGMA library detected")(skipCUDANonDefaultStreamIf(True)(fn))

def has_cusolver():
    return not TEST_WITH_ROCM

def has_hipsolver():
    rocm_version = _get_torch_rocm_version()
    # hipSOLVER is disabled on ROCM < 5.3
    return rocm_version >= (5, 3)

# Skips a test on CUDA/ROCM if cuSOLVER/hipSOLVER is not available
def skipCUDAIfNoCusolver(fn):
    return skipCUDAIf(not has_cusolver() and not has_hipsolver(), "cuSOLVER not available")(fn)


# Skips a test if both cuSOLVER and MAGMA are not available
def skipCUDAIfNoMagmaAndNoCusolver(fn):
    if has_cusolver():
        return fn
    else:
        # cuSolver is disabled on cuda < 10.1.243, tests depend on MAGMA
        return skipCUDAIfNoMagma(fn)

# Skips a test if both cuSOLVER/hipSOLVER and MAGMA are not available
def skipCUDAIfNoMagmaAndNoLinalgsolver(fn):
    if has_cusolver() or has_hipsolver():
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
                rocm_version_tuple = _get_torch_rocm_version()
                if rocm_version_tuple is None or version is None or rocm_version_tuple < tuple(version):
                    reason = "ROCm {0} is available but {1} required".format(rocm_version_tuple, version)
                    raise unittest.SkipTest(reason)

            return fn(self, *args, **kwargs)

        return wrap_fn
    return dec_fn

# Skips a test for specified CUDA versions, given in the form of a list of [major, minor]s.
def skipCUDAVersionIn(versions : List[Tuple[int, int]] = None):
    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            version = _get_torch_cuda_version()
            if version == (0, 0):  # cpu or rocm
                return fn(self, *args, **kwargs)
            if version in (versions or []):
                reason = "test skipped for CUDA version {0}".format(version)
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)

        return wrap_fn
    return dec_fn

# Skips a test for CUDA versions less than specified, given in the form of [major, minor].
def skipCUDAIfVersionLessThan(versions : Tuple[int, int] = None):
    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            version = _get_torch_cuda_version()
            if version == (0, 0):  # cpu or rocm
                return fn(self, *args, **kwargs)
            if version < versions:
                reason = "test skipped for CUDA versions < {0}".format(version)
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

# Skips a test on CUDA if cuSparse generic API is not available
def skipCUDAIfNoCusparseGeneric(fn):
    return skipCUDAIf(not TEST_CUSPARSE_GENERIC, "cuSparse Generic API not available")(fn)

def skipCUDAIfNoHipsparseGeneric(fn):
    return skipCUDAIf(not TEST_HIPSPARSE_GENERIC, "hipSparse Generic API not available")(fn)

def skipCUDAIfNoSparseGeneric(fn):
    return skipCUDAIf(not (TEST_CUSPARSE_GENERIC or TEST_HIPSPARSE_GENERIC), "Sparse Generic API not available")(fn)

def skipCUDAIfNoCudnn(fn):
    return skipCUDAIfCudnnVersionLessThan(0)(fn)

def skipCUDAIfMiopen(fn):
    return skipCUDAIf(torch.version.hip is not None, "Marked as skipped for MIOpen")(fn)

def skipCUDAIfNoMiopen(fn):
    return skipCUDAIf(torch.version.hip is None, "MIOpen is not available")(skipCUDAIfNoCudnn(fn))

def skipMeta(fn):
    return skipMetaIf(True, "test doesn't work with meta tensors")(fn)

def skipXLA(fn):
    return skipXLAIf(True, "Marked as skipped for XLA")(fn)

def skipMPS(fn):
    return skipMPSIf(True, "test doesn't work on MPS backend")(fn)

def skipPRIVATEUSE1(fn):
    return skipPRIVATEUSE1If(True, "test doesn't work on privateuse1 backend")(fn)

# TODO: the "all" in the name isn't true anymore for quite some time as we have also have for example XLA and MPS now.
#  This should probably enumerate all available device type test base classes.
def get_all_device_types() -> List[str]:
    return ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
