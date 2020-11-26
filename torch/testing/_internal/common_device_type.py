import copy
import gc
import inspect
import runpy
import threading
from functools import wraps
from typing import List, Any, ClassVar
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
    skipCUDANonDefaultStreamIf, TEST_WITH_ASAN, TEST_WITH_UBSAN, TEST_WITH_TSAN, \
    IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing import \
    (get_all_dtypes)

try:
    import psutil  # type: ignore[import]
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Note: Generic Device-Type Testing
#
# [WRITING TESTS]
#
# Write your test class as usual except:
#   (1) Each test method should have one of following five signatures:
#
#           (1a) testX(self, device)
#
#           (1b) @deviceCountAtLeast(<minimum number of devices to run test with>)
#                testX(self, devices)
#
#           (1c) @dtypes(<list of dtypes> or <list of tuples of dtypes>)
#                testX(self, device, dtype)
#
#           (1d) @deviceCountAtLeast(<minimum number of devices to run test with>)
#                @dtypes(<list of dtypes> or <list of tuples of dtypes>)
#                testX(self, devices, dtype)
#
#           (1e) @ops(<list of OpInfo instances>)
#                testX(self, device, dtype, op)
#
#       Note that the decorators are required for signatures 1b--1e.
#
#       When a test like (1a) is called it will be given a device string,
#       like 'cpu' or 'cuda:0.'
#
#       Tests like (1b) are called with a list of device strings, like
#       ['cuda:0', 'cuda:1']. The first device string will be the
#       primary device. These tests will be skipped if the device type
#       has fewer available devices than the argument to @deviceCountAtLeast.
#
#       Tests like (1c) are called with a device string and a torch.dtype (or
#       a tuple of torch.dtypes) from the list of dtypes (or list of tuples
#       of torch.dtypes) specified in the @dtypes decorator. Device-specific
#       dtype overrides can be specified using @dtypesIfCPU and @dtypesIfCUDA.
#
#       Tests like (1d) take a devices argument like (1b) and a dtype
#       argument from (1c).
#
#       Tests like (1e) are instantiated for each provided OpInfo instance,
#       with dtypes specified by the OpInfo instance (unless overridden with
#       an additional @dtypes decorator).
#
#   (2) Prefer using test decorators defined in this file to others.
#       For example, using the @skipIfNoLapack decorator instead of the
#       @skipCPUIfNoLapack will cause the test to not run on CUDA if
#       LAPACK is not available, which is wrong. If you need to use a decorator
#       you may want to ask about porting it to this framework.
#
#   See the TestTorchDeviceType class in test_torch.py for an example.
#
# [RUNNING TESTS]
#
# After defining your test class call instantiate_device_type_tests on it
# and pass in globals() for the second argument. This will instantiate
# discoverable device-specific test classes from your generic class. It will
# also hide the tests in your generic class so they're not run.
#
# If you device-generic test class is TestClass then new classes with names
# TestClass<DEVICE_TYPE> will be created for each available device type.
# TestClassCPU and TestClassCUDA, for example. Tests in these classes also
# have the device type and dtype, if provided, appended to their original
# name. testX, for instance, becomes testX_<device_type> or
# testX_<device_type>_<dtype>.
#
# More concretely, TestTorchDeviceType becomes TestTorchDeviceTypeCPU,
# TestTorchDeviceTypeCUDA, ... test_diagonal in TestTorchDeviceType becomes
# test_diagonal_cpu, test_diagonal_cuda, ... test_erfinv, which accepts a dtype,
# becomes test_erfinv_cpu_float, test_erfinv_cpu_double, test_erfinv_cuda_half,
# ...
#
# In short, if you write a test signature like
#   def textX(self, device)
# You are effectively writing
#   def testX_cpu(self, device='cpu')
#   def textX_cuda(self, device='cuda')
#   def testX_xla(self, device='xla')
#   ...
#
# These tests can be run directly like normal tests:
# "python test_torch.py TestTorchDeviceTypeCPU.test_diagonal_cpu"
#
# All the tests for a particular device type can be run using the class, and
# other collections of tests can be run using pytest filtering, like
#
# "pytest test_torch.py -k 'test_diag'"
#
# which will run test_diag on every available device.
#
# To specify particular device types the 'and' keyword can be used:
#
# "pytest test_torch.py -k 'test_erfinv and cpu'"
#
# will run test_erfinv on all cpu dtypes.
#
# [ADDING A DEVICE TYPE]
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
#
# Note [Overriding methods in generic tests]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#

# List of device type test bases that can be used to instantiate tests.
# See below for how this list is populated. If you're adding a device type
# you should check if it's available and (if it is) add it to this list.


def _construct_test_name(test_name, op, device_type, dtype):
    if op is not None:
        test_name += "_" + op.name

    test_name += "_" + device_type

    if dtype is not None:
        if isinstance(dtype, (list, tuple)):
            for d in dtype:
                test_name += "_" + str(d).split('.')[1]
        else:
            test_name += "_" + str(dtype).split('.')[1]

    return test_name

class DeviceTypeTestBase(TestCase):
    device_type: str = 'generic_device_type'

    # Precision is a thread-local setting since it may be overridden per test
    _tls = threading.local()
    _tls.precision = TestCase._precision

    @property
    def precision(self):
        return self._tls.precision

    @precision.setter
    def precision(self, prec):
        self._tls.precision = prec

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

    # Creates device-specific tests.
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):

        def instantiate_test_helper(cls, name, *, test, dtype, op):

            # Constructs the test's name
            test_name = _construct_test_name(name, op, cls.device_type, dtype)

            # wraps instantiated test with op decorators
            # NOTE: test_wrapper exists because we don't want to apply
            #   op-specific decorators to the original test.
            #   Test-sepcific decorators are applied to the original test,
            #   however.
            if op is not None and op.decorators is not None:
                @wraps(test)
                def test_wrapper(*args, **kwargs):
                    return test(*args, **kwargs)

                for decorator in op.decorators:
                    test_wrapper = decorator(test_wrapper)

                test_fn = test_wrapper
            else:
                test_fn = test

            # Constructs the test
            @wraps(test)
            def instantiated_test(self, name=name, test=test_fn, dtype=dtype, op=op):
                if op is not None and op.should_skip(generic_cls.__name__, name,
                                                     self.device_type, dtype):
                    self.skipTest("Skipped!")

                device_arg: str = cls.get_primary_device()
                if hasattr(test_fn, 'num_required_devices'):
                    device_arg = cls.get_all_devices()

                # Sets precision and runs test
                # Note: precision is reset after the test is run
                guard_precision = self.precision
                try:
                    self.precision = self._get_precision_override(test_fn, dtype)
                    args = (arg for arg in (device_arg, dtype, op) if arg is not None)
                    result = test_fn(self, *args)
                finally:
                    self.precision = guard_precision

                return result

            assert not hasattr(cls, test_name), "Redefinition of test {0}".format(test_name)
            setattr(cls, test_name, instantiated_test)

        # Handles tests using the ops decorator
        if hasattr(test, "op_list"):
            for op in test.op_list:
                # Acquires dtypes, using the op data if unspecified
                dtypes = cls._get_dtypes(test)
                if dtypes is None:
                    if test.unsupported_dtypes_only:
                        dtypes = set(get_all_dtypes()).difference(op.supported_dtypes(cls.device_type))
                    elif test.all_dtypes:
                        dtypes = op.supported_dtypes(cls.device_type)
                    else:
                        dtypes = op.tested_dtypes(cls.device_type)

                    if test.dtype_filter is not None:
                        dtypes = dtypes.intersection(test.dtype_filter)

                else:
                    assert test.dtype_filter is None, "ops(dtype_filter=[...]) and the dtypes decorator are incompatible"
                    assert not test.all_dtypes, "ops(all_dtypes=True) and dtypes decorator are incompatible"

                for dtype in dtypes:
                    instantiate_test_helper(cls,
                                            name,
                                            test=test,
                                            dtype=dtype,
                                            op=op)
        else:
            # Handles tests that don't use the ops decorator
            dtypes = cls._get_dtypes(test)
            dtypes = tuple(dtypes) if dtypes is not None else (None,)
            for dtype in dtypes:
                instantiate_test_helper(cls, name, test=test, dtype=dtype, op=None)


class CPUTestBase(DeviceTypeTestBase):
    device_type = 'cpu'


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
            # skip if sanitizer is enabled
            if not TEST_WITH_ASAN and not TEST_WITH_TSAN and not TEST_WITH_UBSAN:
                test_bases.append(CUDATestBase)
        else:
            test_bases.append(CPUTestBase)
    else:
        test_bases.append(CPUTestBase)
        if torch.cuda.is_available():
            test_bases.append(CUDATestBase)

    return test_bases


device_type_test_bases = get_device_type_test_bases()


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
        mod = runpy.run_path(path, init_globals=globals())
        device_type_test_bases.append(mod['TEST_CLASS'])


PYTORCH_CUDA_MEMCHECK = os.getenv('PYTORCH_CUDA_MEMCHECK', '0') == '1'


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

    # Creates device-specific test cases
    for base in device_type_test_bases:
        # Skips bases listed in except_for
        if except_for is not None and only_for is not None:
            assert base.device_type not in except_for or base.device_type not in only_for,\
                "same device cannot appear in except_for and only_for"
        if except_for is not None and base.device_type in except_for:
            continue
        if only_for is not None and base.device_type not in only_for:
            continue

        class_name = generic_test_class.__name__ + base.device_type.upper()

        # type set to Any and suppressed due to unsupport runtime class:
        # https://github.com/python/mypy/wiki/Unsupported-Python-Features
        device_type_test_class: Any = type(class_name, (base, empty_class), {})

        for name in generic_members:
            if name in generic_tests:  # Instantiates test member
                # Requires tests be a function for Python2 compat
                # (In Python2 tests are type checked methods wrapping functions)
                test = getattr(generic_test_class, name)
                if hasattr(test, '__func__'):
                    test = test.__func__
                assert inspect.isfunction(test), "Couldn't extract function from '{0}'".format(name)

                # XLA-compat shim (XLA's instantiate_test takes doesn't take generic_cls)
                sig = inspect.signature(device_type_test_class.instantiate_test)
                if len(sig.parameters) == 3:
                    # Instantiates the device-specific tests
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test), generic_cls=generic_test_class)
                else:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test))
            else:  # Ports non-test member
                assert name not in device_type_test_class.__dict__, "Redefinition of directly defined member {0}".format(name)

                # Unwraps to functions (when available) for Python2 compat
                nontest = getattr(generic_test_class, name)
                if hasattr(nontest, '__func__'):
                    nontest = nontest.__func__

                setattr(device_type_test_class, name, nontest)

        # Mimics defining the instantiated class in the caller's file
        # by setting its module to the given class's and adding
        # the module to the given scope.
        # This lets the instantiated class be discovered by unittest.
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class


# Decorator that defines the ops a test should be run with
# The test signature must be:
#   <test_name>(self, device, dtype, op)
# For example:
# @ops(unary_ufuncs)
# test_numerics(self, device, dtype, op):
#   <test_code>
class ops(object):
    def __init__(self, op_list, *, unsupported_dtypes_only=False, all_dtypes=False, dtype_filter=None):
        self.op_list = op_list
        assert not (unsupported_dtypes_only and all_dtypes), \
            "Cannot have both all_dtypes and unsupported_dtypes_only"
        self.unsupported_dtypes_only = unsupported_dtypes_only
        self.all_dtypes = all_dtypes
        self.dtype_filter = set(dtype_filter) if dtype_filter is not None else None

    def __call__(self, fn):
        fn.op_list = self.op_list
        fn.unsupported_dtypes_only = self.unsupported_dtypes_only
        fn.all_dtypes = self.all_dtypes
        fn.dtype_filter = self.dtype_filter
        return fn

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
        def dep_fn(slf, device, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                if (isinstance(self.dep, str) and getattr(slf, self.dep, True)) or (isinstance(self.dep, bool) and self.dep):
                    raise unittest.SkipTest(self.reason)

            return fn(slf, device, *args, **kwargs)
        return dep_fn


# Skips a test on CPU if the condition is true.
class skipCPUIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='cpu')


# Skips a test on CUDA if the condition is true.
class skipCUDAIf(skipIf):

    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type='cuda')

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
        def efail_fn(slf, device, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                try:
                    fn(slf, device, *args, **kwargs)
                except Exception:
                    return
                else:
                    slf.fail('expected test to fail, but it passed')

            return fn(slf, device, *args, **kwargs)
        return efail_fn


class onlyOn(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def only_fn(slf, device, *args, **kwargs):
            if self.device_type != slf.device_type:
                reason = "Only runs on {0}".format(self.device_type)
                raise unittest.SkipTest(reason)

            return fn(slf, device, *args, **kwargs)

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
    def only_fn(self, device, *args, **kwargs):
        if self.device_type != 'cpu' and self.device_type != 'cuda':
            reason = "Doesn't run on {0}".format(self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, device, *args, **kwargs)

    return only_fn

# Specifies per-dtype precision overrides.
# Ex.
#
# @precisionOverride(torch.half : 1e-2, torch.float : 1e-4)
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

    # Note: *args, **kwargs for Python2 compat.
    # Python 3 allows (self, *args, device_type='all').
    def __init__(self, *args, **kwargs):
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
        self.device_type = kwargs.get('device_type', 'all')

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

class expectedAlertNondeterministic:
    def __init__(self, caller_name, device_type=None, fn_has_device_arg=True):
        self.device_type = device_type
        self.error_message = caller_name + ' does not have a deterministic implementation, but you set'
        self.fn_has_device_arg = fn_has_device_arg

    def __call__(self, fn):
        @wraps(fn)
        def efail_fn(slf, device, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                deterministic_restore = torch.is_deterministic()
                torch.set_deterministic(True)
                try:
                    if self.fn_has_device_arg:
                        fn(slf, device, *args, **kwargs)
                    else:
                        fn(slf, *args, **kwargs)
                except RuntimeError as e:
                    torch.set_deterministic(deterministic_restore)
                    if self.error_message not in str(e):
                        slf.fail(
                            'expected non-deterministic error message to start with "'
                            + self.error_message
                            + '" but got this instead: "' + str(e) + '"')
                    return
                else:
                    torch.set_deterministic(deterministic_restore)
                    slf.fail('expected a non-deterministic error, but it was not raised')

            if self.fn_has_device_arg:
                return fn(slf, device, *args, **kwargs)
            else:
                return fn(slf, *args, **kwargs)

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


# Skips a test on CPU if MKL is not available.
def skipCPUIfNoMkl(fn):
    return skipCPUIf(not TEST_MKL, "PyTorch is built without MKL support")(fn)


# Skips a test on CUDA if MAGMA is not available.
def skipCUDAIfNoMagma(fn):
    return skipCUDAIf('no_magma', "no MAGMA library detected")(skipCUDANonDefaultStreamIf(True)(fn))

def skipCUDAIfNoMagmaAndNoCusolver(fn):
    version = _get_torch_cuda_version()
    if version >= [10, 2]:
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


# Skips a test on CUDA if cuDNN is unavailable or its version is lower than requested.
def skipCUDAIfCudnnVersionLessThan(version=0):

    def dec_fn(fn):
        @wraps(fn)
        def wrap_fn(self, device, *args, **kwargs):
            if self.device_type == 'cuda':
                if self.no_cudnn:
                    reason = "cuDNN not available"
                    raise unittest.SkipTest(reason)
                if self.cudnn_version is None or self.cudnn_version < version:
                    reason = "cuDNN version {0} is available but {1} required".format(self.cudnn_version, version)
                    raise unittest.SkipTest(reason)

            return fn(self, device, *args, **kwargs)

        return wrap_fn
    return dec_fn


def skipCUDAIfNoCudnn(fn):
    return skipCUDAIfCudnnVersionLessThan(0)(fn)
