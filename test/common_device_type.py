import inspect
import threading
from functools import wraps
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
    skipCUDANonDefaultStreamIf

# Note: Generic Device-Type Testing
#
# [WRITING TESTS]
#
# Write your test class as usual except:
#   (1) Each test method should have one of four signatures:
#
#           (1a) testX(self, device)
#
#           (1b) @deviceCountAtLeast(<minimum number of devices to run test with>)
#                testX(self, devices)
#
#           (1c) @dtypes(<list of dtypes>)
#                testX(self, device, dtype)
#
#           (1d) @deviceCountAtLeast(<minimum number of devices to run test with>)
#                @dtypes(<list of dtypes>)
#                testX(self, devices, dtype)
#
#
#       Note that the decorators are required for signatures (1b), (1c) and
#       (1d).
#
#       When a test like (1a) is called it will be given a device string,
#       like 'cpu' or 'cuda:0.'
#
#       Tests like (1b) are called with a list of device strings, like
#       ['cuda:0', 'cuda:1']. The first device string will be the
#       primary device. These tests will be skipped if the device type
#       has fewer available devices than the argument to @deviceCountAtLeast.
#
#       Tests like (1c) are called with a device string and a torch.dtype from
#       the list of dtypes specified in the @dtypes decorator. Device-specific
#       dtype overrides can be specified using @dtypesIfCPU and @dtypesIfCUDA.
#
#       Tests like (1d) take a devices argument like (1b) and a dtype
#       argument from (1c).
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

# List of device type test bases that can be used to instantiate tests.
# See below for how this list is populated. If you're adding a device type
# you should check if it's available and (if it is) add it to this list.
device_type_test_bases = []


class DeviceTypeTestBase(TestCase):
    device_type = 'generic_device_type'

    # Precision is a thread-local setting since it may be overriden per test
    _tls = threading.local()
    _tls.precision = TestCase.precision

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
    def instantiate_test(cls, name, test):
        test_name = name + "_" + cls.device_type

        dtypes = cls._get_dtypes(test)
        if dtypes is None:  # Test has no dtype variants
            assert not hasattr(cls, test_name), "Redefinition of test {0}".format(test_name)

            @wraps(test)
            def instantiated_test(self, test=test):
                device_arg = cls.get_primary_device() if not hasattr(test, 'num_required_devices') else cls.get_all_devices()
                return test(self, device_arg)

            setattr(cls, test_name, instantiated_test)
        else:  # Test has dtype variants
            for dtype in dtypes:
                dtype_str = str(dtype).split('.')[1]
                dtype_test_name = test_name + "_" + dtype_str
                assert not hasattr(cls, dtype_test_name), "Redefinition of test {0}".format(dtype_test_name)

                @wraps(test)
                def instantiated_test(self, test=test, dtype=dtype):
                    device_arg = cls.get_primary_device() if not hasattr(test, 'num_required_devices') else cls.get_all_devices()
                    # Sets precision and runs test
                    # Note: precision is reset after the test is run
                    guard_precision = self.precision
                    try :
                        self.precision = self._get_precision_override(test, dtype)
                        result = test(self, device_arg, dtype)
                    finally:
                        self.precision = guard_precision

                    return result

                setattr(cls, dtype_test_name, instantiated_test)


class CPUTestBase(DeviceTypeTestBase):
    device_type = 'cpu'


class CUDATestBase(DeviceTypeTestBase):
    device_type = 'cuda'
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

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
        cls.no_cudnn = not (TEST_WITH_ROCM or torch.backends.cudnn.is_acceptable(t))
        cls.cudnn_version = None if cls.no_cudnn else torch.backends.cudnn.version()

        # Acquires the current device as the primary (test) device
        cls.primary_device = 'cuda:{0}'.format(torch.cuda.current_device())


# Adds available device-type-specific test base classes
device_type_test_bases.append(CPUTestBase)
if torch.cuda.is_available():
    device_type_test_bases.append(CUDATestBase)

PYTORCH_CUDA_MEMCHECK = os.getenv('PYTORCH_CUDA_MEMCHECK', '0') == '1'


# Adds 'instantiated' device-specific test cases to the given scope.
# The tests in these test cases are derived from the generic tests in
# generic_test_class.
# See note "Generic Device Type Testing."
def instantiate_device_type_tests(generic_test_class, scope, except_for=None):
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
    generic_members = set(dir(generic_test_class)) - set(dir(empty_class))
    generic_tests = [x for x in generic_members if x.startswith('test')]

    # Creates device-specific test cases
    for base in device_type_test_bases:
        # Skips bases listed in except_for
        if except_for is not None and base.device_type in except_for:
            continue

        class_name = generic_test_class.__name__ + base.device_type.upper()
        device_type_test_class = type(class_name, (base, empty_class), {})

        for name in generic_members:
            if name in generic_tests:  # Instantiates test member
                # Requires tests be a function for Python2 compat
                # (In Python2 tests are type checked methods wrapping functions)
                test = getattr(generic_test_class, name)
                if hasattr(test, '__func__'):
                    test = test.__func__
                assert inspect.isfunction(test), "Couldn't extract function from '{0}'".format(name)

                # Instantiates the device-specific tests
                device_type_test_class.instantiate_test(name, test)
            else:  # Ports non-test member
                assert not hasattr(device_type_test_class, name), "Redefinition of non-test member {0}".format(name)

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
        super(skipCPUIf, self).__init__(dep, reason, device_type='cpu')


# Skips a test on CUDA if the condition is true.
class skipCUDAIf(skipIf):

    def __init__(self, dep, reason):
        super(skipCUDAIf, self).__init__(dep, reason, device_type='cuda')


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
#   (2) Can be overriden for the CPU or CUDA, respectively, using dtypesIfCPU
#       or dtypesIfCUDA.
#   (3) Prefer the existing decorators to defining the 'device_type' kwarg.
class dtypes(object):

    # Note: *args, **kwargs for Python2 compat.
    # Python 3 allows (self, *args, device_type='all').
    def __init__(self, *args, **kwargs):
        assert args is not None and len(args) != 0, "No dtypes given"
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
        super(dtypesIfCPU, self).__init__(*args, device_type='cpu')


# Overrides specified dtypes on CUDA.
class dtypesIfCUDA(dtypes):

    def __init__(self, *args):
        super(dtypesIfCUDA, self).__init__(*args, device_type='cuda')


def onlyCPU(fn):
    return onlyOn('cpu')(fn)


def onlyCUDA(fn):
    return onlyOn('cuda')(fn)


def expectedFailureCUDA(fn):
    return expectedFailure('cuda')(fn)


# Skips a test on CPU if LAPACK is not available.
def skipCPUIfNoLapack(fn):
    return skipCPUIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)


# Skips a test on CPU if MKL is not available.
def skipCPUIfNoMkl(fn):
    return skipCPUIf(not TEST_MKL, "PyTorch is built without MKL support")(fn)


# Skips a test on CUDA if MAGMA is not available.
def skipCUDAIfNoMagma(fn):
    return skipCUDAIf('no_magma', "no MAGMA library detected")(skipCUDANonDefaultStreamIf(True)(fn))


# Skips a test on CUDA when using ROCm.
def skipCUDAIfRocm(fn):
    return skipCUDAIf(TEST_WITH_ROCM, "test doesn't currently work on the ROCm stack")(fn)


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
