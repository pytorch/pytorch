import types
import unittest

import torch
from common_utils import TestCase

# Note: Generic Device-Type Testing
#
# [WRITING TESTS]
#
# To write a test that runs on a variety of devices do the following:
#
#   (1) Create a class, "class <name>(<base>)"
#   (2) Write tests as usual, except they must take a 'device' argument
#       and may take a 'dtype' argument. If the test takes the 'dtype'
#       argument then it must specify a set of dtypes using the @dtypes
#       decorator. Additional decorators can specialize the set of dtypes
#       for CPU or CUDA device types.
#
#       The device will be a string, like 'cpu' or 'cuda,' and dtype will be a
#       member of the set passed to @dtypes, like torch.float.
#       The test should use the given device and dtype. See test_torch.py's
#       TestTorchDeviceType for an example.
#   (3) Non-test attributes, like helper methods, must go in the base class.
#       This quirk is for Python 2 compatibility.
#   (5) Call instantiate_device_type_tests(_<name>, globals()) in your test
#       test script. This will create the device-specific test cases from
#       the generic test case and make them discoverable.
#
# See test_torch.py for an example.
#
# [RUNNING TESTS]
#
# The above will create test classes with names <name><device_type.upper()>,
# e.g. if the generic class is called _TestTorchDevice then the created test
# classes will TestTorchDeviceTypeCPU and TestTorchDeviceTypeCUDA.
#
# Tests like test_diag will have "_<device_type>" appended. So an instantiated
# test can be run directly with the command
# "python test_torch.py TestTorchDeviceTypeCPU.test_diag_cpu"
#
# If a test takes dtypes, then the dtype will also be appended, so the test
# name will be <name>_<device_type>_<dtype>, and can be run directly with
# "python test_torch.py TestTorchDeviceTypeCPU.test_neg_cpu_torch.float"
#
# Particular test suites can be run using pytest filtering. For example,
# "pytest test_torch.py -k 'test_diag'"
# will run test_diag on every device (and with every dtype, if applicable).
# To specify particular device types or dtypes the 'and' keyword can be used:
# "pytest test_torch.py -k 'test_diag and cpu'"
# "pytest test_torch.py -k 'test_neg and cuda and float"
# pytest filtering also makes it easy to run all tests on a particular device
# type or all tests for a particular dtype.
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
#   (4) (Optional) Define the "supported_dtypes" attribute to
#       a list of dtypes your device type supports.
#   (5) (Optional) Write setUpClass/tearDownClass class methods.
#   (6) (Optional) Override the _check_if_skip method.
#
# Device type tests are created and discovered BEFORE setUpClass is called.
# setUpClass is called before tests are run, however.
# setupClass is also only called if its tests are being run, and so can
# it can initialize devices. The setUpClass method might be used, for example,
# to inspect the available software and hardware and change the
# supported_dtypes attribute or dynamically exclude some tests.
#
# Before each test is run the _check_if_skip method is called. This method
# can use the information defined in setUpClass to determine if a test should
# be run. See below for examples.
#

# List of device type test bases that can be used to instantiate tests
device_type_test_bases = []

# List of all Torch dtypes
all_dtypes = [torch.float, torch.double, torch.half, torch.uint8,
              torch.int8, torch.short, torch.int, torch.long, torch.bool]

# Holds dtype metadata that applies to all device types (unless overriden)
# test_name : str -> iterable of Torch dtypes
_generic_test_dtypes = {}

# Holds dependency metadata that applies to all device types
# test_name : str -> list of (dependency : [bool, str], skip reason : str) tuples
_generic_test_deps = {}

class DeviceTypeTestBase(TestCase):
    device_type = "generic_device_type"
    supported_dtypes = all_dtypes

    @classmethod
    def _get_dtypes(cls, test):
        if hasattr(cls, 'dtypes') and test.__name__ in cls.dtypes:
            return cls.dtypes[test.__name__]
        return _generic_test_dtypes.get(test.__name__, None)

    def _check_if_skip(self, test, dtype=None):
        # Checks if dtype is supported
        if dtype is not None and dtype not in self.supported_dtypes:
            reason = "unsupported dtype '{0}'".format(str(dtype))
            raise unittest.SkipTest(reason)

        # Checks dependencies
        if hasattr(self, 'deps'):
            deps = self.deps.get(test.__name__, [])
            deps.extend(_generic_test_deps.get(test.__name__, []))
            for dep, reason in deps:
                if isinstance(dep, int) and dep:
                    raise unittest.SkipTest(reason)
                elif isinstance(dep, str) and hasattr(self, dep) and not getattr(self, dep):
                    raise unittest.SkipTest(reason)

    # Creates device-specific tests.
    # Note: called after all non-test members of the generic class have
    # been ported. Called
    @classmethod
    def instantiate_test(cls, test):
        test_name = test.__name__ + "_" + cls.device_type

        dtypes = cls._get_dtypes(test)
        if dtypes is None or len(dtypes) == 0:
            # Test has no dtype variants
            def instantiated_test(self, test=test):
                self._check_if_skip(test)
                return test(self, cls.device_type)

            setattr(cls, test_name, instantiated_test)
        else:
            # Test has dtype variants
            for dtype in dtypes:
                dtype_test_name = test_name + "_" + str(dtype)

                def instantiated_test(self, test=test, dtype=dtype):
                    self._check_if_skip(test, dtype=dtype)
                    return test(self, cls.device_type, dtype)

                setattr(cls, dtype_test_name, instantiated_test)

class CPUTestBase(DeviceTypeTestBase):
    device_type = "cpu"
    dtypes = {}  # Holds device-specific dtype metadata
    deps = {}  # Holds device-specific dependency metadata

    @classmethod
    def setUpClass(cls):
        cls.lapack = torch._C.has_lapack

class CUDATestBase(DeviceTypeTestBase):
    device_type = "cuda"
    dtypes = {}  # Holds device-specific dtype metadata
    deps = {}  # Holds device-specific dependency metadata
    _do_cuda_memory_leak_check = True

    @classmethod
    def setUpClass(cls):
        # has_magma shows up after cuda is initialized
        torch.ones(1).cuda()
        cls.magma = torch.cuda.has_magma

# Adds available device-type-specific classes
device_type_test_bases.append(CPUTestBase)
if torch.cuda.is_available():
    device_type_test_bases.append(CUDATestBase)

# Adds 'instantiated' device-specific test cases to the given scope.
# The tests in these test cases are derived from the generic tests in
# generic_test_class.
# See note "Generic Device Type Testing."
def instantiate_device_type_tests(generic_test_class, scope):
    # Removes the generic test class from its enclosing scope so it's tests
    # are not discoverable.
    del scope[generic_test_class.__name__]

    # Creates an 'empty' version of the generic_test_class
    empty_name = generic_test_class.__name__ + "_base"
    empty_class = type(empty_name, (generic_test_class.__base__,), {})

    # Acquires member names
    generic_members = set(dir(generic_test_class)) - set(dir(empty_class))
    generic_tests = [x for x in generic_members if x.startswith('test_')]
    generic_nontests = generic_members - set(generic_tests)

    assert len(generic_nontests) == 0, "Generic device class has non-test attributes"

    # Makes all tests in generic_test_class staticmethods
    # Note: necessary for Python 2 compatibility
    for test in generic_tests:
        fn = staticmethod(getattr(generic_test_class, test))
        setattr(generic_test_class, test, fn)

    for base in device_type_test_bases:
        # Creates the device-specific test case
        class_name = generic_test_class.__name__ + base.device_type.upper()
        device_type_test_class = type(class_name, (base, empty_class), {})

        # Instantiates device-specific tests
        for name in generic_tests:
            assert not hasattr(device_type_test_class, name), \
                "Device-specific test class attribute '{0}' redefinition".format(name)
            test = getattr(generic_test_class, name)
            if hasattr(test, '__func__'):
                test = test.__func__
            device_type_test_class.instantiate_test(test)

        # Mimics defining the instantiated class in the caller's file
        # by setting its module to the given class's and adding
        # the module to the given scope.
        # This lets the instantiated class be discovered by unittest.
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class

def _get_base(device_type):
    for base in device_type_test_bases:
        if device_type == base.device_type:
            return base
    return None

# Decorator that specifies a variant of the test for each of the given
# dtypes should be instantiated. If device_type is None then the
# variants are instantiated for all device types (unless overridden).
# for all device types (unless overridden). Setting device_type
# will override previous @dtypes decorations for that device type.
class dtypes(object):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.device_type = kwargs.get('device_type', None)

    def __call__(self, fn):
        if self.device_type is None:
            _generic_test_dtypes[fn.__name__] = self.args
        else:
            base = _get_base(self.device_type)
            if base is not None:
                base.dtypes[fn.__name__] = self.args
        return fn

# Sugar for @dtypes(..., device_type='cpu')
class dtypesIfCPU(dtypes):

    def __init__(self, *args):
        super(dtypesIfCPU, self).__init__(*args, device_type='cpu')

# Sugar for @dtypes(..., device_type='cuda')
class dtypesIfCUDA(dtypes):

    def __init__(self, *args):
        super(dtypesIfCUDA, self).__init__(*args, device_type='cuda')

# Decorator that specifes when to skip a test. If device_type is None
# then the dependency is required for all device_types. Unlike @dtypes,
# dependencies are extended, not overwritten, by later calls.
# Dependencies can either be booleans or strings specifiying an
# attribute. If the latter, the test is skipped if the class has the attribute
# and it's false.
class skipIf(object):

    def __init__(self, dep, reason, device_type=None):
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):
        if self.device_type is None:
            if fn.__name__ not in _generic_test_deps:
                _generic_test_deps[fn.__name__] = []
            _generic_test_deps[fn.__name__].append((self.dep, self.reason))
        else:
            base = _get_base(self.device_type)
            if base is not None:
                if fn.__name__ not in base.deps:
                    base.deps[fn.__name__] = []
                base.deps[fn.__name__].append((self.dep, self.reason))
        return fn

# Sugar for @skipIf(..., device_type='cpu')
class skipCPUIf(skipIf):

    def __init__(self, dep, reason):
        super(skipCPUIf, self).__init__(dep, reason, device_type='cpu')

# Sugar for @skipIf(..., device_type='cuda')
class skipCUDAIf(skipIf):

    def __init__(self, dep, reason):
        super(skipCUDAIf, self).__init__(dep, reason, device_type='cuda')

# Sugar for @skipCPUIf('lapack', "PyTorch compiled without Lapack")
def skipCPUIfNoLapack(fn):
    skipper = skipCPUIf('lapack', "PyTorch compiled without Lapack")
    skipper(fn)
    return fn

# Sugar for @skipCUDAIf('magma', "no MAGMA library detected")
def skipCUDAIfNoMagma(fn):
    skipper = skipCUDAIf('magma', "no MAGMA library detected")
    skipper(fn)
    return fn
