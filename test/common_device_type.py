import inspect
from functools import wraps
import unittest
import torch
from common_utils import TestCase

# Note: Generic Device-Type Testing
#
# [WRITING TESTS]
#
# Write your test class as usual except:
#   (1) Only define test methods in the test class itself. Helper methods
#       and non-methods must be inherited. This limitation is for Python2
#       compatibility.
#   (2) Each test method should have have the signature
#           testX(self, device)
#       OR
#           testX(self, device, dtype)
#       The device argument will be a string like 'cpu' or 'cuda.'
#       If the dtype argument is in the signature the test must be decorated
#       with a dtypes decorator (see below). This will instantiate a variant
#       of the test for each dtype.
#   (3) Prefer using test decorators defined in this file to others.
#       For example, using the @skipIfNoLapack decorator instead of the
#       @skipCPUIfNoLapack will cause the test to not run on CUDA if
#       LAPACK is not available, which is wrong. If you need to use a
#       decorator you may have to port it.
#
#   See the TestTorchDeviceType class in test_torch.py for an example.
#
# [RUNNING TESTS]
#
# After defining your test class call instantiate_device_type_tests on it
# and pass in globals() for the second argument. This will instantiate
# discoverable device-specific test classes from your generic class. It will
# also hide the tests in your generic class so they're not run directly.
#
# For each generic testX, a new test textX_<device_type> or a suite of tests
# testX_<device_type>_<dtype> will be created, depending on whether the generic
# test used type in its signature. These tests will be put in classes named
# GenericTestClassName<DEVICE_TYPE>. For example, test_diagonal in TestTorchDeviceType
# becomes test_diagonal_cpu in TestTorchDeviceTypeCPU and test_diagonal_cuda in
# TestTorchDeviceTypeCUDA. Tests with datatypes have the "torch." part of the
# dtype name removed, so tests instantiated from a dtype test like test_neg are
# named test_neg_cpu_float32, test_neg_cuda_int64... These tests receive the
# device_type and dtype corresponding to their names as arguments.
#
# In short, if you write a test signature like
#   def textX(self, device)
# You are effectively writing
#   def testX_cpu(self, device='cpu')
#   def textX_cuda(self, device='cuda')
#   def testX_xla(self, device='xla')
#   ...
# And if you accept a dtype
#   @dtypes(torch.float, torch.double)
#   def testX(self, device, dtype)
# It becomes
#   def testX_cpu_float32(self, device='cpu', dtype=torch.float32)
#   def testX_cpu_float64(self, device='cpu', dtype=torch.float64)
#   def testX_cuda_float32(self, device='cuda', dtype=torch.float32)
#   ...
#
# These tests can be run directly like normal tests:
# "python test_torch.py TestTorchDeviceTypeCPU.test_diagonal_cpu"
# "python test_torch.py TestTorchDeviceTypeCUDA.test_neg_cuda_float64"
#
# Collections of tests can be run using pytest filtering. For example,
# "pytest test_torch.py -k 'test_diag'"
# will run test_diag on every device (and with every dtype, if applicable).
# To specify particular device types or dtypes the 'and' keyword can be used:
# "pytest test_torch.py -k 'test_diag and cpu'"
# "pytest test_torch.py -k 'test_neg and cuda and float'"
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
#   (5) (Optional) Write setUpClass/tearDownClass class methods that
#       instantiate dependencies (see MAGMA in CUDATestBase).
#
# setUpClass is called AFTER tests have been created and BEFORE and ONLY IF
# they are run. This makes it useful for initializing devices and dependencies.
#

# List of device type test bases that can be used to instantiate tests.
# See below for how this list is populated. If you're adding a device type
# you should check if it's available and (if it is) and it to this list.
device_type_test_bases = []

# List of all Torch dtypes.
all_dtypes = [torch.float64, torch.float32, torch.float16,
              torch.int64, torch.int32, torch.int16, torch.int8,
              torch.uint8, torch.bool]

class DeviceTypeTestBase(TestCase):
    device_type = "generic_device_type"
    supported_dtypes = all_dtypes

    # Returns the dtypes the test has requested.
    # Prefers device-specific dtype specifications over generic ones.
    @classmethod
    def _get_dtypes(cls, test):
        if not hasattr(test, 'dtypes'):
            return None
        d = getattr(test, 'dtypes')
        if cls.device_type in d:
            return d[cls.device_type]
        return d.get('all', None)

    # Raises unittest.SkipTest if the dtype in not in supported_dtypes.
    def _skip_if_unsupported_dtype(self, dtype):
        if dtype not in self.supported_dtypes:
            reason = "unsupported dtype '{0}'".format(str(dtype))
            raise unittest.SkipTest(reason)

    # Creates device-specific tests.
    @classmethod
    def instantiate_test(cls, test):
        test_name = test.__name__ + "_" + cls.device_type

        dtypes = cls._get_dtypes(test)
        if dtypes is None or len(dtypes) == 0:
            # Test has no dtype variants
            @wraps(test)
            def instantiated_test(self, test=test):
                return test(self, cls.device_type)

            setattr(cls, test_name, instantiated_test)
        else:
            # Test has dtype variants
            for dtype in dtypes:
                dtype_str = str(dtype).rsplit('.', maxsplit=1)[1]
                dtype_test_name = test_name + "_" + dtype_str

                @wraps(test)
                def instantiated_test(self, test=test, dtype=dtype):
                    self._skip_if_unsupported_dtype(dtype)
                    return test(self, cls.device_type, dtype)

                setattr(cls, dtype_test_name, instantiated_test)

class CPUTestBase(DeviceTypeTestBase):
    device_type = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.has_lapack = torch._C.has_lapack

class CUDATestBase(DeviceTypeTestBase):
    device_type = "cuda"
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @classmethod
    def setUpClass(cls):
        # has_magma shows up after cuda is initialized
        torch.ones(1).cuda()
        cls.has_magma = torch.cuda.has_magma

# Adds available device-type-specific test base classes
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

    # Checks that the generic test suite only has test members
    # Note: for Python2 compat.
    # Note: Nontest members can be inherited, so if you want to use a helper
    #   function you can put it in a base class.
    generic_nontests = generic_members - set(generic_tests)
    assert len(generic_nontests) == 0, "Generic device class has non-test members"

    for base in device_type_test_bases:
        # Creates the device-specific test case
        class_name = generic_test_class.__name__ + base.device_type.upper()
        device_type_test_class = type(class_name, (base, empty_class), {})

        for name in generic_tests:
            # Attempts to acquire a function from the attribute
            test = getattr(generic_test_class, name)
            if hasattr(test, '__func__'):
                test = test.__func__
            assert inspect.isfunction(test), "Couldn't extract function from '{0}'".format(name)
            # Instantiates the device-specific tests
            device_type_test_class.instantiate_test(test)

        # Mimics defining the instantiated class in the caller's file
        # by setting its module to the given class's and adding
        # the module to the given scope.
        # This lets the instantiated class be discovered by unittest.
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class

# Decorator that specifies a variant of the test for each of the given
# dtypes should be instantiated.
# Notes:
#   (1) Tests that accept the dtype argument MUST use this decorator, the
#       allDtypes decorator, or the allDtypesExcept decorator so the test will
#       work with new device types.
#   (2) Can be overriden using dtypesIfCPU or dtypesIfCUDA.
#   (3) Prefer the existing decorators to using kwargs here.
class dtypes(object):

    # Note: *args, **kwargs for Python2 compat.
    # Python 3 allows (self, *args, device_type='all).
    def __init__(self, *args, **kwargs):
        assert all(arg in all_dtypes for arg in args), "Unknown dtype in {0}".format(str(args))
        self.args = args
        self.device_type = kwargs.get('device_type', 'all')

    def __call__(self, fn):
        d = getattr(fn, 'dtypes', {})
        assert self.device_type not in d, "dtypes redefinition for {0}".format(self.device_type)
        d[self.device_type] = self.args
        setattr(fn, 'dtypes', d)
        return fn

# Sugar that requests a variant of the test for each dtype.
def allDtypes(fn):
    return dtypes(*all_dtypes)(fn)

# Sugar that requests a variant of the test for each dtype EXCEPT those given.
class allDtypesExcept(dtypes):

    def __init__(self, *args):
        filtered = [x for x in all_dtypes if x not in args]
        super(allDtypesExcept, self).__init__(*filtered)

# Overrides specified dtypes on the CPU.
class dtypesIfCPU(dtypes):

    def __init__(self, *args):
        super(dtypesIfCPU, self).__init__(*args, device_type='cpu')

# Overrides specified dtypes on CUDA.
class dtypesIfCUDA(dtypes):

    def __init__(self, *args):
        super(dtypesIfCUDA, self).__init__(*args, device_type='cuda')

# Decorator that specifies a test dependency.
# Notes:
#   (1) Dependencies stack. Multiple dependencies are all evaluated.
#   (2) Dependencies can either be bools or strings. If a string the
#       test base must have defined the corresponding attribute to be True
#       for the test to run. If you want to use a string argument you should
#       probably define a new decorator instead (see below).
#   (3) Prefer the existing decorators to defining the 'device_type' kwarg.
class skipIf(object):

    def __init__(self, dep, reason, device_type='all'):
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def dep_fn(slf, device, *args, **kwargs):
            if self.device_type == 'all' or self.device_type == slf.device_type:
                if not self.dep or (isinstance(self.dep, str) and not getattr(slf, self.dep, False)):
                    raise unittest.SkipTest(self.reason)

            return fn(slf, device, *args, **kwargs)
        return dep_fn

# Specifies a CPU dependency.
class skipCPUIf(skipIf):

    def __init__(self, dep, reason):
        super(skipCPUIf, self).__init__(dep, reason, device_type='cpu')

# Specifies a CUDA dependency.
class skipCUDAIf(skipIf):

    def __init__(self, dep, reason):
        super(skipCUDAIf, self).__init__(dep, reason, device_type='cuda')

# Specifies LAPACK as a CPU dependency.
def skipCPUIfNoLapack(fn):
    return skipCPUIf(torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)

# Specifies MAGMA as a CUDA dependency.
def skipCUDAIfNoMagma(fn):
    return skipCUDAIf('has_magma', "no MAGMA library detected")(fn)
