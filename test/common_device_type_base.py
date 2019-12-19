from common_utils import TestCase
from functools import wraps
import threading

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
                    try:
                        self.precision = self._get_precision_override(test, dtype)
                        result = test(self, device_arg, dtype)
                    finally:
                        self.precision = guard_precision

                    return result

                setattr(cls, dtype_test_name, instantiated_test)
