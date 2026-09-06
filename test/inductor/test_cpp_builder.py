# Owner(s): ["oncall: cpu inductor"]

from torch._dynamo.device_interface import (
    device_interfaces,
    DeviceInterface,
    register_interface_for_device,
)
from torch._inductor.cpp_builder import get_cpp_torch_device_options
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCppBuilder(TestCase):
    def test_device_interface_cpp_options(self) -> None:
        class MockDeviceInterface(DeviceInterface):
            @staticmethod
            def get_cpp_device_options(aot_mode, compile_only):
                self.assertFalse(aot_mode)
                self.assertFalse(compile_only)
                return (
                    ["USE_MOCK"],
                    ["/mock/include"],
                    ["mock_cflag"],
                    ["mock_ldflag"],
                    ["/mock/lib"],
                    ["mocklib"],
                    ["-DMOCK_PASSTHROUGH"],
                )

        self.addCleanup(device_interfaces.pop, "mock", None)
        register_interface_for_device("mock", MockDeviceInterface)
        result = get_cpp_torch_device_options("mock", False, False)
        self.assertEqual(
            result,
            (
                ["USE_MOCK"],
                ["/mock/include"],
                ["mock_cflag"],
                ["mock_ldflag"],
                ["/mock/lib"],
                ["mocklib"],
                ["-DMOCK_PASSTHROUGH"],
            ),
        )

    def test_device_interface_without_cpp_options_uses_default_options(self) -> None:
        class MockDeviceInterface(DeviceInterface):
            pass

        self.addCleanup(device_interfaces.pop, "mock_no_options", None)
        register_interface_for_device("mock_no_options", MockDeviceInterface)
        result = get_cpp_torch_device_options("mock_no_options", False, False)
        definitions, _inc, cflags, _ld, _libdirs, libraries, passthrough = result
        self.assertEqual(definitions, [])
        self.assertEqual(cflags, [])
        self.assertEqual(libraries, [])
        self.assertEqual(passthrough, [])


if __name__ == "__main__":
    run_tests()
