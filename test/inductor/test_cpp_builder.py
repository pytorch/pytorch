# Owner(s): ["oncall: cpu inductor"]

import warnings

from torch._inductor.cpp_builder import (
    _cpp_device_options_registry,
    get_cpp_torch_device_options,
    register_cpp_device_options,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCppBuilder(TestCase):
    def tearDown(self) -> None:
        _cpp_device_options_registry.clear()

    def test_register_cpp_device_options(self) -> None:
        def mock_options(device_type, aot_mode, compile_only):
            self.assertEqual(device_type, "mock")
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

        register_cpp_device_options("mock", mock_options)
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

    def test_register_duplicate_warns(self) -> None:
        def mock_options(device_type, aot_mode, compile_only):
            return ([], [], [], [], [], [], [])

        register_cpp_device_options("mock", mock_options)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_cpp_device_options("mock", mock_options)
        self.assertEqual(len(w), 1)
        self.assertIn("already registered", str(w[0].message))

    def test_unregistered_device_options_unchanged(self) -> None:
        result = get_cpp_torch_device_options("unregistered_device_xyz", False, False)
        definitions, _inc, cflags, _ld, _libdirs, libraries, passthrough = result
        self.assertEqual(definitions, [])
        self.assertEqual(cflags, [])
        self.assertEqual(libraries, [])
        self.assertEqual(passthrough, [])


if __name__ == "__main__":
    run_tests()
