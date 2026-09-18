# Owner(s): ["module: inductor", "oncall: cpu inductor"]

import ntpath
import os
from unittest import mock

import torch
from torch._dynamo.device_interface import (
    device_interfaces,
    DeviceInterface,
    register_interface_for_device,
)
from torch._inductor import cpp_builder
from torch._inductor.cpp_builder import get_cpp_torch_device_options
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)


THEROCK_ROCM_HOME = "C:/site-packages/_rocm_sdk_core"
THEROCK_CLANG_CL_CANDIDATE = os.path.join(
    THEROCK_ROCM_HOME, "lib", "llvm", "bin", "clang-cl.exe"
)
THEROCK_CLANG_CL_WINDOWS = ntpath.join(
    THEROCK_ROCM_HOME, "lib", "llvm", "bin", "clang-cl.exe"
)
THEROCK_CLANG_CL = THEROCK_CLANG_CL_WINDOWS.replace("\\", "/")


class TestGetRocmClangClWindows(TestCase):
    def setUp(self):
        super().setUp()
        cpp_builder._get_rocm_clang_cl_windows.cache_clear()

    def tearDown(self):
        cpp_builder._get_rocm_clang_cl_windows.cache_clear()
        super().tearDown()

    def test_uses_rocm_home_clang_cl_when_present(self):
        with (
            mock.patch("torch.utils.cpp_extension.ROCM_HOME", THEROCK_ROCM_HOME),
            mock.patch.object(
                cpp_builder.os.path,
                "exists",
                side_effect=lambda path: path == THEROCK_CLANG_CL_CANDIDATE,
            ) as exists,
        ):
            compiler = cpp_builder._get_rocm_clang_cl_windows()

        self.assertEqual(compiler, THEROCK_CLANG_CL_CANDIDATE)
        exists.assert_called_once_with(THEROCK_CLANG_CL_CANDIDATE)

    def test_warns_and_falls_back_when_clang_cl_is_missing(self):
        with (
            mock.patch("torch.utils.cpp_extension.ROCM_HOME", THEROCK_ROCM_HOME),
            mock.patch.object(cpp_builder.os.path, "exists", return_value=False),
            self.assertLogs(cpp_builder.log, level="WARNING") as logs,
        ):
            compiler = cpp_builder._get_rocm_clang_cl_windows()

        self.assertEqual(compiler, "clang-cl")
        self.assertTrue(any(THEROCK_CLANG_CL_CANDIDATE in line for line in logs.output))

    def test_warns_and_falls_back_when_rocm_home_is_unset(self):
        with (
            mock.patch("torch.utils.cpp_extension.ROCM_HOME", None),
            self.assertLogs(cpp_builder.log, level="WARNING") as logs,
        ):
            compiler = cpp_builder._get_rocm_clang_cl_windows()

        self.assertEqual(compiler, "clang-cl")
        self.assertTrue(any("ROCM_HOME is not set" in line for line in logs.output))


@instantiate_parametrized_tests
class TestGetCppCompilerWindows(TestCase):
    @parametrize(
        "hip,device_type,cxx,expected,uses_rocm_default",
        [
            subtest(
                ("7.14.0", "cuda", None, THEROCK_CLANG_CL, True),
                name="rocm_cuda_default",
            ),
            subtest(
                ("7.14.0", "cuda:0", None, THEROCK_CLANG_CL, True),
                name="rocm_indexed_cuda_default",
            ),
            subtest(
                ("7.14.0", "cpu", None, "cl", False),
                name="rocm_cpu_default",
            ),
            subtest(
                ("7.14.0", None, None, "cl", False),
                name="rocm_no_device_default",
            ),
            subtest(
                (None, "cuda", None, "cl", False),
                name="non_rocm_default",
            ),
            subtest(
                ("7.14.0", "cuda", "my-clang-cl", "my-clang-cl", False),
                name="rocm_cxx_override",
            ),
            subtest(
                (None, None, "icx", "icx", False),
                name="non_rocm_cxx_override",
            ),
        ],
    )
    def test_compiler_selection(
        self, hip, device_type, cxx, expected, uses_rocm_default
    ):
        env = dict(os.environ)
        env.pop("CXX", None)
        if cxx is not None:
            env["CXX"] = cxx

        with (
            mock.patch.object(cpp_builder, "_IS_WINDOWS", True),
            mock.patch.object(cpp_builder.os, "sep", "\\"),
            mock.patch.object(torch.version, "hip", hip),
            mock.patch.object(cpp_builder, "check_compiler_exist_windows"),
            mock.patch.object(cpp_builder, "check_msvc_cl_language_id") as lang_check,
            mock.patch.object(
                cpp_builder,
                "_get_rocm_clang_cl_windows",
                return_value=THEROCK_CLANG_CL_WINDOWS,
            ) as rocm_compiler,
            mock.patch.dict(os.environ, env, clear=True),
        ):
            compiler = cpp_builder.get_cpp_compiler(device_type=device_type)

        self.assertEqual(compiler, expected)
        lang_check.assert_called_once_with(expected)
        if uses_rocm_default:
            rocm_compiler.assert_called_once_with()
        else:
            rocm_compiler.assert_not_called()

    def test_rejects_invalid_device(self):
        with (
            mock.patch.object(cpp_builder, "_IS_WINDOWS", True),
            mock.patch.object(torch.version, "hip", "7.14.0"),
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaises(RuntimeError),
        ):
            cpp_builder.get_cpp_compiler(device_type="cuda:invalid")


class TestMsvcLanguageCheck(TestCase):
    def tearDown(self):
        cpp_builder._is_msvc_cl.cache_clear()
        cpp_builder.check_msvc_cl_language_id.cache_clear()
        super().tearDown()

    def test_clang_cl_skips_msvc_language_pack_check(self):
        result = mock.Mock(stdout=b"OVERVIEW: clang LLVM compiler\n")
        with (
            mock.patch.object(cpp_builder, "_IS_WINDOWS", True),
            mock.patch.object(cpp_builder.subprocess, "run", return_value=result),
            mock.patch.object(cpp_builder, "WinPeFileVersionInfo") as version_info,
        ):
            cpp_builder._is_msvc_cl.cache_clear()
            cpp_builder.check_msvc_cl_language_id.cache_clear()
            cpp_builder.check_msvc_cl_language_id("clang-cl.exe")

        version_info.assert_not_called()


class TestCppTorchDeviceOptionsCompiler(TestCase):
    class StopAfterSuperCall(Exception):
        pass

    def test_resolves_default_compiler_for_device(self):
        with (
            mock.patch.object(
                cpp_builder, "get_cpp_compiler", return_value=THEROCK_CLANG_CL
            ) as get_compiler,
            mock.patch.object(
                cpp_builder.CppTorchOptions,
                "__init__",
                side_effect=self.StopAfterSuperCall,
            ) as init,
            self.assertRaises(self.StopAfterSuperCall),
        ):
            cpp_builder.CppTorchDeviceOptions(device_type="cuda")

        get_compiler.assert_called_once_with(device_type="cuda")
        self.assertEqual(init.call_args.kwargs["compiler"], THEROCK_CLANG_CL)

    def test_preserves_explicit_compiler(self):
        with (
            mock.patch.object(cpp_builder, "get_cpp_compiler") as get_compiler,
            mock.patch.object(
                cpp_builder.CppTorchOptions,
                "__init__",
                side_effect=self.StopAfterSuperCall,
            ) as init,
            self.assertRaises(self.StopAfterSuperCall),
        ):
            cpp_builder.CppTorchDeviceOptions(
                device_type="cuda", compiler="custom-clang-cl"
            )

        get_compiler.assert_not_called()
        self.assertEqual(init.call_args.kwargs["compiler"], "custom-clang-cl")

    def test_normalizes_before_validation(self):
        with (
            mock.patch.object(cpp_builder, "_validate_cpp_stdlib") as validate,
            mock.patch.object(cpp_builder, "get_cpp_compiler") as get_compiler,
            mock.patch.object(
                cpp_builder.CppTorchOptions,
                "__init__",
                side_effect=self.StopAfterSuperCall,
            ),
            self.assertRaises(self.StopAfterSuperCall),
        ):
            cpp_builder.CppTorchDeviceOptions(device_type="cpu:0")

        validate.assert_called_once_with("libstdc++", "cpu", False)
        get_compiler.assert_called_once_with(device_type="cpu")


@instantiate_parametrized_tests
class TestGetCppTorchDeviceOptionsWindows(TestCase):
    @parametrize("device_type", ["cuda", "cuda:0"])
    def test_rocm_link_libtorch_links_amdhip64(self, device_type):
        with (
            mock.patch.object(cpp_builder, "_IS_WINDOWS", True),
            mock.patch.object(torch.version, "hip", "7.14.0"),
            mock.patch.object(cpp_builder.config.aot_inductor, "link_libtorch", True),
            mock.patch(
                "torch.utils.cpp_extension.include_paths", return_value=[]
            ) as include_paths,
            mock.patch(
                "torch.utils.cpp_extension.library_paths", return_value=[]
            ) as library_paths,
        ):
            (
                definitions,
                _include_dirs,
                _cflags,
                _ldflags,
                _libraries_dirs,
                libraries,
                _passthrough_args,
            ) = cpp_builder.get_cpp_torch_device_options(device_type)

        include_paths.assert_called_once_with("cuda", False)
        library_paths.assert_called_once_with(
            "cuda", torch_include_dirs=True, cross_target_platform=None
        )
        self.assertIn(" USE_ROCM", definitions)
        self.assertIn("torch_hip", libraries)
        self.assertIn("amdhip64", libraries)


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
