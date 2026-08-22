# Owner(s): ["module: inductor"]

import os
from unittest import mock

import torch
from torch._inductor import cpp_builder
from torch._inductor.test_case import run_tests, TestCase


# A fake, forward-slashed path so assertions are stable across host platforms.
ROCM_CLANG_CL = "/opt/rocm/lib/llvm/bin/clang-cl.exe"


class TestGetCppCompilerWindows(TestCase):
    def _get_compiler(self, hip, cxx=None):
        env = dict(os.environ)
        env.pop("CXX", None)
        if cxx is not None:
            env["CXX"] = cxx

        with (
            mock.patch.object(cpp_builder, "_IS_WINDOWS", True),
            mock.patch.object(torch.version, "hip", hip),
            mock.patch.object(cpp_builder, "check_compiler_exist_windows"),
            mock.patch.object(
                cpp_builder, "check_msvc_cl_language_id"
            ) as mock_lang_check,
            mock.patch.object(
                cpp_builder,
                "_get_rocm_clang_cl_windows",
                return_value=ROCM_CLANG_CL,
            ),
            mock.patch.object(
                cpp_builder,
                "_is_msvc_cl",
                side_effect=lambda compiler: os.path.basename(compiler).lower() == "cl",
            ),
            mock.patch.dict(os.environ, env, clear=True),
        ):
            compiler = cpp_builder.get_cpp_compiler()
        return compiler, mock_lang_check

    def test_rocm_build_defaults_to_clang_cl(self):
        compiler, mock_lang_check = self._get_compiler(hip="7.14.0")
        self.assertIn("clang-cl", compiler)
        mock_lang_check.assert_not_called()

    def test_non_rocm_build_defaults_to_cl(self):
        compiler, mock_lang_check = self._get_compiler(hip=None)
        self.assertEqual(compiler, "cl")
        mock_lang_check.assert_called_once_with("cl")

    def test_cxx_env_overrides_rocm_default(self):
        compiler, _ = self._get_compiler(hip="7.14.0", cxx="my-clang-cl")
        self.assertEqual(compiler, "my-clang-cl")

    def test_cxx_env_overrides_non_rocm_default(self):
        compiler, _ = self._get_compiler(hip=None, cxx="icx")
        self.assertEqual(compiler, "icx")


class TestGetCppTorchDeviceOptionsWindows(TestCase):
    def test_rocm_link_libtorch_links_amdhip64(self):
        with (
            mock.patch.object(cpp_builder, "_IS_WINDOWS", True),
            mock.patch.object(torch.version, "hip", "7.14.0"),
            mock.patch.object(cpp_builder.config.aot_inductor, "link_libtorch", True),
            mock.patch("torch.utils.cpp_extension.include_paths", return_value=[]),
            mock.patch("torch.utils.cpp_extension.library_paths", return_value=[]),
        ):
            options = cpp_builder.get_cpp_torch_device_options("cuda")

        libraries = options[5]
        self.assertIn("torch_hip", libraries)
        self.assertIn("amdhip64", libraries)


if __name__ == "__main__":
    run_tests()
