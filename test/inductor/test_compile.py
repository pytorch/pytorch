# Owner(s): ["module: inductor"]
import os
import shlex
import subprocess
import sys
import types
import unittest
from unittest import mock

import torch
from torch import _dynamo as dynamo, _inductor as inductor
from torch._inductor import config, cpp_builder as cpp_builder_module, cpu_vec_isa
from torch._inductor.codecache import write
from torch._inductor.cpp_builder import CppBuilder, CppOptions, CppTorchOptions
from torch._inductor.cpu_vec_isa import invalid_vec_isa
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import gen_gm_and_inputs, run_and_get_code
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.inductor_utils import HAS_CPU


_IS_MACOS = sys.platform.startswith("darwin")
_IS_WINDOWS = sys.platform == "win32"


def safe_command_output(cmd, timeout=30):
    try:
        return subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            shell=isinstance(cmd, str),
        ).strip()
    except subprocess.CalledProcessError as e:
        return f"run failed（error code {e.returncode}）: {e.output.strip()}"
    except subprocess.TimeoutExpired:
        return "runt timeout"


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(10, 10)
        self.b = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.a(x))
        x = torch.sigmoid(self.b(x))
        return x


class MyModule2(MyModule):
    def forward(self, x):  # takes a dict of list
        a, b = x["key"]
        return {"result": super().forward(a) + b}


class MyModule3(MyModule):
    def forward(self, x):
        return (super().forward(x),)


class TestStandaloneInductor(TestCase):
    """
    These test check that you can call TorchInductor directly without
    going through TorchDynamo.
    """

    def test_inductor_via_fx(self):
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_tensor_return(self):
        mod = MyModule().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_dict_input(self):
        mod = MyModule2().eval()
        inp = {"key": [torch.randn(10), torch.randn(10)]}
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_make_fx(self):
        mod = MyModule().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(make_fx(mod)(inp), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_bare_module(self):
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        # no FX graph at all (mod must return list/tuple in this case)
        mod_opt = inductor.compile(mod, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export1(self):
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        gm, _ = dynamo.export(mod, inp, aten_graph=True)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export2(self):
        mod = MyModule2().eval()
        inp = {"key": [torch.randn(10), torch.randn(10)]}
        correct = mod(inp)
        gm, _ = dynamo.export(mod, inp)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_op_with_multiple_outputs(self):
        x1 = torch.randn((2, 512, 128))
        x2 = [128]
        x3 = torch.randn(128)
        x4 = torch.randn((128,))
        x5 = 1e-6
        mod, inp = gen_gm_and_inputs(
            torch.ops.aten.native_layer_norm.default, (x1, x2, x3, x4, x5), {}
        )
        mod_opt = inductor.compile(mod, inp)
        self.assertEqual(mod(*inp), mod_opt(*inp))

    @mock.patch.dict(os.environ, {"TORCHINDUCTOR_DEBUG_COMPILE": "1"})
    def test_inductor_generate_debug_compile(self):
        cpp_code = """
        int main(){
            return 0;
        }
        """

        _, source_path = write(
            cpp_code,
            "cpp",
        )
        build_option = CppOptions()
        cpp_builder = CppBuilder(
            name="test_compile",
            sources=source_path,
            output_dir=os.path.dirname(source_path),
            BuildOption=build_option,
        )
        cpp_builder.build()
        binary_path = cpp_builder.get_target_file_path()

        """
        When we turn on generate debug compile.
        On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
        On Linux, it should create some debug sections in binary file.
        """

        def check_linux_debug_section(module_path: str):
            check_cmd = shlex.split(f"readelf -S {module_path}")
            output = safe_command_output(check_cmd)
            has_debug_sym = ".debug_info" in output
            self.assertEqual(has_debug_sym, True)

        def check_windows_pdb_exist(module_path: str):
            file_name_no_ext = os.path.splitext(module_path)[0]
            file_name_pdb = f"{file_name_no_ext}.pdb"
            has_pdb_file = os.path.exists(file_name_pdb)
            self.assertEqual(has_pdb_file, True)

        if _IS_WINDOWS:
            check_windows_pdb_exist(binary_path)
        elif _IS_MACOS:
            pass  # MacOS not sure that if it should be works.
        else:
            check_linux_debug_section(binary_path)

    def test_cpp_prefix_vectorized_bool_mask_cast(self):
        vec_isa = cpu_vec_isa.pick_vec_isa()
        if not vec_isa:
            self.skipTest("requires CPU vectorization")

        cpp_code = """
        #include <torch/csrc/inductor/cpp_prefix.h>
        int main() {
        #if INDUCTOR_USE_VECTOR_TYPES()
          __at_align__ bool in[at::vec::Vectorized<bool>::size()] = {};
          __at_align__ bool out[at::vec::Vectorized<bool>::size()] = {};
          auto mask = at::vec::Vectorized<bool>::loadu(
              in, at::vec::Vectorized<bool>::size());
          auto casted = inductor_vec_mask_cast<float, 1>(mask);
          casted.store(out, at::vec::Vectorized<bool>::size());
        #endif
          return 0;
        }
        """

        _, source_path = write(cpp_code, "cpp")
        cpp_builder = CppBuilder(
            name="test_vectorized_bool_mask_cast",
            sources=source_path,
            output_dir=os.path.dirname(source_path),
            BuildOption=CppTorchOptions(vec_isa=vec_isa),
        )
        cpp_builder.build()

    def test_fbcode_local_build_uses_absolute_linker_script(self):
        fake_build_paths = types.SimpleNamespace(
            cc_include="cc_include",
            glibc_include="glibc_include",
            glibc_lib="glibc_lib",
            libgcc_arch_include="libgcc_arch_include",
            libgcc_backward_include="libgcc_backward_include",
            libgcc_include="libgcc_include",
            linux_kernel_include="linux_kernel_include",
            openmp_include="openmp_include",
            python_include="python_include",
            sleef_include="sleef_include",
        )

        with (
            mock.patch.object(
                cpp_builder_module,
                "config",
                types.SimpleNamespace(is_fbcode=lambda: True),
            ),
            mock.patch.object(
                cpp_builder_module, "build_paths", fake_build_paths, create=True
            ),
        ):
            _, _, _, local_ldflags = cpp_builder_module._setup_standard_sys_libs(
                "clang++", aot_mode=False, use_relative_path=False
            )
            _, _, _, relative_ldflags = cpp_builder_module._setup_standard_sys_libs(
                "clang++", aot_mode=False, use_relative_path=True
            )

        self.assertIn(f"Wl,--script={cpp_builder_module._LINKER_SCRIPT}", local_ldflags)
        self.assertIn("Wl,--script=script.ld", relative_ldflags)

    def test_cpp_codegen_bool_where_uses_mask_cast_helper(self):
        if not cpu_vec_isa.pick_vec_isa():
            self.skipTest("requires CPU vectorization")

        def fn(a):
            b = a > 0
            c = a < 1
            return torch.where(b, c, ~c)

        x = torch.randn(128)
        result, code = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(result, fn(x))
        self.assertIn("inductor_vec_mask_cast<float,1>", "".join(code).replace(" ", ""))

    @mock.patch.dict(os.environ, {"TORCHINDUCTOR_DEBUG_SYMBOL": "1"})
    def test_inductor_generate_debug_symbol(self):
        cpp_code = """
        int main(){
            return 0;
        }
        """

        _, source_path = write(
            cpp_code,
            "cpp",
        )
        build_option = CppOptions()
        cpp_builder = CppBuilder(
            name="test_symbol",
            sources=source_path,
            output_dir=os.path.dirname(source_path),
            BuildOption=build_option,
        )
        cpp_builder.build()
        binary_path = cpp_builder.get_target_file_path()

        """
        When we turn on generate debug symbol.
        On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
        On Linux, it should create some debug sections in binary file.
        """

        def check_linux_debug_section(module_path: str):
            check_cmd = shlex.split(f"readelf -S {module_path}")
            output = safe_command_output(check_cmd)
            has_debug_sym = ".debug_info" in output
            self.assertEqual(has_debug_sym, True)

        def check_windows_pdb_exist(module_path: str):
            file_name_no_ext = os.path.splitext(module_path)[0]
            file_name_pdb = f"{file_name_no_ext}.pdb"
            has_pdb_file = os.path.exists(file_name_pdb)
            self.assertEqual(has_pdb_file, True)

        if _IS_WINDOWS:
            check_windows_pdb_exist(binary_path)
        elif _IS_MACOS:
            pass  # MacOS not sure that if it should be works.
        else:
            check_linux_debug_section(binary_path)

    def _aot_cpp_arch_flags(self):
        build_option = CppTorchOptions(
            aot_mode=True,
            compile_only=True,
            vec_isa=invalid_vec_isa,
        )
        return [
            flag
            for flag in build_option.get_cflags()
            if flag.startswith(("march=", "mcpu="))
        ]

    @unittest.skipIf(config.is_fbcode(), "fbcode does not emit CPU architecture flags")
    def test_aot_cpp_march_config(self):
        with (
            config.patch({"cpp.march": "x86-64"}),
            mock.patch(
                "torch._inductor.cpp_builder.platform.machine",
                return_value="x86_64",
            ),
        ):
            arch_flags = self._aot_cpp_arch_flags()
        self.assertEqual(arch_flags, ["march=x86-64"])

    @unittest.skipIf(config.is_fbcode(), "fbcode does not emit CPU architecture flags")
    def test_aot_cpp_march_config_ppc64le(self):
        with (
            config.patch({"cpp.march": "power9"}),
            mock.patch(
                "torch._inductor.cpp_builder.platform.machine",
                return_value="ppc64le",
            ),
        ):
            arch_flags = self._aot_cpp_arch_flags()
        self.assertEqual(arch_flags, ["mcpu=power9"])

    @unittest.skipIf(config.is_fbcode(), "fbcode does not emit CPU architecture flags")
    def test_cpp_march_config_can_disable_arch_flag(self):
        with config.patch({"cpp.march": ""}):
            arch_flags = self._aot_cpp_arch_flags()
        self.assertEqual(arch_flags, [])


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()
