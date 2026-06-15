# Owner(s): ["module: inductor"]
import os
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import _dynamo as dynamo, _inductor as inductor
from torch._inductor import config
from torch._inductor.codecache import _cuda_fatbin_command, write
from torch._inductor.cpp_builder import (
    BuildOptionsBase,
    CppBuilder,
    CppOptions,
    CppTorchOptions,
)
from torch._inductor.cpu_vec_isa import invalid_vec_isa
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import gen_gm_and_inputs
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

    @mock.patch.dict(
        os.environ,
        {"TORCH_CUDA_ARCH_LIST": "7.0;8.0;8.6;9.0+PTX"},
    )
    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_multi_arch_gencode_options(self):
        from torch._inductor.codegen.cuda import compile_utils

        with self.assertLogs(
            "torch._inductor.codegen.cuda.compile_utils", level="WARNING"
        ) as log_ctx:
            self.assertEqual(
                compile_utils._cuda_multi_arch_gencode_options("80"),
                [
                    "arch=compute_80,code=sm_80",
                    "arch=compute_80,code=compute_80",
                    "arch=compute_86,code=sm_86",
                    "arch=compute_90,code=sm_90",
                ],
            )
        self.assertIn("Ignoring TORCH_CUDA_ARCH_LIST entry sm_70", log_ctx.output[0])

    @mock.patch.dict(
        os.environ,
        {"TORCH_CUDA_ARCH_LIST": "9.0;9.0a;10.0"},
    )
    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_multi_arch_gencode_options_suffix_arch(self):
        from torch._inductor.codegen.cuda import compile_utils

        self.assertEqual(
            compile_utils._cuda_multi_arch_gencode_options("90a"),
            [
                "arch=compute_90a,code=sm_90a",
                "arch=compute_90a,code=compute_90a",
            ],
        )

    @mock.patch(
        "torch._inductor.codegen.cuda.compile_utils._nvcc_arch_as_compile_option",
        return_value="100a",
    )
    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_target_arch_strips_suffix(self, _):
        from torch._inductor.codegen.cuda import compile_utils

        self.assertEqual(compile_utils._aoti_cuda_target_arch(), "100")
        with config.patch({"cuda.arch": "90a"}):
            self.assertEqual(compile_utils._aoti_cuda_target_arch(), "90")

    @mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.0;8.6"})
    @mock.patch(
        "torch._inductor.codegen.cuda.compile_utils._nvcc_arch_as_compile_option",
        return_value="100a",
    )
    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_fatbin_command_uses_nvcc_for_extra_archs(self, _):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_cubin = os.path.join(tmp_dir, "kernel.cubin")
            with open(raw_cubin, "wb"):
                pass
            cmd = _cuda_fatbin_command(
                "kernel.ptx",
                "kernel.fatbin",
                raw_cubin,
                "nvcc",
                "fatbinary",
                "80",
            )

        self.assertEqual(
            cmd[:5], ["nvcc", "-fatbin", "kernel.ptx", "-o", "kernel.fatbin"]
        )
        self.assertIn("arch=compute_80,code=sm_80", cmd)
        self.assertIn("arch=compute_86,code=sm_86", cmd)
        self.assertNotIn("arch=compute_100a,code=sm_100a", cmd)

    @mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "7.0;8.0;8.6;9.0"})
    @mock.patch(
        "torch._inductor.codegen.cuda.compile_utils._nvcc_arch_as_compile_option",
        return_value="100a",
    )
    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_cmake_uses_multi_arch_gencode_flags(self, _):
        build_option = BuildOptionsBase(compiler="c++")
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmake_path = os.path.join(tmp_dir, "CMakeLists.txt")
            cpp_builder = CppBuilder(
                name="test_compile",
                sources=[],
                output_dir=tmp_dir,
                BuildOption=build_option,
            )
            with config.patch({"cuda.arch": "80"}):
                cpp_builder.save_compile_cmd_to_cmake(cmake_path, "cuda")
            with open(cmake_path) as f:
                cmake_contents = f.read()

        self.assertNotIn("compute_70", cmake_contents)
        self.assertNotIn("compute_100a", cmake_contents)
        self.assertIn("-gencode arch=compute_80,code=sm_80", cmake_contents)
        self.assertIn("-gencode arch=compute_86,code=sm_86", cmake_contents)
        self.assertIn("-gencode arch=compute_90,code=sm_90", cmake_contents)

    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_save_kernel_recompiles_for_target_arch(self):
        from torch._inductor.runtime.triton_heuristics import (
            CachingAutotuner,
            TritonCompileResult,
        )

        autotuner = object.__new__(CachingAutotuner)
        autotuner.inductor_meta = {"kernel_name": "triton_kernel"}
        autotuner.triton_meta = {}
        autotuner.device_props = types.SimpleNamespace(type="cuda", cc=100)

        current_binary = types.SimpleNamespace(
            metadata=types.SimpleNamespace(name="kernel", num_warps=1, shared=0),
            asm={"cubin": b"current cubin", "ptx": "current ptx"},
        )
        target_binary = types.SimpleNamespace(
            metadata=types.SimpleNamespace(name="kernel", num_warps=1, shared=0),
            asm={"cubin": b"target cubin", "ptx": "target ptx"},
        )
        target_result = object.__new__(TritonCompileResult)
        target_result.kernel = target_binary

        launcher = types.SimpleNamespace(
            bin=current_binary,
            config=types.SimpleNamespace(kwargs={}, num_warps=1, num_stages=1),
            def_args=[],
            call_args=[],
            global_scratch=None,
            profile_scratch=None,
        )

        with (
            config.patch(
                {
                    "aot_inductor.emit_multi_arch_kernel": True,
                    "cuda.arch": "90a",
                }
            ),
            mock.patch.object(
                CachingAutotuner,
                "_precompile_config",
                return_value=target_result,
            ) as precompile_config,
            mock.patch(
                "torch._inductor.codecache.CudaKernelParamCache.set"
            ) as cache_set,
        ):
            autotuner.save_gpu_kernel("stream", launcher)

        precompile_config.assert_called_once_with(launcher.config, cc_override=90)
        _, params, cubin, bin_type, asm, asm_type = cache_set.call_args.args
        self.assertEqual(params["cuda_arch"], "90")
        self.assertEqual(cubin, b"target cubin")
        self.assertEqual(bin_type, "cubin")
        self.assertEqual(asm, "target ptx")
        self.assertEqual(asm_type, "ptx")

        with (
            config.patch(
                {
                    "aot_inductor.emit_multi_arch_kernel": True,
                    "cuda.arch": None,
                }
            ),
            mock.patch(
                "torch._inductor.codegen.cuda.compile_utils._nvcc_arch_as_compile_option",
                return_value="100a",
            ),
            mock.patch.object(
                CachingAutotuner,
                "_precompile_config",
                return_value=target_result,
            ) as precompile_config,
            mock.patch(
                "torch._inductor.codecache.CudaKernelParamCache.set"
            ) as cache_set,
        ):
            autotuner.save_gpu_kernel("stream", launcher)

        precompile_config.assert_not_called()
        _, params, cubin, bin_type, asm, asm_type = cache_set.call_args.args
        self.assertEqual(params["cuda_arch"], "100")
        self.assertEqual(cubin, b"current cubin")
        self.assertEqual(bin_type, "cubin")
        self.assertEqual(asm, "current ptx")
        self.assertEqual(asm_type, "ptx")


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()
