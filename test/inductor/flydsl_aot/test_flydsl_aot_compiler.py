# Owner(s): ["module: inductor"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import ctypes
import re
import shutil
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import torch
from torch._inductor.codegen.flydsl import flydsl_utils
from torch.testing._internal.common_utils import TestCase


HAS_FLYDSL = torch.cuda.is_available() and flydsl_utils.aot_runtime_available()
if HAS_FLYDSL:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import execution_engine, ir
    from flydsl.compiler import (
        backends,
        jit_executor,
        jit_function,
        kernel_function,
        protocol,
    )
    from flydsl.compiler.jit_argument import PointerJitArg, TorchTensorJitArg

if HAS_FLYDSL:
    from torch._inductor.codegen.flydsl.aot_compile import (
        _argument_abi,
        _bundle_runtime_libraries,
        _publish_runtime_library,
        _rename_export_symbols,
        _runtime_library_dependencies,
        compile_aot,
        CompiledAOTLauncher,
    )


if HAS_FLYDSL:

    @flyc.kernel
    def _aot_test_add_kernel(
        lhs: fx.Tensor,
        rhs: fx.Tensor,
        out: fx.Tensor,
        block_dim: fx.Constexpr[int],
    ):
        block = fx.block_idx.x
        thread = fx.thread_idx.x
        lhs = fx.rocdl.make_buffer_tensor(lhs)
        rhs = fx.rocdl.make_buffer_tensor(rhs)
        out = fx.rocdl.make_buffer_tensor(out)
        tiled_lhs = fx.slice(
            fx.logical_divide(lhs, fx.make_layout(block_dim, 1)),
            (None, block),
        )
        tiled_rhs = fx.slice(
            fx.logical_divide(rhs, fx.make_layout(block_dim, 1)),
            (None, block),
        )
        tiled_out = fx.slice(
            fx.logical_divide(out, fx.make_layout(block_dim, 1)),
            (None, block),
        )
        tiled_lhs = fx.logical_divide(tiled_lhs, fx.make_layout(1, 1))
        tiled_rhs = fx.logical_divide(tiled_rhs, fx.make_layout(1, 1))
        tiled_out = fx.logical_divide(tiled_out, fx.make_layout(1, 1))
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(32), fx.Float32)
        lhs_register = fx.make_rmem_tensor(1, fx.Float32)
        rhs_register = fx.make_rmem_tensor(1, fx.Float32)
        out_register = fx.make_rmem_tensor(1, fx.Float32)
        fx.copy_atom_call(
            copy_atom,
            fx.slice(tiled_lhs, (None, thread)),
            lhs_register,
        )
        fx.copy_atom_call(
            copy_atom,
            fx.slice(tiled_rhs, (None, thread)),
            rhs_register,
        )
        value = fx.arith.addf(
            fx.memref_load_vec(lhs_register),
            fx.memref_load_vec(rhs_register),
        )
        fx.memref_store_vec(value, out_register)
        fx.copy_atom_call(
            copy_atom,
            out_register,
            fx.slice(tiled_out, (None, thread)),
        )

    @flyc.jit
    def _aot_test_add_launcher(
        out: fx.Tensor,
        lhs: fx.Tensor,
        rhs: fx.Tensor,
        elements: fx.Int32,
        block_dim: fx.Constexpr[int],
    ):
        blocks = (elements + block_dim - 1) // block_dim
        _aot_test_add_kernel(lhs, rhs, out, block_dim).launch(
            grid=(blocks, 1, 1),
            block=(block_dim, 1, 1),
        )


class _FakeExecutionEngine:
    module_text = ""
    enable_pic = False

    def __init__(self, module, *, opt_level, shared_libs, enable_pic=False):
        self.__class__.module_text = str(module)
        self.__class__.enable_pic = enable_pic

    def dump_to_object_file(self, path):
        Path(path).write_bytes(b"flydsl-object")


class FlyDSLAOTAvailabilityTest(TestCase):
    def test_aot_runtime_requires_flydsl_0_3_2(self):
        with (
            mock.patch.object(
                flydsl_utils,
                "_flydsl_runtime_unavailable_reason",
                return_value=None,
            ),
            mock.patch.object(
                flydsl_utils,
                "_available_version",
                return_value=SimpleNamespace(release=(0, 3, 1)),
            ),
        ):
            reason = flydsl_utils._flydsl_aot_runtime_unavailable_reason()
        self.assertIn(">=0.3.2", reason)

    def test_aot_runtime_accepts_flydsl_0_3_2(self):
        with (
            mock.patch.object(
                flydsl_utils,
                "_flydsl_runtime_unavailable_reason",
                return_value=None,
            ),
            mock.patch.object(
                flydsl_utils,
                "_available_version",
                return_value=SimpleNamespace(release=(0, 3, 2)),
            ),
        ):
            reason = flydsl_utils._flydsl_aot_runtime_unavailable_reason()
        self.assertIsNone(reason)

    def test_aot_runtime_has_no_upper_version_bound(self):
        package_spec = SimpleNamespace(submodule_search_locations=["package"])
        with (
            mock.patch.object(
                flydsl_utils,
                "find_spec",
                return_value=package_spec,
            ),
            mock.patch.object(
                flydsl_utils,
                "_pathfinder_find_spec",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(
                flydsl_utils,
                "_available_version",
                return_value=SimpleNamespace(release=(1, 0, 0)),
            ),
        ):
            reason = flydsl_utils._flydsl_aot_runtime_unavailable_reason()
        self.assertIsNone(reason)


@unittest.skipUnless(HAS_FLYDSL, "FlyDSL is not available")
class FlyDSLAOTCompilerTest(TestCase):
    @staticmethod
    def _pack_aot_arguments(abi, args):
        storage = []
        for slot in abi:
            arg_index = slot["arg_index"]
            arg = args[arg_index] if arg_index is not None else None
            if slot["kind"] == "tensor_data":
                value = ctypes.c_void_p(arg.data_ptr())
            elif slot["kind"] == "tensor_layout":
                value = ctypes.create_string_buffer(slot["size"])
                offset = 0
                for dim in slot["shape_dims"]:
                    ctypes.c_int32.from_buffer(value, offset).value = arg.shape[dim]
                    offset += ctypes.sizeof(ctypes.c_int32)
                stride_type = (
                    ctypes.c_int32 if slot["stride_bits"] == 32 else ctypes.c_int64
                )
                for dim in slot["stride_dims"]:
                    stride_type.from_buffer(value, offset).value = arg.stride(dim)
                    offset += ctypes.sizeof(stride_type)
            elif slot["kind"] == "scalar":
                scalar_types = {
                    "bool": ctypes.c_bool,
                    "float": ctypes.c_float,
                    "double": ctypes.c_double,
                    **{
                        f"int{bits}": getattr(ctypes, f"c_int{bits}")
                        for bits in (8, 16, 32, 64)
                    },
                    **{
                        f"uint{bits}": getattr(ctypes, f"c_uint{bits}")
                        for bits in (8, 16, 32, 64)
                    },
                }
                value = scalar_types[slot["ctype"]](arg)
            elif slot["kind"] == "stream":
                value = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            else:
                raise AssertionError(f"unsupported test ABI slot: {slot['kind']}")
            storage.append(value)
        packed = (ctypes.c_void_p * len(storage))(
            *(ctypes.addressof(value) for value in storage)
        )
        return packed, storage

    def _compiled_launcher(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = ir.Module.parse(
                """
                module {
                  llvm.func @launcher() {
                    llvm.return
                  }
                }
                """
            )
            return CompiledAOTLauncher(
                module,
                "launcher",
                (
                    {
                        "arg_index": 0,
                        "arg_name": "out",
                        "kind": "tensor_data",
                        "ctype": "pointer",
                        "size": 8,
                        "alignment": 8,
                    },
                ),
            )

    def test_export_uses_packed_entry_and_explicit_module_symbols(self):
        _FakeExecutionEngine.enable_pic = False
        compiled = self._compiled_launcher()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(
                execution_engine,
                "ExecutionEngine",
                _FakeExecutionEngine,
            ),
            mock.patch.object(
                jit_executor,
                "_resolve_runtime_libs",
                return_value=["/runtime/libfly_jit_runtime.so"],
            ),
            mock.patch(
                "torch._inductor.codegen.flydsl.aot_compile._rename_loader_symbols",
                return_value=("launcher__init", "launcher__load"),
            ) as rename_loader_symbols,
            mock.patch(
                "torch._inductor.codegen.flydsl.aot_compile._bundle_runtime_libraries",
                return_value=["/runtime/libfly_jit_runtime.so"],
            ),
        ):
            output = Path(tmpdir) / "kernel.o"
            metadata = compiled.export_to_c(str(output), "flydsl_test_launcher")

            self.assertEqual(b"flydsl-object", output.read_bytes())
            self.assertEqual("_mlir_flydsl_test_launcher", metadata["symbol"])
            self.assertEqual("launcher__init", metadata["module_init_symbol"])
            self.assertEqual("launcher__load", metadata["module_load_symbol"])
            self.assertTrue(_FakeExecutionEngine.enable_pic)
            self.assertIn("@flydsl_test_launcher", _FakeExecutionEngine.module_text)
            self.assertNotIn("@launcher", _FakeExecutionEngine.module_text)
            rename_loader_symbols.assert_called_once_with(
                output,
                "flydsl_test_launcher",
            )

    def test_export_does_not_rename_external_llvm_declarations(self):
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            module = ir.Module.parse(
                """
                module {
                  llvm.func @external()
                  llvm.func @launcher() {
                    llvm.call @external() : () -> ()
                    llvm.return
                  }
                }
                """
            )
            _rename_export_symbols(module, "launcher", "flydsl_test_launcher")
            module_text = str(module)

        self.assertIn("llvm.func @external()", module_text)
        self.assertIn("llvm.call @external()", module_text)
        self.assertNotIn("flydsl_test_launcher__external", module_text)

    def test_real_export_loads_and_matches_jit(self):
        elements = 256
        block_dim = 256
        lhs = torch.arange(elements, device="cuda", dtype=torch.float32)
        rhs = torch.arange(elements, device="cuda", dtype=torch.float32).flip(0)
        jit_out = torch.empty_like(lhs)
        _aot_test_add_launcher(jit_out, lhs, rhs, elements, block_dim)
        torch.cuda.synchronize()

        aot_out = torch.empty_like(lhs)
        compiled = compile_aot(
            _aot_test_add_launcher,
            aot_out,
            lhs,
            rhs,
            elements,
            block_dim,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            object_path = root / "flydsl_test_launcher.o"
            metadata = compiled.export_to_c(
                str(object_path),
                "flydsl_test_launcher",
            )

            nm = shutil.which("llvm-nm") or shutil.which("nm")
            self.assertIsNotNone(nm)
            symbols = subprocess.run(
                [cast(str, nm), "--defined-only", str(object_path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            for symbol in (
                metadata["symbol"],
                metadata["module_init_symbol"],
                metadata["module_load_symbol"],
            ):
                self.assertRegex(symbols, rf"(?m)\b{re.escape(symbol)}$")

            linker = (
                shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
            )
            self.assertIsNotNone(linker)
            shared_path = root / "flydsl_test_launcher.so"
            subprocess.run(
                [
                    cast(str, linker),
                    "-shared",
                    "-o",
                    str(shared_path),
                    str(object_path),
                    *metadata["runtime_libraries"],
                    "-Wl,-rpath,$ORIGIN",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            library = ctypes.CDLL(str(shared_path), mode=ctypes.RTLD_GLOBAL)
            module = ctypes.c_void_p()
            error = ctypes.c_int32()
            for symbol in (
                metadata["module_init_symbol"],
                metadata["module_load_symbol"],
            ):
                loader = getattr(library, symbol)
                loader.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_int32),
                ]
                loader(ctypes.byref(module), ctypes.byref(error))
                self.assertEqual(0, error.value)
            self.assertIsNotNone(module.value)

            packed, storage = self._pack_aot_arguments(
                metadata["abi"],
                (aot_out, lhs, rhs, elements, block_dim),
            )
            entry = getattr(library, metadata["symbol"])
            entry.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            entry(packed)
            self.assertTrue(storage)
            torch.cuda.synchronize()

        torch.testing.assert_close(aot_out, jit_out)

    def test_runtime_dependencies_support_paths_with_spaces(self):
        completed = subprocess.CompletedProcess(
            args=["ldd"],
            returncode=0,
            stdout=("libfly.so => /tmp/flydsl distribution/libfly.so (0x1234)\n"),
            stderr="",
        )
        with (
            mock.patch(
                "torch._inductor.codegen.flydsl.aot_compile.shutil.which",
                return_value="/usr/bin/ldd",
            ),
            mock.patch(
                "torch._inductor.codegen.flydsl.aot_compile.subprocess.run",
                return_value=completed,
            ),
        ):
            dependencies = _runtime_library_dependencies(Path("runtime.so"))

        self.assertEqual(
            Path("/tmp/flydsl distribution/libfly.so"), dependencies["libfly.so"]
        )

    def test_runtime_dependencies_reject_missing_library(self):
        completed = subprocess.CompletedProcess(
            args=["ldd"],
            returncode=0,
            stdout="libmissing.so => not found\n",
            stderr="",
        )
        with (
            mock.patch(
                "torch._inductor.codegen.flydsl.aot_compile.shutil.which",
                return_value="/usr/bin/ldd",
            ),
            mock.patch(
                "torch._inductor.codegen.flydsl.aot_compile.subprocess.run",
                return_value=completed,
            ),
            self.assertRaisesRegex(RuntimeError, "libmissing.so"),
        ):
            _runtime_library_dependencies(Path("runtime.so"))

    def test_runtime_bundle_includes_sonames_and_flydsl_dependencies(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.TemporaryDirectory() as system_tmpdir,
        ):
            root = Path(tmpdir)
            mlir_libs = root / "flydsl" / "_mlir" / "_mlir_libs"
            wheel_libs = root / "flydsl.libs"
            output_dir = root / "output"
            mlir_libs.mkdir(parents=True)
            wheel_libs.mkdir()
            output_dir.mkdir()
            runtime = mlir_libs / "libmlir_c_runner_utils.so"
            float16 = mlir_libs / "libmlir_float16_utils.so.23"
            apfloat = wheel_libs / "libmlir_apfloat.so.23"
            system = Path(system_tmpdir) / "libc.so.6"
            for path in (runtime, float16, apfloat, system):
                path.write_bytes(path.name.encode())

            with (
                mock.patch(
                    "torch._inductor.codegen.flydsl.aot_compile._elf_soname",
                    return_value="libmlir_c_runner_utils.so.23",
                ),
                mock.patch(
                    "torch._inductor.codegen.flydsl.aot_compile._runtime_library_dependencies",
                    return_value={
                        float16.name: float16,
                        apfloat.name: apfloat,
                        system.name: system,
                    },
                ),
            ):
                bundled = _bundle_runtime_libraries([str(runtime)], output_dir)

            self.assertCountEqual(
                [
                    str(output_dir / "libmlir_c_runner_utils.so.23"),
                    str(output_dir / float16.name),
                    str(output_dir / apfloat.name),
                ],
                bundled,
            )
            self.assertFalse((output_dir / system.name).exists())

    def test_runtime_library_publication_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.so"
            second = root / "second.so"
            destination = root / "runtime.so"
            first.write_bytes(b"first runtime")
            second.write_bytes(b"second runtime")

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(_publish_runtime_library, source, destination)
                    for source in (first, second)
                ]
                errors = [future.exception() for future in futures]

            self.assertEqual(1, sum(error is None for error in errors))
            self.assertEqual(
                1, sum(isinstance(error, RuntimeError) for error in errors)
            )
            self.assertIn(
                destination.read_bytes(), (first.read_bytes(), second.read_bytes())
            )

    def test_export_rejects_invalid_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "invalid C function name"):
                self._compiled_launcher().export_to_c(
                    str(Path(tmpdir) / "kernel.o"),
                    "not-a-c-symbol",
                )

    def test_tensor_abi_describes_dynamic_layout(self):
        tensor = cast(Any, torch).empty_strided((2, 3), (5, 1))

        slots = _argument_abi(TorchTensorJitArg(tensor))

        self.assertEqual("tensor_data", slots[0]["kind"])
        self.assertEqual("tensor_layout", slots[1]["kind"])
        self.assertEqual([0, 1], slots[1]["shape_dims"])
        self.assertEqual([0], slots[1]["stride_dims"])
        self.assertEqual(64, slots[1]["stride_bits"])

    def test_integer_abi_uses_fixed_width_names(self):
        self.assertEqual("bool", _argument_abi(fx.Boolean(True))[0]["ctype"])
        self.assertEqual("int32", _argument_abi(fx.Int32(1))[0]["ctype"])
        self.assertEqual("uint64", _argument_abi(fx.Uint64(1))[0]["ctype"])

    def test_pointer_and_floating_point_abi(self):
        pointer = object.__new__(PointerJitArg)

        self.assertEqual(
            [{"kind": "pointer", "ctype": "pointer", "size": 8, "alignment": 8}],
            _argument_abi(pointer),
        )
        float16 = _argument_abi(fx.Float16(1.0))[0]
        bfloat16 = _argument_abi(fx.BFloat16(1.0))[0]
        self.assertEqual("uint16", float16["ctype"])
        self.assertEqual("float16_bits", float16["encoding"])
        self.assertEqual("uint16", bfloat16["ctype"])
        self.assertEqual("bfloat16_bits", bfloat16["encoding"])
        self.assertEqual("float", _argument_abi(fx.Float32(1.0))[0]["ctype"])
        self.assertEqual("double", _argument_abi(fx.Float64(1.0))[0]["ctype"])

    def test_abi_rejects_unsupported_and_multi_slot_arguments(self):
        with self.assertRaisesRegex(NotImplementedError, "unsupported JIT argument"):
            _argument_abi(object())

        with (
            mock.patch.object(
                protocol,
                "c_abi_spec",
                return_value=[
                    (ctypes.c_int32, mock.Mock()),
                    (ctypes.c_int32, mock.Mock()),
                ],
            ),
            self.assertRaisesRegex(NotImplementedError, "exactly one C ABI slot"),
        ):
            _argument_abi(fx.Int32(1))

    def test_unsupported_custom_jit_argument_is_rejected(self):
        class CustomJitArgument:
            def __c_abi_spec__(self):
                return []

        with self.assertRaisesRegex(NotImplementedError, "unsupported JIT argument"):
            _argument_abi(CustomJitArgument())

    def test_compile_aot_supports_positional_only_and_constexpr_arguments(self):
        @flyc.jit
        def launcher(
            inp: fx.Tensor,
            block_dim: fx.Constexpr[int],
            /,
            *,
            rows: fx.Int32,
        ):
            pass

        backend = mock.Mock()
        backend.target.arch = "gfx950"
        backend.gpu_module_targets.return_value = []
        with (
            mock.patch.object(
                type(launcher),
                "__call__",
                side_effect=AssertionError("AOT compilation dispatched the launcher"),
            ),
            mock.patch.object(
                backends,
                "get_backend",
                return_value=backend,
            ),
            mock.patch.object(
                jit_function.MlirCompiler,
                "compile",
                side_effect=lambda module, **_kwargs: module,
            ),
        ):
            compiled = compile_aot(
                launcher,
                cast(Any, torch).empty(8, device="meta"),
                256,
                rows=8,
            )

        self.assertIsInstance(compiled, CompiledAOTLauncher)
        self.assertEqual(
            ["tensor_data", "tensor_layout", "scalar", "stream"],
            [slot["kind"] for slot in compiled.abi],
        )
        self.assertNotIn("block_dim", {slot["arg_name"] for slot in compiled.abi})

    def test_compile_aot_traces_multiple_kernel_launches(self):
        @flyc.kernel
        def first_kernel():
            pass

        @flyc.kernel
        def second_kernel():
            pass

        @flyc.jit
        def launcher():
            first_kernel().launch(grid=(1, 1, 1), block=(1, 1, 1))
            second_kernel().launch(grid=(1, 1, 1), block=(1, 1, 1))

        backend = mock.Mock()
        backend.target.arch = "gfx950"
        backend.gpu_module_targets.return_value = []
        with (
            mock.patch.object(
                backends,
                "get_backend",
                return_value=backend,
            ),
            mock.patch.object(
                jit_function.MlirCompiler,
                "compile",
                side_effect=lambda module, **_kwargs: module,
            ),
        ):
            compiled = compile_aot(launcher)

        self.assertEqual(2, compiled._ir_text.count("gpu.launch_func"))

    def test_compile_aot_caller_hints_override_launcher_defaults(self):
        @flyc.jit
        def launcher():
            pass

        launcher.compile_hints = {"waves_per_eu": 1, "fast_fp_math": False}
        backend = mock.Mock()
        backend.target.arch = "gfx950"
        backend.gpu_module_targets.return_value = []
        observed_hints = {}

        def capture_hints(module, **_kwargs):
            observed_hints.update(
                kernel_function.CompilationContext.get_compile_hints()
            )
            return module

        with (
            kernel_function.CompilationContext.compile_hints(
                {"waves_per_eu": 2, "fast_fp_math": True}
            ),
            mock.patch.object(backends, "get_backend", return_value=backend),
            mock.patch.object(
                jit_function.MlirCompiler,
                "compile",
                side_effect=capture_hints,
            ),
        ):
            compile_aot(launcher)

        self.assertEqual(
            {"waves_per_eu": 2, "fast_fp_math": True},
            observed_hints,
        )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
