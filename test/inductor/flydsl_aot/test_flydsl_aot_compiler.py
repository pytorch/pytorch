# Owner(s): ["module: inductor"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import ctypes
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast
from unittest import mock

import torch
from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available
from torch.testing._internal.common_utils import TestCase


HAS_FLYDSL = runtime_available()
if HAS_FLYDSL:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import execution_engine, ir
    from flydsl.compiler import backends, jit_executor, jit_function, protocol
    from flydsl.compiler.jit_argument import PointerJitArg, TorchTensorJitArg

if HAS_FLYDSL:
    from torch._inductor.codegen.flydsl.aot_compile import (
        _argument_abi,
        _bundle_runtime_libraries,
        _publish_runtime_library,
        compile_aot,
        CompiledAOTLauncher,
    )


class _FakeExecutionEngine:
    module_text = ""
    enable_pic = False

    def __init__(self, module, *, opt_level, shared_libs, enable_pic=False):
        self.__class__.module_text = str(module)
        self.__class__.enable_pic = enable_pic

    def dump_to_object_file(self, path):
        Path(path).write_bytes(b"flydsl-object")


@unittest.skipUnless(HAS_FLYDSL, "FlyDSL is not available")
class FlyDSLAOTCompilerTest(TestCase):
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
        self.assertEqual(
            "float16_bits",
            _argument_abi(fx.Float16(1.0))[0]["encoding"],
        )
        self.assertEqual(
            "bfloat16_bits",
            _argument_abi(fx.BFloat16(1.0))[0]["encoding"],
        )
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

    def test_compile_aot_does_not_dispatch_launcher(self):
        @flyc.jit
        def launcher(
            inp: fx.Tensor,
            block_dim: fx.Constexpr[int],
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


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
