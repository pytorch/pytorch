# Owner(s): ["module: inductor"]
import copy
import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.codegen.common import TritonScratchWorkspace
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.cpp_wrapper_gpu import (
    _launch_pdl_cpp_literal,
    CppWrapperGpu,
    DeferredTritonCallWrapper,
)
from torch._inductor.codegen.cpp_wrapper_mps import CppWrapperMps
from torch._inductor.codegen.cuda.device_op_overrides import CUDADeviceOpOverrides
from torch._inductor.codegen.mps_device_op_overrides import MPSDeviceOpOverrides
from torch._inductor.codegen.mtia.device_op_overrides import MTIADeviceOpOverrides
from torch._inductor.codegen.xpu.device_op_overrides import XPUDeviceOpOverrides
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import DualIndentedBuffer, IndentedBuffer, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    find_library_location,
    HardwareClassification,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    parametrize,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, RUN_GPU
from torch.utils._triton import has_triton_tensor_descriptor_host_tma


try:
    try:
        from . import (
            test_combo_kernels,
            test_foreach,
            test_pattern_matcher,
            test_select_algorithm,
            test_torchinductor,
        )
    except ImportError:
        import test_combo_kernels  # @manual=fbcode//caffe2/test/inductor:combo_kernels-library

        import test_foreach  # @manual=fbcode//caffe2/test/inductor:foreach-library
        import test_pattern_matcher  # @manual=fbcode//caffe2/test/inductor:pattern_matcher-library
        import test_select_algorithm  # @manual=fbcode//caffe2/test/inductor:select_algorithm-library
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


def _register_fbcode_cpp_wrapper_arg_helper_op(m, device):
    def impl(xs, dtype, device_arg, layout, memory_format):
        if dtype != torch.float32:
            raise AssertionError(f"Unexpected dtype: {dtype}")
        if torch.device(device_arg).type != device:
            raise AssertionError(f"Unexpected device: {device_arg}")
        if layout != torch.strided:
            raise AssertionError(f"Unexpected layout: {layout}")
        if memory_format != torch.contiguous_format:
            raise AssertionError(f"Unexpected memory_format: {memory_format}")
        return (
            xs[0]
            .to(device=device_arg, dtype=dtype)
            .contiguous(memory_format=memory_format)
        )

    def meta(xs, dtype, _device_arg, layout, memory_format):
        return torch.empty_like(
            xs[0],
            dtype=dtype,
            layout=layout,
            memory_format=memory_format,
        )

    m.define(
        "arg_helpers("
        "Tensor[] xs, ScalarType dtype, Device device, "
        "Layout layout, MemoryFormat memory_format"
        ") -> Tensor"
    )
    m.impl("arg_helpers", impl, device.upper())
    m.impl("arg_helpers", meta, "Meta")


class TestGpuWrapper(InductorTestCase):
    """cpp_wrapper tests that run on whatever accelerator is instantiated."""

    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("empty_middle", [False, True])
    def test_nested_tensor_list_fallback(self, device, empty_middle):
        with torch.library._scoped_library("gpu_nested_fallback", "FRAGMENT") as lib:
            torch.library.define(
                "gpu_nested_fallback::lists",
                "(Tensor x) -> (Tensor[], Tensor[], Tensor[])",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl(
                "gpu_nested_fallback::lists", "CompositeExplicitAutograd", lib=lib
            )
            @torch.library.register_fake("gpu_nested_fallback::lists", lib=lib)
            def lists(x):
                return [x + 1, x * 2], [] if empty_middle else [x - 3], [x.square()]

            class Model(torch.nn.Module):
                def forward(self, x):
                    first, middle, last = torch.ops.gpu_nested_fallback.lists(x.sin())
                    result = first[0] * first[1] + last[0]
                    if not empty_middle:
                        result = result + middle[0]
                    return result

            model = Model()
            compiled = torch.compile(
                model,
                fullgraph=True,
                dynamic=True,
                options={
                    "cpp_wrapper": True,
                    "triton.autotune_at_compile_time": False,
                },
            )
            x = torch.randn(4, 16, device=device)
            actual, code = run_and_get_code(compiled, x)
            self.assertEqual(actual, model(x))
            code = "\n".join(code)
            self.assertIn('findSchemaOrThrow("gpu_nested_fallback::lists", "")', code)
            self.assertIn(".callBoxed(", code)
            self.assertIn(".toTensorList()", code)
            self.assertNotIn("custom_op_wrapper", code)
            self.assertNotIn("PyObject_CallObject", code)
            for batch in (7, 12):
                x = torch.randn(batch, 16, device=device)
                self.assertEqual(compiled(x), model(x))

    def test_cpp_wrapper_compile_timing_recorded(self, device):
        from torch._dynamo.utils import compilation_time_metrics
        from torch._inductor.utils import fresh_cache

        def fn(x, y):
            return (x @ y).relu()

        x = torch.randn(64, 64, device=device)
        y = torch.randn(64, 64, device=device)
        compilation_time_metrics.pop("cpp_wrapper_compile", None)
        with fresh_cache():
            compiled = torch.compile(options={"cpp_wrapper": True})(fn)
            result = compiled(x, y)
        self.assertEqual(result, fn(x, y))
        self.assertIn("cpp_wrapper_compile", compilation_time_metrics)
        durations = compilation_time_metrics["cpp_wrapper_compile"]
        self.assertTrue(any(t > 0 for t in durations))

    def test_aoti_debug_printer_works_on_constants(self, device):
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def test_fn():
            inp = torch.randn(batch_size, seq_length, hidden_size, device=device)
            weight = torch.randn(hidden_size, hidden_size, device=device)
            matmul_output = inp @ weight
            torch.nn.LayerNorm(hidden_size, device=device)(matmul_output)
            return True

        comp = torch.compile(
            options={
                "cpp_wrapper": True,
                "cpp_wrapper_build_separate": True,
                "aot_inductor.debug_intermediate_value_printer": "2",
            }
        )(test_fn)
        comp()

    def test_non_tensor_args_wrapped_on_cpu(self, device):
        def test_fn(x, s):
            return (x + s).sum()

        compiled = torch.compile(options={"cpp_wrapper": True})(test_fn)
        x = torch.randn(4, device=device)
        with torch.utils._device.DeviceContext(device):
            _, code = test_torchinductor.run_and_get_cpp_code(compiled, x, 3)
        self.assertIn("torch.tensor(arg, device='cpu')", code)

    @unittest.skipIf(
        not config.is_fbcode(), "fbcode-only cpp_wrapper symbol visibility test"
    )
    @config.patch(implicit_fallbacks=True)
    def test_fbcode_custom_op_fallback_python_arg_helpers(self, device):
        if not config.is_fbcode():
            self.skipTest("fbcode-only cpp_wrapper symbol visibility test")

        with torch.library._scoped_library(
            "fbcode_cpp_wrapper_arg_helpers", "FRAGMENT"
        ) as m:
            _register_fbcode_cpp_wrapper_arg_helper_op(m, device)

            def fn(x):
                y = x.sin()
                return torch.ops.fbcode_cpp_wrapper_arg_helpers.arg_helpers(
                    [y],
                    torch.float32,
                    torch.device(device),
                    torch.strided,
                    torch.contiguous_format,
                ).cos()

            x = torch.randn(4, 4, device=device)
            expected = fn(x)
            actual = torch.compile(fullgraph=True, options={"cpp_wrapper": True})(fn)(x)
            self.assertEqual(actual, expected)

    @config.patch(implicit_fallbacks=True)
    def test_custom_op_fallback_symint_dispatch(self, device):
        # A custom op with a SymInt argument (e.g.
        # fbgemm::permute_2D_sparse_data_input1D) must not be routed to the
        # StableIValue path.  SymInt is reported as IntType by JitType.type, so
        # such ops used to pass _compatible_with_stableivalue and then fail at
        # runtime with 'aoti_torch_call_dispatcher(...) API call failed'.  They
        # must use the boxed dispatch path, which handles c10::SymInt correctly.
        with torch.library._scoped_library("mylib_symint", "FRAGMENT") as m:
            m.define("add_symint(Tensor x, SymInt n) -> Tensor")
            m.impl("add_symint", lambda x, n: x + n, device.upper())
            m.impl("add_symint", lambda x, n: torch.empty_like(x), "Meta")

            op = torch.ops.mylib_symint.add_symint.default
            self.assertFalse(CppWrapperCpu._compatible_with_stableivalue(op))
            self.assertTrue(CppWrapperCpu._compatible_with_cpp_boxed_dispatch(op))

            def fn(x):
                # x.shape[0] is symbolic once dim 0 is dynamic, so n is a SymInt.
                return torch.ops.mylib_symint.add_symint(x, x.shape[0])

            x = torch.randn(8, 4, device=device)
            torch._dynamo.mark_dynamic(x, 0)
            expected = fn(x)
            compiled = torch.compile(fullgraph=True, options={"cpp_wrapper": True})(fn)
            actual, code = test_torchinductor.run_and_get_cpp_code(compiled, x)
            self.assertEqual(actual, expected)
            self.assertNotIn(
                'aoti_torch_call_dispatcher("mylib_symint::add_symint"', code
            )

    @config.patch(implicit_fallbacks=True)
    def test_custom_op_fallback_boxes_none_as_undefined_tensor(self, device):
        if IS_FBCODE or IS_SANDCASTLE:
            torch.ops.load_library("//caffe2/test/inductor:custom_ops")
        elif IS_MACOS:
            self.skipTest("non-portable load_library call used in test")
        else:
            lib_path = find_library_location("libaoti_custom_ops.so")
            if IS_WINDOWS:
                lib_path = find_library_location("aoti_custom_ops.dll")
            if not os.path.exists(lib_path):
                self.skipTest("libaoti_custom_ops not built")
            torch.ops.load_library(str(lib_path))

        def fn(x):
            return torch.ops.aoti_custom_ops.maybe_weighted(x, None, x.shape[0])

        x = torch.randn(8, 4, device=device)
        torch._dynamo.mark_dynamic(x, 0)
        expected = fn(x)
        compiled = torch.compile(fullgraph=True, options={"cpp_wrapper": True})(fn)
        actual, code = test_torchinductor.run_and_get_cpp_code(compiled, x)
        self.assertEqual(actual, expected)
        # The composite must decompose to the inner op, which is boxed-dispatched
        # with its undefined weight materialized as an undefined tensor, not None.
        self.assertIn("aoti_custom_ops::forward_maybe_weighted", code)
        self.assertIn("c10::IValue(at::Tensor())", code)

    @parametrize("per_subkernel_blocks", [False, True])
    def test_lazy_compile_combo_kernel_default_config(
        self, device, per_subkernel_blocks
    ):
        """Lazy compile should use default_config from combo_grid_meta for XBLOCK."""
        from unittest.mock import patch

        from torch._inductor.codegen.triton_combo_kernel import (
            DEFAULT_COMBO_BLOCK_SIZE_1D,
        )
        from torch._inductor.runtime import triton_lazy_compile as tlc

        captured = {}
        original = tlc.run_triton_kernel_with_autotune

        def capture(pending_kernels, kernel_name, stream, args):
            result = original(pending_kernels, kernel_name, stream, args)
            if "triton_for_fused" in kernel_name:
                captured[kernel_name] = result.xblocks
            return result

        with patch.object(tlc, "run_triton_kernel_with_autotune", side_effect=capture):
            params = [torch.randn(1024, device=device) for _ in range(4)]
            grads = [torch.randn_like(p) for p in params]

            @torch.compile(
                options={
                    "cpp_wrapper": True,
                    "triton.autotune_at_compile_time": False,
                    "combo_kernels": True,
                    "combo_kernel_per_subkernel_blocks": per_subkernel_blocks,
                }
            )
            def fn(params, grads):
                torch._foreach_add_(params, grads, alpha=-0.1)

            fn(params, grads)

        self.assertTrue(len(captured) > 0, "No combo kernels were lazy-compiled")
        for name, xblocks in captured.items():
            # When per_subkernel_blocks=False, default_config has a single XBLOCK
            # that must be picked up correctly (not the hardcoded fallback of 128).
            # When per_subkernel_blocks=True, default_config uses per-subkernel
            # XBLOCK_N keys, so xblocks has one entry per subkernel; just verify
            # compilation succeeded.
            if not per_subkernel_blocks:
                self.assertEqual(
                    xblocks,
                    [DEFAULT_COMBO_BLOCK_SIZE_1D],
                    lambda msg: f"{msg}\n{name} got xblocks={xblocks}, "
                    f"expected [{DEFAULT_COMBO_BLOCK_SIZE_1D}]",
                )

    def test_cudagraph_no_partition(self, device):
        def test_fn(x, s):
            return (x + s).sum()

        x = torch.randn(4, device=device)
        s = 3
        expected = test_fn(x, s)

        comp = torch.compile(
            options={
                "cpp_wrapper": True,
                "triton.cudagraphs": True,
                "graph_partition": False,
            }
        )(test_fn)
        for i in range(3):
            res = comp(x, s)
            self.assertEqual(res, expected)

    def test_many_args_fold_expression_nesting(self, device):
        if torch.device(device).type == "xpu":
            self.skipTest("ocloc backend compiler crashes with too many kernel args")

        num_params = 130
        params = [torch.randn(64, device=device) for _ in range(num_params)]
        grads = [torch.randn_like(p) for p in params]
        expected = [p.clone() + (-0.1) * g for p, g in zip(params, grads)]

        @torch.compile(
            options={
                "cpp_wrapper": True,
                "combo_kernels": True,
                "combo_kernel_max_num_args": 1000,
            }
        )
        def fn(params, grads):
            torch._foreach_add_(params, grads, alpha=-0.1)

        fn(params, grads)

        for p, e in zip(params, expected):
            self.assertEqual(p, e)

    def test_cpp_wrapper_backward_lazy_compile(self, device):
        """Test that options={"cpp_wrapper": True} works with backward pass.

        Backward graphs may be compiled lazily (after compile_fx returns).
        The cpp_wrapper triton config (store_cubin, autotune_at_compile_time)
        must still be applied. See https://github.com/pytorch/pytorch/issues/178845
        """

        def fn(x, output_grad):
            layer_norm = torch.nn.LayerNorm(normalized_shape=4).to(device)
            output = layer_norm(x)
            output.backward(output_grad)
            return output

        x = torch.randn(2, 3, 4, device=device)
        output_grad = torch.randn(2, 3, 4, device=device)

        opt_fn = torch.compile(options={"cpp_wrapper": True})(fn)
        result = opt_fn(x, output_grad)
        self.assertEqual(result.shape, x.shape)

    def test_cuda_cpp_wrapper_skips_vec_isa_for_device_only_code(self, device):
        def fn(x):
            return (x.sin() + 1).cos()

        x = torch.randn(128, device=device)
        expected = fn(x)
        compiled = torch.compile(options={"cpp_wrapper": True})(fn)
        actual, code = test_torchinductor.run_and_get_cpp_code(compiled, x)

        self.assertEqual(actual, expected)
        self.assertIn("needs_vec_isa=False", code)

    def test_cuda_cpp_wrapper_keeps_vec_isa_for_host_vectorized_code(self, device):
        def fn(x):
            x = x + 1
            x = x + 2
            x = x.to(device=device)
            x = x + 3
            x = x + 4
            x = x.cpu()
            x = x + 5
            x = x + 6
            return x

        x = torch.randn(2, 2, 10)
        expected = fn(x)
        compiled = torch.compile(options={"cpp_wrapper": True})(fn)
        actual, code = test_torchinductor.run_and_get_cpp_code(compiled, x)

        self.assertEqual(actual, expected)
        self.assertIn("at::vec::", code)
        self.assertIn("needs_vec_isa=True", code)

    def test_cuda_cpp_wrapper_build_separate_splits_vec_isa_requirements(self, device):
        def fn(x):
            x = x + 1
            x = x + 2
            x = x.to(device=device)
            x = x + 3
            x = x + 4
            x = x.cpu()
            x = x + 5
            x = x + 6
            return x

        x = torch.randn(2, 2, 10)
        expected = fn(x)
        with config.patch({"cpp_wrapper_build_separate": True}):
            compiled = torch.compile(options={"cpp_wrapper": True})(fn)
            actual, code = test_torchinductor.run_and_get_cpp_code(compiled, x)

        self.assertEqual(actual, expected)
        self.assertIn("kernel_src = (", code)
        self.assertIn("needs_vec_isa=False", code)
        self.assertIn("kernel_needs_vec_isa=True", code)

    def test_map_fullgraph_cpp_wrapper(self, device):
        def body_fn(x):
            return torch.nn.functional.gelu(x)

        def fn(xs):
            return torch._higher_order_ops.map(body_fn, xs).sum()

        xs = torch.randn(8, 64, device=device)
        expected = fn(xs)
        opt_fn = torch.compile(fn, fullgraph=True, options={"cpp_wrapper": True})
        self.assertEqual(opt_fn(xs), expected)

    def test_any_fallback_cpp_wrapper(self, device):
        with torch.library._scoped_library("mylib_fallback", "FRAGMENT") as m:
            m.define("any_fallback(Tensor x, Any y) -> Tensor")

            def any_fallback(x: torch.Tensor, y: Any) -> torch.Tensor:
                torch._check(y is not None)
                return x.clone()

            m.impl("any_fallback", any_fallback, device.upper())
            m.impl("any_fallback", any_fallback, "Meta")

            @torch.compile(fullgraph=True, options={"cpp_wrapper": True})
            def test_fn(x: torch.Tensor, y: Any) -> torch.Tensor:
                return torch.ops.mylib_fallback.any_fallback(x, y)

            test_fn(torch.randn(4, device=device), torch.randn(4, device=device))
            test_fn(torch.randn(4, device=device), "string")


class TestGpuWrapperGeneric(InductorTestCase):
    """Codegen-only checks that do not need an accelerator to run."""

    hw_classification = HardwareClassification.GENERIC

    def test_prologue_stream_declaration_per_backend(self):
        def prologue(wrapper_cls, device, device_codegen, aot_mode=False):
            wrapper = wrapper_cls.__new__(wrapper_cls)
            wrapper.device = device
            wrapper.device_codegen = device_codegen
            wrapper.prefix = IndentedBuffer()
            wrapper._lazy_kernel_names = []
            graph = SimpleNamespace(
                aot_mode=aot_mode, is_dual_wrapper_mode=False, device_idxs=(0,)
            )
            with V.set_graph_handler(graph):
                wrapper._codegen_entry_impl_prologue()
            return wrapper.prefix.getvalue()

        self.assertIn(
            "stream0;", prologue(CppWrapperGpu, "cuda", CUDADeviceOpOverrides())
        )
        # Neither backend has a stream type to query: Metal kernels are
        # dispatched without a raw stream, and pure AOTI takes the stream as a
        # parameter, which is how MTIA runs through CppWrapperGpu.
        self.assertNotIn(
            "stream0", prologue(CppWrapperMps, "mps", MPSDeviceOpOverrides())
        )
        self.assertNotIn(
            "stream0",
            prologue(CppWrapperGpu, "mtia", MTIADeviceOpOverrides(), aot_mode=True),
        )
        self.assertNotIn(
            "stream0",
            prologue(CppWrapperGpu, "xpu", XPUDeviceOpOverrides(), aot_mode=True),
        )

    @parametrize(
        "triton_meta,expected",
        [
            (None, "false"),
            ({}, "false"),
            ({"launch_pdl": False}, "false"),
            ({"launch_pdl": True}, "true"),
            ({"launch_pdl": 0}, "false"),
            ({"launch_pdl": 1}, "true"),
            ({"launch_pdl": None}, "false"),
            (
                {"launch_pdl": False, "backend_options": {"launch_pdl": True}},
                "true",
            ),
            (
                {"launch_pdl": True, "backend_options": {"launch_pdl": False}},
                "false",
            ),
        ],
    )
    def test_launch_pdl_cpp_literal(self, triton_meta, expected):
        self.assertEqual(_launch_pdl_cpp_literal(triton_meta), expected)

    @parametrize("enable_kernel_profile", [False, True])
    @parametrize("supports_pdl,launch_pdl_arg", [(True, ", true"), (False, "")])
    def test_triton_wrapper_forwards_launch_pdl(
        self, enable_kernel_profile, supports_pdl, launch_pdl_arg
    ):
        class FakeDeviceCodegen:
            @staticmethod
            def cpp_kernel_launch_supports_pdl():
                return supports_pdl

        class FakeWrapper:
            device_codegen = FakeDeviceCodegen()

            @staticmethod
            def generate_args_decl(
                prefix,
                call_args,
                arg_types,
                arg_signatures,
                is_triton_kernel=True,
                scratch_spaces=None,
            ):
                return ""

        prefix = IndentedBuffer()
        params = {
            "triton_meta": {
                "signature": {"x": "*fp32"},
                "constants": {},
                "launch_pdl": True,
            },
            "def_args": ["x"],
            "call_args": ["x"],
            "config": {},
            "num_warps": 4,
            "shared_mem": 0,
        }
        with config.patch({"cpp.enable_kernel_profile": enable_kernel_profile}):
            DeferredTritonCallWrapper(
                wrapper_name="wrapper",
                kernel_name="kernel",
                kernel_name_to_body={},
                arg_types=[torch.float32],
            ).generate_launch_kernel(prefix, FakeWrapper(), "kernel_var", params)

        self.assertIn(
            "launchKernel(kernel_var, grid_0, grid_1, grid_2, 4, 0, "
            f"kernel_args_, stream_{launch_pdl_arg});",
            prefix.getvalue(),
        )

    @parametrize("supports_pdl,launch_pdl_arg", [(True, ", true"), (False, "")])
    def test_lazy_triton_wrapper_forwards_launch_pdl(
        self, supports_pdl, launch_pdl_arg
    ):
        class FakeDeviceCodegen:
            @staticmethod
            def cpp_device_ptr():
                return "CUdeviceptr"

            @staticmethod
            def cpp_kernel_launch_supports_pdl():
                return supports_pdl

        class FakeWrapper:
            device = "cuda"  # production wrappers always carry .device (see _generate_lazy_scratch)
            device_codegen = FakeDeviceCodegen()

            @staticmethod
            def generate_args_decl(
                prefix,
                call_args,
                arg_types,
                arg_signatures,
                is_triton_kernel=True,
                scratch_spaces=None,
            ):
                return ""

            @staticmethod
            def codegen_dtype(dtype):
                return "at::ScalarType::Byte"

            @staticmethod
            def codegen_device(device):
                return "c10::DeviceType::CUDA, 0"

        prefix = DualIndentedBuffer()
        deferred = DeferredTritonCallWrapper(
            wrapper_name="wrapper",
            kernel_name="kernel",
            kernel_name_to_body={},
            arg_types=[torch.float32],
            triton_meta={
                "signature": {"x": "*fp32"},
                "constants": {},
                "launch_pdl": True,
            },
        )
        deferred._generate_lazy_launch(prefix, FakeWrapper(), ["x"], ["x"])

        expected_args = (
            "grid_0, grid_1, grid_2, kernel_result.num_warps, "
            f"kernel_result.shared_mem, kernel_args_, stream_{launch_pdl_arg}"
        )
        self.assertIn(f"launchKernel(kernel, {expected_args});", prefix.getvalue())
        self.assertIn(
            f"launchKernel(kernels_.kernel, {expected_args});",
            prefix.aot.getvalue(),
        )


class TestGpuWrapperCudaCodegen(InductorTestCase):
    """Codegen checks that assert CUDA/ROCm specific generated code."""

    hw_classification = HardwareClassification.CUDA

    def test_cpp_wrapper_gpu_debug_sync_codegen(self):
        wrapper = CppWrapperGpu.__new__(CppWrapperGpu)
        wrapper.device = "cuda"

        debug_sync = IndentedBuffer()
        wrapper.generate_debug_sync(debug_sync)
        code = debug_sync.getvalue()
        if torch.version.hip is not None:
            self.assertIn("AOTI_RUNTIME_CUDA_CHECK", code)
            self.assertIn("DeviceSynchronize", code)
        else:
            self.assertIn("CUDA_DRIVER_CHECK", code)
            self.assertIn("CtxSynchronize", code)
        self.assertNotIn("torch.cuda.synchronize()", code)

        wrapper.prefix = IndentedBuffer()
        wrapper._lazy_kernel_names = []
        # The prologue hoists a stream declaration per device, so it reads the
        # devices the graph lowered for.
        graph = SimpleNamespace(
            aot_mode=False, is_dual_wrapper_mode=False, device_idxs=()
        )
        with (
            config.patch({"triton.debug_sync_graph": True}),
            V.set_graph_handler(graph),
        ):
            wrapper._codegen_entry_impl_prologue()
        code = wrapper.prefix.getvalue()
        if torch.version.hip is not None:
            self.assertIn("AOTI_RUNTIME_CUDA_CHECK", code)
            self.assertIn("DeviceSynchronize", code)
        else:
            self.assertIn("CUDA_DRIVER_CHECK", code)
            self.assertIn("CtxSynchronize", code)
        self.assertNotIn("torch.cuda.synchronize()", code)

        wrapper.device = "xpu"
        with self.assertRaisesRegex(
            NotImplementedError, "triton debug sync is not supported"
        ):
            wrapper.generate_debug_sync(IndentedBuffer())

    def test_tunableop_dynamic_dims_guard_codegen(self):
        wrapper = CppWrapperGpu.__new__(CppWrapperGpu)
        wrapper.device = "cuda"
        wrapper.lines = []

        with config.patch({"cuda.autotune_tunableop_dynamic_dims_wildcard": True}):
            self.assertTrue(
                wrapper._begin_tunableop_dynamic_dims_guard(
                    (False, True, False, False), "cuda"
                )
            )
        self.assertEqual(
            wrapper.lines,
            [
                "{",
                "AOTICudaTunableOpDynamicDimsGuard tunable_dynamic_dims_guard(2);",
            ],
        )
        with config.patch({"cuda.autotune_tunableop_dynamic_dims_wildcard": True}):
            self.assertFalse(wrapper._begin_tunableop_dynamic_dims_guard(None, "cuda"))
            self.assertFalse(
                wrapper._begin_tunableop_dynamic_dims_guard(
                    (False, False, False, False), "cuda"
                )
            )
            self.assertFalse(
                wrapper._begin_tunableop_dynamic_dims_guard(
                    (False, True, False, False), "cpu"
                )
            )
        with config.patch({"cuda.autotune_tunableop_dynamic_dims_wildcard": False}):
            self.assertFalse(
                wrapper._begin_tunableop_dynamic_dims_guard(
                    (False, True, False, False), "cuda"
                )
            )

    def test_tunableop_dynamic_dims_guard_preserves_alloc_output_scope(self):
        wrapper = CppWrapperGpu.__new__(CppWrapperGpu)
        wrapper.device = "cuda"
        wrapper.lines = []
        extern_kernel = SimpleNamespace(
            name="buf0",
            op_overload=None,
            outputs=[],
            python_kernel_name=None,
            tunable_dyn_dims_mask=(True, False, False, False),
            get_device=lambda: torch.device("cuda"),
            get_kernel_name=lambda: "at::test",
        )

        def emit_call(wrapper, *_args, **_kwargs):
            wrapper.writeline("CALL;")

        with (
            config.patch({"cuda.autotune_tunableop_dynamic_dims_wildcard": True}),
            patch.object(
                CppWrapperCpu,
                "generate_c_shim_extern_kernel_call",
                emit_call,
            ),
        ):
            wrapper._generate_extern_kernel_alloc_helper(extern_kernel, ["arg0"])

        self.assertEqual(
            wrapper.lines,
            [
                "AtenTensorHandle buf0_handle;",
                "{",
                "AOTICudaTunableOpDynamicDimsGuard tunable_dynamic_dims_guard(1);",
                "CALL;",
                "}",
                "RAIIAtenTensorHandle buf0(buf0_handle);",
            ],
        )

    def test_tma_descriptor_separates_global_and_kernel_shapes(self):
        if torch.version.hip is not None:
            self.skipTest("requires CUDA TMA helpers")

        helpers = CUDADeviceOpOverrides().tma_descriptor_helpers()
        self.assertNotIn("int32_t* shape", helpers)
        self.assertIn("uint64_t* shape", helpers)
        self.assertIn("int32_t kernel_shape[5];", helpers)
        self.assertIn("uint64_t global_shape[5];", helpers)
        self.assertIn("uint64_t dim = shape[i];", helpers)

        wrapper = CppWrapperGpu.__new__(CppWrapperGpu)
        wrapper.arg_var_id = itertools.count()
        launch_args = wrapper.generate_args_decl(
            IndentedBuffer(),
            ["descriptor"],
            [torch.float32],
            ["tensordesc<fp32[16, 32]>"],
        )
        self.assertEqual(
            launch_args,
            "&var_0.m, &var_0.kernel_shape[0], &var_0.kernel_shape[1], "
            "&var_0.strides[0], &var_0.strides[1]",
        )

    def test_cpp_scratch_scales_with_grid_size_for_tma(self):
        scratch_def, scratch_var = CUDADeviceOpOverrides().cpp_scratch(
            0,
            TritonScratchWorkspace(
                size=256,
                generate_dtype_str=lambda: "at::ScalarType::Byte",
                generate_device_type_str=lambda: "cached_torch_device_type_cuda",
            ),
            prefix="global_scratch",
        )
        self.assertEqual(scratch_var, "global_scratch_scratch_0")
        self.assertIn(
            "static_cast<int64_t>(256) * grid_0 * grid_1 * grid_2", scratch_def[0]
        )
        self.assertIn("cached_torch_device_type_cuda", scratch_def[3])

    def test_triton_wrapper_scales_scratch_with_num_ctas(self):
        class FakeDeviceCodegen:
            @staticmethod
            def cpp_kernel_launch_supports_pdl():
                return True

        class FakeWrapper:
            device_codegen = FakeDeviceCodegen()

            def __init__(self):
                self.scratch_spaces = None

            def generate_args_decl(
                self,
                prefix,
                call_args,
                arg_types,
                arg_signatures,
                is_triton_kernel=True,
                scratch_spaces=None,
            ):
                self.scratch_spaces = scratch_spaces

                return ""

        wrapper = FakeWrapper()
        prefix = IndentedBuffer()
        params = {
            "triton_meta": {"signature": {"x": "*fp32"}, "constants": {}},
            "def_args": ["x"],
            "call_args": ["x"],
            "config": {"num_ctas": 8},
            "num_warps": 4,
            "shared_mem": 0,
            "global_scratch": 256,
        }

        DeferredTritonCallWrapper(
            wrapper_name="wrapper",
            kernel_name="kernel",
            kernel_name_to_body={},
            arg_types=[torch.float32],
        ).generate_launch_kernel(prefix, wrapper, "kernel_var", params)

        self.assertEqual(wrapper.scratch_spaces, {"global_scratch": 256 * 8})


class TestGpuWrapperCuda(InductorTestCase):
    """cpp_wrapper tests that execute on a CUDA device."""

    hw_classification = HardwareClassification.CUDA

    def test_debug_sync_graph(self, device):
        def test_fn(x):
            return x * 2

        compiled = torch.compile(
            options={"cpp_wrapper": True, "triton.debug_sync_graph": True}
        )(test_fn)
        x = torch.randn(8, device=device)
        with torch.utils._device.DeviceContext(device):
            result = compiled(x)
        self.assertEqual(result, x * 2)

    def test_debug_sync_kernel(self, device):
        def test_fn(x):
            return x * 2

        compiled = torch.compile(
            options={"cpp_wrapper": True, "triton.debug_sync_kernel": True}
        )(test_fn)
        x = torch.randn(8, device=device)
        with torch.utils._device.DeviceContext(device):
            result = compiled(x)
        self.assertEqual(result, x * 2)

    def test_tunableop_dynamic_dims_guard_in_compiled_wrapper(self, device):
        def fn(bias, x, weight):
            return torch.addmm(bias, x, weight)

        bias = torch.randn(32, device=device)
        x = torch.randn(8, 16, device=device)
        weight = torch.randn(16, 32, device=device)
        torch._dynamo.mark_dynamic(x, 0)

        previous_enabled = torch.cuda.tunable.is_enabled()
        previous_tuning = torch.cuda.tunable.tuning_is_enabled()
        previous_filename = torch.cuda.tunable.get_filename()
        try:
            torch.cuda.tunable.enable(False)
            torch.cuda.tunable.tuning_enable(False)
            with config.patch(
                {
                    "cuda.autotune_tunableop_dynamic_dims_wildcard": True,
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "ATEN",
                }
            ):
                compiled = torch.compile(
                    fn, fullgraph=True, options={"cpp_wrapper": True}
                )
                actual, code = test_torchinductor.run_and_get_cpp_code(
                    compiled, bias, x, weight
                )

            self.assertEqual(actual, fn(bias, x, weight))
            self.assertIn("AOTICudaTunableOpDynamicDimsGuard", code)
            self.assertIn("tunable_dynamic_dims_guard(1)", code)
            self.assertNotIn("ATen/cuda/tunable/Tunable.h", code)
            self.assertNotIn("at::cuda::tunable", code)

            with tempfile.TemporaryDirectory() as tmpdir:
                torch.cuda.tunable.set_filename(
                    os.path.join(tmpdir, "results.csv"), False
                )
                torch.cuda.tunable._clear_all()
                torch.cuda.tunable.enable(True)
                torch.cuda.tunable.tuning_enable(True)
                self.assertEqual(compiled(bias, x, weight), fn(bias, x, weight))
                self.assertTrue(
                    any("*" in entry[1] for entry in torch.cuda.tunable.get_results())
                )
        finally:
            torch.cuda.tunable._clear_all()
            torch.cuda.tunable.set_filename(previous_filename, False)
            torch.cuda.tunable.enable(previous_enabled)
            torch.cuda.tunable.tuning_enable(previous_tuning)


_LAZY_COMPILE_COLLISION_SCRIPT = """\
import os
import torch

from torch._inductor import config

config.cpp_wrapper = True
config.triton.autotune_at_compile_time = False

device = os.environ["TEST_DEVICE"]

def fn(x, y, z, w):
    a = x.sin()
    torch._dynamo.graph_break()
    b = (a * y).cos()
    torch._dynamo.graph_break()
    c = (b * z).sin()
    torch._dynamo.graph_break()
    d = (c * w).cos()
    return d.sum()

args = [torch.randn(32, device=device, requires_grad=True) for _ in range(4)]
ref_args = [a.detach().clone().requires_grad_(True) for a in args]
ref = fn(*ref_args)
ref.backward()

compiled_fn = torch.compile(fn)
res = compiled_fn(*args)
res.backward()

assert torch.allclose(res.detach(), ref.detach()), f"Forward mismatch: {res} vs {ref}"
for i, (a, r) in enumerate(zip(args, ref_args)):
    assert torch.allclose(a.grad, r.grad), f"Grad mismatch for arg {i}"
"""


class TestLazyCompileKernelCollision(InductorTestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_lazy_compile_kernel_name_collision_across_modules(self, device):
        """The collision manifests when a fresh process loads .so modules from
        warm on-disk caches: AOTAutograd cache hits cause both forward and
        backward .so to be loaded (static initializers register kernels in
        _pending_kernels) before either executes.  If two modules share a
        kernel name, the global dict collision corrupts the mapping.

        This requires two process invocations because dlopen within a single
        process reuses loaded libraries without re-running static initializers.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Keep the cache off the subprocess's sys.path: `python -c` puts its
            # cwd first, and a cache subdir such as "z3/" would shadow that module.
            cache_dir = os.path.join(tmp_dir, "inductor_cache")
            env = {
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": cache_dir,
                "TEST_DEVICE": device,
                "INDUCTOR_TEST_DISABLE_FRESH_CACHE": "1",
            }
            # First run: cold compile, populates on-disk caches.
            r1 = subprocess.run(
                [sys.executable, "-c", _LAZY_COMPILE_COLLISION_SCRIPT],
                capture_output=True,
                cwd=tmp_dir,
                text=True,
                env=env,
            )
            self.assertEqual(
                r1.returncode,
                0,
                lambda msg: f"{msg}\nCold run failed:\n{r1.stderr[-2000:]}",
            )
            # Second run: warm caches trigger the collision without the fix.
            r2 = subprocess.run(
                [sys.executable, "-c", _LAZY_COMPILE_COLLISION_SCRIPT],
                capture_output=True,
                cwd=tmp_dir,
                text=True,
                env=env,
            )
            self.assertEqual(
                r2.returncode,
                0,
                lambda msg: f"{msg}\nWarm run failed:\n{r2.stderr[-2000:]}",
            )


_STATIC_INIT_DEADLOCK_SCRIPT = """\
import os
import torch

from torch._inductor import config

config.compile_threads = 1
config.cpp_wrapper = True
config.triton.autotune_at_compile_time = False

device = os.environ["TEST_DEVICE"]


def fn(x):
    return (x * 2 + 1).sum()


compiled = torch.compile(fn)
x = torch.randn(128, device=device)
compiled(x)
"""


class TestCppWrapperStaticInitDeadlock(InductorTestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @skipIfXpu(msg="https://github.com/pytorch/pytorch/issues/184496")
    def test_static_init_dlopen_does_not_deadlock(self, device):
        """The cpp_wrapper-generated .so must not trigger Triton kernel
        compilation from a static initializer (dlopen-time): doing so
        runs Triton -> mlir::verify -> llvm::StdThreadPool, whose workers
        contend for the dynamic linker's global init lock and deadlock
        the main thread inside dlopen.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Keep the cache off sys.path; see the collision test above.
            cache_dir = os.path.join(tmp_dir, "inductor_cache")
            env = {
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": cache_dir,
                "TEST_DEVICE": device,
            }
            try:
                r = subprocess.run(
                    [sys.executable, "-c", _STATIC_INIT_DEADLOCK_SCRIPT],
                    capture_output=True,
                    cwd=tmp_dir,
                    text=True,
                    env=env,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                self.fail(
                    "cpp_wrapper subprocess timed out after 60s -- likely "
                    "deadlocked in dlopen / static-init Triton kernel compile."
                )
        self.assertEqual(
            r.returncode,
            0,
            lambda msg: f"{msg}\nSubprocess failed:\nstderr:\n{r.stderr[-2000:]}\nstdout:\n{r.stdout[-2000:]}",
        )


_LAZY_TMA_GLOBAL_SCRATCH_SCRIPT = """\
import torch
import triton
import triton.language as tl

from torch._library import capture_triton


M = 1048576
N = 16
BLOCK_M = 16
BLOCK_N = 16


def _alloc_scratch(size, alignment, stream):
    return torch.empty(size, device="{device}", dtype=torch.uint8)


triton.set_allocator(_alloc_scratch)


@triton.jit
def _tma_copy_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Each CTA builds a different descriptor so one shared scratch slot is corrupt.
    in_desc = tl.make_tensor_descriptor(
        in_ptr + pid * BLOCK_M * N,
        shape=[BLOCK_M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    tile = in_desc.load([0, 0])

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], tile)


@torch.library.triton_op("test_inductor::lazy_tma_scratch_copy", mutates_args={})
def _tma_copy(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    capture_triton(_tma_copy_kernel)[grid](
        x,
        y,
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=4,
    )
    return y


def run():
    x = (torch.arange(M * N, device="{device}", dtype=torch.float32) % 100).to(
        torch.float16
    )
    x = x.reshape(M, N)

    def fn(x):
        return _tma_copy(x)

    eager = fn(x)
    torch.{device}.synchronize()
    torch.testing.assert_close(eager, x)

    compiled = torch.compile(
        fn,
        fullgraph=True,
        options={
            "cpp_wrapper": True,
            "triton.autotune_at_compile_time": False,
        },
    )
    warmup = compiled(x)
    torch.{device}.synchronize()
    torch.testing.assert_close(warmup, x)
    # The warmup lazy-compiles through Python; the second call uses the cached C++ launch.
    out = compiled(x)
    torch.{device}.synchronize()
    torch.testing.assert_close(out, x)


run()
"""


class TestLazyTmaGlobalScratch(InductorTestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_lazy_tma_global_scratch_scales_with_launch_grid(self, device):
        if torch.version.hip:
            self.skipTest("TMA descriptors are not available on ROCm")
        device_type = torch.device(device).type
        if device_type not in ("cuda", "xpu"):
            self.skipTest("requires CUDA or XPU")
        if (
            device_type == "cuda"
            and torch.get_device_module(device).get_device_capability()[0] < 9
        ):
            self.skipTest("requires Hopper or newer for TMA")
        if not has_triton_tensor_descriptor_host_tma():
            self.skipTest("requires Triton TensorDescriptor TMA support")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # The script's directory is first on sys.path, so keep the cache in a
            # subdir; see TestLazyCompileKernelCollision.
            cache_dir = os.path.join(tmp_dir, "inductor_cache")
            script_path = Path(tmp_dir) / "lazy_tma_global_scratch_repro.py"
            script = _LAZY_TMA_GLOBAL_SCRATCH_SCRIPT.replace("{device}", device)
            script_path.write_text(script, encoding="utf-8")
            env = {
                **os.environ,
                "CUDA_LAUNCH_BLOCKING": "1",
                "TORCHINDUCTOR_CACHE_DIR": cache_dir,
            }
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                cwd=tmp_dir,
                text=True,
                env=env,
                timeout=180,
            )

        stderr_tail = "\n".join(result.stderr.splitlines()[-80:])
        self.assertEqual(
            result.returncode,
            0,
            lambda msg: f"{msg}\nlazy TMA scratch regression subprocess failed:\n"
            f"returncode: {result.returncode}\n"
            f"stderr tail:\n{stderr_tail}",
        )


class DynamicShapesGpuWrapperGpuTests(InductorTestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_annotation_training(self, device):
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def create_test_fn():
            def test_fn():
                inp = torch.randn(batch_size, seq_length, hidden_size, device=device)
                weight = torch.randn(hidden_size, hidden_size, device=device)
                matmul_output = inp @ weight
                torch.nn.LayerNorm(hidden_size, device=device)(matmul_output)
                return True

            return test_fn

        fn = torch.compile(options={"annotate_training": True, "cpp_wrapper": True})(
            create_test_fn()
        )
        fn()


test_failures_gpu_wrapper = {
    "test_mm_plus_mm2_dynamic_shapes": test_torchinductor.TestFailure(
        ("gpu_wrapper",), is_skip=True
    ),
}

# Skip only on CUDA as wrapper dynamic shapes passes on ROCm.
# Per https://github.com/pytorch/pytorch/pull/172780
if not torch.version.hip:
    test_failures_gpu_wrapper["test_mm_plus_mm3_dynamic_shapes"] = (
        test_torchinductor.TestFailure(("gpu_wrapper",), is_skip=False)
    )


class _WrapperVariant(NamedTuple):
    target_cls: type
    fn_suffix: str
    dynamic_shapes: bool = False
    test_failures: dict | None = None
    tf_suffix: str | None = None
    xfail_props: tuple = ()


def make_test_case(name, source_device, tests, variant, check_code=True):
    test_name = f"{name}_{source_device}" if source_device else name
    func = getattr(tests, test_name)
    if not callable(func):
        raise AssertionError(f"{test_name} is not callable")

    def fn(self, device):
        # Some source classes (TestSelectAlgorithm, ComboKernelDynamicShapesTests)
        # do not declare device at all; only restore it when it existed.
        had_device = hasattr(tests, "device")
        previous_device = getattr(tests, "device", None)
        tests.device = device
        tests.setUpClass()
        tests.setUp()
        try:
            with torch._C._PreserveDispatchKeyGuard():
                torch._C._dispatch_tls_set_dispatch_key_included(
                    torch._C.DispatchKey.Dense, True
                )

                _, code = test_torchinductor.run_and_get_cpp_code(func)
                if check_code:
                    self.assertEqual("CppWrapperCodeCache" in code, True)
        finally:
            try:
                tests.tearDown()
                tests.tearDownClass()
            finally:
                if had_device:
                    tests.device = previous_device
                else:
                    del tests.device

    # Metadata must be attached to the bare function: replacing __dict__ after
    # decorating would drop functools.wraps' __wrapped__ and hide the device
    # parameter from instantiate_device_type_tests.
    fn.__name__ = f"{test_name}_{variant.fn_suffix}"
    fn.__dict__ = copy.deepcopy(func.__dict__)
    # A copied __wrapped__/__signature__ would redirect inspect.signature() to
    # the source test and hide the injected device parameter.
    fn.__dict__.pop("__wrapped__", None)
    fn.__dict__.pop("__signature__", None)
    fn = config.patch(cpp_wrapper=True)(fn)
    if variant.dynamic_shapes:
        fn = torch._dynamo.config.patch(assume_static_by_default=False)(fn)

    full_name = fn.__name__

    for prop in variant.xfail_props:
        if hasattr(func, prop):
            fn = unittest.expectedFailure(fn)
            break

    if variant.test_failures is not None:
        tf_suffix = variant.tf_suffix or variant.fn_suffix
        template_name = (
            f"{test_name}_dynamic_shapes" if variant.dynamic_shapes else test_name
        )
        tf = variant.test_failures.get(template_name)
        if tf and tf_suffix in tf.suffixes:
            if tf.is_skip:
                fn = unittest.skip("Skipped!")(fn)
            else:
                fn = unittest.expectedFailure(fn)

    setattr(variant.target_cls, full_name, fn)


if RUN_GPU:

    class BaseTest(NamedTuple):
        name: str
        # Addresses the GPU_TYPE-suffixed copies that copy_tests() put on
        # test_torchinductor.GPUTests; the runtime device comes from the
        # device parameter injected by instantiate_device_type_tests.
        source_device: str | None = GPU_TYPE
        tests: InductorTestCase = test_torchinductor.GPUTests()
        check_code: bool = True

    static_variant = _WrapperVariant(
        target_cls=TestGpuWrapper,
        fn_suffix="gpu_wrapper",
        test_failures=test_failures_gpu_wrapper,
        tf_suffix="gpu_wrapper",
    )
    dynamic_variant = _WrapperVariant(
        target_cls=DynamicShapesGpuWrapperGpuTests,
        fn_suffix="dynamic_shapes_gpu_wrapper",
        dynamic_shapes=True,
        test_failures=test_failures_gpu_wrapper,
        tf_suffix="gpu_wrapper",
        xfail_props=("_expected_failure_dynamic", "_expected_failure_dynamic_wrapper"),
    )

    # Maintain two separate test lists for cuda and cpp for now

    for item in [
        BaseTest("test_add_complex"),
        BaseTest("test_add_complex4"),
        BaseTest("test_as_strided"),  # buffer reuse
        BaseTest("test_batch_norm_2d_2"),
        BaseTest("test_bernoulli1_combo_kernels_False"),
        BaseTest("test_bernoulli1_combo_kernels_True"),
        BaseTest("test_bitwise"),  # int32
        BaseTest("test_bmm1"),
        BaseTest("test_bmm2"),
        BaseTest("test_buffer_use_after_remove"),
        BaseTest("test_cat"),  # alias
        BaseTest("test_convolution1"),
        BaseTest("test_conv_backward"),
        BaseTest("test_custom_op_1"),
        BaseTest("test_custom_op_2"),
        BaseTest("test_custom_op_3"),
        BaseTest("test_efficient_zero_tensor_avoids_oom"),
        BaseTest("test_embedding_bag"),  # test default FallbackKernel
        BaseTest("test_index_put_deterministic_fallback"),
        BaseTest("test_adding_tensor_offsets"),
        BaseTest("test_index_tensor"),
        BaseTest("test_inductor_layout_optimization_input_mutations"),
        BaseTest("test_insignificant_strides"),
        BaseTest("test_layer_norm"),
        BaseTest("test_linear1"),
        BaseTest("test_linear2"),
        BaseTest("test_mm_views"),
        BaseTest("test_multi_device"),
        BaseTest("test_multi_threading"),
        BaseTest("test_pow3"),
        BaseTest("test_profiler_mark_wrapper_call"),
        BaseTest("test_randint"),
        BaseTest("test_reduction1"),  # Reduction
        BaseTest("test_relu"),  # multiple inputs
        BaseTest("test_repeat_interleave_2"),
        BaseTest("test_roi_align"),
        BaseTest("test_scalar_input"),
        BaseTest("test_scaled_dot_product_attention"),
        BaseTest("test_scaled_dot_product_efficient_attention"),
        BaseTest("test_sort"),
        BaseTest("test_silu"),  # single input, single output
        BaseTest("test_sum_dtype"),  # float64
        BaseTest("test_sum_int"),  # bool, int64, int8, uint8
        BaseTest("test_transpose"),  # multiple outputs, buffer clear
        *[
            BaseTest(f"test_unspec_inputs_{str(dtype)[6:]}")
            for dtype in test_torchinductor.test_dtypes
        ],
        BaseTest("test_consecutive_split_cumprod"),
        BaseTest("test_pointwise_hermite_polynomial_he"),
        BaseTest("test_pointwise_hermite_polynomial_h"),
        BaseTest(
            "test_foreach_cpp_wrapper",
            tests=test_foreach.ForeachTests(),
        ),  # test foreach
        BaseTest(
            "test_enable_dynamic_shapes_cpp_wrapper",
            tests=test_foreach.ForeachTests(),
        ),
        BaseTest(
            "test_dynamic_shapes_persistent_reduction_mixed_x_dim",
            tests=test_combo_kernels.ComboKernelDynamicShapesTests(),
        ),
        BaseTest(
            "test_cat_slice_cat",
            tests=test_pattern_matcher.TestPatternMatcher(),
        ),
        # TODO: Re-enable this test after fixing cuda wrapper for conv Triton templates with dynamic shapes.
        # This test is unstable: it succeeds when an ATEN kernel is used, and fails when a Triton kernel is used.
        # Currently it passes on CI (an ATEN kernel is chosen) and fails locally (a Triton kernel is chosen).
        # Ideally, it should succeed for whatever kernels.
        # BaseTest(
        #     "test_convolution1",
        #     source_device=None,
        #     tests=test_select_algorithm.TestSelectAlgorithm(),
        # ),
        BaseTest(
            "test_mm_plus_mm2",
            source_device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        BaseTest(
            "test_mm_plus_mm3",
            source_device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        BaseTest("test_fft_real_input"),
        BaseTest("test_fft_real_input_real_output"),
        *[
            # some dtypes may raise exception and be skipped in test_dtypeview, so set check_code to False here
            BaseTest(
                f"test_dtypeview_{str(dtype_x)[6:]}_{str(dtype_y)[6:]}",
                check_code=False,
            )
            for dtype_x, dtype_y in itertools.product(
                test_torchinductor.test_dtypes, test_torchinductor.test_dtypes
            )
        ],
        BaseTest("test_dtypeview_fusion"),
        # skip the next two tests if not enough SMs, logic for this is handled by TestSelectAlgorithm.setUp()
        BaseTest(
            "test_addmm",
            source_device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
        BaseTest(
            "test_linear_relu",
            source_device=None,
            tests=test_select_algorithm.TestSelectAlgorithm(),
        ),
    ]:
        for variant in (static_variant, dynamic_variant):
            make_test_case(
                item.name,
                item.source_device,
                item.tests,
                variant,
                check_code=item.check_code,
            )


instantiate_parametrized_tests(TestGpuWrapperGeneric)
instantiate_device_type_tests(
    TestGpuWrapper, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(TestGpuWrapperCudaCodegen, globals(), only_for="cuda")
instantiate_device_type_tests(TestGpuWrapperCuda, globals(), only_for="cuda")
instantiate_device_type_tests(
    TestLazyCompileKernelCollision, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestCppWrapperStaticInitDeadlock, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestLazyTmaGlobalScratch, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(
    DynamicShapesGpuWrapperGpuTests, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_GPU:
        run_tests(needs="filelock")
