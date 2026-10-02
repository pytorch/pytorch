# Owner(s): ["module: inductor"]

import sys
import unittest
from types import SimpleNamespace

import torch
from torch._inductor import config, ir
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import GPU_ALIGN_BYTES, run_and_get_cpp_code
from torch._inductor.virtualized import V
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.inductor_utils import RUN_CPU


try:
    try:
        from .test_torchinductor import (
            define_custom_op_for_test,
            target_assert_alignment_regex,
        )
    except ImportError:
        from test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
            define_custom_op_for_test,
            target_assert_alignment_regex,
        )
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


@unittest.skipIf(
    IS_MACOS or not RUN_CPU, "requires a supported CPU cpp_wrapper test configuration"
)
class CppWrapperAssertTests(InductorTestCase):
    def test_inplace_view_alignment_uses_result_classification(self):
        class FakeInplaceViewExternKernel:
            @staticmethod
            def get_assert_name():
                return "mutated_input"

            @staticmethod
            def get_name():
                return "extern_result"

            @staticmethod
            def get_op_name():
                return "torch.ops.aten.set_.source_Tensor"

        class RecordingWrapper:
            comment = "//"

            def __init__(self):
                self.asserts = []
                self.comments = []

            def write_assert_alignment(self, name, alignment, op_name):
                self.asserts.append((name, alignment, op_name))

            def make_comment(self, comment):
                self.comments.append(comment)

        wrapper = RecordingWrapper()
        graph = SimpleNamespace(unaligned_buffers={"extern_result"})
        with config.patch(alignment_asserts=True), V.set_graph_handler(graph):
            ir.ExternKernel.codegen_alignment_asserts(
                FakeInplaceViewExternKernel(), wrapper
            )

        self.assertEqual(wrapper.asserts, [])
        self.assertEqual(
            wrapper.comments,
            [
                "// buffer mutated_input (op: torch.ops.aten.set_.source_Tensor) "
                "is assumed to be not aligned"
            ],
        )

        wrapper = RecordingWrapper()
        graph = SimpleNamespace(unaligned_buffers={"mutated_input"})
        with config.patch(alignment_asserts=True), V.set_graph_handler(graph):
            ir.ExternKernel.codegen_alignment_asserts(
                FakeInplaceViewExternKernel(), wrapper
            )

        self.assertEqual(
            wrapper.asserts,
            [
                (
                    "mutated_input",
                    GPU_ALIGN_BYTES,
                    "torch.ops.aten.set_.source_Tensor",
                )
            ],
        )
        self.assertEqual(wrapper.comments, [])

    @config.patch(
        cpp_wrapper=True,
        fx_graph_cache=False,
        implicit_fallbacks=True,
        alignment_asserts=True,
    )
    def test_fallback_output_asserts_are_generated(self):
        def foo(x):
            return 3 * x

        def foo_meta(x):
            return torch.empty_like(x)

        op_name = "cpp_wrapper_assert_codegen"
        define_custom_op_for_test(op_name, foo, foo_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            return getattr(torch.ops.test, op_name)(a)

        _, code = run_and_get_cpp_code(torch.compile(fn), torch.randn(16, 32))
        qualified_op_name = f"torch.ops.test.{op_name}.default"
        FileCheck().check("assert_size_stride(").check_regex(
            target_assert_alignment_regex(
                cpp_wrapper=True,
                op_name=qualified_op_name,
                alignment=GPU_ALIGN_BYTES,
            )
        ).run(code)

    @config.patch(
        cpp_wrapper=True,
        fx_graph_cache=False,
        implicit_fallbacks=True,
        alignment_asserts=True,
    )
    def test_fallback_output_alignment_assert_fails_for_incorrect_meta(self):
        def slice2d(x):
            return (3 * x)[..., 1:-15]

        def slice2d_meta(x):
            return torch.empty_like(x)[..., 0:-16]

        op_name = "cpp_wrapper_slice2d_incorrect_meta_assert"
        define_custom_op_for_test(op_name, slice2d, slice2d_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            b = getattr(torch.ops.test, op_name)(a)
            return torch.cos(b)

        compiled = torch.compile(fn)
        expected_error = (
            "Expect the tensor to be 16 bytes aligned. "
            "Fail due to storage_offset=1 itemsize=4"
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            compiled(torch.randn(8, 24))

    @config.patch(
        cpp_wrapper=True,
        fx_graph_cache=False,
        implicit_fallbacks=True,
        alignment_asserts=True,
    )
    def test_fallback_output_alignment_assert_uses_storage_offset(self):
        def misaligned_base(x):
            storage = bytearray(x.numel() * x.element_size() + GPU_ALIGN_BYTES)
            base = torch.frombuffer(storage, dtype=torch.uint8)
            offset = 1 if (base.data_ptr() + 1) % GPU_ALIGN_BYTES else 2
            return torch.frombuffer(
                storage,
                dtype=x.dtype,
                count=x.numel(),
                offset=offset,
            ).reshape(x.shape)

        def misaligned_base_meta(x):
            return torch.empty_like(x)

        sample = torch.randn(8, 24)
        eager_result = misaligned_base(sample)
        self.assertEqual(eager_result.storage_offset(), 0)
        self.assertNotEqual(eager_result.data_ptr() % GPU_ALIGN_BYTES, 0)

        op_name = "cpp_wrapper_misaligned_base_assert"
        define_custom_op_for_test(op_name, misaligned_base, misaligned_base_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            return torch.cos(getattr(torch.ops.test, op_name)(a))

        compiled = torch.compile(fn)
        self.assertEqual(compiled(sample), fn(sample))


class CppWrapperMpsDeviceTypeTests(InductorTestCase):
    """Tests that CppWrapperMps consumes the class-level device_type attribute.

    Companion to the device_type attribute in codegen/mps.py. The guard in
    _generate_kernel_call_helper and the device filter in
    codegen_additional_funcs are driven directly (production code paths, not
    copies). The wrapper is built with object.__new__ to avoid the full
    __init__ chain, which requires a complete graph lowering.
    """

    def _make_wrapper(self, cls=None):
        from types import SimpleNamespace

        from torch._inductor.codegen.cpp_wrapper_mps import CppWrapperMps
        from torch._inductor.utils import make_codegen_buffer
        from torch._inductor.virtualized import V
        from torch.utils._ordered_set import OrderedSet

        V.set_graph_handler(
            SimpleNamespace(
                is_dual_wrapper_mode=False,
                aot_mode=False,
                cpp_wrapper=True,
            )
        )

        if cls is None:
            cls = CppWrapperMps
        wrapper = object.__new__(cls)
        wrapper._used_kernel_names = OrderedSet()
        wrapper._lambda_counter = 0
        wrapper._cpu_triton_kernel_names = OrderedSet()
        wrapper.lines = []
        wrapper.prefix = make_codegen_buffer()
        return wrapper

    def _make_kernel_call_line(self, device_type):
        from torch._inductor.codegen.wrapper import KernelCallLine

        return KernelCallLine(
            wrapper=None,
            kernel_name="mps_lib_0",
            call_args=(),
            raw_keys=(),
            raw_args=(),
            arg_types=[],
            triton=False,
            triton_meta=None,
            inductor_meta=None,
            device=torch.device(device_type),
            graph_name="graph_0",
            original_fxnode_name="l0",
            current_stream_idx=None,
        )

    def test_guard_error_message_uses_device_type(self):
        wrapper = self._make_wrapper()
        # xpu (not cpu, not mps) reaches the class guard without taking the
        # CppWrapperCpu delegation path that cpu would take.
        with self.assertRaisesRegex(
            AssertionError, "expected device.type == 'mps', got xpu"
        ):
            wrapper._generate_kernel_call_helper(
                device=torch.device("xpu"),
                kernel_name="mps_lib_0",
                call_args=[],
                arg_types=[],
            )

    def test_subclass_device_type_override_passes_guard(self):
        from torch._inductor.codegen.cpp_wrapper_mps import CppWrapperMps

        class CppWrapperXpu(CppWrapperMps):
            device_type = "xpu"

        wrapper = self._make_wrapper(CppWrapperXpu)
        # The guard must read the subclass attribute: an mps device fails for
        # the subclass with the parameterized message (and would pass for the
        # base class), proving the attribute is actually consumed.
        with self.assertRaisesRegex(
            AssertionError, "expected device.type == 'xpu', got mps"
        ):
            wrapper._generate_kernel_call_helper(
                device=torch.device("mps"),
                kernel_name="x_lib_0",
                call_args=[],
                arg_types=[],
            )

    def test_additional_funcs_filters_by_device_type(self):
        wrapper = self._make_wrapper()
        wrapper.lines = [self._make_kernel_call_line("xpu")]
        # With only a non-matching device line, no shader library is emitted.
        wrapper.codegen_additional_funcs()
        self.assertEqual(wrapper.prefix.getvalue(), "")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")
