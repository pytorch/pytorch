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


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")
