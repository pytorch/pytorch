# Owner(s): ["module: inductor"]

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor import config, ir
from torch._inductor.codegen import cpp_wrapper_cpu as cwc, cpp_wrapper_gpu as cwg
from torch._inductor.codegen.cpp_wrapper_mps import CppWrapperMps
from torch._inductor.codegen.wrapper import KernelCallLine
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import (
    GPU_ALIGN_BYTES,
    make_codegen_buffer,
    run_and_get_cpp_code,
)
from torch._inductor.virtualized import V
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.inductor_utils import RUN_CPU
from torch.utils._ordered_set import OrderedSet


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


class CppWrapperXpu(CppWrapperMps):
    """A retargeting subclass, shared by the tests below."""

    device = "xpu"


class CppWrapperMpsDeviceTests(InductorTestCase):
    """Tests that CppWrapperMps guards and filters on its class-level ``device``.

    Production paths under test: the guard in _generate_kernel_call_helper and the
    device filter in codegen_additional_funcs, both reading ``self.device``, which
    CppWrapperGpu leaves alone when a subclass (or CppWrapperMps itself) sets
    it -- the CppWrapperCpu hasattr pattern. The factory test runs the real
    CppWrapperMps and CppWrapperGpu __init__ (only CppWrapperCpu's is stubbed), so
    a revert of that hasattr check cannot stay green; the other tests build
    wrappers via object.__new__ to skip the chain, which requires a complete graph
    lowering.
    """

    def _install_fake_graph(self):
        self.enterContext(
            V.set_graph_handler(
                SimpleNamespace(is_dual_wrapper_mode=False, aot_mode=False)
            )
        )

    def _make_wrapper(self, cls=None):
        self._install_fake_graph()
        wrapper = object.__new__(CppWrapperMps if cls is None else cls)
        wrapper._used_kernel_names = OrderedSet()
        wrapper.lines = []
        wrapper.prefix = make_codegen_buffer()
        return wrapper

    def _make_kernel_call_line(self, device_type, kernel_name):
        return KernelCallLine(
            wrapper=None,
            kernel_name=kernel_name,
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

    def test_guard_rejects_foreign_device(self):
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

    def test_matching_line_emits_handle(self):
        wrapper = self._make_wrapper()
        wrapper.lines = [self._make_kernel_call_line("mps", "mps_lib_0")]
        wrapper.codegen_additional_funcs()
        self.assertIn("get_mps_lib_0_handle", wrapper.prefix.getvalue())

    def test_nonmatching_line_emits_nothing(self):
        wrapper = self._make_wrapper()
        wrapper.lines = [self._make_kernel_call_line("xpu", "xpu_lib_0")]
        wrapper.codegen_additional_funcs()
        self.assertEqual(wrapper.prefix.getvalue(), "")

    def test_subclass_device_filters_for_its_own_device(self):
        wrapper = self._make_wrapper(CppWrapperXpu)
        # The same line shapes, judged against the subclass override: the mps
        # line is filtered out and the xpu line emits its handle.
        wrapper.lines = [
            self._make_kernel_call_line("mps", "mps_lib_0"),
            self._make_kernel_call_line("xpu", "xpu_lib_0"),
        ]
        wrapper.codegen_additional_funcs()
        prefix = wrapper.prefix.getvalue()
        self.assertNotIn("get_mps_lib_0_handle", prefix)
        self.assertIn("get_xpu_lib_0_handle", prefix)

    def test_subclass_guard_rejects_base_device(self):
        wrapper = self._make_wrapper(CppWrapperXpu)
        # An mps device fails for the subclass with the overridden message (and
        # would pass the base-class guard), proving the guard reads
        # ``self.device``.
        with self.assertRaisesRegex(
            AssertionError, "expected device.type == 'xpu', got mps"
        ):
            wrapper._generate_kernel_call_helper(
                device=torch.device("mps"),
                kernel_name="x_lib_0",
                call_args=[],
                arg_types=[],
            )

    def test_init_and_factory_keep_the_class_device_field(self):
        # CppWrapperCpu.__init__ stubbed: the point is the hasattr guard in
        # CppWrapperGpu.__init__ and the device routing into
        # get_device_op_overrides, both observed without a real graph.
        with (
            mock.patch.object(cwg, "get_gpu_type", lambda: "cuda"),
            mock.patch.object(cwc.CppWrapperCpu, "__init__", lambda self: None),
            mock.patch.object(
                cwg, "get_device_op_overrides", lambda d: f"overrides:{d}"
            ),
        ):
            subclass = CppWrapperXpu()
            base = CppWrapperMps()
            made = CppWrapperXpu.create(False, None, None)
        # Without the hasattr guard the real __init__ would have overwritten both
        # devices with "cuda", so the subclass's guard/filter would run against the
        # wrong device. create still returns a CppWrapperXpu there because the base
        # CppWrapperGpu.create is a classmethod returning cls(); a staticmethod
        # revert would hand back a plain CppWrapperGpu.
        self.assertEqual(subclass.device, "xpu")
        self.assertEqual(subclass.device_codegen, "overrides:xpu")
        self.assertEqual(base.device, "mps")
        self.assertEqual(base.device_codegen, "overrides:mps")
        self.assertIs(type(made), CppWrapperXpu)
        self.assertEqual(made.device, "xpu")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")
