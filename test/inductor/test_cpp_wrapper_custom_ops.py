# Owner(s): ["module: inductor"]

import os
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE
from torch.testing._internal.inductor_utils import HAS_CPU


class TestCppWrapperCustomOps(InductorTestCase):
    @staticmethod
    def _load_issue153478_extension():
        from torch.utils.cpp_extension import load_inline

        cpp_src = r"""
        #include <ATen/ATen.h>
        #include <torch/library.h>

        at::Tensor sum_list(c10::List<at::Tensor> xs) {
            at::Tensor out = at::zeros_like(xs.get(0));
            for (const auto& x : xs) {
                out = out + x;
            }
            return out;
        }

        at::Tensor sum_list_with_n(c10::List<at::Tensor> xs, c10::SymInt n) {
            at::Tensor out = sum_list(xs);
            return out + n.expect_int();
        }

        void add_scaled_(at::Tensor& x, const at::Tensor& y, double scale) {
            x.add_(y, scale);
        }

        at::Tensor add_list_return(at::Tensor& x, c10::List<at::Tensor> xs) {
            for (const auto& y : xs) {
                x.add_(y);
            }
            return x + 1;
        }

        std::tuple<c10::optional<at::Tensor>, at::Tensor> optional_first(
            c10::List<at::Tensor> xs
        ) {
            return std::make_tuple(c10::nullopt, xs.get(0) + 1);
        }

        TORCH_LIBRARY(issue153478_cpp_wrapper, m) {
            m.def("sum_list(Tensor[] xs) -> Tensor");
            m.def("sum_list_with_n(Tensor[] xs, SymInt n) -> Tensor");
            m.def("add_scaled_(Tensor(a!) x, Tensor y, float scale) -> ()");
            m.def("add_list_return(Tensor(a!) x, Tensor[] xs) -> Tensor");
            m.def("optional_first(Tensor[] xs) -> (Tensor?, Tensor)");
        }

        TORCH_LIBRARY_IMPL(issue153478_cpp_wrapper, CPU, m) {
            m.impl("sum_list", sum_list);
            m.impl("sum_list_with_n", sum_list_with_n);
            m.impl("add_scaled_", add_scaled_);
            m.impl("add_list_return", add_list_return);
            m.impl("optional_first", optional_first);
        }
        """

        load_inline(
            name=f"issue153478_cpp_wrapper_ext_{os.getpid()}",
            cpp_sources=cpp_src,
            functions=[],
            extra_cflags=["-O0"],
            verbose=False,
        )

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "cpp_extension doesn't work here")
    @config.patch(cpp_wrapper=True)
    def test_cpp_custom_op_runtime_dispatch_stays_in_cpp(self):
        self._load_issue153478_extension()

        @torch.library.register_fake("issue153478_cpp_wrapper::sum_list")
        def _(xs):
            return torch.empty_like(xs[0])

        @torch.library.register_fake("issue153478_cpp_wrapper::sum_list_with_n")
        def _(xs, n):
            return torch.empty_like(xs[0])

        @torch.library.register_fake("issue153478_cpp_wrapper::add_scaled_")
        def _(x, y, scale: float):
            return None

        @torch.library.register_fake("issue153478_cpp_wrapper::add_list_return")
        def _(x, xs):
            return x

        @torch.library.register_fake("issue153478_cpp_wrapper::optional_first")
        def _(xs):
            return None, torch.empty_like(xs[0])

        def tensor_list_fn(x, y):
            return torch.ops.issue153478_cpp_wrapper.sum_list.default([x, y])

        def tensor_list_symint_fn(x, y):
            return torch.ops.issue153478_cpp_wrapper.sum_list_with_n.default(
                [x, y], x.shape[0]
            )

        def none_return_fn(x, y):
            z = x.clone()
            torch.ops.issue153478_cpp_wrapper.add_scaled_.default(z, y, 2.0)
            return z

        def mutation_and_return_fn(x, y):
            z = x.clone()
            return torch.ops.issue153478_cpp_wrapper.add_list_return.default(z, [y])

        def optional_first_fn(x):
            return torch.ops.issue153478_cpp_wrapper.optional_first.default([x])[1]

        x = torch.randn(4)
        y = torch.randn(4)

        out, code = run_and_get_code(
            torch.compile(tensor_list_fn, fullgraph=True), x, y
        )
        self.assertEqual(out, x + y)
        code_str = "\n".join(code)
        self.assertNotIn("PyObject_CallObject", code_str)
        self.assertNotIn("custom_op_wrapper", code_str)
        self.assertIn(
            'findSchemaOrThrow("issue153478_cpp_wrapper::sum_list", "")',
            code_str,
        )
        self.assertIn("callBoxed", code_str)

        out, code = run_and_get_code(
            torch.compile(tensor_list_symint_fn, fullgraph=True), x, y
        )
        self.assertEqual(out, x + y + x.shape[0])
        code_str = "\n".join(code)
        self.assertNotIn("PyObject_CallObject", code_str)
        self.assertNotIn("custom_op_wrapper", code_str)
        self.assertIn(
            'findSchemaOrThrow("issue153478_cpp_wrapper::sum_list_with_n", "")',
            code_str,
        )
        self.assertIn("callBoxed", code_str)

        out, code = run_and_get_code(
            torch.compile(none_return_fn, fullgraph=True), x, y
        )
        self.assertEqual(out, x + 2 * y)
        code_str = "\n".join(code)
        self.assertNotIn("PyObject_CallObject", code_str)
        self.assertNotIn("custom_op_wrapper", code_str)
        self.assertIn(
            'aoti_torch_call_dispatcher("issue153478_cpp_wrapper::add_scaled_", ""',
            code_str,
        )

        out, code = run_and_get_code(
            torch.compile(mutation_and_return_fn, fullgraph=True), x, y
        )
        self.assertEqual(out, x + y + 1)
        code_str = "\n".join(code)
        self.assertNotIn("PyObject_CallObject", code_str)
        self.assertNotIn("custom_op_wrapper", code_str)
        self.assertIn(
            'findSchemaOrThrow("issue153478_cpp_wrapper::add_list_return", "")',
            code_str,
        )
        self.assertIn("callBoxed", code_str)

        out, code = run_and_get_code(
            torch.compile(optional_first_fn, fullgraph=True), x
        )
        self.assertEqual(out, x + 1)
        code_str = "\n".join(code)
        self.assertNotIn("PyObject_CallObject", code_str)
        self.assertNotIn("custom_op_wrapper", code_str)
        self.assertIn(
            'findSchemaOrThrow("issue153478_cpp_wrapper::optional_first", "")',
            code_str,
        )
        self.assertIn("callBoxed", code_str)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @config.patch(implicit_fallbacks=True)
    def test_optional_output_size_assert_is_null_guarded(self):
        # A fallback custom op can declare an optional output (Tensor?) that is
        # absent (None) at runtime. Under cpp_wrapper the C-shim writes a null
        # AtenTensorHandle for the absent output; an unconditional
        # assert_size_stride on that slot (its fake kernel returned a tensor)
        # would dereference the null handle -> SIGSEGV (seen with fbgemm's
        # block_bucketize_sparse_features_inference). The fix emits the assert
        # for schema-optional outputs under a runtime null guard.
        #
        # Force the op onto the C-shim path (custom_ops_to_c_shims) and inspect
        # the generated code: the optional output (distinct fake shape 11x13) is
        # asserted inside `if (buf != nullptr) { ... }`, while the required (8x4)
        # output keeps an unconditional assert. Codegen runs before the .so is
        # loaded, so the absent hand-written C-shim symbol does not matter here.
        import io
        import logging

        from torch._inductor.codecache import output_code_log

        with torch.library._scoped_library("test_null_guard", "FRAGMENT") as lib:
            lib.define("f(Tensor x) -> (Tensor, Tensor?)")
            lib.impl("f", lambda x: (x + 1, None), "CompositeExplicitAutograd")
            lib.impl(
                "f",
                lambda x: (
                    torch.empty_like(x),
                    torch.empty(11, 13, dtype=x.dtype, device=x.device),
                ),
                "Meta",
            )

            op = torch.ops.test_null_guard.f.default
            c_shims = {
                op: [
                    "AOTITorchError aoti_torch_cpu_f(AtenTensorHandle x, "
                    "AtenTensorHandle* r0, AtenTensorHandle* r1)"
                ]
            }

            def fn(x):
                a, b = torch.ops.test_null_guard.f(x)
                return a + 1, b

            x = torch.randn(8, 4)

            cap = io.StringIO()
            handler = logging.StreamHandler(cap)
            output_code_log.addHandler(handler)
            prev_level = output_code_log.level
            output_code_log.setLevel(logging.DEBUG)
            try:
                with config.patch(
                    {"debug": True, "aot_inductor.custom_ops_to_c_shims": c_shims}
                ):
                    compiled = torch.compile(
                        fn, fullgraph=True, options={"cpp_wrapper": True}
                    )
                    try:
                        compiled(x)
                    except Exception:
                        # No real C-shim symbol is provided, so the generated .so
                        # fails to load; codegen has already emitted the wrapper.
                        pass
            finally:
                output_code_log.setLevel(prev_level)
                output_code_log.removeHandler(handler)

        code = cap.getvalue()
        assert_lines = [ln for ln in code.splitlines() if "assert_size_stride" in ln]
        self.assertTrue(assert_lines, "expected assert_size_stride in generated code")
        # Optional Tensor? output (fake shape 11x13) must be null-guarded.
        optional = [ln for ln in assert_lines if "13" in ln]
        self.assertTrue(optional, "optional output should still be size-asserted")
        self.assertTrue(
            all("nullptr" in ln for ln in optional),
            f"optional Tensor? output assert must be runtime null-guarded: {optional}",
        )
        # Required (non-optional) outputs keep an unconditional assert.
        self.assertTrue(
            any("nullptr" not in ln for ln in assert_lines),
            f"required outputs must keep unconditional asserts: {assert_lines}",
        )


if __name__ == "__main__":
    run_tests(needs="filelock")
