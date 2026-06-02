# Owner(s): ["module: inductor"]

import os
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
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

        TORCH_LIBRARY(issue153478_cpp_wrapper, m) {
            m.def("sum_list(Tensor[] xs) -> Tensor");
            m.def("sum_list_with_n(Tensor[] xs, SymInt n) -> Tensor");
            m.def("add_scaled_(Tensor(a!) x, Tensor y, float scale) -> ()");
        }

        TORCH_LIBRARY_IMPL(issue153478_cpp_wrapper, CPU, m) {
            m.impl("sum_list", sum_list);
            m.impl("sum_list_with_n", sum_list_with_n);
            m.impl("add_scaled_", add_scaled_);
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


if __name__ == "__main__":
    run_tests(needs="filelock")
