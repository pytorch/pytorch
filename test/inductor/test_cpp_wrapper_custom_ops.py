# Owner(s): ["module: inductor"]

import os
import sys
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_SANDCASTLE,
    parametrize,
)
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
    @config.patch(cpp_wrapper=True, implicit_fallbacks=True)
    @parametrize("dispatch", ("boxed", "python"))
    def test_cpp_wrapper_single_element_tensor_list_return(self, dispatch):
        with torch.library._scoped_library("mylib_single_list", "FRAGMENT") as m:
            return_type = "Tensor[]" if dispatch == "boxed" else "Tensor?[]"
            m.define(f"single_list(Tensor x) -> {return_type}")
            m.impl("single_list", lambda x: [x + 1], "CPU")
            m.impl("single_list", lambda x: [torch.empty_like(x)], "Meta")

            class Model(torch.nn.Module):
                def forward(self, x):
                    (y,) = torch.ops.mylib_single_list.single_list.default(x)
                    return y * 2

            model = Model()
            x = torch.randn(4)
            out, code = run_and_get_code(torch.compile(model, fullgraph=True), x)
            self.assertEqual(out, model(x))
            code = "\n".join(code)
            if dispatch == "boxed":
                self.assertIn(".callBoxed(", code)
                self.assertIn(".toTensorList()", code)
                self.assertNotIn("custom_op_wrapper", code)
                self.assertNotIn("PyObject_CallObject", code)
            else:
                self.assertIn("custom_op_wrapper", code)
                self.assertIn("PyList_GET_ITEM", code)
                self.assertNotIn(".callBoxed(", code)

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @unittest.skipUnless(
        sys.platform in ("linux", "win32"), "kernel profiling platform"
    )
    @config.patch(cpp_wrapper=True, implicit_fallbacks=True)
    @parametrize("enable_kernel_profile", (False, True))
    def test_cpp_wrapper_boxed_tensor_list_profiling(self, enable_kernel_profile):
        with torch.library._scoped_library("mylib_list_profile", "FRAGMENT") as m:
            m.define("single_list(Tensor x) -> Tensor[]")
            m.impl("single_list", lambda x: [x + 1], "CPU")
            m.impl("single_list", lambda x: [torch.empty_like(x)], "Meta")

            class Model(torch.nn.Module):
                def forward(self, x):
                    return torch.ops.mylib_list_profile.single_list(x)[0] * 2

            model = Model()
            x = torch.randn(4)
            with config.patch("cpp.enable_kernel_profile", enable_kernel_profile):
                compiled = torch.compile(model, fullgraph=True)
                actual, codes = run_and_get_code(compiled, x)
                self.assertEqual(actual, model(x))
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as profile:
                    actual = compiled(x)
                self.assertEqual(actual, model(x))

            code = "\n".join(codes)
            self.assertIn(".callBoxed(", code)
            self.assertNotIn("custom_op_wrapper", code)
            event_names = {event.key for event in profile.key_averages()}
            self.assertEqual(
                "mylib_list_profile.single_list.default" in event_names,
                enable_kernel_profile,
            )

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @config.patch(cpp_wrapper=True, implicit_fallbacks=True)
    @parametrize("expected_length,runtime_length", ((2, 0), (2, 1), (2, 3), (0, 1)))
    def test_cpp_wrapper_boxed_tensor_list_length_mismatch(
        self, expected_length, runtime_length
    ):
        returned_length = [expected_length]
        with torch.library._scoped_library("mylib_list_bounds", "FRAGMENT") as m:
            m.define("mixed(Tensor x) -> (Tensor[], Tensor)")
            m.impl(
                "mixed",
                lambda x: ([x + i for i in range(returned_length[0])], x + 1),
                "CPU",
            )
            m.impl(
                "mixed",
                lambda x: (
                    [torch.empty_like(x) for _ in range(expected_length)],
                    torch.empty_like(x),
                ),
                "Meta",
            )

            class Model(torch.nn.Module):
                def forward(self, x):
                    _, y = torch.ops.mylib_list_bounds.mixed(x)
                    return y * 2

            model = Model()
            x = torch.randn(4)
            compiled = torch.compile(model, fullgraph=True)
            actual, codes = run_and_get_code(compiled, x)
            self.assertEqual(actual, model(x))
            self.assertIn(".callBoxed(", "\n".join(codes))
            returned_length[0] = runtime_length
            self.assertEqual(model(x), (x + 1) * 2)
            with self.assertRaisesRegex(
                RuntimeError,
                f"mylib_list_bounds::mixed return 0 expected {expected_length} "
                f"tensors, got {runtime_length}",
            ):
                compiled(x)
            returned_length[0] = expected_length
            self.assertEqual(compiled(x), model(x))

    @unittest.skipIf(not HAS_CPU, "requires CPU")
    @config.patch(cpp_wrapper=True, implicit_fallbacks=True)
    @parametrize("optional_present", (False, True))
    @parametrize("empty_middle_list", (False, True))
    def test_cpp_wrapper_mixed_tensor_list_returns(
        self, optional_present, empty_middle_list
    ):
        with torch.library._scoped_library("mylib_mixed_lists", "FRAGMENT") as m:
            m.define("mixed(Tensor x) -> (Tensor[], Tensor?, Tensor[], Tensor)")
            m.impl(
                "mixed",
                lambda x: (
                    [x + 1, x + 2],
                    x + 3 if optional_present else None,
                    [] if empty_middle_list else [x + 4],
                    x + 5,
                ),
                "CPU",
            )
            m.impl(
                "mixed",
                lambda x: (
                    [torch.empty_like(x), torch.empty_like(x)],
                    torch.empty_like(x) if optional_present else None,
                    [] if empty_middle_list else [torch.empty_like(x)],
                    torch.empty_like(x),
                ),
                "Meta",
            )

            class Model(torch.nn.Module):
                def forward(self, x):
                    first, optional, middle, last = torch.ops.mylib_mixed_lists.mixed(x)
                    return first, optional, middle, last, first[0] + last

            model = Model()
            x = torch.randn(4)
            expected = model(x)
            compiled = torch.compile(model, fullgraph=True)
            actual, codes = run_and_get_code(compiled, x)
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual[0]), 2)
            self.assertEqual(len(actual[2]), 0 if empty_middle_list else 1)
            if not optional_present:
                self.assertIsNone(actual[1])
            next_x = torch.randn(4)
            self.assertEqual(compiled(next_x), model(next_x))
            self.assertEqual(actual, expected)
            code = "\n".join(codes)
            self.assertIn(".callBoxed(", code)
            self.assertIn(".toTensorList()", code)
            self.assertNotIn("custom_op_wrapper", code)
            self.assertRegex(
                code,
                r"std::move\(dispatch_stack_\d+\[3\]\)\.toTensor\(\)",
            )


instantiate_parametrized_tests(TestCppWrapperCustomOps)


if __name__ == "__main__":
    run_tests(needs="filelock")
