# Owner(s): ["module: inductor"]

import os
import sys
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_cpp_code


libtest = torch.library.Library(  # noqa: SCOPED_LIBRARY
    "test_cpp_wrapper_asserts", "FRAGMENT"
)
ids = set()


def define_custom_op_for_test(id_, fn, fn_meta):
    if id_ not in ids:
        libtest.define(f"{id_}(Tensor self) -> Tensor")
        libtest.impl(id_, fn, "CPU")
        libtest.impl(id_, fn_meta, "Meta")
        ids.add(id_)


def unique_op_name(name):
    return f"{name}_{os.getpid()}"


@unittest.skipIf(sys.platform == "darwin", "CPU cpp_wrapper tests are not run on macOS")
class CppWrapperAssertTests(InductorTestCase):
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

        op_name = unique_op_name("foo_assert_codegen")
        define_custom_op_for_test(op_name, foo, foo_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            return getattr(torch.ops.test_cpp_wrapper_asserts, op_name)(a)

        _, code = run_and_get_cpp_code(torch.compile(fn), torch.randn(16, 32))
        self.assertIn("assert_size_stride(buf", code)
        self.assertIn("assert_alignment(buf", code)
        self.assertIn(f"torch.ops.test_cpp_wrapper_asserts.{op_name}.default", code)

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

        op_name = unique_op_name("slice2d_incorrect_meta_assert")
        define_custom_op_for_test(op_name, slice2d, slice2d_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            b = getattr(torch.ops.test_cpp_wrapper_asserts, op_name)(a)
            return torch.cos(b)

        compiled = torch.compile(fn)
        expected_error = (
            "Expect the tensor to be 16 bytes aligned. "
            r"Data pointer is misaligned by [1-9][0-9]* bytes "
            r"\(storage_offset=1, itemsize=4\)"
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            compiled(torch.randn(8, 24))

    @config.patch(
        cpp_wrapper=True,
        fx_graph_cache=False,
        implicit_fallbacks=True,
        alignment_asserts=True,
    )
    def test_fallback_output_alignment_assert_checks_data_pointer(self):
        def misaligned_base(x):
            storage = bytearray(x.numel() * x.element_size() + 16)
            base = torch.frombuffer(storage, dtype=torch.uint8)
            offset = 1 if (base.data_ptr() + 1) % 16 else 2
            result = torch.frombuffer(
                storage,
                dtype=x.dtype,
                count=x.numel(),
                offset=offset,
            ).reshape(x.shape)
            if result.storage_offset() != 0:
                raise AssertionError("expected external storage_offset to be zero")
            if result.data_ptr() % 16 == 0:
                raise AssertionError("expected external storage to be misaligned")
            return result

        def misaligned_base_meta(x):
            return torch.empty_like(x)

        op_name = unique_op_name("misaligned_base_assert")
        define_custom_op_for_test(op_name, misaligned_base, misaligned_base_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            return torch.cos(getattr(torch.ops.test_cpp_wrapper_asserts, op_name)(a))

        compiled = torch.compile(fn)
        expected_error = (
            "Expect the tensor to be 16 bytes aligned. "
            r"Data pointer is misaligned by [1-9][0-9]* bytes "
            r"\(storage_offset=0, itemsize=4\)"
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            compiled(torch.randn(8, 24))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")
