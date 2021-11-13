import torch
import unittest

import numpy as np

import torch
from torch import fx
import functorch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase
import unittest

from functorch.compile import memory_efficient_operator_authoring
from functorch.compile import compiled_function

from torch.testing._internal.common_utils import TestCase, run_tests


class TestCompileCache(TestCase):
    def test_recompilation_on_broadcast(self):
        def fn(x, bias):
            return x + bias

        def check(a, b, mem_optimized_fn, fn):
            a_clone = a.clone().detach().requires_grad_(True)
            b_clone = b.clone().detach().requires_grad_(True)
            ref = fn(a, b)
            ref.sum().backward()

            res = mem_optimized_fn(a_clone, b_clone)
            res.sum().backward()
            assert torch.allclose(res, ref)
            assert torch.allclose(a.grad, a_clone.grad)
            assert torch.allclose(b.grad, b_clone.grad)

        for hasher_type in ["DynamicShapeHasher", "StaticShapeHasher"]:
            functorch.compile.clear_compile_cache()
            start_num_recomps = functorch.compile.num_of_recompilations()
            mem_optimized_fn = memory_efficient_operator_authoring(
                fn,
                compiler_name="torchscript_nnc",
                hasher_type=hasher_type,
            )

            a = torch.randn(10, 20, requires_grad=True)
            b = torch.randn(20, requires_grad=True)
            check(a, b, mem_optimized_fn, fn)

            a = torch.randn(10, 20, requires_grad=True)
            b = torch.randn(10, 20, requires_grad=True)
            check(a, b, mem_optimized_fn, fn)

            end_num_recomps = functorch.compile.num_of_recompilations()

            total_recomps = end_num_recomps - start_num_recomps
            assert total_recomps == 2

    def test_compilation_for_dynamic_shape(self):
        def fn(x, bias):
            return x + bias

        def check(a, b, mem_optimized_fn, fn):
            a_clone = a.clone().detach().requires_grad_(True)
            b_clone = b.clone().detach().requires_grad_(True)
            ref = fn(a, b)
            ref.sum().backward()

            res = mem_optimized_fn(a_clone, b_clone)
            res.sum().backward()
            assert torch.allclose(res, ref)
            assert torch.allclose(a.grad, a_clone.grad)
            assert torch.allclose(b.grad, b_clone.grad)

        for hasher_type in ["DynamicShapeHasher", "StaticShapeHasher"]:
            functorch.compile.clear_compile_cache()
            start_num_recomps = functorch.compile.num_of_recompilations()
            mem_optimized_fn = memory_efficient_operator_authoring(
                fn, compiler_name="torchscript_nnc", hasher_type=hasher_type
            )

            for s in range(10, 20):
                a = torch.randn(s, requires_grad=True)
                b = torch.randn(s, requires_grad=True)
                check(a, b, mem_optimized_fn, fn)

            for s in range(10, 20):
                a = torch.randn(s, requires_grad=True)
                b = torch.randn(s, requires_grad=True)
                check(a, b, mem_optimized_fn, fn)

            end_num_recomps = functorch.compile.num_of_recompilations()

            total_recomps = end_num_recomps - start_num_recomps
            if hasher_type == "DynamicShapeHasher":
                assert total_recomps == 1
            elif hasher_type == "StaticShapeHasher":
                assert total_recomps == 10

            for s in range(10, 20):
                a = torch.randn(s, s, requires_grad=True)
                b = torch.randn(s, s, requires_grad=True)
                check(a, b, mem_optimized_fn, fn)

            end_num_recomps = functorch.compile.num_of_recompilations()

            total_recomps = end_num_recomps - start_num_recomps
            if hasher_type == "DynamicShapeHasher":
                assert total_recomps == 2
            elif hasher_type == "StaticShapeHasher":
                assert total_recomps == 20

    def test_global_cache_no_recompilations(self):
        def f(x, bias):
            return x + bias

        def _nop_compile(x, _):
            return x

        def g(x, bias):
            return compiled_function(
                f, _nop_compile, _nop_compile, hasher_type="DynamicShapeHasher"
            )(x, bias)

        def check(a, b, mem_optimized_fn, fn):
            a_clone = a.clone().detach().requires_grad_(True)
            b_clone = b.clone().detach().requires_grad_(True)
            ref = fn(a, b)
            ref.sum().backward()

            res = mem_optimized_fn(a_clone, b_clone)
            res.sum().backward()
            assert torch.allclose(res, ref)
            assert torch.allclose(a.grad, a_clone.grad)
            assert torch.allclose(b.grad, b_clone.grad)

        start_num_recomps = functorch.compile.num_of_recompilations()
        for _ in range(10):
            a = torch.randn(10, 20, requires_grad=True)
            b = torch.randn(10, 20, requires_grad=True)
            check(a, b, g, f)

        end_num_recomps = functorch.compile.num_of_recompilations()
        total_recomps = end_num_recomps - start_num_recomps
        assert total_recomps == 1

    def test_multiple_functions(self):
        def f(x, bias):
            return x + bias

        def g(x, y):
            return x * y

        def check(a, b, mem_optimized_fn, fn):
            a_clone = a.clone().detach().requires_grad_(True)
            b_clone = b.clone().detach().requires_grad_(True)
            ref = fn(a, b)
            ref.sum().backward()

            res = mem_optimized_fn(a_clone, b_clone)
            res.sum().backward()
            assert torch.allclose(res, ref)
            assert torch.allclose(a.grad, a_clone.grad)
            assert torch.allclose(b.grad, b_clone.grad)

        for hasher_type in ["DynamicShapeHasher", "StaticShapeHasher"]:
            functorch.compile.clear_compile_cache()
            mem_optimized_f = memory_efficient_operator_authoring(
                f, compiler_name="torchscript_nnc", hasher_type=hasher_type
            )
            mem_optimized_g = memory_efficient_operator_authoring(
                g, compiler_name="torchscript_nnc", hasher_type=hasher_type
            )

            start_num_recomps = functorch.compile.num_of_recompilations()
            a = torch.randn(10, requires_grad=True)
            b = torch.randn(10, requires_grad=True)
            check(a, b, mem_optimized_f, f)

            a = torch.randn(10, requires_grad=True)
            b = torch.randn(10, requires_grad=True)
            check(a, b, mem_optimized_g, g)

            end_num_recomps = functorch.compile.num_of_recompilations()
            total_recomps = end_num_recomps - start_num_recomps
            assert total_recomps == 2

            # Force recompilation for function f and check num of recompilations again
            a = torch.randn(10, 20, requires_grad=True)
            b = torch.randn(10, 20, requires_grad=True)
            check(a, b, mem_optimized_f, f)

            end_num_recomps = functorch.compile.num_of_recompilations()
            total_recomps = end_num_recomps - start_num_recomps
            assert total_recomps == 3


if __name__ == "__main__":
    run_tests()
