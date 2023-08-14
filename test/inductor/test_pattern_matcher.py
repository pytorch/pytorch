# Owner(s): ["module: inductor"]
import copy
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import expectedFailureDynamicWrapper
from torch._dynamo.utils import count_calls, counters
from torch._inductor.fx_passes import joint_graph
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestPaternMatcher(TestCase):
    def test_mm_plus_mm(self):
        def fn(a, b, c, d):
            return torch.add(torch.mm(a, b), torch.mm(c, d))

        args_list = [
            (
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ),
            # https://github.com/pytorch/pytorch/issues/100670.
            (
                torch.randn(1, 4, device="cuda"),
                torch.randn(4, 2, device="cuda"),
                torch.randn(1, 2, device="cuda"),
                torch.randn(2, 1, device="cuda"),
            ),
            (
                torch.randn(1, 2, device="cuda"),
                torch.randn(2, 1, device="cuda"),
                torch.randn(1, 4, device="cuda"),
                torch.randn(4, 2, device="cuda"),
            ),
            (
                torch.randn(1, 4, device="cuda"),
                torch.randn(4, 2, device="cuda"),
                torch.randn(1, 5, device="cuda"),
                torch.randn(5, 2, device="cuda"),
            ),
        ]
        for args in args_list:
            counters.clear()
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 3)

    def test_addmm(self):
        def fn(a, b, c):
            return torch.add(a, torch.mm(b, c)), torch.mm(b, c) + a

        args_list = [
            (
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ),
            (
                torch.randn(16, 16, device="cuda"),
                torch.randn(1, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ),
            (
                torch.randn(1, 16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
                torch.randn(16, 16, device="cuda"),
            ),
            (4, torch.randn(16, 16, device="cuda"), torch.randn(16, 16, device="cuda")),
        ]
        for args in args_list:
            torch._dynamo.reset()
            counters.clear()
            e1, e2 = fn(*args)
            a1, a2 = torch.compile(fn)(*args)
            torch.testing.assert_close(a1, e1)
            torch.testing.assert_close(a2, e2)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 2)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    def test_addmm_activation(self):
        def fn_addmm_relu(input, mat1, mat2):
            return torch.nn.functional.relu(torch.addmm(input, mat1, mat2))

        def fn_addmm_gelu(input, mat1, mat2):
            return torch.nn.functional.gelu(torch.addmm(input, mat1, mat2))

        args = [
            torch.randn(20, device="cuda"),  # input
            torch.randn(10, 15, device="cuda"),  # mat1
            torch.randn(15, 20, device="cuda"),  # mat2
        ]

        for fn, atol in (
            (fn_addmm_relu, 1e-8),
            # higher tolerance due to the "tanh" approximation
            # in fused GELU epilogue vs. "none" without fusion
            (fn_addmm_gelu, 1e-3),
        ):
            expected = fn(*args)
            actual, (code,) = run_and_get_code(torch.compile(fn), *args)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
            self.assertTrue("_addmm_activation" in code)

        for fn in (fn_addmm_relu, fn_addmm_gelu):
            counters.clear()
            torch.compile(
                fn,
                # replacement disabled on max_autotune_gemm
                options={"max_autotune_gemm": True},
            )(*args)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 0)

        args_not_replaced = [
            # addmm + activation with a rank-2 input
            # is not fusable, hence not replaced
            torch.randn(10, 20, device="cuda"),  # input
            torch.randn(10, 15, device="cuda"),  # mat1
            torch.randn(15, 20, device="cuda"),  # mat2
        ]

        for fn in (fn_addmm_relu, fn_addmm_gelu):
            counters.clear()
            torch.compile(fn)(*args_not_replaced)
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)
            self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 0)

    def test_cat_mm(self):
        def fn(a, b, c):
            return torch.cat(
                [
                    torch.mm(a, b),
                    torch.mm(b, c),
                    torch.mm(a, c),
                ],
                1,
            )

        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 2)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 5)

    def test_cat_addmm(self):
        def fn(a, b, c):
            return torch.cat(
                [
                    torch.addmm(a, b, c),
                    torch.addmm(b, c, a),
                    torch.addmm(c, a, b),
                ],
                1,
            )

        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 2)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 5)

    @expectedFailureDynamicWrapper
    def test_cat_slice_cat(self):
        def check_counter(counter, expected):
            if not inductor_config.cpp_wrapper:
                self.assertEqual(counter, expected)
            else:
                # cpp_wrapper for the CUDA backend runs two passes
                self.assertEqual(counter, 2 * expected)

        def fn(a, b):
            cat_1 = torch.ops.aten.cat.default([a, b], 1)
            slice_1 = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
            slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
            return torch.ops.aten.cat.default([cat_1, slice_2], 1)

        args = [
            torch.randn(2, 32, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        check_counter(counters["inductor"]["pattern_matcher_count"], 1)
        check_counter(counters["inductor"]["pattern_matcher_nodes"], 4)

        counters.clear()
        args = [
            torch.randn(2, 8, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        check_counter(counters["inductor"]["pattern_matcher_count"], 1)
        check_counter(counters["inductor"]["pattern_matcher_nodes"], 4)

        # Verify we fallback to non-optimal path for negative `end`.
        def fn(a, b):
            cat_1 = torch.ops.aten.cat.default([a, b], 1)
            slice_1 = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
            slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, -1)
            return torch.ops.aten.cat.default([cat_1, slice_2], 1)

        counters.clear()
        args = [
            torch.randn(2, 8, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        check_counter(counters["inductor"]["pattern_matcher_count"], 1)
        check_counter(counters["inductor"]["pattern_matcher_nodes"], 4)

    def test_pointless_convert(self):
        def fn1(x):
            x = torch.ops.prims.convert_element_type.default(x, torch.float16)
            x = torch.ops.prims.convert_element_type.default(x, torch.float32)
            return x

        gm = torch.fx.symbolic_trace(fn1)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 1)

        def fn2(x):
            x = torch.ops.prims.convert_element_type.default(x, torch.int32)
            x = torch.ops.prims.convert_element_type.default(x, torch.float32)
            return x

        gm = torch.fx.symbolic_trace(fn2)
        self.assertEqual(count_calls(gm.graph), 2)
        joint_graph.joint_graph_passes(gm)
        self.assertEqual(count_calls(gm.graph), 2)

    def test_pointless_cumsum(self):
        def fn1():
            ones = torch.full(
                [1, 128], 1, layout=torch.strided, dtype=torch.float32
            ).to(torch.int64)
            return torch.cumsum(ones, 1) * ones

        def fn2():
            ones = torch.full(
                [55, 10], 1, layout=torch.strided, dtype=torch.float32
            ).to(torch.int64)
            return torch.cumsum(ones, 1)

        for fn in (fn1, fn2):
            result, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True))
            self.assertNotIn("aten.cumsum", code)
            self.assertEqual(result, fn())
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
            counters.clear()

    def test_splitwithsizes_cat(self):
        # Good case
        def fn(a):
            split_with_sizes = torch.ops.aten.split_with_sizes.default(a, [8, 24], 1)
            getitem = split_with_sizes[0]
            getitem_1 = split_with_sizes[1]
            cat = torch.ops.aten.cat.default([getitem, getitem_1], 1)
            return cat**2

        args = [
            torch.randn(2, 32, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)
        counters.clear()

        # Not all getitems are passed to cat
        def fn(a):
            split_with_sizes = torch.ops.aten.split_with_sizes.default(a, [8, 8, 16], 1)
            getitem = split_with_sizes[0]
            getitem_1 = split_with_sizes[1]
            getitem_2 = split_with_sizes[2]
            cat = torch.ops.aten.cat.default([getitem, getitem_1], 1)
            return cat**2 + getitem_2

        args = [
            torch.randn(2, 32, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 0)
        counters.clear()

        # Different dimensions  (TODO this case should be handled by replacing with a reshape)
        def fn(a):
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                a, [8, 8, 8, 8], 1
            )
            cat = torch.ops.aten.cat.default(split_with_sizes, 0)
            return cat**2

        args = [
            torch.randn(2, 32, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 0)

        # https://github.com/pytorch/pytorch/issues/99686.
        def fn(a):
            x = torch.ops.aten.split_with_sizes.default(a, [3, 2, 3], dim=1)
            cat = torch.ops.aten.cat.default([x[1], x[0], x[2]], dim=1)
            return cat

        args = [
            torch.randn(1, 8, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 0)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 0)

    def test_match_with_mutation(self):
        from torch._inductor.pattern_matcher import (
            CallFunction,
            KeywordArg,
            PatternMatcherPass,
            register_graph_pattern,
        )

        counter = 0
        test_pass = PatternMatcherPass(prevent_match_across_mutations=True)

        @register_graph_pattern(
            CallFunction(
                torch.add, KeywordArg("x"), CallFunction(torch.sin, KeywordArg("x"))
            ),
            pass_dict=test_pass,
        )
        def _test(match, x):
            nonlocal counter
            counter += 1

        def fn0(x, y):
            a = torch.sin(x)
            b = torch.add(x, a)
            return b

        def fn1(x, y):
            a = torch.sin(x)
            x.copy_(y)
            b = torch.add(x, a)
            return b

        def fn2(x, y):
            a = torch.sin(x)
            with torch.no_grad():
                b = torch.add(x, a)
            return b

        def fn3(x, y):
            a = torch.sin(x)
            with torch.autocast("cuda"):
                b = torch.add(x, a)
            return b

        def fn4(x, y):
            a = torch.sin(x)
            torch.manual_seed(1234)
            b = torch.add(x, a)
            return b

        def fn5(x, y):
            a = torch.sin(x)
            torch.add(y, 1, out=x)
            b = torch.add(x, a)
            return b

        args = [
            torch.randn(5, 5, device="cuda"),
            torch.randn(5, 5, device="cuda"),
        ]

        with unittest.mock.patch(
            "torch._inductor.fx_passes.pre_grad.pattern_matcher_passes", [test_pass]
        ):
            for fn in (fn0, fn1, fn2, fn3, fn4, fn5):
                counter = 0
                expected = fn(*copy.deepcopy(args))
                actual = torch.compile(fn)(*copy.deepcopy(args))
                # should not match
                self.assertEqual(counter, int(fn is fn0))
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
