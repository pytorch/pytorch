# Owner(s): ["module: inductor"]
import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


class TestPaternMatcher(TestCase):
    def test_mm_plus_mm(self):
        def fn(a, b, c, d):
            return torch.add(torch.mm(a, b), torch.mm(c, d))

        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 3)

    def test_addmm(self):
        def fn(a, b, c):
            return torch.add(a, torch.mm(b, c)), torch.mm(a, b) + c

        args = [
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
            torch.randn(16, 16, device="cuda"),
        ]
        e1, e2 = fn(*args)
        a1, a2 = torch.compile(fn)(*args)
        torch.testing.assert_close(a1, e1)
        torch.testing.assert_close(a2, e2)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 2)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

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
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

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
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

    def test_cat_slice_cat(self):
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
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)

        counters.clear()
        args = [
            torch.randn(2, 8, device="cuda"),
            torch.randn(2, 16, device="cuda"),
        ]
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)
        self.assertEqual(counters["inductor"]["pattern_matcher_nodes"], 4)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
