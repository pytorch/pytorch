# Owner(s): ["module: functorch"]

"""Adapted from https://github.com/arogozhnikov/einops/blob/230ac1526c1f42c9e1f7373912c7f8047496df11/tests/test_ops.py.

MIT License

Copyright (c) 2018 Alex Rogozhnikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List, Tuple

import numpy as np

import torch

from functorch.einops import rearrange
from torch.testing._internal.common_utils import run_tests, TestCase

identity_patterns: List[str] = [
    "...->...",
    "a b c d e-> a b c d e",
    "a b c d e ...-> ... a b c d e",
    "a b c d e ...-> a ... b c d e",
    "... a b c d e -> ... a b c d e",
    "a ... e-> a ... e",
    "a ... -> a ... ",
    "a ... c d e -> a (...) c d e",
]

equivalent_rearrange_patterns: List[Tuple[str, str]] = [
    ("a b c d e -> (a b) c d e", "a b ... -> (a b) ... "),
    ("a b c d e -> a b (c d) e", "... c d e -> ... (c d) e"),
    ("a b c d e -> a b c d e", "... -> ... "),
    ("a b c d e -> (a b c d e)", "... ->  (...)"),
    ("a b c d e -> b (c d e) a", "a b ... -> b (...) a"),
    ("a b c d e -> b (a c d) e", "a b ... e -> b (a ...) e"),
]


class TestRearrange(TestCase):
    def test_collapsed_ellipsis_errors_out(self) -> None:
        x = torch.zeros([1, 1, 1, 1, 1])
        rearrange(x, "a b c d ... ->  a b c ... d")
        with self.assertRaises(ValueError):
            rearrange(x, "a b c d (...) ->  a b c ... d")

        rearrange(x, "... ->  (...)")
        with self.assertRaises(ValueError):
            rearrange(x, "(...) -> (...)")

    def test_ellipsis_ops(self) -> None:
        x = torch.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
        for pattern in identity_patterns:
            torch.testing.assert_close(rearrange(x, pattern), x, msg=pattern)

        for pattern1, pattern2 in equivalent_rearrange_patterns:
            torch.testing.assert_close(
                rearrange(x, pattern1),
                rearrange(x, pattern2),
                msg=f"{pattern1} vs {pattern2}",
            )

    def test_rearrange_consistency(self) -> None:
        shape = [1, 2, 3, 5, 7, 11]
        x = torch.arange(int(np.prod(shape, dtype=int))).reshape(shape)
        for pattern in [
            "a b c d e f -> a b c d e f",
            "b a c d e f -> a b d e f c",
            "a b c d e f -> f e d c b a",
            "a b c d e f -> (f e) d (c b a)",
            "a b c d e f -> (f e d c b a)",
        ]:
            result = rearrange(x, pattern)
            self.assertEqual(len(np.setdiff1d(x, result)), 0)
            self.assertIs(result.dtype, x.dtype)

        result = rearrange(x, "a b c d e f -> a (b) (c d e) f")
        torch.testing.assert_close(x.flatten(), result.flatten())

        result = rearrange(x, "a aa aa1 a1a1 aaaa a11 -> a aa aa1 a1a1 aaaa a11")
        torch.testing.assert_close(x, result)

        result1 = rearrange(x, "a b c d e f -> f e d c b a")
        result2 = rearrange(x, "f e d c b a -> a b c d e f")
        torch.testing.assert_close(result1, result2)

        result = rearrange(
            rearrange(x, "a b c d e f -> (f d) c (e b) a"),
            "(f d) c (e b) a -> a b c d e f",
            b=2,
            d=5,
        )
        torch.testing.assert_close(x, result)

        sizes = dict(zip("abcdef", shape))
        temp = rearrange(x, "a b c d e f -> (f d) c (e b) a", **sizes)
        result = rearrange(temp, "(f d) c (e b) a -> a b c d e f", **sizes)
        torch.testing.assert_close(x, result)

        x2 = torch.arange(2 * 3 * 4).reshape([2, 3, 4])
        result = rearrange(x2, "a b c -> b c a")
        self.assertEqual(x2[1, 2, 3], result[2, 3, 1])
        self.assertEqual(x2[0, 1, 2], result[1, 2, 0])

    def test_rearrange_permutations(self) -> None:
        # tests random permutation of axes against two independent numpy ways
        for n_axes in range(1, 10):
            input = torch.arange(2**n_axes).reshape([2] * n_axes)
            permutation = np.random.permutation(n_axes)
            left_expression = " ".join("i" + str(axis) for axis in range(n_axes))
            right_expression = " ".join("i" + str(axis) for axis in permutation)
            expression = left_expression + " -> " + right_expression
            result = rearrange(input, expression)

            for pick in np.random.randint(0, 2, [10, n_axes]):
                self.assertEqual(input[tuple(pick)], result[tuple(pick[permutation])])

        for n_axes in range(1, 10):
            input = torch.arange(2**n_axes).reshape([2] * n_axes)
            permutation = np.random.permutation(n_axes)
            left_expression = " ".join("i" + str(axis) for axis in range(n_axes)[::-1])
            right_expression = " ".join("i" + str(axis) for axis in permutation[::-1])
            expression = left_expression + " -> " + right_expression
            result = rearrange(input, expression)
            self.assertEqual(result.shape, input.shape)
            expected_result = torch.zeros_like(input)
            for original_axis, result_axis in enumerate(permutation):
                expected_result |= ((input >> original_axis) & 1) << result_axis

            torch.testing.assert_close(result, expected_result)

    def test_concatenations_and_stacking(self) -> None:
        for n_arrays in [1, 2, 5]:
            shapes: List[List[int]] = [[], [1], [1, 1], [2, 3, 5, 7], [1] * 6]
            for shape in shapes:
                arrays1 = [
                    torch.arange(i, i + np.prod(shape, dtype=int)).reshape(shape)
                    for i in range(n_arrays)
                ]
                result0 = torch.stack(arrays1)
                result1 = rearrange(arrays1, "...->...")
                torch.testing.assert_close(result0, result1)

    def test_unsqueeze(self) -> None:
        x = torch.randn((2, 3, 4, 5))
        actual = rearrange(x, "b h w c -> b 1 h w 1 c")
        expected = x.unsqueeze(1).unsqueeze(-2)
        torch.testing.assert_close(actual, expected)

    def test_squeeze(self) -> None:
        x = torch.randn((2, 1, 3, 4, 1, 5))
        actual = rearrange(x, "b 1 h w 1 c -> b h w c")
        expected = x.squeeze()
        torch.testing.assert_close(actual, expected)

    def test_0_dim_tensor(self) -> None:
        x = expected = torch.tensor(1)
        actual = rearrange(x, "->")
        torch.testing.assert_close(actual, expected)

        actual = rearrange(x, "... -> ...")
        torch.testing.assert_close(actual, expected)

    def test_dimension_mismatch_no_ellipsis(self) -> None:
        x = torch.randn((1, 2, 3))
        with self.assertRaises(ValueError):
            rearrange(x, "a b -> b a")

        with self.assertRaises(ValueError):
            rearrange(x, "a b c d -> c d b a")

    def test_dimension_mismatch_with_ellipsis(self) -> None:
        x = torch.tensor(1)
        with self.assertRaises(ValueError):
            rearrange(x, "a ... -> ... a")


if __name__ == "__main__":
    run_tests()
