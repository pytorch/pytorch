# Owner(s): ["module: inductor"]

import sympy

import torch
from torch._inductor.codegen.block_analysis import BlockPatternMatcher
from torch._inductor.utils import sympy_dot
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import dummy_graph
from torch.utils._sympy.functions import FloorDiv, Identity, ModularIndexing


# Some useful symbols
x, y = sympy.symbols("x y")


@instantiate_parametrized_tests
class BlockAnalysisTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Create a GraphLowering, so we can access V.graph.
        cls.graph = dummy_graph()

    @parametrize(
        "stride,symbol,expr",
        [
            (5, x, Identity(5 * x)),
            (4, y, 4 * Identity(y)),
            (3, x, Identity(3) * x),
        ],
    )
    def test_affine_identity(self, stride: int, symbol: sympy.Symbol, expr: sympy.Expr):
        # Test that we can handle an identity expression in affine indexing.
        matched_stride = BlockPatternMatcher.match_affine_block_expr(expr, symbol)
        self.assertEqual(matched_stride, stride)

    @parametrize(
        "dims,strides,symbol,expr",
        [
            (
                (2, 4),
                (4, 1),
                x,
                4 * FloorDiv(Identity(x), 4) + ModularIndexing(x, 1, 4),
            ),
            (
                (3, 9),
                (5, 2),
                x,
                5 * FloorDiv(x, 9) + 2 * ModularIndexing(Identity(x), 1, 9),
            ),
            ((2, 7), (1, 1), x, Identity(FloorDiv(x, 7) + ModularIndexing(x, 1, 7))),
        ],
    )
    def test_mod_div_identity(
        self,
        dims: tuple[int],
        strides: tuple[int],
        symbol: sympy.Symbol,
        expr: sympy.Expr,
    ):
        # Test that we can handle an identity expression in modular indexing.
        numel = int(torch.prod(torch.Tensor(dims)))
        num_dims = len(dims)
        with V.set_graph_handler(self.graph):
            match_result = BlockPatternMatcher.match_mod_div_block_expr(
                expr, symbol, numel, num_dims
            )

        # Check the matched block dimensions.
        self.assertNotEqual(match_result, None)
        matched_dims, matched_strides, matched_block_index_exprs = match_result
        self.assertEqual(matched_dims, dims)
        self.assertEqual(matched_strides, strides)

    @parametrize(
        "symbol,expr,subexpr",
        [
            (x, Identity(x), x),
            (x, Identity(x + 5), x),
            (y, Identity(x + 2 * y) + 5, 2 * y),
        ],
    )
    def test_subexpr_identity(
        self,
        symbol: sympy.Symbol,
        expr: sympy.Expr,
        subexpr: sympy.Expr,
    ):
        matched_subexpr = BlockPatternMatcher.get_subexpr_involving_symbol(expr, symbol)
        self.assertEqual(matched_subexpr, subexpr)

    def test_index_with_dynamic_shapes(self):
        s0 = sympy.var("s0", integer=True)
        s1 = sympy.var("s1", integer=True)

        dims = [s1, sympy.Integer(3)]
        num_dims = len(dims)
        numel = dims[0] * dims[1]
        strides = [sympy.Integer(3) * s0, sympy.Integer(1)]
        block_index_exprs = [
            FloorDiv(y, sympy.Integer(3)),
            ModularIndexing(y, sympy.Integer(1), sympy.Integer(3)),
        ]
        index = sympy_dot(strides, block_index_exprs)

        with V.set_graph_handler(self.graph):
            match = BlockPatternMatcher.match_mod_div_block_expr(
                index, y, numel, num_dims
            )
            sizevars = V.graph.sizevars
            for expected, actual in zip((dims, strides, block_index_exprs), match):
                if not (
                    isinstance(expected, (list, tuple))
                    and isinstance(actual, (list, tuple))
                ):
                    raise AssertionError(
                        f"Expected list/tuple types, got {type(expected)} and {type(actual)}"
                    )
                for expected_expr, actual_expr in zip(expected, actual):
                    if not (
                        isinstance(expected_expr, sympy.Expr)
                        and isinstance(actual_expr, sympy.Expr)
                    ):
                        raise AssertionError(
                            f"Expected sympy.Expr types, got {type(expected_expr)} and {type(actual_expr)}"
                        )
                    self.assertTrue(
                        sizevars.statically_known_equals(
                            sizevars.remove_precomputed_replacements(expected_expr),
                            sizevars.remove_precomputed_replacements(actual_expr),
                        )
                    )


if __name__ == "__main__":
    run_tests()
