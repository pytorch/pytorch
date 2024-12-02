import collections
import functools
import textwrap
from typing import List, Optional, Tuple

import sympy
from sympy import Expr, Symbol

from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from ..utils import sympy_dot, sympy_subs
from ..virtualized import V


class BlockPatternMatcher:
    """
    Matches block indexing expressions.
    """

    @staticmethod
    def get_subexpr_involving_symbol(expr: Expr, symbol: Symbol) -> Expr:
        """
        Given a sympy expression, return the subexpression comprised only of terms
        involving the specified symbol.

        For example, if `expr` is `x * 5 + x ** 2 + y * 2 + 5`, and `symbol` is `x`,
        this returns `x * 5 + x ** 2`.
        """
        return sympy.S.Zero + sum(
            term for term in sympy.Add.make_args(expr) if symbol in term.free_symbols
        )

    @staticmethod
    def get_slice_numels(dims: List[Expr]) -> List[Expr]:
        """
        Compute the cumulative size of each dimension's slice.
        This proceeds from the last dim up to the second.
        """
        numels = collections.deque([sympy.S.One])
        for dim in dims[:0:-1]:
            numel = dim * numels[0]
            numels.appendleft(numel)
        return [*numels]

    @classmethod
    def match_mod_div_block_expr(
        cls,
        index: Expr,
        index_var: Symbol,
        numel: Expr,
        num_dims: int,
    ) -> Optional[Tuple[List[Expr], List[Expr], List[Expr]]]:
        """
        Matches modular indexing expressions, converting them to implied block dimensions and strides.
        See triton.py for more information.
        """

        # Pattern match to find the strides and offset.
        wild = functools.partial(sympy.Wild, exclude=[index_var])
        dims: List[Expr] = [wild(f"dim_mod{idx}") for idx in range(num_dims)]
        strides: List[Expr] = [wild(f"stride_mod{idx}") for idx in range(num_dims)]

        # The first dimension's index is computed by division.
        # The remaining are computed by modulo.
        slice_numels = cls.get_slice_numels(dims[:num_dims])
        block_index_exprs = [FloorDiv(index_var, slice_numels[0])] + [
            ModularIndexing(index_var, numel, dim)
            for dim, numel in zip(dims[1:], slice_numels[1:])
        ]

        # Calculate a linear index from block indices.
        match_expr = sympy_dot(strides, block_index_exprs)

        # Pattern match.
        match = index.match(match_expr)
        if match is None:
            return None

        # Provide default values for unmatched dims and strides.
        for dim in dims[1:]:
            if dim not in match:
                match[dim] = sympy.S.One
        for stride in strides[1:]:
            if stride not in match:
                match[stride] = sympy.S.Zero

        sizevars = V.graph.sizevars

        def get_match(expr: Expr) -> Expr:
            return sizevars.lookup_precomputed_size(match[expr])

        # Replace wildcards with matched expressions.
        dims = [dims[0]] + [get_match(dim) for dim in dims[1:]]
        strides = [get_match(stride) for stride in strides]
        slice_numels = cls.get_slice_numels(dims)
        block_index_exprs = [sympy_subs(expr, match) for expr in block_index_exprs]

        # The leading dimension is not directly matched in our expression.
        # We solve for it by dividing the range tree numel by the product of
        # all other dimensions. We quit if they are not known to be divisible.
        assert dims[0] not in match, "Expected not to match the leading dimension!"
        if not sizevars.statically_known_multiple_of(numel, slice_numels[0]):
            return None
        dims[0] = numel / slice_numels[0]

        # Sanity check that we can recover the index from the matched subexpressions.
        matched_index = sympy_dot(strides, block_index_exprs)
        assert sizevars.statically_known_equals(matched_index, index), textwrap.dedent(
            f"""
            Invalid match!
            Index: {index}
            Matched expression: {matched_index}
            """
        )

        return dims, strides, block_index_exprs
