import collections
import functools
import textwrap

import sympy
from sympy import Expr, Symbol

from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from ..utils import sympy_dot, sympy_subs
from ..virtualized import V


class BlockPatternMatcher:
    """
    Matches block indexing expressions.
    """

    _indexing_wild_signed_int = functools.partial(
        sympy.Wild, properties=[lambda x: x.is_integer]
    )
    _indexing_wild_unsigned_int = functools.partial(
        sympy.Wild, properties=[lambda x: x.is_integer and x.is_nonnegative]
    )

    @classmethod
    def get_subexpr_involving_symbol(cls, expr: Expr, symbol: Symbol) -> Expr:
        """
        Given a sympy expression, return the subexpression comprised only of terms
        involving the specified symbol.

        For example, if `expr` is `x * 5 + x ** 2 + y * 2 + 5`, and `symbol` is `x`,
        this returns `x * 5 + x ** 2`.
        """
        expr = cls._preprocess(expr)
        return sympy.S.Zero + sum(
            term for term in sympy.Add.make_args(expr) if symbol in term.free_symbols
        )

    @staticmethod
    def factor_index_expr(expr: sympy.Expr, index_var: Symbol) -> sympy.Expr:
        """
        Given an index expression, factor the expression around
        - FloorDiv(index_var, ...)
        - ModularIndexing(index_var, ...)
        - xindex

        e.g. FloorDiv(index_var, d0)*s0 + FloorDiv(index_var, d0)*s1 ->
        FloorDiv(index_var, d0) * (s0 + s1)
        """
        centres = OrderedSet()
        for sub in sympy.preorder_traversal(expr):
            if isinstance(sub, FloorDiv) and sub.args[0] == index_var:
                centres.add(sub)
            elif isinstance(sub, ModularIndexing) and sub.args[0] == index_var:
                centres.add(sub)
        centres.add(index_var)

        expr_out = expr
        for c in centres:
            expr_out = sympy.collect(expr_out, c)
        return expr_out

    @staticmethod
    def get_slice_numels(dims: list[Expr]) -> list[Expr]:
        """
        Compute the cumulative size of each dimension's slice.
        This proceeds from the last dim up to the second.
        """
        numels = collections.deque([sympy.S.One])
        for dim in dims[:0:-1]:
            numel = dim * numels[0]
            numels.appendleft(numel)
        return [*numels]

    @staticmethod
    def _preprocess(expr: Expr) -> Expr:
        # Remove any Identity nodes, e.g. expand x + (5 * y) to x + 5 * y.
        # Disable mul and multinomial as those expansions affect op trees:
        # e.g. sympy expects to match Mul(a, b), but expansion has simplified
        # to Add(...). Even though they may be algebraically equivalent,
        # sympy `match` performs structural pattern matching
        return expr.expand(mul=False, multinomial=False, identity=True)

    @classmethod
    def match_mod_div_block_expr(
        cls,
        index: Expr,
        index_var: Symbol,
        numel: Expr,
        num_dims: int,
    ) -> tuple[list[Expr], list[Expr], list[Expr]] | None:
        """
        Matches modular indexing expressions, converting them to implied block dimensions and strides.
        See triton.py for more information.

        Warning: this function requires that `index`, `numel` and any other sympy
        expression does not have precomputed replacements since otherwise block
        pattern matching may fail.
        See [Note: Precomputed replacements with BlockPatternMatch]
        """
        index = cls._preprocess(index)

        # Pattern match to find the strides and offset.
        wild_unsigned_int = functools.partial(
            cls._indexing_wild_unsigned_int, exclude=[index_var]
        )
        wild_signed_int = functools.partial(
            cls._indexing_wild_signed_int, exclude=[index_var]
        )
        dims: list[Expr] = [
            wild_unsigned_int(f"dim_mod{idx}") for idx in range(num_dims)
        ]
        strides: list[Expr] = [
            wild_signed_int(f"stride_mod{idx}") for idx in range(num_dims)
        ]

        # The first dimension's index is computed by division.
        # The remaining are computed by modulo.
        slice_numels = cls.get_slice_numels(dims[:num_dims])
        block_index_exprs = [FloorDiv(index_var, slice_numels[0])] + [
            ModularIndexing(index_var, numel, dim)
            for dim, numel in zip(dims[1:], slice_numels[1:])
        ]

        # Calculate a linear index from block indices.
        match_expr = sympy_dot(strides, block_index_exprs)

        # Heuristic: if the number of dimensions is high, check that the minimum requirements
        # are met before attempting an expensive full match. see triton.py:match_mod_div_block
        # for more details. In short, here we check that each subexpression in sympy.Add contains
        # only FloorDiv or ModularIndexing expressions.
        if num_dims >= 5:
            stride = sympy.symbols("stride", cls=wild_signed_int)
            denom, other = sympy.symbols("denominator other", cls=wild_unsigned_int)
            mod_div_pattern = stride * ModularIndexing(index_var, denom, other)
            floor_div_pattern = stride * FloorDiv(index_var, denom)
            first_dim_floor_div_matched = False
            match_failed = False
            for arg in sympy.Add.make_args(index):
                if arg.match(floor_div_pattern):
                    # There should only be a single FloorDiv(index, denom) expression
                    # corresponding to the first dimension
                    if first_dim_floor_div_matched:
                        match_failed = True
                        break
                    first_dim_floor_div_matched = True
                elif arg.match(mod_div_pattern):
                    continue
                else:
                    match_failed = True
                    break

            if match_failed:
                return None

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

        # Replace wildcards with matched expressions.
        dims = [dims[0]] + [match[dim] for dim in dims[1:]]
        strides = [match[stride] for stride in strides]
        slice_numels = cls.get_slice_numels(dims)
        block_index_exprs = [sympy_subs(expr, match) for expr in block_index_exprs]

        sizevars = V.graph.sizevars

        # The leading dimension is not directly matched in our expression.
        # We solve for it by dividing the range tree numel by the product of
        # all other dimensions. We quit if they are not known to be divisible.
        assert dims[0] not in match, "Expected not to match the leading dimension!"
        if not sizevars.statically_known_multiple_of(numel, slice_numels[0]):
            return None
        dims[0] = numel / slice_numels[0]

        # Sanity check that we can recover the index from the matched subexpressions.
        matched_index = sympy_dot(strides, block_index_exprs)
        assert sizevars.statically_known_equals(
            matched_index,
            index,
        ), textwrap.dedent(
            f"""
            Invalid match!
            Index: {index}
            Matched expression: {matched_index}
            """
        )

        return dims, strides, block_index_exprs

    @classmethod
    def match_affine_block_expr(
        cls,
        index: Expr,
        index_var: Symbol,
    ) -> Expr | None:
        """
        Matches simple expressions of the form stride * index, returning the
        stride.
        """
        index = cls._preprocess(index)
        stride = cls._indexing_wild_signed_int(name="stride", exclude=[index_var])
        m = index.match(index_var * stride)
        if m is None:
            return None

        return m[stride]
