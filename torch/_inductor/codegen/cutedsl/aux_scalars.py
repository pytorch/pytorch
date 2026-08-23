# mypy: allow-untyped-defs

import dataclasses
from collections.abc import Callable
from typing import ClassVar

import sympy


@dataclasses.dataclass(frozen=True, slots=True)
class CuteDSLAuxScalarBindings:
    """Render symbolic shape captures through FA4's runtime aux_scalars tuple.

    Inductor represents captured dynamic ints/floats as SymPy expressions in the
    other-buffer list, while tensor captures remain TensorBox-backed aux_tensors.
    CuTe kernels receive those scalar values as ordinary kernel arguments, wrap
    them for FA4, and rewrite matching symbols in generated score_mod/mask_mod
    expressions to tuple lookups.
    """

    symbols: tuple[sympy.Symbol, ...] = ()
    tuple_name: ClassVar[str] = "aux_scalars"

    def symbol_codes(self) -> dict[sympy.Symbol, str]:
        """Render symbols as tuple lookups for CuTe expressions."""
        return {
            symbol: f"{self.tuple_name}[{index}]"
            for index, symbol in enumerate(self.symbols)
        }

    def symbol_codes_with_renames(
        self, rename: Callable[[sympy.Symbol], sympy.Expr]
    ) -> dict[sympy.Symbol, str]:
        """Include Inductor-renamed symbols used in generated kernel signatures."""
        codes = self.symbol_codes()
        for symbol, code in list(codes.items()):
            renamed = rename(symbol)
            if isinstance(renamed, sympy.Symbol):
                codes[renamed] = code
        return codes

    def tuple_expr(
        self,
        rename: Callable[[sympy.Symbol], sympy.Expr],
        print_expr: Callable[[sympy.Expr], str],
    ) -> str:
        """Render runtime scalar values with FA4-compatible scalar wrapper types."""
        if not self.symbols:
            return "None"
        scalar_values = []
        for symbol in self.symbols:
            scalar_type = "cutlass.Int64" if symbol.is_integer else "cutlass.Float64"
            scalar_values.append(f"{scalar_type}({print_expr(rename(symbol))})")
        if len(scalar_values) == 1:
            return f"({scalar_values[0]},)"
        return f"({', '.join(scalar_values)})"
