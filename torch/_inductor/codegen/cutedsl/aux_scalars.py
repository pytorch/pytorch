# mypy: allow-untyped-defs

import dataclasses
from collections.abc import Callable
from typing import ClassVar

import sympy


@dataclasses.dataclass(frozen=True, slots=True)
class CuteDSLAuxScalarBindings:
    symbols: tuple[sympy.Symbol, ...] = ()
    tuple_name: ClassVar[str] = "aux_scalars"

    def symbol_codes(
        self, *, cast_integer_to_int32: bool = False
    ) -> dict[sympy.Symbol, str]:
        codes: dict[sympy.Symbol, str] = {}
        for index, symbol in enumerate(self.symbols):
            code = f"{self.tuple_name}[{index}]"
            if cast_integer_to_int32 and symbol.is_integer:
                code = f"cutlass.Int32({code})"
            codes[symbol] = code
        return codes

    def symbol_codes_with_renames(
        self, rename: Callable[[sympy.Symbol], sympy.Expr]
    ) -> dict[sympy.Symbol, str]:
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
        if not self.symbols:
            return "None"
        scalar_values = []
        for symbol in self.symbols:
            scalar_type = "cutlass.Int64" if symbol.is_integer else "cutlass.Float64"
            scalar_values.append(f"{scalar_type}({print_expr(rename(symbol))})")
        if len(scalar_values) == 1:
            return f"({scalar_values[0]},)"
        return f"({', '.join(scalar_values)})"
