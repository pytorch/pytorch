import re
from typing import Union

import sympy
from sympy.printing.printer import Printer


class ExprPrinter(Printer):
    def is_symbol(self, string: str) -> bool:
        # Customization point for indutor's ExprPrinter
        # Variables shouldn't be surrounded in parens (for brevity)
        return False

    @staticmethod
    def all_in_parens(string: str) -> bool:
        if string[0] != "(" or len(string) < 2:
            return False
        count = 1
        for i, char in enumerate(string[1:]):
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
            if count == 0 and i != len(string) - 2:
                return False
        assert count == 0
        return True

    def paren(self, expr: Union[sympy.Expr, str]) -> str:
        if isinstance(expr, sympy.Symbol):
            return self._print(expr)

        string = self._print(expr) if not isinstance(expr, str) else expr

        if (
            self.is_symbol(string)
            or re.match(r"^[a-z0-9_.]+$", string, re.I)
            or re.match(r"^\([^)]*\)$", string, re.I)
            or string == ""
        ):
            return string
        # don't put extra parens for strings that are already wrapped in parens
        if self.all_in_parens(string):
            return string
        return f"({string})"

    def _print_Pow(self, expr):
        # Pow() confuses triton
        base, exp = expr.args
        # NB: Remember this is sizevar computation!  You don't typically
        # expect to have to do floating point computation including exponents
        # in sizevar compute.  Instead of adding support for floating
        # point pow, you should make upstream retranslate the Sympy expression
        # into Tensor expressions earlier and do that instead.
        if exp == 0.5:
            return self._helper_sqrt(base)  # type: ignore[attr-defined]
        elif exp == -0.5:
            return "1/" + self._helper_sqrt(base)  # type: ignore[attr-defined]
        base = self.paren(base)
        assert exp == int(exp), exp
        exp = int(exp)
        if exp > 0:
            return "*".join([base] * exp)
        elif exp < 0:
            return "1/" + self.paren("*".join([base] * abs(exp)))
        else:  # exp == 0
            return "1"

    def _print_Relational(self, expr):
        return f"{self.paren(expr.lhs)} {expr.rel_op} {self.paren(expr.rhs)}"

    def _print_Mul(self, expr):
        return "*".join(map(self.paren, expr.args))

    def _print_Add(self, expr):
        return " + ".join(map(self.paren, expr.args))

    def _print_Mod(self, expr):
        return " % ".join(map(self.paren, expr.args))

    def _print_CleanDiv(self, expr):
        return self._print_FloorDiv(expr)  # type: ignore[attr-defined]


class PythonPrinter(ExprPrinter):
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(x)
        div = self.paren(div)
        mod = self.paren(mod)
        if div != "1":
            x = f"({x} // {div})"
        return f"{x} % {mod}"

    def _print_FloorDiv(self, expr):
        x, div = expr.args
        x = self.paren(x)
        div = self.paren(div)
        return f"({x} // {div})"

    def _helper_sqrt(self, expr):
        return f"math.sqrt({self._print(expr)})"

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return f"math.floor({self._print(expr.args[0])})"

    def _print_ceiling(self, expr):
        assert len(expr.args) == 1
        return f"math.ceil({self._print(expr.args[0])})"

    def _print_Abs(self, expr):
        assert len(expr.args) == 1
        return f"abs({self._print(expr.args[0])})"
