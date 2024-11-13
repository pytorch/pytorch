import re
import sys

import sympy
from sympy.printing.printer import Printer


INDEX_TYPE = "int64_t"


# This printer contains rules that are supposed to be generic for both C/C++ and
# Python
class ExprPrinter(Printer):
    @staticmethod
    def paren(string: str) -> str:
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

        if (
            re.match(r"^[a-z0-9_.]+$", string, re.IGNORECASE)
            or re.match(r"^\([^)]*\)$", string, re.IGNORECASE)
            or string == ""
        ):
            return string
        # don't put extra parens for strings that are already wrapped in parens
        if all_in_parens(string):
            return string
        return f"({string})"

    def _print_Relational(self, expr: sympy.Expr) -> str:
        return f" {expr.rel_op} ".join(map(self.paren, map(self._print, expr.args)))

    def _print_Mul(self, expr: sympy.Expr) -> str:
        return "*".join(map(self.paren, map(self._print, expr.args)))

    def _print_Add(self, expr: sympy.Expr) -> str:
        return " + ".join(map(self.paren, map(self._print, expr.args)))

    # NB: this is OK to put here, because Mod is only defined for positive
    # numbers, and so across C/Python its behavior is consistent
    def _print_Mod(self, expr: sympy.Expr) -> str:
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    def _print_FloatTrueDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        return f"{self.paren(self._print(lhs))} / {self.paren(self._print(rhs))}"

    def _print_CleanDiv(self, expr: sympy.Expr) -> str:
        return self._print_FloorDiv(expr)

    def _print_Identity(self, expr: sympy.Expr) -> str:
        return self._print(expr.args[0])

    def _print_GreaterThan(self, expr: sympy.Expr) -> str:
        # GreaterThan:          >=
        # StrictlyGreaterThan:  >
        # Go figure...
        return " >= ".join(map(self.paren, map(self._print, expr.args)))

    # NB: The C implementation is injected into codegen at
    # torch/_inductor/codegen/wrapper.py
    def _print_align(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"align({self._print(expr.args[0])})"

    # This must be implemented because sympy will collect x * x into Pow(x, 2), without
    # any explicit intervention.  We print it just like x * x, notably, we
    # never generate sympy.Pow with floats.
    #
    # NB: this pow by natural, you should never have used builtin sympy.pow
    # for FloatPow, and a symbolic exponent should be PowByNatural.  These
    # means exp is guaranteed to be integer.
    def _print_Pow(self, expr: sympy.Expr) -> str:
        base, exp = expr.args
        base = self._print(base)
        assert exp == int(exp), exp
        exp = int(exp)
        assert exp >= 0
        if exp > 0:
            return "*".join([self.paren(base)] * exp)
        return "1"

    # Explicit NotImplemented functions are to prevent default sympy printing
    # behavior, which will just barf out ToFloat(...) to your IR.  The error
    # message is better here because it tells you which printer class it needs
    # to go in.

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_ToFloat not implemented for {type(self)}")

    def _print_Infinity(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_Infinity not implemented for {type(self)}")

    def _print_NegativeInfinity(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(
            f"_print_NegativeInfinity not implemented for {type(self)}"
        )

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_FloorDiv not implemented for {type(self)}")

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_PythonMod not implemented for {type(self)}")

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_IntTrueDiv not implemented for {type(self)}")

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(
            f"_print_PowByNatural not implemented for {type(self)}"
        )

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_FloatPow not implemented for {type(self)}")

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_TruncToInt not implemented for {type(self)}")

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(f"_print_RoundToInt not implemented for {type(self)}")

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(
            f"_print_RoundDecimal not implemented for {type(self)}"
        )

    # NB: Some float operations are INTENTIONALLY not implemented for
    # printers.  You can implement them as a quick unblock, but it is better
    # to ask yourself why we haven't done this computation in the Tensor
    # universe instead

    def _print_TruncToFloat(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(
            f"_print_TruncToFloat not implemented for {type(self)}"
        )


class PythonPrinter(ExprPrinter):
    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"float({self._print(expr.args[0])})"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} // {div})"
        return f"{x} % {mod}"

    def _print_Infinity(self, expr: sympy.Expr) -> str:
        return "math.inf"

    def _print_NegativeInfinity(self, expr: sympy.Expr) -> str:
        return "-math.inf"

    # WARNING: this is dangerous for Triton, which has C-style modulus
    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    # WARNING: this is dangerous for Triton, which has C-style modulus
    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} // {div})"

    # WARNING: this is dangerous for Triton, when lhs, rhs > 2**53, Python
    # does a special algorithm
    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        return f"{self.paren(self._print(lhs))} / {self.paren(self._print(rhs))}"

    def _helper_sqrt(self, expr: sympy.Expr) -> str:
        return f"math.sqrt({self._print(expr)})"

    def _print_OpaqueUnaryFn_sqrt(self, expr: sympy.Expr) -> str:
        return self._helper_sqrt(expr.args[0])

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        base, exp = expr.args
        return f"{self.paren(self._print(base))} ** {self.paren(self._print(exp))}"

    # TODO: Not sure this works with Triton, even when base/exp are integral
    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        base, exp = expr.args
        return f"{self.paren(self._print(base))} ** {self.paren(self._print(exp))}"

    def _print_floor(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.floor({self._print(expr.args[0])})"

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.floor({self._print(expr.args[0])})"

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        # This also could have been int(), they'll do the same thing for float
        return f"math.trunc({self._print(expr.args[0])})"

    def _print_ceiling(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.ceil({self._print(expr.args[0])})"

    def _print_CeilToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.ceil({self._print(expr.args[0])})"

    def _print_Abs(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"abs({self._print(expr.args[0])})"

    # NB: It's expected that we've made explicit any promotion in the sympy
    # expression, so it doesn't matter that Python max/min doesn't perform
    # promotion
    def _print_Max(self, expr: sympy.Expr) -> str:
        assert len(expr.args) >= 2
        return f"max({', '.join(map(self._print, expr.args))})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        assert len(expr.args) >= 2
        return f"min({', '.join(map(self._print, expr.args))})"

    def _print_OpaqueUnaryFn_cos(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.cos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.cosh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.acos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.sin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.sinh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.asin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tan(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.tan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tanh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.tanh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_atan(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"math.atan({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"round({self._print(expr.args[0])})"

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 2
        number, ndigits = expr.args
        assert isinstance(ndigits, sympy.Integer)
        return f"round({self._print(number)}, {ndigits})"


class CppPrinter(ExprPrinter):
    def _print_Integer(self, expr: sympy.Expr) -> str:
        return (
            f"{int(expr)}LL" if sys.platform in ["darwin", "win32"] else f"{int(expr)}L"
        )

    def _print_Where(self, expr: sympy.Expr) -> str:
        c = self.paren(self.doprint(expr.args[0]))
        p = self.paren(self.doprint(expr.args[1]))
        q = self.paren(self.doprint(expr.args[2]))
        return f"{c} ? {p} : {q}"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        if div != 1:
            div = self.paren(self.doprint(div))
            if expr.is_integer:
                x = f"c10::div_floor_integer(static_cast<int64_t>({x}), static_cast<int64_t>({div}))"
            else:
                x = f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"
        mod = self.paren(self.doprint(mod))
        return f"static_cast<{INDEX_TYPE}>({x}) % static_cast<{INDEX_TYPE}>({mod})"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        if expr.is_integer:
            return f"c10::div_floor_integer(static_cast<int64_t>({x}), static_cast<int64_t>({div}))"
        return f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"

    def _print_floor(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        r = f"std::floor({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        r = f"std::floor({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        r = f"std::trunc({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})"

    def _print_TruncToFloat(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::trunc({self._print(expr.args[0])})"

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"static_cast<double>({self._print(expr.args[0])})"

    # TODO: This is wrong if one of the inputs is negative.  This is hard to
    # tickle though, as the inputs are typically positive (and if we can prove
    # they are positive, we will have used Mod instead, for which this codegen
    # is right).
    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    def _print_CMod(self, expr: sympy.Expr) -> str:
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        # TODO: This is only accurate up to 2**53
        return f"static_cast<double>({self._print(lhs)}) / static_cast<double>({self._print(rhs)})"

    # TODO: PowByNatural: we need to implement our own int-int pow.  Do NOT
    # use std::pow, that operates on floats
    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        raise NotImplementedError(
            f"_print_PowByNatural not implemented for {type(self)}"
        )

    def _print_FloatTrueDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        return f"{self.paren(self._print(lhs))} / {self.paren(self._print(rhs))}"

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        base, exp = expr.args
        return f"std::pow({self._print(base)}, {self._print(exp)})"

    def _print_Pow(self, expr: sympy.Expr) -> str:
        # Uses float constants to perform FP div
        base, exp = expr.args
        base = self._print(base)

        if exp == 0.5 or exp == -0.5:
            return f"std::sqrt({base})" if exp == 0.5 else f"1.0/std::sqrt({base})"
        if exp.is_integer:
            exp = int(exp)
            if exp > 0:
                r = "*".join([self.paren(base)] * exp)
            elif exp < 0:
                r = "1.0/" + self.paren("*".join([self.paren(base)] * abs(exp)))
            else:  # exp == 0
                r = "1.0"

            return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r
        else:
            # TODO: float vs double
            return f"std::pow({base}, {float(exp)})"

    def _print_Rational(self, expr: sympy.Expr) -> str:
        # Uses float constants to perform FP div
        if expr.q == 1:
            r = f"{expr.p}"
        else:
            r = f"{expr.p}.0/{expr.q}.0"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_ceiling(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        r = f"std::ceil({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_CeilToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        r = f"std::ceil({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_Min(self, expr: sympy.Expr) -> str:
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f"std::min(static_cast<{INDEX_TYPE}>({args[0]}), static_cast<{INDEX_TYPE}>({args[1]}))"
        else:
            # Initializer list overload
            il = "{" + ", ".join(args) + "}"
            return f"std::min({il})"

    def _print_Max(self, expr: sympy.Expr) -> str:
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f"std::max(static_cast<{INDEX_TYPE}>({args[0]}), static_cast<{INDEX_TYPE}>({args[1]}))"
        else:
            # Initializer list overload
            il = "{" + ", ".join(args) + "}"
            return f"std::max({il})"

    def _print_Abs(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::abs({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cos(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::cos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::cosh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::acos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::sin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::sinh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::asin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tan(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::tan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tanh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::tanh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_atan(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"std::atan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sqrt(self, expr: sympy.Expr) -> str:
        return f"std::sqrt({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        # TODO: dispatch to llrint depending on index type
        return f"std::lrint({self._print(expr.args[0])})"

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 2
        number, ndigits = expr.args
        if number.is_integer:
            # ndigits < 0 should have been filtered by the sympy function
            assert ndigits < 0
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )
        return f"static_cast<double>(std::nearbyint(1e{ndigits} * {self.paren(self._print(number))}) * 1e{-ndigits})"

    def _print_BooleanTrue(self, expr: sympy.Expr) -> str:
        return "true"

    def _print_BooleanFalse(self, expr: sympy.Expr) -> str:
        return "false"
