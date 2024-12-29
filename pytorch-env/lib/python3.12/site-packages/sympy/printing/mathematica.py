"""
Mathematica code printer
"""

from __future__ import annotations
from typing import Any

from sympy.core import Basic, Expr, Float
from sympy.core.sorting import default_sort_key

from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence

# Used in MCodePrinter._print_Function(self)
known_functions = {
    "exp": [(lambda x: True, "Exp")],
    "log": [(lambda x: True, "Log")],
    "sin": [(lambda x: True, "Sin")],
    "cos": [(lambda x: True, "Cos")],
    "tan": [(lambda x: True, "Tan")],
    "cot": [(lambda x: True, "Cot")],
    "sec": [(lambda x: True, "Sec")],
    "csc": [(lambda x: True, "Csc")],
    "asin": [(lambda x: True, "ArcSin")],
    "acos": [(lambda x: True, "ArcCos")],
    "atan": [(lambda x: True, "ArcTan")],
    "acot": [(lambda x: True, "ArcCot")],
    "asec": [(lambda x: True, "ArcSec")],
    "acsc": [(lambda x: True, "ArcCsc")],
    "atan2": [(lambda *x: True, "ArcTan")],
    "sinh": [(lambda x: True, "Sinh")],
    "cosh": [(lambda x: True, "Cosh")],
    "tanh": [(lambda x: True, "Tanh")],
    "coth": [(lambda x: True, "Coth")],
    "sech": [(lambda x: True, "Sech")],
    "csch": [(lambda x: True, "Csch")],
    "asinh": [(lambda x: True, "ArcSinh")],
    "acosh": [(lambda x: True, "ArcCosh")],
    "atanh": [(lambda x: True, "ArcTanh")],
    "acoth": [(lambda x: True, "ArcCoth")],
    "asech": [(lambda x: True, "ArcSech")],
    "acsch": [(lambda x: True, "ArcCsch")],
    "sinc": [(lambda x: True, "Sinc")],
    "conjugate": [(lambda x: True, "Conjugate")],
    "Max": [(lambda *x: True, "Max")],
    "Min": [(lambda *x: True, "Min")],
    "erf": [(lambda x: True, "Erf")],
    "erf2": [(lambda *x: True, "Erf")],
    "erfc": [(lambda x: True, "Erfc")],
    "erfi": [(lambda x: True, "Erfi")],
    "erfinv": [(lambda x: True, "InverseErf")],
    "erfcinv": [(lambda x: True, "InverseErfc")],
    "erf2inv": [(lambda *x: True, "InverseErf")],
    "expint": [(lambda *x: True, "ExpIntegralE")],
    "Ei": [(lambda x: True, "ExpIntegralEi")],
    "fresnelc": [(lambda x: True, "FresnelC")],
    "fresnels": [(lambda x: True, "FresnelS")],
    "gamma": [(lambda x: True, "Gamma")],
    "uppergamma": [(lambda *x: True, "Gamma")],
    "polygamma": [(lambda *x: True, "PolyGamma")],
    "loggamma": [(lambda x: True, "LogGamma")],
    "beta": [(lambda *x: True, "Beta")],
    "Ci": [(lambda x: True, "CosIntegral")],
    "Si": [(lambda x: True, "SinIntegral")],
    "Chi": [(lambda x: True, "CoshIntegral")],
    "Shi": [(lambda x: True, "SinhIntegral")],
    "li": [(lambda x: True, "LogIntegral")],
    "factorial": [(lambda x: True, "Factorial")],
    "factorial2": [(lambda x: True, "Factorial2")],
    "subfactorial": [(lambda x: True, "Subfactorial")],
    "catalan": [(lambda x: True, "CatalanNumber")],
    "harmonic": [(lambda *x: True, "HarmonicNumber")],
    "lucas": [(lambda x: True, "LucasL")],
    "RisingFactorial": [(lambda *x: True, "Pochhammer")],
    "FallingFactorial": [(lambda *x: True, "FactorialPower")],
    "laguerre": [(lambda *x: True, "LaguerreL")],
    "assoc_laguerre": [(lambda *x: True, "LaguerreL")],
    "hermite": [(lambda *x: True, "HermiteH")],
    "jacobi": [(lambda *x: True, "JacobiP")],
    "gegenbauer": [(lambda *x: True, "GegenbauerC")],
    "chebyshevt": [(lambda *x: True, "ChebyshevT")],
    "chebyshevu": [(lambda *x: True, "ChebyshevU")],
    "legendre": [(lambda *x: True, "LegendreP")],
    "assoc_legendre": [(lambda *x: True, "LegendreP")],
    "mathieuc": [(lambda *x: True, "MathieuC")],
    "mathieus": [(lambda *x: True, "MathieuS")],
    "mathieucprime": [(lambda *x: True, "MathieuCPrime")],
    "mathieusprime": [(lambda *x: True, "MathieuSPrime")],
    "stieltjes": [(lambda x: True, "StieltjesGamma")],
    "elliptic_e": [(lambda *x: True, "EllipticE")],
    "elliptic_f": [(lambda *x: True, "EllipticE")],
    "elliptic_k": [(lambda x: True, "EllipticK")],
    "elliptic_pi": [(lambda *x: True, "EllipticPi")],
    "zeta": [(lambda *x: True, "Zeta")],
    "dirichlet_eta": [(lambda x: True, "DirichletEta")],
    "riemann_xi": [(lambda x: True, "RiemannXi")],
    "besseli": [(lambda *x: True, "BesselI")],
    "besselj": [(lambda *x: True, "BesselJ")],
    "besselk": [(lambda *x: True, "BesselK")],
    "bessely": [(lambda *x: True, "BesselY")],
    "hankel1": [(lambda *x: True, "HankelH1")],
    "hankel2": [(lambda *x: True, "HankelH2")],
    "airyai": [(lambda x: True, "AiryAi")],
    "airybi": [(lambda x: True, "AiryBi")],
    "airyaiprime": [(lambda x: True, "AiryAiPrime")],
    "airybiprime": [(lambda x: True, "AiryBiPrime")],
    "polylog": [(lambda *x: True, "PolyLog")],
    "lerchphi": [(lambda *x: True, "LerchPhi")],
    "gcd": [(lambda *x: True, "GCD")],
    "lcm": [(lambda *x: True, "LCM")],
    "jn": [(lambda *x: True, "SphericalBesselJ")],
    "yn": [(lambda *x: True, "SphericalBesselY")],
    "hyper": [(lambda *x: True, "HypergeometricPFQ")],
    "meijerg": [(lambda *x: True, "MeijerG")],
    "appellf1": [(lambda *x: True, "AppellF1")],
    "DiracDelta": [(lambda x: True, "DiracDelta")],
    "Heaviside": [(lambda x: True, "HeavisideTheta")],
    "KroneckerDelta": [(lambda *x: True, "KroneckerDelta")],
    "sqrt": [(lambda x: True, "Sqrt")],  # For automatic rewrites
}


class MCodePrinter(CodePrinter):
    """A printer to convert Python expressions to
    strings of the Wolfram's Mathematica code
    """
    printmethod = "_mcode"
    language = "Wolfram Language"

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 15,
        'user_functions': {},
    })

    _number_symbols: set[tuple[Expr, Float]] = set()
    _not_supported: set[Basic] = set()

    def __init__(self, settings={}):
        """Register function mappings supplied by user"""
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {}).copy()
        for k, v in userfuncs.items():
            if not isinstance(v, list):
                userfuncs[k] = [(lambda *x: True, v)]
        self.known_functions.update(userfuncs)

    def _format_code(self, lines):
        return lines

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))

    def _print_Mul(self, expr):
        PREC = precedence(expr)
        c, nc = expr.args_cnc()
        res = super()._print_Mul(expr.func(*c))
        if nc:
            res += '*'
            res += '**'.join(self.parenthesize(a, PREC) for a in nc)
        return res

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    # Primitive numbers
    def _print_Zero(self, expr):
        return '0'

    def _print_One(self, expr):
        return '1'

    def _print_NegativeOne(self, expr):
        return '-1'

    def _print_Half(self, expr):
        return '1/2'

    def _print_ImaginaryUnit(self, expr):
        return 'I'


    # Infinity and invalid numbers
    def _print_Infinity(self, expr):
        return 'Infinity'

    def _print_NegativeInfinity(self, expr):
        return '-Infinity'

    def _print_ComplexInfinity(self, expr):
        return 'ComplexInfinity'

    def _print_NaN(self, expr):
        return 'Indeterminate'


    # Mathematical constants
    def _print_Exp1(self, expr):
        return 'E'

    def _print_Pi(self, expr):
        return 'Pi'

    def _print_GoldenRatio(self, expr):
        return 'GoldenRatio'

    def _print_TribonacciConstant(self, expr):
        expanded = expr.expand(func=True)
        PREC = precedence(expr)
        return self.parenthesize(expanded, PREC)

    def _print_EulerGamma(self, expr):
        return 'EulerGamma'

    def _print_Catalan(self, expr):
        return 'Catalan'


    def _print_list(self, expr):
        return '{' + ', '.join(self.doprint(a) for a in expr) + '}'
    _print_tuple = _print_list
    _print_Tuple = _print_list

    def _print_ImmutableDenseMatrix(self, expr):
        return self.doprint(expr.tolist())

    def _print_ImmutableSparseMatrix(self, expr):

        def print_rule(pos, val):
            return '{} -> {}'.format(
            self.doprint((pos[0]+1, pos[1]+1)), self.doprint(val))

        def print_data():
            items = sorted(expr.todok().items(), key=default_sort_key)
            return '{' + \
                ', '.join(print_rule(k, v) for k, v in items) + \
                '}'

        def print_dims():
            return self.doprint(expr.shape)

        return 'SparseArray[{}, {}]'.format(print_data(), print_dims())

    def _print_ImmutableDenseNDimArray(self, expr):
        return self.doprint(expr.tolist())

    def _print_ImmutableSparseNDimArray(self, expr):
        def print_string_list(string_list):
            return '{' + ', '.join(a for a in string_list) + '}'

        def to_mathematica_index(*args):
            """Helper function to change Python style indexing to
            Pathematica indexing.

            Python indexing (0, 1 ... n-1)
            -> Mathematica indexing (1, 2 ... n)
            """
            return tuple(i + 1 for i in args)

        def print_rule(pos, val):
            """Helper function to print a rule of Mathematica"""
            return '{} -> {}'.format(self.doprint(pos), self.doprint(val))

        def print_data():
            """Helper function to print data part of Mathematica
            sparse array.

            It uses the fourth notation ``SparseArray[data,{d1,d2,...}]``
            from
            https://reference.wolfram.com/language/ref/SparseArray.html

            ``data`` must be formatted with rule.
            """
            return print_string_list(
                [print_rule(
                    to_mathematica_index(*(expr._get_tuple_index(key))),
                    value)
                for key, value in sorted(expr._sparse_array.items())]
            )

        def print_dims():
            """Helper function to print dimensions part of Mathematica
            sparse array.

            It uses the fourth notation ``SparseArray[data,{d1,d2,...}]``
            from
            https://reference.wolfram.com/language/ref/SparseArray.html
            """
            return self.doprint(expr.shape)

        return 'SparseArray[{}, {}]'.format(print_data(), print_dims())

    def _print_Function(self, expr):
        if expr.func.__name__ in self.known_functions:
            cond_mfunc = self.known_functions[expr.func.__name__]
            for cond, mfunc in cond_mfunc:
                if cond(*expr.args):
                    return "%s[%s]" % (mfunc, self.stringify(expr.args, ", "))
        elif expr.func.__name__ in self._rewriteable_functions:
            # Simple rewrite to supported function possible
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all(self._can_print(f) for f in required_fs):
                return self._print(expr.rewrite(target_f))
        return expr.func.__name__ + "[%s]" % self.stringify(expr.args, ", ")

    _print_MinMaxBase = _print_Function

    def _print_LambertW(self, expr):
        if len(expr.args) == 1:
            return "ProductLog[{}]".format(self._print(expr.args[0]))
        return "ProductLog[{}, {}]".format(
            self._print(expr.args[1]), self._print(expr.args[0]))

    def _print_Integral(self, expr):
        if len(expr.variables) == 1 and not expr.limits[0][1:]:
            args = [expr.args[0], expr.variables[0]]
        else:
            args = expr.args
        return "Hold[Integrate[" + ', '.join(self.doprint(a) for a in args) + "]]"

    def _print_Sum(self, expr):
        return "Hold[Sum[" + ', '.join(self.doprint(a) for a in expr.args) + "]]"

    def _print_Derivative(self, expr):
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        return "Hold[D[" + ', '.join(self.doprint(a) for a in [dexpr] + dvars) + "]]"


    def _get_comment(self, text):
        return "(* {} *)".format(text)


def mathematica_code(expr, **settings):
    r"""Converts an expr to a string of the Wolfram Mathematica code

    Examples
    ========

    >>> from sympy import mathematica_code as mcode, symbols, sin
    >>> x = symbols('x')
    >>> mcode(sin(x).series(x).removeO())
    '(1/120)*x^5 - 1/6*x^3 + x'
    """
    return MCodePrinter(settings).doprint(expr)
