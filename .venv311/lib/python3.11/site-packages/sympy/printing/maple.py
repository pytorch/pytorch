"""
Maple code printer

The MapleCodePrinter converts single SymPy expressions into single
Maple expressions, using the functions defined in the Maple objects where possible.


FIXME: This module is still under actively developed. Some functions may be not completed.
"""

from sympy.core import S
from sympy.core.numbers import Integer, IntegerConstant, equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE

import sympy

_known_func_same_name = (
    'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinh', 'cosh', 'tanh', 'sech',
    'csch', 'coth', 'exp', 'floor', 'factorial', 'bernoulli',  'euler',
    'fibonacci', 'gcd', 'lcm', 'conjugate', 'Ci', 'Chi', 'Ei', 'Li', 'Si', 'Shi',
    'erf', 'erfc', 'harmonic', 'LambertW',
    'sqrt', # For automatic rewrites
)

known_functions = {
    # SymPy -> Maple
    'Abs': 'abs',
    'log': 'ln',
    'asin': 'arcsin',
    'acos': 'arccos',
    'atan': 'arctan',
    'asec': 'arcsec',
    'acsc': 'arccsc',
    'acot': 'arccot',
    'asinh': 'arcsinh',
    'acosh': 'arccosh',
    'atanh': 'arctanh',
    'asech': 'arcsech',
    'acsch': 'arccsch',
    'acoth': 'arccoth',
    'ceiling': 'ceil',
    'Max' : 'max',
    'Min' : 'min',

    'factorial2': 'doublefactorial',
    'RisingFactorial': 'pochhammer',
    'besseli': 'BesselI',
    'besselj': 'BesselJ',
    'besselk': 'BesselK',
    'bessely': 'BesselY',
    'hankelh1': 'HankelH1',
    'hankelh2': 'HankelH2',
    'airyai': 'AiryAi',
    'airybi': 'AiryBi',
    'appellf1': 'AppellF1',
    'fresnelc': 'FresnelC',
    'fresnels': 'FresnelS',
    'lerchphi' : 'LerchPhi',
}

for _func in _known_func_same_name:
    known_functions[_func] = _func

number_symbols = {
    # SymPy -> Maple
    S.Pi: 'Pi',
    S.Exp1: 'exp(1)',
    S.Catalan: 'Catalan',
    S.EulerGamma: 'gamma',
    S.GoldenRatio: '(1/2 + (1/2)*sqrt(5))'
}

spec_relational_ops = {
    # SymPy -> Maple
    '==': '=',
    '!=': '<>'
}

not_supported_symbol = [
    S.ComplexInfinity
]

class MapleCodePrinter(CodePrinter):
    """
    Printer which converts a SymPy expression into a maple code.
    """
    printmethod = "_maple"
    language = "maple"

    _operators = {
        'and': 'and',
        'or': 'or',
        'not': 'not ',
    }

    _default_settings = dict(CodePrinter._default_settings, **{
        'inline': True,
        'allow_unknown_functions': True,
    })

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        super().__init__(settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "# {}".format(text)

    def _declare_number_const(self, name, value):
        return "{} := {};".format(name,
                                    value.evalf(self._settings['precision']))

    def _format_code(self, lines):
        return lines

    def _print_tuple(self, expr):
        return self._print(list(expr))

    def _print_Tuple(self, expr):
        return self._print(list(expr))

    def _print_Assignment(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return "{lhs} := {rhs}".format(lhs=lhs, rhs=rhs)

    def _print_Pow(self, expr, **kwargs):
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1/%s' % (self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        elif equal_valued(expr.exp, -0.5):
            return '1/sqrt(%s)' % self._print(expr.base)
        else:
            return '{base}^{exp}'.format(
                base=self.parenthesize(expr.base, PREC),
                exp=self.parenthesize(expr.exp, PREC))

    def _print_Piecewise(self, expr):
        if (expr.args[-1].cond is not True) and (expr.args[-1].cond != S.BooleanTrue):
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        _coup_list = [
            ("{c}, {e}".format(c=self._print(c),
                               e=self._print(e)) if c is not True and c is not S.BooleanTrue else "{e}".format(
                e=self._print(e)))
            for e, c in expr.args]
        _inbrace = ', '.join(_coup_list)
        return 'piecewise({_inbrace})'.format(_inbrace=_inbrace)

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return "{p}/{q}".format(p=str(p), q=str(q))

    def _print_Relational(self, expr):
        PREC=precedence(expr)
        lhs_code = self.parenthesize(expr.lhs, PREC)
        rhs_code = self.parenthesize(expr.rhs, PREC)
        op = expr.rel_op
        if op in spec_relational_ops:
            op = spec_relational_ops[op]
        return "{lhs} {rel_op} {rhs}".format(lhs=lhs_code, rel_op=op, rhs=rhs_code)

    def _print_NumberSymbol(self, expr):
        return number_symbols[expr]

    def _print_NegativeInfinity(self, expr):
        return '-infinity'

    def _print_Infinity(self, expr):
        return 'infinity'

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    def _print_bool(self, expr):
        return 'true' if expr else 'false'

    def _print_NaN(self, expr):
        return 'undefined'

    def _get_matrix(self, expr, sparse=False):
        if S.Zero in expr.shape:
            _strM = 'Matrix([], storage = {storage})'.format(
                storage='sparse' if sparse else 'rectangular')
        else:
            _strM = 'Matrix({list}, storage = {storage})'.format(
                list=self._print(expr.tolist()),
                storage='sparse' if sparse else 'rectangular')
        return _strM

    def _print_MatrixElement(self, expr):
        return "{parent}[{i_maple}, {j_maple}]".format(
            parent=self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True),
            i_maple=self._print(expr.i + 1),
            j_maple=self._print(expr.j + 1))

    def _print_MatrixBase(self, expr):
        return self._get_matrix(expr, sparse=False)

    def _print_SparseRepMatrix(self, expr):
        return self._get_matrix(expr, sparse=True)

    def _print_Identity(self, expr):
        if isinstance(expr.rows, (Integer, IntegerConstant)):
            return self._print(sympy.SparseMatrix(expr))
        else:
            return "Matrix({var_size}, shape = identity)".format(var_size=self._print(expr.rows))

    def _print_MatMul(self, expr):
        PREC=precedence(expr)
        _fact_list = list(expr.args)
        _const = None
        if not isinstance(_fact_list[0], (sympy.MatrixBase, sympy.MatrixExpr,
                                          sympy.MatrixSlice, sympy.MatrixSymbol)):
            _const, _fact_list = _fact_list[0], _fact_list[1:]

        if _const is None or _const == 1:
            return '.'.join(self.parenthesize(_m, PREC) for _m in _fact_list)
        else:
            return '{c}*{m}'.format(c=_const, m='.'.join(self.parenthesize(_m, PREC) for _m in _fact_list))

    def _print_MatPow(self, expr):
        # This function requires LinearAlgebra Function in Maple
        return 'MatrixPower({A}, {n})'.format(A=self._print(expr.base), n=self._print(expr.exp))

    def _print_HadamardProduct(self, expr):
        PREC = precedence(expr)
        _fact_list = list(expr.args)
        return '*'.join(self.parenthesize(_m, PREC) for _m in _fact_list)

    def _print_Derivative(self, expr):
        _f, (_var, _order) = expr.args

        if _order != 1:
            _second_arg = '{var}${order}'.format(var=self._print(_var),
                                                 order=self._print(_order))
        else:
            _second_arg = '{var}'.format(var=self._print(_var))
        return 'diff({func_expr}, {sec_arg})'.format(func_expr=self._print(_f), sec_arg=_second_arg)


def maple_code(expr, assign_to=None, **settings):
    r"""Converts ``expr`` to a string of Maple code.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned.  Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
        expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi  [default=16].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations.  Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)].  See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols.  If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text).  [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    inline: bool, optional
        If True, we try to create single-statement code instead of multiple
        statements.  [default=True].

    """
    return MapleCodePrinter(settings).doprint(expr, assign_to)


def print_maple_code(expr, **settings):
    """Prints the Maple representation of the given expression.

    See :func:`maple_code` for the meaning of the optional arguments.

    Examples
    ========

    >>> from sympy import print_maple_code, symbols
    >>> x, y = symbols('x y')
    >>> print_maple_code(x, assign_to=y)
    y := x
    """
    print(maple_code(expr, **settings))
