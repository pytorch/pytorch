"""
R code printer

The RCodePrinter converts single SymPy expressions into single R expressions,
using the functions defined in math.h where possible.



"""

from __future__ import annotations
from typing import Any

from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range

# dictionary mapping SymPy function to (argument_conditions, C_function).
# Used in RCodePrinter._print_Function(self)
known_functions = {
    #"Abs": [(lambda x: not x.is_integer, "fabs")],
    "Abs": "abs",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "exp": "exp",
    "log": "log",
    "erf": "erf",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "floor": "floor",
    "ceiling": "ceiling",
    "sign": "sign",
    "Max": "max",
    "Min": "min",
    "factorial": "factorial",
    "gamma": "gamma",
    "digamma": "digamma",
    "trigamma": "trigamma",
    "beta": "beta",
    "sqrt": "sqrt",  # To enable automatic rewrite
}

# These are the core reserved words in the R language. Taken from:
# https://cran.r-project.org/doc/manuals/r-release/R-lang.html#Reserved-words

reserved_words = ['if',
                  'else',
                  'repeat',
                  'while',
                  'function',
                  'for',
                  'in',
                  'next',
                  'break',
                  'TRUE',
                  'FALSE',
                  'NULL',
                  'Inf',
                  'NaN',
                  'NA',
                  'NA_integer_',
                  'NA_real_',
                  'NA_complex_',
                  'NA_character_',
                  'volatile']


class RCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of R code"""
    printmethod = "_rcode"
    language = "R"

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 15,
        'user_functions': {},
        'contract': True,
        'dereference': set(),
    })
    _operators = {
       'and': '&',
        'or': '|',
       'not': '!',
    }

    _relationals: dict[str, str] = {}

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.reserved_words = set(reserved_words)

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "// {}".format(text)

    def _declare_number_const(self, name, value):
        return "{} = {};".format(name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        """Returns a tuple (open_lines, close_lines) containing lists of codelines
        """
        open_lines = []
        close_lines = []
        loopstart = "for (%(var)s in %(start)s:%(end)s){"
        for i in indices:
            # R arrays start at 1 and end at dimension
            open_lines.append(loopstart % {
                'var': self._print(i.label),
                'start': self._print(i.lower+1),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            return '%s^%s' % (self.parenthesize(expr.base, PREC),
                                 self.parenthesize(expr.exp, PREC))


    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d.0/%d.0' % (p, q)

    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s[%s]" % (self._print(expr.base.label), ", ".join(inds))

    def _print_Exp1(self, expr):
        return "exp(1)"

    def _print_Pi(self, expr):
        return 'pi'

    def _print_Infinity(self, expr):
        return 'Inf'

    def _print_NegativeInfinity(self, expr):
        return '-Inf'

    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment

        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        #if isinstance(expr.rhs, Piecewise):
        #    from sympy.functions.elementary.piecewise import Piecewise
        #    # Here we modify Piecewise so each expression is now
        #    # an Assignment, and then continue on the print.
        #    expressions = []
        #    conditions = []
        #    for (e, c) in rhs.args:
        #        expressions.append(Assignment(lhs, e))
        #        conditions.append(c)
        #    temp = Piecewise(*zip(expressions, conditions))
        #    return self._print(temp)
        #elif isinstance(lhs, MatrixSymbol):
        if isinstance(lhs, MatrixSymbol):
            # Here we form an Assignment for each element in the array,
            # printing each one.
            lines = []
            for (i, j) in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return "\n".join(lines)
        elif self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    def _print_Piecewise(self, expr):
        # This method is called only for inline if constructs
        # Top level piecewise is handled in doprint()
        if expr.args[-1].cond == True:
            last_line = "%s" % self._print(expr.args[-1].expr)
        else:
            last_line = "ifelse(%s,%s,NA)" % (self._print(expr.args[-1].cond), self._print(expr.args[-1].expr))
        code=last_line
        for e, c in reversed(expr.args[:-1]):
            code= "ifelse(%s,%s," % (self._print(c), self._print(e))+code+")"
        return(code)

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_MatrixElement(self, expr):
        return "{}[{}]".format(self.parenthesize(expr.parent, PRECEDENCE["Atom"],
            strict=True), expr.j + expr.i*expr.parent.shape[1])

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        if expr in self._dereference:
            return '(*{})'.format(name)
        else:
            return name

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        op = expr.op
        rhs_code = self._print(expr.rhs)
        return "{} {} {};".format(lhs_code, op, rhs_code)

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return 'for({target} in seq(from={start}, to={stop}, by={step}){{\n{body}\n}}'.format(target=target, start=start,
                stop=stop-1, step=step, body=body)


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "   "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty


def rcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of r code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired R string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        rfunction_string)] or [(argument_test, rfunction_formater)]. See below
        for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import rcode, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> rcode((2*tau)**Rational(7, 2))
    '8*sqrt(2)*tau^(7.0/2.0)'
    >>> rcode(sin(x), assign_to="s")
    's = sin(x);'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> rcode(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    'f(fabs(x) + CEIL(x))'

    or if the R-function takes a subset of the original arguments:

    >>> rcode(2**x + 3**x, user_functions={'Pow': [
    ...   (lambda b, e: b == 2, lambda b, e: 'exp2(%s)' % e),
    ...   (lambda b, e: b != 2, 'pow')]})
    'exp2(x) + pow(3, x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(rcode(expr, assign_to=tau))
    tau = ifelse(x > 0,x + 1,x);

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> rcode(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(rcode(mat, A))
    A[0] = x^2;
    A[1] = ifelse(x > 0,x + 1,x);
    A[2] = sin(x);

    """

    return RCodePrinter(settings).doprint(expr, assign_to)


def print_rcode(expr, **settings):
    """Prints R representation of the given expression."""
    print(rcode(expr, **settings))
