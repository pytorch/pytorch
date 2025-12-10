"""
Octave (and Matlab) code printer

The `OctaveCodePrinter` converts SymPy expressions into Octave expressions.
It uses a subset of the Octave language for Matlab compatibility.

A complete code generator, which uses `octave_code` extensively, can be found
in `sympy.utilities.codegen`.  The `codegen` module can be used to generate
complete source code files.

"""

from __future__ import annotations
from typing import Any

from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search

# List of known functions.  First, those that have the same name in
# SymPy and Octave.   This is almost certainly incomplete!
known_fcns_src1 = ["sin", "cos", "tan", "cot", "sec", "csc",
                   "asin", "acos", "acot", "atan", "atan2", "asec", "acsc",
                   "sinh", "cosh", "tanh", "coth", "csch", "sech",
                   "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
                   "erfc", "erfi", "erf", "erfinv", "erfcinv",
                   "besseli", "besselj", "besselk", "bessely",
                   "bernoulli", "beta", "euler", "exp", "factorial", "floor",
                   "fresnelc", "fresnels", "gamma", "harmonic", "log",
                   "polylog", "sign", "zeta", "legendre"]

# These functions have different names ("SymPy": "Octave"), more
# generally a mapping to (argument_conditions, octave_function).
known_fcns_src2 = {
    "Abs": "abs",
    "arg": "angle",  # arg/angle ok in Octave but only angle in Matlab
    "binomial": "bincoeff",
    "ceiling": "ceil",
    "chebyshevu": "chebyshevU",
    "chebyshevt": "chebyshevT",
    "Chi": "coshint",
    "Ci": "cosint",
    "conjugate": "conj",
    "DiracDelta": "dirac",
    "Heaviside": "heaviside",
    "im": "imag",
    "laguerre": "laguerreL",
    "LambertW": "lambertw",
    "li": "logint",
    "loggamma": "gammaln",
    "Max": "max",
    "Min": "min",
    "Mod": "mod",
    "polygamma": "psi",
    "re": "real",
    "RisingFactorial": "pochhammer",
    "Shi": "sinhint",
    "Si": "sinint",
}


class OctaveCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Octave/Matlab code.
    """
    printmethod = "_octave"
    language = "Octave"

    _operators = {
        'and': '&',
        'or': '|',
        'not': '~',
    }

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
        'inline': True,
    })
    # Note: contract is for expressing tensors as loops (if True), or just
    # assignment (if False).  FIXME: this should be looked a more carefully
    # for Octave.


    def __init__(self, settings={}):
        super().__init__(settings)
        self.known_functions = dict(zip(known_fcns_src1, known_fcns_src1))
        self.known_functions.update(dict(known_fcns_src2))
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)


    def _rate_index_position(self, p):
        return p*5


    def _get_statement(self, codestring):
        return "%s;" % codestring


    def _get_comment(self, text):
        return "% {}".format(text)


    def _declare_number_const(self, name, value):
        return "{} = {};".format(name, value)


    def _format_code(self, lines):
        return self.indent_code(lines)


    def _traverse_matrix_indices(self, mat):
        # Octave uses Fortran order (column-major)
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))


    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # Octave arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("for %s = %s:%s" % (var, start, stop))
            close_lines.append("end")
        return open_lines, close_lines


    def _print_Mul(self, expr):
        # print complex numbers nicely in Octave
        if (expr.is_number and expr.is_imaginary and
                (S.ImaginaryUnit*expr).is_Integer):
            return "%si" % self._print(-S.ImaginaryUnit*expr)

        # cribbed from str.py
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if (item.is_commutative and item.is_Pow and item.exp.is_Rational
                    and item.exp.is_negative):
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # from here it differs from str.py to deal with "*" and ".*"
        def multjoin(a, a_str):
            # here we probably are assuming the constants will come first
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i-1].is_number else '.*'
                r = r + mulsym + a_str[i]
            return r

        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return sign + multjoin(a, a_str) + divsym + b_str[0]
        else:
            divsym = '/' if all(bi.is_number for bi in b) else './'
            return (sign + multjoin(a, a_str) +
                    divsym + "(%s)" % multjoin(b, b_str))

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Pow(self, expr):
        powsymbol = '^' if all(x.is_number for x in expr.args) else '.^'

        PREC = precedence(expr)

        if equal_valued(expr.exp, 0.5):
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if equal_valued(expr.exp, -0.5):
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "sqrt(%s)" % self._print(expr.base)
            if equal_valued(expr.exp, -1):
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "%s" % self.parenthesize(expr.base, PREC)

        return '%s%s%s' % (self.parenthesize(expr.base, PREC), powsymbol,
                           self.parenthesize(expr.exp, PREC))


    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))

    def _print_MatrixSolve(self, expr):
        PREC = precedence(expr)
        return "%s \\ %s" % (self.parenthesize(expr.matrix, PREC),
                             self.parenthesize(expr.vector, PREC))

    def _print_Pi(self, expr):
        return 'pi'


    def _print_ImaginaryUnit(self, expr):
        return "1i"


    def _print_Exp1(self, expr):
        return "exp(1)"


    def _print_GoldenRatio(self, expr):
        # FIXME: how to do better, e.g., for octave_code(2*GoldenRatio)?
        #return self._print((1+sqrt(S(5)))/2)
        return "(1+sqrt(5))/2"


    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.tensor.indexed import IndexedBase
        # Copied from codeprinter, but remove special MatrixSymbol treatment
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        if not self._settings["inline"] and isinstance(expr.rhs, Piecewise):
            # Here we modify Piecewise so each expression is now
            # an Assignment, and then continue on the print.
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        if self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))


    def _print_Infinity(self, expr):
        return 'inf'


    def _print_NegativeInfinity(self, expr):
        return '-inf'


    def _print_NaN(self, expr):
        return 'NaN'


    def _print_list(self, expr):
        return '{' + ', '.join(self._print(a) for a in expr) + '}'
    _print_tuple = _print_list
    _print_Tuple = _print_list
    _print_List = _print_list


    def _print_BooleanTrue(self, expr):
        return "true"


    def _print_BooleanFalse(self, expr):
        return "false"


    def _print_bool(self, expr):
        return str(expr).lower()


    # Could generate quadrature code for definite Integrals?
    #_print_Integral = _print_not_supported


    def _print_MatrixBase(self, A):
        # Handle zero dimensions:
        if (A.rows, A.cols) == (0, 0):
            return '[]'
        elif S.Zero in A.shape:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        elif (A.rows, A.cols) == (1, 1):
            # Octave does not distinguish between scalars and 1x1 matrices
            return self._print(A[0, 0])
        return "[%s]" % "; ".join(" ".join([self._print(a) for a in A[r, :]])
                                  for r in range(A.rows))


    def _print_SparseRepMatrix(self, A):
        from sympy.matrices import Matrix
        L = A.col_list()
        # make row vectors of the indices and entries
        I = Matrix([[k[0] + 1 for k in L]])
        J = Matrix([[k[1] + 1 for k in L]])
        AIJ = Matrix([[k[2] for k in L]])
        return "sparse(%s, %s, %s, %s, %s)" % (self._print(I), self._print(J),
                                            self._print(AIJ), A.rows, A.cols)


    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '(%s, %s)' % (expr.i + 1, expr.j + 1)


    def _print_MatrixSlice(self, expr):
        def strslice(x, lim):
            l = x[0] + 1
            h = x[1]
            step = x[2]
            lstr = self._print(l)
            hstr = 'end' if h == lim else self._print(h)
            if step == 1:
                if l == 1 and h == lim:
                    return ':'
                if l == h:
                    return lstr
                else:
                    return lstr + ':' + hstr
            else:
                return ':'.join((lstr, self._print(step), hstr))
        return (self._print(expr.parent) + '(' +
                strslice(expr.rowslice, expr.parent.shape[0]) + ', ' +
                strslice(expr.colslice, expr.parent.shape[1]) + ')')


    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))


    def _print_KroneckerDelta(self, expr):
        prec = PRECEDENCE["Pow"]
        return "double(%s == %s)" % tuple(self.parenthesize(x, prec)
                                          for x in expr.args)

    def _print_HadamardProduct(self, expr):
        return '.*'.join([self.parenthesize(arg, precedence(expr))
                          for arg in expr.args])

    def _print_HadamardPower(self, expr):
        PREC = precedence(expr)
        return '.**'.join([
            self.parenthesize(expr.base, PREC),
            self.parenthesize(expr.exp, PREC)
            ])

    def _print_Identity(self, expr):
        shape = expr.shape
        if len(shape) == 2 and shape[0] == shape[1]:
            shape = [shape[0]]
        s = ", ".join(self._print(n) for n in shape)
        return "eye(" + s + ")"

    def _print_lowergamma(self, expr):
        # Octave implements regularized incomplete gamma function
        return "(gammainc({1}, {0}).*gamma({0}))".format(
            self._print(expr.args[0]), self._print(expr.args[1]))


    def _print_uppergamma(self, expr):
        return "(gammainc({1}, {0}, 'upper').*gamma({0}))".format(
            self._print(expr.args[0]), self._print(expr.args[1]))


    def _print_sinc(self, expr):
        #Note: Divide by pi because Octave implements normalized sinc function.
        return "sinc(%s)" % self._print(expr.args[0]/S.Pi)


    def _print_hankel1(self, expr):
        return "besselh(%s, 1, %s)" % (self._print(expr.order),
                                       self._print(expr.argument))


    def _print_hankel2(self, expr):
        return "besselh(%s, 2, %s)" % (self._print(expr.order),
                                       self._print(expr.argument))


    # Note: as of 2015, Octave doesn't have spherical Bessel functions
    def _print_jn(self, expr):
        from sympy.functions import sqrt, besselj
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*besselj(expr.order + S.Half, x)
        return self._print(expr2)


    def _print_yn(self, expr):
        from sympy.functions import sqrt, bessely
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*bessely(expr.order + S.Half, x)
        return self._print(expr2)


    def _print_airyai(self, expr):
        return "airy(0, %s)" % self._print(expr.args[0])


    def _print_airyaiprime(self, expr):
        return "airy(1, %s)" % self._print(expr.args[0])


    def _print_airybi(self, expr):
        return "airy(2, %s)" % self._print(expr.args[0])


    def _print_airybiprime(self, expr):
        return "airy(3, %s)" % self._print(expr.args[0])


    def _print_expint(self, expr):
        mu, x = expr.args
        if mu != 1:
            return self._print_not_supported(expr)
        return "expint(%s)" % self._print(x)


    def _one_or_two_reversed_args(self, expr):
        assert len(expr.args) <= 2
        return '{name}({args})'.format(
            name=self.known_functions[expr.__class__.__name__],
            args=", ".join([self._print(x) for x in reversed(expr.args)])
        )


    _print_DiracDelta = _print_LambertW = _one_or_two_reversed_args


    def _nested_binary_math_func(self, expr):
        return '{name}({arg1}, {arg2})'.format(
            name=self.known_functions[expr.__class__.__name__],
            arg1=self._print(expr.args[0]),
            arg2=self._print(expr.func(*expr.args[1:]))
            )

    _print_Max = _print_Min = _nested_binary_math_func


    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if self._settings["inline"]:
            # Express each (cond, expr) pair in a nested Horner form:
            #   (condition) .* (expr) + (not cond) .* (<others>)
            # Expressions that result in multiple statements won't work here.
            ecpairs = ["({0}).*({1}) + (~({0})).*(".format
                       (self._print(c), self._print(e))
                       for e, c in expr.args[:-1]]
            elast = "%s" % self._print(expr.args[-1].expr)
            pw = " ...\n".join(ecpairs) + elast + ")"*len(ecpairs)
            # Note: current need these outer brackets for 2*pw.  Would be
            # nicer to teach parenthesize() to do this for us when needed!
            return "(" + pw + ")"
        else:
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s)" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("elseif (%s)" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                if i == len(expr.args) - 1:
                    lines.append("end")
            return "\n".join(lines)


    def _print_zeta(self, expr):
        if len(expr.args) == 1:
            return "zeta(%s)" % self._print(expr.args[0])
        else:
            # Matlab two argument zeta is not equivalent to SymPy's
            return self._print_not_supported(expr)


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        # code mostly copied from ccode
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "  "
        inc_regex = ('^function ', '^if ', '^elseif ', '^else$', '^for ')
        dec_regex = ('^end$', '^elseif ', '^else$')

        # pre-strip left-space from the code
        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(search(re, line) for re in inc_regex))
                     for line in code ]
        decrease = [ int(any(search(re, line) for re in dec_regex))
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


def octave_code(expr, assign_to=None, **settings):
    r"""Converts `expr` to a string of Octave (or Matlab) code.

    The string uses a subset of the Octave language for Matlab compatibility.

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

    Examples
    ========

    >>> from sympy import octave_code, symbols, sin, pi
    >>> x = symbols('x')
    >>> octave_code(sin(x).series(x).removeO())
    'x.^5/120 - x.^3/6 + x'

    >>> from sympy import Rational, ceiling
    >>> x, y, tau = symbols("x, y, tau")
    >>> octave_code((2*tau)**Rational(7, 2))
    '8*sqrt(2)*tau.^(7/2)'

    Note that element-wise (Hadamard) operations are used by default between
    symbols.  This is because its very common in Octave to write "vectorized"
    code.  It is harmless if the values are scalars.

    >>> octave_code(sin(pi*x*y), assign_to="s")
    's = sin(pi*x.*y);'

    If you need a matrix product "*" or matrix power "^", you can specify the
    symbol as a ``MatrixSymbol``.

    >>> from sympy import Symbol, MatrixSymbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> octave_code(3*pi*A**3)
    '(3*pi)*A^3'

    This class uses several rules to decide which symbol to use a product.
    Pure numbers use "*", Symbols use ".*" and MatrixSymbols use "*".
    A HadamardProduct can be used to specify componentwise multiplication ".*"
    of two MatrixSymbols.  There is currently there is no easy way to specify
    scalar symbols, so sometimes the code might have some minor cosmetic
    issues.  For example, suppose x and y are scalars and A is a Matrix, then
    while a human programmer might write "(x^2*y)*A^3", we generate:

    >>> octave_code(x**2*y*A**3)
    '(x.^2.*y)*A^3'

    Matrices are supported using Octave inline notation.  When using
    ``assign_to`` with matrices, the name can be specified either as a string
    or as a ``MatrixSymbol``.  The dimensions must align in the latter case.

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([[x**2, sin(x), ceiling(x)]])
    >>> octave_code(mat, assign_to='A')
    'A = [x.^2 sin(x) ceil(x)];'

    ``Piecewise`` expressions are implemented with logical masking by default.
    Alternatively, you can pass "inline=False" to use if-else conditionals.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> octave_code(pw, assign_to=tau)
    'tau = ((x > 0).*(x + 1) + (~(x > 0)).*(x));'

    Note that any expression that can be generated normally can also exist
    inside a Matrix:

    >>> mat = Matrix([[x**2, pw, sin(x)]])
    >>> octave_code(mat, assign_to='A')
    'A = [x.^2 ((x > 0).*(x + 1) + (~(x > 0)).*(x)) sin(x)];'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg.  Alternatively, the
    dictionary value can be a list of tuples i.e., [(argument_test,
    cfunction_string)].  This can be used to call a custom Octave function.

    >>> from sympy import Function
    >>> f = Function('f')
    >>> g = Function('g')
    >>> custom_functions = {
    ...   "f": "existing_octave_fcn",
    ...   "g": [(lambda x: x.is_Matrix, "my_mat_fcn"),
    ...         (lambda x: not x.is_Matrix, "my_fcn")]
    ... }
    >>> mat = Matrix([[1, x]])
    >>> octave_code(f(x) + g(x) + g(mat), user_functions=custom_functions)
    'existing_octave_fcn(x) + my_fcn(x) + my_mat_fcn([1 x])'

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
    >>> e = Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> octave_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy(i) = (y(i + 1) - y(i))./(t(i + 1) - t(i));'
    """
    return OctaveCodePrinter(settings).doprint(expr, assign_to)


def print_octave_code(expr, **settings):
    """Prints the Octave (or Matlab) representation of the given expression.

    See `octave_code` for the meaning of the optional arguments.
    """
    print(octave_code(expr, **settings))
