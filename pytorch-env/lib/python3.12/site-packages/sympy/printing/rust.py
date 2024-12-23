"""
Rust code printer

The `RustCodePrinter` converts SymPy expressions into Rust expressions.

A complete code generator, which uses `rust_code` extensively, can be found
in `sympy.utilities.codegen`. The `codegen` module can be used to generate
complete source code files.

"""

# Possible Improvement
#
# * make sure we follow Rust Style Guidelines_
# * make use of pattern matching
# * better support for reference
# * generate generic code and use trait to make sure they have specific methods
# * use crates_ to get more math support
#     - num_
#         + BigInt_, BigUint_
#         + Complex_
#         + Rational64_, Rational32_, BigRational_
#
# .. _crates: https://crates.io/
# .. _Guidelines: https://github.com/rust-lang/rust/tree/master/src/doc/style
# .. _num: http://rust-num.github.io/num/num/
# .. _BigInt: http://rust-num.github.io/num/num/bigint/struct.BigInt.html
# .. _BigUint: http://rust-num.github.io/num/num/bigint/struct.BigUint.html
# .. _Complex: http://rust-num.github.io/num/num/complex/struct.Complex.html
# .. _Rational32: http://rust-num.github.io/num/num/rational/type.Rational32.html
# .. _Rational64: http://rust-num.github.io/num/num/rational/type.Rational64.html
# .. _BigRational: http://rust-num.github.io/num/num/rational/type.BigRational.html

from __future__ import annotations
from typing import Any

from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter

# Rust's methods for integer and float can be found at here :
#
# * `Rust - Primitive Type f64 <https://doc.rust-lang.org/std/primitive.f64.html>`_
# * `Rust - Primitive Type i64 <https://doc.rust-lang.org/std/primitive.i64.html>`_
#
# Function Style :
#
# 1. args[0].func(args[1:]), method with arguments
# 2. args[0].func(), method without arguments
# 3. args[1].func(), method without arguments (e.g. (e, x) => x.exp())
# 4. func(args), function with arguments

# dictionary mapping SymPy function to (argument_conditions, Rust_function).
# Used in RustCodePrinter._print_Function(self)

# f64 method in Rust
known_functions = {
    # "": "is_nan",
    # "": "is_infinite",
    # "": "is_finite",
    # "": "is_normal",
    # "": "classify",
    "floor": "floor",
    "ceiling": "ceil",
    # "": "round",
    # "": "trunc",
    # "": "fract",
    "Abs": "abs",
    "sign": "signum",
    # "": "is_sign_positive",
    # "": "is_sign_negative",
    # "": "mul_add",
    "Pow": [(lambda base, exp: equal_valued(exp, -1), "recip", 2),   # 1.0/x
            (lambda base, exp: equal_valued(exp, 0.5), "sqrt", 2),   # x ** 0.5
            (lambda base, exp: equal_valued(exp, -0.5), "sqrt().recip", 2),   # 1/(x ** 0.5)
            (lambda base, exp: exp == Rational(1, 3), "cbrt", 2),    # x ** (1/3)
            (lambda base, exp: equal_valued(base, 2), "exp2", 3),    # 2 ** x
            (lambda base, exp: exp.is_integer, "powi", 1),           # x ** y, for i32
            (lambda base, exp: not exp.is_integer, "powf", 1)],      # x ** y, for f64
    "exp": [(lambda exp: True, "exp", 2)],   # e ** x
    "log": "ln",
    # "": "log",          # number.log(base)
    # "": "log2",
    # "": "log10",
    # "": "to_degrees",
    # "": "to_radians",
    "Max": "max",
    "Min": "min",
    # "": "hypot",        # (x**2 + y**2) ** 0.5
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    # "": "sin_cos",
    # "": "exp_m1",       # e ** x - 1
    # "": "ln_1p",        # ln(1 + x)
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "sqrt": "sqrt",  # To enable automatic rewrites
}

# i64 method in Rust
# known_functions_i64 = {
#     "": "min_value",
#     "": "max_value",
#     "": "from_str_radix",
#     "": "count_ones",
#     "": "count_zeros",
#     "": "leading_zeros",
#     "": "trainling_zeros",
#     "": "rotate_left",
#     "": "rotate_right",
#     "": "swap_bytes",
#     "": "from_be",
#     "": "from_le",
#     "": "to_be",    # to big endian
#     "": "to_le",    # to little endian
#     "": "checked_add",
#     "": "checked_sub",
#     "": "checked_mul",
#     "": "checked_div",
#     "": "checked_rem",
#     "": "checked_neg",
#     "": "checked_shl",
#     "": "checked_shr",
#     "": "checked_abs",
#     "": "saturating_add",
#     "": "saturating_sub",
#     "": "saturating_mul",
#     "": "wrapping_add",
#     "": "wrapping_sub",
#     "": "wrapping_mul",
#     "": "wrapping_div",
#     "": "wrapping_rem",
#     "": "wrapping_neg",
#     "": "wrapping_shl",
#     "": "wrapping_shr",
#     "": "wrapping_abs",
#     "": "overflowing_add",
#     "": "overflowing_sub",
#     "": "overflowing_mul",
#     "": "overflowing_div",
#     "": "overflowing_rem",
#     "": "overflowing_neg",
#     "": "overflowing_shl",
#     "": "overflowing_shr",
#     "": "overflowing_abs",
#     "Pow": "pow",
#     "Abs": "abs",
#     "sign": "signum",
#     "": "is_positive",
#     "": "is_negnative",
# }

# These are the core reserved words in the Rust language. Taken from:
# http://doc.rust-lang.org/grammar.html#keywords

reserved_words = ['abstract',
                  'alignof',
                  'as',
                  'become',
                  'box',
                  'break',
                  'const',
                  'continue',
                  'crate',
                  'do',
                  'else',
                  'enum',
                  'extern',
                  'false',
                  'final',
                  'fn',
                  'for',
                  'if',
                  'impl',
                  'in',
                  'let',
                  'loop',
                  'macro',
                  'match',
                  'mod',
                  'move',
                  'mut',
                  'offsetof',
                  'override',
                  'priv',
                  'proc',
                  'pub',
                  'pure',
                  'ref',
                  'return',
                  'Self',
                  'self',
                  'sizeof',
                  'static',
                  'struct',
                  'super',
                  'trait',
                  'true',
                  'type',
                  'typeof',
                  'unsafe',
                  'unsized',
                  'use',
                  'virtual',
                  'where',
                  'while',
                  'yield']


class RustCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of Rust code"""
    printmethod = "_rust_code"
    language = "Rust"

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
        'dereference': set(),
    })

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
        return "// %s" % text

    def _declare_number_const(self, name, value):
        return "const %s: f64 = %s;" % (name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for %(var)s in %(start)s..%(end)s {"
        for i in indices:
            # Rust arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'var': self._print(i),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

    def _print_caller_var(self, expr):
        if len(expr.args) > 1:
            # for something like `sin(x + y + z)`,
            # make sure we can get '(x + y + z).sin()'
            # instead of 'x + y + z.sin()'
            return '(' + self._print(expr) + ')'
        elif expr.is_number:
            return self._print(expr, _type=True)
        else:
            return self._print(expr)

    def _print_Function(self, expr):
        """
        basic function for printing `Function`

        Function Style :

        1. args[0].func(args[1:]), method with arguments
        2. args[0].func(), method without arguments
        3. args[1].func(), method without arguments (e.g. (e, x) => x.exp())
        4. func(args), function with arguments
        """

        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            func = None
            style = 1
            if isinstance(cond_func, str):
                func = cond_func
            else:
                for cond, func, style in cond_func:
                    if cond(*expr.args):
                        break
            if func is not None:
                if style == 1:
                    ret = "%(var)s.%(method)s(%(args)s)" % {
                        'var': self._print_caller_var(expr.args[0]),
                        'method': func,
                        'args': self.stringify(expr.args[1:], ", ") if len(expr.args) > 1 else ''
                    }
                elif style == 2:
                    ret = "%(var)s.%(method)s()" % {
                        'var': self._print_caller_var(expr.args[0]),
                        'method': func,
                    }
                elif style == 3:
                    ret = "%(var)s.%(method)s()" % {
                        'var': self._print_caller_var(expr.args[1]),
                        'method': func,
                    }
                else:
                    ret = "%(func)s(%(args)s)" % {
                        'func': func,
                        'args': self.stringify(expr.args, ", "),
                    }
                return ret
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            # inlined function
            return self._print(expr._imp_(*expr.args))
        elif expr.func.__name__ in self._rewriteable_functions:
            # Simple rewrite to supported function possible
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all(self._can_print(f) for f in required_fs):
                return self._print(expr.rewrite(target_f))
        else:
            return self._print_not_supported(expr)

    def _print_Pow(self, expr):
        if expr.base.is_integer and not expr.exp.is_integer:
            expr = type(expr)(Float(expr.base), expr.exp)
            return self._print(expr)
        return self._print_Function(expr)

    def _print_Float(self, expr, _type=False):
        ret = super()._print_Float(expr)
        if _type:
            return ret + '_f64'
        else:
            return ret

    def _print_Integer(self, expr, _type=False):
        ret = super()._print_Integer(expr)
        if _type:
            return ret + '_i32'
        else:
            return ret

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d_f64/%d.0' % (p, q)

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "%s[%s]" % (self._print(expr.base.label), self._print(elem))

    def _print_Idx(self, expr):
        return expr.label.name

    def _print_Dummy(self, expr):
        return expr.name

    def _print_Exp1(self, expr, _type=False):
        return "E"

    def _print_Pi(self, expr, _type=False):
        return 'PI'

    def _print_Infinity(self, expr, _type=False):
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr, _type=False):
        return 'NEG_INFINITY'

    def _print_BooleanTrue(self, expr, _type=False):
        return "true"

    def _print_BooleanFalse(self, expr, _type=False):
        return "false"

    def _print_bool(self, expr, _type=False):
        return str(expr).lower()

    def _print_NaN(self, expr, _type=False):
        return "NAN"

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

        for i, (e, c) in enumerate(expr.args):
            if i == 0:
                lines.append("if (%s) {" % self._print(c))
            elif i == len(expr.args) - 1 and c == True:
                lines[-1] += " else {"
            else:
                lines[-1] += " else if (%s) {" % self._print(c)
            code0 = self._print(e)
            lines.append(code0)
            lines.append("}")

        if self._settings['inline']:
            return " ".join(lines)
        else:
            return "\n".join(lines)

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise, deep=False))

    def _print_MatrixBase(self, A):
        if A.cols == 1:
            return "[%s]" % ", ".join(self._print(a) for a in A)
        else:
            raise ValueError("Full Matrix Support in Rust need Crates (https://crates.io/keywords/matrix).")

    def _print_SparseRepMatrix(self, mat):
        # do not allow sparse matrices to be made dense
        return self._print_not_supported(mat)

    def _print_MatrixElement(self, expr):
        return "%s[%s]" % (expr.parent,
                           expr.j + expr.i*expr.parent.shape[1])

    def _print_Symbol(self, expr):

        name = super()._print_Symbol(expr)

        if expr in self._dereference:
            return '(*%s)' % name
        else:
            return name

    def _print_Assignment(self, expr):
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        if self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "    "
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


def rust_code(expr, assign_to=None, **settings):
    """Converts an expr to a string of Rust code

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
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)].  See below for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
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

    >>> from sympy import rust_code, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> rust_code((2*tau)**Rational(7, 2))
    '8*1.4142135623731*tau.powf(7_f64/2.0)'
    >>> rust_code(sin(x), assign_to="s")
    's = x.sin();'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs", 4),
    ...           (lambda x: x.is_integer, "ABS", 4)],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> rust_code(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    '(fabs(x) + x.CEIL()).f()'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(rust_code(expr, tau))
    tau = if (x > 0) {
        x + 1
    } else {
        x
    };

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
    >>> rust_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(rust_code(mat, A))
    A = [x.powi(2), if (x > 0) {
        x + 1
    } else {
        x
    }, x.sin()];
    """

    return RustCodePrinter(settings).doprint(expr, assign_to)


def print_rust_code(expr, **settings):
    """Prints Rust representation of the given expression."""
    print(rust_code(expr, **settings))
