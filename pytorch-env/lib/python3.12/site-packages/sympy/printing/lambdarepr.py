from .pycode import (
    PythonCodePrinter,
    MpmathPrinter,
)
from .numpy import NumPyPrinter  # NumPyPrinter is imported for backward compatibility
from sympy.core.sorting import default_sort_key


__all__ = [
    'PythonCodePrinter',
    'MpmathPrinter',  # MpmathPrinter is published for backward compatibility
    'NumPyPrinter',
    'LambdaPrinter',
    'NumPyPrinter',
    'IntervalPrinter',
    'lambdarepr',
]


class LambdaPrinter(PythonCodePrinter):
    """
    This printer converts expressions into strings that can be used by
    lambdify.
    """
    printmethod = "_lambdacode"


    def _print_And(self, expr):
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' and ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Or(self, expr):
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' or ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Not(self, expr):
        result = ['(', 'not (', self._print(expr.args[0]), '))']
        return ''.join(result)

    def _print_BooleanTrue(self, expr):
        return "True"

    def _print_BooleanFalse(self, expr):
        return "False"

    def _print_ITE(self, expr):
        result = [
            '((', self._print(expr.args[1]),
            ') if (', self._print(expr.args[0]),
            ') else (', self._print(expr.args[2]), '))'
        ]
        return ''.join(result)

    def _print_NumberSymbol(self, expr):
        return str(expr)

    def _print_Pow(self, expr, **kwargs):
        # XXX Temporary workaround. Should Python math printer be
        # isolated from PythonCodePrinter?
        return super(PythonCodePrinter, self)._print_Pow(expr, **kwargs)


# numexpr works by altering the string passed to numexpr.evaluate
# rather than by populating a namespace.  Thus a special printer...
class NumExprPrinter(LambdaPrinter):
    # key, value pairs correspond to SymPy name and numexpr name
    # functions not appearing in this dict will raise a TypeError
    printmethod = "_numexprcode"

    _numexpr_functions = {
        'sin' : 'sin',
        'cos' : 'cos',
        'tan' : 'tan',
        'asin': 'arcsin',
        'acos': 'arccos',
        'atan': 'arctan',
        'atan2' : 'arctan2',
        'sinh' : 'sinh',
        'cosh' : 'cosh',
        'tanh' : 'tanh',
        'asinh': 'arcsinh',
        'acosh': 'arccosh',
        'atanh': 'arctanh',
        'ln' : 'log',
        'log': 'log',
        'exp': 'exp',
        'sqrt' : 'sqrt',
        'Abs' : 'abs',
        'conjugate' : 'conj',
        'im' : 'imag',
        're' : 'real',
        'where' : 'where',
        'complex' : 'complex',
        'contains' : 'contains',
    }

    module = 'numexpr'

    def _print_ImaginaryUnit(self, expr):
        return '1j'

    def _print_seq(self, seq, delimiter=', '):
        # simplified _print_seq taken from pretty.py
        s = [self._print(item) for item in seq]
        if s:
            return delimiter.join(s)
        else:
            return ""

    def _print_Function(self, e):
        func_name = e.func.__name__

        nstr = self._numexpr_functions.get(func_name, None)
        if nstr is None:
            # check for implemented_function
            if hasattr(e, '_imp_'):
                return "(%s)" % self._print(e._imp_(*e.args))
            else:
                raise TypeError("numexpr does not support function '%s'" %
                                func_name)
        return "%s(%s)" % (nstr, self._print_seq(e.args))

    def _print_Piecewise(self, expr):
        "Piecewise function printer"
        exprs = [self._print(arg.expr) for arg in expr.args]
        conds = [self._print(arg.cond) for arg in expr.args]
        # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
        #     it will behave the same as passing the 'default' kwarg to select()
        #     *as long as* it is the last element in expr.args.
        # If this is not the case, it may be triggered prematurely.
        ans = []
        parenthesis_count = 0
        is_last_cond_True = False
        for cond, expr in zip(conds, exprs):
            if cond == 'True':
                ans.append(expr)
                is_last_cond_True = True
                break
            else:
                ans.append('where(%s, %s, ' % (cond, expr))
                parenthesis_count += 1
        if not is_last_cond_True:
            # See https://github.com/pydata/numexpr/issues/298
            #
            # simplest way to put a nan but raises
            # 'RuntimeWarning: invalid value encountered in log'
            #
            # There are other ways to do this such as
            #
            #   >>> import numexpr as ne
            #   >>> nan = float('nan')
            #   >>> ne.evaluate('where(x < 0, -1, nan)', {'x': [-1, 2, 3], 'nan':nan})
            #   array([-1., nan, nan])
            #
            # That needs to be handled in the lambdified function though rather
            # than here in the printer.
            ans.append('log(-1)')
        return ''.join(ans) + ')' * parenthesis_count

    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def blacklisted(self, expr):
        raise TypeError("numexpr cannot be used with %s" %
                        expr.__class__.__name__)

    # blacklist all Matrix printing
    _print_SparseRepMatrix = \
    _print_MutableSparseMatrix = \
    _print_ImmutableSparseMatrix = \
    _print_Matrix = \
    _print_DenseMatrix = \
    _print_MutableDenseMatrix = \
    _print_ImmutableMatrix = \
    _print_ImmutableDenseMatrix = \
    blacklisted
    # blacklist some Python expressions
    _print_list = \
    _print_tuple = \
    _print_Tuple = \
    _print_dict = \
    _print_Dict = \
    blacklisted

    def _print_NumExprEvaluate(self, expr):
        evaluate = self._module_format(self.module +".evaluate")
        return "%s('%s', truediv=True)" % (evaluate, self._print(expr.expr))

    def doprint(self, expr):
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        if not isinstance(expr, CodegenAST):
            expr = NumExprEvaluate(expr)
        return super().doprint(expr)

    def _print_Return(self, expr):
        from sympy.codegen.pynodes import NumExprEvaluate
        r, = expr.args
        if not isinstance(r, NumExprEvaluate):
            expr = expr.func(NumExprEvaluate(r))
        return super()._print_Return(expr)

    def _print_Assignment(self, expr):
        from sympy.codegen.pynodes import NumExprEvaluate
        lhs, rhs, *args = expr.args
        if not isinstance(rhs, NumExprEvaluate):
            expr = expr.func(lhs, NumExprEvaluate(rhs), *args)
        return super()._print_Assignment(expr)

    def _print_CodeBlock(self, expr):
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        args = [ arg if isinstance(arg, CodegenAST) else NumExprEvaluate(arg) for arg in expr.args ]
        return super()._print_CodeBlock(self, expr.func(*args))


class IntervalPrinter(MpmathPrinter, LambdaPrinter):
    """Use ``lambda`` printer but print numbers as ``mpi`` intervals. """

    def _print_Integer(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Integer(expr)

    def _print_Rational(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Half(self, expr):
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Pow(self, expr):
        return super(MpmathPrinter, self)._print_Pow(expr, rational=True)


for k in NumExprPrinter._numexpr_functions:
    setattr(NumExprPrinter, '_print_%s' % k, NumExprPrinter._print_Function)

def lambdarepr(expr, **settings):
    """
    Returns a string usable for lambdifying.
    """
    return LambdaPrinter(settings).doprint(expr)
