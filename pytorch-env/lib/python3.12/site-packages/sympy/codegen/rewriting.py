"""
Classes and functions useful for rewriting expressions for optimized code
generation. Some languages (or standards thereof), e.g. C99, offer specialized
math functions for better performance and/or precision.

Using the ``optimize`` function in this module, together with a collection of
rules (represented as instances of ``Optimization``), one can rewrite the
expressions for this purpose::

    >>> from sympy import Symbol, exp, log
    >>> from sympy.codegen.rewriting import optimize, optims_c99
    >>> x = Symbol('x')
    >>> optimize(3*exp(2*x) - 3, optims_c99)
    3*expm1(2*x)
    >>> optimize(exp(2*x) - 1 - exp(-33), optims_c99)
    expm1(2*x) - exp(-33)
    >>> optimize(log(3*x + 3), optims_c99)
    log1p(x) + log(3)
    >>> optimize(log(2*x + 3), optims_c99)
    log(2*x + 3)

The ``optims_c99`` imported above is tuple containing the following instances
(which may be imported from ``sympy.codegen.rewriting``):

- ``expm1_opt``
- ``log1p_opt``
- ``exp2_opt``
- ``log2_opt``
- ``log2const_opt``


"""
from sympy.core.function import expand_log
from sympy.core.singleton import S
from sympy.core.symbol import Wild
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min)
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.assumptions import Q, ask
from sympy.codegen.cfunctions import log1p, log2, exp2, expm1
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core.expr import UnevaluatedExpr
from sympy.core.power import Pow
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.core.mul import Mul
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.utilities.iterables import sift


class Optimization:
    """ Abstract base class for rewriting optimization.

    Subclasses should implement ``__call__`` taking an expression
    as argument.

    Parameters
    ==========
    cost_function : callable returning number
    priority : number

    """
    def __init__(self, cost_function=None, priority=1):
        self.cost_function = cost_function
        self.priority=priority

    def cheapest(self, *args):
        return min(args, key=self.cost_function)


class ReplaceOptim(Optimization):
    """ Rewriting optimization calling replace on expressions.

    Explanation
    ===========

    The instance can be used as a function on expressions for which
    it will apply the ``replace`` method (see
    :meth:`sympy.core.basic.Basic.replace`).

    Parameters
    ==========

    query :
        First argument passed to replace.
    value :
        Second argument passed to replace.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.codegen.rewriting import ReplaceOptim
    >>> from sympy.codegen.cfunctions import exp2
    >>> x = Symbol('x')
    >>> exp2_opt = ReplaceOptim(lambda p: p.is_Pow and p.base == 2,
    ...     lambda p: exp2(p.exp))
    >>> exp2_opt(2**x)
    exp2(x)

    """

    def __init__(self, query, value, **kwargs):
        super().__init__(**kwargs)
        self.query = query
        self.value = value

    def __call__(self, expr):
        return expr.replace(self.query, self.value)


def optimize(expr, optimizations):
    """ Apply optimizations to an expression.

    Parameters
    ==========

    expr : expression
    optimizations : iterable of ``Optimization`` instances
        The optimizations will be sorted with respect to ``priority`` (highest first).

    Examples
    ========

    >>> from sympy import log, Symbol
    >>> from sympy.codegen.rewriting import optims_c99, optimize
    >>> x = Symbol('x')
    >>> optimize(log(x+3)/log(2) + log(x**2 + 1), optims_c99)
    log1p(x**2) + log2(x + 3)

    """

    for optim in sorted(optimizations, key=lambda opt: opt.priority, reverse=True):
        new_expr = optim(expr)
        if optim.cost_function is None:
            expr = new_expr
        else:
            expr = optim.cheapest(expr, new_expr)
    return expr


exp2_opt = ReplaceOptim(
    lambda p: p.is_Pow and p.base == 2,
    lambda p: exp2(p.exp)
)


_d = Wild('d', properties=[lambda x: x.is_Dummy])
_u = Wild('u', properties=[lambda x: not x.is_number and not x.is_Add])
_v = Wild('v')
_w = Wild('w')
_n = Wild('n', properties=[lambda x: x.is_number])

sinc_opt1 = ReplaceOptim(
    sin(_w)/_w, sinc(_w)
)
sinc_opt2 = ReplaceOptim(
    sin(_n*_w)/_w, _n*sinc(_n*_w)
)
sinc_opts = (sinc_opt1, sinc_opt2)

log2_opt = ReplaceOptim(_v*log(_w)/log(2), _v*log2(_w), cost_function=lambda expr: expr.count(
    lambda e: (  # division & eval of transcendentals are expensive floating point operations...
        e.is_Pow and e.exp.is_negative  # division
        or (isinstance(e, (log, log2)) and not e.args[0].is_number))  # transcendental
    )
)

log2const_opt = ReplaceOptim(log(2)*log2(_w), log(_w))

logsumexp_2terms_opt = ReplaceOptim(
    lambda l: (isinstance(l, log)
               and l.args[0].is_Add
               and len(l.args[0].args) == 2
               and all(isinstance(t, exp) for t in l.args[0].args)),
    lambda l: (
        Max(*[e.args[0] for e in l.args[0].args]) +
        log1p(exp(Min(*[e.args[0] for e in l.args[0].args])))
    )
)


class FuncMinusOneOptim(ReplaceOptim):
    """Specialization of ReplaceOptim for functions evaluating "f(x) - 1".

    Explanation
    ===========

    Numerical functions which go toward one as x go toward zero is often best
    implemented by a dedicated function in order to avoid catastrophic
    cancellation. One such example is ``expm1(x)`` in the C standard library
    which evaluates ``exp(x) - 1``. Such functions preserves many more
    significant digits when its argument is much smaller than one, compared
    to subtracting one afterwards.

    Parameters
    ==========

    func :
        The function which is subtracted by one.
    func_m_1 :
        The specialized function evaluating ``func(x) - 1``.
    opportunistic : bool
        When ``True``, apply the transformation as long as the magnitude of the
        remaining number terms decreases. When ``False``, only apply the
        transformation if it completely eliminates the number term.

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy.codegen.rewriting import FuncMinusOneOptim
    >>> from sympy.codegen.cfunctions import expm1
    >>> x, y = symbols('x y')
    >>> expm1_opt = FuncMinusOneOptim(exp, expm1)
    >>> expm1_opt(exp(x) + 2*exp(5*y) - 3)
    expm1(x) + 2*expm1(5*y)


    """

    def __init__(self, func, func_m_1, opportunistic=True):
        weight = 10  # <-- this is an arbitrary number (heuristic)
        super().__init__(lambda e: e.is_Add, self.replace_in_Add,
                         cost_function=lambda expr: expr.count_ops() - weight*expr.count(func_m_1))
        self.func = func
        self.func_m_1 = func_m_1
        self.opportunistic = opportunistic

    def _group_Add_terms(self, add):
        numbers, non_num = sift(add.args, lambda arg: arg.is_number, binary=True)
        numsum = sum(numbers)
        terms_with_func, other = sift(non_num, lambda arg: arg.has(self.func), binary=True)
        return numsum, terms_with_func, other

    def replace_in_Add(self, e):
        """ passed as second argument to Basic.replace(...) """
        numsum, terms_with_func, other_non_num_terms = self._group_Add_terms(e)
        if numsum == 0:
            return e
        substituted, untouched = [], []
        for with_func in terms_with_func:
            if with_func.is_Mul:
                func, coeff = sift(with_func.args, lambda arg: arg.func == self.func, binary=True)
                if len(func) == 1 and len(coeff) == 1:
                    func, coeff = func[0], coeff[0]
                else:
                    coeff = None
            elif with_func.func == self.func:
                func, coeff = with_func, S.One
            else:
                coeff = None

            if coeff is not None and coeff.is_number and sign(coeff) == -sign(numsum):
                if self.opportunistic:
                    do_substitute = abs(coeff+numsum) < abs(numsum)
                else:
                    do_substitute = coeff+numsum == 0

                if do_substitute:  # advantageous substitution
                    numsum += coeff
                    substituted.append(coeff*self.func_m_1(*func.args))
                    continue
            untouched.append(with_func)

        return e.func(numsum, *substituted, *untouched, *other_non_num_terms)

    def __call__(self, expr):
        alt1 = super().__call__(expr)
        alt2 = super().__call__(expr.factor())
        return self.cheapest(alt1, alt2)


expm1_opt = FuncMinusOneOptim(exp, expm1)
cosm1_opt = FuncMinusOneOptim(cos, cosm1)
powm1_opt = FuncMinusOneOptim(Pow, powm1)

log1p_opt = ReplaceOptim(
    lambda e: isinstance(e, log),
    lambda l: expand_log(l.replace(
        log, lambda arg: log(arg.factor())
    )).replace(log(_u+1), log1p(_u))
)

def create_expand_pow_optimization(limit, *, base_req=lambda b: b.is_symbol):
    """ Creates an instance of :class:`ReplaceOptim` for expanding ``Pow``.

    Explanation
    ===========

    The requirements for expansions are that the base needs to be a symbol
    and the exponent needs to be an Integer (and be less than or equal to
    ``limit``).

    Parameters
    ==========

    limit : int
         The highest power which is expanded into multiplication.
    base_req : function returning bool
         Requirement on base for expansion to happen, default is to return
         the ``is_symbol`` attribute of the base.

    Examples
    ========

    >>> from sympy import Symbol, sin
    >>> from sympy.codegen.rewriting import create_expand_pow_optimization
    >>> x = Symbol('x')
    >>> expand_opt = create_expand_pow_optimization(3)
    >>> expand_opt(x**5 + x**3)
    x**5 + x*x*x
    >>> expand_opt(x**5 + x**3 + sin(x)**3)
    x**5 + sin(x)**3 + x*x*x
    >>> opt2 = create_expand_pow_optimization(3, base_req=lambda b: not b.is_Function)
    >>> opt2((x+1)**2 + sin(x)**2)
    sin(x)**2 + (x + 1)*(x + 1)

    """
    return ReplaceOptim(
        lambda e: e.is_Pow and base_req(e.base) and e.exp.is_Integer and abs(e.exp) <= limit,
        lambda p: (
            UnevaluatedExpr(Mul(*([p.base]*+p.exp), evaluate=False)) if p.exp > 0 else
            1/UnevaluatedExpr(Mul(*([p.base]*-p.exp), evaluate=False))
        ))

# Optimization procedures for turning A**(-1) * x into MatrixSolve(A, x)
def _matinv_predicate(expr):
    # TODO: We should be able to support more than 2 elements
    if expr.is_MatMul and len(expr.args) == 2:
        left, right = expr.args
        if left.is_Inverse and right.shape[1] == 1:
            inv_arg = left.arg
            if isinstance(inv_arg, MatrixSymbol):
                return bool(ask(Q.fullrank(left.arg)))

    return False

def _matinv_transform(expr):
    left, right = expr.args
    inv_arg = left.arg
    return MatrixSolve(inv_arg, right)


matinv_opt = ReplaceOptim(_matinv_predicate, _matinv_transform)


logaddexp_opt = ReplaceOptim(log(exp(_v)+exp(_w)), logaddexp(_v, _w))
logaddexp2_opt = ReplaceOptim(log(Pow(2, _v)+Pow(2, _w)), logaddexp2(_v, _w)*log(2))

# Collections of optimizations:
optims_c99 = (expm1_opt, log1p_opt, exp2_opt, log2_opt, log2const_opt)

optims_numpy = optims_c99 + (logaddexp_opt, logaddexp2_opt,) + sinc_opts

optims_scipy = (cosm1_opt, powm1_opt)
