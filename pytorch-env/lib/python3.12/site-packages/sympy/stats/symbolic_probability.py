import itertools
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import Not
from sympy.core.parameters import global_parameters
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.stats import variance, covariance
from sympy.stats.rv import (RandomSymbol, pspace, dependent,
                            given, sampling_E, RandomIndexedSymbol, is_random,
                            PSpace, sampling_P, random_symbols)

__all__ = ['Probability', 'Expectation', 'Variance', 'Covariance']


@is_random.register(Expr)
def _(x):
    atoms = x.free_symbols
    if len(atoms) == 1 and next(iter(atoms)) == x:
        return False
    return any(is_random(i) for i in atoms)

@is_random.register(RandomSymbol)  # type: ignore
def _(x):
    return True


class Probability(Expr):
    """
    Symbolic expression for the probability.

    Examples
    ========

    >>> from sympy.stats import Probability, Normal
    >>> from sympy import Integral
    >>> X = Normal("X", 0, 1)
    >>> prob = Probability(X > 1)
    >>> prob
    Probability(X > 1)

    Integral representation:

    >>> prob.rewrite(Integral)
    Integral(sqrt(2)*exp(-_z**2/2)/(2*sqrt(pi)), (_z, 1, oo))

    Evaluation of the integral:

    >>> prob.evaluate_integral()
    sqrt(2)*(-sqrt(2)*sqrt(pi)*erf(sqrt(2)/2) + sqrt(2)*sqrt(pi))/(4*sqrt(pi))
    """

    is_commutative = True

    def __new__(cls, prob, condition=None, **kwargs):
        prob = _sympify(prob)
        if condition is None:
            obj = Expr.__new__(cls, prob)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, prob, condition)
        obj._condition = condition
        return obj

    def doit(self, **hints):
        condition = self.args[0]
        given_condition = self._condition
        numsamples = hints.get('numsamples', False)
        evaluate = hints.get('evaluate', True)

        if isinstance(condition, Not):
            return S.One - self.func(condition.args[0], given_condition,
                                    evaluate=evaluate).doit(**hints)

        if condition.has(RandomIndexedSymbol):
            return pspace(condition).probability(condition, given_condition,
                                             evaluate=evaluate)

        if isinstance(given_condition, RandomSymbol):
            condrv = random_symbols(condition)
            if len(condrv) == 1 and condrv[0] == given_condition:
                from sympy.stats.frv_types import BernoulliDistribution
                return BernoulliDistribution(self.func(condition).doit(**hints), 0, 1)
            if any(dependent(rv, given_condition) for rv in condrv):
                return Probability(condition, given_condition)
            else:
                return Probability(condition).doit()

        if given_condition is not None and \
                not isinstance(given_condition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (given_condition))

        if given_condition == False or condition is S.false:
            return S.Zero
        if not isinstance(condition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (condition))
        if condition is S.true:
            return S.One

        if numsamples:
            return sampling_P(condition, given_condition, numsamples=numsamples)
        if given_condition is not None:  # If there is a condition
            # Recompute on new conditional expr
            return Probability(given(condition, given_condition)).doit()

        # Otherwise pass work off to the ProbabilitySpace
        if pspace(condition) == PSpace():
            return Probability(condition, given_condition)

        result = pspace(condition).probability(condition)
        if hasattr(result, 'doit') and evaluate:
            return result.doit()
        else:
            return result

    def _eval_rewrite_as_Integral(self, arg, condition=None, **kwargs):
        return self.func(arg, condition=condition).doit(evaluate=False)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()


class Expectation(Expr):
    """
    Symbolic expression for the expectation.

    Examples
    ========

    >>> from sympy.stats import Expectation, Normal, Probability, Poisson
    >>> from sympy import symbols, Integral, Sum
    >>> mu = symbols("mu")
    >>> sigma = symbols("sigma", positive=True)
    >>> X = Normal("X", mu, sigma)
    >>> Expectation(X)
    Expectation(X)
    >>> Expectation(X).evaluate_integral().simplify()
    mu

    To get the integral expression of the expectation:

    >>> Expectation(X).rewrite(Integral)
    Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    The same integral expression, in more abstract terms:

    >>> Expectation(X).rewrite(Probability)
    Integral(x*Probability(Eq(X, x)), (x, -oo, oo))

    To get the Summation expression of the expectation for discrete random variables:

    >>> lamda = symbols('lamda', positive=True)
    >>> Z = Poisson('Z', lamda)
    >>> Expectation(Z).rewrite(Sum)
    Sum(Z*lamda**Z*exp(-lamda)/factorial(Z), (Z, 0, oo))

    This class is aware of some properties of the expectation:

    >>> from sympy.abc import a
    >>> Expectation(a*X)
    Expectation(a*X)
    >>> Y = Normal("Y", 1, 2)
    >>> Expectation(X + Y)
    Expectation(X + Y)

    To expand the ``Expectation`` into its expression, use ``expand()``:

    >>> Expectation(X + Y).expand()
    Expectation(X) + Expectation(Y)
    >>> Expectation(a*X + Y).expand()
    a*Expectation(X) + Expectation(Y)
    >>> Expectation(a*X + Y)
    Expectation(a*X + Y)
    >>> Expectation((X + Y)*(X - Y)).expand()
    Expectation(X**2) - Expectation(Y**2)

    To evaluate the ``Expectation``, use ``doit()``:

    >>> Expectation(X + Y).doit()
    mu + 1
    >>> Expectation(X + Expectation(Y + Expectation(2*X))).doit()
    3*mu + 1

    To prevent evaluating nested ``Expectation``, use ``doit(deep=False)``

    >>> Expectation(X + Expectation(Y)).doit(deep=False)
    mu + Expectation(Expectation(Y))
    >>> Expectation(X + Expectation(Y + Expectation(2*X))).doit(deep=False)
    mu + Expectation(Expectation(Expectation(2*X) + Y))

    """

    def __new__(cls, expr, condition=None, **kwargs):
        expr = _sympify(expr)
        if expr.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import ExpectationMatrix
            return ExpectationMatrix(expr, condition)
        if condition is None:
            if not is_random(expr):
                return expr
            obj = Expr.__new__(cls, expr)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, expr, condition)
        obj._condition = condition
        return obj

    def _eval_is_commutative(self):
        return(self.args[0].is_commutative)

    def expand(self, **hints):
        expr = self.args[0]
        condition = self._condition

        if not is_random(expr):
            return expr

        if isinstance(expr, Add):
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expr.args)

        expand_expr = _expand(expr)
        if isinstance(expand_expr, Add):
            return Add.fromiter(Expectation(a, condition=condition).expand()
                    for a in expand_expr.args)

        elif isinstance(expr, Mul):
            rv = []
            nonrv = []
            for a in expr.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a)
            return Mul.fromiter(nonrv)*Expectation(Mul.fromiter(rv), condition=condition)

        return self

    def doit(self, **hints):
        deep = hints.get('deep', True)
        condition = self._condition
        expr = self.args[0]
        numsamples = hints.get('numsamples', False)
        evaluate = hints.get('evaluate', True)

        if deep:
            expr = expr.doit(**hints)

        if not is_random(expr) or isinstance(expr, Expectation):  # expr isn't random?
            return expr
        if numsamples:  # Computing by monte carlo sampling?
            evalf = hints.get('evalf', True)
            return sampling_E(expr, condition, numsamples=numsamples, evalf=evalf)

        if expr.has(RandomIndexedSymbol):
            return pspace(expr).compute_expectation(expr, condition)

        # Create new expr and recompute E
        if condition is not None:  # If there is a condition
            return self.func(given(expr, condition)).doit(**hints)

        # A few known statements for efficiency

        if expr.is_Add:  # We know that E is Linear
            return Add(*[self.func(arg, condition).doit(**hints)
                    if not isinstance(arg, Expectation) else self.func(arg, condition)
                         for arg in expr.args])
        if expr.is_Mul:
            if expr.atoms(Expectation):
                return expr

        if pspace(expr) == PSpace():
            return self.func(expr)
        # Otherwise case is simple, pass work off to the ProbabilitySpace
        result = pspace(expr).compute_expectation(expr, evaluate=evaluate)
        if hasattr(result, 'doit') and evaluate:
            return result.doit(**hints)
        else:
            return result


    def _eval_rewrite_as_Probability(self, arg, condition=None, **kwargs):
        rvs = arg.atoms(RandomSymbol)
        if len(rvs) > 1:
            raise NotImplementedError()
        if len(rvs) == 0:
            return arg

        rv = rvs.pop()
        if rv.pspace is None:
            raise ValueError("Probability space not known")

        symbol = rv.symbol
        if symbol.name[0].isupper():
            symbol = Symbol(symbol.name.lower())
        else :
            symbol = Symbol(symbol.name + "_1")

        if rv.pspace.is_Continuous:
            return Integral(arg.replace(rv, symbol)*Probability(Eq(rv, symbol), condition), (symbol, rv.pspace.domain.set.inf, rv.pspace.domain.set.sup))
        else:
            if rv.pspace.is_Finite:
                raise NotImplementedError
            else:
                return Sum(arg.replace(rv, symbol)*Probability(Eq(rv, symbol), condition), (symbol, rv.pspace.domain.set.inf, rv.pspace.set.sup))

    def _eval_rewrite_as_Integral(self, arg, condition=None, evaluate=False, **kwargs):
        return self.func(arg, condition=condition).doit(deep=False, evaluate=evaluate)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral # For discrete this will be Sum

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()

    evaluate_sum = evaluate_integral

class Variance(Expr):
    """
    Symbolic expression for the variance.

    Examples
    ========

    >>> from sympy import symbols, Integral
    >>> from sympy.stats import Normal, Expectation, Variance, Probability
    >>> mu = symbols("mu", positive=True)
    >>> sigma = symbols("sigma", positive=True)
    >>> X = Normal("X", mu, sigma)
    >>> Variance(X)
    Variance(X)
    >>> Variance(X).evaluate_integral()
    sigma**2

    Integral representation of the underlying calculations:

    >>> Variance(X).rewrite(Integral)
    Integral(sqrt(2)*(X - Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo)))**2*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    Integral representation, without expanding the PDF:

    >>> Variance(X).rewrite(Probability)
    -Integral(x*Probability(Eq(X, x)), (x, -oo, oo))**2 + Integral(x**2*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the variance in terms of the expectation

    >>> Variance(X).rewrite(Expectation)
    -Expectation(X)**2 + Expectation(X**2)

    Some transformations based on the properties of the variance may happen:

    >>> from sympy.abc import a
    >>> Y = Normal("Y", 0, 1)
    >>> Variance(a*X)
    Variance(a*X)

    To expand the variance in its expression, use ``expand()``:

    >>> Variance(a*X).expand()
    a**2*Variance(X)
    >>> Variance(X + Y)
    Variance(X + Y)
    >>> Variance(X + Y).expand()
    2*Covariance(X, Y) + Variance(X) + Variance(Y)

    """
    def __new__(cls, arg, condition=None, **kwargs):
        arg = _sympify(arg)

        if arg.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import VarianceMatrix
            return VarianceMatrix(arg, condition)
        if condition is None:
            obj = Expr.__new__(cls, arg)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, arg, condition)
        obj._condition = condition
        return obj

    def _eval_is_commutative(self):
        return self.args[0].is_commutative

    def expand(self, **hints):
        arg = self.args[0]
        condition = self._condition

        if not is_random(arg):
            return S.Zero

        if isinstance(arg, RandomSymbol):
            return self
        elif isinstance(arg, Add):
            rv = []
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
            variances = Add(*(Variance(xv, condition).expand() for xv in rv))
            map_to_covar = lambda x: 2*Covariance(*x, condition=condition).expand()
            covariances = Add(*map(map_to_covar, itertools.combinations(rv, 2)))
            return variances + covariances
        elif isinstance(arg, Mul):
            nonrv = []
            rv = []
            for a in arg.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a**2)
            if len(rv) == 0:
                return S.Zero
            return Mul.fromiter(nonrv)*Variance(Mul.fromiter(rv), condition)

        # this expression contains a RandomSymbol somehow:
        return self

    def _eval_rewrite_as_Expectation(self, arg, condition=None, **kwargs):
            e1 = Expectation(arg**2, condition)
            e2 = Expectation(arg, condition)**2
            return e1 - e2

    def _eval_rewrite_as_Probability(self, arg, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, arg, condition=None, **kwargs):
        return variance(self.args[0], self._condition, evaluate=False)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()


class Covariance(Expr):
    """
    Symbolic expression for the covariance.

    Examples
    ========

    >>> from sympy.stats import Covariance
    >>> from sympy.stats import Normal
    >>> X = Normal("X", 3, 2)
    >>> Y = Normal("Y", 0, 1)
    >>> Z = Normal("Z", 0, 1)
    >>> W = Normal("W", 0, 1)
    >>> cexpr = Covariance(X, Y)
    >>> cexpr
    Covariance(X, Y)

    Evaluate the covariance, `X` and `Y` are independent,
    therefore zero is the result:

    >>> cexpr.evaluate_integral()
    0

    Rewrite the covariance expression in terms of expectations:

    >>> from sympy.stats import Expectation
    >>> cexpr.rewrite(Expectation)
    Expectation(X*Y) - Expectation(X)*Expectation(Y)

    In order to expand the argument, use ``expand()``:

    >>> from sympy.abc import a, b, c, d
    >>> Covariance(a*X + b*Y, c*Z + d*W)
    Covariance(a*X + b*Y, c*Z + d*W)
    >>> Covariance(a*X + b*Y, c*Z + d*W).expand()
    a*c*Covariance(X, Z) + a*d*Covariance(W, X) + b*c*Covariance(Y, Z) + b*d*Covariance(W, Y)

    This class is aware of some properties of the covariance:

    >>> Covariance(X, X).expand()
    Variance(X)
    >>> Covariance(a*X, b*Y).expand()
    a*b*Covariance(X, Y)
    """

    def __new__(cls, arg1, arg2, condition=None, **kwargs):
        arg1 = _sympify(arg1)
        arg2 = _sympify(arg2)

        if arg1.is_Matrix or arg2.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import CrossCovarianceMatrix
            return CrossCovarianceMatrix(arg1, arg2, condition)

        if kwargs.pop('evaluate', global_parameters.evaluate):
            arg1, arg2 = sorted([arg1, arg2], key=default_sort_key)

        if condition is None:
            obj = Expr.__new__(cls, arg1, arg2)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, arg1, arg2, condition)
        obj._condition = condition
        return obj

    def _eval_is_commutative(self):
        return self.args[0].is_commutative

    def expand(self, **hints):
        arg1 = self.args[0]
        arg2 = self.args[1]
        condition = self._condition

        if arg1 == arg2:
            return Variance(arg1, condition).expand()

        if not is_random(arg1):
            return S.Zero
        if not is_random(arg2):
            return S.Zero

        arg1, arg2 = sorted([arg1, arg2], key=default_sort_key)

        if isinstance(arg1, RandomSymbol) and isinstance(arg2, RandomSymbol):
            return Covariance(arg1, arg2, condition)

        coeff_rv_list1 = self._expand_single_argument(arg1.expand())
        coeff_rv_list2 = self._expand_single_argument(arg2.expand())

        addends = [a*b*Covariance(*sorted([r1, r2], key=default_sort_key), condition=condition)
                   for (a, r1) in coeff_rv_list1 for (b, r2) in coeff_rv_list2]
        return Add.fromiter(addends)

    @classmethod
    def _expand_single_argument(cls, expr):
        # return (coefficient, random_symbol) pairs:
        if isinstance(expr, RandomSymbol):
            return [(S.One, expr)]
        elif isinstance(expr, Add):
            outval = []
            for a in expr.args:
                if isinstance(a, Mul):
                    outval.append(cls._get_mul_nonrv_rv_tuple(a))
                elif is_random(a):
                    outval.append((S.One, a))

            return outval
        elif isinstance(expr, Mul):
            return [cls._get_mul_nonrv_rv_tuple(expr)]
        elif is_random(expr):
            return [(S.One, expr)]

    @classmethod
    def _get_mul_nonrv_rv_tuple(cls, m):
        rv = []
        nonrv = []
        for a in m.args:
            if is_random(a):
                rv.append(a)
            else:
                nonrv.append(a)
        return (Mul.fromiter(nonrv), Mul.fromiter(rv))

    def _eval_rewrite_as_Expectation(self, arg1, arg2, condition=None, **kwargs):
        e1 = Expectation(arg1*arg2, condition)
        e2 = Expectation(arg1, condition)*Expectation(arg2, condition)
        return e1 - e2

    def _eval_rewrite_as_Probability(self, arg1, arg2, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, arg1, arg2, condition=None, **kwargs):
        return covariance(self.args[0], self.args[1], self._condition, evaluate=False)

    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()


class Moment(Expr):
    """
    Symbolic class for Moment

    Examples
    ========

    >>> from sympy import Symbol, Integral
    >>> from sympy.stats import Normal, Expectation, Probability, Moment
    >>> mu = Symbol('mu', real=True)
    >>> sigma = Symbol('sigma', positive=True)
    >>> X = Normal('X', mu, sigma)
    >>> M = Moment(X, 3, 1)

    To evaluate the result of Moment use `doit`:

    >>> M.doit()
    mu**3 - 3*mu**2 + 3*mu*sigma**2 + 3*mu - 3*sigma**2 - 1

    Rewrite the Moment expression in terms of Expectation:

    >>> M.rewrite(Expectation)
    Expectation((X - 1)**3)

    Rewrite the Moment expression in terms of Probability:

    >>> M.rewrite(Probability)
    Integral((x - 1)**3*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the Moment expression in terms of Integral:

    >>> M.rewrite(Integral)
    Integral(sqrt(2)*(X - 1)**3*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    """
    def __new__(cls, X, n, c=0, condition=None, **kwargs):
        X = _sympify(X)
        n = _sympify(n)
        c = _sympify(c)
        if condition is not None:
            condition = _sympify(condition)
            return super().__new__(cls, X, n, c, condition)
        else:
            return super().__new__(cls, X, n, c)

    def doit(self, **hints):
        return self.rewrite(Expectation).doit(**hints)

    def _eval_rewrite_as_Expectation(self, X, n, c=0, condition=None, **kwargs):
        return Expectation((X - c)**n, condition)

    def _eval_rewrite_as_Probability(self, X, n, c=0, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, X, n, c=0, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Integral)


class CentralMoment(Expr):
    """
    Symbolic class Central Moment

    Examples
    ========

    >>> from sympy import Symbol, Integral
    >>> from sympy.stats import Normal, Expectation, Probability, CentralMoment
    >>> mu = Symbol('mu', real=True)
    >>> sigma = Symbol('sigma', positive=True)
    >>> X = Normal('X', mu, sigma)
    >>> CM = CentralMoment(X, 4)

    To evaluate the result of CentralMoment use `doit`:

    >>> CM.doit().simplify()
    3*sigma**4

    Rewrite the CentralMoment expression in terms of Expectation:

    >>> CM.rewrite(Expectation)
    Expectation((-Expectation(X) + X)**4)

    Rewrite the CentralMoment expression in terms of Probability:

    >>> CM.rewrite(Probability)
    Integral((x - Integral(x*Probability(True), (x, -oo, oo)))**4*Probability(Eq(X, x)), (x, -oo, oo))

    Rewrite the CentralMoment expression in terms of Integral:

    >>> CM.rewrite(Integral)
    Integral(sqrt(2)*(X - Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo)))**4*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    """
    def __new__(cls, X, n, condition=None, **kwargs):
        X = _sympify(X)
        n = _sympify(n)
        if condition is not None:
            condition = _sympify(condition)
            return super().__new__(cls, X, n, condition)
        else:
            return super().__new__(cls, X, n)

    def doit(self, **hints):
        return self.rewrite(Expectation).doit(**hints)

    def _eval_rewrite_as_Expectation(self, X, n, condition=None, **kwargs):
        mu = Expectation(X, condition, **kwargs)
        return Moment(X, n, mu, condition, **kwargs).rewrite(Expectation)

    def _eval_rewrite_as_Probability(self, X, n, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Probability)

    def _eval_rewrite_as_Integral(self, X, n, condition=None, **kwargs):
        return self.rewrite(Expectation).rewrite(Integral)
