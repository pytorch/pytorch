"""
The objects in this module allow the usage of the MatchPy pattern matching
library on SymPy expressions.
"""
import re
from typing import List, Callable, NamedTuple, Any, Dict

from sympy.core.sympify import _sympify
from sympy.external import import_module
from sympy.functions import (log, sin, cos, tan, cot, csc, sec, erf, gamma, uppergamma)
from sympy.functions.elementary.hyperbolic import acosh, asinh, atanh, acoth, acsch, asech, cosh, sinh, tanh, coth, sech, csch
from sympy.functions.elementary.trigonometric import atan, acsc, asin, acot, acos, asec
from sympy.functions.special.error_functions import fresnelc, fresnels, erfc, erfi, Ei
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import (Equality, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.integrals.integrals import Integral
from sympy.printing.repr import srepr
from sympy.utilities.decorator import doctest_depends_on


matchpy = import_module("matchpy")


__doctest_requires__ = {('*',): ['matchpy']}


if matchpy:
    from matchpy import Operation, CommutativeOperation, AssociativeOperation, OneIdentityOperation
    from matchpy.expressions.functions import op_iter, create_operation_expression, op_len

    Operation.register(Integral)
    Operation.register(Pow)
    OneIdentityOperation.register(Pow)

    Operation.register(Add)
    OneIdentityOperation.register(Add)
    CommutativeOperation.register(Add)
    AssociativeOperation.register(Add)

    Operation.register(Mul)
    OneIdentityOperation.register(Mul)
    CommutativeOperation.register(Mul)
    AssociativeOperation.register(Mul)

    Operation.register(Equality)
    CommutativeOperation.register(Equality)
    Operation.register(Unequality)
    CommutativeOperation.register(Unequality)

    Operation.register(exp)
    Operation.register(log)
    Operation.register(gamma)
    Operation.register(uppergamma)
    Operation.register(fresnels)
    Operation.register(fresnelc)
    Operation.register(erf)
    Operation.register(Ei)
    Operation.register(erfc)
    Operation.register(erfi)
    Operation.register(sin)
    Operation.register(cos)
    Operation.register(tan)
    Operation.register(cot)
    Operation.register(csc)
    Operation.register(sec)
    Operation.register(sinh)
    Operation.register(cosh)
    Operation.register(tanh)
    Operation.register(coth)
    Operation.register(csch)
    Operation.register(sech)
    Operation.register(asin)
    Operation.register(acos)
    Operation.register(atan)
    Operation.register(acot)
    Operation.register(acsc)
    Operation.register(asec)
    Operation.register(asinh)
    Operation.register(acosh)
    Operation.register(atanh)
    Operation.register(acoth)
    Operation.register(acsch)
    Operation.register(asech)

    @op_iter.register(Integral)  # type: ignore
    def _(operation):
        return iter((operation._args[0],) + operation._args[1])

    @op_iter.register(Basic)  # type: ignore
    def _(operation):
        return iter(operation._args)

    @op_len.register(Integral)  # type: ignore
    def _(operation):
        return 1 + len(operation._args[1])

    @op_len.register(Basic)  # type: ignore
    def _(operation):
        return len(operation._args)

    @create_operation_expression.register(Basic)
    def sympy_op_factory(old_operation, new_operands, variable_name=True):
        return type(old_operation)(*new_operands)


if matchpy:
    from matchpy import Wildcard
else:
    class Wildcard: # type: ignore
        def __init__(self, min_length, fixed_size, variable_name, optional):
            self.min_count = min_length
            self.fixed_size = fixed_size
            self.variable_name = variable_name
            self.optional = optional


@doctest_depends_on(modules=('matchpy',))
class _WildAbstract(Wildcard, Symbol):
    min_length: int  # abstract field required in subclasses
    fixed_size: bool  # abstract field required in subclasses

    def __init__(self, variable_name=None, optional=None, **assumptions):
        min_length = self.min_length
        fixed_size = self.fixed_size
        if optional is not None:
            optional = _sympify(optional)
        Wildcard.__init__(self, min_length, fixed_size, str(variable_name), optional)

    def __getstate__(self):
        return {
            "min_length": self.min_length,
            "fixed_size": self.fixed_size,
            "min_count": self.min_count,
            "variable_name": self.variable_name,
            "optional": self.optional,
        }

    def __new__(cls, variable_name=None, optional=None, **assumptions):
        cls._sanitize(assumptions, cls)
        return _WildAbstract.__xnew__(cls, variable_name, optional, **assumptions)

    def __getnewargs__(self):
        return self.variable_name, self.optional

    @staticmethod
    def __xnew__(cls, variable_name=None, optional=None, **assumptions):
        obj = Symbol.__xnew__(cls, variable_name, **assumptions)
        return obj

    def _hashable_content(self):
        if self.optional:
            return super()._hashable_content() + (self.min_count, self.fixed_size, self.variable_name, self.optional)
        else:
            return super()._hashable_content() + (self.min_count, self.fixed_size, self.variable_name)

    def __copy__(self) -> '_WildAbstract':
        return type(self)(variable_name=self.variable_name, optional=self.optional)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name


@doctest_depends_on(modules=('matchpy',))
class WildDot(_WildAbstract):
    min_length = 1
    fixed_size = True


@doctest_depends_on(modules=('matchpy',))
class WildPlus(_WildAbstract):
    min_length = 1
    fixed_size = False


@doctest_depends_on(modules=('matchpy',))
class WildStar(_WildAbstract):
    min_length = 0
    fixed_size = False


def _get_srepr(expr):
    s = srepr(expr)
    s = re.sub(r"WildDot\('(\w+)'\)", r"\1", s)
    s = re.sub(r"WildPlus\('(\w+)'\)", r"*\1", s)
    s = re.sub(r"WildStar\('(\w+)'\)", r"*\1", s)
    return s


class ReplacementInfo(NamedTuple):
    replacement: Any
    info: Any


@doctest_depends_on(modules=('matchpy',))
class Replacer:
    """
    Replacer object to perform multiple pattern matching and subexpression
    replacements in SymPy expressions.

    Examples
    ========

    Example to construct a simple first degree equation solver:

    >>> from sympy.utilities.matchpy_connector import WildDot, Replacer
    >>> from sympy import Equality, Symbol
    >>> x = Symbol("x")
    >>> a_ = WildDot("a_", optional=1)
    >>> b_ = WildDot("b_", optional=0)

    The lines above have defined two wildcards, ``a_`` and ``b_``, the
    coefficients of the equation `a x + b = 0`. The optional values specified
    indicate which expression to return in case no match is found, they are
    necessary in equations like `a x = 0` and `x + b = 0`.

    Create two constraints to make sure that ``a_`` and ``b_`` will not match
    any expression containing ``x``:

    >>> from matchpy import CustomConstraint
    >>> free_x_a = CustomConstraint(lambda a_: not a_.has(x))
    >>> free_x_b = CustomConstraint(lambda b_: not b_.has(x))

    Now create the rule replacer with the constraints:

    >>> replacer = Replacer(common_constraints=[free_x_a, free_x_b])

    Add the matching rule:

    >>> replacer.add(Equality(a_*x + b_, 0), -b_/a_)

    Let's try it:

    >>> replacer.replace(Equality(3*x + 4, 0))
    -4/3

    Notice that it will not match equations expressed with other patterns:

    >>> eq = Equality(3*x, 4)
    >>> replacer.replace(eq)
    Eq(3*x, 4)

    In order to extend the matching patterns, define another one (we also need
    to clear the cache, because the previous result has already been memorized
    and the pattern matcher will not iterate again if given the same expression)

    >>> replacer.add(Equality(a_*x, b_), b_/a_)
    >>> replacer._matcher.clear()
    >>> replacer.replace(eq)
    4/3
    """

    def __init__(self, common_constraints: list = [], lambdify: bool = False, info: bool = False):
        self._matcher = matchpy.ManyToOneMatcher()
        self._common_constraint = common_constraints
        self._lambdify = lambdify
        self._info = info
        self._wildcards: Dict[str, Wildcard] = {}

    def _get_lambda(self, lambda_str: str) -> Callable[..., Expr]:
        exec("from sympy import *")
        return eval(lambda_str, locals())

    def _get_custom_constraint(self, constraint_expr: Expr, condition_template: str) -> Callable[..., Expr]:
        wilds = [x.name for x in constraint_expr.atoms(_WildAbstract)]
        lambdaargs = ', '.join(wilds)
        fullexpr = _get_srepr(constraint_expr)
        condition = condition_template.format(fullexpr)
        return matchpy.CustomConstraint(
            self._get_lambda(f"lambda {lambdaargs}: ({condition})"))

    def _get_custom_constraint_nonfalse(self, constraint_expr: Expr) -> Callable[..., Expr]:
        return self._get_custom_constraint(constraint_expr, "({}) != False")

    def _get_custom_constraint_true(self, constraint_expr: Expr) -> Callable[..., Expr]:
        return self._get_custom_constraint(constraint_expr, "({}) == True")

    def add(self, expr: Expr, replacement, conditions_true: List[Expr] = [],
            conditions_nonfalse: List[Expr] = [], info: Any = None) -> None:
        expr = _sympify(expr)
        replacement = _sympify(replacement)
        constraints = self._common_constraint[:]
        constraint_conditions_true = [
            self._get_custom_constraint_true(cond) for cond in conditions_true]
        constraint_conditions_nonfalse = [
            self._get_custom_constraint_nonfalse(cond) for cond in conditions_nonfalse]
        constraints.extend(constraint_conditions_true)
        constraints.extend(constraint_conditions_nonfalse)
        pattern = matchpy.Pattern(expr, *constraints)
        if self._lambdify:
            lambda_str = f"lambda {', '.join((x.name for x in expr.atoms(_WildAbstract)))}: {_get_srepr(replacement)}"
            lambda_expr = self._get_lambda(lambda_str)
            replacement = lambda_expr
        else:
            self._wildcards.update({str(i): i for i in expr.atoms(Wildcard)})
        if self._info:
            replacement = ReplacementInfo(replacement, info)
        self._matcher.add(pattern, replacement)

    def replace(self, expression, max_count: int = -1):
        # This method partly rewrites the .replace method of ManyToOneReplacer
        # in MatchPy.
        # License: https://github.com/HPAC/matchpy/blob/master/LICENSE
        infos = []
        replaced = True
        replace_count = 0
        while replaced and (max_count < 0 or replace_count < max_count):
            replaced = False
            for subexpr, pos in matchpy.preorder_iter_with_position(expression):
                try:
                    replacement_data, subst = next(iter(self._matcher.match(subexpr)))
                    if self._info:
                        replacement = replacement_data.replacement
                        infos.append(replacement_data.info)
                    else:
                        replacement = replacement_data

                    if self._lambdify:
                        result = replacement(**subst)
                    else:
                        result = replacement.xreplace({self._wildcards[k]: v for k, v in subst.items()})

                    expression = matchpy.functions.replace(expression, pos, result)
                    replaced = True
                    break
                except StopIteration:
                    pass
            replace_count += 1
        if self._info:
            return expression, infos
        else:
            return expression
