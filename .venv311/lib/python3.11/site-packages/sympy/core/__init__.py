"""Core module. Provides the basic operations needed in sympy.
"""

from .sympify import sympify, SympifyError
from .cache import cacheit
from .assumptions import assumptions, check_assumptions, failing_assumptions, common_assumptions
from .basic import Basic, Atom
from .singleton import S
from .expr import Expr, AtomicExpr, UnevaluatedExpr
from .symbol import Symbol, Wild, Dummy, symbols, var
from .numbers import Number, Float, Rational, Integer, NumberSymbol, \
    RealNumber, igcd, ilcm, seterr, E, I, nan, oo, pi, zoo, \
    AlgebraicNumber, comp, mod_inverse
from .power import Pow
from .intfunc import integer_nthroot, integer_log, num_digits, trailing
from .mul import Mul, prod
from .add import Add
from .mod import Mod
from .relational import ( Rel, Eq, Ne, Lt, Le, Gt, Ge,
    Equality, GreaterThan, LessThan, Unequality, StrictGreaterThan,
    StrictLessThan )
from .multidimensional import vectorize
from .function import Lambda, WildFunction, Derivative, diff, FunctionClass, \
    Function, Subs, expand, PoleError, count_ops, \
    expand_mul, expand_log, expand_func, \
    expand_trig, expand_complex, expand_multinomial, nfloat, \
    expand_power_base, expand_power_exp, arity
from .evalf import PrecisionExhausted, N
from .containers import Tuple, Dict
from .exprtools import gcd_terms, factor_terms, factor_nc
from .parameters import evaluate
from .kind import UndefinedKind, NumberKind, BooleanKind
from .traversal import preorder_traversal, bottom_up, use, postorder_traversal
from .sorting import default_sort_key, ordered

# expose singletons
Catalan = S.Catalan
EulerGamma = S.EulerGamma
GoldenRatio = S.GoldenRatio
TribonacciConstant = S.TribonacciConstant

__all__ = [
    'sympify', 'SympifyError',

    'cacheit',

    'assumptions', 'check_assumptions', 'failing_assumptions',
    'common_assumptions',

    'Basic', 'Atom',

    'S',

    'Expr', 'AtomicExpr', 'UnevaluatedExpr',

    'Symbol', 'Wild', 'Dummy', 'symbols', 'var',

    'Number', 'Float', 'Rational', 'Integer', 'NumberSymbol', 'RealNumber',
    'igcd', 'ilcm', 'seterr', 'E', 'I', 'nan', 'oo', 'pi', 'zoo',
    'AlgebraicNumber', 'comp', 'mod_inverse',

    'Pow',

    'integer_nthroot', 'integer_log', 'num_digits', 'trailing',

    'Mul', 'prod',

    'Add',

    'Mod',

    'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality', 'GreaterThan',
    'LessThan', 'Unequality', 'StrictGreaterThan', 'StrictLessThan',

    'vectorize',

    'Lambda', 'WildFunction', 'Derivative', 'diff', 'FunctionClass',
    'Function', 'Subs', 'expand', 'PoleError', 'count_ops', 'expand_mul',
    'expand_log', 'expand_func', 'expand_trig', 'expand_complex',
    'expand_multinomial', 'nfloat', 'expand_power_base', 'expand_power_exp',
    'arity',

    'PrecisionExhausted', 'N',

    'evalf', # The module?

    'Tuple', 'Dict',

    'gcd_terms', 'factor_terms', 'factor_nc',

    'evaluate',

    'Catalan',
    'EulerGamma',
    'GoldenRatio',
    'TribonacciConstant',

    'UndefinedKind', 'NumberKind', 'BooleanKind',

    'preorder_traversal', 'bottom_up', 'use', 'postorder_traversal',

    'default_sort_key', 'ordered',
]
