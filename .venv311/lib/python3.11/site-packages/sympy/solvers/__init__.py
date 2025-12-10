"""A module for solving all kinds of equations.

    Examples
    ========

    >>> from sympy.solvers import solve
    >>> from sympy.abc import x
    >>> solve(((x + 1)**5).expand(), x)
    [-1]
"""
from sympy.core.assumptions import check_assumptions, failing_assumptions

from .solvers import solve, solve_linear_system, solve_linear_system_LU, \
    solve_undetermined_coeffs, nsolve, solve_linear, checksol, \
    det_quick, inv_quick

from sympy.solvers.diophantine.diophantine import diophantine

from .recurr import rsolve, rsolve_poly, rsolve_ratio, rsolve_hyper

from .ode import checkodesol, classify_ode, dsolve, \
    homogeneous_order

from .polysys import solve_poly_system, solve_triangulated, factor_system

from .pde import pde_separate, pde_separate_add, pde_separate_mul, \
    pdsolve, classify_pde, checkpdesol

from .deutils import ode_order

from .inequalities import reduce_inequalities, reduce_abs_inequality, \
    reduce_abs_inequalities, solve_poly_inequality, solve_rational_inequalities, solve_univariate_inequality

from .decompogen import decompogen

from .solveset import solveset, linsolve, linear_eq_to_matrix, nonlinsolve, substitution

from .simplex import lpmin, lpmax, linprog

# This is here instead of sympy/sets/__init__.py to avoid circular import issues
from ..core.singleton import S
Complexes = S.Complexes

__all__ = [
    'solve', 'solve_linear_system', 'solve_linear_system_LU',
    'solve_undetermined_coeffs', 'nsolve', 'solve_linear', 'checksol',
    'det_quick', 'inv_quick', 'check_assumptions', 'failing_assumptions',

    'diophantine',

    'rsolve', 'rsolve_poly', 'rsolve_ratio', 'rsolve_hyper',

    'checkodesol', 'classify_ode', 'dsolve', 'homogeneous_order',

    'solve_poly_system', 'solve_triangulated', 'factor_system',

    'pde_separate', 'pde_separate_add', 'pde_separate_mul', 'pdsolve',
    'classify_pde', 'checkpdesol',

    'ode_order',

    'reduce_inequalities', 'reduce_abs_inequality', 'reduce_abs_inequalities',
    'solve_poly_inequality', 'solve_rational_inequalities',
    'solve_univariate_inequality',

    'decompogen',

    'solveset', 'linsolve', 'linear_eq_to_matrix', 'nonlinsolve',
    'substitution',

    # This is here instead of sympy/sets/__init__.py to avoid circular import issues
    'Complexes',

    'lpmin', 'lpmax', 'linprog'
]
