from sympy.core.symbol import Symbol
from sympy.matrices.dense import (eye, zeros)
from sympy.solvers.solvers import solve_linear_system

N = 8
M = zeros(N, N + 1)
M[:, :N] = eye(N)
S = [Symbol('A%i' % i) for i in range(N)]


def timeit_linsolve_trivial():
    solve_linear_system(M, *S)
