from sympy.core.numbers import Rational
from sympy.core.relational import Eq, Ne
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.core.singleton import S
from sympy.core.random import random, choice
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.generate import randprime
from sympy.matrices.dense import Matrix
from sympy.solvers.solveset import linear_eq_to_matrix
from sympy.solvers.simplex import (_lp as lp, _primal_dual,
    UnboundedLPError, InfeasibleLPError, lpmin, lpmax,
    _m, _abcd, _simplex, linprog)

from sympy.external.importtools import import_module

from sympy.testing.pytest import raises

from sympy.abc import x, y, z


np = import_module("numpy")
scipy = import_module("scipy")


def test_lp():
    r1 = y + 2*z <= 3
    r2 = -x - 3*z <= -2
    r3 = 2*x + y + 7*z <= 5
    constraints = [r1, r2, r3, x >= 0, y >= 0, z >= 0]
    objective = -x - y - 5 * z
    ans = optimum, argmax = lp(max, objective, constraints)
    assert ans == lpmax(objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True

    r1 = x - y + 2*z <= 3
    r2 = -x + 2*y - 3*z <= -2
    r3 = 2*x + y - 7*z <= -5
    constraints = [r1, r2, r3, x >= 0, y >= 0, z >= 0]
    objective = -x - y - 5*z
    ans = optimum, argmax = lp(max, objective, constraints)
    assert ans == lpmax(objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True

    r1 = x - y + 2*z <= -4
    r2 = -x + 2*y - 3*z <= 8
    r3 = 2*x + y - 7*z <= 10
    constraints = [r1, r2, r3, x >= 0, y >= 0, z >= 0]
    const = 2
    objective = -x-y-5*z+const # has constant term
    ans = optimum, argmax = lp(max, objective, constraints)
    assert ans == lpmax(objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True

    # Section 4 Problem 1 from
    # http://web.tecnico.ulisboa.pt/mcasquilho/acad/or/ftp/FergusonUCLA_LP.pdf
    # answer on page 55
    v = x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
    r1 = x1 - x2 - 2*x3 - x4 <= 4
    r2 = 2*x1 + x3 -4*x4 <= 2
    r3 = -2*x1 + x2 + x4 <= 1
    objective, constraints = x1 - 2*x2 - 3*x3 - x4, [r1, r2, r3] + [
        i >= 0 for i in v]
    ans = optimum, argmax = lp(max, objective, constraints)
    assert ans == lpmax(objective, constraints)
    assert ans == (4, {x1: 7, x2: 0, x3: 0, x4: 3})

    # input contains Floats
    r1 = x - y + 2.0*z <= -4
    r2 = -x + 2*y - 3.0*z <= 8
    r3 = 2*x + y - 7*z <= 10
    constraints = [r1, r2, r3] + [i >= 0 for i in (x, y, z)]
    objective = -x-y-5*z
    optimum, argmax = lp(max, objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True

    # input contains non-float or non-Rational
    r1 = x - y + sqrt(2) * z <= -4
    r2 = -x + 2*y - 3*z <= 8
    r3 = 2*x + y - 7*z <= 10
    raises(TypeError, lambda: lp(max, -x-y-5*z, [r1, r2, r3]))

    r1 = x >= 0
    raises(UnboundedLPError, lambda: lp(max, x, [r1]))
    r2 = x <= -1
    raises(InfeasibleLPError, lambda: lp(max, x, [r1, r2]))

    # strict inequalities are not allowed
    r1 = x > 0
    raises(TypeError, lambda: lp(max, x, [r1]))

    # not equals not allowed
    r1 = Ne(x, 0)
    raises(TypeError, lambda: lp(max, x, [r1]))

    def make_random_problem(nvar=2, num_constraints=2, sparsity=.1):
        def rand():
            if random() < sparsity:
                return sympify(0)
            int1, int2 = [randprime(0, 200) for _ in range(2)]
            return Rational(int1, int2)*choice([-1, 1])
        variables = symbols('x1:%s' % (nvar + 1))
        constraints = [(sum(rand()*x for x in variables) <= rand())
                       for _ in range(num_constraints)]
        objective = sum(rand() * x for x in variables)
        return objective, constraints, variables

    # equality
    r1 = Eq(x, y)
    r2 = Eq(y, z)
    r3 = z <= 3
    constraints = [r1, r2, r3]
    objective = x
    ans = optimum, argmax = lp(max, objective, constraints)
    assert ans == lpmax(objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True


def test_simplex():
    L = [
        [[1, 1], [-1, 1], [0, 1], [-1, 0]],
        [5, 1, 2, -1],
        [[1, 1]],
        [-1]]
    A, B, C, D = _abcd(_m(*L), list=False)
    assert _simplex(A, B, -C, -D) == (-6, [3, 2], [1, 0, 0, 0])
    assert _simplex(A, B, -C, -D, dual=True) == (-6,
        [1, 0, 0, 0], [5, 0])

    assert _simplex([[]],[],[[1]],[0]) == (0, [0], [])

    # handling of Eq (or Eq-like x<=y, x>=y conditions)
    assert lpmax(x - y, [x <= y + 2, x >= y + 2, x >= 0, y >= 0]
        ) == (2, {x: 2, y: 0})
    assert lpmax(x - y, [x <= y + 2, Eq(x, y + 2), x >= 0, y >= 0]
        ) == (2, {x: 2, y: 0})
    assert lpmax(x - y, [x <= y + 2, Eq(x, 2)]) == (2, {x: 2, y: 0})
    assert lpmax(y, [Eq(y, 2)]) == (2, {y: 2})

    # the conditions are equivalent to Eq(x, y + 2)
    assert lpmin(y, [x <= y + 2, x >= y + 2, y >= 0]
        ) == (0, {x: 2, y: 0})
    # equivalent to Eq(y, -2)
    assert lpmax(y, [0 <= y + 2, 0 >= y + 2]) == (-2, {y: -2})
    assert lpmax(y, [0 <= y + 2, 0 >= y + 2, y <= 0]
        ) == (-2, {y: -2})

    # extra symbols symbols
    assert lpmin(x, [y >= 1, x >= y]) == (1, {x: 1, y: 1})
    assert lpmin(x, [y >= 1, x >= y + z, x >= 0, z >= 0]
        ) == (1, {x: 1, y: 1, z: 0})

    # detect oscillation
    # o1
    v = x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
    raises(InfeasibleLPError, lambda: lpmin(
        9*x2 - 8*x3 + 3*x4 + 6,
        [5*x2 - 2*x3 <= 0,
        -x1 - 8*x2 + 9*x3 <= -3,
        10*x1 - x2+ 9*x4 <= -4] + [i >= 0 for i in v]))
    # o2 - equations fed to lpmin are changed into a matrix
    # system that doesn't oscillate and has the same solution
    # as below
    M = linear_eq_to_matrix
    f = 5*x2 + x3 + 4*x4 - x1
    L = 5*x2 + 2*x3 + 5*x4 - (x1 + 5)
    cond = [L <= 0] + [Eq(3*x2 + x4, 2), Eq(-x1 + x3 + 2*x4, 1)]
    c, d = M(f, v)
    a, b = M(L, v)
    aeq, beq = M(cond[1:], v)
    ans = (S(9)/2, [0, S(1)/2, 0, S(1)/2])
    assert linprog(c, a, b, aeq, beq, bounds=(0, 1)) == ans
    lpans = lpmin(f, cond + [x1 >= 0, x1 <= 1,
        x2 >= 0, x2 <= 1, x3 >= 0, x3 <= 1, x4 >= 0, x4 <= 1])
    assert (lpans[0], list(lpans[1].values())) == ans


def test_lpmin_lpmax():
    v = x1, x2, y1, y2 = symbols('x1 x2 y1 y2')
    L = [[1, -1]], [1], [[1, 1]], [2]
    a, b, c, d = [Matrix(i) for i in L]
    m = Matrix([[a, b], [c, d]])
    f, constr = _primal_dual(m)[0]
    ans = lpmin(f, constr + [i >= 0 for i in v[:2]])
    assert ans == (-1, {x1: 1, x2: 0}),ans

    L = [[1, -1], [1, 1]], [1, 1], [[1, 1]], [2]
    a, b, c, d = [Matrix(i) for i in L]
    m = Matrix([[a, b], [c, d]])
    f, constr = _primal_dual(m)[1]
    ans = lpmax(f, constr + [i >= 0 for i in v[-2:]])
    assert ans == (-1, {y1: 1, y2: 0})


def test_linprog():
    for do in range(2):
        if not do:
            M = lambda a, b: linear_eq_to_matrix(a, b)
        else:
            # check matrices as list
            M = lambda a, b: tuple([
                i.tolist() for i in linear_eq_to_matrix(a, b)])

        v = x, y, z = symbols('x1:4')
        f = x + y - 2*z
        c = M(f, v)[0]
        ineq = [7*x + 4*y - 7*z <= 3,
            3*x - y + 10*z <= 6,
            x >= 0, y >= 0, z >= 0]
        ab = M([i.lts - i.gts for i in ineq], v)
        ans = (-S(6)/5, [0, 0, S(3)/5])
        assert lpmin(f, ineq) == (ans[0], dict(zip(v, ans[1])))
        assert linprog(c, *ab) == ans

        f += 1
        c = M(f, v)[0]
        eq = [Eq(y - 9*x, 1)]
        abeq = M([i.lhs - i.rhs for i in eq], v)
        ans = (1 - S(2)/5, [0, 1, S(7)/10])
        assert lpmin(f, ineq + eq) == (ans[0], dict(zip(v, ans[1])))
        assert linprog(c, *ab, *abeq) == (ans[0] - 1, ans[1])

        eq = [z - y <= S.Half]
        abeq = M([i.lhs - i.rhs for i in eq], v)
        ans = (1 - S(10)/9, [0, S(1)/9, S(11)/18])
        assert lpmin(f, ineq + eq) == (ans[0], dict(zip(v, ans[1])))
        assert linprog(c, *ab, *abeq) == (ans[0] - 1, ans[1])

        bounds = [(0, None), (0, None), (None, S.Half)]
        ans = (0, [0, 0, S.Half])
        assert lpmin(f, ineq + [z <= S.Half]) == (
            ans[0], dict(zip(v, ans[1])))
        assert linprog(c, *ab, bounds=bounds) == (ans[0] - 1, ans[1])
        assert linprog(c, *ab, bounds={v.index(z): bounds[-1]}
            ) == (ans[0] - 1, ans[1])
        eq = [z - y <= S.Half]

    assert linprog([[1]], [], [], bounds=(2, 3)) == (2, [2])
    assert linprog([1], [], [], bounds=(2, 3)) == (2, [2])
    assert linprog([1], bounds=(2, 3)) == (2, [2])
    assert linprog([1, -1], [[1, 1]], [2], bounds={1:(None, None)}
        ) == (-2, [0, 2])
    assert linprog([1, -1], [[1, 1]], [5], bounds={1:(3, None)}
        ) == (-5, [0, 5])
