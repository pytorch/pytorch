from functools import reduce
import itertools
from operator import add

from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import Inverse, MatAdd, MatMul, Transpose
from sympy.polys.rootoftools import CRootOf
from sympy.series.order import O
from sympy.simplify.cse_main import cse
from sympy.simplify.simplify import signsimp
from sympy.tensor.indexed import (Idx, IndexedBase)

from sympy.core.function import count_ops
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.functions.special.hyper import meijerg
from sympy.simplify import cse_main, cse_opts
from sympy.utilities.iterables import subsets
from sympy.testing.pytest import XFAIL, raises
from sympy.matrices import (MutableDenseMatrix, MutableSparseMatrix,
        ImmutableDenseMatrix, ImmutableSparseMatrix)
from sympy.matrices.expressions import MatrixSymbol


w, x, y, z = symbols('w,x,y,z')
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = symbols('x:13')


def test_numbered_symbols():
    ns = cse_main.numbered_symbols(prefix='y')
    assert list(itertools.islice(
        ns, 0, 10)) == [Symbol('y%s' % i) for i in range(0, 10)]
    ns = cse_main.numbered_symbols(prefix='y')
    assert list(itertools.islice(
        ns, 10, 20)) == [Symbol('y%s' % i) for i in range(10, 20)]
    ns = cse_main.numbered_symbols()
    assert list(itertools.islice(
        ns, 0, 10)) == [Symbol('x%s' % i) for i in range(0, 10)]

# Dummy "optimization" functions for testing.


def opt1(expr):
    return expr + y


def opt2(expr):
    return expr*z


def test_preprocess_for_cse():
    assert cse_main.preprocess_for_cse(x, [(opt1, None)]) == x + y
    assert cse_main.preprocess_for_cse(x, [(None, opt1)]) == x
    assert cse_main.preprocess_for_cse(x, [(None, None)]) == x
    assert cse_main.preprocess_for_cse(x, [(opt1, opt2)]) == x + y
    assert cse_main.preprocess_for_cse(
        x, [(opt1, None), (opt2, None)]) == (x + y)*z


def test_postprocess_for_cse():
    assert cse_main.postprocess_for_cse(x, [(opt1, None)]) == x
    assert cse_main.postprocess_for_cse(x, [(None, opt1)]) == x + y
    assert cse_main.postprocess_for_cse(x, [(None, None)]) == x
    assert cse_main.postprocess_for_cse(x, [(opt1, opt2)]) == x*z
    # Note the reverse order of application.
    assert cse_main.postprocess_for_cse(
        x, [(None, opt1), (None, opt2)]) == x*z + y


def test_cse_single():
    # Simple substitution.
    e = Add(Pow(x + y, 2), sqrt(x + y))
    substs, reduced = cse([e])
    assert substs == [(x0, x + y)]
    assert reduced == [sqrt(x0) + x0**2]

    subst42, (red42,) = cse([42])  # issue_15082
    assert len(subst42) == 0 and red42 == 42
    subst_half, (red_half,) = cse([0.5])
    assert len(subst_half) == 0 and red_half == 0.5


def test_cse_single2():
    # Simple substitution, test for being able to pass the expression directly
    e = Add(Pow(x + y, 2), sqrt(x + y))
    substs, reduced = cse(e)
    assert substs == [(x0, x + y)]
    assert reduced == [sqrt(x0) + x0**2]
    substs, reduced = cse(Matrix([[1]]))
    assert isinstance(reduced[0], Matrix)

    subst42, (red42,) = cse(42)  # issue 15082
    assert len(subst42) == 0 and red42 == 42
    subst_half, (red_half,) = cse(0.5)  # issue 15082
    assert len(subst_half) == 0 and red_half == 0.5


def test_cse_not_possible():
    # No substitution possible.
    e = Add(x, y)
    substs, reduced = cse([e])
    assert substs == []
    assert reduced == [x + y]
    # issue 6329
    eq = (meijerg((1, 2), (y, 4), (5,), [], x) +
          meijerg((1, 3), (y, 4), (5,), [], x))
    assert cse(eq) == ([], [eq])


def test_nested_substitution():
    # Substitution within a substitution.
    e = Add(Pow(w*x + y, 2), sqrt(w*x + y))
    substs, reduced = cse([e])
    assert substs == [(x0, w*x + y)]
    assert reduced == [sqrt(x0) + x0**2]


def test_subtraction_opt():
    # Make sure subtraction is optimized.
    e = (x - y)*(z - y) + exp((x - y)*(z - y))
    substs, reduced = cse(
        [e], optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])
    assert substs == [(x0, (x - y)*(y - z))]
    assert reduced == [-x0 + exp(-x0)]
    e = -(x - y)*(z - y) + exp(-(x - y)*(z - y))
    substs, reduced = cse(
        [e], optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])
    assert substs == [(x0, (x - y)*(y - z))]
    assert reduced == [x0 + exp(x0)]
    # issue 4077
    n = -1 + 1/x
    e = n/x/(-n)**2 - 1/n/x
    assert cse(e, optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)]) == \
        ([], [0])
    assert cse(((w + x + y + z)*(w - y - z))/(w + x)**3) == \
        ([(x0, w + x), (x1, y + z)], [(w - x1)*(x0 + x1)/x0**3])


def test_multiple_expressions():
    e1 = (x + y)*z
    e2 = (x + y)*w
    substs, reduced = cse([e1, e2])
    assert substs == [(x0, x + y)]
    assert reduced == [x0*z, x0*w]
    l = [w*x*y + z, w*y]
    substs, reduced = cse(l)
    rsubsts, _ = cse(reversed(l))
    assert substs == rsubsts
    assert reduced == [z + x*x0, x0]
    l = [w*x*y, w*x*y + z, w*y]
    substs, reduced = cse(l)
    rsubsts, _ = cse(reversed(l))
    assert substs == rsubsts
    assert reduced == [x1, x1 + z, x0]
    l = [(x - z)*(y - z), x - z, y - z]
    substs, reduced = cse(l)
    rsubsts, _ = cse(reversed(l))
    assert substs == [(x0, -z), (x1, x + x0), (x2, x0 + y)]
    assert rsubsts == [(x0, -z), (x1, x0 + y), (x2, x + x0)]
    assert reduced == [x1*x2, x1, x2]
    l = [w*y + w + x + y + z, w*x*y]
    assert cse(l) == ([(x0, w*y)], [w + x + x0 + y + z, x*x0])
    assert cse([x + y, x + y + z]) == ([(x0, x + y)], [x0, z + x0])
    assert cse([x + y, x + z]) == ([], [x + y, x + z])
    assert cse([x*y, z + x*y, x*y*z + 3]) == \
        ([(x0, x*y)], [x0, z + x0, 3 + x0*z])


@XFAIL # CSE of non-commutative Mul terms is disabled
def test_non_commutative_cse():
    A, B, C = symbols('A B C', commutative=False)
    l = [A*B*C, A*C]
    assert cse(l) == ([], l)
    l = [A*B*C, A*B]
    assert cse(l) == ([(x0, A*B)], [x0*C, x0])


# Test if CSE of non-commutative Mul terms is disabled
def test_bypass_non_commutatives():
    A, B, C = symbols('A B C', commutative=False)
    l = [A*B*C, A*C]
    assert cse(l) == ([], l)
    l = [A*B*C, A*B]
    assert cse(l) == ([], l)
    l = [B*C, A*B*C]
    assert cse(l) == ([], l)


@XFAIL # CSE fails when replacing non-commutative sub-expressions
def test_non_commutative_order():
    A, B, C = symbols('A B C', commutative=False)
    x0 = symbols('x0', commutative=False)
    l = [B+C, A*(B+C)]
    assert cse(l) == ([(x0, B+C)], [x0, A*x0])


@XFAIL # Worked in gh-11232, but was reverted due to performance considerations
def test_issue_10228():
    assert cse([x*y**2 + x*y]) == ([(x0, x*y)], [x0*y + x0])
    assert cse([x + y, 2*x + y]) == ([(x0, x + y)], [x0, x + x0])
    assert cse((w + 2*x + y + z, w + x + 1)) == (
        [(x0, w + x)], [x0 + x + y + z, x0 + 1])
    assert cse(((w + x + y + z)*(w - x))/(w + x)) == (
        [(x0, w + x)], [(x0 + y + z)*(w - x)/x0])
    a, b, c, d, f, g, j, m = symbols('a, b, c, d, f, g, j, m')
    exprs = (d*g**2*j*m, 4*a*f*g*m, a*b*c*f**2)
    assert cse(exprs) == (
        [(x0, g*m), (x1, a*f)], [d*g*j*x0, 4*x0*x1, b*c*f*x1]
)

@XFAIL
def test_powers():
    assert cse(x*y**2 + x*y) == ([(x0, x*y)], [x0*y + x0])


def test_issue_4498():
    assert cse(w/(x - y) + z/(y - x), optimizations='basic') == \
        ([], [(w - z)/(x - y)])


def test_issue_4020():
    assert cse(x**5 + x**4 + x**3 + x**2, optimizations='basic') \
        == ([(x0, x**2)], [x0*(x**3 + x + x0 + 1)])


def test_issue_4203():
    assert cse(sin(x**x)/x**x) == ([(x0, x**x)], [sin(x0)/x0])


def test_issue_6263():
    e = Eq(x*(-x + 1) + x*(x - 1), 0)
    assert cse(e, optimizations='basic') == ([], [True])


def test_issue_25043():
    c = symbols("c")
    x = symbols("x0", real=True)
    cse_expr = cse(c*x**2 + c*(x**4 - x**2))[-1][-1]
    free = cse_expr.free_symbols
    assert len(free) == len({i.name for i in free})


def test_dont_cse_tuples():
    from sympy.core.function import Subs
    f = Function("f")
    g = Function("g")

    name_val, (expr,) = cse(
        Subs(f(x, y), (x, y), (0, 1))
        + Subs(g(x, y), (x, y), (0, 1)))

    assert name_val == []
    assert expr == (Subs(f(x, y), (x, y), (0, 1))
            + Subs(g(x, y), (x, y), (0, 1)))

    name_val, (expr,) = cse(
        Subs(f(x, y), (x, y), (0, x + y))
        + Subs(g(x, y), (x, y), (0, x + y)))

    assert name_val == [(x0, x + y)]
    assert expr == Subs(f(x, y), (x, y), (0, x0)) + \
        Subs(g(x, y), (x, y), (0, x0))


def test_pow_invpow():
    assert cse(1/x**2 + x**2) == \
        ([(x0, x**2)], [x0 + 1/x0])
    assert cse(x**2 + (1 + 1/x**2)/x**2) == \
        ([(x0, x**2), (x1, 1/x0)], [x0 + x1*(x1 + 1)])
    assert cse(1/x**2 + (1 + 1/x**2)*x**2) == \
        ([(x0, x**2), (x1, 1/x0)], [x0*(x1 + 1) + x1])
    assert cse(cos(1/x**2) + sin(1/x**2)) == \
        ([(x0, x**(-2))], [sin(x0) + cos(x0)])
    assert cse(cos(x**2) + sin(x**2)) == \
        ([(x0, x**2)], [sin(x0) + cos(x0)])
    assert cse(y/(2 + x**2) + z/x**2/y) == \
        ([(x0, x**2)], [y/(x0 + 2) + z/(x0*y)])
    assert cse(exp(x**2) + x**2*cos(1/x**2)) == \
        ([(x0, x**2)], [x0*cos(1/x0) + exp(x0)])
    assert cse((1 + 1/x**2)/x**2) == \
        ([(x0, x**(-2))], [x0*(x0 + 1)])
    assert cse(x**(2*y) + x**(-2*y)) == \
        ([(x0, x**(2*y))], [x0 + 1/x0])


def test_postprocess():
    eq = (x + 1 + exp((x + 1)/(y + 1)) + cos(y + 1))
    assert cse([eq, Eq(x, z + 1), z - 2, (z + 1)*(x + 1)],
        postprocess=cse_main.cse_separate) == \
        [[(x0, y + 1), (x2, z + 1), (x, x2), (x1, x + 1)],
        [x1 + exp(x1/x0) + cos(x0), z - 2, x1*x2]]


def test_issue_4499():
    # previously, this gave 16 constants
    from sympy.abc import a, b
    B = Function('B')
    G = Function('G')
    t = Tuple(*
        (a, a + S.Half, 2*a, b, 2*a - b + 1, (sqrt(z)/2)**(-2*a + 1)*B(2*a -
        b, sqrt(z))*B(b - 1, sqrt(z))*G(b)*G(2*a - b + 1),
        sqrt(z)*(sqrt(z)/2)**(-2*a + 1)*B(b, sqrt(z))*B(2*a - b,
        sqrt(z))*G(b)*G(2*a - b + 1), sqrt(z)*(sqrt(z)/2)**(-2*a + 1)*B(b - 1,
        sqrt(z))*B(2*a - b + 1, sqrt(z))*G(b)*G(2*a - b + 1),
        (sqrt(z)/2)**(-2*a + 1)*B(b, sqrt(z))*B(2*a - b + 1,
        sqrt(z))*G(b)*G(2*a - b + 1), 1, 0, S.Half, z/2, -b + 1, -2*a + b,
        -2*a))
    c = cse(t)
    ans = (
        [(x0, 2*a), (x1, -b + x0), (x2, x1 + 1), (x3, b - 1), (x4, sqrt(z)),
         (x5, B(x3, x4)), (x6, (x4/2)**(1 - x0)*G(b)*G(x2)), (x7, x6*B(x1, x4)),
         (x8, B(b, x4)), (x9, x6*B(x2, x4))],
        [(a, a + S.Half, x0, b, x2, x5*x7, x4*x7*x8, x4*x5*x9, x8*x9,
          1, 0, S.Half, z/2, -x3, -x1, -x0)])
    assert ans == c


def test_issue_6169():
    r = CRootOf(x**6 - 4*x**5 - 2, 1)
    assert cse(r) == ([], [r])
    # and a check that the right thing is done with the new
    # mechanism
    assert sub_post(sub_pre((-x - y)*z - x - y)) == -z*(x + y) - x - y


def test_cse_Indexed():
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    i = Idx('i', len_y-1)

    expr1 = (y[i+1]-y[i])/(x[i+1]-x[i])
    expr2 = 1/(x[i+1]-x[i])
    replacements, reduced_exprs = cse([expr1, expr2])
    assert len(replacements) > 0


def test_cse_MatrixSymbol():
    # MatrixSymbols have non-Basic args, so make sure that works
    A = MatrixSymbol("A", 3, 3)
    assert cse(A) == ([], [A])

    n = symbols('n', integer=True)
    B = MatrixSymbol("B", n, n)
    assert cse(B) == ([], [B])

    assert cse(A[0] * A[0]) == ([], [A[0]*A[0]])

    assert cse(A[0,0]*A[0,1] + A[0,0]*A[0,1]*A[0,2]) == ([(x0, A[0, 0]*A[0, 1])], [x0*A[0, 2] + x0])

def test_cse_MatrixExpr():
    A = MatrixSymbol('A', 3, 3)
    y = MatrixSymbol('y', 3, 1)

    expr1 = (A.T*A).I * A * y
    expr2 = (A.T*A) * A * y
    replacements, reduced_exprs = cse([expr1, expr2])
    assert len(replacements) > 0

    replacements, reduced_exprs = cse([expr1 + expr2, expr1])
    assert replacements

    replacements, reduced_exprs = cse([A**2, A + A**2])
    assert replacements


def test_Piecewise():
    f = Piecewise((-z + x*y, Eq(y, 0)), (-z - x*y, True))
    ans = cse(f)
    actual_ans = ([(x0, x*y)],
        [Piecewise((x0 - z, Eq(y, 0)), (-z - x0, True))])
    assert ans == actual_ans


def test_ignore_order_terms():
    eq = exp(x).series(x,0,3) + sin(y+x**3) - 1
    assert cse(eq) == ([], [sin(x**3 + y) + x + x**2/2 + O(x**3)])


def test_name_conflict():
    z1 = x0 + y
    z2 = x2 + x3
    l = [cos(z1) + z1, cos(z2) + z2, x0 + x2]
    substs, reduced = cse(l)
    assert [e.subs(reversed(substs)) for e in reduced] == l


def test_name_conflict_cust_symbols():
    z1 = x0 + y
    z2 = x2 + x3
    l = [cos(z1) + z1, cos(z2) + z2, x0 + x2]
    substs, reduced = cse(l, symbols("x:10"))
    assert [e.subs(reversed(substs)) for e in reduced] == l


def test_symbols_exhausted_error():
    l = cos(x+y)+x+y+cos(w+y)+sin(w+y)
    sym = [x, y, z]
    with raises(ValueError):
        cse(l, symbols=sym)


def test_issue_7840():
    # daveknippers' example
    C393 = sympify( \
        'Piecewise((C391 - 1.65, C390 < 0.5), (Piecewise((C391 - 1.65, \
        C391 > 2.35), (C392, True)), True))'
    )
    C391 = sympify( \
        'Piecewise((2.05*C390**(-1.03), C390 < 0.5), (2.5*C390**(-0.625), True))'
    )
    C393 = C393.subs('C391',C391)
    # simple substitution
    sub = {}
    sub['C390'] = 0.703451854
    sub['C392'] = 1.01417794
    ss_answer = C393.subs(sub)
    # cse
    substitutions,new_eqn = cse(C393)
    for pair in substitutions:
        sub[pair[0].name] = pair[1].subs(sub)
    cse_answer = new_eqn[0].subs(sub)
    # both methods should be the same
    assert ss_answer == cse_answer

    # GitRay's example
    expr = sympify(
        "Piecewise((Symbol('ON'), Equality(Symbol('mode'), Symbol('ON'))), \
        (Piecewise((Piecewise((Symbol('OFF'), StrictLessThan(Symbol('x'), \
        Symbol('threshold'))), (Symbol('ON'), true)), Equality(Symbol('mode'), \
        Symbol('AUTO'))), (Symbol('OFF'), true)), true))"
    )
    substitutions, new_eqn = cse(expr)
    # this Piecewise should be exactly the same
    assert new_eqn[0] == expr
    # there should not be any replacements
    assert len(substitutions) < 1


def test_issue_8891():
    for cls in (MutableDenseMatrix, MutableSparseMatrix,
            ImmutableDenseMatrix, ImmutableSparseMatrix):
        m = cls(2, 2, [x + y, 0, 0, 0])
        res = cse([x + y, m])
        ans = ([(x0, x + y)], [x0, cls([[x0, 0], [0, 0]])])
        assert res == ans
        assert isinstance(res[1][-1], cls)


def test_issue_11230():
    # a specific test that always failed
    a, b, f, k, l, i = symbols('a b f k l i')
    p = [a*b*f*k*l, a*i*k**2*l, f*i*k**2*l]
    R, C = cse(p)
    assert not any(i.is_Mul for a in C for i in a.args)

    # random tests for the issue
    from sympy.core.random import choice
    from sympy.core.function import expand_mul
    s = symbols('a:m')
    # 35 Mul tests, none of which should ever fail
    ex = [Mul(*[choice(s) for i in range(5)]) for i in range(7)]
    for p in subsets(ex, 3):
        p = list(p)
        R, C = cse(p)
        assert not any(i.is_Mul for a in C for i in a.args)
        for ri in reversed(R):
            for i in range(len(C)):
                C[i] = C[i].subs(*ri)
        assert p == C
    # 35 Add tests, none of which should ever fail
    ex = [Add(*[choice(s[:7]) for i in range(5)]) for i in range(7)]
    for p in subsets(ex, 3):
        p = list(p)
        R, C = cse(p)
        assert not any(i.is_Add for a in C for i in a.args)
        for ri in reversed(R):
            for i in range(len(C)):
                C[i] = C[i].subs(*ri)
        # use expand_mul to handle cases like this:
        # p = [a + 2*b + 2*e, 2*b + c + 2*e, b + 2*c + 2*g]
        # x0 = 2*(b + e) is identified giving a rebuilt p that
        # is now `[a + 2*(b + e), c + 2*(b + e), b + 2*c + 2*g]`
        assert p == [expand_mul(i) for i in C]


@XFAIL
def test_issue_11577():
    def check(eq):
        r, c = cse(eq)
        assert eq.count_ops() >= \
            len(r) + sum(i[1].count_ops() for i in r) + \
            count_ops(c)

    eq = x**5*y**2 + x**5*y + x**5
    assert cse(eq) == (
        [(x0, x**4), (x1, x*y)], [x**5 + x0*x1*y + x0*x1])
        # ([(x0, x**5*y)], [x0*y + x0 + x**5]) or
        # ([(x0, x**5)], [x0*y**2 + x0*y + x0])
    check(eq)

    eq = x**2/(y + 1)**2 + x/(y + 1)
    assert cse(eq) == (
        [(x0, y + 1)], [x**2/x0**2 + x/x0])
        # ([(x0, x/(y + 1))], [x0**2 + x0])
    check(eq)


def test_hollow_rejection():
    eq = [x + 3, x + 4]
    assert cse(eq) == ([], eq)


def test_cse_ignore():
    exprs = [exp(y)*(3*y + 3*sqrt(x+1)), exp(y)*(5*y + 5*sqrt(x+1))]
    subst1, red1 = cse(exprs)
    assert any(y in sub.free_symbols for _, sub in subst1), "cse failed to identify any term with y"

    subst2, red2 = cse(exprs, ignore=(y,))  # y is not allowed in substitutions
    assert not any(y in sub.free_symbols for _, sub in subst2), "Sub-expressions containing y must be ignored"
    assert any(sub - sqrt(x + 1) == 0 for _, sub in subst2), "cse failed to identify sqrt(x + 1) as sub-expression"


def test_cse_ignore_issue_15002():
    l = [
        w*exp(x)*exp(-z),
        exp(y)*exp(x)*exp(-z)
    ]
    substs, reduced = cse(l, ignore=(x,))
    rl = [e.subs(reversed(substs)) for e in reduced]
    assert rl == l


def test_cse_unevaluated():
    xp1 = UnevaluatedExpr(x + 1)
    # This used to cause RecursionError
    [(x0, ue)], [red] = cse([(-1 - xp1) / (1 - xp1)])
    if ue == xp1:
        assert red == (-1 - x0) / (1 - x0)
    elif ue == -xp1:
        assert red == (-1 + x0) / (1 + x0)
    else:
        msg = f'Expected common subexpression {xp1} or {-xp1}, instead got {ue}'
        assert False, msg


def test_cse__performance():
    nexprs, nterms = 3, 20
    x = symbols('x:%d' % nterms)
    exprs = [
        reduce(add, [x[j]*(-1)**(i+j) for j in range(nterms)])
        for i in range(nexprs)
    ]
    assert (exprs[0] + exprs[1]).simplify() == 0
    subst, red = cse(exprs)
    assert len(subst) > 0, "exprs[0] == -exprs[2], i.e. a CSE"
    for i, e in enumerate(red):
        assert (e.subs(reversed(subst)) - exprs[i]).simplify() == 0


def test_issue_12070():
    exprs = [x + y, 2 + x + y, x + y + z, 3 + x + y + z]
    subst, red = cse(exprs)
    assert 6 >= (len(subst) + sum(v.count_ops() for k, v in subst) +
                 count_ops(red))


def test_issue_13000():
    eq = x/(-4*x**2 + y**2)
    cse_eq = cse(eq)[1][0]
    assert cse_eq == eq


def test_issue_18203():
    eq = CRootOf(x**5 + 11*x - 2, 0) + CRootOf(x**5 + 11*x - 2, 1)
    assert cse(eq) == ([], [eq])


def test_unevaluated_mul():
    eq = Mul(x + y, x + y, evaluate=False)
    assert cse(eq) == ([(x0, x + y)], [x0**2])


def test_cse_release_variables():
    from sympy.simplify.cse_main import cse_release_variables
    _0, _1, _2, _3, _4 = symbols('_:5')
    eqs = [(x + y - 1)**2, x,
        x + y, (x + y)/(2*x + 1) + (x + y - 1)**2,
        (2*x + 1)**(x + y)]
    r, e = cse(eqs, postprocess=cse_release_variables)
    # this can change in keeping with the intention of the function
    assert r, e == ([
    (x0, x + y), (x1, (x0 - 1)**2), (x2, 2*x + 1),
    (_3, x0/x2 + x1), (_4, x2**x0), (x2, None), (_0, x1),
    (x1, None), (_2, x0), (x0, None), (_1, x)], (_0, _1, _2, _3, _4))
    r.reverse()
    r = [(s, v) for s, v in r if v is not None]
    assert eqs == [i.subs(r) for i in e]


def test_cse_list():
    _cse = lambda x: cse(x, list=False)
    assert _cse(x) == ([], x)
    assert _cse('x') == ([], 'x')
    it = [x]
    for c in (list, tuple, set):
        assert _cse(c(it)) == ([], c(it))
    #Tuple works different from tuple:
    assert _cse(Tuple(*it)) == ([], Tuple(*it))
    d = {x: 1}
    assert _cse(d) == ([], d)

def test_issue_18991():
    A = MatrixSymbol('A', 2, 2)
    assert signsimp(-A * A - A) == -A * A - A


def test_unevaluated_Mul():
    m = [Mul(1, 2, evaluate=False)]
    assert cse(m) == ([], m)


def test_cse_matrix_expression_inverse():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = Inverse(A)
    cse_expr = cse(x)
    assert cse_expr == ([], [Inverse(A)])


def test_cse_matrix_expression_matmul_inverse():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    b = ImmutableDenseMatrix(symbols('b:2'))
    x = MatMul(Inverse(A), b)
    cse_expr = cse(x)
    assert cse_expr == ([], [x])


def test_cse_matrix_negate_matrix():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = MatMul(S.NegativeOne, A)
    cse_expr = cse(x)
    assert cse_expr == ([], [x])


def test_cse_matrix_negate_matmul_not_extracted():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    B = ImmutableDenseMatrix(symbols('B:4')).reshape(2, 2)
    x = MatMul(S.NegativeOne, A, B)
    cse_expr = cse(x)
    assert cse_expr == ([], [x])


@XFAIL  # No simplification rule for nested associative operations
def test_cse_matrix_nested_matmul_collapsed():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    B = ImmutableDenseMatrix(symbols('B:4')).reshape(2, 2)
    x = MatMul(S.NegativeOne, MatMul(A, B))
    cse_expr = cse(x)
    assert cse_expr == ([], [MatMul(S.NegativeOne, A, B)])


def test_cse_matrix_optimize_out_single_argument_mul():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = MatMul(MatMul(MatMul(A)))
    cse_expr = cse(x)
    assert cse_expr == ([], [A])


@XFAIL  # Multiple simplification passed not supported in CSE
def test_cse_matrix_optimize_out_single_argument_mul_combined():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = MatAdd(MatMul(MatMul(MatMul(A))), MatMul(MatMul(A)), MatMul(A), A)
    cse_expr = cse(x)
    assert cse_expr == ([], [MatMul(4, A)])


def test_cse_matrix_optimize_out_single_argument_add():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = MatAdd(MatAdd(MatAdd(MatAdd(A))))
    cse_expr = cse(x)
    assert cse_expr == ([], [A])


@XFAIL  # Multiple simplification passed not supported in CSE
def test_cse_matrix_optimize_out_single_argument_add_combined():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    x = MatMul(MatAdd(MatAdd(MatAdd(A))), MatAdd(MatAdd(A)), MatAdd(A), A)
    cse_expr = cse(x)
    assert cse_expr == ([], [MatMul(4, A)])


def test_cse_matrix_expression_matrix_solve():
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    b = ImmutableDenseMatrix(symbols('b:2'))
    x = MatrixSolve(A, b)
    cse_expr = cse(x)
    assert cse_expr == ([], [x])


def test_cse_matrix_matrix_expression():
    X = ImmutableDenseMatrix(symbols('X:4')).reshape(2, 2)
    y = ImmutableDenseMatrix(symbols('y:2'))
    b = MatMul(Inverse(MatMul(Transpose(X), X)), Transpose(X), y)
    cse_expr = cse(b)
    x0 = MatrixSymbol('x0', 2, 2)
    reduced_expr_expected = MatMul(Inverse(MatMul(x0, X)), x0, y)
    assert cse_expr == ([(x0, Transpose(X))], [reduced_expr_expected])


def test_cse_matrix_kalman_filter():
    """Kalman Filter example from Matthew Rocklin's SciPy 2013 talk.

    Talk titled: "Matrix Expressions and BLAS/LAPACK; SciPy 2013 Presentation"

    Video: https://pyvideo.org/scipy-2013/matrix-expressions-and-blaslapack-scipy-2013-pr.html

    Notes
    =====

    Equations are:

    new_mu = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
           = MatAdd(mu, MatMul(Sigma, Transpose(H), Inverse(MatAdd(R, MatMul(H, Sigma, Transpose(H)))), MatAdd(MatMul(H, mu), MatMul(S.NegativeOne, data))))
    new_Sigma = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma
              = MatAdd(Sigma, MatMul(S.NegativeOne, Sigma, Transpose(H)), Inverse(MatAdd(R, MatMul(H*Sigma*Transpose(H)))), H, Sigma))

    """
    N = 2
    mu = ImmutableDenseMatrix(symbols(f'mu:{N}'))
    Sigma = ImmutableDenseMatrix(symbols(f'Sigma:{N * N}')).reshape(N, N)
    H = ImmutableDenseMatrix(symbols(f'H:{N * N}')).reshape(N, N)
    R = ImmutableDenseMatrix(symbols(f'R:{N * N}')).reshape(N, N)
    data = ImmutableDenseMatrix(symbols(f'data:{N}'))
    new_mu = MatAdd(mu, MatMul(Sigma, Transpose(H), Inverse(MatAdd(R, MatMul(H, Sigma, Transpose(H)))), MatAdd(MatMul(H, mu), MatMul(S.NegativeOne, data))))
    new_Sigma = MatAdd(Sigma, MatMul(S.NegativeOne, Sigma, Transpose(H), Inverse(MatAdd(R, MatMul(H, Sigma, Transpose(H)))), H, Sigma))
    cse_expr = cse([new_mu, new_Sigma])
    x0 = MatrixSymbol('x0', N, N)
    x1 = MatrixSymbol('x1', N, N)
    replacements_expected = [
        (x0, Transpose(H)),
        (x1, Inverse(MatAdd(R, MatMul(H, Sigma, x0)))),
    ]
    reduced_exprs_expected = [
        MatAdd(mu, MatMul(Sigma, x0, x1, MatAdd(MatMul(H, mu), MatMul(S.NegativeOne, data)))),
        MatAdd(Sigma, MatMul(S.NegativeOne, Sigma, x0, x1, H, Sigma)),
    ]
    assert cse_expr == (replacements_expected, reduced_exprs_expected)
