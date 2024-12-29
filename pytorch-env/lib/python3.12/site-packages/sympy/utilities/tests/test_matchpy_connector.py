import pickle

from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar, Replacer

matchpy = import_module("matchpy")

x, y, z = symbols("x y z")


def _get_first_match(expr, pattern):
    from matchpy import ManyToOneMatcher, Pattern

    matcher = ManyToOneMatcher()
    matcher.add(Pattern(pattern))
    return next(iter(matcher.match(expr)))


def test_matchpy_connector():
    if matchpy is None:
        skip("matchpy not installed")

    from multiset import Multiset
    from matchpy import Pattern, Substitution

    w_ = WildDot("w_")
    w__ = WildPlus("w__")
    w___ = WildStar("w___")

    expr = x + y
    pattern = x + w_
    p, subst = _get_first_match(expr, pattern)
    assert p == Pattern(pattern)
    assert subst == Substitution({'w_': y})

    expr = x + y + z
    pattern = x + w__
    p, subst = _get_first_match(expr, pattern)
    assert p == Pattern(pattern)
    assert subst == Substitution({'w__': Multiset([y, z])})

    expr = x + y + z
    pattern = x + y + z + w___
    p, subst = _get_first_match(expr, pattern)
    assert p == Pattern(pattern)
    assert subst == Substitution({'w___': Multiset()})


def test_matchpy_optional():
    if matchpy is None:
        skip("matchpy not installed")

    from matchpy import Pattern, Substitution
    from matchpy import ManyToOneReplacer, ReplacementRule

    p = WildDot("p", optional=1)
    q = WildDot("q", optional=0)

    pattern = p*x + q

    expr1 = 2*x
    pa, subst = _get_first_match(expr1, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': 2, 'q': 0})

    expr2 = x + 3
    pa, subst = _get_first_match(expr2, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': 1, 'q': 3})

    expr3 = x
    pa, subst = _get_first_match(expr3, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': 1, 'q': 0})

    expr4 = x*y + z
    pa, subst = _get_first_match(expr4, pattern)
    assert pa == Pattern(pattern)
    assert subst == Substitution({'p': y, 'q': z})

    replacer = ManyToOneReplacer()
    replacer.add(ReplacementRule(Pattern(pattern), lambda p, q: sin(p)*cos(q)))
    assert replacer.replace(expr1) == sin(2)*cos(0)
    assert replacer.replace(expr2) == sin(1)*cos(3)
    assert replacer.replace(expr3) == sin(1)*cos(0)
    assert replacer.replace(expr4) == sin(y)*cos(z)


def test_replacer():
    if matchpy is None:
        skip("matchpy not installed")

    for info in [True, False]:
        for lambdify in [True, False]:
            _perform_test_replacer(info, lambdify)


def _perform_test_replacer(info, lambdify):

    x1_ = WildDot("x1_")
    x2_ = WildDot("x2_")

    a_ = WildDot("a_", optional=S.One)
    b_ = WildDot("b_", optional=S.One)
    c_ = WildDot("c_", optional=S.Zero)

    replacer = Replacer(common_constraints=[
        matchpy.CustomConstraint(lambda a_: not a_.has(x)),
        matchpy.CustomConstraint(lambda b_: not b_.has(x)),
        matchpy.CustomConstraint(lambda c_: not c_.has(x)),
    ], lambdify=lambdify, info=info)

    # Rewrite the equation into implicit form, unless it's already solved:
    replacer.add(Eq(x1_, x2_), Eq(x1_ - x2_, 0), conditions_nonfalse=[Ne(x2_, 0), Ne(x1_, 0), Ne(x1_, x), Ne(x2_, x)], info=1)

    # Simple equation solver for real numbers:
    replacer.add(Eq(a_*x + b_, 0), Eq(x, -b_/a_), info=2)
    disc = b_**2 - 4*a_*c_
    replacer.add(
        Eq(a_*x**2 + b_*x + c_, 0),
        Eq(x, (-b_ - sqrt(disc))/(2*a_)) | Eq(x, (-b_ + sqrt(disc))/(2*a_)),
        conditions_nonfalse=[disc >= 0],
        info=3
    )
    replacer.add(
        Eq(a_*x**2 + c_, 0),
        Eq(x, sqrt(-c_/a_)) | Eq(x, -sqrt(-c_/a_)),
        conditions_nonfalse=[-c_*a_ > 0],
        info=4
    )

    g = lambda expr, infos: (expr, infos) if info else expr

    assert replacer.replace(Eq(3*x, y)) == g(Eq(x, y/3), [1, 2])
    assert replacer.replace(Eq(x**2 + 1, 0)) == g(Eq(x**2 + 1, 0), [])
    assert replacer.replace(Eq(x**2, 4)) == g((Eq(x, 2) | Eq(x, -2)), [1, 4])
    assert replacer.replace(Eq(x**2 + 4*y*x + 4*y**2, 0)) == g(Eq(x, -2*y), [3])


def test_matchpy_object_pickle():
    if matchpy is None:
        return

    a1 = WildDot("a")
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2

    a1 = WildDot("a", S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2

    a1 = WildPlus("a", S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2

    a1 = WildStar("a", S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2
