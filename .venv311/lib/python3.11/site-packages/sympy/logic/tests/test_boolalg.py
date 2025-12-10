from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.numbers import oo
from sympy.core.relational import Equality, Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.sets.sets import Interval, Union
from sympy.sets.contains import Contains
from sympy.simplify.simplify import simplify
from sympy.logic.boolalg import (
    And, Boolean, Equivalent, ITE, Implies, Nand, Nor, Not, Or,
    POSform, SOPform, Xor, Xnor, conjuncts, disjuncts,
    distribute_or_over_and, distribute_and_over_or,
    eliminate_implications, is_nnf, is_cnf, is_dnf, simplify_logic,
    to_nnf, to_cnf, to_dnf, to_int_repr, bool_map, true, false,
    BooleanAtom, is_literal, term_to_integer,
    truth_table, as_Boolean, to_anf, is_anf, distribute_xor_over_and,
    anf_coeffs, ANFform, bool_minterm, bool_maxterm, bool_monomial,
    _check_pair, _convert_to_varsSOP, _convert_to_varsPOS, Exclusive,
    gateinputcount)
from sympy.assumptions.cnf import CNF

from sympy.testing.pytest import raises, XFAIL, slow

from itertools import combinations, permutations, product

A, B, C, D = symbols('A:D')
a, b, c, d, e, w, x, y, z = symbols('a:e w:z')


def test_overloading():
    """Test that |, & are overloaded as expected"""

    assert A & B == And(A, B)
    assert A | B == Or(A, B)
    assert (A & B) | C == Or(And(A, B), C)
    assert A >> B == Implies(A, B)
    assert A << B == Implies(B, A)
    assert ~A == Not(A)
    assert A ^ B == Xor(A, B)


def test_And():
    assert And() is true
    assert And(A) == A
    assert And(True) is true
    assert And(False) is false
    assert And(True, True) is true
    assert And(True, False) is false
    assert And(False, False) is false
    assert And(True, A) == A
    assert And(False, A) is false
    assert And(True, True, True) is true
    assert And(True, True, A) == A
    assert And(True, False, A) is false
    assert And(1, A) == A
    raises(TypeError, lambda: And(2, A))
    assert And(A < 1, A >= 1) is false
    e = A > 1
    assert And(e, e.canonical) == e.canonical
    g, l, ge, le = A > B, B < A, A >= B, B <= A
    assert And(g, l, ge, le) == And(ge, g)
    assert {And(*i) for i in permutations((l, g, le, ge))} == {And(ge, g)}
    assert And(And(Eq(a, 0), Eq(b, 0)), And(Ne(a, 0), Eq(c, 0))) is false


def test_Or():
    assert Or() is false
    assert Or(A) == A
    assert Or(True) is true
    assert Or(False) is false
    assert Or(True, True) is true
    assert Or(True, False) is true
    assert Or(False, False) is false
    assert Or(True, A) is true
    assert Or(False, A) == A
    assert Or(True, False, False) is true
    assert Or(True, False, A) is true
    assert Or(False, False, A) == A
    assert Or(1, A) is true
    raises(TypeError, lambda: Or(2, A))
    assert Or(A < 1, A >= 1) is true
    e = A > 1
    assert Or(e, e.canonical) == e
    g, l, ge, le = A > B, B < A, A >= B, B <= A
    assert Or(g, l, ge, le) == Or(g, ge)


def test_Xor():
    assert Xor() is false
    assert Xor(A) == A
    assert Xor(A, A) is false
    assert Xor(True, A, A) is true
    assert Xor(A, A, A, A, A) == A
    assert Xor(True, False, False, A, B) == ~Xor(A, B)
    assert Xor(True) is true
    assert Xor(False) is false
    assert Xor(True, True) is false
    assert Xor(True, False) is true
    assert Xor(False, False) is false
    assert Xor(True, A) == ~A
    assert Xor(False, A) == A
    assert Xor(True, False, False) is true
    assert Xor(True, False, A) == ~A
    assert Xor(False, False, A) == A
    assert isinstance(Xor(A, B), Xor)
    assert Xor(A, B, Xor(C, D)) == Xor(A, B, C, D)
    assert Xor(A, B, Xor(B, C)) == Xor(A, C)
    assert Xor(A < 1, A >= 1, B) == Xor(0, 1, B) == Xor(1, 0, B)
    e = A > 1
    assert Xor(e, e.canonical) == Xor(0, 0) == Xor(1, 1)


def test_rewrite_as_And():
    expr = x ^ y
    assert expr.rewrite(And) == (x | y) & (~x | ~y)


def test_rewrite_as_Or():
    expr = x ^ y
    assert expr.rewrite(Or) == (x & ~y) | (y & ~x)


def test_rewrite_as_Nand():
    expr = (y & z) | (z & ~w)
    assert expr.rewrite(Nand) == ~(~(y & z) & ~(z & ~w))


def test_rewrite_as_Nor():
    expr = z & (y | ~w)
    assert expr.rewrite(Nor) == ~(~z | ~(y | ~w))


def test_Not():
    raises(TypeError, lambda: Not(True, False))
    assert Not(True) is false
    assert Not(False) is true
    assert Not(0) is true
    assert Not(1) is false
    assert Not(2) is false


def test_Nand():
    assert Nand() is false
    assert Nand(A) == ~A
    assert Nand(True) is false
    assert Nand(False) is true
    assert Nand(True, True) is false
    assert Nand(True, False) is true
    assert Nand(False, False) is true
    assert Nand(True, A) == ~A
    assert Nand(False, A) is true
    assert Nand(True, True, True) is false
    assert Nand(True, True, A) == ~A
    assert Nand(True, False, A) is true


def test_Nor():
    assert Nor() is true
    assert Nor(A) == ~A
    assert Nor(True) is false
    assert Nor(False) is true
    assert Nor(True, True) is false
    assert Nor(True, False) is false
    assert Nor(False, False) is true
    assert Nor(True, A) is false
    assert Nor(False, A) == ~A
    assert Nor(True, True, True) is false
    assert Nor(True, True, A) is false
    assert Nor(True, False, A) is false


def test_Xnor():
    assert Xnor() is true
    assert Xnor(A) == ~A
    assert Xnor(A, A) is true
    assert Xnor(True, A, A) is false
    assert Xnor(A, A, A, A, A) == ~A
    assert Xnor(True) is false
    assert Xnor(False) is true
    assert Xnor(True, True) is true
    assert Xnor(True, False) is false
    assert Xnor(False, False) is true
    assert Xnor(True, A) == A
    assert Xnor(False, A) == ~A
    assert Xnor(True, False, False) is false
    assert Xnor(True, False, A) == A
    assert Xnor(False, False, A) == ~A


def test_Implies():
    raises(ValueError, lambda: Implies(A, B, C))
    assert Implies(True, True) is true
    assert Implies(True, False) is false
    assert Implies(False, True) is true
    assert Implies(False, False) is true
    assert Implies(0, A) is true
    assert Implies(1, 1) is true
    assert Implies(1, 0) is false
    assert A >> B == B << A
    assert (A < 1) >> (A >= 1) == (A >= 1)
    assert (A < 1) >> (S.One > A) is true
    assert A >> A is true


def test_Equivalent():
    assert Equivalent(A, B) == Equivalent(B, A) == Equivalent(A, B, A)
    assert Equivalent() is true
    assert Equivalent(A, A) == Equivalent(A) is true
    assert Equivalent(True, True) == Equivalent(False, False) is true
    assert Equivalent(True, False) == Equivalent(False, True) is false
    assert Equivalent(A, True) == A
    assert Equivalent(A, False) == Not(A)
    assert Equivalent(A, B, True) == A & B
    assert Equivalent(A, B, False) == ~A & ~B
    assert Equivalent(1, A) == A
    assert Equivalent(0, A) == Not(A)
    assert Equivalent(A, Equivalent(B, C)) != Equivalent(Equivalent(A, B), C)
    assert Equivalent(A < 1, A >= 1) is false
    assert Equivalent(A < 1, A >= 1, 0) is false
    assert Equivalent(A < 1, A >= 1, 1) is false
    assert Equivalent(A < 1, S.One > A) == Equivalent(1, 1) == Equivalent(0, 0)
    assert Equivalent(Equality(A, B), Equality(B, A)) is true


def test_Exclusive():
    assert Exclusive(False, False, False) is true
    assert Exclusive(True, False, False) is true
    assert Exclusive(True, True, False) is false
    assert Exclusive(True, True, True) is false


def test_equals():
    assert Not(Or(A, B)).equals(And(Not(A), Not(B))) is True
    assert Equivalent(A, B).equals((A >> B) & (B >> A)) is True
    assert ((A | ~B) & (~A | B)).equals((~A & ~B) | (A & B)) is True
    assert (A >> B).equals(~A >> ~B) is False
    assert (A >> (B >> A)).equals(A >> (C >> A)) is False
    raises(NotImplementedError, lambda: (A & B).equals(A > B))


def test_simplification_boolalg():
    """
    Test working of simplification methods.
    """
    set1 = [[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]]
    set2 = [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]
    assert SOPform([x, y, z], set1) == Or(And(Not(x), z), And(Not(z), x))
    assert Not(SOPform([x, y, z], set2)) == \
           Not(Or(And(Not(x), Not(z)), And(x, z)))
    assert POSform([x, y, z], set1 + set2) is true
    assert SOPform([x, y, z], set1 + set2) is true
    assert SOPform([Dummy(), Dummy(), Dummy()], set1 + set2) is true

    minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1],
                [1, 1, 1, 1]]
    dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x))))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)

    minterms = [1, 3, 7, 11, 15]
    dontcares = [0, 2, 5]
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x))))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)

    minterms = [1, [0, 0, 1, 1], 7, [1, 0, 1, 1],
                [1, 1, 1, 1]]
    dontcares = [0, [0, 0, 1, 0], 5]
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x))))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)

    minterms = [1, {y: 1, z: 1}]
    dontcares = [0, [0, 0, 1, 0], 5]
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x))))
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)

    minterms = [{y: 1, z: 1}, 1]
    dontcares = [[0, 0, 0, 0]]

    minterms = [[0, 0, 0]]
    raises(ValueError, lambda: SOPform([w, x, y, z], minterms))
    raises(ValueError, lambda: POSform([w, x, y, z], minterms))

    raises(TypeError, lambda: POSform([w, x, y, z], ["abcdefg"]))

    # test simplification
    ans = And(A, Or(B, C))
    assert simplify_logic(A & (B | C)) == ans
    assert simplify_logic((A & B) | (A & C)) == ans
    assert simplify_logic(Implies(A, B)) == Or(Not(A), B)
    assert simplify_logic(Equivalent(A, B)) == \
           Or(And(A, B), And(Not(A), Not(B)))
    assert simplify_logic(And(Equality(A, 2), C)) == And(Equality(A, 2), C)
    assert simplify_logic(And(Equality(A, 2), A)) == And(Equality(A, 2), A)
    assert simplify_logic(And(Equality(A, B), C)) == And(Equality(A, B), C)
    assert simplify_logic(Or(And(Equality(A, 3), B), And(Equality(A, 3), C))) \
           == And(Equality(A, 3), Or(B, C))
    b = (~x & ~y & ~z) | (~x & ~y & z)
    e = And(A, b)
    assert simplify_logic(e) == A & ~x & ~y
    raises(ValueError, lambda: simplify_logic(A & (B | C), form='blabla'))
    assert simplify(Or(x <= y, And(x < y, z))) == (x <= y)
    assert simplify(Or(x <= y, And(y > x, z))) == (x <= y)
    assert simplify(Or(x >= y, And(y < x, z))) == (x >= y)

    # Check that expressions with nine variables or more are not simplified
    # (without the force-flag)
    a, b, c, d, e, f, g, h, j = symbols('a b c d e f g h j')
    expr = a & b & c & d & e & f & g & h & j | \
           a & b & c & d & e & f & g & h & ~j
    # This expression can be simplified to get rid of the j variables
    assert simplify_logic(expr) == expr

    # Test dontcare
    assert simplify_logic((a & b) | c | d, dontcare=(a & b)) == c | d

    # check input
    ans = SOPform([x, y], [[1, 0]])
    assert SOPform([x, y], [[1, 0]]) == ans
    assert POSform([x, y], [[1, 0]]) == ans

    raises(ValueError, lambda: SOPform([x], [[1]], [[1]]))
    assert SOPform([x], [[1]], [[0]]) is true
    assert SOPform([x], [[0]], [[1]]) is true
    assert SOPform([x], [], []) is false

    raises(ValueError, lambda: POSform([x], [[1]], [[1]]))
    assert POSform([x], [[1]], [[0]]) is true
    assert POSform([x], [[0]], [[1]]) is true
    assert POSform([x], [], []) is false

    # check working of simplify
    assert simplify((A & B) | (A & C)) == And(A, Or(B, C))
    assert simplify(And(x, Not(x))) == False
    assert simplify(Or(x, Not(x))) == True
    assert simplify(And(Eq(x, 0), Eq(x, y))) == And(Eq(x, 0), Eq(y, 0))
    assert And(Eq(x - 1, 0), Eq(x, y)).simplify() == And(Eq(x, 1), Eq(y, 1))
    assert And(Ne(x - 1, 0), Ne(x, y)).simplify() == And(Ne(x, 1), Ne(x, y))
    assert And(Eq(x - 1, 0), Ne(x, y)).simplify() == And(Eq(x, 1), Ne(y, 1))
    assert And(Eq(x - 1, 0), Eq(x, z + y), Eq(y + x, 0)).simplify(
    ) == And(Eq(x, 1), Eq(y, -1), Eq(z, 2))
    assert And(Eq(x - 1, 0), Eq(x + 2, 3)).simplify() == Eq(x, 1)
    assert And(Ne(x - 1, 0), Ne(x + 2, 3)).simplify() == Ne(x, 1)
    assert And(Eq(x - 1, 0), Eq(x + 2, 2)).simplify() == False
    assert And(Ne(x - 1, 0), Ne(x + 2, 2)).simplify(
    ) == And(Ne(x, 1), Ne(x, 0))
    assert simplify(Xor(x, ~x)) == True


def test_bool_map():
    """
    Test working of bool_map function.
    """

    minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1],
                [1, 1, 1, 1]]
    assert bool_map(Not(Not(a)), a) == (a, {a: a})
    assert bool_map(SOPform([w, x, y, z], minterms),
                    POSform([w, x, y, z], minterms)) == \
           (And(Or(Not(w), y), Or(Not(x), y), z), {x: x, w: w, z: z, y: y})
    assert bool_map(SOPform([x, z, y], [[1, 0, 1]]),
                    SOPform([a, b, c], [[1, 0, 1]])) != False
    function1 = SOPform([x, z, y], [[1, 0, 1], [0, 0, 1]])
    function2 = SOPform([a, b, c], [[1, 0, 1], [1, 0, 0]])
    assert bool_map(function1, function2) == \
           (function1, {y: a, z: b})
    assert bool_map(Xor(x, y), ~Xor(x, y)) == False
    assert bool_map(And(x, y), Or(x, y)) is None
    assert bool_map(And(x, y), And(x, y, z)) is None
    # issue 16179
    assert bool_map(Xor(x, y, z), ~Xor(x, y, z)) == False
    assert bool_map(Xor(a, x, y, z), ~Xor(a, x, y, z)) == False


def test_bool_symbol():
    """Test that mixing symbols with boolean values
    works as expected"""

    assert And(A, True) == A
    assert And(A, True, True) == A
    assert And(A, False) is false
    assert And(A, True, False) is false
    assert Or(A, True) is true
    assert Or(A, False) == A


def test_is_boolean():
    assert isinstance(True, Boolean) is False
    assert isinstance(true, Boolean) is True
    assert 1 == True
    assert 1 != true
    assert (1 == true) is False
    assert 0 == False
    assert 0 != false
    assert (0 == false) is False
    assert true.is_Boolean is True
    assert (A & B).is_Boolean
    assert (A | B).is_Boolean
    assert (~A).is_Boolean
    assert (A ^ B).is_Boolean
    assert A.is_Boolean != isinstance(A, Boolean)
    assert isinstance(A, Boolean)


def test_subs():
    assert (A & B).subs(A, True) == B
    assert (A & B).subs(A, False) is false
    assert (A & B).subs(B, True) == A
    assert (A & B).subs(B, False) is false
    assert (A & B).subs({A: True, B: True}) is true
    assert (A | B).subs(A, True) is true
    assert (A | B).subs(A, False) == B
    assert (A | B).subs(B, True) is true
    assert (A | B).subs(B, False) == A
    assert (A | B).subs({A: True, B: True}) is true


"""
we test for axioms of boolean algebra
see https://en.wikipedia.org/wiki/Boolean_algebra_(structure)
"""


def test_commutative():
    """Test for commutativity of And and Or"""
    A, B = map(Boolean, symbols('A,B'))

    assert A & B == B & A
    assert A | B == B | A


def test_and_associativity():
    """Test for associativity of And"""

    assert (A & B) & C == A & (B & C)


def test_or_assicativity():
    assert ((A | B) | C) == (A | (B | C))


def test_double_negation():
    a = Boolean()
    assert ~(~a) == a


# test methods

def test_eliminate_implications():
    assert eliminate_implications(Implies(A, B, evaluate=False)) == (~A) | B
    assert eliminate_implications(
        A >> (C >> Not(B))) == Or(Or(Not(B), Not(C)), Not(A))
    assert eliminate_implications(Equivalent(A, B, C, D)) == \
           (~A | B) & (~B | C) & (~C | D) & (~D | A)


def test_conjuncts():
    assert conjuncts(A & B & C) == {A, B, C}
    assert conjuncts((A | B) & C) == {A | B, C}
    assert conjuncts(A) == {A}
    assert conjuncts(True) == {True}
    assert conjuncts(False) == {False}


def test_disjuncts():
    assert disjuncts(A | B | C) == {A, B, C}
    assert disjuncts((A | B) & C) == {(A | B) & C}
    assert disjuncts(A) == {A}
    assert disjuncts(True) == {True}
    assert disjuncts(False) == {False}


def test_distribute():
    assert distribute_and_over_or(Or(And(A, B), C)) == And(Or(A, C), Or(B, C))
    assert distribute_or_over_and(And(A, Or(B, C))) == Or(And(A, B), And(A, C))
    assert distribute_xor_over_and(And(A, Xor(B, C))) == Xor(And(A, B), And(A, C))


def test_to_anf():
    x, y, z = symbols('x,y,z')
    assert to_anf(And(x, y)) == And(x, y)
    assert to_anf(Or(x, y)) == Xor(x, y, And(x, y))
    assert to_anf(Or(Implies(x, y), And(x, y), y)) == \
           Xor(x, True, x & y, remove_true=False)
    assert to_anf(Or(Nand(x, y), Nor(x, y), Xnor(x, y), Implies(x, y))) == True
    assert to_anf(Or(x, Not(y), Nor(x, z), And(x, y), Nand(y, z))) == \
           Xor(True, And(y, z), And(x, y, z), remove_true=False)
    assert to_anf(Xor(x, y)) == Xor(x, y)
    assert to_anf(Not(x)) == Xor(x, True, remove_true=False)
    assert to_anf(Nand(x, y)) == Xor(True, And(x, y), remove_true=False)
    assert to_anf(Nor(x, y)) == Xor(x, y, True, And(x, y), remove_true=False)
    assert to_anf(Implies(x, y)) == Xor(x, True, And(x, y), remove_true=False)
    assert to_anf(Equivalent(x, y)) == Xor(x, y, True, remove_true=False)
    assert to_anf(Nand(x | y, x >> y), deep=False) == \
           Xor(True, And(Or(x, y), Implies(x, y)), remove_true=False)
    assert to_anf(Nor(x ^ y, x & y), deep=False) == \
           Xor(True, Or(Xor(x, y), And(x, y)), remove_true=False)
    # issue 25218
    assert to_anf(x ^ ~(x ^ y ^ ~y)) == False


def test_to_nnf():
    assert to_nnf(true) is true
    assert to_nnf(false) is false
    assert to_nnf(A) == A
    assert to_nnf(A | ~A | B) is true
    assert to_nnf(A & ~A & B) is false
    assert to_nnf(A >> B) == ~A | B
    assert to_nnf(Equivalent(A, B, C)) == (~A | B) & (~B | C) & (~C | A)
    assert to_nnf(A ^ B ^ C) == \
           (A | B | C) & (~A | ~B | C) & (A | ~B | ~C) & (~A | B | ~C)
    assert to_nnf(ITE(A, B, C)) == (~A | B) & (A | C)
    assert to_nnf(Not(A | B | C)) == ~A & ~B & ~C
    assert to_nnf(Not(A & B & C)) == ~A | ~B | ~C
    assert to_nnf(Not(A >> B)) == A & ~B
    assert to_nnf(Not(Equivalent(A, B, C))) == And(Or(A, B, C), Or(~A, ~B, ~C))
    assert to_nnf(Not(A ^ B ^ C)) == \
           (~A | B | C) & (A | ~B | C) & (A | B | ~C) & (~A | ~B | ~C)
    assert to_nnf(Not(ITE(A, B, C))) == (~A | ~B) & (A | ~C)
    assert to_nnf((A >> B) ^ (B >> A)) == (A & ~B) | (~A & B)
    assert to_nnf((A >> B) ^ (B >> A), False) == \
           (~A | ~B | A | B) & ((A & ~B) | (~A & B))
    assert ITE(A, 1, 0).to_nnf() == A
    assert ITE(A, 0, 1).to_nnf() == ~A
    # although ITE can hold non-Boolean, it will complain if
    # an attempt is made to convert the ITE to Boolean nnf
    raises(TypeError, lambda: ITE(A < 1, [1], B).to_nnf())


def test_to_cnf():
    assert to_cnf(~(B | C)) == And(Not(B), Not(C))
    assert to_cnf((A & B) | C) == And(Or(A, C), Or(B, C))
    assert to_cnf(A >> B) == (~A) | B
    assert to_cnf(A >> (B & C)) == (~A | B) & (~A | C)
    assert to_cnf(A & (B | C) | ~A & (B | C), True) == B | C
    assert to_cnf(A & B) == And(A, B)

    assert to_cnf(Equivalent(A, B)) == And(Or(A, Not(B)), Or(B, Not(A)))
    assert to_cnf(Equivalent(A, B & C)) == \
           (~A | B) & (~A | C) & (~B | ~C | A)
    assert to_cnf(Equivalent(A, B | C), True) == \
           And(Or(Not(B), A), Or(Not(C), A), Or(B, C, Not(A)))
    assert to_cnf(A + 1) == A + 1


def test_issue_18904():
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = symbols('x1:16')
    eq = ((x1 & x2 & x3 & x4 & x5 & x6 & x7 & x8 & x9) |
          (x1 & x2 & x3 & x4 & x5 & x6 & x7 & x10 & x9) |
          (x1 & x11 & x3 & x12 & x5 & x13 & x14 & x15 & x9))
    assert is_cnf(to_cnf(eq))
    raises(ValueError, lambda: to_cnf(eq, simplify=True))
    for f, t in zip((And, Or), (to_cnf, to_dnf)):
        eq = f(x1, x2, x3, x4, x5, x6, x7, x8, x9)
        raises(ValueError, lambda: to_cnf(eq, simplify=True))
        assert t(eq, simplify=True, force=True) == eq


def test_issue_9949():
    assert is_cnf(to_cnf((b > -5) | (a > 2) & (a < 4)))


def test_to_CNF():
    assert CNF.CNF_to_cnf(CNF.to_CNF(~(B | C))) == to_cnf(~(B | C))
    assert CNF.CNF_to_cnf(CNF.to_CNF((A & B) | C)) == to_cnf((A & B) | C)
    assert CNF.CNF_to_cnf(CNF.to_CNF(A >> B)) == to_cnf(A >> B)
    assert CNF.CNF_to_cnf(CNF.to_CNF(A >> (B & C))) == to_cnf(A >> (B & C))
    assert CNF.CNF_to_cnf(CNF.to_CNF(A & (B | C) | ~A & (B | C))) == to_cnf(A & (B | C) | ~A & (B | C))
    assert CNF.CNF_to_cnf(CNF.to_CNF(A & B)) == to_cnf(A & B)


def test_to_dnf():
    assert to_dnf(~(B | C)) == And(Not(B), Not(C))
    assert to_dnf(A & (B | C)) == Or(And(A, B), And(A, C))
    assert to_dnf(A >> B) == (~A) | B
    assert to_dnf(A >> (B & C)) == (~A) | (B & C)
    assert to_dnf(A | B) == A | B

    assert to_dnf(Equivalent(A, B), True) == \
           Or(And(A, B), And(Not(A), Not(B)))
    assert to_dnf(Equivalent(A, B & C), True) == \
           Or(And(A, B, C), And(Not(A), Not(B)), And(Not(A), Not(C)))
    assert to_dnf(A + 1) == A + 1


def test_to_int_repr():
    x, y, z = map(Boolean, symbols('x,y,z'))

    def sorted_recursive(arg):
        try:
            return sorted(sorted_recursive(x) for x in arg)
        except TypeError:  # arg is not a sequence
            return arg

    assert sorted_recursive(to_int_repr([x | y, z | x], [x, y, z])) == \
           sorted_recursive([[1, 2], [1, 3]])
    assert sorted_recursive(to_int_repr([x | y, z | ~x], [x, y, z])) == \
           sorted_recursive([[1, 2], [3, -1]])


def test_is_anf():
    x, y = symbols('x,y')
    assert is_anf(true) is True
    assert is_anf(false) is True
    assert is_anf(x) is True
    assert is_anf(And(x, y)) is True
    assert is_anf(Xor(x, y, And(x, y))) is True
    assert is_anf(Xor(x, y, Or(x, y))) is False
    assert is_anf(Xor(Not(x), y)) is False


def test_is_nnf():
    assert is_nnf(true) is True
    assert is_nnf(A) is True
    assert is_nnf(~A) is True
    assert is_nnf(A & B) is True
    assert is_nnf((A & B) | (~A & A) | (~B & B) | (~A & ~B), False) is True
    assert is_nnf((A | B) & (~A | ~B)) is True
    assert is_nnf(Not(Or(A, B))) is False
    assert is_nnf(A ^ B) is False
    assert is_nnf((A & B) | (~A & A) | (~B & B) | (~A & ~B), True) is False


def test_is_cnf():
    assert is_cnf(x) is True
    assert is_cnf(x | y | z) is True
    assert is_cnf(x & y & z) is True
    assert is_cnf((x | y) & z) is True
    assert is_cnf((x & y) | z) is False
    assert is_cnf(~(x & y) | z) is False


def test_is_dnf():
    assert is_dnf(x) is True
    assert is_dnf(x | y | z) is True
    assert is_dnf(x & y & z) is True
    assert is_dnf((x & y) | z) is True
    assert is_dnf((x | y) & z) is False
    assert is_dnf(~(x | y) & z) is False


def test_ITE():
    A, B, C = symbols('A:C')
    assert ITE(True, False, True) is false
    assert ITE(True, True, False) is true
    assert ITE(False, True, False) is false
    assert ITE(False, False, True) is true
    assert isinstance(ITE(A, B, C), ITE)

    A = True
    assert ITE(A, B, C) == B
    A = False
    assert ITE(A, B, C) == C
    B = True
    assert ITE(And(A, B), B, C) == C
    assert ITE(Or(A, False), And(B, True), False) is false
    assert ITE(x, A, B) == Not(x)
    assert ITE(x, B, A) == x
    assert ITE(1, x, y) == x
    assert ITE(0, x, y) == y
    raises(TypeError, lambda: ITE(2, x, y))
    raises(TypeError, lambda: ITE(1, [], y))
    raises(TypeError, lambda: ITE(1, (), y))
    raises(TypeError, lambda: ITE(1, y, []))
    assert ITE(1, 1, 1) is S.true
    assert isinstance(ITE(1, 1, 1, evaluate=False), ITE)

    assert ITE(Eq(x, True), y, x) == ITE(x, y, x)
    assert ITE(Eq(x, False), y, x) == ITE(~x, y, x)
    assert ITE(Ne(x, True), y, x) == ITE(~x, y, x)
    assert ITE(Ne(x, False), y, x) == ITE(x, y, x)
    assert ITE(Eq(S.true, x), y, x) == ITE(x, y, x)
    assert ITE(Eq(S.false, x), y, x) == ITE(~x, y, x)
    assert ITE(Ne(S.true, x), y, x) == ITE(~x, y, x)
    assert ITE(Ne(S.false, x), y, x) == ITE(x, y, x)
    # 0 and 1 in the context are not treated as True/False
    # so the equality must always be False since dissimilar
    # objects cannot be equal
    assert ITE(Eq(x, 0), y, x) == x
    assert ITE(Eq(x, 1), y, x) == x
    assert ITE(Ne(x, 0), y, x) == y
    assert ITE(Ne(x, 1), y, x) == y
    assert ITE(Eq(x, 0), y, z).subs(x, 0) == y
    assert ITE(Eq(x, 0), y, z).subs(x, 1) == z
    raises(ValueError, lambda: ITE(x > 1, y, x, z))


def test_is_literal():
    assert is_literal(True) is True
    assert is_literal(False) is True
    assert is_literal(A) is True
    assert is_literal(~A) is True
    assert is_literal(Or(A, B)) is False
    assert is_literal(Q.zero(A)) is True
    assert is_literal(Not(Q.zero(A))) is True
    assert is_literal(Or(A, B)) is False
    assert is_literal(And(Q.zero(A), Q.zero(B))) is False
    assert is_literal(x < 3)
    assert not is_literal(x + y < 3)


def test_operators():
    # Mostly test __and__, __rand__, and so on
    assert True & A == A & True == A
    assert False & A == A & False == False
    assert A & B == And(A, B)
    assert True | A == A | True == True
    assert False | A == A | False == A
    assert A | B == Or(A, B)
    assert ~A == Not(A)
    assert True >> A == A << True == A
    assert False >> A == A << False == True
    assert A >> True == True << A == True
    assert A >> False == False << A == ~A
    assert A >> B == B << A == Implies(A, B)
    assert True ^ A == A ^ True == ~A
    assert False ^ A == A ^ False == A
    assert A ^ B == Xor(A, B)


def test_true_false():
    assert true is S.true
    assert false is S.false
    assert true is not True
    assert false is not False
    assert true
    assert not false
    assert true == True
    assert false == False
    assert not (true == False)
    assert not (false == True)
    assert not (true == false)

    assert hash(true) == hash(True)
    assert hash(false) == hash(False)
    assert len({true, True}) == len({false, False}) == 1

    assert isinstance(true, BooleanAtom)
    assert isinstance(false, BooleanAtom)
    # We don't want to subclass from bool, because bool subclasses from
    # int. But operators like &, |, ^, <<, >>, and ~ act differently on 0 and
    # 1 then we want them to on true and false.  See the docstrings of the
    # various And, Or, etc. functions for examples.
    assert not isinstance(true, bool)
    assert not isinstance(false, bool)

    # Note: using 'is' comparison is important here. We want these to return
    # true and false, not True and False

    assert Not(true) is false
    assert Not(True) is false
    assert Not(false) is true
    assert Not(False) is true
    assert ~true is false
    assert ~false is true

    for T, F in product((True, true), (False, false)):
        assert And(T, F) is false
        assert And(F, T) is false
        assert And(F, F) is false
        assert And(T, T) is true
        assert And(T, x) == x
        assert And(F, x) is false
        if not (T is True and F is False):
            assert T & F is false
            assert F & T is false
        if F is not False:
            assert F & F is false
        if T is not True:
            assert T & T is true

        assert Or(T, F) is true
        assert Or(F, T) is true
        assert Or(F, F) is false
        assert Or(T, T) is true
        assert Or(T, x) is true
        assert Or(F, x) == x
        if not (T is True and F is False):
            assert T | F is true
            assert F | T is true
        if F is not False:
            assert F | F is false
        if T is not True:
            assert T | T is true

        assert Xor(T, F) is true
        assert Xor(F, T) is true
        assert Xor(F, F) is false
        assert Xor(T, T) is false
        assert Xor(T, x) == ~x
        assert Xor(F, x) == x
        if not (T is True and F is False):
            assert T ^ F is true
            assert F ^ T is true
        if F is not False:
            assert F ^ F is false
        if T is not True:
            assert T ^ T is false

        assert Nand(T, F) is true
        assert Nand(F, T) is true
        assert Nand(F, F) is true
        assert Nand(T, T) is false
        assert Nand(T, x) == ~x
        assert Nand(F, x) is true

        assert Nor(T, F) is false
        assert Nor(F, T) is false
        assert Nor(F, F) is true
        assert Nor(T, T) is false
        assert Nor(T, x) is false
        assert Nor(F, x) == ~x

        assert Implies(T, F) is false
        assert Implies(F, T) is true
        assert Implies(F, F) is true
        assert Implies(T, T) is true
        assert Implies(T, x) == x
        assert Implies(F, x) is true
        assert Implies(x, T) is true
        assert Implies(x, F) == ~x
        if not (T is True and F is False):
            assert T >> F is false
            assert F << T is false
            assert F >> T is true
            assert T << F is true
        if F is not False:
            assert F >> F is true
            assert F << F is true
        if T is not True:
            assert T >> T is true
            assert T << T is true

        assert Equivalent(T, F) is false
        assert Equivalent(F, T) is false
        assert Equivalent(F, F) is true
        assert Equivalent(T, T) is true
        assert Equivalent(T, x) == x
        assert Equivalent(F, x) == ~x
        assert Equivalent(x, T) == x
        assert Equivalent(x, F) == ~x

        assert ITE(T, T, T) is true
        assert ITE(T, T, F) is true
        assert ITE(T, F, T) is false
        assert ITE(T, F, F) is false
        assert ITE(F, T, T) is true
        assert ITE(F, T, F) is false
        assert ITE(F, F, T) is true
        assert ITE(F, F, F) is false

    assert all(i.simplify(1, 2) is i for i in (S.true, S.false))


def test_bool_as_set():
    assert ITE(y <= 0, False, y >= 1).as_set() == Interval(1, oo)
    assert And(x <= 2, x >= -2).as_set() == Interval(-2, 2)
    assert Or(x >= 2, x <= -2).as_set() == Interval(-oo, -2) + Interval(2, oo)
    assert Not(x > 2).as_set() == Interval(-oo, 2)
    # issue 10240
    assert Not(And(x > 2, x < 3)).as_set() == \
           Union(Interval(-oo, 2), Interval(3, oo))
    assert true.as_set() == S.UniversalSet
    assert false.as_set() is S.EmptySet
    assert x.as_set() == S.UniversalSet
    assert And(Or(x < 1, x > 3), x < 2).as_set() == Interval.open(-oo, 1)
    assert And(x < 1, sin(x) < 3).as_set() == (x < 1).as_set()
    raises(NotImplementedError, lambda: (sin(x) < 1).as_set())
    # watch for object morph in as_set
    assert Eq(-1, cos(2 * x) ** 2 / sin(2 * x) ** 2).as_set() is S.EmptySet


@XFAIL
def test_multivariate_bool_as_set():
    x, y = symbols('x,y')

    assert And(x >= 0, y >= 0).as_set() == Interval(0, oo) * Interval(0, oo)
    assert Or(x >= 0, y >= 0).as_set() == S.Reals * S.Reals - \
           Interval(-oo, 0, True, True) * Interval(-oo, 0, True, True)


def test_all_or_nothing():
    x = symbols('x', extended_real=True)
    args = x >= -oo, x <= oo
    v = And(*args)
    if v.func is And:
        assert len(v.args) == len(args) - args.count(S.true)
    else:
        assert v == True
    v = Or(*args)
    if v.func is Or:
        assert len(v.args) == 2
    else:
        assert v == True


def test_canonical_atoms():
    assert true.canonical == true
    assert false.canonical == false


def test_negated_atoms():
    assert true.negated == false
    assert false.negated == true


def test_issue_8777():
    assert And(x > 2, x < oo).as_set() == Interval(2, oo, left_open=True)
    assert And(x >= 1, x < oo).as_set() == Interval(1, oo)
    assert (x < oo).as_set() == Interval(-oo, oo)
    assert (x > -oo).as_set() == Interval(-oo, oo)


def test_issue_8975():
    assert Or(And(-oo < x, x <= -2), And(2 <= x, x < oo)).as_set() == \
           Interval(-oo, -2) + Interval(2, oo)


def test_term_to_integer():
    assert term_to_integer([1, 0, 1, 0, 0, 1, 0]) == 82
    assert term_to_integer('0010101000111001') == 10809


def test_issue_21971():
    a, b, c, d = symbols('a b c d')
    f = a & b & c | a & c
    assert f.subs(a & c, d) == b & d | d
    assert f.subs(a & b & c, d) == a & c | d

    f = (a | b | c) & (a | c)
    assert f.subs(a | c, d) == (b | d) & d
    assert f.subs(a | b | c, d) == (a | c) & d

    f = (a ^ b ^ c) & (a ^ c)
    assert f.subs(a ^ c, d) == (b ^ d) & d
    assert f.subs(a ^ b ^ c, d) == (a ^ c) & d


def test_truth_table():
    assert list(truth_table(And(x, y), [x, y], input=False)) == \
           [False, False, False, True]
    assert list(truth_table(x | y, [x, y], input=False)) == \
           [False, True, True, True]
    assert list(truth_table(x >> y, [x, y], input=False)) == \
           [True, True, False, True]
    assert list(truth_table(And(x, y), [x, y])) == \
           [([0, 0], False), ([0, 1], False), ([1, 0], False), ([1, 1], True)]


def test_issue_8571():
    for t in (S.true, S.false):
        raises(TypeError, lambda: +t)
        raises(TypeError, lambda: -t)
        raises(TypeError, lambda: abs(t))
        # use int(bool(t)) to get 0 or 1
        raises(TypeError, lambda: int(t))

        for o in [S.Zero, S.One, x]:
            for _ in range(2):
                raises(TypeError, lambda: o + t)
                raises(TypeError, lambda: o - t)
                raises(TypeError, lambda: o % t)
                raises(TypeError, lambda: o * t)
                raises(TypeError, lambda: o / t)
                raises(TypeError, lambda: o ** t)
                o, t = t, o  # do again in reversed order


def test_expand_relational():
    n = symbols('n', negative=True)
    p, q = symbols('p q', positive=True)
    r = ((n + q * (-n / q + 1)) / (q * (-n / q + 1)) < 0)
    assert r is not S.false
    assert r.expand() is S.false
    assert (q > 0).expand() is S.true


def test_issue_12717():
    assert S.true.is_Atom == True
    assert S.false.is_Atom == True


def test_as_Boolean():
    nz = symbols('nz', nonzero=True)
    assert all(as_Boolean(i) is S.true for i in (True, S.true, 1, nz))
    z = symbols('z', zero=True)
    assert all(as_Boolean(i) is S.false for i in (False, S.false, 0, z))
    assert all(as_Boolean(i) == i for i in (x, x < 0))
    for i in (2, S(2), x + 1, []):
        raises(TypeError, lambda: as_Boolean(i))


def test_binary_symbols():
    assert ITE(x < 1, y, z).binary_symbols == {y, z}
    for f in (Eq, Ne):
        assert f(x, 1).binary_symbols == set()
        assert f(x, True).binary_symbols == {x}
        assert f(x, False).binary_symbols == {x}
    assert S.true.binary_symbols == set()
    assert S.false.binary_symbols == set()
    assert x.binary_symbols == {x}
    assert And(x, Eq(y, False), Eq(z, 1)).binary_symbols == {x, y}
    assert Q.prime(x).binary_symbols == set()
    assert Q.lt(x, 1).binary_symbols == set()
    assert Q.is_true(x).binary_symbols == {x}
    assert Q.eq(x, True).binary_symbols == {x}
    assert Q.prime(x).binary_symbols == set()


def test_BooleanFunction_diff():
    assert And(x, y).diff(x) == Piecewise((0, Eq(y, False)), (1, True))


def test_issue_14700():
    A, B, C, D, E, F, G, H = symbols('A B C D E F G H')
    q = ((B & D & H & ~F) | (B & H & ~C & ~D) | (B & H & ~C & ~F) |
         (B & H & ~D & ~G) | (B & H & ~F & ~G) | (C & G & ~B & ~D) |
         (C & G & ~D & ~H) | (C & G & ~F & ~H) | (D & F & H & ~B) |
         (D & F & ~G & ~H) | (B & D & F & ~C & ~H) | (D & E & F & ~B & ~C) |
         (D & F & ~A & ~B & ~C) | (D & F & ~A & ~C & ~H) |
         (A & B & D & F & ~E & ~H))
    soldnf = ((B & D & H & ~F) | (D & F & H & ~B) | (B & H & ~C & ~D) |
              (B & H & ~D & ~G) | (C & G & ~B & ~D) | (C & G & ~D & ~H) |
              (C & G & ~F & ~H) | (D & F & ~G & ~H) | (D & E & F & ~C & ~H) |
              (D & F & ~A & ~C & ~H) | (A & B & D & F & ~E & ~H))
    solcnf = ((B | C | D) & (B | D | G) & (C | D | H) & (C | F | H) &
              (D | G | H) & (F | G | H) & (B | F | ~D | ~H) &
              (~B | ~D | ~F | ~H) & (D | ~B | ~C | ~G | ~H) &
              (A | H | ~C | ~D | ~F | ~G) & (H | ~C | ~D | ~E | ~F | ~G) &
              (B | E | H | ~A | ~D | ~F | ~G))
    assert simplify_logic(q, "dnf") == soldnf
    assert simplify_logic(q, "cnf") == solcnf

    minterms = [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                [0, 0, 1, 1], [1, 0, 1, 1]]
    dontcares = [[1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 0, 1]]
    assert SOPform([w, x, y, z], minterms) == (x & ~w) | (y & z & ~x)
    # Should not be more complicated with don't cares
    assert SOPform([w, x, y, z], minterms, dontcares) == \
           (x & ~w) | (y & z & ~x)


def test_issue_25115():
    cond = Contains(x, S.Integers)
    # Previously this raised an exception:
    assert simplify_logic(cond) == cond


def test_relational_simplification():
    w, x, y, z = symbols('w x y z', real=True)
    d, e = symbols('d e', real=False)
    # Test all combinations or sign and order
    assert Or(x >= y, x < y).simplify() == S.true
    assert Or(x >= y, y > x).simplify() == S.true
    assert Or(x >= y, -x > -y).simplify() == S.true
    assert Or(x >= y, -y < -x).simplify() == S.true
    assert Or(-x <= -y, x < y).simplify() == S.true
    assert Or(-x <= -y, -x > -y).simplify() == S.true
    assert Or(-x <= -y, y > x).simplify() == S.true
    assert Or(-x <= -y, -y < -x).simplify() == S.true
    assert Or(y <= x, x < y).simplify() == S.true
    assert Or(y <= x, y > x).simplify() == S.true
    assert Or(y <= x, -x > -y).simplify() == S.true
    assert Or(y <= x, -y < -x).simplify() == S.true
    assert Or(-y >= -x, x < y).simplify() == S.true
    assert Or(-y >= -x, y > x).simplify() == S.true
    assert Or(-y >= -x, -x > -y).simplify() == S.true
    assert Or(-y >= -x, -y < -x).simplify() == S.true

    assert Or(x < y, x >= y).simplify() == S.true
    assert Or(y > x, x >= y).simplify() == S.true
    assert Or(-x > -y, x >= y).simplify() == S.true
    assert Or(-y < -x, x >= y).simplify() == S.true
    assert Or(x < y, -x <= -y).simplify() == S.true
    assert Or(-x > -y, -x <= -y).simplify() == S.true
    assert Or(y > x, -x <= -y).simplify() == S.true
    assert Or(-y < -x, -x <= -y).simplify() == S.true
    assert Or(x < y, y <= x).simplify() == S.true
    assert Or(y > x, y <= x).simplify() == S.true
    assert Or(-x > -y, y <= x).simplify() == S.true
    assert Or(-y < -x, y <= x).simplify() == S.true
    assert Or(x < y, -y >= -x).simplify() == S.true
    assert Or(y > x, -y >= -x).simplify() == S.true
    assert Or(-x > -y, -y >= -x).simplify() == S.true
    assert Or(-y < -x, -y >= -x).simplify() == S.true

    # Some other tests
    assert Or(x >= y, w < z, x <= y).simplify() == S.true
    assert And(x >= y, x < y).simplify() == S.false
    assert Or(x >= y, Eq(y, x)).simplify() == (x >= y)
    assert And(x >= y, Eq(y, x)).simplify() == Eq(x, y)
    assert And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y).simplify() == \
           (Eq(x, y) & (x >= 1) & (y >= 5) & (y > z))
    assert Or(Eq(x, y), x >= y, w < y, z < y).simplify() == \
           (x >= y) | (y > z) | (w < y)
    assert And(Eq(x, y), x >= y, w < y, y >= z, z < y).simplify() == \
           Eq(x, y) & (y > z) & (w < y)
    # assert And(Eq(x, y), x >= y, w < y, y >= z, z < y).simplify(relational_minmax=True) == \
    #    And(Eq(x, y), y > Max(w, z))
    # assert Or(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y).simplify(relational_minmax=True) == \
    #    (Eq(x, y) | (x >= 1) | (y > Min(2, z)))
    assert And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y).simplify() == \
           (Eq(x, y) & (x >= 1) & (y >= 5) & (y > z))
    assert (Eq(x, y) & Eq(d, e) & (x >= y) & (d >= e)).simplify() == \
           (Eq(x, y) & Eq(d, e) & (d >= e))
    assert And(Eq(x, y), Eq(x, -y)).simplify() == And(Eq(x, 0), Eq(y, 0))
    assert Xor(x >= y, x <= y).simplify() == Ne(x, y)
    assert And(x > 1, x < -1, Eq(x, y)).simplify() == S.false
    # From #16690
    assert And(x >= y, Eq(y, 0)).simplify() == And(x >= 0, Eq(y, 0))
    assert Or(Ne(x, 1), Ne(x, 2)).simplify() == S.true
    assert And(Eq(x, 1), Ne(2, x)).simplify() == Eq(x, 1)
    assert Or(Eq(x, 1), Ne(2, x)).simplify() == Ne(x, 2)


def test_issue_8373():
    x = symbols('x', real=True)
    assert Or(x < 1, x > -1).simplify() == S.true
    assert Or(x < 1, x >= 1).simplify() == S.true
    assert And(x < 1, x >= 1).simplify() == S.false
    assert Or(x <= 1, x >= 1).simplify() == S.true


def test_issue_7950():
    x = symbols('x', real=True)
    assert And(Eq(x, 1), Eq(x, 2)).simplify() == S.false


@slow
def test_relational_simplification_numerically():
    def test_simplification_numerically_function(original, simplified):
        symb = original.free_symbols
        n = len(symb)
        valuelist = list(set(combinations(list(range(-(n - 1), n)) * n, n)))
        for values in valuelist:
            sublist = dict(zip(symb, values))
            originalvalue = original.subs(sublist)
            simplifiedvalue = simplified.subs(sublist)
            assert originalvalue == simplifiedvalue, "Original: {}\nand" \
                                                     " simplified: {}\ndo not evaluate to the same value for {}" \
                                                     "".format(original, simplified, sublist)

    w, x, y, z = symbols('w x y z', real=True)
    d, e = symbols('d e', real=False)

    expressions = (And(Eq(x, y), x >= y, w < y, y >= z, z < y),
                   And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y),
                   Or(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y),
                   And(x >= y, Eq(y, x)),
                   Or(And(Eq(x, y), x >= y, w < y, Or(y >= z, z < y)),
                      And(Eq(x, y), x >= 1, 2 < y, y >= -1, z < y)),
                   (Eq(x, y) & Eq(d, e) & (x >= y) & (d >= e)),
                   )

    for expression in expressions:
        test_simplification_numerically_function(expression,
                                                 expression.simplify())


def test_relational_simplification_patterns_numerically():
    from sympy.core import Wild
    from sympy.logic.boolalg import _simplify_patterns_and, \
        _simplify_patterns_or, _simplify_patterns_xor
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    symb = [a, b, c]
    patternlists = [[And, _simplify_patterns_and()],
                    [Or, _simplify_patterns_or()],
                    [Xor, _simplify_patterns_xor()]]
    valuelist = list(set(combinations(list(range(-2, 3)) * 3, 3)))
    # Skip combinations of +/-2 and 0, except for all 0
    valuelist = [v for v in valuelist if any(w % 2 for w in v) or not any(v)]
    for func, patternlist in patternlists:
        for pattern in patternlist:
            original = func(*pattern[0].args)
            simplified = pattern[1]
            for values in valuelist:
                sublist = dict(zip(symb, values))
                originalvalue = original.xreplace(sublist)
                simplifiedvalue = simplified.xreplace(sublist)
                assert originalvalue == simplifiedvalue, "Original: {}\nand" \
                                                         " simplified: {}\ndo not evaluate to the same value for" \
                                                         "{}".format(pattern[0], simplified, sublist)


def test_issue_16803():
    n = symbols('n')
    # No simplification done, but should not raise an exception
    assert ((n > 3) | (n < 0) | ((n > 0) & (n < 3))).simplify() == \
           (n > 3) | (n < 0) | ((n > 0) & (n < 3))


def test_issue_17530():
    r = {x: oo, y: oo}
    assert Or(x + y > 0, x - y < 0).subs(r)
    assert not And(x + y < 0, x - y < 0).subs(r)
    raises(TypeError, lambda: Or(x + y < 0, x - y < 0).subs(r))
    raises(TypeError, lambda: And(x + y > 0, x - y < 0).subs(r))
    raises(TypeError, lambda: And(x + y > 0, x - y < 0).subs(r))


def test_anf_coeffs():
    assert anf_coeffs([1, 0]) == [1, 1]
    assert anf_coeffs([0, 0, 0, 1]) == [0, 0, 0, 1]
    assert anf_coeffs([0, 1, 1, 1]) == [0, 1, 1, 1]
    assert anf_coeffs([1, 1, 1, 0]) == [1, 0, 0, 1]
    assert anf_coeffs([1, 0, 0, 0]) == [1, 1, 1, 1]
    assert anf_coeffs([1, 0, 0, 1]) == [1, 1, 1, 0]
    assert anf_coeffs([1, 1, 0, 1]) == [1, 0, 1, 1]


def test_ANFform():
    x, y = symbols('x,y')
    assert ANFform([x], [1, 1]) == True
    assert ANFform([x], [0, 0]) == False
    assert ANFform([x], [1, 0]) == Xor(x, True, remove_true=False)
    assert ANFform([x, y], [1, 1, 1, 0]) == \
           Xor(True, And(x, y), remove_true=False)


def test_bool_minterm():
    x, y = symbols('x,y')
    assert bool_minterm(3, [x, y]) == And(x, y)
    assert bool_minterm([1, 0], [x, y]) == And(Not(y), x)


def test_bool_maxterm():
    x, y = symbols('x,y')
    assert bool_maxterm(2, [x, y]) == Or(Not(x), y)
    assert bool_maxterm([0, 1], [x, y]) == Or(Not(y), x)


def test_bool_monomial():
    x, y = symbols('x,y')
    assert bool_monomial(1, [x, y]) == y
    assert bool_monomial([1, 1], [x, y]) == And(x, y)


def test_check_pair():
    assert _check_pair([0, 1, 0], [0, 1, 1]) == 2
    assert _check_pair([0, 1, 0], [1, 1, 1]) == -1


def test_issue_19114():
    expr = (B & C) | (A & ~C) | (~A & ~B)
    # Expression is minimal, but there are multiple minimal forms possible
    res1 = (A & B) | (C & ~A) | (~B & ~C)
    result = to_dnf(expr, simplify=True)
    assert result in (expr, res1)


def test_issue_20870():
    result = SOPform([a, b, c, d], [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15])
    expected = ((d & ~b) | (a & b & c) | (a & ~c & ~d) |
                (b & ~a & ~c) | (c & ~a & ~d))
    assert result == expected


def test_convert_to_varsSOP():
    assert _convert_to_varsSOP([0, 1, 0], [x, y, z]) == And(Not(x), y, Not(z))
    assert _convert_to_varsSOP([3, 1, 0], [x, y, z]) == And(y, Not(z))


def test_convert_to_varsPOS():
    assert _convert_to_varsPOS([0, 1, 0], [x, y, z]) == Or(x, Not(y), z)
    assert _convert_to_varsPOS([3, 1, 0], [x, y, z]) == Or(Not(y), z)


def test_gateinputcount():
    a, b, c, d, e = symbols('a:e')
    assert gateinputcount(And(a, b)) == 2
    assert gateinputcount(a | b & c & d ^ (e | a)) == 9
    assert gateinputcount(And(a, True)) == 0
    raises(TypeError, lambda: gateinputcount(a * b))


def test_refine():
    # relational
    assert not refine(x < 0, ~(x < 0))
    assert refine(x < 0, (x < 0))
    assert refine(x < 0, (0 > x)) is S.true
    assert refine(x < 0, (y < 0)) == (x < 0)
    assert not refine(x <= 0, ~(x <= 0))
    assert refine(x <= 0, (x <= 0))
    assert refine(x <= 0, (0 >= x)) is S.true
    assert refine(x <= 0, (y <= 0)) == (x <= 0)
    assert not refine(x > 0, ~(x > 0))
    assert refine(x > 0, (x > 0))
    assert refine(x > 0, (0 < x)) is S.true
    assert refine(x > 0, (y > 0)) == (x > 0)
    assert not refine(x >= 0, ~(x >= 0))
    assert refine(x >= 0, (x >= 0))
    assert refine(x >= 0, (0 <= x)) is S.true
    assert refine(x >= 0, (y >= 0)) == (x >= 0)
    assert not refine(Eq(x, 0), ~(Eq(x, 0)))
    assert refine(Eq(x, 0), (Eq(x, 0)))
    assert refine(Eq(x, 0), (Eq(0, x))) is S.true
    assert refine(Eq(x, 0), (Eq(y, 0))) == Eq(x, 0)
    assert not refine(Ne(x, 0), ~(Ne(x, 0)))
    assert refine(Ne(x, 0), (Ne(0, x))) is S.true
    assert refine(Ne(x, 0), (Ne(x, 0)))
    assert refine(Ne(x, 0), (Ne(y, 0))) == (Ne(x, 0))

    # boolean functions
    assert refine(And(x > 0, y > 0), (x > 0)) == (y > 0)
    assert refine(And(x > 0, y > 0), (x > 0) & (y > 0)) is S.true

    # predicates
    assert refine(Q.positive(x), Q.positive(x)) is S.true
    assert refine(Q.positive(x), Q.negative(x)) is S.false
    assert refine(Q.positive(x), Q.real(x)) == Q.positive(x)


def test_relational_threeterm_simplification_patterns_numerically():
    from sympy.core import Wild
    from sympy.logic.boolalg import _simplify_patterns_and3
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    symb = [a, b, c]
    patternlists = [[And, _simplify_patterns_and3()]]
    valuelist = list(set(combinations(list(range(-2, 3)) * 3, 3)))
    # Skip combinations of +/-2 and 0, except for all 0
    valuelist = [v for v in valuelist if any(w % 2 for w in v) or not any(v)]
    for func, patternlist in patternlists:
        for pattern in patternlist:
            original = func(*pattern[0].args)
            simplified = pattern[1]
            for values in valuelist:
                sublist = dict(zip(symb, values))
                originalvalue = original.xreplace(sublist)
                simplifiedvalue = simplified.xreplace(sublist)
                assert originalvalue == simplifiedvalue, "Original: {}\nand" \
                                                         " simplified: {}\ndo not evaluate to the same value for" \
                                                         "{}".format(pattern[0], simplified, sublist)


def test_issue_25451():
    x = Or(And(a, c), Eq(a, b))
    assert isinstance(x, Or)
    assert set(x.args) == {And(a, c), Eq(a, b)}


def test_issue_26985():
    a, b, c, d = symbols('a b c d')

    # Expression before applying to_anf
    x = Xor(c, And(a, b), And(a, c))
    y = Xor(a, b, And(a, c))

    # Applying to_anf
    result = Xor(Xor(d, And(x, y)), And(x, y))
    result_anf = to_anf(Xor(to_anf(Xor(d, And(x, y))), And(x, y)))

    assert result_anf == d
    assert result == d
