from sympy import sin, cos
from sympy.testing.pytest import raises

from sympy.tensor.toperators import PartialDerivative
from sympy.tensor.tensor import (TensorIndexType,
                                 tensor_indices,
                                 TensorHead, tensor_heads)
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import diag
from sympy.tensor.array import Array

from sympy.core.random import randint


L = TensorIndexType("L")
i, j, k, m, m1, m2, m3, m4 = tensor_indices("i j k m m1 m2 m3 m4", L)
i0 = tensor_indices("i0", L)
L_0, L_1 = tensor_indices("L_0 L_1", L)

A, B, C, D = tensor_heads("A B C D", [L])

H = TensorHead("H", [L, L])


def test_invalid_partial_derivative_valence():
    raises(ValueError, lambda: PartialDerivative(C(j), D(-j)))
    raises(ValueError, lambda: PartialDerivative(C(-j), D(j)))


def test_tensor_partial_deriv():
    # Test flatten:
    expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(k))
    assert expr == PartialDerivative(A(i), A(j), A(k))
    assert expr.expr == A(i)
    assert expr.variables == (A(j), A(k))
    assert expr.get_indices() == [i, -j, -k]
    assert expr.get_free_indices() == [i, -j, -k]

    expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(i))
    assert expr.expr == A(L_0)
    assert expr.variables == (A(j), A(L_0))

    expr1 = PartialDerivative(A(i), A(j))
    assert expr1.expr == A(i)
    assert expr1.variables == (A(j),)

    expr2 = A(i)*PartialDerivative(H(k, -i), A(j))
    assert expr2.get_indices() == [L_0, k, -L_0, -j]

    expr2b = A(i)*PartialDerivative(H(k, -i), A(-j))
    assert expr2b.get_indices() == [L_0, k, -L_0, j]

    expr3 = A(i)*PartialDerivative(B(k)*C(-i) + 3*H(k, -i), A(j))
    assert expr3.get_indices() == [L_0, k, -L_0, -j]

    expr4 = (A(i) + B(i))*PartialDerivative(C(j), D(j))
    assert expr4.get_indices() == [i, L_0, -L_0]

    expr4b = (A(i) + B(i))*PartialDerivative(C(-j), D(-j))
    assert expr4b.get_indices() == [i, -L_0, L_0]

    expr5 = (A(i) + B(i))*PartialDerivative(C(-i), D(j))
    assert expr5.get_indices() == [L_0, -L_0, -j]


def test_replace_arrays_partial_derivative():

    x, y, z, t = symbols("x y z t")

    expr = PartialDerivative(A(i), B(j))
    repl = expr.replace_with_arrays({A(i): [sin(x)*cos(y), x**3*y**2], B(i): [x, y]})
    assert repl == Array([[cos(x)*cos(y), -sin(x)*sin(y)], [3*x**2*y**2, 2*x**3*y]])
    repl = expr.replace_with_arrays({A(i): [sin(x)*cos(y), x**3*y**2], B(i): [x, y]}, [-j, i])
    assert repl == Array([[cos(x)*cos(y), 3*x**2*y**2], [-sin(x)*sin(y), 2*x**3*y]])

    # d(A^i)/d(A_j) = d(g^ik A_k)/d(A_j) = g^ik delta_jk
    expr = PartialDerivative(A(i), A(-j))
    assert expr.get_free_indices() == [i, j]
    assert expr.get_indices() == [i, j]
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, [i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, [i, j]) == Array([[1, 0], [0, -1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, 1)}, [i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, -1)}, [i, j]) == Array([[1, 0], [0, -1]])

    expr = PartialDerivative(A(i), A(j))
    assert expr.get_free_indices() == [i, -j]
    assert expr.get_indices() == [i, -j]
    assert expr.replace_with_arrays({A(i): [x, y]}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, 1)}, [i, -j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, -1)}, [i, -j]) == Array([[1, 0], [0, 1]])

    expr = PartialDerivative(A(-i), A(-j))
    assert expr.get_free_indices() == [-i, j]
    assert expr.get_indices() == [-i, j]
    assert expr.replace_with_arrays({A(-i): [x, y]}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, 1)}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(-i): [x, y], L: diag(1, -1)}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, [-i, j]) == Array([[1, 0], [0, 1]])
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, [-i, j]) == Array([[1, 0], [0, 1]])

    expr = PartialDerivative(A(i), A(i))
    assert expr.get_free_indices() == []
    assert expr.get_indices() == [L_0, -L_0]
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, []) == 2
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, []) == 2

    expr = PartialDerivative(A(-i), A(-i))
    assert expr.get_free_indices() == []
    assert expr.get_indices() == [-L_0, L_0]
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, 1)}, []) == 2
    assert expr.replace_with_arrays({A(i): [x, y], L: diag(1, -1)}, []) == 2

    expr = PartialDerivative(H(i, j) + H(j, i), A(i))
    assert expr.get_indices() == [L_0, j, -L_0]
    assert expr.get_free_indices() == [j]

    expr = PartialDerivative(H(i, j) + H(j, i), A(k))*B(-i)
    assert expr.get_indices() == [L_0, j, -k, -L_0]
    assert expr.get_free_indices() == [j, -k]

    expr = PartialDerivative(A(i)*(H(-i, j) + H(j, -i)), A(j))
    assert expr.get_indices() == [L_0, -L_0, L_1, -L_1]
    assert expr.get_free_indices() == []

    expr = A(j)*A(-j) + expr
    assert expr.get_indices() == [L_0, -L_0, L_1, -L_1]
    assert expr.get_free_indices() == []

    expr = A(i)*(B(j)*PartialDerivative(C(-j), D(i)) + C(j)*PartialDerivative(D(-j), B(i)))
    assert expr.get_indices() == [L_0, L_1, -L_1, -L_0]
    assert expr.get_free_indices() == []

    expr = A(i)*PartialDerivative(C(-j), D(i))
    assert expr.get_indices() == [L_0, -j, -L_0]
    assert expr.get_free_indices() == [-j]


def test_expand_partial_derivative_sum_rule():
    tau = symbols("tau")

    # check sum rule for D(tensor, symbol)
    expr1aa = PartialDerivative(A(i), tau)

    assert expr1aa._expand_partial_derivative() == PartialDerivative(A(i), tau)

    expr1ab = PartialDerivative(A(i) + B(i), tau)

    assert (expr1ab._expand_partial_derivative() ==
            PartialDerivative(A(i), tau) +
            PartialDerivative(B(i), tau))

    expr1ac = PartialDerivative(A(i) + B(i) + C(i), tau)

    assert (expr1ac._expand_partial_derivative() ==
            PartialDerivative(A(i), tau) +
            PartialDerivative(B(i), tau) +
            PartialDerivative(C(i), tau))

    # check sum rule for D(tensor, D(j))
    expr1ba = PartialDerivative(A(i), D(j))

    assert expr1ba._expand_partial_derivative() ==\
        PartialDerivative(A(i), D(j))
    expr1bb = PartialDerivative(A(i) + B(i), D(j))

    assert (expr1bb._expand_partial_derivative() ==
            PartialDerivative(A(i), D(j)) +
            PartialDerivative(B(i), D(j)))

    expr1bc = PartialDerivative(A(i) + B(i) + C(i), D(j))
    assert expr1bc._expand_partial_derivative() ==\
        PartialDerivative(A(i), D(j))\
        + PartialDerivative(B(i), D(j))\
        + PartialDerivative(C(i), D(j))

    # check sum rule for D(tensor, H(j, k))
    expr1ca = PartialDerivative(A(i), H(j, k))
    assert expr1ca._expand_partial_derivative() ==\
        PartialDerivative(A(i), H(j, k))
    expr1cb = PartialDerivative(A(i) + B(i), H(j, k))
    assert (expr1cb._expand_partial_derivative() ==
            PartialDerivative(A(i), H(j, k))
            + PartialDerivative(B(i), H(j, k)))
    expr1cc = PartialDerivative(A(i) + B(i) + C(i), H(j, k))
    assert (expr1cc._expand_partial_derivative() ==
            PartialDerivative(A(i), H(j, k))
            + PartialDerivative(B(i), H(j, k))
            + PartialDerivative(C(i), H(j, k)))

    # check sum rule for D(D(tensor, D(j)), H(k, m))
    expr1da = PartialDerivative(A(i), (D(j), H(k, m)))
    assert expr1da._expand_partial_derivative() ==\
        PartialDerivative(A(i), (D(j), H(k, m)))
    expr1db = PartialDerivative(A(i) + B(i), (D(j), H(k, m)))
    assert expr1db._expand_partial_derivative() ==\
        PartialDerivative(A(i), (D(j), H(k, m)))\
        + PartialDerivative(B(i), (D(j), H(k, m)))
    expr1dc = PartialDerivative(A(i) + B(i) + C(i), (D(j), H(k, m)))
    assert expr1dc._expand_partial_derivative() ==\
        PartialDerivative(A(i), (D(j), H(k, m)))\
        + PartialDerivative(B(i), (D(j), H(k, m)))\
        + PartialDerivative(C(i), (D(j), H(k, m)))


def test_expand_partial_derivative_constant_factor_rule():
    nneg = randint(0, 1000)
    pos = randint(1, 1000)
    neg = -randint(1, 1000)

    c1 = Rational(nneg, pos)
    c2 = Rational(neg, pos)
    c3 = Rational(nneg, neg)

    expr2a = PartialDerivative(nneg*A(i), D(j))
    assert expr2a._expand_partial_derivative() ==\
        nneg*PartialDerivative(A(i), D(j))

    expr2b = PartialDerivative(neg*A(i), D(j))
    assert expr2b._expand_partial_derivative() ==\
        neg*PartialDerivative(A(i), D(j))

    expr2ca = PartialDerivative(c1*A(i), D(j))
    assert expr2ca._expand_partial_derivative() ==\
        c1*PartialDerivative(A(i), D(j))

    expr2cb = PartialDerivative(c2*A(i), D(j))
    assert expr2cb._expand_partial_derivative() ==\
        c2*PartialDerivative(A(i), D(j))

    expr2cc = PartialDerivative(c3*A(i), D(j))
    assert expr2cc._expand_partial_derivative() ==\
        c3*PartialDerivative(A(i), D(j))


def test_expand_partial_derivative_full_linearity():
    nneg = randint(0, 1000)
    pos = randint(1, 1000)
    neg = -randint(1, 1000)

    c1 = Rational(nneg, pos)
    c2 = Rational(neg, pos)
    c3 = Rational(nneg, neg)

    # check full linearity
    p = PartialDerivative(42, D(j))
    assert p and not p._expand_partial_derivative()

    expr3a = PartialDerivative(nneg*A(i) + pos*B(i), D(j))
    assert expr3a._expand_partial_derivative() ==\
        nneg*PartialDerivative(A(i), D(j))\
        + pos*PartialDerivative(B(i), D(j))

    expr3b = PartialDerivative(nneg*A(i) + neg*B(i), D(j))
    assert expr3b._expand_partial_derivative() ==\
        nneg*PartialDerivative(A(i), D(j))\
        + neg*PartialDerivative(B(i), D(j))

    expr3c = PartialDerivative(neg*A(i) + pos*B(i), D(j))
    assert expr3c._expand_partial_derivative() ==\
        neg*PartialDerivative(A(i), D(j))\
        + pos*PartialDerivative(B(i), D(j))

    expr3d = PartialDerivative(c1*A(i) + c2*B(i), D(j))
    assert expr3d._expand_partial_derivative() ==\
        c1*PartialDerivative(A(i), D(j))\
        + c2*PartialDerivative(B(i), D(j))

    expr3e = PartialDerivative(c2*A(i) + c1*B(i), D(j))
    assert expr3e._expand_partial_derivative() ==\
        c2*PartialDerivative(A(i), D(j))\
        + c1*PartialDerivative(B(i), D(j))

    expr3f = PartialDerivative(c2*A(i) + c3*B(i), D(j))
    assert expr3f._expand_partial_derivative() ==\
        c2*PartialDerivative(A(i), D(j))\
        + c3*PartialDerivative(B(i), D(j))

    expr3g = PartialDerivative(c3*A(i) + c2*B(i), D(j))
    assert expr3g._expand_partial_derivative() ==\
        c3*PartialDerivative(A(i), D(j))\
        + c2*PartialDerivative(B(i), D(j))

    expr3h = PartialDerivative(c3*A(i) + c1*B(i), D(j))
    assert expr3h._expand_partial_derivative() ==\
        c3*PartialDerivative(A(i), D(j))\
        + c1*PartialDerivative(B(i), D(j))

    expr3i = PartialDerivative(c1*A(i) + c3*B(i), D(j))
    assert expr3i._expand_partial_derivative() ==\
        c1*PartialDerivative(A(i), D(j))\
        + c3*PartialDerivative(B(i), D(j))


def test_expand_partial_derivative_product_rule():
    # check product rule
    expr4a = PartialDerivative(A(i)*B(j), D(k))

    assert expr4a._expand_partial_derivative() == \
        PartialDerivative(A(i), D(k))*B(j)\
        + A(i)*PartialDerivative(B(j), D(k))

    expr4b = PartialDerivative(A(i)*B(j)*C(k), D(m))
    assert expr4b._expand_partial_derivative() ==\
        PartialDerivative(A(i), D(m))*B(j)*C(k)\
        + A(i)*PartialDerivative(B(j), D(m))*C(k)\
        + A(i)*B(j)*PartialDerivative(C(k), D(m))

    expr4c = PartialDerivative(A(i)*B(j), C(k), D(m))
    assert expr4c._expand_partial_derivative() ==\
        PartialDerivative(A(i), C(k), D(m))*B(j) \
        + PartialDerivative(A(i), C(k))*PartialDerivative(B(j), D(m))\
        + PartialDerivative(A(i), D(m))*PartialDerivative(B(j), C(k))\
        + A(i)*PartialDerivative(B(j), C(k), D(m))


def test_eval_partial_derivative_expr_by_symbol():

    tau, alpha = symbols("tau alpha")

    expr1 = PartialDerivative(tau**alpha, tau)
    assert expr1._perform_derivative() == alpha * 1 / tau * tau ** alpha

    expr2 = PartialDerivative(2*tau + 3*tau**4, tau)
    assert expr2._perform_derivative() == 2 + 12 * tau ** 3

    expr3 = PartialDerivative(2*tau + 3*tau**4, alpha)
    assert expr3._perform_derivative() == 0


def test_eval_partial_derivative_single_tensors_by_scalar():

    tau, mu = symbols("tau mu")

    expr = PartialDerivative(tau**mu, tau)
    assert expr._perform_derivative() == mu*tau**mu/tau

    expr1a = PartialDerivative(A(i), tau)
    assert expr1a._perform_derivative() == 0

    expr1b = PartialDerivative(A(-i), tau)
    assert expr1b._perform_derivative() == 0

    expr2a = PartialDerivative(H(i, j), tau)
    assert expr2a._perform_derivative() == 0

    expr2b = PartialDerivative(H(i, -j), tau)
    assert expr2b._perform_derivative() == 0

    expr2c = PartialDerivative(H(-i, j), tau)
    assert expr2c._perform_derivative() == 0

    expr2d = PartialDerivative(H(-i, -j), tau)
    assert expr2d._perform_derivative() == 0


def test_eval_partial_derivative_single_1st_rank_tensors_by_tensor():

    expr1 = PartialDerivative(A(i), A(j))
    assert expr1._perform_derivative() - L.delta(i, -j) == 0

    expr2 = PartialDerivative(A(i), A(-j))
    assert expr2._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, j) == 0

    expr3 = PartialDerivative(A(-i), A(-j))
    assert expr3._perform_derivative() - L.delta(-i, j) == 0

    expr4 = PartialDerivative(A(-i), A(j))
    assert expr4._perform_derivative() - L.metric(-i, -L_0) * L.delta(L_0, -j) == 0

    expr5 = PartialDerivative(A(i), B(j))
    expr6 = PartialDerivative(A(i), C(j))
    expr7 = PartialDerivative(A(i), D(j))
    expr8 = PartialDerivative(A(i), H(j, k))
    assert expr5._perform_derivative() == 0
    assert expr6._perform_derivative() == 0
    assert expr7._perform_derivative() == 0
    assert expr8._perform_derivative() == 0

    expr9 = PartialDerivative(A(i), A(i))
    assert expr9._perform_derivative() - L.delta(L_0, -L_0) == 0

    expr10 = PartialDerivative(A(-i), A(-i))
    assert expr10._perform_derivative() - L.delta(-L_0, L_0) == 0


def test_eval_partial_derivative_single_2nd_rank_tensors_by_tensor():

    expr1 = PartialDerivative(H(i, j), H(m, m1))
    assert expr1._perform_derivative() - L.delta(i, -m) * L.delta(j, -m1) == 0

    expr2 = PartialDerivative(H(i, j), H(-m, m1))
    assert expr2._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, m) * L.delta(j, -m1) == 0

    expr3 = PartialDerivative(H(i, j), H(m, -m1))
    assert expr3._perform_derivative() - L.delta(i, -m) * L.metric(j, L_0) * L.delta(-L_0, m1) == 0

    expr4 = PartialDerivative(H(i, j), H(-m, -m1))
    assert expr4._perform_derivative() - L.metric(i, L_0) * L.delta(-L_0, m) * L.metric(j, L_1) * L.delta(-L_1, m1) == 0

def test_eval_partial_derivative_divergence_type():
    expr1a = PartialDerivative(A(i), A(i))
    expr1b = PartialDerivative(A(i), A(k))
    expr1c = PartialDerivative(L.delta(-i, k) * A(i), A(k))

    assert (expr1a._perform_derivative()
            - (L.delta(-i, k) * expr1b._perform_derivative())).contract_delta(L.delta) == 0

    assert (expr1a._perform_derivative()
            - expr1c._perform_derivative()).contract_delta(L.delta) == 0

    expr2a = PartialDerivative(H(i, j), H(i, j))
    expr2b = PartialDerivative(H(i, j), H(k, m))
    expr2c = PartialDerivative(L.delta(-i, k) * L.delta(-j, m) * H(i, j), H(k, m))

    assert (expr2a._perform_derivative()
            - (L.delta(-i, k) * L.delta(-j, m) * expr2b._perform_derivative())).contract_delta(L.delta) == 0

    assert (expr2a._perform_derivative()
            - expr2c._perform_derivative()).contract_delta(L.delta) == 0


def test_eval_partial_derivative_expr1():

    tau, alpha = symbols("tau alpha")

    # this is only some special expression
    # tested: vector derivative
    # tested: scalar derivative
    # tested: tensor derivative
    base_expr1 = A(i)*H(-i, j) + A(i)*A(-i)*A(j) + tau**alpha*A(j)

    tensor_derivative = PartialDerivative(base_expr1, H(k, m))._perform_derivative()
    vector_derivative = PartialDerivative(base_expr1, A(k))._perform_derivative()
    scalar_derivative = PartialDerivative(base_expr1, tau)._perform_derivative()

    assert (tensor_derivative - A(L_0)*L.metric(-L_0, -L_1)*L.delta(L_1, -k)*L.delta(j, -m)) == 0

    assert (vector_derivative - (tau**alpha*L.delta(j, -k) +
        L.delta(L_0, -k)*A(-L_0)*A(j) +
        A(L_0)*L.metric(-L_0, -L_1)*L.delta(L_1, -k)*A(j) +
        A(L_0)*A(-L_0)*L.delta(j, -k) +
        L.delta(L_0, -k)*H(-L_0, j))).expand() == 0

    assert (vector_derivative.contract_metric(L.metric).contract_delta(L.delta) -
        (tau**alpha*L.delta(j, -k) + A(L_0)*A(-L_0)*L.delta(j, -k) + H(-k, j) + 2*A(j)*A(-k))).expand() == 0

    assert scalar_derivative - alpha*1/tau*tau**alpha*A(j) == 0


def test_eval_partial_derivative_mixed_scalar_tensor_expr2():

    tau, alpha = symbols("tau alpha")

    base_expr2 = A(i)*A(-i) + tau**2

    vector_expression = PartialDerivative(base_expr2, A(k))._perform_derivative()
    assert  (vector_expression -
        (L.delta(L_0, -k)*A(-L_0) + A(L_0)*L.metric(-L_0, -L_1)*L.delta(L_1, -k))).expand() == 0

    scalar_expression = PartialDerivative(base_expr2, tau)._perform_derivative()
    assert scalar_expression == 2*tau
