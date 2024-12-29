from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import sin
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.physics.quantum.operator import (Operator, UnitaryOperator,
                                            HermitianOperator, OuterProduct,
                                            DifferentialOperator,
                                            IdentityOperator)
from sympy.physics.quantum.state import Ket, Bra, Wavefunction
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.spin import JzKet, JzBra
from sympy.physics.quantum.trace import Tr
from sympy.matrices import eye


class CustomKet(Ket):
    @classmethod
    def default_args(self):
        return ("t",)


class CustomOp(HermitianOperator):
    @classmethod
    def default_args(self):
        return ("T",)

t_ket = CustomKet()
t_op = CustomOp()


def test_operator():
    A = Operator('A')
    B = Operator('B')
    C = Operator('C')

    assert isinstance(A, Operator)
    assert isinstance(A, QExpr)

    assert A.label == (Symbol('A'),)
    assert A.is_commutative is False
    assert A.hilbert_space == HilbertSpace()

    assert A*B != B*A

    assert (A*(B + C)).expand() == A*B + A*C
    assert ((A + B)**2).expand() == A**2 + A*B + B*A + B**2

    assert t_op.label[0] == Symbol(t_op.default_args()[0])

    assert Operator() == Operator("O")
    assert A*IdentityOperator() == A


def test_operator_inv():
    A = Operator('A')
    assert A*A.inv() == 1
    assert A.inv()*A == 1


def test_hermitian():
    H = HermitianOperator('H')

    assert isinstance(H, HermitianOperator)
    assert isinstance(H, Operator)

    assert Dagger(H) == H
    assert H.inv() != H
    assert H.is_commutative is False
    assert Dagger(H).is_commutative is False


def test_unitary():
    U = UnitaryOperator('U')

    assert isinstance(U, UnitaryOperator)
    assert isinstance(U, Operator)

    assert U.inv() == Dagger(U)
    assert U*Dagger(U) == 1
    assert Dagger(U)*U == 1
    assert U.is_commutative is False
    assert Dagger(U).is_commutative is False


def test_identity():
    I = IdentityOperator()
    O = Operator('O')
    x = Symbol("x")

    assert isinstance(I, IdentityOperator)
    assert isinstance(I, Operator)

    assert I * O == O
    assert O * I == O
    assert I * Dagger(O) == Dagger(O)
    assert Dagger(O) * I == Dagger(O)
    assert isinstance(I * I, IdentityOperator)
    assert isinstance(3 * I, Mul)
    assert isinstance(I * x, Mul)
    assert I.inv() == I
    assert Dagger(I) == I
    assert qapply(I * O) == O
    assert qapply(O * I) == O

    for n in [2, 3, 5]:
        assert represent(IdentityOperator(n)) == eye(n)


def test_outer_product():
    k = Ket('k')
    b = Bra('b')
    op = OuterProduct(k, b)

    assert isinstance(op, OuterProduct)
    assert isinstance(op, Operator)

    assert op.ket == k
    assert op.bra == b
    assert op.label == (k, b)
    assert op.is_commutative is False

    op = k*b

    assert isinstance(op, OuterProduct)
    assert isinstance(op, Operator)

    assert op.ket == k
    assert op.bra == b
    assert op.label == (k, b)
    assert op.is_commutative is False

    op = 2*k*b

    assert op == Mul(Integer(2), k, b)

    op = 2*(k*b)

    assert op == Mul(Integer(2), OuterProduct(k, b))

    assert Dagger(k*b) == OuterProduct(Dagger(b), Dagger(k))
    assert Dagger(k*b).is_commutative is False

    #test the _eval_trace
    assert Tr(OuterProduct(JzKet(1, 1), JzBra(1, 1))).doit() == 1

    # test scaled kets and bras
    assert OuterProduct(2 * k, b) == 2 * OuterProduct(k, b)
    assert OuterProduct(k, 2 * b) == 2 * OuterProduct(k, b)

    # test sums of kets and bras
    k1, k2 = Ket('k1'), Ket('k2')
    b1, b2 = Bra('b1'), Bra('b2')
    assert (OuterProduct(k1 + k2, b1) ==
            OuterProduct(k1, b1) + OuterProduct(k2, b1))
    assert (OuterProduct(k1, b1 + b2) ==
            OuterProduct(k1, b1) + OuterProduct(k1, b2))
    assert (OuterProduct(1 * k1 + 2 * k2, 3 * b1 + 4 * b2) ==
            3 * OuterProduct(k1, b1) +
            4 * OuterProduct(k1, b2) +
            6 * OuterProduct(k2, b1) +
            8 * OuterProduct(k2, b2))


def test_operator_dagger():
    A = Operator('A')
    B = Operator('B')
    assert Dagger(A*B) == Dagger(B)*Dagger(A)
    assert Dagger(A + B) == Dagger(A) + Dagger(B)
    assert Dagger(A**2) == Dagger(A)**2


def test_differential_operator():
    x = Symbol('x')
    f = Function('f')
    d = DifferentialOperator(Derivative(f(x), x), f(x))
    g = Wavefunction(x**2, x)
    assert qapply(d*g) == Wavefunction(2*x, x)
    assert d.expr == Derivative(f(x), x)
    assert d.function == f(x)
    assert d.variables == (x,)
    assert diff(d, x) == DifferentialOperator(Derivative(f(x), x, 2), f(x))

    d = DifferentialOperator(Derivative(f(x), x, 2), f(x))
    g = Wavefunction(x**3, x)
    assert qapply(d*g) == Wavefunction(6*x, x)
    assert d.expr == Derivative(f(x), x, 2)
    assert d.function == f(x)
    assert d.variables == (x,)
    assert diff(d, x) == DifferentialOperator(Derivative(f(x), x, 3), f(x))

    d = DifferentialOperator(1/x*Derivative(f(x), x), f(x))
    assert d.expr == 1/x*Derivative(f(x), x)
    assert d.function == f(x)
    assert d.variables == (x,)
    assert diff(d, x) == \
        DifferentialOperator(Derivative(1/x*Derivative(f(x), x), x), f(x))
    assert qapply(d*g) == Wavefunction(3*x, x)

    # 2D cartesian Laplacian
    y = Symbol('y')
    d = DifferentialOperator(Derivative(f(x, y), x, 2) +
                             Derivative(f(x, y), y, 2), f(x, y))
    w = Wavefunction(x**3*y**2 + y**3*x**2, x, y)
    assert d.expr == Derivative(f(x, y), x, 2) + Derivative(f(x, y), y, 2)
    assert d.function == f(x, y)
    assert d.variables == (x, y)
    assert diff(d, x) == \
        DifferentialOperator(Derivative(d.expr, x), f(x, y))
    assert diff(d, y) == \
        DifferentialOperator(Derivative(d.expr, y), f(x, y))
    assert qapply(d*w) == Wavefunction(2*x**3 + 6*x*y**2 + 6*x**2*y + 2*y**3,
                                       x, y)

    # 2D polar Laplacian (th = theta)
    r, th = symbols('r th')
    d = DifferentialOperator(1/r*Derivative(r*Derivative(f(r, th), r), r) +
                             1/(r**2)*Derivative(f(r, th), th, 2), f(r, th))
    w = Wavefunction(r**2*sin(th), r, (th, 0, pi))
    assert d.expr == \
        1/r*Derivative(r*Derivative(f(r, th), r), r) + \
        1/(r**2)*Derivative(f(r, th), th, 2)
    assert d.function == f(r, th)
    assert d.variables == (r, th)
    assert diff(d, r) == \
        DifferentialOperator(Derivative(d.expr, r), f(r, th))
    assert diff(d, th) == \
        DifferentialOperator(Derivative(d.expr, th), f(r, th))
    assert qapply(d*w) == Wavefunction(3*sin(th), r, (th, 0, pi))


def test_eval_power():
    from sympy.core import Pow
    from sympy.core.expr import unchanged
    O = Operator('O')
    U = UnitaryOperator('U')
    H = HermitianOperator('H')
    assert O**-1 == O.inv() # same as doc test
    assert U**-1 == U.inv()
    assert H**-1 == H.inv()
    x = symbols("x", commutative = True)
    assert unchanged(Pow, H, x) # verify Pow(H,x)=="X^n"
    assert H**x == Pow(H, x)
    assert Pow(H,x) == Pow(H, x, evaluate=False) # Just check
    from sympy.physics.quantum.gate import XGate
    X = XGate(0) # is hermitian and unitary
    assert unchanged(Pow, X, x) # verify Pow(X,x)=="X^x"
    assert X**x == Pow(X, x)
    assert Pow(X, x, evaluate=False) == Pow(X, x) # Just check
    n = symbols("n", integer=True, even=True)
    assert X**n == 1
    n = symbols("n", integer=True, odd=True)
    assert X**n == X
    n = symbols("n", integer=True)
    assert unchanged(Pow, X, n) # verify Pow(X,n)=="X^n"
    assert X**n == Pow(X, n)
    assert Pow(X, n, evaluate=False)==Pow(X, n) # Just check
    assert X**4 == 1
    assert X**7 == X
