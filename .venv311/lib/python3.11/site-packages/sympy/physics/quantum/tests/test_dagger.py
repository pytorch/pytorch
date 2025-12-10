from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import conjugate
from sympy.matrices.dense import Matrix

from sympy.physics.quantum.dagger import adjoint, Dagger
from sympy.external import import_module
from sympy.testing.pytest import skip, warns_deprecated_sympy
from sympy.physics.quantum.operator import Operator, IdentityOperator


def test_scalars():
    x = symbols('x', complex=True)
    assert Dagger(x) == conjugate(x)
    assert Dagger(I*x) == -I*conjugate(x)

    i = symbols('i', real=True)
    assert Dagger(i) == i

    p = symbols('p')
    assert isinstance(Dagger(p), conjugate)

    i = Integer(3)
    assert Dagger(i) == i

    A = symbols('A', commutative=False)
    assert Dagger(A).is_commutative is False


def test_matrix():
    x = symbols('x')
    m = Matrix([[I, x*I], [2, 4]])
    assert Dagger(m) == m.H


def test_dagger_mul():
    O = Operator('O')
    assert Dagger(O)*O == Dagger(O)*O
    with warns_deprecated_sympy():
        I = IdentityOperator()
        assert Dagger(O)*O*I == Mul(Dagger(O), O)*I
    assert Dagger(O)*Dagger(O) == Dagger(O)**2
    assert Dagger(O)*Dagger(I) == Dagger(O)


class Foo(Expr):

    def _eval_adjoint(self):
        return I


def test_eval_adjoint():
    f = Foo()
    d = Dagger(f)
    assert d == I

np = import_module('numpy')


def test_numpy_dagger():
    if not np:
        skip("numpy not installed.")

    a = np.array([[1.0, 2.0j], [-1.0j, 2.0]])
    adag = a.copy().transpose().conjugate()
    assert (Dagger(a) == adag).all()


scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})


def test_scipy_sparse_dagger():
    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")
    else:
        sparse = scipy.sparse

    a = sparse.csr_matrix([[1.0 + 0.0j, 2.0j], [-1.0j, 2.0 + 0.0j]])
    adag = a.copy().transpose().conjugate()
    assert np.linalg.norm((Dagger(a) - adag).todense()) == 0.0


def test_unknown():
    """Check treatment of unknown objects.
    Objects without adjoint or conjugate/transpose methods
    are sympified and wrapped in dagger.
    """
    x = symbols("x", commutative=False)
    result = Dagger(x)
    assert result.args == (x,) and isinstance(result, adjoint)


def test_unevaluated():
    """Check that evaluate=False returns unevaluated Dagger.
    """
    x = symbols("x", real=True)
    assert Dagger(x) == x
    result = Dagger(x, evaluate=False)
    assert result.args == (x,) and isinstance(result, adjoint)
