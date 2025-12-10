from sympy.concrete.products import Product
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.trace import Trace
from sympy.tensor.indexed import IndexedBase
from sympy.stats import (GaussianUnitaryEnsemble as GUE, density,
                         GaussianOrthogonalEnsemble as GOE,
                         GaussianSymplecticEnsemble as GSE,
                         joint_eigen_distribution,
                         CircularUnitaryEnsemble as CUE,
                         CircularOrthogonalEnsemble as COE,
                         CircularSymplecticEnsemble as CSE,
                         JointEigenDistribution,
                         level_spacing_distribution,
                         Normal, Beta)
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import RandomMatrixSymbol
from sympy.stats.random_matrix_models import GaussianEnsemble, RandomMatrixPSpace
from sympy.testing.pytest import raises

def test_GaussianEnsemble():
    G = GaussianEnsemble('G', 3)
    assert density(G) == G.pspace.model
    raises(ValueError, lambda: GaussianEnsemble('G', 3.5))

def test_GaussianUnitaryEnsemble():
    H = RandomMatrixSymbol('H', 3, 3)
    G = GUE('U', 3)
    assert density(G)(H) == sqrt(2)*exp(-3*Trace(H**2)/2)/(4*pi**Rational(9, 2))
    i, j = (Dummy('i', integer=True, positive=True),
            Dummy('j', integer=True, positive=True))
    l = IndexedBase('l')
    assert joint_eigen_distribution(G).dummy_eq(
            Lambda((l[1], l[2], l[3]),
            27*sqrt(6)*exp(-3*(l[1]**2)/2 - 3*(l[2]**2)/2 - 3*(l[3]**2)/2)*
            Product(Abs(l[i] - l[j])**2, (j, i + 1, 3), (i, 1, 2))/(16*pi**Rational(3, 2))))
    s = Dummy('s')
    assert level_spacing_distribution(G).dummy_eq(Lambda(s, 32*s**2*exp(-4*s**2/pi)/pi**2))


def test_GaussianOrthogonalEnsemble():
    H = RandomMatrixSymbol('H', 3, 3)
    _H = MatrixSymbol('_H', 3, 3)
    G = GOE('O', 3)
    assert density(G)(H) == exp(-3*Trace(H**2)/4)/Integral(exp(-3*Trace(_H**2)/4), _H)
    i, j = (Dummy('i', integer=True, positive=True),
            Dummy('j', integer=True, positive=True))
    l = IndexedBase('l')
    assert joint_eigen_distribution(G).dummy_eq(
            Lambda((l[1], l[2], l[3]),
            9*sqrt(2)*exp(-3*l[1]**2/2 - 3*l[2]**2/2 - 3*l[3]**2/2)*
            Product(Abs(l[i] - l[j]), (j, i + 1, 3), (i, 1, 2))/(32*pi)))
    s = Dummy('s')
    assert level_spacing_distribution(G).dummy_eq(Lambda(s, s*pi*exp(-s**2*pi/4)/2))

def test_GaussianSymplecticEnsemble():
    H = RandomMatrixSymbol('H', 3, 3)
    _H = MatrixSymbol('_H', 3, 3)
    G = GSE('O', 3)
    assert density(G)(H) == exp(-3*Trace(H**2))/Integral(exp(-3*Trace(_H**2)), _H)
    i, j = (Dummy('i', integer=True, positive=True),
            Dummy('j', integer=True, positive=True))
    l = IndexedBase('l')
    assert joint_eigen_distribution(G).dummy_eq(
            Lambda((l[1], l[2], l[3]),
            162*sqrt(3)*exp(-3*l[1]**2/2 - 3*l[2]**2/2 - 3*l[3]**2/2)*
            Product(Abs(l[i] - l[j])**4, (j, i + 1, 3), (i, 1, 2))/(5*pi**Rational(3, 2))))
    s = Dummy('s')
    assert level_spacing_distribution(G).dummy_eq(Lambda(s, S(262144)*s**4*exp(-64*s**2/(9*pi))/(729*pi**3)))

def test_CircularUnitaryEnsemble():
    CU = CUE('U', 3)
    j, k = (Dummy('j', integer=True, positive=True),
            Dummy('k', integer=True, positive=True))
    t = IndexedBase('t')
    assert joint_eigen_distribution(CU).dummy_eq(
            Lambda((t[1], t[2], t[3]),
            Product(Abs(exp(I*t[j]) - exp(I*t[k]))**2,
            (j, k + 1, 3), (k, 1, 2))/(48*pi**3))
    )

def test_CircularOrthogonalEnsemble():
    CO = COE('U', 3)
    j, k = (Dummy('j', integer=True, positive=True),
            Dummy('k', integer=True, positive=True))
    t = IndexedBase('t')
    assert joint_eigen_distribution(CO).dummy_eq(
            Lambda((t[1], t[2], t[3]),
            Product(Abs(exp(I*t[j]) - exp(I*t[k])),
            (j, k + 1, 3), (k, 1, 2))/(48*pi**2))
    )

def test_CircularSymplecticEnsemble():
    CS = CSE('U', 3)
    j, k = (Dummy('j', integer=True, positive=True),
            Dummy('k', integer=True, positive=True))
    t = IndexedBase('t')
    assert joint_eigen_distribution(CS).dummy_eq(
            Lambda((t[1], t[2], t[3]),
            Product(Abs(exp(I*t[j]) - exp(I*t[k]))**4,
            (j, k + 1, 3), (k, 1, 2))/(720*pi**3))
    )

def test_JointEigenDistribution():
    A = Matrix([[Normal('A00', 0, 1), Normal('A01', 1, 1)],
                [Beta('A10', 1, 1), Beta('A11', 1, 1)]])
    assert JointEigenDistribution(A) == \
    JointDistributionHandmade(-sqrt(A[0, 0]**2 - 2*A[0, 0]*A[1, 1] + 4*A[0, 1]*A[1, 0] + A[1, 1]**2)/2 +
    A[0, 0]/2 + A[1, 1]/2, sqrt(A[0, 0]**2 - 2*A[0, 0]*A[1, 1] + 4*A[0, 1]*A[1, 0] + A[1, 1]**2)/2 + A[0, 0]/2 + A[1, 1]/2)
    raises(ValueError, lambda: JointEigenDistribution(Matrix([[1, 0], [2, 1]])))

def test_issue_19841():
    G1 = GUE('U', 2)
    G2 = G1.xreplace({2: 2})
    assert G1.args == G2.args

    X = MatrixSymbol('X', 2, 2)
    G = GSE('U', 2)
    h_pspace = RandomMatrixPSpace('P', model=density(G))
    H = RandomMatrixSymbol('H', 2, 2, pspace=h_pspace)
    H2 = RandomMatrixSymbol('H', 2, 2, pspace=None)
    assert H.doit() == H

    assert (2*H).xreplace({H: X}) == 2*X
    assert (2*H).xreplace({H2: X}) == 2*H
    assert (2*H2).xreplace({H: X}) == 2*H2
    assert (2*H2).xreplace({H2: X}) == 2*X
