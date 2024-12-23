"""Tests for sho1d.py"""

from sympy.core.numbers import (I, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum import Commutator
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.cartesian import X, Px
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.represent import represent
from sympy.external import import_module
from sympy.testing.pytest import skip

from sympy.physics.quantum.sho1d import (RaisingOp, LoweringOp,
                                        SHOKet, SHOBra,
                                        Hamiltonian, NumberOp)

ad = RaisingOp('a')
a = LoweringOp('a')
k = SHOKet('k')
kz = SHOKet(0)
kf = SHOKet(1)
k3 = SHOKet(3)
b = SHOBra('b')
b3 = SHOBra(3)
H = Hamiltonian('H')
N = NumberOp('N')
omega = Symbol('omega')
m = Symbol('m')
ndim = Integer(4)

np = import_module('numpy')
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})

ad_rep_sympy = represent(ad, basis=N, ndim=4, format='sympy')
a_rep = represent(a, basis=N, ndim=4, format='sympy')
N_rep = represent(N, basis=N, ndim=4, format='sympy')
H_rep = represent(H, basis=N, ndim=4, format='sympy')
k3_rep = represent(k3, basis=N, ndim=4, format='sympy')
b3_rep = represent(b3, basis=N, ndim=4, format='sympy')

def test_RaisingOp():
    assert Dagger(ad) == a
    assert Commutator(ad, a).doit() == Integer(-1)
    assert Commutator(ad, N).doit() == Integer(-1)*ad
    assert qapply(ad*k) == (sqrt(k.n + 1)*SHOKet(k.n + 1)).expand()
    assert qapply(ad*kz) == (sqrt(kz.n + 1)*SHOKet(kz.n + 1)).expand()
    assert qapply(ad*kf) == (sqrt(kf.n + 1)*SHOKet(kf.n + 1)).expand()
    assert ad.rewrite('xp').doit() == \
        (Integer(1)/sqrt(Integer(2)*hbar*m*omega))*(Integer(-1)*I*Px + m*omega*X)
    assert ad.hilbert_space == ComplexSpace(S.Infinity)
    for i in range(ndim - 1):
        assert ad_rep_sympy[i + 1,i] == sqrt(i + 1)

    if not np:
        skip("numpy not installed.")

    ad_rep_numpy = represent(ad, basis=N, ndim=4, format='numpy')
    for i in range(ndim - 1):
        assert ad_rep_numpy[i + 1,i] == float(sqrt(i + 1))

    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")

    ad_rep_scipy = represent(ad, basis=N, ndim=4, format='scipy.sparse', spmatrix='lil')
    for i in range(ndim - 1):
        assert ad_rep_scipy[i + 1,i] == float(sqrt(i + 1))

    assert ad_rep_numpy.dtype == 'float64'
    assert ad_rep_scipy.dtype == 'float64'

def test_LoweringOp():
    assert Dagger(a) == ad
    assert Commutator(a, ad).doit() == Integer(1)
    assert Commutator(a, N).doit() == a
    assert qapply(a*k) == (sqrt(k.n)*SHOKet(k.n-Integer(1))).expand()
    assert qapply(a*kz) == Integer(0)
    assert qapply(a*kf) == (sqrt(kf.n)*SHOKet(kf.n-Integer(1))).expand()
    assert a.rewrite('xp').doit() == \
        (Integer(1)/sqrt(Integer(2)*hbar*m*omega))*(I*Px + m*omega*X)
    for i in range(ndim - 1):
        assert a_rep[i,i + 1] == sqrt(i + 1)

def test_NumberOp():
    assert Commutator(N, ad).doit() == ad
    assert Commutator(N, a).doit() == Integer(-1)*a
    assert Commutator(N, H).doit() == Integer(0)
    assert qapply(N*k) == (k.n*k).expand()
    assert N.rewrite('a').doit() == ad*a
    assert N.rewrite('xp').doit() == (Integer(1)/(Integer(2)*m*hbar*omega))*(
        Px**2 + (m*omega*X)**2) - Integer(1)/Integer(2)
    assert N.rewrite('H').doit() == H/(hbar*omega) - Integer(1)/Integer(2)
    for i in range(ndim):
        assert N_rep[i,i] == i
    assert N_rep == ad_rep_sympy*a_rep

def test_Hamiltonian():
    assert Commutator(H, N).doit() == Integer(0)
    assert qapply(H*k) == ((hbar*omega*(k.n + Integer(1)/Integer(2)))*k).expand()
    assert H.rewrite('a').doit() == hbar*omega*(ad*a + Integer(1)/Integer(2))
    assert H.rewrite('xp').doit() == \
        (Integer(1)/(Integer(2)*m))*(Px**2 + (m*omega*X)**2)
    assert H.rewrite('N').doit() == hbar*omega*(N + Integer(1)/Integer(2))
    for i in range(ndim):
        assert H_rep[i,i] == hbar*omega*(i + Integer(1)/Integer(2))

def test_SHOKet():
    assert SHOKet('k').dual_class() == SHOBra
    assert SHOBra('b').dual_class() == SHOKet
    assert InnerProduct(b,k).doit() == KroneckerDelta(k.n, b.n)
    assert k.hilbert_space == ComplexSpace(S.Infinity)
    assert k3_rep[k3.n, 0] == Integer(1)
    assert b3_rep[0, b3.n] == Integer(1)
