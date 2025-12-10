from sympy.physics.quantum.hilbert import (
    HilbertSpace, ComplexSpace, L2, FockSpace, TensorProductHilbertSpace,
    DirectSumHilbertSpace, TensorPowerHilbertSpace
)

from sympy.core.numbers import oo
from sympy.core.symbol import Symbol
from sympy.printing.repr import srepr
from sympy.printing.str import sstr
from sympy.sets.sets import Interval


def test_hilbert_space():
    hs = HilbertSpace()
    assert isinstance(hs, HilbertSpace)
    assert sstr(hs) == 'H'
    assert srepr(hs) == 'HilbertSpace()'


def test_complex_space():
    c1 = ComplexSpace(2)
    assert isinstance(c1, ComplexSpace)
    assert c1.dimension == 2
    assert sstr(c1) == 'C(2)'
    assert srepr(c1) == 'ComplexSpace(Integer(2))'

    n = Symbol('n')
    c2 = ComplexSpace(n)
    assert isinstance(c2, ComplexSpace)
    assert c2.dimension == n
    assert sstr(c2) == 'C(n)'
    assert srepr(c2) == "ComplexSpace(Symbol('n'))"
    assert c2.subs(n, 2) == ComplexSpace(2)


def test_L2():
    b1 = L2(Interval(-oo, 1))
    assert isinstance(b1, L2)
    assert b1.dimension is oo
    assert b1.interval == Interval(-oo, 1)

    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    b2 = L2(Interval(x, y))
    assert b2.dimension is oo
    assert b2.interval == Interval(x, y)
    assert b2.subs(x, -1) == L2(Interval(-1, y))


def test_fock_space():
    f1 = FockSpace()
    f2 = FockSpace()
    assert isinstance(f1, FockSpace)
    assert f1.dimension is oo
    assert f1 == f2


def test_tensor_product():
    n = Symbol('n')
    hs1 = ComplexSpace(2)
    hs2 = ComplexSpace(n)

    h = hs1*hs2
    assert isinstance(h, TensorProductHilbertSpace)
    assert h.dimension == 2*n
    assert h.spaces == (hs1, hs2)

    h = hs2*hs2
    assert isinstance(h, TensorPowerHilbertSpace)
    assert h.base == hs2
    assert h.exp == 2
    assert h.dimension == n**2

    f = FockSpace()
    h = hs1*hs2*f
    assert h.dimension is oo


def test_tensor_power():
    n = Symbol('n')
    hs1 = ComplexSpace(2)
    hs2 = ComplexSpace(n)

    h = hs1**2
    assert isinstance(h, TensorPowerHilbertSpace)
    assert h.base == hs1
    assert h.exp == 2
    assert h.dimension == 4

    h = hs2**3
    assert isinstance(h, TensorPowerHilbertSpace)
    assert h.base == hs2
    assert h.exp == 3
    assert h.dimension == n**3


def test_direct_sum():
    n = Symbol('n')
    hs1 = ComplexSpace(2)
    hs2 = ComplexSpace(n)

    h = hs1 + hs2
    assert isinstance(h, DirectSumHilbertSpace)
    assert h.dimension == 2 + n
    assert h.spaces == (hs1, hs2)

    f = FockSpace()
    h = hs1 + f + hs2
    assert h.dimension is oo
    assert h.spaces == (hs1, f, hs2)
