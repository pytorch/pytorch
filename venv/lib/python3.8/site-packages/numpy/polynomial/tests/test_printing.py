import numpy.polynomial as poly
from numpy.testing import assert_equal


class TestStr:
    def test_polynomial_str(self):
        res = str(poly.Polynomial([0, 1]))
        tgt = 'poly([0. 1.])'
        assert_equal(res, tgt)

    def test_chebyshev_str(self):
        res = str(poly.Chebyshev([0, 1]))
        tgt = 'cheb([0. 1.])'
        assert_equal(res, tgt)

    def test_legendre_str(self):
        res = str(poly.Legendre([0, 1]))
        tgt = 'leg([0. 1.])'
        assert_equal(res, tgt)

    def test_hermite_str(self):
        res = str(poly.Hermite([0, 1]))
        tgt = 'herm([0. 1.])'
        assert_equal(res, tgt)

    def test_hermiteE_str(self):
        res = str(poly.HermiteE([0, 1]))
        tgt = 'herme([0. 1.])'
        assert_equal(res, tgt)

    def test_laguerre_str(self):
        res = str(poly.Laguerre([0, 1]))
        tgt = 'lag([0. 1.])'
        assert_equal(res, tgt)


class TestRepr:
    def test_polynomial_str(self):
        res = repr(poly.Polynomial([0, 1]))
        tgt = 'Polynomial([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_chebyshev_str(self):
        res = repr(poly.Chebyshev([0, 1]))
        tgt = 'Chebyshev([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_legendre_repr(self):
        res = repr(poly.Legendre([0, 1]))
        tgt = 'Legendre([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_hermite_repr(self):
        res = repr(poly.Hermite([0, 1]))
        tgt = 'Hermite([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_hermiteE_repr(self):
        res = repr(poly.HermiteE([0, 1]))
        tgt = 'HermiteE([0., 1.], domain=[-1,  1], window=[-1,  1])'
        assert_equal(res, tgt)

    def test_laguerre_repr(self):
        res = repr(poly.Laguerre([0, 1]))
        tgt = 'Laguerre([0., 1.], domain=[0, 1], window=[0, 1])'
        assert_equal(res, tgt)


class TestLatexRepr:
    """Test the latex repr used by Jupyter"""

    def as_latex(self, obj):
        # right now we ignore the formatting of scalars in our tests, since
        # it makes them too verbose. Ideally, the formatting of scalars will
        # be fixed such that tests below continue to pass
        obj._repr_latex_scalar = lambda x: str(x)
        try:
            return obj._repr_latex_()
        finally:
            del obj._repr_latex_scalar

    def test_simple_polynomial(self):
        # default input
        p = poly.Polynomial([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,x + 3.0\,x^{2}$')

        # translated input
        p = poly.Polynomial([1, 2, 3], domain=[-2, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(1.0 + x\right) + 3.0\,\left(1.0 + x\right)^{2}$')

        # scaled input
        p = poly.Polynomial([1, 2, 3], domain=[-0.5, 0.5])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(2.0x\right) + 3.0\,\left(2.0x\right)^{2}$')

        # affine input
        p = poly.Polynomial([1, 2, 3], domain=[-1, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0 + 2.0\,\left(1.0 + 2.0x\right) + 3.0\,\left(1.0 + 2.0x\right)^{2}$')

    def test_basis_func(self):
        p = poly.Chebyshev([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{T}_{0}(x) + 2.0\,{T}_{1}(x) + 3.0\,{T}_{2}(x)$')
        # affine input - check no surplus parens are added
        p = poly.Chebyshev([1, 2, 3], domain=[-1, 0])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{T}_{0}(1.0 + 2.0x) + 2.0\,{T}_{1}(1.0 + 2.0x) + 3.0\,{T}_{2}(1.0 + 2.0x)$')

    def test_multichar_basis_func(self):
        p = poly.HermiteE([1, 2, 3])
        assert_equal(self.as_latex(p),
            r'$x \mapsto 1.0\,{He}_{0}(x) + 2.0\,{He}_{1}(x) + 3.0\,{He}_{2}(x)$')
