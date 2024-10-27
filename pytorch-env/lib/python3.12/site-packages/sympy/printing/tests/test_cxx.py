from sympy.core.numbers import Float, Integer, Rational
from sympy.core.symbol import symbols
from sympy.functions import beta, Ei, zeta, Max, Min, sqrt, riemann_xi, frac
from sympy.printing.cxx import CXX98CodePrinter, CXX11CodePrinter, CXX17CodePrinter, cxxcode
from sympy.codegen.cfunctions import log1p


x, y, u, v = symbols('x y u v')


def test_CXX98CodePrinter():
    assert CXX98CodePrinter().doprint(Max(x, 3)) in ('std::max(x, 3)', 'std::max(3, x)')
    assert CXX98CodePrinter().doprint(Min(x, 3, sqrt(x))) == 'std::min(3, std::min(x, std::sqrt(x)))'
    cxx98printer = CXX98CodePrinter()
    assert cxx98printer.language == 'C++'
    assert cxx98printer.standard == 'C++98'
    assert 'template' in cxx98printer.reserved_words
    assert 'alignas' not in cxx98printer.reserved_words


def test_CXX11CodePrinter():
    assert CXX11CodePrinter().doprint(log1p(x)) == 'std::log1p(x)'

    cxx11printer = CXX11CodePrinter()
    assert cxx11printer.language == 'C++'
    assert cxx11printer.standard == 'C++11'
    assert 'operator' in cxx11printer.reserved_words
    assert 'noexcept' in cxx11printer.reserved_words
    assert 'concept' not in cxx11printer.reserved_words


def test_subclass_print_method():
    class MyPrinter(CXX11CodePrinter):
        def _print_log1p(self, expr):
            return 'my_library::log1p(%s)' % ', '.join(map(self._print, expr.args))

    assert MyPrinter().doprint(log1p(x)) == 'my_library::log1p(x)'


def test_subclass_print_method__ns():
    class MyPrinter(CXX11CodePrinter):
        _ns = 'my_library::'

    p = CXX11CodePrinter()
    myp = MyPrinter()

    assert p.doprint(log1p(x)) == 'std::log1p(x)'
    assert myp.doprint(log1p(x)) == 'my_library::log1p(x)'


def test_CXX17CodePrinter():
    assert CXX17CodePrinter().doprint(beta(x, y)) == 'std::beta(x, y)'
    assert CXX17CodePrinter().doprint(Ei(x)) == 'std::expint(x)'
    assert CXX17CodePrinter().doprint(zeta(x)) == 'std::riemann_zeta(x)'

    # Automatic rewrite
    assert CXX17CodePrinter().doprint(frac(x)) == '(x - std::floor(x))'
    assert CXX17CodePrinter().doprint(riemann_xi(x)) == '((1.0/2.0)*std::pow(M_PI, -1.0/2.0*x)*x*(x - 1)*std::tgamma((1.0/2.0)*x)*std::riemann_zeta(x))'


def test_cxxcode():
    assert sorted(cxxcode(sqrt(x)*.5).split('*')) == sorted(['0.5', 'std::sqrt(x)'])

def test_cxxcode_nested_minmax():
    assert cxxcode(Max(Min(x, y), Min(u, v))) \
        == 'std::max(std::min(u, v), std::min(x, y))'
    assert cxxcode(Min(Max(x, y), Max(u, v))) \
        == 'std::min(std::max(u, v), std::max(x, y))'

def test_subclass_Integer_Float():
    class MyPrinter(CXX17CodePrinter):
        def _print_Integer(self, arg):
            return 'bigInt("%s")' % super()._print_Integer(arg)

        def _print_Float(self, arg):
            rat = Rational(arg)
            return 'bigFloat(%s, %s)' % (
                self._print(Integer(rat.p)),
                self._print(Integer(rat.q))
            )

    p = MyPrinter()
    for i in range(13):
        assert p.doprint(i) == 'bigInt("%d")' % i
    assert p.doprint(Float(0.5)) == 'bigFloat(bigInt("1"), bigInt("2"))'
    assert p.doprint(x**-1.0) == 'bigFloat(bigInt("1"), bigInt("1"))/x'
