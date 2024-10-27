from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.external import import_module
from sympy.stats import Geometric, Poisson, Zeta, sample, Skellam, DiscreteRV, Logarithmic, NegativeBinomial, YuleSimon
from sympy.testing.pytest import skip, raises, slow


def test_sample_numpy():
    distribs_numpy = [
        Geometric('G', 0.5),
        Poisson('P', 1),
        Zeta('Z', 2)
    ]
    size = 3
    numpy = import_module('numpy')
    if not numpy:
        skip('Numpy is not installed. Abort tests for _sample_numpy.')
    else:
        for X in distribs_numpy:
            samps = sample(X, size=size, library='numpy')
            for sam in samps:
                assert sam in X.pspace.domain.set
        raises(NotImplementedError,
               lambda: sample(Skellam('S', 1, 1), library='numpy'))
    raises(NotImplementedError,
           lambda: Skellam('S', 1, 1).pspace.distribution.sample(library='tensorflow'))


def test_sample_scipy():
    p = S(2)/3
    x = Symbol('x', integer=True, positive=True)
    pdf = p*(1 - p)**(x - 1) # pdf of Geometric Distribution
    distribs_scipy = [
        DiscreteRV(x, pdf, set=S.Naturals),
        Geometric('G', 0.5),
        Logarithmic('L', 0.5),
        NegativeBinomial('N', 5, 0.4),
        Poisson('P', 1),
        Skellam('S', 1, 1),
        YuleSimon('Y', 1),
        Zeta('Z', 2)
    ]
    size = 3
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests for _sample_scipy.')
    else:
        for X in distribs_scipy:
            samps = sample(X, size=size, library='scipy')
            samps2 = sample(X, size=(2, 2), library='scipy')
            for sam in samps:
                assert sam in X.pspace.domain.set
            for i in range(2):
                for j in range(2):
                    assert samps2[i][j] in X.pspace.domain.set


def test_sample_pymc():
    distribs_pymc = [
        Geometric('G', 0.5),
        Poisson('P', 1),
        NegativeBinomial('N', 5, 0.4)
    ]
    size = 3
    pymc = import_module('pymc')
    if not pymc:
        skip('PyMC is not installed. Abort tests for _sample_pymc.')
    else:
        for X in distribs_pymc:
            samps = sample(X, size=size, library='pymc')
            for sam in samps:
                assert sam in X.pspace.domain.set
        raises(NotImplementedError,
               lambda: sample(Skellam('S', 1, 1), library='pymc'))

@slow
def test_sample_discrete():
    X = Geometric('X', S.Half)
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy not installed. Abort tests')
    assert sample(X) in X.pspace.domain.set
    samps = sample(X, size=2) # This takes long time if ran without scipy
    for samp in samps:
        assert samp in X.pspace.domain.set

    libraries = ['scipy', 'numpy', 'pymc']
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            if imported_lib:
                s0, s1, s2 = [], [], []
                s0 = sample(X, size=10, library=lib, seed=0)
                s1 = sample(X, size=10, library=lib, seed=0)
                s2 = sample(X, size=10, library=lib, seed=1)
                assert all(s0 == s1)
                assert not all(s1 == s2)
        except NotImplementedError:
            continue
