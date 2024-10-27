from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.external import import_module
from sympy.stats import Binomial, sample, Die, FiniteRV, DiscreteUniform, Bernoulli, BetaBinomial, Hypergeometric, \
    Rademacher
from sympy.testing.pytest import skip, raises

def test_given_sample():
    X = Die('X', 6)
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy is not installed. Abort tests')
    assert sample(X, X > 5) == 6

def test_sample_numpy():
    distribs_numpy = [
        Binomial("B", 5, 0.4),
        Hypergeometric("H", 2, 1, 1)
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
               lambda: sample(Die("D"), library='numpy'))
    raises(NotImplementedError,
           lambda: Die("D").pspace.sample(library='tensorflow'))


def test_sample_scipy():
    distribs_scipy = [
        FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)}),
        DiscreteUniform("Y", list(range(5))),
        Die("D"),
        Bernoulli("Be", 0.3),
        Binomial("Bi", 5, 0.4),
        BetaBinomial("Bb", 2, 1, 1),
        Hypergeometric("H", 1, 1, 1),
        Rademacher("R")
    ]

    size = 3
    scipy = import_module('scipy')
    if not scipy:
        skip('Scipy not installed. Abort tests for _sample_scipy.')
    else:
        for X in distribs_scipy:
            samps = sample(X, size=size)
            samps2 = sample(X, size=(2, 2))
            for sam in samps:
                assert sam in X.pspace.domain.set
            for i in range(2):
                for j in range(2):
                    assert samps2[i][j] in X.pspace.domain.set


def test_sample_pymc():
    distribs_pymc = [
        Bernoulli('B', 0.2),
        Binomial('N', 5, 0.4)
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
                          lambda: (sample(Die("D"), library='pymc')))


def test_sample_seed():
    F = FiniteRV('F', {1: S.Half, 2: Rational(1, 4), 3: Rational(1, 4)})
    size = 10
    libraries = ['scipy', 'numpy', 'pymc']
    for lib in libraries:
        try:
            imported_lib = import_module(lib)
            if imported_lib:
                s0 = sample(F, size=size, library=lib, seed=0)
                s1 = sample(F, size=size, library=lib, seed=0)
                s2 = sample(F, size=size, library=lib, seed=1)
                assert all(s0 == s1)
                assert not all(s1 == s2)
        except NotImplementedError:
            continue
