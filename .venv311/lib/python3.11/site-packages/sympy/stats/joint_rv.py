"""
Joint Random Variables Module

See Also
========
sympy.stats.rv
sympy.stats.frv
sympy.stats.crv
sympy.stats.drv
"""
from math import prod

from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.sets.sets import ProductSet
from sympy.tensor.indexed import Indexed
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum, summation
from sympy.core.containers import Tuple
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices import ImmutableMatrix, matrix2numpy, list2numpy
from sympy.stats.crv import SingleContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import (ProductPSpace, NamedArgsMixin, Distribution,
                            ProductDomain, RandomSymbol, random_symbols,
                            SingleDomain, _symbol_converter)
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import filldedent
from sympy.external import import_module

# __all__ = ['marginal_distribution']

class JointPSpace(ProductPSpace):
    """
    Represents a joint probability space. Represented using symbols for
    each component and a distribution.
    """
    def __new__(cls, sym, dist):
        if isinstance(dist, SingleContinuousDistribution):
            return SingleContinuousPSpace(sym, dist)
        if isinstance(dist, SingleDiscreteDistribution):
            return SingleDiscretePSpace(sym, dist)
        sym = _symbol_converter(sym)
        return Basic.__new__(cls, sym, dist)

    @property
    def set(self):
        return self.domain.set

    @property
    def symbol(self):
        return self.args[0]

    @property
    def distribution(self):
        return self.args[1]

    @property
    def value(self):
        return JointRandomSymbol(self.symbol, self)

    @property
    def component_count(self):
        _set = self.distribution.set
        if isinstance(_set, ProductSet):
            return S(len(_set.args))
        elif isinstance(_set, Product):
            return _set.limits[0][-1]
        return S.One

    @property
    def pdf(self):
        sym = [Indexed(self.symbol, i) for i in range(self.component_count)]
        return self.distribution(*sym)

    @property
    def domain(self):
        rvs = random_symbols(self.distribution)
        if not rvs:
            return SingleDomain(self.symbol, self.distribution.set)
        return ProductDomain(*[rv.pspace.domain for rv in rvs])

    def component_domain(self, index):
        return self.set.args[index]

    def marginal_distribution(self, *indices):
        count = self.component_count
        if count.atoms(Symbol):
            raise ValueError("Marginal distributions cannot be computed "
                                "for symbolic dimensions. It is a work under progress.")
        orig = [Indexed(self.symbol, i) for i in range(count)]
        all_syms = [Symbol(str(i)) for i in orig]
        replace_dict = dict(zip(all_syms, orig))
        sym = tuple(Symbol(str(Indexed(self.symbol, i))) for i in indices)
        limits = [[i,] for i in all_syms if i not in sym]
        index = 0
        for i in range(count):
            if i not in indices:
                limits[index].append(self.distribution.set.args[i])
                limits[index] = tuple(limits[index])
                index += 1
        if self.distribution.is_Continuous:
            f = Lambda(sym, integrate(self.distribution(*all_syms), *limits))
        elif self.distribution.is_Discrete:
            f = Lambda(sym, summation(self.distribution(*all_syms), *limits))
        return f.xreplace(replace_dict)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        syms = tuple(self.value[i] for i in range(self.component_count))
        rvs = rvs or syms
        if not any(i in rvs for i in syms):
            return expr
        expr = expr*self.pdf
        for rv in rvs:
            if isinstance(rv, Indexed):
                expr = expr.xreplace({rv: Indexed(str(rv.base), rv.args[1])})
            elif isinstance(rv, RandomSymbol):
                expr = expr.xreplace({rv: rv.symbol})
        if self.value in random_symbols(expr):
            raise NotImplementedError(filldedent('''
            Expectations of expression with unindexed joint random symbols
            cannot be calculated yet.'''))
        limits = tuple((Indexed(str(rv.base),rv.args[1]),
            self.distribution.set.args[rv.args[1]]) for rv in syms)
        return Integral(expr, *limits)

    def where(self, condition):
        raise NotImplementedError()

    def compute_density(self, expr):
        raise NotImplementedError()

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """
        return {RandomSymbol(self.symbol, self): self.distribution.sample(size,
                    library=library, seed=seed)}

    def probability(self, condition):
        raise NotImplementedError()


class SampleJointScipy:
    """Returns the sample from scipy of the given distribution"""
    def __new__(cls, dist, size, seed=None):
        return cls._sample_scipy(dist, size, seed)

    @classmethod
    def _sample_scipy(cls, dist, size, seed):
        """Sample from SciPy."""

        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        from scipy import stats as scipy_stats
        scipy_rv_map = {
            'MultivariateNormalDistribution': lambda dist, size: scipy_stats.multivariate_normal.rvs(
                mean=matrix2numpy(dist.mu).flatten(),
                cov=matrix2numpy(dist.sigma), size=size, random_state=rand_state),
            'MultivariateBetaDistribution': lambda dist, size: scipy_stats.dirichlet.rvs(
                alpha=list2numpy(dist.alpha, float).flatten(), size=size, random_state=rand_state),
            'MultinomialDistribution': lambda dist, size: scipy_stats.multinomial.rvs(
                n=int(dist.n), p=list2numpy(dist.p, float).flatten(), size=size, random_state=rand_state)
        }

        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        dist_list = scipy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        samples = scipy_rv_map[dist.__class__.__name__](dist, size)
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))

class SampleJointNumpy:
    """Returns the sample from numpy of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_numpy(dist, size, seed)

    @classmethod
    def _sample_numpy(cls, dist, size, seed):
        """Sample from NumPy."""

        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        numpy_rv_map = {
            'MultivariateNormalDistribution': lambda dist, size: rand_state.multivariate_normal(
                mean=matrix2numpy(dist.mu, float).flatten(),
                cov=matrix2numpy(dist.sigma, float), size=size),
            'MultivariateBetaDistribution': lambda dist, size: rand_state.dirichlet(
                alpha=list2numpy(dist.alpha, float).flatten(), size=size),
            'MultinomialDistribution': lambda dist, size: rand_state.multinomial(
                n=int(dist.n), pvals=list2numpy(dist.p, float).flatten(), size=size)
        }

        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        dist_list = numpy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        samples = numpy_rv_map[dist.__class__.__name__](dist, prod(size))
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))

class SampleJointPymc:
    """Returns the sample from pymc of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_pymc(dist, size, seed)

    @classmethod
    def _sample_pymc(cls, dist, size, seed):
        """Sample from PyMC."""

        try:
            import pymc
        except ImportError:
            import pymc3 as pymc
        pymc_rv_map = {
            'MultivariateNormalDistribution': lambda dist:
                pymc.MvNormal('X', mu=matrix2numpy(dist.mu, float).flatten(),
                cov=matrix2numpy(dist.sigma, float), shape=(1, dist.mu.shape[0])),
            'MultivariateBetaDistribution': lambda dist:
                pymc.Dirichlet('X', a=list2numpy(dist.alpha, float).flatten()),
            'MultinomialDistribution': lambda dist:
                pymc.Multinomial('X', n=int(dist.n),
                p=list2numpy(dist.p, float).flatten(), shape=(1, len(dist.p)))
        }

        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        dist_list = pymc_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        import logging
        logging.getLogger("pymc3").setLevel(logging.ERROR)
        with pymc.Model():
            pymc_rv_map[dist.__class__.__name__](dist)
            samples = pymc.sample(draws=prod(size), chains=1, progressbar=False, random_seed=seed, return_inferencedata=False, compute_convergence_checks=False)[:]['X']
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))


_get_sample_class_jrv = {
    'scipy': SampleJointScipy,
    'pymc3': SampleJointPymc,
    'pymc': SampleJointPymc,
    'numpy': SampleJointNumpy
}

class JointDistribution(Distribution, NamedArgsMixin):
    """
    Represented by the random variables part of the joint distribution.
    Contains methods for PDF, CDF, sampling, marginal densities, etc.
    """

    _argnames = ('pdf', )

    def __new__(cls, *args):
        args = list(map(sympify, args))
        for i in range(len(args)):
            if isinstance(args[i], list):
                args[i] = ImmutableMatrix(args[i])
        return Basic.__new__(cls, *args)

    @property
    def domain(self):
        return ProductDomain(self.symbols)

    @property
    def pdf(self):
        return self.density.args[1]

    def cdf(self, other):
        if not isinstance(other, dict):
            raise ValueError("%s should be of type dict, got %s"%(other, type(other)))
        rvs = other.keys()
        _set = self.domain.set.sets
        expr = self.pdf(tuple(i.args[0] for i in self.symbols))
        for i in range(len(other)):
            if rvs[i].is_Continuous:
                density = Integral(expr, (rvs[i], _set[i].inf,
                    other[rvs[i]]))
            elif rvs[i].is_Discrete:
                density = Sum(expr, (rvs[i], _set[i].inf,
                    other[rvs[i]]))
        return density

    def sample(self, size=(), library='scipy', seed=None):
        """ A random realization from the distribution """

        libraries = ('scipy', 'numpy', 'pymc3', 'pymc')
        if library not in libraries:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                        % str(library))
        if not import_module(library):
            raise ValueError("Failed to import %s" % library)

        samps = _get_sample_class_jrv[library](self, size, seed=seed)

        if samps is not None:
            return samps
        raise NotImplementedError(
                "Sampling for %s is not currently implemented from %s"
                % (self.__class__.__name__, library)
                )

    def __call__(self, *args):
        return self.pdf(*args)

class JointRandomSymbol(RandomSymbol):
    """
    Representation of random symbols with joint probability distributions
    to allow indexing."
    """
    def __getitem__(self, key):
        if isinstance(self.pspace, JointPSpace):
            if (self.pspace.component_count <= key) == True:
                raise ValueError("Index keys for %s can only up to %s." %
                    (self.name, self.pspace.component_count - 1))
            return Indexed(self, key)



class MarginalDistribution(Distribution):
    """
    Represents the marginal distribution of a joint probability space.

    Initialised using a probability distribution and random variables(or
    their indexed components) which should be a part of the resultant
    distribution.
    """

    def __new__(cls, dist, *rvs):
        if len(rvs) == 1 and iterable(rvs[0]):
            rvs = tuple(rvs[0])
        if not all(isinstance(rv, (Indexed, RandomSymbol)) for rv in rvs):
            raise ValueError(filldedent('''Marginal distribution can be
             intitialised only in terms of random variables or indexed random
             variables'''))
        rvs = Tuple.fromiter(rv for rv in rvs)
        if not isinstance(dist, JointDistribution) and len(random_symbols(dist)) == 0:
            return dist
        return Basic.__new__(cls, dist, rvs)

    def check(self):
        pass

    @property
    def set(self):
        rvs = [i for i in self.args[1] if isinstance(i, RandomSymbol)]
        return ProductSet(*[rv.pspace.set for rv in rvs])

    @property
    def symbols(self):
        rvs = self.args[1]
        return {rv.pspace.symbol for rv in rvs}

    def pdf(self, *x):
        expr, rvs = self.args[0], self.args[1]
        marginalise_out = [i for i in random_symbols(expr) if i not in rvs]
        if isinstance(expr, JointDistribution):
            count = len(expr.domain.args)
            x = Dummy('x', real=True)
            syms = tuple(Indexed(x, i) for i in count)
            expr = expr.pdf(syms)
        else:
            syms = tuple(rv.pspace.symbol if isinstance(rv, RandomSymbol) else rv.args[0] for rv in rvs)
        return Lambda(syms, self.compute_pdf(expr, marginalise_out))(*x)

    def compute_pdf(self, expr, rvs):
        for rv in rvs:
            lpdf = 1
            if isinstance(rv, RandomSymbol):
                lpdf = rv.pspace.pdf
            expr = self.marginalise_out(expr*lpdf, rv)
        return expr

    def marginalise_out(self, expr, rv):
        from sympy.concrete.summations import Sum
        if isinstance(rv, RandomSymbol):
            dom = rv.pspace.set
        elif isinstance(rv, Indexed):
            dom = rv.base.component_domain(
                rv.pspace.component_domain(rv.args[1]))
        expr = expr.xreplace({rv: rv.pspace.symbol})
        if rv.pspace.is_Continuous:
            #TODO: Modify to support integration
            #for all kinds of sets.
            expr = Integral(expr, (rv.pspace.symbol, dom))
        elif rv.pspace.is_Discrete:
            #incorporate this into `Sum`/`summation`
            if dom in (S.Integers, S.Naturals, S.Naturals0):
                dom = (dom.inf, dom.sup)
            expr = Sum(expr, (rv.pspace.symbol, dom))
        return expr

    def __call__(self, *args):
        return self.pdf(*args)
