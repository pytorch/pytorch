from functools import singledispatch
from sympy.external import import_module
from sympy.stats.crv_types import BetaDistribution, CauchyDistribution, ChiSquaredDistribution, ExponentialDistribution, \
    GammaDistribution, LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, \
    GaussianInverseDistribution
from sympy.stats.drv_types import PoissonDistribution, GeometricDistribution, NegativeBinomialDistribution
from sympy.stats.frv_types import BinomialDistribution, BernoulliDistribution


try:
    import pymc
except ImportError:
    pymc = import_module('pymc3')

@singledispatch
def do_sample_pymc(dist):
    return None


# CRV:

@do_sample_pymc.register(BetaDistribution)
def _(dist: BetaDistribution):
    return pymc.Beta('X', alpha=float(dist.alpha), beta=float(dist.beta))


@do_sample_pymc.register(CauchyDistribution)
def _(dist: CauchyDistribution):
    return pymc.Cauchy('X', alpha=float(dist.x0), beta=float(dist.gamma))


@do_sample_pymc.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution):
    return pymc.ChiSquared('X', nu=float(dist.k))


@do_sample_pymc.register(ExponentialDistribution)
def _(dist: ExponentialDistribution):
    return pymc.Exponential('X', lam=float(dist.rate))


@do_sample_pymc.register(GammaDistribution)
def _(dist: GammaDistribution):
    return pymc.Gamma('X', alpha=float(dist.k), beta=1 / float(dist.theta))


@do_sample_pymc.register(LogNormalDistribution)
def _(dist: LogNormalDistribution):
    return pymc.Lognormal('X', mu=float(dist.mean), sigma=float(dist.std))


@do_sample_pymc.register(NormalDistribution)
def _(dist: NormalDistribution):
    return pymc.Normal('X', float(dist.mean), float(dist.std))


@do_sample_pymc.register(GaussianInverseDistribution)
def _(dist: GaussianInverseDistribution):
    return pymc.Wald('X', mu=float(dist.mean), lam=float(dist.shape))


@do_sample_pymc.register(ParetoDistribution)
def _(dist: ParetoDistribution):
    return pymc.Pareto('X', alpha=float(dist.alpha), m=float(dist.xm))


@do_sample_pymc.register(UniformDistribution)
def _(dist: UniformDistribution):
    return pymc.Uniform('X', lower=float(dist.left), upper=float(dist.right))


# DRV:

@do_sample_pymc.register(GeometricDistribution)
def _(dist: GeometricDistribution):
    return pymc.Geometric('X', p=float(dist.p))


@do_sample_pymc.register(NegativeBinomialDistribution)
def _(dist: NegativeBinomialDistribution):
    return pymc.NegativeBinomial('X', mu=float((dist.p * dist.r) / (1 - dist.p)),
                                  alpha=float(dist.r))


@do_sample_pymc.register(PoissonDistribution)
def _(dist: PoissonDistribution):
    return pymc.Poisson('X', mu=float(dist.lamda))


# FRV:

@do_sample_pymc.register(BernoulliDistribution)
def _(dist: BernoulliDistribution):
    return pymc.Bernoulli('X', p=float(dist.p))


@do_sample_pymc.register(BinomialDistribution)
def _(dist: BinomialDistribution):
    return pymc.Binomial('X', n=int(dist.n), p=float(dist.p))
