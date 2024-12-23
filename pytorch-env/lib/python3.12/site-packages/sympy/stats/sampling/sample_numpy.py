from functools import singledispatch

from sympy.external import import_module
from sympy.stats.crv_types import BetaDistribution, ChiSquaredDistribution, ExponentialDistribution, GammaDistribution, \
    LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, FDistributionDistribution, GumbelDistribution, LaplaceDistribution, \
    LogisticDistribution, RayleighDistribution, TriangularDistribution
from sympy.stats.drv_types import GeometricDistribution, PoissonDistribution, ZetaDistribution
from sympy.stats.frv_types import BinomialDistribution, HypergeometricDistribution


numpy = import_module('numpy')


@singledispatch
def do_sample_numpy(dist, size, rand_state):
    return None


# CRV:

@do_sample_numpy.register(BetaDistribution)
def _(dist: BetaDistribution, size, rand_state):
    return rand_state.beta(a=float(dist.alpha), b=float(dist.beta), size=size)


@do_sample_numpy.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution, size, rand_state):
    return rand_state.chisquare(df=float(dist.k), size=size)


@do_sample_numpy.register(ExponentialDistribution)
def _(dist: ExponentialDistribution, size, rand_state):
    return rand_state.exponential(1 / float(dist.rate), size=size)

@do_sample_numpy.register(FDistributionDistribution)
def _(dist: FDistributionDistribution, size, rand_state):
    return rand_state.f(dfnum = float(dist.d1), dfden = float(dist.d2), size=size)

@do_sample_numpy.register(GammaDistribution)
def _(dist: GammaDistribution, size, rand_state):
    return rand_state.gamma(shape = float(dist.k), scale = float(dist.theta), size=size)

@do_sample_numpy.register(GumbelDistribution)
def _(dist: GumbelDistribution, size, rand_state):
    return rand_state.gumbel(loc = float(dist.mu), scale = float(dist.beta), size=size)

@do_sample_numpy.register(LaplaceDistribution)
def _(dist: LaplaceDistribution, size, rand_state):
    return rand_state.laplace(loc = float(dist.mu), scale = float(dist.b), size=size)

@do_sample_numpy.register(LogisticDistribution)
def _(dist: LogisticDistribution, size, rand_state):
    return rand_state.logistic(loc = float(dist.mu), scale = float(dist.s), size=size)

@do_sample_numpy.register(LogNormalDistribution)
def _(dist: LogNormalDistribution, size, rand_state):
    return rand_state.lognormal(mean = float(dist.mean), sigma = float(dist.std), size=size)

@do_sample_numpy.register(NormalDistribution)
def _(dist: NormalDistribution, size, rand_state):
    return rand_state.normal(loc = float(dist.mean), scale = float(dist.std), size=size)

@do_sample_numpy.register(RayleighDistribution)
def _(dist: RayleighDistribution, size, rand_state):
    return rand_state.rayleigh(scale = float(dist.sigma), size=size)

@do_sample_numpy.register(ParetoDistribution)
def _(dist: ParetoDistribution, size, rand_state):
    return (numpy.random.pareto(a=float(dist.alpha), size=size) + 1) * float(dist.xm)

@do_sample_numpy.register(TriangularDistribution)
def _(dist: TriangularDistribution, size, rand_state):
    return rand_state.triangular(left = float(dist.a), mode = float(dist.b), right = float(dist.c), size=size)

@do_sample_numpy.register(UniformDistribution)
def _(dist: UniformDistribution, size, rand_state):
    return rand_state.uniform(low=float(dist.left), high=float(dist.right), size=size)


# DRV:

@do_sample_numpy.register(GeometricDistribution)
def _(dist: GeometricDistribution, size, rand_state):
    return rand_state.geometric(p=float(dist.p), size=size)


@do_sample_numpy.register(PoissonDistribution)
def _(dist: PoissonDistribution, size, rand_state):
    return rand_state.poisson(lam=float(dist.lamda), size=size)


@do_sample_numpy.register(ZetaDistribution)
def _(dist: ZetaDistribution, size, rand_state):
    return rand_state.zipf(a=float(dist.s), size=size)


# FRV:

@do_sample_numpy.register(BinomialDistribution)
def _(dist: BinomialDistribution, size, rand_state):
    return rand_state.binomial(n=int(dist.n), p=float(dist.p), size=size)

@do_sample_numpy.register(HypergeometricDistribution)
def _(dist: HypergeometricDistribution, size, rand_state):
    return rand_state.hypergeometric(ngood = int(dist.N), nbad = int(dist.m), nsample = int(dist.n), size=size)
