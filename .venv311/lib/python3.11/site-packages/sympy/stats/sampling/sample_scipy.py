from functools import singledispatch

from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import exp
from sympy.utilities.lambdify import lambdify
from sympy.external import import_module
from sympy.stats import DiscreteDistributionHandmade
from sympy.stats.crv import SingleContinuousDistribution
from sympy.stats.crv_types import ChiSquaredDistribution, ExponentialDistribution, GammaDistribution, \
    LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, BetaDistribution, \
    StudentTDistribution, CauchyDistribution
from sympy.stats.drv_types import GeometricDistribution, LogarithmicDistribution, NegativeBinomialDistribution, \
    PoissonDistribution, SkellamDistribution, YuleSimonDistribution, ZetaDistribution
from sympy.stats.frv import SingleFiniteDistribution


scipy = import_module("scipy", import_kwargs={'fromlist':['stats']})


@singledispatch
def do_sample_scipy(dist, size, seed):
    return None


# CRV

@do_sample_scipy.register(SingleContinuousDistribution)
def _(dist: SingleContinuousDistribution, size, seed):
    # if we don't need to make a handmade pdf, we won't
    import scipy.stats

    z = Dummy('z')
    handmade_pdf = lambdify(z, dist.pdf(z), ['numpy', 'scipy'])

    class scipy_pdf(scipy.stats.rv_continuous):
        def _pdf(dist, x):
            return handmade_pdf(x)

    scipy_rv = scipy_pdf(a=float(dist.set._inf),
                         b=float(dist.set._sup), name='scipy_pdf')
    return scipy_rv.rvs(size=size, random_state=seed)


@do_sample_scipy.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution, size, seed):
    # same parametrisation
    return scipy.stats.chi2.rvs(df=float(dist.k), size=size, random_state=seed)


@do_sample_scipy.register(ExponentialDistribution)
def _(dist: ExponentialDistribution, size, seed):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html#scipy.stats.expon
    return scipy.stats.expon.rvs(scale=1 / float(dist.rate), size=size, random_state=seed)


@do_sample_scipy.register(GammaDistribution)
def _(dist: GammaDistribution, size, seed):
    # https://stackoverflow.com/questions/42150965/how-to-plot-gamma-distribution-with-alpha-and-beta-parameters-in-python
    return scipy.stats.gamma.rvs(a=float(dist.k), scale=float(dist.theta), size=size, random_state=seed)


@do_sample_scipy.register(LogNormalDistribution)
def _(dist: LogNormalDistribution, size, seed):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
    return scipy.stats.lognorm.rvs(scale=float(exp(dist.mean)), s=float(dist.std), size=size, random_state=seed)


@do_sample_scipy.register(NormalDistribution)
def _(dist: NormalDistribution, size, seed):
    return scipy.stats.norm.rvs(loc=float(dist.mean), scale=float(dist.std), size=size, random_state=seed)


@do_sample_scipy.register(ParetoDistribution)
def _(dist: ParetoDistribution, size, seed):
    # https://stackoverflow.com/questions/42260519/defining-pareto-distribution-in-python-scipy
    return scipy.stats.pareto.rvs(b=float(dist.alpha), scale=float(dist.xm), size=size, random_state=seed)


@do_sample_scipy.register(StudentTDistribution)
def _(dist: StudentTDistribution, size, seed):
    return scipy.stats.t.rvs(df=float(dist.nu), size=size, random_state=seed)


@do_sample_scipy.register(UniformDistribution)
def _(dist: UniformDistribution, size, seed):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
    return scipy.stats.uniform.rvs(loc=float(dist.left), scale=float(dist.right - dist.left), size=size, random_state=seed)


@do_sample_scipy.register(BetaDistribution)
def _(dist: BetaDistribution, size, seed):
    # same parametrisation
    return scipy.stats.beta.rvs(a=float(dist.alpha), b=float(dist.beta), size=size, random_state=seed)


@do_sample_scipy.register(CauchyDistribution)
def _(dist: CauchyDistribution, size, seed):
    return scipy.stats.cauchy.rvs(loc=float(dist.x0), scale=float(dist.gamma), size=size, random_state=seed)


# DRV:

@do_sample_scipy.register(DiscreteDistributionHandmade)
def _(dist: DiscreteDistributionHandmade, size, seed):
    from scipy.stats import rv_discrete

    z = Dummy('z')
    handmade_pmf = lambdify(z, dist.pdf(z), ['numpy', 'scipy'])

    class scipy_pmf(rv_discrete):
        def _pmf(dist, x):
            return handmade_pmf(x)

    scipy_rv = scipy_pmf(a=float(dist.set._inf), b=float(dist.set._sup),
                         name='scipy_pmf')
    return scipy_rv.rvs(size=size, random_state=seed)


@do_sample_scipy.register(GeometricDistribution)
def _(dist: GeometricDistribution, size, seed):
    return scipy.stats.geom.rvs(p=float(dist.p), size=size, random_state=seed)


@do_sample_scipy.register(LogarithmicDistribution)
def _(dist: LogarithmicDistribution, size, seed):
    return scipy.stats.logser.rvs(p=float(dist.p), size=size, random_state=seed)


@do_sample_scipy.register(NegativeBinomialDistribution)
def _(dist: NegativeBinomialDistribution, size, seed):
    return scipy.stats.nbinom.rvs(n=float(dist.r), p=float(dist.p), size=size, random_state=seed)


@do_sample_scipy.register(PoissonDistribution)
def _(dist: PoissonDistribution, size, seed):
    return scipy.stats.poisson.rvs(mu=float(dist.lamda), size=size, random_state=seed)


@do_sample_scipy.register(SkellamDistribution)
def _(dist: SkellamDistribution, size, seed):
    return scipy.stats.skellam.rvs(mu1=float(dist.mu1), mu2=float(dist.mu2), size=size, random_state=seed)


@do_sample_scipy.register(YuleSimonDistribution)
def _(dist: YuleSimonDistribution, size, seed):
    return scipy.stats.yulesimon.rvs(alpha=float(dist.rho), size=size, random_state=seed)


@do_sample_scipy.register(ZetaDistribution)
def _(dist: ZetaDistribution, size, seed):
    return scipy.stats.zipf.rvs(a=float(dist.s), size=size, random_state=seed)


# FRV:

@do_sample_scipy.register(SingleFiniteDistribution)
def _(dist: SingleFiniteDistribution, size, seed):
    # scipy can handle with custom distributions

    from scipy.stats import rv_discrete
    density_ = dist.dict
    x, y = [], []
    for k, v in density_.items():
        x.append(int(k))
        y.append(float(v))
    scipy_rv = rv_discrete(name='scipy_rv', values=(x, y))
    return scipy_rv.rvs(size=size, random_state=seed)
