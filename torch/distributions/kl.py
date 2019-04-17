import math
import warnings
from functools import total_ordering

import torch
from torch._six import inf

from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exponential import Exponential
from .exp_family import ExponentialFamily
from .gamma import Gamma
from .geometric import Geometric
from .gumbel import Gumbel
from .half_normal import HalfNormal
from .independent import Independent
from .laplace import Laplace
from .lowrank_multivariate_normal import (LowRankMultivariateNormal, _batch_lowrank_logdet,
                                          _batch_lowrank_mahalanobis)
from .multivariate_normal import (MultivariateNormal, _batch_mahalanobis)
from .normal import Normal
from .one_hot_categorical import OneHotCategorical
from .pareto import Pareto
from .poisson import Poisson
from .transformed_distribution import TransformedDistribution
from .uniform import Uniform
from .utils import _sum_rightmost

_KL_REGISTRY = {}  # Source of truth mapping a few general (type, type) pairs to functions.
_KL_MEMOIZE = {}  # Memoized version mapping many specific (type, type) pairs to functions.


def register_kl(type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised. For example to
    resolve the ambiguous situation::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(p, q): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(p, q): ...

    you should register a third most-specific implementation, e.g.::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
        type_q (type): A subclass of :class:`~torch.distributions.Distribution`.
    """
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError('Expected type_p to be a Distribution subclass but got {}'.format(type_p))
    if not isinstance(type_q, type) and issubclass(type_q, Distribution):
        raise TypeError('Expected type_q to be a Distribution subclass but got {}'.format(type_q))

    def decorator(fun):
        _KL_REGISTRY[type_p, type_q] = fun
        _KL_MEMOIZE.clear()  # reset since lookup order may have changed
        return fun

    return decorator


@total_ordering
class _Match(object):
    __slots__ = ['types']

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


def _dispatch_kl(type_p, type_q):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [(super_p, super_q) for super_p, super_q in _KL_REGISTRY
               if issubclass(type_p, super_p) and issubclass(type_q, super_q)]
    if not matches:
        return NotImplemented
    # Check that the left- and right- lexicographic orders agree.
    left_p, left_q = min(_Match(*m) for m in matches).types
    right_q, right_p = min(_Match(*reversed(m)) for m in matches).types
    left_fun = _KL_REGISTRY[left_p, left_q]
    right_fun = _KL_REGISTRY[right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn('Ambiguous kl_divergence({}, {}). Please register_kl({}, {})'.format(
            type_p.__name__, type_q.__name__, left_p.__name__, right_q.__name__),
            RuntimeWarning)
    return left_fun


def _infinite_like(tensor):
    """
    Helper function for obtaining infinite KL Divergence throughout
    """
    return tensor.new_tensor(inf).expand_as(tensor)


def _x_log_x(tensor):
    """
    Utility function for calculating x log x
    """
    return tensor * tensor.log()


def _batch_trace_XXT(bmat):
    """
    Utility function for calculating the trace of XX^{T} with X having arbitrary trailing batch dimensions
    """
    n = bmat.size(-1)
    m = bmat.size(-2)
    flat_trace = bmat.reshape(-1, m * n).pow(2).sum(-1)
    return flat_trace.reshape(bmat.shape[:-2])


def kl_divergence(p, q):
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

    .. math::

        KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

    Args:
        p (Distribution): A :class:`~torch.distributions.Distribution` object.
        q (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    try:
        fun = _KL_MEMOIZE[type(p), type(q)]
    except KeyError:
        fun = _dispatch_kl(type(p), type(q))
        _KL_MEMOIZE[type(p), type(q)] = fun
    if fun is NotImplemented:
        raise NotImplementedError
    return fun(p, q)


################################################################################
# KL Divergence Implementations
################################################################################

_euler_gamma = 0.57721566490153286060

# Same distributions


@register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(p, q):
    t1 = p.probs * (p.probs / q.probs).log()
    t1[q.probs == 0] = inf
    t1[p.probs == 0] = 0
    t2 = (1 - p.probs) * ((1 - p.probs) / (1 - q.probs)).log()
    t2[q.probs == 1] = inf
    t2[p.probs == 1] = 0
    return t1 + t2


@register_kl(Beta, Beta)
def _kl_beta_beta(p, q):
    sum_params_p = p.concentration1 + p.concentration0
    sum_params_q = q.concentration1 + q.concentration0
    t1 = q.concentration1.lgamma() + q.concentration0.lgamma() + (sum_params_p).lgamma()
    t2 = p.concentration1.lgamma() + p.concentration0.lgamma() + (sum_params_q).lgamma()
    t3 = (p.concentration1 - q.concentration1) * torch.digamma(p.concentration1)
    t4 = (p.concentration0 - q.concentration0) * torch.digamma(p.concentration0)
    t5 = (sum_params_q - sum_params_p) * torch.digamma(sum_params_p)
    return t1 - t2 + t3 + t4 + t5


@register_kl(Binomial, Binomial)
def _kl_binomial_binomial(p, q):
    # from https://math.stackexchange.com/questions/2214993/
    # kullback-leibler-divergence-for-binomial-distributions-p-and-q
    if (p.total_count < q.total_count).any():
        raise NotImplementedError('KL between Binomials where q.total_count > p.total_count is not implemented')
    kl = p.total_count * (p.probs * (p.logits - q.logits) + (-p.probs).log1p() - (-q.probs).log1p())
    inf_idxs = p.total_count > q.total_count
    kl[inf_idxs] = _infinite_like(kl[inf_idxs])
    return kl


@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    t = p.probs * (p.logits - q.logits)
    t[(q.probs == 0).expand_as(t)] = inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)


@register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(p, q):
    # From http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
    sum_p_concentration = p.concentration.sum(-1)
    sum_q_concentration = q.concentration.sum(-1)
    t1 = sum_p_concentration.lgamma() - sum_q_concentration.lgamma()
    t2 = (p.concentration.lgamma() - q.concentration.lgamma()).sum(-1)
    t3 = p.concentration - q.concentration
    t4 = p.concentration.digamma() - sum_p_concentration.digamma().unsqueeze(-1)
    return t1 - t2 + (t3 * t4).sum(-1)


@register_kl(Exponential, Exponential)
def _kl_exponential_exponential(p, q):
    rate_ratio = q.rate / p.rate
    t1 = -rate_ratio.log()
    return t1 + rate_ratio - 1


@register_kl(ExponentialFamily, ExponentialFamily)
def _kl_expfamily_expfamily(p, q):
    if not type(p) == type(q):
        raise NotImplementedError("The cross KL-divergence between different exponential families cannot \
                            be computed using Bregman divergences")
    p_nparams = [np.detach().requires_grad_() for np in p._natural_params]
    q_nparams = q._natural_params
    lg_normal = p._log_normalizer(*p_nparams)
    gradients = torch.autograd.grad(lg_normal.sum(), p_nparams, create_graph=True)
    result = q._log_normalizer(*q_nparams) - lg_normal.clone()
    for pnp, qnp, g in zip(p_nparams, q_nparams, gradients):
        term = (qnp - pnp) * g
        result -= _sum_rightmost(term, len(q.event_shape))
    return result


@register_kl(Gamma, Gamma)
def _kl_gamma_gamma(p, q):
    t1 = q.concentration * (p.rate / q.rate).log()
    t2 = torch.lgamma(q.concentration) - torch.lgamma(p.concentration)
    t3 = (p.concentration - q.concentration) * torch.digamma(p.concentration)
    t4 = (q.rate - p.rate) * (p.concentration / p.rate)
    return t1 + t2 + t3 + t4


@register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(p, q):
    ct1 = p.scale / q.scale
    ct2 = q.loc / q.scale
    ct3 = p.loc / q.scale
    t1 = -ct1.log() - ct2 + ct3
    t2 = ct1 * _euler_gamma
    t3 = torch.exp(ct2 + (1 + ct1).lgamma() - ct3)
    return t1 + t2 + t3 - (1 + _euler_gamma)


@register_kl(Geometric, Geometric)
def _kl_geometric_geometric(p, q):
    return -p.entropy() - torch.log1p(-q.probs) / p.probs - q.logits


@register_kl(HalfNormal, HalfNormal)
def _kl_halfnormal_halfnormal(p, q):
    return _kl_normal_normal(p.base_dist, q.base_dist)


@register_kl(Laplace, Laplace)
def _kl_laplace_laplace(p, q):
    # From http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf
    scale_ratio = p.scale / q.scale
    loc_abs_diff = (p.loc - q.loc).abs()
    t1 = -scale_ratio.log()
    t2 = loc_abs_diff / q.scale
    t3 = scale_ratio * torch.exp(-loc_abs_diff / p.scale)
    return t1 + t2 + t3 - 1


@register_kl(LowRankMultivariateNormal, LowRankMultivariateNormal)
def _kl_lowrankmultivariatenormal_lowrankmultivariatenormal(p, q):
    if p.event_shape != q.event_shape:
        raise ValueError("KL-divergence between two Low Rank Multivariate Normals with\
                          different event shapes cannot be computed")

    term1 = (_batch_lowrank_logdet(q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag,
                                   q._capacitance_tril) -
             _batch_lowrank_logdet(p._unbroadcasted_cov_factor, p._unbroadcasted_cov_diag,
                                   p._capacitance_tril))
    term3 = _batch_lowrank_mahalanobis(q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag,
                                       q.loc - p.loc,
                                       q._capacitance_tril)
    # Expands term2 according to
    # inv(qcov) @ pcov = [inv(qD) - inv(qD) @ qW @ inv(qC) @ qW.T @ inv(qD)] @ (pW @ pW.T + pD)
    #                  = [inv(qD) - A.T @ A] @ (pD + pW @ pW.T)
    qWt_qDinv = (q._unbroadcasted_cov_factor.transpose(-1, -2) /
                 q._unbroadcasted_cov_diag.unsqueeze(-2))
    A = torch.triangular_solve(qWt_qDinv, q._capacitance_tril, upper=False)[0]
    term21 = (p._unbroadcasted_cov_diag / q._unbroadcasted_cov_diag).sum(-1)
    term22 = _batch_trace_XXT(p._unbroadcasted_cov_factor *
                              q._unbroadcasted_cov_diag.rsqrt().unsqueeze(-1))
    term23 = _batch_trace_XXT(A * p._unbroadcasted_cov_diag.sqrt().unsqueeze(-2))
    term24 = _batch_trace_XXT(A.matmul(p._unbroadcasted_cov_factor))
    term2 = term21 + term22 - term23 - term24
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


@register_kl(MultivariateNormal, LowRankMultivariateNormal)
def _kl_multivariatenormal_lowrankmultivariatenormal(p, q):
    if p.event_shape != q.event_shape:
        raise ValueError("KL-divergence between two (Low Rank) Multivariate Normals with\
                          different event shapes cannot be computed")

    term1 = (_batch_lowrank_logdet(q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag,
                                   q._capacitance_tril) -
             2 * p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    term3 = _batch_lowrank_mahalanobis(q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag,
                                       q.loc - p.loc,
                                       q._capacitance_tril)
    # Expands term2 according to
    # inv(qcov) @ pcov = [inv(qD) - inv(qD) @ qW @ inv(qC) @ qW.T @ inv(qD)] @ p_tril @ p_tril.T
    #                  = [inv(qD) - A.T @ A] @ p_tril @ p_tril.T
    qWt_qDinv = (q._unbroadcasted_cov_factor.transpose(-1, -2) /
                 q._unbroadcasted_cov_diag.unsqueeze(-2))
    A = torch.triangular_solve(qWt_qDinv, q._capacitance_tril, upper=False)[0]
    term21 = _batch_trace_XXT(p._unbroadcasted_scale_tril *
                              q._unbroadcasted_cov_diag.rsqrt().unsqueeze(-1))
    term22 = _batch_trace_XXT(A.matmul(p._unbroadcasted_scale_tril))
    term2 = term21 - term22
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


@register_kl(LowRankMultivariateNormal, MultivariateNormal)
def _kl_lowrankmultivariatenormal_multivariatenormal(p, q):
    if p.event_shape != q.event_shape:
        raise ValueError("KL-divergence between two (Low Rank) Multivariate Normals with\
                          different event shapes cannot be computed")

    term1 = (2 * q._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) -
             _batch_lowrank_logdet(p._unbroadcasted_cov_factor, p._unbroadcasted_cov_diag,
                                   p._capacitance_tril))
    term3 = _batch_mahalanobis(q._unbroadcasted_scale_tril, (q.loc - p.loc))
    # Expands term2 according to
    # inv(qcov) @ pcov = inv(q_tril @ q_tril.T) @ (pW @ pW.T + pD)
    combined_batch_shape = torch._C._infer_size(q._unbroadcasted_scale_tril.shape[:-2],
                                                p._unbroadcasted_cov_factor.shape[:-2])
    n = p.event_shape[0]
    q_scale_tril = q._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_cov_factor = p._unbroadcasted_cov_factor.expand(combined_batch_shape +
                                                      (n, p.cov_factor.size(-1)))
    p_cov_diag = (torch.diag_embed(p._unbroadcasted_cov_diag.sqrt())
                  .expand(combined_batch_shape + (n, n)))
    term21 = _batch_trace_XXT(torch.triangular_solve(p_cov_factor, q_scale_tril, upper=False)[0])
    term22 = _batch_trace_XXT(torch.triangular_solve(p_cov_diag, q_scale_tril, upper=False)[0])
    term2 = term21 + term22
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


@register_kl(MultivariateNormal, MultivariateNormal)
def _kl_multivariatenormal_multivariatenormal(p, q):
    # From https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
    if p.event_shape != q.event_shape:
        raise ValueError("KL-divergence between two Multivariate Normals with\
                          different event shapes cannot be computed")

    half_term1 = (q._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) -
                  p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    combined_batch_shape = torch._C._infer_size(q._unbroadcasted_scale_tril.shape[:-2],
                                                p._unbroadcasted_scale_tril.shape[:-2])
    n = p.event_shape[0]
    q_scale_tril = q._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_scale_tril = p._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    term2 = _batch_trace_XXT(torch.triangular_solve(p_scale_tril, q_scale_tril, upper=False)[0])
    term3 = _batch_mahalanobis(q._unbroadcasted_scale_tril, (q.loc - p.loc))
    return half_term1 + 0.5 * (term2 + term3 - n)


@register_kl(Normal, Normal)
def _kl_normal_normal(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


@register_kl(OneHotCategorical, OneHotCategorical)
def _kl_onehotcategorical_onehotcategorical(p, q):
    return _kl_categorical_categorical(p._categorical, q._categorical)


@register_kl(Pareto, Pareto)
def _kl_pareto_pareto(p, q):
    # From http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf
    scale_ratio = p.scale / q.scale
    alpha_ratio = q.alpha / p.alpha
    t1 = q.alpha * scale_ratio.log()
    t2 = -alpha_ratio.log()
    result = t1 + t2 + alpha_ratio - 1
    result[p.support.lower_bound < q.support.lower_bound] = inf
    return result


@register_kl(Poisson, Poisson)
def _kl_poisson_poisson(p, q):
    return p.rate * (p.rate.log() - q.rate.log()) - (p.rate - q.rate)


@register_kl(TransformedDistribution, TransformedDistribution)
def _kl_transformed_transformed(p, q):
    if p.transforms != q.transforms:
        raise NotImplementedError
    if p.event_shape != q.event_shape:
        raise NotImplementedError
    # extra_event_dim = len(p.event_shape) - len(p.base_dist.event_shape)
    extra_event_dim = len(p.event_shape)
    base_kl_divergence = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(base_kl_divergence, extra_event_dim)


@register_kl(Uniform, Uniform)
def _kl_uniform_uniform(p, q):
    result = ((q.high - q.low) / (p.high - p.low)).log()
    result[(q.low > p.low) | (q.high < p.high)] = inf
    return result


# Different distributions
@register_kl(Bernoulli, Poisson)
def _kl_bernoulli_poisson(p, q):
    return -p.entropy() - (p.probs * q.rate.log() - q.rate)


@register_kl(Beta, Pareto)
def _kl_beta_infinity(p, q):
    return _infinite_like(p.concentration1)


@register_kl(Beta, Exponential)
def _kl_beta_exponential(p, q):
    return -p.entropy() - q.rate.log() + q.rate * (p.concentration1 / (p.concentration1 + p.concentration0))


@register_kl(Beta, Gamma)
def _kl_beta_gamma(p, q):
    t1 = -p.entropy()
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    t3 = (q.concentration - 1) * (p.concentration1.digamma() - (p.concentration1 + p.concentration0).digamma())
    t4 = q.rate * p.concentration1 / (p.concentration1 + p.concentration0)
    return t1 + t2 - t3 + t4

# TODO: Add Beta-Laplace KL Divergence


@register_kl(Beta, Normal)
def _kl_beta_normal(p, q):
    E_beta = p.concentration1 / (p.concentration1 + p.concentration0)
    var_normal = q.scale.pow(2)
    t1 = -p.entropy()
    t2 = 0.5 * (var_normal * 2 * math.pi).log()
    t3 = (E_beta * (1 - E_beta) / (p.concentration1 + p.concentration0 + 1) + E_beta.pow(2)) * 0.5
    t4 = q.loc * E_beta
    t5 = q.loc.pow(2) * 0.5
    return t1 + t2 + (t3 - t4 + t5) / var_normal


@register_kl(Beta, Uniform)
def _kl_beta_uniform(p, q):
    result = -p.entropy() + (q.high - q.low).log()
    result[(q.low > p.support.lower_bound) | (q.high < p.support.upper_bound)] = inf
    return result


@register_kl(Exponential, Beta)
@register_kl(Exponential, Pareto)
@register_kl(Exponential, Uniform)
def _kl_exponential_infinity(p, q):
    return _infinite_like(p.rate)


@register_kl(Exponential, Gamma)
def _kl_exponential_gamma(p, q):
    ratio = q.rate / p.rate
    t1 = -q.concentration * torch.log(ratio)
    return t1 + ratio + q.concentration.lgamma() + q.concentration * _euler_gamma - (1 + _euler_gamma)


@register_kl(Exponential, Gumbel)
def _kl_exponential_gumbel(p, q):
    scale_rate_prod = p.rate * q.scale
    loc_scale_ratio = q.loc / q.scale
    t1 = scale_rate_prod.log() - 1
    t2 = torch.exp(loc_scale_ratio) * scale_rate_prod / (scale_rate_prod + 1)
    t3 = scale_rate_prod.reciprocal()
    return t1 - loc_scale_ratio + t2 + t3

# TODO: Add Exponential-Laplace KL Divergence


@register_kl(Exponential, Normal)
def _kl_exponential_normal(p, q):
    var_normal = q.scale.pow(2)
    rate_sqr = p.rate.pow(2)
    t1 = 0.5 * torch.log(rate_sqr * var_normal * 2 * math.pi)
    t2 = rate_sqr.reciprocal()
    t3 = q.loc / p.rate
    t4 = q.loc.pow(2) * 0.5
    return t1 - 1 + (t2 - t3 + t4) / var_normal


@register_kl(Gamma, Beta)
@register_kl(Gamma, Pareto)
@register_kl(Gamma, Uniform)
def _kl_gamma_infinity(p, q):
    return _infinite_like(p.concentration)


@register_kl(Gamma, Exponential)
def _kl_gamma_exponential(p, q):
    return -p.entropy() - q.rate.log() + q.rate * p.concentration / p.rate


@register_kl(Gamma, Gumbel)
def _kl_gamma_gumbel(p, q):
    beta_scale_prod = p.rate * q.scale
    loc_scale_ratio = q.loc / q.scale
    t1 = (p.concentration - 1) * p.concentration.digamma() - p.concentration.lgamma() - p.concentration
    t2 = beta_scale_prod.log() + p.concentration / beta_scale_prod
    t3 = torch.exp(loc_scale_ratio) * (1 + beta_scale_prod.reciprocal()).pow(-p.concentration) - loc_scale_ratio
    return t1 + t2 + t3

# TODO: Add Gamma-Laplace KL Divergence


@register_kl(Gamma, Normal)
def _kl_gamma_normal(p, q):
    var_normal = q.scale.pow(2)
    beta_sqr = p.rate.pow(2)
    t1 = 0.5 * torch.log(beta_sqr * var_normal * 2 * math.pi) - p.concentration - p.concentration.lgamma()
    t2 = 0.5 * (p.concentration.pow(2) + p.concentration) / beta_sqr
    t3 = q.loc * p.concentration / p.rate
    t4 = 0.5 * q.loc.pow(2)
    return t1 + (p.concentration - 1) * p.concentration.digamma() + (t2 - t3 + t4) / var_normal


@register_kl(Gumbel, Beta)
@register_kl(Gumbel, Exponential)
@register_kl(Gumbel, Gamma)
@register_kl(Gumbel, Pareto)
@register_kl(Gumbel, Uniform)
def _kl_gumbel_infinity(p, q):
    return _infinite_like(p.loc)

# TODO: Add Gumbel-Laplace KL Divergence


@register_kl(Gumbel, Normal)
def _kl_gumbel_normal(p, q):
    param_ratio = p.scale / q.scale
    t1 = (param_ratio / math.sqrt(2 * math.pi)).log()
    t2 = (math.pi * param_ratio * 0.5).pow(2) / 3
    t3 = ((p.loc + p.scale * _euler_gamma - q.loc) / q.scale).pow(2) * 0.5
    return -t1 + t2 + t3 - (_euler_gamma + 1)


@register_kl(Laplace, Beta)
@register_kl(Laplace, Exponential)
@register_kl(Laplace, Gamma)
@register_kl(Laplace, Pareto)
@register_kl(Laplace, Uniform)
def _kl_laplace_infinity(p, q):
    return _infinite_like(p.loc)


@register_kl(Laplace, Normal)
def _kl_laplace_normal(p, q):
    var_normal = q.scale.pow(2)
    scale_sqr_var_ratio = p.scale.pow(2) / var_normal
    t1 = 0.5 * torch.log(2 * scale_sqr_var_ratio / math.pi)
    t2 = 0.5 * p.loc.pow(2)
    t3 = p.loc * q.loc
    t4 = 0.5 * q.loc.pow(2)
    return -t1 + scale_sqr_var_ratio + (t2 - t3 + t4) / var_normal - 1


@register_kl(Normal, Beta)
@register_kl(Normal, Exponential)
@register_kl(Normal, Gamma)
@register_kl(Normal, Pareto)
@register_kl(Normal, Uniform)
def _kl_normal_infinity(p, q):
    return _infinite_like(p.loc)


@register_kl(Normal, Gumbel)
def _kl_normal_gumbel(p, q):
    mean_scale_ratio = p.loc / q.scale
    var_scale_sqr_ratio = (p.scale / q.scale).pow(2)
    loc_scale_ratio = q.loc / q.scale
    t1 = var_scale_sqr_ratio.log() * 0.5
    t2 = mean_scale_ratio - loc_scale_ratio
    t3 = torch.exp(-mean_scale_ratio + 0.5 * var_scale_sqr_ratio + loc_scale_ratio)
    return -t1 + t2 + t3 - (0.5 * (1 + math.log(2 * math.pi)))

# TODO: Add Normal-Laplace KL Divergence


@register_kl(Pareto, Beta)
@register_kl(Pareto, Uniform)
def _kl_pareto_infinity(p, q):
    return _infinite_like(p.scale)


@register_kl(Pareto, Exponential)
def _kl_pareto_exponential(p, q):
    scale_rate_prod = p.scale * q.rate
    t1 = (p.alpha / scale_rate_prod).log()
    t2 = p.alpha.reciprocal()
    t3 = p.alpha * scale_rate_prod / (p.alpha - 1)
    result = t1 - t2 + t3 - 1
    result[p.alpha <= 1] = inf
    return result


@register_kl(Pareto, Gamma)
def _kl_pareto_gamma(p, q):
    common_term = p.scale.log() + p.alpha.reciprocal()
    t1 = p.alpha.log() - common_term
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    t3 = (1 - q.concentration) * common_term
    t4 = q.rate * p.alpha * p.scale / (p.alpha - 1)
    result = t1 + t2 + t3 + t4 - 1
    result[p.alpha <= 1] = inf
    return result

# TODO: Add Pareto-Laplace KL Divergence


@register_kl(Pareto, Normal)
def _kl_pareto_normal(p, q):
    var_normal = 2 * q.scale.pow(2)
    common_term = p.scale / (p.alpha - 1)
    t1 = (math.sqrt(2 * math.pi) * q.scale * p.alpha / p.scale).log()
    t2 = p.alpha.reciprocal()
    t3 = p.alpha * common_term.pow(2) / (p.alpha - 2)
    t4 = (p.alpha * common_term - q.loc).pow(2)
    result = t1 - t2 + (t3 + t4) / var_normal - 1
    result[p.alpha <= 2] = inf
    return result


@register_kl(Poisson, Bernoulli)
@register_kl(Poisson, Binomial)
def _kl_poisson_infinity(p, q):
    return _infinite_like(p.rate)


@register_kl(Uniform, Beta)
def _kl_uniform_beta(p, q):
    common_term = p.high - p.low
    t1 = torch.log(common_term)
    t2 = (q.concentration1 - 1) * (_x_log_x(p.high) - _x_log_x(p.low) - common_term) / common_term
    t3 = (q.concentration0 - 1) * (_x_log_x((1 - p.high)) - _x_log_x((1 - p.low)) + common_term) / common_term
    t4 = q.concentration1.lgamma() + q.concentration0.lgamma() - (q.concentration1 + q.concentration0).lgamma()
    result = t3 + t4 - t1 - t2
    result[(p.high > q.support.upper_bound) | (p.low < q.support.lower_bound)] = inf
    return result


@register_kl(Uniform, Exponential)
def _kl_uniform_exponetial(p, q):
    result = q.rate * (p.high + p.low) / 2 - ((p.high - p.low) * q.rate).log()
    result[p.low < q.support.lower_bound] = inf
    return result


@register_kl(Uniform, Gamma)
def _kl_uniform_gamma(p, q):
    common_term = p.high - p.low
    t1 = common_term.log()
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    t3 = (1 - q.concentration) * (_x_log_x(p.high) - _x_log_x(p.low) - common_term) / common_term
    t4 = q.rate * (p.high + p.low) / 2
    result = -t1 + t2 + t3 + t4
    result[p.low < q.support.lower_bound] = inf
    return result


@register_kl(Uniform, Gumbel)
def _kl_uniform_gumbel(p, q):
    common_term = q.scale / (p.high - p.low)
    high_loc_diff = (p.high - q.loc) / q.scale
    low_loc_diff = (p.low - q.loc) / q.scale
    t1 = common_term.log() + 0.5 * (high_loc_diff + low_loc_diff)
    t2 = common_term * (torch.exp(-high_loc_diff) - torch.exp(-low_loc_diff))
    return t1 - t2

# TODO: Uniform-Laplace KL Divergence


@register_kl(Uniform, Normal)
def _kl_uniform_normal(p, q):
    common_term = p.high - p.low
    t1 = (math.sqrt(math.pi * 2) * q.scale / common_term).log()
    t2 = (common_term).pow(2) / 12
    t3 = ((p.high + p.low - 2 * q.loc) / 2).pow(2)
    return t1 + 0.5 * (t2 + t3) / q.scale.pow(2)


@register_kl(Uniform, Pareto)
def _kl_uniform_pareto(p, q):
    support_uniform = p.high - p.low
    t1 = (q.alpha * q.scale.pow(q.alpha) * (support_uniform)).log()
    t2 = (_x_log_x(p.high) - _x_log_x(p.low) - support_uniform) / support_uniform
    result = t2 * (q.alpha + 1) - t1
    result[p.low < q.support.lower_bound] = inf
    return result


@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)
