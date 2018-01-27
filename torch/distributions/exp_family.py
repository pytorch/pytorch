import torch
from torch.distributions.distribution import Distribution
from torch.autograd import Variable


class ExponentialFamily(Distribution):
    r"""
    ExponentialFamily is the abstract base class for probability distributions belonging to an
    exponential family, whose probability mass/density function has the form is defined below

    ..math::

        p_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle) - F(\theta) + k(x))

    where :math:`\theta` denotes the natural parameters, :math:`t(x)` denotes the sufficient statistic,
    :math:`F(\theta)` is the log normalizer function for a given family and :math:`k(x)` is the carrier
    measure.

    Note:
        This class is an intermediary between the `Distribution` class and distributions which belong
        to an exponential family mainly to check the correctness of the `.entropy()` and analytic KL
        divergence methods. We use this class to compute the entropy and KL divergence using the AD frame-
        work and Bregman divergences (courtesy of: Frank Nielsen and Richard Nock, Entropies and
        Cross-entropies of Exponential Families).
    """

    @property
    def natural_params(self):
        """
        Abstract method for natural parameters. Returns a tuple of Variables based
        on the distribution
        """
        raise NotImplementedError

    def log_normalizer(self):
        """
        Abstract method for log normalizer function. Returns a log normalizer based on
        the distribution and input
        """
        raise NotImplementedError

    @property
    def mean_carrier_measure(self):
        """
        Abstract method for expected carrier measure, which is required for computing
        entropy.
        """
        raise NotImplementedError

    def entropy(self):
        """
        Method to compute the entropy using Bregman divergence of the log normalizer.
        """
        result = -self.mean_carrier_measure
        nparams = [Variable(p, requires_grad=True) for p in self.natural_params]
        lg_normal = self.log_normalizer(*nparams)
        gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
        result += lg_normal.clone().data
        for np, g in zip(nparams, gradients):
            result -= np.data * g.data
        return result

    def kl_divergence(self, dist):
        """
        Method to compute the KL-divergence between self and dist :math:`KL(self || dist)`
        """
        if not type(self) == type(dist):
            raise ValueError("The cross KL-divergence between different exponential families cannot \
                                be computed")
        self_nparams = [Variable(p, requires_grad=True) for p in self.natural_params]
        dist_nparams = dist.natural_params
        lg_normal = self.log_normalizer(*self_nparams)
        gradients = torch.autograd.grad(lg_normal.sum(), self_nparams, create_graph=True)
        result = dist.log_normalizer(*dist_nparams) - lg_normal.clone().data
        for snp, dnp, g in zip(self_nparams, dist_nparams, gradients):
            term = (dnp - snp.data) * g.data
            for _ in range(len(dist.event_shape)):
                term = term.sum(-1)
            result -= term
        return result
