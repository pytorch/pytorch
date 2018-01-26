import torch
from torch.distributions.distribution import Distribution
from torch.autograd import Variable


class ExponentialFamily(Distribution):
    r"""
    ExponentialFamily is the abstract base class for probability distributions belonging to an
    exponential family, whose form is defined below

    ..math::

        p_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle) - F(\theta) + k(x))

    where :math:`\theta` denotes the natural parameters, :math:`t(x)` denotes the sufficient statistic,
    :math:`F(\theta)` is the log normalizer function for a given family and :math:`k(x)` is the carrier
    measure.
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
        Abstract method for log normalizer function. Returns a Variable of shape
        `batch_shape` based on the distribution
        """
        raise NotImplementedError

    @property
    def mean_carrier_measure(self):
        """
        Abstract method for expected carrier measure, which is required for computing
        entropy
        """
        raise NotImplementedError

    def entropy(self):
        """
        Method to compute the entropy using Bregman divergence of the log normalizer
        """
        if not (self.mean_carrier_measure == 0):
            raise ValueError("There is no closed form solution for non-zero carrier measure")
        nparams = [p for p in self.natural_params if p.requires_grad]
        lg_normal = self.log_normalizer()
        gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
        result = lg_normal.clone()
        for np, g in zip(nparams, gradients):
            result -= np * g
        return result

    def kl_divergence(self, dist):
        """
        Method to compute the KL-divergence between self and dist :math:`KL(self || dist)`
        """
        if not type(self) == type(dist):
            raise ValueError("The cross KL-divergence between different exponential families cannot \
                                be computed")
        self_nparams = [p for p in self.natural_params if p.requires_grad]
        dist_nparams = dist.natural_params
        lg_normal = self.log_normalizer()
        gradients = torch.autograd.grad(lg_normal.sum(), self_nparams, create_graph=True)
        result = dist.log_normalizer() - lg_normal.clone()
        for snp, dnp, g in zip(self_nparams, dist_nparams, gradients):
            term = (dnp - snp) * g
            for _ in range(len(dist.event_shape)):
                term = term.sum(-1)
            result -= term
        return result
