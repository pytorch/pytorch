import torch
from torch.distributions.distribution import Distribution
from torch.autograd import Variable


class ExponentialFamily(Distribution):

    _zero_carrier_measure = None

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

    def entropy(self):
        """
        Method to compute the entropy using Bregman divergence of the log normalizer
        """
        if not self._zero_carrier_measure:
            raise ValueError("There is no closed form solution for non-zero carrier measure")
        nparams = self.natural_params()
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
        self_nparams = self.natural_params()
        dist_nparams = dist.natural_params()
        lg_normal = self.log_normalizer()
        gradients = torch.autograd.grad(lg_normal.sum(), self_nparams, create_graph=True)
        result = dist.log_normalizer() - lg_normal.clone()
        for snp, dnp, g in zip(self_nparams, dist_nparams, gradients):
            result -= (dnp - snp) * g
        return result
