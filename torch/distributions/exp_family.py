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
        if self._zero_carrier_measure:
            nparams = self.natural_params()
            lg_normal = self.log_normalizer()
            gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
            result = lg_normal.clone()
            for np, g in zip(nparams, gradients):
                result -= torch.dot(np, g)
            return result
        else:
            raise ValueError("There is no closed form solution for non-zero carrier measure")
