import torch
from torch.distributions.distribution import Distribution


class ExponentialFamily(Distribution):
    r"""
    ExponentialFamily is the abstract base class for probability distributions belonging to an
    exponential family, whose probability mass/density function has the form is defined below

    .. math::

        p_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))

    where :math:`\theta` denotes the natural parameters, :math:`t(x)` denotes the sufficient statistic,
    :math:`F(\theta)` is the log normalizer function for a given family and :math:`k(x)` is the carrier
    measure.

    Note:
        This class is an intermediary between the `Distribution` class and distributions which belong
        to an exponential family mainly to check the correctness of the `.entropy()` and analytic KL
        divergence methods. We use this class to compute the entropy and KL divergence using the AD
        framework and Bregman divergences (courtesy of: Frank Nielsen and Richard Nock, Entropies and
        Cross-entropies of Exponential Families).
    """

    @property
    def _natural_params(self):
        """
        Abstract method for natural parameters. Returns a tuple of Tensors based
        on the distribution
        """
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        """
        Abstract method for log normalizer function. Returns a log normalizer based on
        the distribution and input
        """
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self):
        """
        Abstract method for expected carrier measure, which is required for computing
        entropy.
        """
        raise NotImplementedError

    def entropy(self):
        """
        Method to compute the entropy using Bregman divergence of the log normalizer.
        """
        result = -self._mean_carrier_measure
        nparams = [p.detach().requires_grad_() for p in self._natural_params]
        lg_normal = self._log_normalizer(*nparams)
        gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
        result += lg_normal
        for np, g in zip(nparams, gradients):
            result -= np * g
        return result

    @staticmethod
    def _from_natural_params(*params):
        raise NotImplementedError

    def normalized_product(self, dim=-1, keepdim=False):
        r"""
        Returns a new distribution object whose probability mass/density function is the normalized product
        of the probability mass/density functions along given axis. The product of exponential family
        distributions results in a distribution in the same family, but with new natural parameters :math:`\sum(\theta)`

        .. math::

            \frac{1}{c} \prod_i p_{F}(x; \theta_i) = \exp(\langle t(x), \sum_i\theta_i\rangle - F(\sum_i\theta_i) + k(x))

        where :math:`c` is the normalization constant and :math:`i` is the index along given dimension.

        For a batched distribution d and a single observation x::

            d.prod(dim).log_prob(x) = d.log_prob(x).sum(dim) + constant

        This might be useful for models such as Product-of-Experts (Hinton, 1999):

        .. math::

            p(\textbf{d}; \theta_1...\theta_n) = \frac{\prod_m p_m(\textbf{d}|\theta_m)}{\sum_i\prod_m p_m(\textbf{c}_i|\theta_m)}

        where :math:`\textbf{d}` is a data vector in a discrete space, :math:`\theta_m` is all the parameters of individual model m,
        :math:`p_m(\textbf{d}|\theta_m)` is the probability of :math:`\textbf{d}` under model :math:`m`, and :math:`i` is an index
        over all possible vectors in the data space.
        """

        if dim >= 0:
            dim = dim - len(self.batch_shape) - 1
        dim = dim - len(self.event_shape)

        # This assumes all parameters are univariate, and will need to be fixed to support multivariate parameters. TODO
        params = [p.sum(dim, keepdim=keepdim) for p in self._natural_params]
        return self._from_natural_params(*params)
