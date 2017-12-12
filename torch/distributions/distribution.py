from torch.autograd import Variable


class Distribution(object):
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    def sample(self):
        """
        Generates a single sample or single batch of samples if the distribution
        parameters are batched.
        """
        raise NotImplementedError

    def sample_n(self, n):
        """
        Generates n samples or n batches of samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor or Variable):
        """
        raise NotImplementedError
