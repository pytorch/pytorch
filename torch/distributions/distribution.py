from torch.autograd import Variable
import warnings


class Distribution(object):
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    has_rsample = False
    has_enumerate_support = False

    def sample(self, sample_shape=()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Currently only
        supports `len(sample_shape)<2`.
        """
        z = self.rsample(sample_shape)
        return z.detach() if hasattr(z, 'detach') else z

    def rsample(self, sample_shape=()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Currently only supports `len(sample_shape)<2`.
        """
        raise NotImplementedError

    def sample_n(self, n):
        """
        Generates n samples or n batches of samples if the distribution
        parameters are batched.
        """
        warnings.warn('sample_n will be deprecated. Use .sample((n,)) instead', UserWarning)
        return self.sample((n,))

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor or Variable):
        """
        raise NotImplementedError

    def enumerate_support(self):
        """
        Returns tensor containing all values supported by a discrete
        distribution. The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched variables in lock-step
        `[[0, 0], [1, 1], ...]`. To iterate over the full Cartesian product
        use `itertools.product(m.enumerate_support())`.

        Returns:
            Variable or Tensor iterating over dimension 0.
        """
        raise NotImplementedError
