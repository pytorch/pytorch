import torch
from torch.autograd import Variable
import warnings


class Distribution(object):
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    has_rsample = False
    has_enumerate_support = False

    def __init__(self, batch_shape=torch.Size(), event_shape=torch.Size()):
        self._batch_shape = batch_shape
        self._event_shape = event_shape

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        z = self.rsample(sample_shape)
        return z.detach() if hasattr(z, 'detach') else z

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
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

    def _extended_shape(self, sample_shape=()):
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (torch.Size): the size of the sample to be drawn.
        """
        shape = sample_shape + self._batch_shape + self._event_shape
        if not shape:
            shape = torch.Size((1,))
        return shape

    def _validate_log_prob_arg(self, value):
        """
        Argument validation for `log_prob` methods. The rightmost dimensions
        of a value to be scored via `log_prob` must agree with the distribution's
        batch and event shapes.

        Args:
            value (Tensor or Variable): the tensor whose log probability is to be
                computed by the `log_prob` method.
        Raises
            ValueError: when the rightmost dimensions of `value` do not match the
                distribution's batch and event shapes.
        """
        if not (torch.is_tensor(value) or isinstance(value, Variable)):
            raise ValueError('The value argument to log_prob must be a Tensor or Variable instance.')

        event_dim_start = len(value.size()) - len(self._event_shape)
        if value.size()[event_dim_start:] != self._event_shape:
            raise ValueError('The right-most size of value must match event_shape: {} vs {}.'.
                             format(value.size(), self._event_shape))

        actual_shape = value.size()
        expected_shape = self._batch_shape + self._event_shape
        for i, j in zip(reversed(actual_shape), reversed(expected_shape)):
            if i != 1 and j != 1 and i != j:
                raise ValueError('Value is not broadcastable with batch_shape+event_shape: {} vs {}.'.
                                 format(actual_shape, expected_shape))
