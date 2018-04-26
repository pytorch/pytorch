from numbers import Number
import torch
import math
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions import Chi2
from torch.distributions.utils import broadcast_all


class StudentT(Distribution):
    r"""
    Creates a Student's t-distribution parameterized by `df`.

    Example::

        >>> m = StudentT(torch.tensor([2.0]))
        >>> m.sample()  # Student's t-distributed with degrees of freedom=2
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        df (float or Tensor): degrees of freedom
    """
    arg_constraints = {'df': constraints.positive, 'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        m = self.loc.clone()
        m[self.df <= 1] = float('nan')
        return m

    @property
    def variance(self):
        m = self.df.clone()
        m[self.df > 2] = self.scale[self.df > 2].pow(2) * self.df[self.df > 2] / (self.df[self.df > 2] - 2)
        m[(self.df <= 2) & (self.df > 1)] = float('inf')
        m[self.df <= 1] = float('nan')
        return m

    def __init__(self, df, loc=0., scale=1., validate_args=None):
        self.df, self.loc, self.scale = broadcast_all(df, loc, scale)
        self._chi2 = Chi2(df)
        batch_shape = torch.Size() if isinstance(df, Number) else self.df.size()
        super(StudentT, self).__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        # NOTE: This does not agree with scipy implementation as much as other distributions.
        # (see https://github.com/fritzo/notebooks/blob/master/debug-student-t.ipynb). Using DoubleTensor
        # parameters seems to help.

        #   X ~ Normal(0, 1)
        #   Z ~ Chi2(df)
        #   Y = X / sqrt(Z / df) ~ StudentT(df)
        shape = self._extended_shape(sample_shape)
        X = self.df.new(shape).normal_()
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df)
        return self.loc + self.scale * Y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (value - self.loc) / self.scale
        Z = (self.scale.log() +
             0.5 * self.df.log() +
             0.5 * math.log(math.pi) +
             torch.lgamma(0.5 * self.df) -
             torch.lgamma(0.5 * (self.df + 1.)))
        return -0.5 * (self.df + 1.) * torch.log1p(y**2. / self.df) - Z

    def entropy(self):
        lbeta = torch.lgamma(0.5 * self.df) + math.lgamma(0.5) - torch.lgamma(0.5 * (self.df + 1))
        return (self.scale.log() +
                0.5 * (self.df + 1) *
                (torch.digamma(0.5 * (self.df + 1)) - torch.digamma(0.5 * self.df)) +
                0.5 * self.df.log() + lbeta)
