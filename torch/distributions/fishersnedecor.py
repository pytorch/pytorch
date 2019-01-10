from numbers import Number
import torch
import math
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.utils import broadcast_all, _finfo


class FisherSnedecor(Distribution):
    r"""
    Creates a Fisher-Snedecor distribution parameterized by `df1` and `df2`.

    Example::

        >>> m = FisherSnedecor(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2
         0.2453
        [torch.FloatTensor of size 1]

    Args:
        df1 (float or Tensor): degrees of freedom parameter 1
        df2 (float or Tensor): degrees of freedom parameter 2
    """
    arg_constraints = {'df1': constraints.positive, 'df2': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, df1, df2, validate_args=None):
        self.df1, self.df2 = broadcast_all(df1, df2)
        self._gamma1 = Gamma(self.df1 * 0.5, self.df1)
        self._gamma2 = Gamma(self.df2 * 0.5, self.df2)

        if isinstance(df1, Number) and isinstance(df2, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.df1.size()
        super(FisherSnedecor, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        df2 = self.df2.clone()
        df2[df2 <= 2] = float('nan')
        return df2 / (df2 - 2)

    @property
    def variance(self):
        df2 = self.df2.clone()
        df2[df2 <= 4] = float('nan')
        return 2 * df2.pow(2) * (self.df1 + df2 - 2) / (self.df1 * (df2 - 2).pow(2) * (df2 - 4))

    def rsample(self, sample_shape=torch.Size(())):
        shape = self._extended_shape(sample_shape)
        #   X1 ~ Gamma(df1 / 2, 1 / df1), X2 ~ Gamma(df2 / 2, 1 / df2)
        #   Y = df2 * df1 * X1 / (df1 * df2 * X2) = X1 / X2 ~ F(df1, df2)
        X1 = self._gamma1.rsample(sample_shape).view(shape)
        X2 = self._gamma2.rsample(sample_shape).view(shape)
        X2.clamp_(min=_finfo(X2).tiny)
        Y = X1 / X2
        Y.clamp_(min=_finfo(X2).tiny)
        return Y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        ct1 = self.df1 * 0.5
        ct2 = self.df2 * 0.5
        ct3 = self.df1 / self.df2
        t1 = (ct1 + ct2).lgamma() - ct1.lgamma() - ct2.lgamma()
        t2 = ct1 * ct3.log() + (ct1 - 1) * torch.log(value)
        t3 = (ct1 + ct2) * torch.log1p(ct3 * value)
        return t1 + t2 - t3
