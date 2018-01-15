from numbers import Number
import torch
import math
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions import Beta, Chi2
from torch.distributions.utils import broadcast_all


class F(Distribution):
    r"""
    Creates a F-distribution parameterized by `df1` and `df2`.

    Example::

        >>> m = StudentT(torch.Tensor([1.0]), torch.Tensor([2.0]))
        >>> m.sample()  # Student's t-distributed with df1=1 and df2=2
         0.2453
        [torch.FloatTensor of size 1]

    Args:
        df1 (float or Tensor or Variable): degrees of freedom parameter 1
        df2 (float or Tensor or Variable): degrees of freedom parameter 2
    """
    params = {'df1': constraints.positive, 'df2': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, df1, df2, sampler):  # sampler is temporary, for debug purposes only
        self.df1, self.df2 = broadcast_all(df1, df2)
        self.sampler = sampler
        if sampler == 'beta':
            self._beta = Beta(df1/2, df2/2)
        elif sampler == 'chi2':
            self._chi2_df1 = Chi2(df1)
            self._chi2_df2 = Chi2(df2)
        if isinstance(df1, Number) and isinstance(df2, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.df1.size()
        super(F, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        if self.sampler == 'beta':
            #   X ~ Beta(df1/2, df2/2)
            #   Y = df2 * X / df1 * (1 - X) ~ F(df1, df2)
            X = self._beta.rsample(sample_shape)
            Y = X * self.df2 / ((1 - X) * self.df1)

        elif self.sampler == 'chi2':
            #   X1 ~ Chi2(df1), X2 ~ Chi2(df2)
            #   Y = X1 * df2 / X2 * df1 ~ F(df1, df2)
            X1 = self._chi2_df1.rsample(sample_shape)
            X2 = self._chi2_df2.rsample(sample_shape)
            Y = X1 * self.df2 / (X2 * self.df1)
        return Y

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        ct1 = (self.df1 + self.df2) / 2
        ct2 = self.df1 / 2
        ct3 = self.df1 / self.df2
        t1 = ct1.lgamma() - ct2.lgamma() - (ct1 - ct2).lgamma()
        t2 = ct2 * ct3.log() + (ct2 - 1) * torch.log(value)
        t3 = ct1 * torch.log1p(ct3 * value)
        return t1 + t2 - t3

    def entropy(self):
        ct1 = (self.df1 + self.df2) / 2
        ct2 = self.df1 / 2
        ct3 = self.df2 / 2
        t1 = (self.df1 / self.df2).log() * torch.exp(ct2.lgamma() + ct3.lgamma() - ct1.lgamma())
        t2 = (1 - ct2) * ct2.digamma() - (1 + ct3) * ct3.digamma()
        t3 = ct1 * ct1.digamma()
        return t1 + t2 + t3
