from torch.distributions import constraints
from torch.distributions.gamma import Gamma


class Chi2(Gamma):
    r"""
    Creates a Chi2 distribution parameterized by shape parameter `df`.
    This is exactly equivalent to Gamma(alpha=0.5*df, beta=0.5)

    Example::

        >>> m = Chi2(torch.Tensor([1.0]))
        >>> m.sample()  # Chi2 distributed with shape df=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        df (float or Tensor): shape parameter of the distribution
    """
    params = {'df': constraints.positive}

    def __init__(self, df):
        super(Chi2, self).__init__(0.5 * df, 0.5)
