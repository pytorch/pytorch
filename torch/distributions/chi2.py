from torch.distributions import Gamma


class Chi2(Gamma):
    r"""
    Creates a Gamma distribution parameterized by shape `alpha` and rate `beta`.

    Example::

        >>> m = Gamma(torch.Tensor([1.0]), torch.Tensor([1.0]))
        >>> m.sample()  # Gamma distributed with shape alpha=1 and rate beta=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        alpha (float or Tensor or Variable): shape parameter of the distribution
        beta (float or Tensor or Variable): rate = 1 / scale of the distribution
    """

    def __init__(self, df):
        super(Chi2, self).__init__(0.5 * df, 0.5)
