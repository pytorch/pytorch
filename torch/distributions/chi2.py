# mypy: allow-untyped-defs
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.gamma import Gamma


__all__ = ["Chi2"]


class Chi2(Gamma):
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`.
    This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)``

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Chi2(torch.tensor([1.0]))
        >>> m.sample()  # Chi2 distributed with shape df=1
        tensor([ 0.1046])

    Args:
        df (float or Tensor): shape parameter of the distribution
    """

    arg_constraints = {"df": constraints.positive}

    def __init__(self, df, validate_args=None):
        super().__init__(0.5 * df, 0.5, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Chi2, _instance)
        return super().expand(batch_shape, new)

    @property
    def df(self) -> Tensor:
        return self.concentration * 2
