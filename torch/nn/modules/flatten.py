from .module import Module
from ..._jit_internal import weak_module, weak_script_method


@weak_module
class Flatten(Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with nn.Sequential.
    Args:
        start_dim: first dim to flatten
        end_dim: last dim to flatten

    Shape:
        - Input: :math:`(N, *dims)
        - Output: :math:`(N, product of dims)`. Output is of the same shape as input

    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
    """
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    @weak_script_method
    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
