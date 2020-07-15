from .module import Module

from torch import Tensor


class Flatten(Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).


    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
    """
    __constants__ = ['start_dim', 'end_dim']
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return 'start_dim={}, end_dim={}'.format(
            self.start_dim, self.end_dim
        )


class Unflatten(Module):
    r"""
    Unflattens a tensor into another tensor of shape (N, C, H, W). For use with :class:`~nn.Sequential`.
    Args:
        channels: numbers of channels or depth of the output tensor
        height: dimension corresponding to the height of the output tensor.
        width: dimension corresponding to the width of the output tensor.

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, C_out, H_out, W_out)`


    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Linear(49152, 49152),
        >>>     nn.Unflatten(3, 128, 128)
        >>> )
    """
    __constants__ = ['channels', 'height', 'width']
    channels: int
    height: int
    width: int

    def __init__(self, channels: int, height: int, width: int) -> None:
        super(Unflatten, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, input: Tensor) -> Tensor:
        input = input.refine_names('N', 'features')
        return input.unflatten('features', (('C', self.channels), ('H', self.height), ('W', self.width)))

    def extra_repr(self) -> str:
        return 'channels={}, height={}, width={}'.format(
            self.channels, self.height, self.width
        )
