import torch.nn.functional as F
from torch import Tensor

from .module import Module


__all__ = ["ChannelShuffle"]


class ChannelShuffle(Module):
    r"""Divides and rearranges the channels in a tensor.

    This operation divides the channels in a tensor of shape :math:`(*, C , H, W)`
    into g groups as :math:`(*, \frac{C}{g}, g, H, W)` and shuffles them,
    while retaining the original tensor shape in the final output.

    Args:
        groups (int): number of groups to divide channels in.

    Examples::

        >>> channel_shuffle = nn.ChannelShuffle(2)
        >>> input = torch.arange(1, 17, dtype=torch.float32).view(1, 4, 2, 2)
        >>> input
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]],
                 [[ 5.,  6.],
                  [ 7.,  8.]],
                 [[ 9., 10.],
                  [11., 12.]],
                 [[13., 14.],
                  [15., 16.]]]])
        >>> output = channel_shuffle(input)
        >>> output
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]],
                 [[ 9., 10.],
                  [11., 12.]],
                 [[ 5.,  6.],
                  [ 7.,  8.]],
                 [[13., 14.],
                  [15., 16.]]]])
    """

    __constants__ = ["groups"]
    groups: int

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, input: Tensor) -> Tensor:
        return F.channel_shuffle(input, self.groups)

    def extra_repr(self) -> str:
        return f"groups={self.groups}"
