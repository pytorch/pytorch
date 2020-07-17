from .module import Module

from typing import Union
from torch import Tensor
from torch import Size


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
        dim: tensor dimension to be unflattened
        unflattened_size: shape of the output tensor

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, C_out, H_out, W_out)`


    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Linear(49152, 49152),
        >>>     nn.Unflatten(1, (3, 128, 128))
        >>> )

        >>> m = nn.Sequential(
        >>>     nn.Linear(49152, 49152),
        >>>     nn.Unflatten('features', (('C', 3), ('H', 128), ('W',128)))
        >>> )
    """
    __constants__ = ['dim', 'unflattened_size']
    dim: Union[int, str]
    unflattened_size: Union[tuple, Size]

    def __init__(self, dim: Union[int, str], unflattened_size: Union[tuple, Size]) -> None:
        super(Unflatten, self).__init__()
        if isinstance(dim, int):
            self._require_tuple_int(unflattened_size)
            self.named = False
        else:
            self._require_tuple_tuple(unflattened_size)
            self.named = True

        self.dim = dim
        self.unflattened_size = unflattened_size

    def _require_tuple_tuple(self, input):
        if (isinstance(input, tuple)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, tuple):
                    raise TypeError("unflattened_size must be tuple of tuples, " + 
                                    "but found element of type {} at pos {}".format(type(elem).__name__, idx))
            return
        raise TypeError("unflattened_size must be a tuple of tuples, " +
                        "but found type {}".format(type(input).__name__))

    def _require_tuple_int(self, input):
        if (isinstance(input, tuple)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError("unflattened_size must be tuple of ints, " + 
                                    "but found element of type {} at pos {}".format(type(elem).__name__, idx))
            return
        raise TypeError("unflattened_size must be a tuple of ints, but found type {}".format(type(input).__name__))

    def forward(self, input: Tensor) -> Tensor:
        if self.named:
            return input.unflatten(self.dim, self.unflattened_size)
        else:
            dim = self.dim
            if self.dim < 0:
                dim += input.ndim()
            inp_size = list(input.size())
            new_size = inp_size[:dim] + list(self.unflattened_size) + inp_size[dim + 1:]
            return input.view(new_size)

    def extra_repr(self) -> str:
        return 'dim={}, unflattened_size={}'.format(
            self.dim, self.unflattened_size
        )
