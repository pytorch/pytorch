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
    Unflattens a tensor into another tensor of a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be flattened, and it can 
      be either `str` or `int` when `NamedTensor` or `Tensor` is used, respectively.

    * :attr:`unflattened_size` is the size of the unflattened dimension of the tensor and it can be a
      `namedshape` (`tuple` of tuples) if :attr:`dim` is `str` or a `tuple` of ints as well as `torch.Size` if
      :attr:`dim` is an `int`.

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`

    Args:
        dim (Union[int, str]): Dimension to be flattened
        unflattened_size (Union[tuple, torch.Size]): Size of the output tensor

    Examples:
        >>> input = torch.randn(2, 50)
        >>> # With tuple of ints
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, (2, 5, 5))
        >>> )
        >>> output = m(output)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With torch.Size
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, torch.Size([2, 5, 5]))
        >>> )
        >>> output = m(output)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With namedshape (tuple of tuples)
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten('features', (('C', 2), ('H', 50), ('W',50)))
        >>> )
        >>> output = m(output)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
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
            dim = int(self.dim)
            if dim < 0:
                dim += input.dim()
            inp_size = list(input.size())
            new_size = inp_size[:dim] + list(self.unflattened_size) + inp_size[dim + 1:]
            return input.view(new_size)

    def extra_repr(self) -> str:
        return 'dim={}, unflattened_size={}'.format(
            self.dim, self.unflattened_size
        )
