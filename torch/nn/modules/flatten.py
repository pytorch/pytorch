# mypy: allow-untyped-defs
from typing import Union

from torch import Tensor
from torch.types import _size

from .module import Module


__all__ = ["Flatten", "Unflatten"]


class Flatten(Module):
    r"""
    Flattens a contiguous range of dims into a tensor.

    For use with :class:`~nn.Sequential`, see :meth:`torch.flatten` for details.

    Shape:
        - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)`,'
          where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
          number of dimensions including none.
        - Output: :math:`(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)`.

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Examples::
        >>> input = torch.randn(32, 1, 5, 5)
        >>> # With default parameters
        >>> m = nn.Flatten()
        >>> output = m(input)
        >>> output.size()
        torch.Size([32, 25])
        >>> # With non-default parameters
        >>> m = nn.Flatten(0, 2)
        >>> output = m(input)
        >>> output.size()
        torch.Size([160, 5])
    """

    __constants__ = ["start_dim", "end_dim"]
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        """
        Returns the extra representation of the module.
        """
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Unflatten(Module):
    r"""
    Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be unflattened.

    * :attr:`unflattened_size` is the new shape of the unflattened dimension of the tensor and it can be
      a `tuple` of ints or a `list` of ints or `torch.Size`.

    Shape:
        - Input: :math:`(*, S_{\text{dim}}, *)`, where :math:`S_{\text{dim}}` is the size at
          dimension :attr:`dim` and :math:`*` means any number of dimensions including none.
        - Output: :math:`(*, U_1, ..., U_n, *)`, where :math:`U` = :attr:`unflattened_size` and
          :math:`\prod_{i=1}^n U_i = S_{\text{dim}}`.

    Args:
        dim: Dimension to be unflattened
        unflattened_size (Union[torch.Size, Tuple, List]): New shape of the unflattened dimension

    Examples:
        >>> input = torch.randn(2, 50)
        >>> # With tuple of ints
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, (2, 5, 5))
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With torch.Size
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, torch.Size([2, 5, 5]))
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
    """

    __constants__ = ["dim", "unflattened_size"]
    dim: int
    unflattened_size: _size

    def __init__(self, dim: int, unflattened_size: _size) -> None:
        super().__init__()

        if not isinstance(dim, int):
            raise TypeError(f"dim must be an int, but found {type(dim).__name__}")
        self._require_tuple_int(unflattened_size)

        self.dim = dim
        self.unflattened_size = unflattened_size

    def _require_tuple_int(self, input) -> None:
        if isinstance(input, (tuple, list)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError(
                        "unflattened_size must be tuple of ints, "
                        + f"but found element of type {type(elem).__name__} at pos {idx}"
                    )
            return
        raise TypeError(
            f"unflattened_size must be a tuple of ints, but found type {type(input).__name__}"
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return input.unflatten(self.dim, self.unflattened_size)

    def extra_repr(self) -> str:
        """
        Returns the extra representation of the module.
        """
        return f"dim={self.dim}, unflattened_size={self.unflattened_size}"
