from typing import Tuple, Optional, TYPE_CHECKING
import torch
from torch._C import _linalg  # type: ignore[attr-defined]
from torch.overrides import has_torch_function_variadic, handle_torch_function
from torch._jit_internal import _overload as overload
from torch._jit_internal import List

if TYPE_CHECKING:
    pass
    # There's no good way to use this type annotation without breaking JIT
    # overloads. So leave untyped for mypy for now.
else:
    @overload
    def tensordot(a, b, dims: int = 2, out: Optional[torch.Tensor] = None):
        pass

    @overload
    def tensordot(a, b, dims: Tuple[List[int], List[int]], out: Optional[torch.Tensor] = None):  # noqa: F811
        pass

    @overload
    def tensordot(a, b, dims: List[List[int]], out: Optional[torch.Tensor] = None):  # noqa: F811
        pass

    @overload
    def tensordot(a, b, dims: torch.Tensor, out: Optional[torch.Tensor] = None):  # noqa: F811
        pass

def tensordot(a, b, dims=2, out: Optional[torch.Tensor] = None):  # noqa: F811
    r"""Returns a contraction of a and b over multiple dimensions.

    :attr:`tensordot` implements a generalized matrix product.

    Args:
      a (Tensor): Left tensor to contract
      b (Tensor): Right tensor to contract
      dims (int or Tuple[List[int], List[int]] or List[List[int]] containing two lists or Tensor): number of dimensions to
         contract or explicit lists of dimensions for :attr:`a` and
         :attr:`b` respectively

    When called with a non-negative integer argument :attr:`dims` = :math:`d`, and
    the number of dimensions of :attr:`a` and :attr:`b` is :math:`m` and :math:`n`,
    respectively, :func:`torch.linalg.tensordot` computes

    .. math::
        r_{i_0,...,i_{m-d}, i_d,...,i_n}
          = \sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \times b_{k_0,...,k_{d-1}, i_d,...,i_n}.

    When called with :attr:`dims` of the list form, the given dimensions will be contracted
    in place of the last :math:`d` of :attr:`a` and the first :math:`d` of :math:`b`. The sizes
    in these dimensions must match, but :func:`torch.linalg.tensordot` will deal with broadcasted
    dimensions.

    Examples::

        >>> a = torch.arange(60.).reshape(3, 4, 5)
        >>> b = torch.arange(24.).reshape(4, 3, 2)
        >>> torch.linalg.tensordot(a, b, dims=([1, 0], [0, 1]))
        tensor([[4400., 4730.],
                [4532., 4874.],
                [4664., 5018.],
                [4796., 5162.],
                [4928., 5306.]])

        >>> a = torch.randn(3, 4, 5, device='cuda')
        >>> b = torch.randn(4, 5, 6, device='cuda')
        >>> c = torch.linalg.tensordot(a, b, dims=2).cpu()
        tensor([[ 8.3504, -2.5436,  6.2922,  2.7556, -1.0732,  3.2741],
                [ 3.3161,  0.0704,  5.0187, -0.4079, -4.3126,  4.8744],
                [ 0.8223,  3.9445,  3.2168, -0.2400,  3.4117,  1.7780]])

        >>> a = torch.randn(3, 5, 4, 6)
        >>> b = torch.randn(6, 4, 5, 3)
        >>> torch.linalg.tensordot(a, b, dims=([2, 1, 3], [1, 2, 0]))
        tensor([[  7.7193,  -2.4867, -10.3204],
                [  1.5513, -14.4737,  -6.5113],
                [ -0.2850,   4.2573,  -3.5997]])
    """
    if has_torch_function_variadic(a, b):
        return handle_torch_function(tensordot, (a, b), a, b, dims=dims)

    if not isinstance(dims, (tuple, list, torch.Tensor, int)):
        raise RuntimeError("tensordot expects dims to be int or "
                           + "Tuple[List[int], List[int]] or "
                           + "List[List[int]] containing two lists, but got "
                           + f"dims={dims}")

    dims_a: List[int] = []
    dims_b: List[int] = []

    if isinstance(dims, (tuple, list)):
        dims_a, dims_b = dims

    if isinstance(dims, torch.Tensor):
        num_elements = dims.numel()
        if num_elements > 1:
            assert dims.size()[0] == 2
            dims_a = torch.jit.annotate(List[int], dims[0].tolist())
            dims_b = torch.jit.annotate(List[int], dims[1].tolist())
        else:
            dims_val = int(dims.item())
            if dims_val < 0:
                raise RuntimeError(f"tensordot expects dims >= 0, but got dims={dims}")
            dims_a = list(range(-dims_val, 0))
            dims_b = list(range(dims_val))

    if isinstance(dims, int):
        if dims < 0:
            raise RuntimeError(f"tensordot expects dims >= 0, but got dims={dims}")
        dims_a = list(range(-dims, 0))
        dims_b = list(range(dims))

    if out is None:
        return _linalg.linalg_tensordot(a, b, dims_a, dims_b)  # type: ignore[attr-defined]
    else:
        return _linalg.linalg_tensordot(a, b, dims_a, dims_b, out=out)  # type: ignore[attr-defined]
