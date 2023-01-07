from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch._C import _add_docstr, _nested  # type: ignore[attr-defined]

from torch.types import _device as Device, _dtype as DType

__all__ = [
    "to_padded_tensor",
    "as_nested_tensor",
    "nested_tensor",
    "empty_nested",
]

# Nested Tensor constructor functions


def as_nested_tensor(
    tensor_list: List[Tensor],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Tensor:
    r"""
    Constructs a nested tensor preserving autograd history from :attr:`tensor_list` a list of tensors.

    .. note::
        Tensors within the list are always copied by this function due to current nested tensor semantics.

    Args:
        tensor_list (List[Tensor]): a list of tensors with the same ndim

    Keyword arguments:
        dtype (:class:`torch.dtype`, optional): the desired type of returned nested tensor.
            Default: if None, same :class:`torch.dtype` as leftmost tensor in the list.
        device (:class:`torch.device`, optional): the desired device of returned nested tensor.
            Default: if None, same :class:`torch.device` as leftmost tensor in the list

    Example::

        >>> a = torch.arange(3, dtype=torch.float, requires_grad=True)
        >>> b = torch.arange(5, dtype=torch.float, requires_grad=True)
        >>> nt = torch.nested.as_nested_tensor([a, b])
        >>> nt.is_leaf
        False
        >>> fake_grad = torch.nested.nested_tensor([torch.ones_like(a), torch.zeros_like(b)])
        >>> nt.backward(fake_grad)
        >>> a.grad
        tensor([1., 1., 1.])
        >>> b.grad
        tensor([0., 0., 0., 0., 0.])
    """
    if not isinstance(tensor_list, list) or any(
        [not isinstance(t, Tensor) for t in tensor_list]
    ):
        raise TypeError(
            "nested_tensor(): Expected first argument to be a list of tensors "
        )
    return torch._nested_tensor_from_tensor_list(tensor_list, dtype, None, device, None)


# Note: This not only adds doc strings for the nested ops, but
# also connects the torch.nested Python namespace to the torch._C._nested builtins.

to_padded_tensor = _add_docstr(
    _nested.nested_to_padded_tensor,
    r"""
to_padded_tensor(input, padding, output_size=None, out=None) -> Tensor

Returns a new (non-nested) Tensor by padding the :attr:`input` nested tensor.
The leading entries will be filled with the nested data,
while the trailing entries will be padded.

.. warning::

    :func:`to_padded_tensor` always copies the underlying data,
    since the nested and the non-nested tensors differ in memory layout.

Args:
    padding (float): The padding value for the trailing entries.

Keyword args:
    output_size (Tuple[int]): The size of the output tensor.
                              If given, it must be large enough to contain all nested data;
                              else, will infer by taking the max size of each nested sub-tensor along each dimension.
    out (Tensor, optional): the output tensor.

Example::

    >>> nt = torch.nested.nested_tensor([torch.randn((2, 5)), torch.randn((3, 4))])
    nested_tensor([
      tensor([[ 1.6862, -1.1282,  1.1031,  0.0464, -1.3276],
              [-1.9967, -1.0054,  1.8972,  0.9174, -1.4995]]),
      tensor([[-1.8546, -0.7194, -0.2918, -0.1846],
              [ 0.2773,  0.8793, -0.5183, -0.6447],
              [ 1.8009,  1.8468, -0.9832, -1.5272]])
    ])
    >>> pt_infer = torch.nested.to_padded_tensor(nt, 0.0)
    tensor([[[ 1.6862, -1.1282,  1.1031,  0.0464, -1.3276],
             [-1.9967, -1.0054,  1.8972,  0.9174, -1.4995],
             [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
            [[-1.8546, -0.7194, -0.2918, -0.1846,  0.0000],
             [ 0.2773,  0.8793, -0.5183, -0.6447,  0.0000],
             [ 1.8009,  1.8468, -0.9832, -1.5272,  0.0000]]])
    >>> pt_large = torch.nested.to_padded_tensor(nt, 1.0, (2, 4, 6))
    tensor([[[ 1.6862, -1.1282,  1.1031,  0.0464, -1.3276,  1.0000],
             [-1.9967, -1.0054,  1.8972,  0.9174, -1.4995,  1.0000],
             [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]],
            [[-1.8546, -0.7194, -0.2918, -0.1846,  1.0000,  1.0000],
             [ 0.2773,  0.8793, -0.5183, -0.6447,  1.0000,  1.0000],
             [ 1.8009,  1.8468, -0.9832, -1.5272,  1.0000,  1.0000],
             [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]]])
    >>> pt_small = torch.nested.to_padded_tensor(nt, 2.0, (2, 2, 2))
    RuntimeError: Value in output_size is less than NestedTensor padded size. Truncation is not supported.

""",
)

nested_tensor = _add_docstr(
    _nested.nested_tensor,
    r"""
nested_tensor(tensor_list, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Constructs a nested tensor with no autograd history (also known as a “leaf tensor”, see
:ref:`Autograd mechanics <autograd-mechanics>`) from :attr:`tensor_list` a list of tensors.

Args:
    tensor_list (List[array_like]): a list of tensors, or anything that can be passed to torch.tensor,
    where each element of the list has the same dimensionality.

Keyword arguments:
    dtype (:class:`torch.dtype`, optional): the desired type of returned nested tensor.
        Default: if None, same :class:`torch.dtype` as leftmost tensor in the list.
    device (:class:`torch.device`, optional): the desired device of returned nested tensor.
        Default: if None, same :class:`torch.device` as leftmost tensor in the list
    requires_grad (bool, optional): If autograd should record operations on the
        returned nested tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned nested tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.

Example::

    >>> a = torch.arange(3, dtype=torch.float, requires_grad=True)
    >>> b = torch.arange(5, dtype=torch.float, requires_grad=True)
    >>> nt = torch.nested.nested_tensor([a, b], requires_grad=True)
    >>> nt.is_leaf
    True
    """,
)

class NestedSize():
    """Thin wrapper to abstract the internal representation of nested sizes from the user.

    This class is used to represent the sizes of a NestedTensor. It is a list of tuples where
    each tuple represents the size of a tensor_component in the NestedTensor. This is used when
    creating nested_tensors from size information alone.

    """
    def __init__(self, sizes: List[Tuple[int]]):
        self.validate_input(sizes)
        self.sizes_ = sizes

    def convert_list_to_nested_size(self) -> Tensor:
        r"""Converts a list of tuples to a 2d tensor on cpu.
            This is the current datastructure used to represent nested sizes in C++. However
            this may change in the future.
        """
        return torch.Tensor(self.sizes_, device='cpu').to(torch.int64)

    @staticmethod
    def validate_input(sizes: List[Tuple[int]]) -> None:
        r"""Validates the input sizes to class constructor.
        """
        if not isinstance(sizes, list):
            raise TypeError("sizes must be a list of tuples")

        if len(sizes) == 0:
            raise ValueError("sizes must be non-empty")

        # Check first element is a tuple
        if not isinstance(sizes[0], tuple):
            raise ValueError("sizes must be a list of tuples")

        tuple_size = len(sizes[0])
        for size in sizes[1:]:
            if not isinstance(size, tuple):
                raise ValueError("sizes must be a list of tuples")
            if len(size) != tuple_size:
                raise ValueError("All tuples in sizes must have the same length")


def empty(nested_size: Union[NestedSize, Tensor], dtype=None, device=None, pin_memory=False) -> Tensor:
    r"""Constructs a contiguous NestedTensor with the shape specified by nested_size.

    Args:
        nested_size (NestedSize): a NestedSize object that specifies the shape of the NestedTensor
            For Jedis: nested_size is also allowed to be a 2d tensor on cpu. But this may not always be true

    Keyword Arguments:
        dtype (:class:`torch.dtype`, optional): the desired type of returned NestedTensor.
            Default: if None: :class:`torch.float32`.
        device (:class:`torch.device`, optional): the desired device of returned nested tensor.
            Default: if None: :class:`cpu`
        pin_memory (bool, optional): If set, returned nested tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.

    Example::

        >>> nt_size = torch.nested.NestedSize([(2,4), (3,5)])
        >>> nt = torch.nested.empty(nt_size, device='cuda', dtype=torch.float16)
        """
    if isinstance(nested_size, NestedSize):
        nested_size = nested_size.convert_list_to_nested_size()
    elif isinstance(nested_size, Tensor):
        if nested_size.device != torch.device('cpu'):
            raise ValueError("nested_size must be a cpu tensor")
        if nested_size.dim() != 2:
            raise ValueError("nested_size must be a 2d tensor")
        if nested_size.dtype != torch.int64:
            raise ValueError("nested_size must be a int64 tensor")
    else:
        raise ValueError("nested_size must be a NestedSize object or a 2d cpu int64 tensor")
    return _nested.empty_nested(nested_size,
                                dtype=dtype, layout=None,
                                device=device,
                                pin_memory=pin_memory)  # type: ignore[attr-defined]
