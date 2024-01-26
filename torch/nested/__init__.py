from typing import List, Optional, Union, Sequence

import torch
from torch import SymInt, Tensor
from torch._C import _add_docstr, _nested  # type: ignore[attr-defined]

from torch.types import _device as Device, _dtype as DType

__all__ = [
    "to_padded_tensor",
    "as_nested_tensor",
    "nested_tensor",
    "narrow",
]

# Nested Tensor constructor functions


def as_nested_tensor(
    ts: Union[Tensor, Sequence[Tensor]],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    layout=None
) -> Tensor:
    r"""
    Constructs a nested tensor preserving autograd history from a tensor or a list of tensors.

    If a nested tensor is passed, it will be returned directly unless the device / dtype / layout
    differ. Note that converting device / dtype will result in a copy, while converting layout
    is not currently supported by this function.

    If a non-nested tensor is passed, it is treated as a batch of constituents of consistent size.
    A copy will be incurred if the passed device / dtype differ from those of the input OR if
    the input is non-contiguous. Otherwise, the input's storage will be used directly.

    If a tensor list is provided, tensors in the list are always copied during construction of
    the nested tensor.

    Args:
        ts (Tensor or List[Tensor]): a tensor to treat as a nested tensor OR a list of tensors
            with the same ndim

    Keyword arguments:
        dtype (:class:`torch.dtype`, optional): the desired type of returned nested tensor.
            Default: if None, same :class:`torch.dtype` as leftmost tensor in the list.
        device (:class:`torch.device`, optional): the desired device of returned nested tensor.
            Default: if None, same :class:`torch.device` as leftmost tensor in the list
        layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
            Only strided and jagged layouts are supported. Default: if None, the strided layout.

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
        >>> c = torch.randn(3, 5, requires_grad=True)
        >>> nt2 = torch.nested.as_nested_tensor(c)
    """
    is_tensor_list = isinstance(ts, list) and all(isinstance(t, Tensor) for t in ts)
    if not isinstance(ts, Tensor) and not is_tensor_list:
        raise TypeError(
            "as_nested_tensor(): Expected first argument to be a tensor or a list of tensors "
        )

    if isinstance(ts, Tensor) and ts.dim() < 2:
        raise RuntimeError("as_nested_tensor(): Expected tensor argument to have dim() > 1")

    if isinstance(ts, Tensor) and ts.is_nested:
        if layout == ts.layout:
            # return input directly or input copied to device / dtype
            return ts.to(device=device, dtype=dtype)
        else:
            # TODO: Just use nt.to(layout=layout) when it exists.
            raise RuntimeError(
                "as_nested_tensor(): Converting between nested tensor layouts is not supported")

    if layout is None:
        layout = torch.strided
    if layout == torch.strided:
        if isinstance(ts, Tensor):
            # contiguous() might be necessary to get flattened view.
            # we could probably be more precise about when to do this as an optimization
            buffer = ts.contiguous().view(-1).to(device=device, dtype=dtype)
            nested_sizes = torch.tensor([t.shape for t in ts])
            return torch._nested_view_from_buffer(
                buffer,
                nested_sizes,
                *torch._nested_compute_contiguous_strides_offsets(nested_sizes))
        else:
            return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    elif layout == torch.jagged:
        if isinstance(ts, Tensor):
            # contiguous() might be necessary to get flattened view.
            # we could probably be more precise about when to do this as an optimization
            values = ts.contiguous().flatten(0, 1).to(device=device, dtype=dtype)
            batch_size = ts.shape[0]
            seq_len = ts.shape[1]
            offsets = torch.arange(0, batch_size * seq_len + 1, seq_len,
                                   device=device, dtype=torch.int64)

            from torch.nested._internal.nested_tensor import nested_view_from_values_offsets

            return nested_view_from_values_offsets(values, offsets)
        else:
            from torch.nested._internal.nested_tensor import jagged_from_list

            nt, _ = jagged_from_list(ts, offsets=None, device=device, dtype=dtype)
            return nt
    else:
        raise RuntimeError(f"Specified layout is unsupported for nested tensors: {layout}")


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

def nested_tensor(tensor_list, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor:
    r"""
Constructs a nested tensor with no autograd history (also known as a “leaf tensor”, see
:ref:`Autograd mechanics <autograd-mechanics>`) from :attr:`tensor_list` a list of tensors.

Args:
    tensor_list (List[array_like]): a list of tensors, or anything that can be passed to torch.tensor,
    where each element of the list has the same dimensionality.

Keyword arguments:
    dtype (:class:`torch.dtype`, optional): the desired type of returned nested tensor.
        Default: if None, same :class:`torch.dtype` as leftmost tensor in the list.
    layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
        Only strided and jagged layouts are supported. Default: if None, the strided layout.
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
    """
    if layout is None:
        layout = torch.strided
    if layout == torch.strided:
        return _nested.nested_tensor(
            tensor_list,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory)
    elif layout == torch.jagged:
        # Need to wrap lists of scalars as tensors
        list_of_tensors = [t if isinstance(t, Tensor) else torch.as_tensor(t) for t in tensor_list]

        from torch.nested._internal.nested_tensor import jagged_from_list

        with torch.no_grad():
            nt, _ = jagged_from_list(list_of_tensors, offsets=None, device=device, dtype=dtype)

        nt.requires_grad_(requires_grad)
        if pin_memory:
            nt = nt.pin_memory()  # type: ignore[assignment]

        return nt
    else:
        raise RuntimeError(f"Specified layout is unsupported for nested tensors: {layout}")


def narrow(tensor: Tensor, dim: int, start: Union[int, Tensor], length: Union[int, Tensor], layout=torch.strided) -> Tensor:
    r"""
Constructs a nested tensor (which might be a view) from :attr:`tensor`, a strided tensor. This follows
similar semantics to torch.Tensor.narrow, where in the :attr:`dim`-th dimension the new nested tensor
shows only the elements in the interval `[start, start+length)`. As nested representations
allow for a different `start` and `length` at each 'row' of that dimension, :attr:`start` and :attr:`length`
can also be tensors of shape `tensor.shape[0]`.

There's some differences depending on the layout you use for the nested tensor. If using strided layout,
torch.narrow will do a copy of the narrowed data into a contiguous NT with strided layout, while
jagged layout narrow() will create a non-contiguous view of your original strided tensor. This particular
representation is really useful for representing kv-caches in Transformer models, as specialized
SDPA kernels can deal with format easily, resulting in performance improvements.


Args:
    tensor (:class:`torch.Tensor`): a strided tensor, which will be used as the underlying data
        for the nested tensor if using the jagged layout or will be copied for the strided layout.
    dim (int): the dimension where narrow will be applied. Only `dim=1` is supported for the
        jagged layout, while strided supports all dim
    start (Union[int, :class:`torch.Tensor`]): starting element for the narrow operation
    length (Union[int, :class:`torch.Tensor`]): number of elements taken during the narrow op

Keyword arguments:
    layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
        Only strided and jagged layouts are supported. Default: if None, the strided layout.

Example::

    >>> starts = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    >>> lengths = torch.tensor([3, 2, 2, 1, 5], dtype=torch.int64)
    >>> narrow_base = torch.randn(5, 10, 20)
    >>> nt_narrowed = torch.nested.narrow(narrow_base, 1, starts, lengths, layout=torch.jagged)
    >>> nt_narrowed.is_contiguous()
    False
    """
    if not isinstance(start, (int, SymInt, Tensor)):
        raise RuntimeError("start must be an integer or a tensor")

    if not isinstance(length, (int, SymInt, Tensor)):
        raise RuntimeError("length must be an integer or a tensor")

    if layout == torch.strided:
        if isinstance(start, Tensor) or isinstance(length, Tensor):
            raise RuntimeError("start and length must be integers for the strided layout NT impl")
        # TODO: switch to as_nested_tensor(tensor) when it is available
        nt = as_nested_tensor(torch.unbind(tensor), layout=torch.strided).narrow(dim, start, length)
    elif layout == torch.jagged:
        if dim != 1:
            raise RuntimeError("jagged layout only supports dim=1")

        from torch.nested._internal.nested_tensor import jagged_from_tensor_and_lengths

        if isinstance(start, (int, SymInt)):
            start = torch.tensor([start], device=tensor.device, dtype=torch.int64)

        if isinstance(length, (int, SymInt)):
            length = torch.tensor([length], device=tensor.device, dtype=torch.int64)

        nt, _, _ = jagged_from_tensor_and_lengths(tensor, start, length)
    else:
        raise RuntimeError(f"Specified layout is unsupported for nested narrow: {layout}")

    return nt
