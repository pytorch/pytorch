# mypy: allow-untyped-defs
"""Adds docstrings to Tensor functions"""

import torch._C
from torch._C import _add_docstr as add_docstr
from torch._torch_docs import parse_kwargs, reproducibility_notes


def add_docstr_all(method: str, docstr: str) -> None:
    add_docstr(getattr(torch._C.TensorBase, method), docstr)


common_args = parse_kwargs(
    """
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.
"""
)

new_common_args = parse_kwargs(
    """
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired type of returned tensor.
        Default: if None, same :class:`torch.dtype` as this tensor.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, same :class:`torch.device` as this tensor.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
"""
)

add_docstr_all(
    "new_tensor",
    """
new_tensor(data, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a new Tensor with :attr:`data` as the tensor data.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

.. warning::

    :func:`new_tensor` always copies :attr:`data`. If you have a Tensor
    ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
    or :func:`torch.Tensor.detach`.
    If you have a numpy array and want to avoid a copy, use
    :func:`torch.from_numpy`.

.. warning::

    When data is a tensor `x`, :func:`new_tensor()` reads out 'the data' from whatever it is passed,
    and constructs a leaf variable. Therefore ``tensor.new_tensor(x)`` is equivalent to ``x.detach().clone()``
    and ``tensor.new_tensor(x, requires_grad=True)`` is equivalent to ``x.detach().clone().requires_grad_(True)``.
    The equivalents using ``detach()`` and ``clone()`` are recommended.

Args:
    data (array_like): The returned Tensor copies :attr:`data`.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.ones((2,), dtype=torch.int8)
    >>> data = [[0, 1], [2, 3]]
    >>> tensor.new_tensor(data)
    tensor([[ 0,  1],
            [ 2,  3]], dtype=torch.int8)

""".format(**new_common_args),
)

add_docstr_all(
    "new_full",
    """
new_full(size, fill_value, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a Tensor of size :attr:`size` filled with :attr:`fill_value`.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    fill_value (scalar): the number to fill the output tensor with.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.ones((2,), dtype=torch.float64)
    >>> tensor.new_full((3, 4), 3.141592)
    tensor([[ 3.1416,  3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416,  3.1416]], dtype=torch.float64)

""".format(**new_common_args),
)

add_docstr_all(
    "new_empty",
    """
new_empty(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a Tensor of size :attr:`size` filled with uninitialized data.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.ones(())
    >>> tensor.new_empty((2, 3))
    tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
            [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])

""".format(**new_common_args),
)

add_docstr_all(
    "new_empty_strided",
    """
new_empty_strided(size, stride, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a Tensor of size :attr:`size` and strides :attr:`stride` filled with
uninitialized data. By default, the returned Tensor has the same
:class:`torch.dtype` and :class:`torch.device` as this tensor.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.ones(())
    >>> tensor.new_empty_strided((2, 3), (3, 1))
    tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
            [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])

""".format(**new_common_args),
)

add_docstr_all(
    "new_ones",
    """
new_ones(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a Tensor of size :attr:`size` filled with ``1``.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.tensor((), dtype=torch.int32)
    >>> tensor.new_ones((2, 3))
    tensor([[ 1,  1,  1],
            [ 1,  1,  1]], dtype=torch.int32)

""".format(**new_common_args),
)

add_docstr_all(
    "new_zeros",
    """
new_zeros(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, \
pin_memory=False) -> Tensor
"""
    + r"""

Returns a Tensor of size :attr:`size` filled with ``0``.
By default, the returned Tensor has the same :class:`torch.dtype` and
:class:`torch.device` as this tensor.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.

Keyword args:
    {dtype}
    {device}
    {requires_grad}
    {layout}
    {pin_memory}

Example::

    >>> tensor = torch.tensor((), dtype=torch.float64)
    >>> tensor.new_zeros((2, 3))
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]], dtype=torch.float64)

""".format(**new_common_args),
)

add_docstr_all(
    "abs",
    r"""
abs() -> Tensor

See :func:`torch.abs`
""",
)

add_docstr_all(
    "abs_",
    r"""
abs_() -> Tensor

In-place version of :meth:`~Tensor.abs`
""",
)

add_docstr_all(
    "absolute",
    r"""
absolute() -> Tensor

Alias for :func:`abs`
""",
)

add_docstr_all(
    "absolute_",
    r"""
absolute_() -> Tensor

In-place version of :meth:`~Tensor.absolute`
Alias for :func:`abs_`
""",
)

add_docstr_all(
    "acos",
    r"""
acos() -> Tensor

See :func:`torch.acos`
""",
)

add_docstr_all(
    "acos_",
    r"""
acos_() -> Tensor

In-place version of :meth:`~Tensor.acos`
""",
)

add_docstr_all(
    "arccos",
    r"""
arccos() -> Tensor

See :func:`torch.arccos`
""",
)

add_docstr_all(
    "arccos_",
    r"""
arccos_() -> Tensor

In-place version of :meth:`~Tensor.arccos`
""",
)

add_docstr_all(
    "acosh",
    r"""
acosh() -> Tensor

See :func:`torch.acosh`
""",
)

add_docstr_all(
    "acosh_",
    r"""
acosh_() -> Tensor

In-place version of :meth:`~Tensor.acosh`
""",
)

add_docstr_all(
    "arccosh",
    r"""
acosh() -> Tensor

See :func:`torch.arccosh`
""",
)

add_docstr_all(
    "arccosh_",
    r"""
acosh_() -> Tensor

In-place version of :meth:`~Tensor.arccosh`
""",
)

add_docstr_all(
    "add",
    r"""
add(other, *, alpha=1) -> Tensor

Add a scalar or tensor to :attr:`self` tensor. If both :attr:`alpha`
and :attr:`other` are specified, each element of :attr:`other` is scaled by
:attr:`alpha` before being used.

When :attr:`other` is a tensor, the shape of :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
tensor

See :func:`torch.add`
""",
)

add_docstr_all(
    "add_",
    r"""
add_(other, *, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.add`
""",
)

add_docstr_all(
    "addbmm",
    r"""
addbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addbmm`
""",
)

add_docstr_all(
    "addbmm_",
    r"""
addbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addbmm`
""",
)

add_docstr_all(
    "addcdiv",
    r"""
addcdiv(tensor1, tensor2, *, value=1) -> Tensor

See :func:`torch.addcdiv`
""",
)

add_docstr_all(
    "addcdiv_",
    r"""
addcdiv_(tensor1, tensor2, *, value=1) -> Tensor

In-place version of :meth:`~Tensor.addcdiv`
""",
)

add_docstr_all(
    "addcmul",
    r"""
addcmul(tensor1, tensor2, *, value=1) -> Tensor

See :func:`torch.addcmul`
""",
)

add_docstr_all(
    "addcmul_",
    r"""
addcmul_(tensor1, tensor2, *, value=1) -> Tensor

In-place version of :meth:`~Tensor.addcmul`
""",
)

add_docstr_all(
    "addmm",
    r"""
addmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addmm`
""",
)

add_docstr_all(
    "addmm_",
    r"""
addmm_(mat1, mat2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addmm`
""",
)

add_docstr_all(
    "addmv",
    r"""
addmv(mat, vec, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addmv`
""",
)

add_docstr_all(
    "addmv_",
    r"""
addmv_(mat, vec, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addmv`
""",
)

add_docstr_all(
    "sspaddmm",
    r"""
sspaddmm(mat1, mat2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.sspaddmm`
""",
)

add_docstr_all(
    "smm",
    r"""
smm(mat) -> Tensor

See :func:`torch.smm`
""",
)

add_docstr_all(
    "addr",
    r"""
addr(vec1, vec2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.addr`
""",
)

add_docstr_all(
    "addr_",
    r"""
addr_(vec1, vec2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.addr`
""",
)

add_docstr_all(
    "align_as",
    r"""
align_as(other) -> Tensor

Permutes the dimensions of the :attr:`self` tensor to match the dimension order
in the :attr:`other` tensor, adding size-one dims for any new names.

This operation is useful for explicit broadcasting by names (see examples).

All of the dims of :attr:`self` must be named in order to use this method.
The resulting tensor is a view on the original tensor.

All dimension names of :attr:`self` must be present in ``other.names``.
:attr:`other` may contain named dimensions that are not in ``self.names``;
the output tensor has a size-one dimension for each of those new names.

To align a tensor to a specific order, use :meth:`~Tensor.align_to`.

Examples::

    # Example 1: Applying a mask
    >>> mask = torch.randint(2, [127, 128], dtype=torch.bool).refine_names('W', 'H')
    >>> imgs = torch.randn(32, 128, 127, 3, names=('N', 'H', 'W', 'C'))
    >>> imgs.masked_fill_(mask.align_as(imgs), 0)


    # Example 2: Applying a per-channel-scale
    >>> def scale_channels(input, scale):
    >>>    scale = scale.refine_names('C')
    >>>    return input * scale.align_as(input)

    >>> num_channels = 3
    >>> scale = torch.randn(num_channels, names=('C',))
    >>> imgs = torch.rand(32, 128, 128, num_channels, names=('N', 'H', 'W', 'C'))
    >>> more_imgs = torch.rand(32, num_channels, 128, 128, names=('N', 'C', 'H', 'W'))
    >>> videos = torch.randn(3, num_channels, 128, 128, 128, names=('N', 'C', 'H', 'W', 'D'))

    # scale_channels is agnostic to the dimension order of the input
    >>> scale_channels(imgs, scale)
    >>> scale_channels(more_imgs, scale)
    >>> scale_channels(videos, scale)

.. warning::
    The named tensor API is experimental and subject to change.

""",
)

add_docstr_all(
    "all",
    r"""
all(dim=None, keepdim=False) -> Tensor

See :func:`torch.all`
""",
)

add_docstr_all(
    "allclose",
    r"""
allclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

See :func:`torch.allclose`
""",
)

add_docstr_all(
    "angle",
    r"""
angle() -> Tensor

See :func:`torch.angle`
""",
)

add_docstr_all(
    "any",
    r"""
any(dim=None, keepdim=False) -> Tensor

See :func:`torch.any`
""",
)

add_docstr_all(
    "apply_",
    r"""
apply_(callable) -> Tensor

Applies the function :attr:`callable` to each element in the tensor, replacing
each element with the value returned by :attr:`callable`.

.. note::

    This function only works with CPU tensors and should not be used in code
    sections that require high performance.
""",
)

add_docstr_all(
    "asin",
    r"""
asin() -> Tensor

See :func:`torch.asin`
""",
)

add_docstr_all(
    "asin_",
    r"""
asin_() -> Tensor

In-place version of :meth:`~Tensor.asin`
""",
)

add_docstr_all(
    "arcsin",
    r"""
arcsin() -> Tensor

See :func:`torch.arcsin`
""",
)

add_docstr_all(
    "arcsin_",
    r"""
arcsin_() -> Tensor

In-place version of :meth:`~Tensor.arcsin`
""",
)

add_docstr_all(
    "asinh",
    r"""
asinh() -> Tensor

See :func:`torch.asinh`
""",
)

add_docstr_all(
    "asinh_",
    r"""
asinh_() -> Tensor

In-place version of :meth:`~Tensor.asinh`
""",
)

add_docstr_all(
    "arcsinh",
    r"""
arcsinh() -> Tensor

See :func:`torch.arcsinh`
""",
)

add_docstr_all(
    "arcsinh_",
    r"""
arcsinh_() -> Tensor

In-place version of :meth:`~Tensor.arcsinh`
""",
)

add_docstr_all(
    "as_strided",
    r"""
as_strided(size, stride, storage_offset=None) -> Tensor

See :func:`torch.as_strided`
""",
)

add_docstr_all(
    "as_strided_",
    r"""
as_strided_(size, stride, storage_offset=None) -> Tensor

In-place version of :meth:`~Tensor.as_strided`
""",
)

add_docstr_all(
    "atan",
    r"""
atan() -> Tensor

See :func:`torch.atan`
""",
)

add_docstr_all(
    "atan_",
    r"""
atan_() -> Tensor

In-place version of :meth:`~Tensor.atan`
""",
)

add_docstr_all(
    "arctan",
    r"""
arctan() -> Tensor

See :func:`torch.arctan`
""",
)

add_docstr_all(
    "arctan_",
    r"""
arctan_() -> Tensor

In-place version of :meth:`~Tensor.arctan`
""",
)

add_docstr_all(
    "atan2",
    r"""
atan2(other) -> Tensor

See :func:`torch.atan2`
""",
)

add_docstr_all(
    "atan2_",
    r"""
atan2_(other) -> Tensor

In-place version of :meth:`~Tensor.atan2`
""",
)

add_docstr_all(
    "arctan2",
    r"""
arctan2(other) -> Tensor

See :func:`torch.arctan2`
""",
)

add_docstr_all(
    "arctan2_",
    r"""
atan2_(other) -> Tensor

In-place version of :meth:`~Tensor.arctan2`
""",
)

add_docstr_all(
    "atanh",
    r"""
atanh() -> Tensor

See :func:`torch.atanh`
""",
)

add_docstr_all(
    "atanh_",
    r"""
atanh_(other) -> Tensor

In-place version of :meth:`~Tensor.atanh`
""",
)

add_docstr_all(
    "arctanh",
    r"""
arctanh() -> Tensor

See :func:`torch.arctanh`
""",
)

add_docstr_all(
    "arctanh_",
    r"""
arctanh_(other) -> Tensor

In-place version of :meth:`~Tensor.arctanh`
""",
)

add_docstr_all(
    "baddbmm",
    r"""
baddbmm(batch1, batch2, *, beta=1, alpha=1) -> Tensor

See :func:`torch.baddbmm`
""",
)

add_docstr_all(
    "baddbmm_",
    r"""
baddbmm_(batch1, batch2, *, beta=1, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.baddbmm`
""",
)

add_docstr_all(
    "bernoulli",
    r"""
bernoulli(*, generator=None) -> Tensor

Returns a result tensor where each :math:`\texttt{result[i]}` is independently
sampled from :math:`\text{Bernoulli}(\texttt{self[i]})`. :attr:`self` must have
floating point ``dtype``, and the result will have the same ``dtype``.

See :func:`torch.bernoulli`
""",
)

add_docstr_all(
    "bernoulli_",
    r"""
bernoulli_(p=0.5, *, generator=None) -> Tensor

Fills each location of :attr:`self` with an independent sample from
:math:`\text{Bernoulli}(\texttt{p})`. :attr:`self` can have integral
``dtype``.

:attr:`p` should either be a scalar or tensor containing probabilities to be
used for drawing the binary random number.

If it is a tensor, the :math:`\text{i}^{th}` element of :attr:`self` tensor
will be set to a value sampled from
:math:`\text{Bernoulli}(\texttt{p\_tensor[i]})`. In this case `p` must have
floating point ``dtype``.

See also :meth:`~Tensor.bernoulli` and :func:`torch.bernoulli`
""",
)

add_docstr_all(
    "bincount",
    r"""
bincount(weights=None, minlength=0) -> Tensor

See :func:`torch.bincount`
""",
)

add_docstr_all(
    "bitwise_not",
    r"""
bitwise_not() -> Tensor

See :func:`torch.bitwise_not`
""",
)

add_docstr_all(
    "bitwise_not_",
    r"""
bitwise_not_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_not`
""",
)

add_docstr_all(
    "bitwise_and",
    r"""
bitwise_and() -> Tensor

See :func:`torch.bitwise_and`
""",
)

add_docstr_all(
    "bitwise_and_",
    r"""
bitwise_and_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_and`
""",
)

add_docstr_all(
    "bitwise_or",
    r"""
bitwise_or() -> Tensor

See :func:`torch.bitwise_or`
""",
)

add_docstr_all(
    "bitwise_or_",
    r"""
bitwise_or_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_or`
""",
)

add_docstr_all(
    "bitwise_xor",
    r"""
bitwise_xor() -> Tensor

See :func:`torch.bitwise_xor`
""",
)

add_docstr_all(
    "bitwise_xor_",
    r"""
bitwise_xor_() -> Tensor

In-place version of :meth:`~Tensor.bitwise_xor`
""",
)

add_docstr_all(
    "bitwise_left_shift",
    r"""
bitwise_left_shift(other) -> Tensor

See :func:`torch.bitwise_left_shift`
""",
)

add_docstr_all(
    "bitwise_left_shift_",
    r"""
bitwise_left_shift_(other) -> Tensor

In-place version of :meth:`~Tensor.bitwise_left_shift`
""",
)

add_docstr_all(
    "bitwise_right_shift",
    r"""
bitwise_right_shift(other) -> Tensor

See :func:`torch.bitwise_right_shift`
""",
)

add_docstr_all(
    "bitwise_right_shift_",
    r"""
bitwise_right_shift_(other) -> Tensor

In-place version of :meth:`~Tensor.bitwise_right_shift`
""",
)

add_docstr_all(
    "broadcast_to",
    r"""
broadcast_to(shape) -> Tensor

See :func:`torch.broadcast_to`.
""",
)

add_docstr_all(
    "logical_and",
    r"""
logical_and() -> Tensor

See :func:`torch.logical_and`
""",
)

add_docstr_all(
    "logical_and_",
    r"""
logical_and_() -> Tensor

In-place version of :meth:`~Tensor.logical_and`
""",
)

add_docstr_all(
    "logical_not",
    r"""
logical_not() -> Tensor

See :func:`torch.logical_not`
""",
)

add_docstr_all(
    "logical_not_",
    r"""
logical_not_() -> Tensor

In-place version of :meth:`~Tensor.logical_not`
""",
)

add_docstr_all(
    "logical_or",
    r"""
logical_or() -> Tensor

See :func:`torch.logical_or`
""",
)

add_docstr_all(
    "logical_or_",
    r"""
logical_or_() -> Tensor

In-place version of :meth:`~Tensor.logical_or`
""",
)

add_docstr_all(
    "logical_xor",
    r"""
logical_xor() -> Tensor

See :func:`torch.logical_xor`
""",
)

add_docstr_all(
    "logical_xor_",
    r"""
logical_xor_() -> Tensor

In-place version of :meth:`~Tensor.logical_xor`
""",
)

add_docstr_all(
    "bmm",
    r"""
bmm(batch2) -> Tensor

See :func:`torch.bmm`
""",
)

add_docstr_all(
    "cauchy_",
    r"""
cauchy_(median=0, sigma=1, *, generator=None) -> Tensor

Fills the tensor with numbers drawn from the Cauchy distribution:

.. math::

    f(x) = \dfrac{1}{\pi} \dfrac{\sigma}{(x - \text{median})^2 + \sigma^2}

.. note::
  Sigma (:math:`\sigma`) is used to denote the scale parameter in Cauchy distribution.
""",
)

add_docstr_all(
    "ceil",
    r"""
ceil() -> Tensor

See :func:`torch.ceil`
""",
)

add_docstr_all(
    "ceil_",
    r"""
ceil_() -> Tensor

In-place version of :meth:`~Tensor.ceil`
""",
)

add_docstr_all(
    "cholesky",
    r"""
cholesky(upper=False) -> Tensor

See :func:`torch.cholesky`
""",
)

add_docstr_all(
    "cholesky_solve",
    r"""
cholesky_solve(input2, upper=False) -> Tensor

See :func:`torch.cholesky_solve`
""",
)

add_docstr_all(
    "cholesky_inverse",
    r"""
cholesky_inverse(upper=False) -> Tensor

See :func:`torch.cholesky_inverse`
""",
)

add_docstr_all(
    "clamp",
    r"""
clamp(min=None, max=None) -> Tensor

See :func:`torch.clamp`
""",
)

add_docstr_all(
    "clamp_",
    r"""
clamp_(min=None, max=None) -> Tensor

In-place version of :meth:`~Tensor.clamp`
""",
)

add_docstr_all(
    "clip",
    r"""
clip(min=None, max=None) -> Tensor

Alias for :meth:`~Tensor.clamp`.
""",
)

add_docstr_all(
    "clip_",
    r"""
clip_(min=None, max=None) -> Tensor

Alias for :meth:`~Tensor.clamp_`.
""",
)

add_docstr_all(
    "clone",
    r"""
clone(*, memory_format=torch.preserve_format) -> Tensor

See :func:`torch.clone`
""".format(**common_args),
)

add_docstr_all(
    "coalesce",
    r"""
coalesce() -> Tensor

Returns a coalesced copy of :attr:`self` if :attr:`self` is an
:ref:`uncoalesced tensor <sparse-uncoalesced-coo-docs>`.

Returns :attr:`self` if :attr:`self` is a coalesced tensor.

.. warning::
  Throws an error if :attr:`self` is not a sparse COO tensor.
""",
)

add_docstr_all(
    "contiguous",
    r"""
contiguous(memory_format=torch.contiguous_format) -> Tensor

Returns a contiguous in memory tensor containing the same data as :attr:`self` tensor. If
:attr:`self` tensor is already in the specified memory format, this function returns the
:attr:`self` tensor.

Args:
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.contiguous_format``.
""",
)

add_docstr_all(
    "copy_",
    r"""
copy_(src, non_blocking=False) -> Tensor

Copies the elements from :attr:`src` into :attr:`self` tensor and returns
:attr:`self`.

The :attr:`src` tensor must be :ref:`broadcastable <broadcasting-semantics>`
with the :attr:`self` tensor. It may be of a different data type or reside on a
different device.

Args:
    src (Tensor): the source tensor to copy from
    non_blocking (bool, optional): if ``True`` and this copy is between CPU and GPU,
        the copy may occur asynchronously with respect to the host. For other
        cases, this argument has no effect. Default: ``False``
""",
)

add_docstr_all(
    "conj",
    r"""
conj() -> Tensor

See :func:`torch.conj`
""",
)

add_docstr_all(
    "conj_physical",
    r"""
conj_physical() -> Tensor

See :func:`torch.conj_physical`
""",
)

add_docstr_all(
    "conj_physical_",
    r"""
conj_physical_() -> Tensor

In-place version of :meth:`~Tensor.conj_physical`
""",
)

add_docstr_all(
    "resolve_conj",
    r"""
resolve_conj() -> Tensor

See :func:`torch.resolve_conj`
""",
)

add_docstr_all(
    "resolve_neg",
    r"""
resolve_neg() -> Tensor

See :func:`torch.resolve_neg`
""",
)

add_docstr_all(
    "copysign",
    r"""
copysign(other) -> Tensor

See :func:`torch.copysign`
""",
)

add_docstr_all(
    "copysign_",
    r"""
copysign_(other) -> Tensor

In-place version of :meth:`~Tensor.copysign`
""",
)

add_docstr_all(
    "cos",
    r"""
cos() -> Tensor

See :func:`torch.cos`
""",
)

add_docstr_all(
    "cos_",
    r"""
cos_() -> Tensor

In-place version of :meth:`~Tensor.cos`
""",
)

add_docstr_all(
    "cosh",
    r"""
cosh() -> Tensor

See :func:`torch.cosh`
""",
)

add_docstr_all(
    "cosh_",
    r"""
cosh_() -> Tensor

In-place version of :meth:`~Tensor.cosh`
""",
)

add_docstr_all(
    "cpu",
    r"""
cpu(memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in CPU memory.

If this object is already in CPU memory,
then no copy is performed and the original object is returned.

Args:
    {memory_format}

""".format(**common_args),
)

add_docstr_all(
    "count_nonzero",
    r"""
count_nonzero(dim=None) -> Tensor

See :func:`torch.count_nonzero`
""",
)

add_docstr_all(
    "cov",
    r"""
cov(*, correction=1, fweights=None, aweights=None) -> Tensor

See :func:`torch.cov`
""",
)

add_docstr_all(
    "corrcoef",
    r"""
corrcoef() -> Tensor

See :func:`torch.corrcoef`
""",
)

add_docstr_all(
    "cross",
    r"""
cross(other, dim=None) -> Tensor

See :func:`torch.cross`
""",
)

add_docstr_all(
    "cuda",
    r"""
cuda(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in CUDA memory.

If this object is already in CUDA memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`, optional): The destination GPU device.
        Defaults to the current CUDA device.
    non_blocking (bool, optional): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "mtia",
    r"""
mtia(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in MTIA memory.

If this object is already in MTIA memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`, optional): The destination MTIA device.
        Defaults to the current MTIA device.
    non_blocking (bool, optional): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "ipu",
    r"""
ipu(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in IPU memory.

If this object is already in IPU memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`, optional): The destination IPU device.
        Defaults to the current IPU device.
    non_blocking (bool, optional): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "xpu",
    r"""
xpu(device=None, non_blocking=False, memory_format=torch.preserve_format) -> Tensor

Returns a copy of this object in XPU memory.

If this object is already in XPU memory and on the correct device,
then no copy is performed and the original object is returned.

Args:
    device (:class:`torch.device`, optional): The destination XPU device.
        Defaults to the current XPU device.
    non_blocking (bool, optional): If ``True`` and the source is in pinned memory,
        the copy will be asynchronous with respect to the host.
        Otherwise, the argument has no effect. Default: ``False``.
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "logcumsumexp",
    r"""
logcumsumexp(dim) -> Tensor

See :func:`torch.logcumsumexp`
""",
)

add_docstr_all(
    "cummax",
    r"""
cummax(dim) -> (Tensor, Tensor)

See :func:`torch.cummax`
""",
)

add_docstr_all(
    "cummin",
    r"""
cummin(dim) -> (Tensor, Tensor)

See :func:`torch.cummin`
""",
)

add_docstr_all(
    "cumprod",
    r"""
cumprod(dim, dtype=None) -> Tensor

See :func:`torch.cumprod`
""",
)

add_docstr_all(
    "cumprod_",
    r"""
cumprod_(dim, dtype=None) -> Tensor

In-place version of :meth:`~Tensor.cumprod`
""",
)

add_docstr_all(
    "cumsum",
    r"""
cumsum(dim, dtype=None) -> Tensor

See :func:`torch.cumsum`
""",
)

add_docstr_all(
    "cumsum_",
    r"""
cumsum_(dim, dtype=None) -> Tensor

In-place version of :meth:`~Tensor.cumsum`
""",
)

add_docstr_all(
    "data_ptr",
    r"""
data_ptr() -> int

Returns the address of the first element of :attr:`self` tensor.
""",
)

add_docstr_all(
    "dequantize",
    r"""
dequantize() -> Tensor

Given a quantized Tensor, dequantize it and return the dequantized float Tensor.
""",
)

add_docstr_all(
    "dense_dim",
    r"""
dense_dim() -> int

Return the number of dense dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.

.. note::
  Returns ``len(self.shape)`` if :attr:`self` is not a sparse tensor.

See also :meth:`Tensor.sparse_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.
""",
)

add_docstr_all(
    "diag",
    r"""
diag(diagonal=0) -> Tensor

See :func:`torch.diag`
""",
)

add_docstr_all(
    "diag_embed",
    r"""
diag_embed(offset=0, dim1=-2, dim2=-1) -> Tensor

See :func:`torch.diag_embed`
""",
)

add_docstr_all(
    "diagflat",
    r"""
diagflat(offset=0) -> Tensor

See :func:`torch.diagflat`
""",
)

add_docstr_all(
    "diagonal",
    r"""
diagonal(offset=0, dim1=0, dim2=1) -> Tensor

See :func:`torch.diagonal`
""",
)

add_docstr_all(
    "diagonal_scatter",
    r"""
diagonal_scatter(src, offset=0, dim1=0, dim2=1) -> Tensor

See :func:`torch.diagonal_scatter`
""",
)

add_docstr_all(
    "as_strided_scatter",
    r"""
as_strided_scatter(src, size, stride, storage_offset=None) -> Tensor

See :func:`torch.as_strided_scatter`
""",
)

add_docstr_all(
    "fill_diagonal_",
    r"""
fill_diagonal_(fill_value, wrap=False) -> Tensor

Fill the main diagonal of a tensor that has at least 2-dimensions.
When dims>2, all dimensions of input must be of equal length.
This function modifies the input tensor in-place, and returns the input tensor.

Arguments:
    fill_value (Scalar): the fill value
    wrap (bool, optional): the diagonal 'wrapped' after N columns for tall matrices. Default: ``False``

Example::

    >>> a = torch.zeros(3, 3)
    >>> a.fill_diagonal_(5)
    tensor([[5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.]])
    >>> b = torch.zeros(7, 3)
    >>> b.fill_diagonal_(5)
    tensor([[5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])
    >>> c = torch.zeros(7, 3)
    >>> c.fill_diagonal_(5, wrap=True)
    tensor([[5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.],
            [0., 0., 0.],
            [5., 0., 0.],
            [0., 5., 0.],
            [0., 0., 5.]])

""",
)

add_docstr_all(
    "floor_divide",
    r"""
floor_divide(value) -> Tensor

See :func:`torch.floor_divide`
""",
)

add_docstr_all(
    "floor_divide_",
    r"""
floor_divide_(value) -> Tensor

In-place version of :meth:`~Tensor.floor_divide`
""",
)

add_docstr_all(
    "diff",
    r"""
diff(n=1, dim=-1, prepend=None, append=None) -> Tensor

See :func:`torch.diff`
""",
)

add_docstr_all(
    "digamma",
    r"""
digamma() -> Tensor

See :func:`torch.digamma`
""",
)

add_docstr_all(
    "digamma_",
    r"""
digamma_() -> Tensor

In-place version of :meth:`~Tensor.digamma`
""",
)

add_docstr_all(
    "dim",
    r"""
dim() -> int

Returns the number of dimensions of :attr:`self` tensor.
""",
)

add_docstr_all(
    "dist",
    r"""
dist(other, p=2) -> Tensor

See :func:`torch.dist`
""",
)

add_docstr_all(
    "div",
    r"""
div(value, *, rounding_mode=None) -> Tensor

See :func:`torch.div`
""",
)

add_docstr_all(
    "div_",
    r"""
div_(value, *, rounding_mode=None) -> Tensor

In-place version of :meth:`~Tensor.div`
""",
)

add_docstr_all(
    "divide",
    r"""
divide(value, *, rounding_mode=None) -> Tensor

See :func:`torch.divide`
""",
)

add_docstr_all(
    "divide_",
    r"""
divide_(value, *, rounding_mode=None) -> Tensor

In-place version of :meth:`~Tensor.divide`
""",
)

add_docstr_all(
    "dot",
    r"""
dot(other) -> Tensor

See :func:`torch.dot`
""",
)

add_docstr_all(
    "element_size",
    r"""
element_size() -> int

Returns the size in bytes of an individual element.

Example::

    >>> torch.tensor([]).element_size()
    4
    >>> torch.tensor([], dtype=torch.uint8).element_size()
    1

""",
)

add_docstr_all(
    "eq",
    r"""
eq(other) -> Tensor

See :func:`torch.eq`
""",
)

add_docstr_all(
    "eq_",
    r"""
eq_(other) -> Tensor

In-place version of :meth:`~Tensor.eq`
""",
)

add_docstr_all(
    "equal",
    r"""
equal(other) -> bool

See :func:`torch.equal`
""",
)

add_docstr_all(
    "erf",
    r"""
erf() -> Tensor

See :func:`torch.erf`
""",
)

add_docstr_all(
    "erf_",
    r"""
erf_() -> Tensor

In-place version of :meth:`~Tensor.erf`
""",
)

add_docstr_all(
    "erfc",
    r"""
erfc() -> Tensor

See :func:`torch.erfc`
""",
)

add_docstr_all(
    "erfc_",
    r"""
erfc_() -> Tensor

In-place version of :meth:`~Tensor.erfc`
""",
)

add_docstr_all(
    "erfinv",
    r"""
erfinv() -> Tensor

See :func:`torch.erfinv`
""",
)

add_docstr_all(
    "erfinv_",
    r"""
erfinv_() -> Tensor

In-place version of :meth:`~Tensor.erfinv`
""",
)

add_docstr_all(
    "exp",
    r"""
exp() -> Tensor

See :func:`torch.exp`
""",
)

add_docstr_all(
    "exp_",
    r"""
exp_() -> Tensor

In-place version of :meth:`~Tensor.exp`
""",
)

add_docstr_all(
    "exp2",
    r"""
exp2() -> Tensor

See :func:`torch.exp2`
""",
)

add_docstr_all(
    "exp2_",
    r"""
exp2_() -> Tensor

In-place version of :meth:`~Tensor.exp2`
""",
)

add_docstr_all(
    "expm1",
    r"""
expm1() -> Tensor

See :func:`torch.expm1`
""",
)

add_docstr_all(
    "expm1_",
    r"""
expm1_() -> Tensor

In-place version of :meth:`~Tensor.expm1`
""",
)

add_docstr_all(
    "exponential_",
    r"""
exponential_(lambd=1, *, generator=None) -> Tensor

Fills :attr:`self` tensor with elements drawn from the PDF (probability density function):

.. math::

    f(x) = \lambda e^{-\lambda x}, x > 0

.. note::
  In probability theory, exponential distribution is supported on interval [0, :math:`\inf`) (i.e., :math:`x >= 0`)
  implying that zero can be sampled from the exponential distribution.
  However, :func:`torch.Tensor.exponential_` does not sample zero,
  which means that its actual support is the interval (0, :math:`\inf`).

  Note that :func:`torch.distributions.exponential.Exponential` is supported on the interval [0, :math:`\inf`) and can sample zero.
""",
)

add_docstr_all(
    "fill_",
    r"""
fill_(value) -> Tensor

Fills :attr:`self` tensor with the specified value.
""",
)

add_docstr_all(
    "floor",
    r"""
floor() -> Tensor

See :func:`torch.floor`
""",
)

add_docstr_all(
    "flip",
    r"""
flip(dims) -> Tensor

See :func:`torch.flip`
""",
)

add_docstr_all(
    "fliplr",
    r"""
fliplr() -> Tensor

See :func:`torch.fliplr`
""",
)

add_docstr_all(
    "flipud",
    r"""
flipud() -> Tensor

See :func:`torch.flipud`
""",
)

add_docstr_all(
    "roll",
    r"""
roll(shifts, dims) -> Tensor

See :func:`torch.roll`
""",
)

add_docstr_all(
    "floor_",
    r"""
floor_() -> Tensor

In-place version of :meth:`~Tensor.floor`
""",
)

add_docstr_all(
    "fmod",
    r"""
fmod(divisor) -> Tensor

See :func:`torch.fmod`
""",
)

add_docstr_all(
    "fmod_",
    r"""
fmod_(divisor) -> Tensor

In-place version of :meth:`~Tensor.fmod`
""",
)

add_docstr_all(
    "frac",
    r"""
frac() -> Tensor

See :func:`torch.frac`
""",
)

add_docstr_all(
    "frac_",
    r"""
frac_() -> Tensor

In-place version of :meth:`~Tensor.frac`
""",
)

add_docstr_all(
    "frexp",
    r"""
frexp(input) -> (Tensor mantissa, Tensor exponent)

See :func:`torch.frexp`
""",
)

add_docstr_all(
    "flatten",
    r"""
flatten(start_dim=0, end_dim=-1) -> Tensor

See :func:`torch.flatten`
""",
)

add_docstr_all(
    "gather",
    r"""
gather(dim, index) -> Tensor

See :func:`torch.gather`
""",
)

add_docstr_all(
    "gcd",
    r"""
gcd(other) -> Tensor

See :func:`torch.gcd`
""",
)

add_docstr_all(
    "gcd_",
    r"""
gcd_(other) -> Tensor

In-place version of :meth:`~Tensor.gcd`
""",
)

add_docstr_all(
    "ge",
    r"""
ge(other) -> Tensor

See :func:`torch.ge`.
""",
)

add_docstr_all(
    "ge_",
    r"""
ge_(other) -> Tensor

In-place version of :meth:`~Tensor.ge`.
""",
)

add_docstr_all(
    "greater_equal",
    r"""
greater_equal(other) -> Tensor

See :func:`torch.greater_equal`.
""",
)

add_docstr_all(
    "greater_equal_",
    r"""
greater_equal_(other) -> Tensor

In-place version of :meth:`~Tensor.greater_equal`.
""",
)

add_docstr_all(
    "geometric_",
    r"""
geometric_(p, *, generator=None) -> Tensor

Fills :attr:`self` tensor with elements drawn from the geometric distribution:

.. math::

    P(X=k) = (1 - p)^{k - 1} p, k = 1, 2, ...

.. note::
  :func:`torch.Tensor.geometric_` `k`-th trial is the first success hence draws samples in :math:`\{1, 2, \ldots\}`, whereas
  :func:`torch.distributions.geometric.Geometric` :math:`(k+1)`-th trial is the first success
  hence draws samples in :math:`\{0, 1, \ldots\}`.
""",
)

add_docstr_all(
    "geqrf",
    r"""
geqrf() -> (Tensor, Tensor)

See :func:`torch.geqrf`
""",
)

add_docstr_all(
    "ger",
    r"""
ger(vec2) -> Tensor

See :func:`torch.ger`
""",
)

add_docstr_all(
    "inner",
    r"""
inner(other) -> Tensor

See :func:`torch.inner`.
""",
)

add_docstr_all(
    "outer",
    r"""
outer(vec2) -> Tensor

See :func:`torch.outer`.
""",
)

add_docstr_all(
    "hypot",
    r"""
hypot(other) -> Tensor

See :func:`torch.hypot`
""",
)

add_docstr_all(
    "hypot_",
    r"""
hypot_(other) -> Tensor

In-place version of :meth:`~Tensor.hypot`
""",
)

add_docstr_all(
    "i0",
    r"""
i0() -> Tensor

See :func:`torch.i0`
""",
)

add_docstr_all(
    "i0_",
    r"""
i0_() -> Tensor

In-place version of :meth:`~Tensor.i0`
""",
)

add_docstr_all(
    "igamma",
    r"""
igamma(other) -> Tensor

See :func:`torch.igamma`
""",
)

add_docstr_all(
    "igamma_",
    r"""
igamma_(other) -> Tensor

In-place version of :meth:`~Tensor.igamma`
""",
)

add_docstr_all(
    "igammac",
    r"""
igammac(other) -> Tensor
See :func:`torch.igammac`
""",
)

add_docstr_all(
    "igammac_",
    r"""
igammac_(other) -> Tensor
In-place version of :meth:`~Tensor.igammac`
""",
)

add_docstr_all(
    "indices",
    r"""
indices() -> Tensor

Return the indices tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.

.. warning::
  Throws an error if :attr:`self` is not a sparse COO tensor.

See also :meth:`Tensor.values`.

.. note::
  This method can only be called on a coalesced sparse tensor. See
  :meth:`Tensor.coalesce` for details.
""",
)

add_docstr_all(
    "get_device",
    r"""
get_device() -> Device ordinal (Integer)

For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.
For CPU tensors, this function returns `-1`.

Example::

    >>> x = torch.randn(3, 4, 5, device='cuda:0')
    >>> x.get_device()
    0
    >>> x.cpu().get_device()
    -1
""",
)

add_docstr_all(
    "values",
    r"""
values() -> Tensor

Return the values tensor of a :ref:`sparse COO tensor <sparse-coo-docs>`.

.. warning::
  Throws an error if :attr:`self` is not a sparse COO tensor.

See also :meth:`Tensor.indices`.

.. note::
  This method can only be called on a coalesced sparse tensor. See
  :meth:`Tensor.coalesce` for details.
""",
)

add_docstr_all(
    "gt",
    r"""
gt(other) -> Tensor

See :func:`torch.gt`.
""",
)

add_docstr_all(
    "gt_",
    r"""
gt_(other) -> Tensor

In-place version of :meth:`~Tensor.gt`.
""",
)

add_docstr_all(
    "greater",
    r"""
greater(other) -> Tensor

See :func:`torch.greater`.
""",
)

add_docstr_all(
    "greater_",
    r"""
greater_(other) -> Tensor

In-place version of :meth:`~Tensor.greater`.
""",
)

add_docstr_all(
    "has_names",
    r"""
Is ``True`` if any of this tensor's dimensions are named. Otherwise, is ``False``.
""",
)

add_docstr_all(
    "hardshrink",
    r"""
hardshrink(lambd=0.5) -> Tensor

See :func:`torch.nn.functional.hardshrink`
""",
)

add_docstr_all(
    "heaviside",
    r"""
heaviside(values) -> Tensor

See :func:`torch.heaviside`
""",
)

add_docstr_all(
    "heaviside_",
    r"""
heaviside_(values) -> Tensor

In-place version of :meth:`~Tensor.heaviside`
""",
)

add_docstr_all(
    "histc",
    r"""
histc(bins=100, min=0, max=0) -> Tensor

See :func:`torch.histc`
""",
)

add_docstr_all(
    "histogram",
    r"""
histogram(input, bins, *, range=None, weight=None, density=False) -> (Tensor, Tensor)

See :func:`torch.histogram`
""",
)

add_docstr_all(
    "index_add_",
    r"""
index_add_(dim, index, source, *, alpha=1) -> Tensor

Accumulate the elements of :attr:`alpha` times ``source`` into the :attr:`self`
tensor by adding to the indices in the order given in :attr:`index`. For example,
if ``dim == 0``, ``index[i] == j``, and ``alpha=-1``, then the ``i``\ th row of
``source`` is subtracted from the ``j``\ th row of :attr:`self`.

The :attr:`dim`\ th dimension of ``source`` must have the same size as the
length of :attr:`index` (which must be a vector), and all other dimensions must
match :attr:`self`, or an error will be raised.

For a 3-D tensor the output is given as::

    self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
    self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
    self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2

Note:
    {forward_reproducibility_note}

Args:
    dim (int): dimension along which to index
    index (Tensor): indices of ``source`` to select from,
            should have dtype either `torch.int64` or `torch.int32`
    source (Tensor): the tensor containing values to add

Keyword args:
    alpha (Number): the scalar multiplier for ``source``

Example::

    >>> x = torch.ones(5, 3)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2])
    >>> x.index_add_(0, index, t)
    tensor([[  2.,   3.,   4.],
            [  1.,   1.,   1.],
            [  8.,   9.,  10.],
            [  1.,   1.,   1.],
            [  5.,   6.,   7.]])
    >>> x.index_add_(0, index, t, alpha=-1)
    tensor([[  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.],
            [  1.,   1.,   1.]])
""".format(**reproducibility_notes),
)

add_docstr_all(
    "index_copy_",
    r"""
index_copy_(dim, index, tensor) -> Tensor

Copies the elements of :attr:`tensor` into the :attr:`self` tensor by selecting
the indices in the order given in :attr:`index`. For example, if ``dim == 0``
and ``index[i] == j``, then the ``i``\ th row of :attr:`tensor` is copied to the
``j``\ th row of :attr:`self`.

The :attr:`dim`\ th dimension of :attr:`tensor` must have the same size as the
length of :attr:`index` (which must be a vector), and all other dimensions must
match :attr:`self`, or an error will be raised.

.. note::
    If :attr:`index` contains duplicate entries, multiple elements from
    :attr:`tensor` will be copied to the same index of :attr:`self`. The result
    is nondeterministic since it depends on which copy occurs last.

Args:
    dim (int): dimension along which to index
    index (LongTensor): indices of :attr:`tensor` to select from
    tensor (Tensor): the tensor containing values to copy

Example::

    >>> x = torch.zeros(5, 3)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2])
    >>> x.index_copy_(0, index, t)
    tensor([[ 1.,  2.,  3.],
            [ 0.,  0.,  0.],
            [ 7.,  8.,  9.],
            [ 0.,  0.,  0.],
            [ 4.,  5.,  6.]])
""",
)

add_docstr_all(
    "index_fill_",
    r"""
index_fill_(dim, index, value) -> Tensor

Fills the elements of the :attr:`self` tensor with value :attr:`value` by
selecting the indices in the order given in :attr:`index`.

Args:
    dim (int): dimension along which to index
    index (LongTensor): indices of :attr:`self` tensor to fill in
    value (float): the value to fill with

Example::

    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    >>> index = torch.tensor([0, 2])
    >>> x.index_fill_(1, index, -1)
    tensor([[-1.,  2., -1.],
            [-1.,  5., -1.],
            [-1.,  8., -1.]])
""",
)

add_docstr_all(
    "index_put_",
    r"""
index_put_(indices, values, accumulate=False) -> Tensor

Puts values from the tensor :attr:`values` into the tensor :attr:`self` using
the indices specified in :attr:`indices` (which is a tuple of Tensors). The
expression ``tensor.index_put_(indices, values)`` is equivalent to
``tensor[indices] = values``. Returns :attr:`self`.

If :attr:`accumulate` is ``True``, the elements in :attr:`values` are added to
:attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
contain duplicate elements.

Args:
    indices (tuple of LongTensor): tensors used to index into `self`.
    values (Tensor): tensor of same dtype as `self`.
    accumulate (bool): whether to accumulate into self
""",
)

add_docstr_all(
    "index_put",
    r"""
index_put(indices, values, accumulate=False) -> Tensor

Out-place version of :meth:`~Tensor.index_put_`.
""",
)

add_docstr_all(
    "index_reduce_",
    r"""
index_reduce_(dim, index, source, reduce, *, include_self=True) -> Tensor

Accumulate the elements of ``source`` into the :attr:`self`
tensor by accumulating to the indices in the order given in :attr:`index`
using the reduction given by the ``reduce`` argument. For example, if ``dim == 0``,
``index[i] == j``, ``reduce == prod`` and ``include_self == True`` then the ``i``\ th
row of ``source`` is multiplied by the ``j``\ th row of :attr:`self`. If
:obj:`include_self="True"`, the values in the :attr:`self` tensor are included
in the reduction, otherwise, rows in the :attr:`self` tensor that are accumulated
to are treated as if they were filled with the reduction identities.

The :attr:`dim`\ th dimension of ``source`` must have the same size as the
length of :attr:`index` (which must be a vector), and all other dimensions must
match :attr:`self`, or an error will be raised.

For a 3-D tensor with :obj:`reduce="prod"` and :obj:`include_self=True` the
output is given as::

    self[index[i], :, :] *= src[i, :, :]  # if dim == 0
    self[:, index[i], :] *= src[:, i, :]  # if dim == 1
    self[:, :, index[i]] *= src[:, :, i]  # if dim == 2

Note:
    {forward_reproducibility_note}

.. note::

    This function only supports floating point tensors.

.. warning::

    This function is in beta and may change in the near future.

Args:
    dim (int): dimension along which to index
    index (Tensor): indices of ``source`` to select from,
        should have dtype either `torch.int64` or `torch.int32`
    source (FloatTensor): the tensor containing values to accumulate
    reduce (str): the reduction operation to apply
        (:obj:`"prod"`, :obj:`"mean"`, :obj:`"amax"`, :obj:`"amin"`)

Keyword args:
    include_self (bool): whether the elements from the ``self`` tensor are
        included in the reduction

Example::

    >>> x = torch.empty(5, 3).fill_(2)
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float)
    >>> index = torch.tensor([0, 4, 2, 0])
    >>> x.index_reduce_(0, index, t, 'prod')
    tensor([[20., 44., 72.],
            [ 2.,  2.,  2.],
            [14., 16., 18.],
            [ 2.,  2.,  2.],
            [ 8., 10., 12.]])
    >>> x = torch.empty(5, 3).fill_(2)
    >>> x.index_reduce_(0, index, t, 'prod', include_self=False)
    tensor([[10., 22., 36.],
            [ 2.,  2.,  2.],
            [ 7.,  8.,  9.],
            [ 2.,  2.,  2.],
            [ 4.,  5.,  6.]])
""".format(**reproducibility_notes),
)

add_docstr_all(
    "index_select",
    r"""
index_select(dim, index) -> Tensor

See :func:`torch.index_select`
""",
)

add_docstr_all(
    "sparse_mask",
    r"""
sparse_mask(mask) -> Tensor

Returns a new :ref:`sparse tensor <sparse-docs>` with values from a
strided tensor :attr:`self` filtered by the indices of the sparse
tensor :attr:`mask`. The values of :attr:`mask` sparse tensor are
ignored. :attr:`self` and :attr:`mask` tensors must have the same
shape.

.. note::

  The returned sparse tensor might contain duplicate values if :attr:`mask`
  is not coalesced. It is therefore advisable to pass ``mask.coalesce()``
  if such behavior is not desired.

.. note::

  The returned sparse tensor has the same indices as the sparse tensor
  :attr:`mask`, even when the corresponding values in :attr:`self` are
  zeros.

Args:
    mask (Tensor): a sparse tensor whose indices are used as a filter

Example::

    >>> nse = 5
    >>> dims = (5, 5, 2, 2)
    >>> I = torch.cat([torch.randint(0, dims[0], size=(nse,)),
    ...                torch.randint(0, dims[1], size=(nse,))], 0).reshape(2, nse)
    >>> V = torch.randn(nse, dims[2], dims[3])
    >>> S = torch.sparse_coo_tensor(I, V, dims).coalesce()
    >>> D = torch.randn(dims)
    >>> D.sparse_mask(S)
    tensor(indices=tensor([[0, 0, 0, 2],
                           [0, 1, 4, 3]]),
           values=tensor([[[ 1.6550,  0.2397],
                           [-0.1611, -0.0779]],

                          [[ 0.2326, -1.0558],
                           [ 1.4711,  1.9678]],

                          [[-0.5138, -0.0411],
                           [ 1.9417,  0.5158]],

                          [[ 0.0793,  0.0036],
                           [-0.2569, -0.1055]]]),
           size=(5, 5, 2, 2), nnz=4, layout=torch.sparse_coo)
""",
)

add_docstr_all(
    "inverse",
    r"""
inverse() -> Tensor

See :func:`torch.inverse`
""",
)

add_docstr_all(
    "isnan",
    r"""
isnan() -> Tensor

See :func:`torch.isnan`
""",
)

add_docstr_all(
    "isinf",
    r"""
isinf() -> Tensor

See :func:`torch.isinf`
""",
)

add_docstr_all(
    "isposinf",
    r"""
isposinf() -> Tensor

See :func:`torch.isposinf`
""",
)

add_docstr_all(
    "isneginf",
    r"""
isneginf() -> Tensor

See :func:`torch.isneginf`
""",
)

add_docstr_all(
    "isfinite",
    r"""
isfinite() -> Tensor

See :func:`torch.isfinite`
""",
)

add_docstr_all(
    "isclose",
    r"""
isclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

See :func:`torch.isclose`
""",
)

add_docstr_all(
    "isreal",
    r"""
isreal() -> Tensor

See :func:`torch.isreal`
""",
)

add_docstr_all(
    "is_coalesced",
    r"""
is_coalesced() -> bool

Returns ``True`` if :attr:`self` is a :ref:`sparse COO tensor
<sparse-coo-docs>` that is coalesced, ``False`` otherwise.

.. warning::
  Throws an error if :attr:`self` is not a sparse COO tensor.

See :meth:`coalesce` and :ref:`uncoalesced tensors <sparse-uncoalesced-coo-docs>`.
""",
)

add_docstr_all(
    "is_contiguous",
    r"""
is_contiguous(memory_format=torch.contiguous_format) -> bool

Returns True if :attr:`self` tensor is contiguous in memory in the order specified
by memory format.

Args:
    memory_format (:class:`torch.memory_format`, optional): Specifies memory allocation
        order. Default: ``torch.contiguous_format``.
""",
)

add_docstr_all(
    "is_pinned",
    r"""
Returns true if this tensor resides in pinned memory.
By default, the device pinned memory on will be the current :ref:`accelerator<accelerators>`.
""",
)

add_docstr_all(
    "is_floating_point",
    r"""
is_floating_point() -> bool

Returns True if the data type of :attr:`self` is a floating point data type.
""",
)

add_docstr_all(
    "is_complex",
    r"""
is_complex() -> bool

Returns True if the data type of :attr:`self` is a complex data type.
""",
)

add_docstr_all(
    "is_inference",
    r"""
is_inference() -> bool

See :func:`torch.is_inference`
""",
)

add_docstr_all(
    "is_conj",
    r"""
is_conj() -> bool

Returns True if the conjugate bit of :attr:`self` is set to true.
""",
)

add_docstr_all(
    "is_neg",
    r"""
is_neg() -> bool

Returns True if the negative bit of :attr:`self` is set to true.
""",
)

add_docstr_all(
    "is_signed",
    r"""
is_signed() -> bool

Returns True if the data type of :attr:`self` is a signed data type.
""",
)

add_docstr_all(
    "is_set_to",
    r"""
is_set_to(tensor) -> bool

Returns True if both tensors are pointing to the exact same memory (same
storage, offset, size and stride).
""",
)

add_docstr_all(
    "item",
    r"""
item() -> number

Returns the value of this tensor as a standard Python number. This only works
for tensors with one element. For other cases, see :meth:`~Tensor.tolist`.

This operation is not differentiable.

Example::

    >>> x = torch.tensor([1.0])
    >>> x.item()
    1.0

""",
)

add_docstr_all(
    "kron",
    r"""
kron(other) -> Tensor

See :func:`torch.kron`
""",
)

add_docstr_all(
    "kthvalue",
    r"""
kthvalue(k, dim=None, keepdim=False) -> (Tensor, LongTensor)

See :func:`torch.kthvalue`
""",
)

add_docstr_all(
    "ldexp",
    r"""
ldexp(other) -> Tensor

See :func:`torch.ldexp`
""",
)

add_docstr_all(
    "ldexp_",
    r"""
ldexp_(other) -> Tensor

In-place version of :meth:`~Tensor.ldexp`
""",
)

add_docstr_all(
    "lcm",
    r"""
lcm(other) -> Tensor

See :func:`torch.lcm`
""",
)

add_docstr_all(
    "lcm_",
    r"""
lcm_(other) -> Tensor

In-place version of :meth:`~Tensor.lcm`
""",
)

add_docstr_all(
    "le",
    r"""
le(other) -> Tensor

See :func:`torch.le`.
""",
)

add_docstr_all(
    "le_",
    r"""
le_(other) -> Tensor

In-place version of :meth:`~Tensor.le`.
""",
)

add_docstr_all(
    "less_equal",
    r"""
less_equal(other) -> Tensor

See :func:`torch.less_equal`.
""",
)

add_docstr_all(
    "less_equal_",
    r"""
less_equal_(other) -> Tensor

In-place version of :meth:`~Tensor.less_equal`.
""",
)

add_docstr_all(
    "lerp",
    r"""
lerp(end, weight) -> Tensor

See :func:`torch.lerp`
""",
)

add_docstr_all(
    "lerp_",
    r"""
lerp_(end, weight) -> Tensor

In-place version of :meth:`~Tensor.lerp`
""",
)

add_docstr_all(
    "lgamma",
    r"""
lgamma() -> Tensor

See :func:`torch.lgamma`
""",
)

add_docstr_all(
    "lgamma_",
    r"""
lgamma_() -> Tensor

In-place version of :meth:`~Tensor.lgamma`
""",
)

add_docstr_all(
    "log",
    r"""
log() -> Tensor

See :func:`torch.log`
""",
)

add_docstr_all(
    "log_",
    r"""
log_() -> Tensor

In-place version of :meth:`~Tensor.log`
""",
)

add_docstr_all(
    "log10",
    r"""
log10() -> Tensor

See :func:`torch.log10`
""",
)

add_docstr_all(
    "log10_",
    r"""
log10_() -> Tensor

In-place version of :meth:`~Tensor.log10`
""",
)

add_docstr_all(
    "log1p",
    r"""
log1p() -> Tensor

See :func:`torch.log1p`
""",
)

add_docstr_all(
    "log1p_",
    r"""
log1p_() -> Tensor

In-place version of :meth:`~Tensor.log1p`
""",
)

add_docstr_all(
    "log2",
    r"""
log2() -> Tensor

See :func:`torch.log2`
""",
)

add_docstr_all(
    "log2_",
    r"""
log2_() -> Tensor

In-place version of :meth:`~Tensor.log2`
""",
)

add_docstr_all(
    "logaddexp",
    r"""
logaddexp(other) -> Tensor

See :func:`torch.logaddexp`
""",
)

add_docstr_all(
    "logaddexp2",
    r"""
logaddexp2(other) -> Tensor

See :func:`torch.logaddexp2`
""",
)

add_docstr_all(
    "log_normal_",
    r"""
log_normal_(mean=1, std=2, *, generator=None)

Fills :attr:`self` tensor with numbers samples from the log-normal distribution
parameterized by the given mean :math:`\mu` and standard deviation
:math:`\sigma`. Note that :attr:`mean` and :attr:`std` are the mean and
standard deviation of the underlying normal distribution, and not of the
returned distribution:

.. math::

    f(x) = \dfrac{1}{x \sigma \sqrt{2\pi}}\ e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}
""",
)

add_docstr_all(
    "logsumexp",
    r"""
logsumexp(dim, keepdim=False) -> Tensor

See :func:`torch.logsumexp`
""",
)

add_docstr_all(
    "lt",
    r"""
lt(other) -> Tensor

See :func:`torch.lt`.
""",
)

add_docstr_all(
    "lt_",
    r"""
lt_(other) -> Tensor

In-place version of :meth:`~Tensor.lt`.
""",
)

add_docstr_all(
    "less",
    r"""
lt(other) -> Tensor

See :func:`torch.less`.
""",
)

add_docstr_all(
    "less_",
    r"""
less_(other) -> Tensor

In-place version of :meth:`~Tensor.less`.
""",
)

add_docstr_all(
    "lu_solve",
    r"""
lu_solve(LU_data, LU_pivots) -> Tensor

See :func:`torch.lu_solve`
""",
)

add_docstr_all(
    "map_",
    r"""
map_(tensor, callable)

Applies :attr:`callable` for each element in :attr:`self` tensor and the given
:attr:`tensor` and stores the results in :attr:`self` tensor. :attr:`self` tensor and
the given :attr:`tensor` must be :ref:`broadcastable <broadcasting-semantics>`.

The :attr:`callable` should have the signature::

    def callable(a, b) -> number
""",
)

add_docstr_all(
    "masked_scatter_",
    r"""
masked_scatter_(mask, source)

Copies elements from :attr:`source` into :attr:`self` tensor at positions where
the :attr:`mask` is True. Elements from :attr:`source` are copied into :attr:`self`
starting at position 0 of :attr:`source` and continuing in order one-by-one for each
occurrence of :attr:`mask` being True.
The shape of :attr:`mask` must be :ref:`broadcastable <broadcasting-semantics>`
with the shape of the underlying tensor. The :attr:`source` should have at least
as many elements as the number of ones in :attr:`mask`.

Args:
    mask (BoolTensor): the boolean mask
    source (Tensor): the tensor to copy from

.. note::

    The :attr:`mask` operates on the :attr:`self` tensor, not on the given
    :attr:`source` tensor.

Example:

    >>> self = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    >>> mask = torch.tensor(
    ...     [[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]],
    ...     dtype=torch.bool,
    ... )
    >>> source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    >>> self.masked_scatter_(mask, source)
    tensor([[0, 0, 0, 0, 1],
            [2, 3, 0, 4, 5]])

""",
)

add_docstr_all(
    "masked_fill_",
    r"""
masked_fill_(mask, value)

Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is
True. The shape of :attr:`mask` must be
:ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
tensor.

Args:
    mask (BoolTensor): the boolean mask
    value (float): the value to fill in with
""",
)

add_docstr_all(
    "masked_select",
    r"""
masked_select(mask) -> Tensor

See :func:`torch.masked_select`
""",
)

add_docstr_all(
    "matrix_power",
    r"""
matrix_power(n) -> Tensor

.. note:: :meth:`~Tensor.matrix_power` is deprecated, use :func:`torch.linalg.matrix_power` instead.

Alias for :func:`torch.linalg.matrix_power`
""",
)

add_docstr_all(
    "matrix_exp",
    r"""
matrix_exp() -> Tensor

See :func:`torch.matrix_exp`
""",
)

add_docstr_all(
    "max",
    r"""
max(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

See :func:`torch.max`
""",
)

add_docstr_all(
    "amax",
    r"""
amax(dim=None, keepdim=False) -> Tensor

See :func:`torch.amax`
""",
)

add_docstr_all(
    "maximum",
    r"""
maximum(other) -> Tensor

See :func:`torch.maximum`
""",
)

add_docstr_all(
    "fmax",
    r"""
fmax(other) -> Tensor

See :func:`torch.fmax`
""",
)

add_docstr_all(
    "argmax",
    r"""
argmax(dim=None, keepdim=False) -> LongTensor

See :func:`torch.argmax`
""",
)

add_docstr_all(
    "argwhere",
    r"""
argwhere() -> Tensor

See :func:`torch.argwhere`
""",
)

add_docstr_all(
    "mean",
    r"""
mean(dim=None, keepdim=False, *, dtype=None) -> Tensor

See :func:`torch.mean`
""",
)

add_docstr_all(
    "nanmean",
    r"""
nanmean(dim=None, keepdim=False, *, dtype=None) -> Tensor

See :func:`torch.nanmean`
""",
)

add_docstr_all(
    "median",
    r"""
median(dim=None, keepdim=False) -> (Tensor, LongTensor)

See :func:`torch.median`
""",
)

add_docstr_all(
    "nanmedian",
    r"""
nanmedian(dim=None, keepdim=False) -> (Tensor, LongTensor)

See :func:`torch.nanmedian`
""",
)

add_docstr_all(
    "min",
    r"""
min(dim=None, keepdim=False) -> Tensor or (Tensor, Tensor)

See :func:`torch.min`
""",
)

add_docstr_all(
    "amin",
    r"""
amin(dim=None, keepdim=False) -> Tensor

See :func:`torch.amin`
""",
)

add_docstr_all(
    "minimum",
    r"""
minimum(other) -> Tensor

See :func:`torch.minimum`
""",
)

add_docstr_all(
    "aminmax",
    r"""
aminmax(*, dim=None, keepdim=False) -> (Tensor min, Tensor max)

See :func:`torch.aminmax`
""",
)

add_docstr_all(
    "fmin",
    r"""
fmin(other) -> Tensor

See :func:`torch.fmin`
""",
)

add_docstr_all(
    "argmin",
    r"""
argmin(dim=None, keepdim=False) -> LongTensor

See :func:`torch.argmin`
""",
)

add_docstr_all(
    "mm",
    r"""
mm(mat2) -> Tensor

See :func:`torch.mm`
""",
)

add_docstr_all(
    "mode",
    r"""
mode(dim=None, keepdim=False) -> (Tensor, LongTensor)

See :func:`torch.mode`
""",
)

add_docstr_all(
    "movedim",
    r"""
movedim(source, destination) -> Tensor

See :func:`torch.movedim`
""",
)

add_docstr_all(
    "moveaxis",
    r"""
moveaxis(source, destination) -> Tensor

See :func:`torch.moveaxis`
""",
)

add_docstr_all(
    "mul",
    r"""
mul(value) -> Tensor

See :func:`torch.mul`.
""",
)

add_docstr_all(
    "mul_",
    r"""
mul_(value) -> Tensor

In-place version of :meth:`~Tensor.mul`.
""",
)

add_docstr_all(
    "multiply",
    r"""
multiply(value) -> Tensor

See :func:`torch.multiply`.
""",
)

add_docstr_all(
    "multiply_",
    r"""
multiply_(value) -> Tensor

In-place version of :meth:`~Tensor.multiply`.
""",
)

add_docstr_all(
    "multinomial",
    r"""
multinomial(num_samples, replacement=False, *, generator=None) -> Tensor

See :func:`torch.multinomial`
""",
)

add_docstr_all(
    "mv",
    r"""
mv(vec) -> Tensor

See :func:`torch.mv`
""",
)

add_docstr_all(
    "mvlgamma",
    r"""
mvlgamma(p) -> Tensor

See :func:`torch.mvlgamma`
""",
)

add_docstr_all(
    "mvlgamma_",
    r"""
mvlgamma_(p) -> Tensor

In-place version of :meth:`~Tensor.mvlgamma`
""",
)

add_docstr_all(
    "narrow",
    r"""
narrow(dimension, start, length) -> Tensor

See :func:`torch.narrow`.
""",
)

add_docstr_all(
    "narrow_copy",
    r"""
narrow_copy(dimension, start, length) -> Tensor

See :func:`torch.narrow_copy`.
""",
)

add_docstr_all(
    "ndimension",
    r"""
ndimension() -> int

Alias for :meth:`~Tensor.dim()`
""",
)

add_docstr_all(
    "nan_to_num",
    r"""
nan_to_num(nan=0.0, posinf=None, neginf=None) -> Tensor

See :func:`torch.nan_to_num`.
""",
)

add_docstr_all(
    "nan_to_num_",
    r"""
nan_to_num_(nan=0.0, posinf=None, neginf=None) -> Tensor

In-place version of :meth:`~Tensor.nan_to_num`.
""",
)

add_docstr_all(
    "ne",
    r"""
ne(other) -> Tensor

See :func:`torch.ne`.
""",
)

add_docstr_all(
    "ne_",
    r"""
ne_(other) -> Tensor

In-place version of :meth:`~Tensor.ne`.
""",
)

add_docstr_all(
    "not_equal",
    r"""
not_equal(other) -> Tensor

See :func:`torch.not_equal`.
""",
)

add_docstr_all(
    "not_equal_",
    r"""
not_equal_(other) -> Tensor

In-place version of :meth:`~Tensor.not_equal`.
""",
)

add_docstr_all(
    "neg",
    r"""
neg() -> Tensor

See :func:`torch.neg`
""",
)

add_docstr_all(
    "negative",
    r"""
negative() -> Tensor

See :func:`torch.negative`
""",
)

add_docstr_all(
    "neg_",
    r"""
neg_() -> Tensor

In-place version of :meth:`~Tensor.neg`
""",
)

add_docstr_all(
    "negative_",
    r"""
negative_() -> Tensor

In-place version of :meth:`~Tensor.negative`
""",
)

add_docstr_all(
    "nelement",
    r"""
nelement() -> int

Alias for :meth:`~Tensor.numel`
""",
)

add_docstr_all(
    "nextafter",
    r"""
nextafter(other) -> Tensor
See :func:`torch.nextafter`
""",
)

add_docstr_all(
    "nextafter_",
    r"""
nextafter_(other) -> Tensor
In-place version of :meth:`~Tensor.nextafter`
""",
)

add_docstr_all(
    "nonzero",
    r"""
nonzero() -> LongTensor

See :func:`torch.nonzero`
""",
)

add_docstr_all(
    "nonzero_static",
    r"""
nonzero_static(input, *, size, fill_value=-1) -> Tensor

Returns a 2-D tensor where each row is the index for a non-zero value.
The returned Tensor has the same `torch.dtype` as `torch.nonzero()`.

Args:
    input (Tensor): the input tensor to count non-zero elements.

Keyword args:
    size (int): the size of non-zero elements expected to be included in the out
        tensor. Pad the out tensor with `fill_value` if the `size` is larger
        than total number of non-zero elements, truncate out tensor if `size`
        is smaller. The size must be a non-negative integer.
    fill_value (int, optional): the value to fill the output tensor with when `size` is larger
        than the total number of non-zero elements. Default is `-1` to represent
        invalid index.

Example:

    # Example 1: Padding
    >>> input_tensor = torch.tensor([[1, 0], [3, 2]])
    >>> static_size = 4
    >>> t = torch.nonzero_static(input_tensor, size=static_size)
    tensor([[  0,   0],
            [  1,   0],
            [  1,   1],
            [  -1, -1]], dtype=torch.int64)

    # Example 2: Truncating
    >>> input_tensor = torch.tensor([[1, 0], [3, 2]])
    >>> static_size = 2
    >>> t = torch.nonzero_static(input_tensor, size=static_size)
    tensor([[  0,   0],
            [  1,   0]], dtype=torch.int64)

    # Example 3: 0 size
    >>> input_tensor = torch.tensor([10])
    >>> static_size = 0
    >>> t = torch.nonzero_static(input_tensor, size=static_size)
    tensor([], size=(0, 1), dtype=torch.int64)

    # Example 4: 0 rank input
    >>> input_tensor = torch.tensor(10)
    >>> static_size = 2
    >>> t = torch.nonzero_static(input_tensor, size=static_size)
    tensor([], size=(2, 0), dtype=torch.int64)
""",
)

add_docstr_all(
    "norm",
    r"""
norm(p=2, dim=None, keepdim=False) -> Tensor

See :func:`torch.norm`
""",
)

add_docstr_all(
    "normal_",
    r"""
normal_(mean=0, std=1, *, generator=None) -> Tensor

Fills :attr:`self` tensor with elements samples from the normal distribution
parameterized by :attr:`mean` and :attr:`std`.
""",
)

add_docstr_all(
    "numel",
    r"""
numel() -> int

See :func:`torch.numel`
""",
)

add_docstr_all(
    "numpy",
    r"""
numpy(*, force=False) -> numpy.ndarray

Returns the tensor as a NumPy :class:`ndarray`.

If :attr:`force` is ``False`` (the default), the conversion
is performed only if the tensor is on the CPU, does not require grad,
does not have its conjugate bit set, and is a dtype and layout that
NumPy supports. The returned ndarray and the tensor will share their
storage, so changes to the tensor will be reflected in the ndarray
and vice versa.

If :attr:`force` is ``True`` this is equivalent to
calling ``t.detach().cpu().resolve_conj().resolve_neg().numpy()``.
If the tensor isn't on the CPU or the conjugate or negative bit is set,
the tensor won't share its storage with the returned ndarray.
Setting :attr:`force` to ``True`` can be a useful shorthand.

Args:
    force (bool): if ``True``, the ndarray may be a copy of the tensor
               instead of always sharing memory, defaults to ``False``.
""",
)

add_docstr_all(
    "orgqr",
    r"""
orgqr(input2) -> Tensor

See :func:`torch.orgqr`
""",
)

add_docstr_all(
    "ormqr",
    r"""
ormqr(input2, input3, left=True, transpose=False) -> Tensor

See :func:`torch.ormqr`
""",
)

add_docstr_all(
    "permute",
    r"""
permute(*dims) -> Tensor

See :func:`torch.permute`
""",
)

add_docstr_all(
    "polygamma",
    r"""
polygamma(n) -> Tensor

See :func:`torch.polygamma`
""",
)

add_docstr_all(
    "polygamma_",
    r"""
polygamma_(n) -> Tensor

In-place version of :meth:`~Tensor.polygamma`
""",
)

add_docstr_all(
    "positive",
    r"""
positive() -> Tensor

See :func:`torch.positive`
""",
)

add_docstr_all(
    "pow",
    r"""
pow(exponent) -> Tensor

See :func:`torch.pow`
""",
)

add_docstr_all(
    "pow_",
    r"""
pow_(exponent) -> Tensor

In-place version of :meth:`~Tensor.pow`
""",
)

add_docstr_all(
    "float_power",
    r"""
float_power(exponent) -> Tensor

See :func:`torch.float_power`
""",
)

add_docstr_all(
    "float_power_",
    r"""
float_power_(exponent) -> Tensor

In-place version of :meth:`~Tensor.float_power`
""",
)

add_docstr_all(
    "prod",
    r"""
prod(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.prod`
""",
)

add_docstr_all(
    "put_",
    r"""
put_(index, source, accumulate=False) -> Tensor

Copies the elements from :attr:`source` into the positions specified by
:attr:`index`. For the purpose of indexing, the :attr:`self` tensor is treated as if
it were a 1-D tensor.

:attr:`index` and :attr:`source` need to have the same number of elements, but not necessarily
the same shape.

If :attr:`accumulate` is ``True``, the elements in :attr:`source` are added to
:attr:`self`. If accumulate is ``False``, the behavior is undefined if :attr:`index`
contain duplicate elements.

Args:
    index (LongTensor): the indices into self
    source (Tensor): the tensor containing values to copy from
    accumulate (bool, optional): whether to accumulate into self. Default: ``False``

Example::

    >>> src = torch.tensor([[4, 3, 5],
    ...                     [6, 7, 8]])
    >>> src.put_(torch.tensor([1, 3]), torch.tensor([9, 10]))
    tensor([[  4,   9,   5],
            [ 10,   7,   8]])
""",
)

add_docstr_all(
    "put",
    r"""
put(input, index, source, accumulate=False) -> Tensor

Out-of-place version of :meth:`torch.Tensor.put_`.
`input` corresponds to `self` in :meth:`torch.Tensor.put_`.
""",
)

add_docstr_all(
    "qr",
    r"""
qr(some=True) -> (Tensor, Tensor)

See :func:`torch.qr`
""",
)

add_docstr_all(
    "qscheme",
    r"""
qscheme() -> torch.qscheme

Returns the quantization scheme of a given QTensor.
""",
)

add_docstr_all(
    "quantile",
    r"""
quantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

See :func:`torch.quantile`
""",
)

add_docstr_all(
    "nanquantile",
    r"""
nanquantile(q, dim=None, keepdim=False, *, interpolation='linear') -> Tensor

See :func:`torch.nanquantile`
""",
)

add_docstr_all(
    "q_scale",
    r"""
q_scale() -> float

Given a Tensor quantized by linear(affine) quantization,
returns the scale of the underlying quantizer().
""",
)

add_docstr_all(
    "q_zero_point",
    r"""
q_zero_point() -> int

Given a Tensor quantized by linear(affine) quantization,
returns the zero_point of the underlying quantizer().
""",
)

add_docstr_all(
    "q_per_channel_scales",
    r"""
q_per_channel_scales() -> Tensor

Given a Tensor quantized by linear (affine) per-channel quantization,
returns a Tensor of scales of the underlying quantizer. It has the number of
elements that matches the corresponding dimensions (from q_per_channel_axis) of
the tensor.
""",
)

add_docstr_all(
    "q_per_channel_zero_points",
    r"""
q_per_channel_zero_points() -> Tensor

Given a Tensor quantized by linear (affine) per-channel quantization,
returns a tensor of zero_points of the underlying quantizer. It has the number of
elements that matches the corresponding dimensions (from q_per_channel_axis) of
the tensor.
""",
)

add_docstr_all(
    "q_per_channel_axis",
    r"""
q_per_channel_axis() -> int

Given a Tensor quantized by linear (affine) per-channel quantization,
returns the index of dimension on which per-channel quantization is applied.
""",
)

add_docstr_all(
    "random_",
    r"""
random_(from=0, to=None, *, generator=None) -> Tensor

Fills :attr:`self` tensor with numbers sampled from the discrete uniform
distribution over ``[from, to - 1]``. If not specified, the values are usually
only bounded by :attr:`self` tensor's data type. However, for floating point
types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
value is representable. For example, `torch.tensor(1, dtype=torch.double).random_()`
will be uniform in ``[0, 2^53]``.
""",
)

add_docstr_all(
    "rad2deg",
    r"""
rad2deg() -> Tensor

See :func:`torch.rad2deg`
""",
)

add_docstr_all(
    "rad2deg_",
    r"""
rad2deg_() -> Tensor

In-place version of :meth:`~Tensor.rad2deg`
""",
)

add_docstr_all(
    "deg2rad",
    r"""
deg2rad() -> Tensor

See :func:`torch.deg2rad`
""",
)

add_docstr_all(
    "deg2rad_",
    r"""
deg2rad_() -> Tensor

In-place version of :meth:`~Tensor.deg2rad`
""",
)

add_docstr_all(
    "ravel",
    r"""
ravel() -> Tensor

see :func:`torch.ravel`
""",
)

add_docstr_all(
    "reciprocal",
    r"""
reciprocal() -> Tensor

See :func:`torch.reciprocal`
""",
)

add_docstr_all(
    "reciprocal_",
    r"""
reciprocal_() -> Tensor

In-place version of :meth:`~Tensor.reciprocal`
""",
)

add_docstr_all(
    "record_stream",
    r"""
record_stream(stream)

Marks the tensor as having been used by this stream.  When the tensor
is deallocated, ensure the tensor memory is not reused for another tensor
until all work queued on :attr:`stream` at the time of deallocation is
complete.

.. note::

    The caching allocator is aware of only the stream where a tensor was
    allocated. Due to the awareness, it already correctly manages the life
    cycle of tensors on only one stream. But if a tensor is used on a stream
    different from the stream of origin, the allocator might reuse the memory
    unexpectedly. Calling this method lets the allocator know which streams
    have used the tensor.

.. warning::

    This method is most suitable for use cases where you are providing a
    function that created a tensor on a side stream, and want users to be able
    to make use of the tensor without having to think carefully about stream
    safety when making use of them.  These safety guarantees come at some
    performance and predictability cost (analogous to the tradeoff between GC
    and manual memory management), so if you are in a situation where
    you manage the full lifetime of your tensors, you may consider instead
    manually managing CUDA events so that calling this method is not necessary.
    In particular, when you call this method, on later allocations the
    allocator will poll the recorded stream to see if all operations have
    completed yet; you can potentially race with side stream computation and
    non-deterministically reuse or fail to reuse memory for an allocation.

    You can safely use tensors allocated on side streams without
    :meth:`~Tensor.record_stream`; you must manually ensure that
    any non-creation stream uses of a tensor are synced back to the creation
    stream before you deallocate the tensor.  As the CUDA caching allocator
    guarantees that the memory will only be reused with the same creation stream,
    this is sufficient to ensure that writes to future reallocations of the
    memory will be delayed until non-creation stream uses are done.
    (Counterintuitively, you may observe that on the CPU side we have already
    reallocated the tensor, even though CUDA kernels on the old tensor are
    still in progress.  This is fine, because CUDA operations on the new
    tensor will appropriately wait for the old operations to complete, as they
    are all on the same stream.)

    Concretely, this looks like this::

        with torch.cuda.stream(s0):
            x = torch.zeros(N)

        s1.wait_stream(s0)
        with torch.cuda.stream(s1):
            y = some_comm_op(x)

        ... some compute on s0 ...

        # synchronize creation stream s0 to side stream s1
        # before deallocating x
        s0.wait_stream(s1)
        del x

    Note that some discretion is required when deciding when to perform
    ``s0.wait_stream(s1)``.  In particular, if we were to wait immediately
    after ``some_comm_op``, there wouldn't be any point in having the side
    stream; it would be equivalent to have run ``some_comm_op`` on ``s0``.
    Instead, the synchronization must be placed at some appropriate, later
    point in time where you expect the side stream ``s1`` to have finished
    work.  This location is typically identified via profiling, e.g., using
    Chrome traces produced
    :meth:`torch.autograd.profiler.profile.export_chrome_trace`.  If you
    place the wait too early, work on s0 will block until ``s1`` has finished,
    preventing further overlapping of communication and computation.  If you
    place the wait too late, you will use more memory than is strictly
    necessary (as you are keeping ``x`` live for longer.)  For a concrete
    example of how this guidance can be applied in practice, see this post:
    `FSDP and CUDACachingAllocator
    <https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486>`_.
""",
)

add_docstr_all(
    "remainder",
    r"""
remainder(divisor) -> Tensor

See :func:`torch.remainder`
""",
)

add_docstr_all(
    "remainder_",
    r"""
remainder_(divisor) -> Tensor

In-place version of :meth:`~Tensor.remainder`
""",
)

add_docstr_all(
    "renorm",
    r"""
renorm(p, dim, maxnorm) -> Tensor

See :func:`torch.renorm`
""",
)

add_docstr_all(
    "renorm_",
    r"""
renorm_(p, dim, maxnorm) -> Tensor

In-place version of :meth:`~Tensor.renorm`
""",
)

add_docstr_all(
    "repeat",
    r"""
repeat(*repeats) -> Tensor

Repeats this tensor along the specified dimensions.

Unlike :meth:`~Tensor.expand`, this function copies the tensor's data.

.. warning::

    :meth:`~Tensor.repeat` behaves differently from
    `numpy.repeat <https://numpy.org/doc/stable/reference/generated/numpy.repeat.html>`_,
    but is more similar to
    `numpy.tile <https://numpy.org/doc/stable/reference/generated/numpy.tile.html>`_.
    For the operator similar to `numpy.repeat`, see :func:`torch.repeat_interleave`.

Args:
    repeat (torch.Size, int..., tuple of int or list of int): The number of times to repeat this tensor along each dimension

Example::

    >>> x = torch.tensor([1, 2, 3])
    >>> x.repeat(4, 2)
    tensor([[ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3],
            [ 1,  2,  3,  1,  2,  3]])
    >>> x.repeat(4, 2, 1).size()
    torch.Size([4, 2, 3])
""",
)

add_docstr_all(
    "repeat_interleave",
    r"""
repeat_interleave(repeats, dim=None, *, output_size=None) -> Tensor

See :func:`torch.repeat_interleave`.
""",
)

add_docstr_all(
    "requires_grad_",
    r"""
requires_grad_(requires_grad=True) -> Tensor

Change if autograd should record operations on this tensor: sets this tensor's
:attr:`requires_grad` attribute in-place. Returns this tensor.

:func:`requires_grad_`'s main use case is to tell autograd to begin recording
operations on a Tensor ``tensor``. If ``tensor`` has ``requires_grad=False``
(because it was obtained through a DataLoader, or required preprocessing or
initialization), ``tensor.requires_grad_()`` makes it so that autograd will
begin to record operations on ``tensor``.

Args:
    requires_grad (bool): If autograd should record operations on this tensor.
        Default: ``True``.

Example::

    >>> # Let's say we want to preprocess some saved weights and use
    >>> # the result as new weights.
    >>> saved_weights = [0.1, 0.2, 0.3, 0.25]
    >>> loaded_weights = torch.tensor(saved_weights)
    >>> weights = preprocess(loaded_weights)  # some function
    >>> weights
    tensor([-0.5503,  0.4926, -2.1158, -0.8303])

    >>> # Now, start to record operations done to weights
    >>> weights.requires_grad_()
    >>> out = weights.pow(2).sum()
    >>> out.backward()
    >>> weights.grad
    tensor([-1.1007,  0.9853, -4.2316, -1.6606])

""",
)

add_docstr_all(
    "reshape",
    r"""
reshape(*shape) -> Tensor

Returns a tensor with the same data and number of elements as :attr:`self`
but with the specified shape. This method returns a view if :attr:`shape` is
compatible with the current shape. See :meth:`torch.Tensor.view` on when it is
possible to return a view.

See :func:`torch.reshape`

Args:
    shape (tuple of ints or int...): the desired shape

""",
)

add_docstr_all(
    "reshape_as",
    r"""
reshape_as(other) -> Tensor

Returns this tensor as the same shape as :attr:`other`.
``self.reshape_as(other)`` is equivalent to ``self.reshape(other.sizes())``.
This method returns a view if ``other.sizes()`` is compatible with the current
shape. See :meth:`torch.Tensor.view` on when it is possible to return a view.

Please see :meth:`reshape` for more information about ``reshape``.

Args:
    other (:class:`torch.Tensor`): The result tensor has the same shape
        as :attr:`other`.
""",
)

add_docstr_all(
    "resize_",
    r"""
resize_(*sizes, memory_format=torch.contiguous_format) -> Tensor

Resizes :attr:`self` tensor to the specified size. If the number of elements is
larger than the current storage size, then the underlying storage is resized
to fit the new number of elements. If the number of elements is smaller, the
underlying storage is not changed. Existing elements are preserved but any new
memory is uninitialized.

.. warning::

    This is a low-level method. The storage is reinterpreted as C-contiguous,
    ignoring the current strides (unless the target size equals the current
    size, in which case the tensor is left unchanged). For most purposes, you
    will instead want to use :meth:`~Tensor.view()`, which checks for
    contiguity, or :meth:`~Tensor.reshape()`, which copies data if needed. To
    change the size in-place with custom strides, see :meth:`~Tensor.set_()`.

.. note::

    If :func:`torch.use_deterministic_algorithms()` and
    :attr:`torch.utils.deterministic.fill_uninitialized_memory` are both set to
    ``True``, new elements are initialized to prevent nondeterministic behavior
    from using the result as an input to an operation. Floating point and
    complex values are set to NaN, and integer values are set to the maximum
    value.

Args:
    sizes (torch.Size or int...): the desired size
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        Tensor. Default: ``torch.contiguous_format``. Note that memory format of
        :attr:`self` is going to be unaffected if ``self.size()`` matches ``sizes``.

Example::

    >>> x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    >>> x.resize_(2, 2)
    tensor([[ 1,  2],
            [ 3,  4]])
""",
)

add_docstr_all(
    "resize_as_",
    r"""
resize_as_(tensor, memory_format=torch.contiguous_format) -> Tensor

Resizes the :attr:`self` tensor to be the same size as the specified
:attr:`tensor`. This is equivalent to ``self.resize_(tensor.size())``.

Args:
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        Tensor. Default: ``torch.contiguous_format``. Note that memory format of
        :attr:`self` is going to be unaffected if ``self.size()`` matches ``tensor.size()``.

""",
)

add_docstr_all(
    "rot90",
    r"""
rot90(k, dims) -> Tensor

See :func:`torch.rot90`
""",
)

add_docstr_all(
    "round",
    r"""
round(decimals=0) -> Tensor

See :func:`torch.round`
""",
)

add_docstr_all(
    "round_",
    r"""
round_(decimals=0) -> Tensor

In-place version of :meth:`~Tensor.round`
""",
)

add_docstr_all(
    "rsqrt",
    r"""
rsqrt() -> Tensor

See :func:`torch.rsqrt`
""",
)

add_docstr_all(
    "rsqrt_",
    r"""
rsqrt_() -> Tensor

In-place version of :meth:`~Tensor.rsqrt`
""",
)

add_docstr_all(
    "scatter_",
    r"""
scatter_(dim, index, src, *, reduce=None) -> Tensor

Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
index is specified by its index in :attr:`src` for ``dimension != dim`` and by
the corresponding value in :attr:`index` for ``dimension = dim``.

For a 3-D tensor, :attr:`self` is updated as::

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

It is also required that
``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
Note that ``input`` and ``index`` do not broadcast against each other for NPUs,
so when running on NPUs, :attr:`input` and :attr:`index` must have the same number of dimensions.
Standard broadcasting occurs in all other cases.

Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
between ``0`` and ``self.size(dim) - 1`` inclusive.

.. warning::

    When indices are not unique, the behavior is non-deterministic (one of the
    values from ``src`` will be picked arbitrarily) and the gradient will be
    incorrect (it will be propagated to all locations in the source that
    correspond to the same index)!

.. note::

    The backward pass is implemented only for ``src.shape == index.shape``.

Additionally accepts an optional :attr:`reduce` argument that allows
specification of an optional reduction operation, which is applied to all
values in the tensor :attr:`src` into :attr:`self` at the indices
specified in the :attr:`index`. For each value in :attr:`src`, the reduction
operation is applied to an index in :attr:`self` which is specified by
its index in :attr:`src` for ``dimension != dim`` and by the corresponding
value in :attr:`index` for ``dimension = dim``.

Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
is updated as::

    self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

Reducing with the addition operation is the same as using
:meth:`~torch.Tensor.scatter_add_`.

.. warning::
    The reduce argument with Tensor ``src`` is deprecated and will be removed in
    a future PyTorch release. Please use :meth:`~torch.Tensor.scatter_reduce_`
    instead for more reduction options.

Args:
    dim (int): the axis along which to index
    index (LongTensor): the indices of elements to scatter, can be either empty
        or of the same dimensionality as ``src``. When empty, the operation
        returns ``self`` unchanged.
    src (Tensor): the source element(s) to scatter.

Keyword args:
    reduce (str, optional): reduction operation to apply, can be either
        ``'add'`` or ``'multiply'``.

Example::

    >>> src = torch.arange(1, 11).reshape((2, 5))
    >>> src
    tensor([[ 1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10]])
    >>> index = torch.tensor([[0, 1, 2, 0]])
    >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
    tensor([[1, 0, 0, 4, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0]])
    >>> index = torch.tensor([[0, 1, 2], [0, 1, 4]])
    >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
    tensor([[1, 2, 3, 0, 0],
            [6, 7, 0, 0, 8],
            [0, 0, 0, 0, 0]])

    >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
    ...            1.23, reduce='multiply')
    tensor([[2.0000, 2.0000, 2.4600, 2.0000],
            [2.0000, 2.0000, 2.0000, 2.4600]])
    >>> torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
    ...            1.23, reduce='add')
    tensor([[2.0000, 2.0000, 3.2300, 2.0000],
            [2.0000, 2.0000, 2.0000, 3.2300]])

.. function:: scatter_(dim, index, value, *, reduce=None) -> Tensor:
   :noindex:

Writes the value from :attr:`value` into :attr:`self` at the indices
specified in the :attr:`index` tensor.  This operation is equivalent to the previous version,
with the :attr:`src` tensor filled entirely with :attr:`value`.

Args:
    dim (int): the axis along which to index
    index (LongTensor): the indices of elements to scatter, can be either empty
        or of the same dimensionality as ``src``. When empty, the operation
        returns ``self`` unchanged.
    value (Scalar): the value to scatter.

Keyword args:
    reduce (str, optional): reduction operation to apply, can be either
        ``'add'`` or ``'multiply'``.

Example::

    >>> index = torch.tensor([[0, 1]])
    >>> value = 2
    >>> torch.zeros(3, 5).scatter_(0, index, value)
    tensor([[2., 0., 0., 0., 0.],
            [0., 2., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])
""",
)

add_docstr_all(
    "scatter_add_",
    r"""
scatter_add_(dim, index, src) -> Tensor

Adds all values from the tensor :attr:`src` into :attr:`self` at the indices
specified in the :attr:`index` tensor in a similar fashion as
:meth:`~torch.Tensor.scatter_`. For each value in :attr:`src`, it is added to
an index in :attr:`self` which is specified by its index in :attr:`src`
for ``dimension != dim`` and by the corresponding value in :attr:`index` for
``dimension = dim``.

For a 3-D tensor, :attr:`self` is updated as::

    self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

:attr:`self`, :attr:`index` and :attr:`src` should have same number of
dimensions. It is also required that ``index.size(d) <= src.size(d)`` for all
dimensions ``d``, and that ``index.size(d) <= self.size(d)`` for all dimensions
``d != dim``. Note that ``index`` and ``src`` do not broadcast.
When :attr:`index` is empty, we always return the original tensor
without further error checking.

Note:
    {forward_reproducibility_note}

.. note::

    The backward pass is implemented only for ``src.shape == index.shape``.

Args:
    dim (int): the axis along which to index
    index (LongTensor): the indices of elements to scatter and add, can be
        either empty or of the same dimensionality as ``src``. When empty, the
        operation returns ``self`` unchanged.
    src (Tensor): the source elements to scatter and add

Example::

    >>> src = torch.ones((2, 5))
    >>> index = torch.tensor([[0, 1, 2, 0, 0]])
    >>> torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)
    tensor([[1., 0., 0., 1., 1.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.]])
    >>> index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]])
    >>> torch.zeros(3, 5, dtype=src.dtype).scatter_add_(0, index, src)
    tensor([[2., 0., 0., 1., 1.],
            [0., 2., 0., 0., 0.],
            [0., 0., 2., 1., 1.]])

""".format(**reproducibility_notes),
)

add_docstr_all(
    "scatter_reduce_",
    r"""
scatter_reduce_(dim, index, src, reduce, *, include_self=True) -> Tensor

Reduces all values from the :attr:`src` tensor to the indices specified in
the :attr:`index` tensor in the :attr:`self` tensor using the applied reduction
defined via the :attr:`reduce` argument (:obj:`"sum"`, :obj:`"prod"`, :obj:`"mean"`,
:obj:`"amax"`, :obj:`"amin"`). For each value in :attr:`src`, it is reduced to an
index in :attr:`self` which is specified by its index in :attr:`src` for
``dimension != dim`` and by the corresponding value in :attr:`index` for
``dimension = dim``. If :obj:`include_self="True"`, the values in the :attr:`self`
tensor are included in the reduction.

:attr:`self`, :attr:`index` and :attr:`src` should all have
the same number of dimensions. It is also required that
``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
Note that ``index`` and ``src`` do not broadcast.

For a 3-D tensor with :obj:`reduce="sum"` and :obj:`include_self=True` the
output is given as::

    self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

Note:
    {forward_reproducibility_note}

.. note::

    The backward pass is implemented only for ``src.shape == index.shape``.

.. warning::

    This function is in beta and may change in the near future.

Args:
    dim (int): the axis along which to index
    index (LongTensor): the indices of elements to scatter and reduce.
    src (Tensor): the source elements to scatter and reduce
    reduce (str): the reduction operation to apply for non-unique indices
        (:obj:`"sum"`, :obj:`"prod"`, :obj:`"mean"`, :obj:`"amax"`, :obj:`"amin"`)
    include_self (bool): whether elements from the :attr:`self` tensor are
        included in the reduction

Example::

    >>> src = torch.tensor([1., 2., 3., 4., 5., 6.])
    >>> index = torch.tensor([0, 1, 0, 1, 2, 1])
    >>> input = torch.tensor([1., 2., 3., 4.])
    >>> input.scatter_reduce(0, index, src, reduce="sum")
    tensor([5., 14., 8., 4.])
    >>> input.scatter_reduce(0, index, src, reduce="sum", include_self=False)
    tensor([4., 12., 5., 4.])
    >>> input2 = torch.tensor([5., 4., 3., 2.])
    >>> input2.scatter_reduce(0, index, src, reduce="amax")
    tensor([5., 6., 5., 2.])
    >>> input2.scatter_reduce(0, index, src, reduce="amax", include_self=False)
    tensor([3., 6., 5., 2.])


""".format(**reproducibility_notes),
)

add_docstr_all(
    "select",
    r"""
select(dim, index) -> Tensor

See :func:`torch.select`
""",
)

add_docstr_all(
    "select_scatter",
    r"""
select_scatter(src, dim, index) -> Tensor

See :func:`torch.select_scatter`
""",
)

add_docstr_all(
    "slice_scatter",
    r"""
slice_scatter(src, dim=0, start=None, end=None, step=1) -> Tensor

See :func:`torch.slice_scatter`
""",
)

add_docstr_all(
    "set_",
    r"""
set_(source=None, storage_offset=0, size=None, stride=None) -> Tensor

Sets the underlying storage, size, and strides. If :attr:`source` is a tensor,
:attr:`self` tensor will share the same storage and have the same size and
strides as :attr:`source`. Changes to elements in one tensor will be reflected
in the other.

If :attr:`source` is a :class:`~torch.Storage`, the method sets the underlying
storage, offset, size, and stride.

Args:
    source (Tensor or Storage): the tensor or storage to use
    storage_offset (int, optional): the offset in the storage
    size (torch.Size, optional): the desired size. Defaults to the size of the source.
    stride (tuple, optional): the desired stride. Defaults to C-contiguous strides.
""",
)

add_docstr_all(
    "sigmoid",
    r"""
sigmoid() -> Tensor

See :func:`torch.sigmoid`
""",
)

add_docstr_all(
    "sigmoid_",
    r"""
sigmoid_() -> Tensor

In-place version of :meth:`~Tensor.sigmoid`
""",
)

add_docstr_all(
    "logit",
    r"""
logit() -> Tensor

See :func:`torch.logit`
""",
)

add_docstr_all(
    "logit_",
    r"""
logit_() -> Tensor

In-place version of :meth:`~Tensor.logit`
""",
)

add_docstr_all(
    "sign",
    r"""
sign() -> Tensor

See :func:`torch.sign`
""",
)

add_docstr_all(
    "sign_",
    r"""
sign_() -> Tensor

In-place version of :meth:`~Tensor.sign`
""",
)

add_docstr_all(
    "signbit",
    r"""
signbit() -> Tensor

See :func:`torch.signbit`
""",
)

add_docstr_all(
    "sgn",
    r"""
sgn() -> Tensor

See :func:`torch.sgn`
""",
)

add_docstr_all(
    "sgn_",
    r"""
sgn_() -> Tensor

In-place version of :meth:`~Tensor.sgn`
""",
)

add_docstr_all(
    "sin",
    r"""
sin() -> Tensor

See :func:`torch.sin`
""",
)

add_docstr_all(
    "sin_",
    r"""
sin_() -> Tensor

In-place version of :meth:`~Tensor.sin`
""",
)

add_docstr_all(
    "sinc",
    r"""
sinc() -> Tensor

See :func:`torch.sinc`
""",
)

add_docstr_all(
    "sinc_",
    r"""
sinc_() -> Tensor

In-place version of :meth:`~Tensor.sinc`
""",
)

add_docstr_all(
    "sinh",
    r"""
sinh() -> Tensor

See :func:`torch.sinh`
""",
)

add_docstr_all(
    "sinh_",
    r"""
sinh_() -> Tensor

In-place version of :meth:`~Tensor.sinh`
""",
)

add_docstr_all(
    "size",
    r"""
size(dim=None) -> torch.Size or int

Returns the size of the :attr:`self` tensor. If ``dim`` is not specified,
the returned value is a :class:`torch.Size`, a subclass of :class:`tuple`.
If ``dim`` is specified, returns an int holding the size of that dimension.

Args:
  dim (int, optional): The dimension for which to retrieve the size.

Example::

    >>> t = torch.empty(3, 4, 5)
    >>> t.size()
    torch.Size([3, 4, 5])
    >>> t.size(dim=1)
    4

""",
)

add_docstr_all(
    "shape",
    r"""
shape() -> torch.Size

Returns the size of the :attr:`self` tensor. Alias for :attr:`size`.

See also :meth:`Tensor.size`.

Example::

    >>> t = torch.empty(3, 4, 5)
    >>> t.size()
    torch.Size([3, 4, 5])
    >>> t.shape
    torch.Size([3, 4, 5])

""",
)

add_docstr_all(
    "sort",
    r"""
sort(dim=-1, descending=False) -> (Tensor, LongTensor)

See :func:`torch.sort`
""",
)

add_docstr_all(
    "msort",
    r"""
msort() -> Tensor

See :func:`torch.msort`
""",
)

add_docstr_all(
    "argsort",
    r"""
argsort(dim=-1, descending=False) -> LongTensor

See :func:`torch.argsort`
""",
)

add_docstr_all(
    "sparse_dim",
    r"""
sparse_dim() -> int

Return the number of sparse dimensions in a :ref:`sparse tensor <sparse-docs>` :attr:`self`.

.. note::
  Returns ``0`` if :attr:`self` is not a sparse tensor.

See also :meth:`Tensor.dense_dim` and :ref:`hybrid tensors <sparse-hybrid-coo-docs>`.
""",
)

add_docstr_all(
    "sparse_resize_",
    r"""
sparse_resize_(size, sparse_dim, dense_dim) -> Tensor

Resizes :attr:`self` :ref:`sparse tensor <sparse-docs>` to the desired
size and the number of sparse and dense dimensions.

.. note::
  If the number of specified elements in :attr:`self` is zero, then
  :attr:`size`, :attr:`sparse_dim`, and :attr:`dense_dim` can be any
  size and positive integers such that ``len(size) == sparse_dim +
  dense_dim``.

  If :attr:`self` specifies one or more elements, however, then each
  dimension in :attr:`size` must not be smaller than the corresponding
  dimension of :attr:`self`, :attr:`sparse_dim` must equal the number
  of sparse dimensions in :attr:`self`, and :attr:`dense_dim` must
  equal the number of dense dimensions in :attr:`self`.

.. warning::
  Throws an error if :attr:`self` is not a sparse tensor.

Args:
    size (torch.Size): the desired size. If :attr:`self` is non-empty
      sparse tensor, the desired size cannot be smaller than the
      original size.
    sparse_dim (int): the number of sparse dimensions
    dense_dim (int): the number of dense dimensions
""",
)

add_docstr_all(
    "sparse_resize_and_clear_",
    r"""
sparse_resize_and_clear_(size, sparse_dim, dense_dim) -> Tensor

Removes all specified elements from a :ref:`sparse tensor
<sparse-docs>` :attr:`self` and resizes :attr:`self` to the desired
size and the number of sparse and dense dimensions.

.. warning:
  Throws an error if :attr:`self` is not a sparse tensor.

Args:
    size (torch.Size): the desired size.
    sparse_dim (int): the number of sparse dimensions
    dense_dim (int): the number of dense dimensions
""",
)

add_docstr_all(
    "sqrt",
    r"""
sqrt() -> Tensor

See :func:`torch.sqrt`
""",
)

add_docstr_all(
    "sqrt_",
    r"""
sqrt_() -> Tensor

In-place version of :meth:`~Tensor.sqrt`
""",
)

add_docstr_all(
    "square",
    r"""
square() -> Tensor

See :func:`torch.square`
""",
)

add_docstr_all(
    "square_",
    r"""
square_() -> Tensor

In-place version of :meth:`~Tensor.square`
""",
)

add_docstr_all(
    "squeeze",
    r"""
squeeze(dim=None) -> Tensor

See :func:`torch.squeeze`
""",
)

add_docstr_all(
    "squeeze_",
    r"""
squeeze_(dim=None) -> Tensor

In-place version of :meth:`~Tensor.squeeze`
""",
)

add_docstr_all(
    "std",
    r"""
std(dim=None, *, correction=1, keepdim=False) -> Tensor

See :func:`torch.std`
""",
)

add_docstr_all(
    "storage_offset",
    r"""
storage_offset() -> int

Returns :attr:`self` tensor's offset in the underlying storage in terms of
number of storage elements (not bytes).

Example::

    >>> x = torch.tensor([1, 2, 3, 4, 5])
    >>> x.storage_offset()
    0
    >>> x[3:].storage_offset()
    3

""",
)

add_docstr_all(
    "untyped_storage",
    r"""
untyped_storage() -> torch.UntypedStorage

Returns the underlying :class:`UntypedStorage`.
""",
)

add_docstr_all(
    "stride",
    r"""
stride(dim) -> tuple or int

Returns the stride of :attr:`self` tensor.

Stride is the jump necessary to go from one element to the next one in the
specified dimension :attr:`dim`. A tuple of all strides is returned when no
argument is passed in. Otherwise, an integer value is returned as the stride in
the particular dimension :attr:`dim`.

Args:
    dim (int, optional): the desired dimension in which stride is required

Example::

    >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)
    >>> x.stride(0)
    5
    >>> x.stride(-1)
    1

""",
)

add_docstr_all(
    "sub",
    r"""
sub(other, *, alpha=1) -> Tensor

See :func:`torch.sub`.
""",
)

add_docstr_all(
    "sub_",
    r"""
sub_(other, *, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.sub`
""",
)

add_docstr_all(
    "subtract",
    r"""
subtract(other, *, alpha=1) -> Tensor

See :func:`torch.subtract`.
""",
)

add_docstr_all(
    "subtract_",
    r"""
subtract_(other, *, alpha=1) -> Tensor

In-place version of :meth:`~Tensor.subtract`.
""",
)

add_docstr_all(
    "sum",
    r"""
sum(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.sum`
""",
)

add_docstr_all(
    "nansum",
    r"""
nansum(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.nansum`
""",
)

add_docstr_all(
    "svd",
    r"""
svd(some=True, compute_uv=True) -> (Tensor, Tensor, Tensor)

See :func:`torch.svd`
""",
)

add_docstr_all(
    "swapdims",
    r"""
swapdims(dim0, dim1) -> Tensor

See :func:`torch.swapdims`
""",
)

add_docstr_all(
    "swapdims_",
    r"""
swapdims_(dim0, dim1) -> Tensor

In-place version of :meth:`~Tensor.swapdims`
""",
)

add_docstr_all(
    "swapaxes",
    r"""
swapaxes(axis0, axis1) -> Tensor

See :func:`torch.swapaxes`
""",
)

add_docstr_all(
    "swapaxes_",
    r"""
swapaxes_(axis0, axis1) -> Tensor

In-place version of :meth:`~Tensor.swapaxes`
""",
)

add_docstr_all(
    "t",
    r"""
t() -> Tensor

See :func:`torch.t`
""",
)

add_docstr_all(
    "t_",
    r"""
t_() -> Tensor

In-place version of :meth:`~Tensor.t`
""",
)

add_docstr_all(
    "tile",
    r"""
tile(dims) -> Tensor

See :func:`torch.tile`
""",
)

add_docstr_all(
    "to",
    r"""
to(*args, **kwargs) -> Tensor

Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
inferred from the arguments of ``self.to(*args, **kwargs)``.

.. note::

    If the ``self`` Tensor already
    has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
    Otherwise, the returned tensor is a copy of ``self`` with the desired
    :class:`torch.dtype` and :class:`torch.device`.

.. note::

    If ``self`` requires gradients (``requires_grad=True``) but the target
    ``dtype`` specified is an integer type, the returned tensor will implicitly
    set ``requires_grad=False``. This is because only tensors with
    floating-point or complex dtypes can require gradients.

Here are the ways to call ``to``:

.. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
   :noindex:

    Returns a Tensor with the specified :attr:`dtype`

    Args:
        {memory_format}

.. note::

    According to `C++ type conversion rules <https://en.cppreference.com/w/cpp/language/implicit_conversion.html>`_,
    converting floating point value to integer type will truncate the fractional part.
    If the truncated value cannot fit into the target type (e.g., casting ``torch.inf`` to ``torch.long``),
    the behavior is undefined and the result may vary across platforms.

.. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
   :noindex:

    Returns a Tensor with the specified :attr:`device` and (optional)
    :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
    When :attr:`non_blocking` is set to ``True``, the function attempts to perform
    the conversion asynchronously with respect to the host, if possible. This
    asynchronous behavior applies to both pinned and pageable memory. However,
    caution is advised when using this feature. For more information, refer to the
    `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.
    When :attr:`copy` is set, a new Tensor is created even when the Tensor
    already matches the desired conversion.

    Args:
        {memory_format}

.. method:: to(other, non_blocking=False, copy=False) -> Tensor
   :noindex:

    Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
    the Tensor :attr:`other`.
    When :attr:`non_blocking` is set to ``True``, the function attempts to perform
    the conversion asynchronously with respect to the host, if possible. This
    asynchronous behavior applies to both pinned and pageable memory. However,
    caution is advised when using this feature. For more information, refer to the
    `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.
    When :attr:`copy` is set, a new Tensor is created even when the Tensor
    already matches the desired conversion.

Example::

    >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
    >>> tensor.to(torch.float64)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64)

    >>> cuda0 = torch.device('cuda:0')
    >>> tensor.to(cuda0)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], device='cuda:0')

    >>> tensor.to(cuda0, dtype=torch.float64)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

    >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
    >>> tensor.to(other, non_blocking=True)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
""".format(**common_args),
)

add_docstr_all(
    "byte",
    r"""
byte(memory_format=torch.preserve_format) -> Tensor

``self.byte()`` is equivalent to ``self.to(torch.uint8)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "bool",
    r"""
bool(memory_format=torch.preserve_format) -> Tensor

``self.bool()`` is equivalent to ``self.to(torch.bool)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "char",
    r"""
char(memory_format=torch.preserve_format) -> Tensor

``self.char()`` is equivalent to ``self.to(torch.int8)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "bfloat16",
    r"""
bfloat16(memory_format=torch.preserve_format) -> Tensor
``self.bfloat16()`` is equivalent to ``self.to(torch.bfloat16)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "double",
    r"""
double(memory_format=torch.preserve_format) -> Tensor

``self.double()`` is equivalent to ``self.to(torch.float64)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "float",
    r"""
float(memory_format=torch.preserve_format) -> Tensor

``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "cdouble",
    r"""
cdouble(memory_format=torch.preserve_format) -> Tensor

``self.cdouble()`` is equivalent to ``self.to(torch.complex128)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "cfloat",
    r"""
cfloat(memory_format=torch.preserve_format) -> Tensor

``self.cfloat()`` is equivalent to ``self.to(torch.complex64)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "chalf",
    r"""
chalf(memory_format=torch.preserve_format) -> Tensor

``self.chalf()`` is equivalent to ``self.to(torch.complex32)``. See :func:`to`.

Args:
     {memory_format}
 """.format(**common_args),
)

add_docstr_all(
    "half",
    r"""
half(memory_format=torch.preserve_format) -> Tensor

``self.half()`` is equivalent to ``self.to(torch.float16)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "int",
    r"""
int(memory_format=torch.preserve_format) -> Tensor

``self.int()`` is equivalent to ``self.to(torch.int32)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "int_repr",
    r"""
int_repr() -> Tensor

Given a quantized Tensor,
``self.int_repr()`` returns a CPU Tensor with uint8_t as data type that stores the
underlying uint8_t values of the given Tensor.
""",
)


add_docstr_all(
    "long",
    r"""
long(memory_format=torch.preserve_format) -> Tensor

``self.long()`` is equivalent to ``self.to(torch.int64)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "short",
    r"""
short(memory_format=torch.preserve_format) -> Tensor

``self.short()`` is equivalent to ``self.to(torch.int16)``. See :func:`to`.

Args:
    {memory_format}
""".format(**common_args),
)

add_docstr_all(
    "take",
    r"""
take(indices) -> Tensor

See :func:`torch.take`
""",
)

add_docstr_all(
    "take_along_dim",
    r"""
take_along_dim(indices, dim) -> Tensor

See :func:`torch.take_along_dim`
""",
)

add_docstr_all(
    "tan",
    r"""
tan() -> Tensor

See :func:`torch.tan`
""",
)

add_docstr_all(
    "tan_",
    r"""
tan_() -> Tensor

In-place version of :meth:`~Tensor.tan`
""",
)

add_docstr_all(
    "tanh",
    r"""
tanh() -> Tensor

See :func:`torch.tanh`
""",
)

add_docstr_all(
    "softmax",
    r"""
softmax(dim) -> Tensor

Alias for :func:`torch.nn.functional.softmax`.
""",
)

add_docstr_all(
    "tanh_",
    r"""
tanh_() -> Tensor

In-place version of :meth:`~Tensor.tanh`
""",
)

add_docstr_all(
    "tolist",
    r"""
tolist() -> list or number

Returns the tensor as a (nested) list. For scalars, a standard
Python number is returned, just like with :meth:`~Tensor.item`.
Tensors are automatically moved to the CPU first if necessary.

This operation is not differentiable.

Examples::

    >>> a = torch.randn(2, 2)
    >>> a.tolist()
    [[0.012766935862600803, 0.5415473580360413],
     [-0.08909505605697632, 0.7729271650314331]]
    >>> a[0,0].tolist()
    0.012766935862600803
""",
)

add_docstr_all(
    "topk",
    r"""
topk(k, dim=None, largest=True, sorted=True) -> (Tensor, LongTensor)

See :func:`torch.topk`
""",
)

add_docstr_all(
    "to_dense",
    r"""
to_dense(dtype=None, *, masked_grad=True) -> Tensor

Creates a strided copy of :attr:`self` if :attr:`self` is not a strided tensor, otherwise returns :attr:`self`.

Keyword args:
    {dtype}
    masked_grad (bool, optional): If set to ``True`` (default) and
      :attr:`self` has a sparse layout then the backward of
      :meth:`to_dense` returns ``grad.sparse_mask(self)``.

Example::

    >>> s = torch.sparse_coo_tensor(
    ...        torch.tensor([[1, 1],
    ...                      [0, 2]]),
    ...        torch.tensor([9, 10]),
    ...        size=(3, 3))
    >>> s.to_dense()
    tensor([[ 0,  0,  0],
            [ 9,  0, 10],
            [ 0,  0,  0]])
""",
)

add_docstr_all(
    "to_sparse",
    r"""
to_sparse(sparseDims) -> Tensor

Returns a sparse copy of the tensor.  PyTorch supports sparse tensors in
:ref:`coordinate format <sparse-coo-docs>`.

Args:
    sparseDims (int, optional): the number of sparse dimensions to include in the new sparse tensor

Example::

    >>> d = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])
    >>> d
    tensor([[ 0,  0,  0],
            [ 9,  0, 10],
            [ 0,  0,  0]])
    >>> d.to_sparse()
    tensor(indices=tensor([[1, 1],
                           [0, 2]]),
           values=tensor([ 9, 10]),
           size=(3, 3), nnz=2, layout=torch.sparse_coo)
    >>> d.to_sparse(1)
    tensor(indices=tensor([[1]]),
           values=tensor([[ 9,  0, 10]]),
           size=(3, 3), nnz=1, layout=torch.sparse_coo)

.. method:: to_sparse(*, layout=None, blocksize=None, dense_dim=None) -> Tensor
   :noindex:

Returns a sparse tensor with the specified layout and blocksize.  If
the :attr:`self` is strided, the number of dense dimensions could be
specified, and a hybrid sparse tensor will be created, with
`dense_dim` dense dimensions and `self.dim() - 2 - dense_dim` batch
dimension.

.. note:: If the :attr:`self` layout and blocksize parameters match
          with the specified layout and blocksize, return
          :attr:`self`. Otherwise, return a sparse tensor copy of
          :attr:`self`.

Args:

    layout (:class:`torch.layout`, optional): The desired sparse
      layout. One of ``torch.sparse_coo``, ``torch.sparse_csr``,
      ``torch.sparse_csc``, ``torch.sparse_bsr``, or
      ``torch.sparse_bsc``. Default: if ``None``,
      ``torch.sparse_coo``.

    blocksize (list, tuple, :class:`torch.Size`, optional): Block size
      of the resulting BSR or BSC tensor. For other layouts,
      specifying the block size that is not ``None`` will result in a
      RuntimeError exception.  A block size must be a tuple of length
      two such that its items evenly divide the two sparse dimensions.

    dense_dim (int, optional): Number of dense dimensions of the
      resulting CSR, CSC, BSR or BSC tensor.  This argument should be
      used only if :attr:`self` is a strided tensor, and must be a
      value between 0 and dimension of :attr:`self` tensor minus two.

Example::

    >>> x = torch.tensor([[1, 0], [0, 0], [2, 3]])
    >>> x.to_sparse(layout=torch.sparse_coo)
    tensor(indices=tensor([[0, 2, 2],
                           [0, 0, 1]]),
           values=tensor([1, 2, 3]),
           size=(3, 2), nnz=3, layout=torch.sparse_coo)
    >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(1, 2))
    tensor(crow_indices=tensor([0, 1, 1, 2]),
           col_indices=tensor([0, 0]),
           values=tensor([[[1, 0]],
                          [[2, 3]]]), size=(3, 2), nnz=2, layout=torch.sparse_bsr)
    >>> x.to_sparse(layout=torch.sparse_bsr, blocksize=(2, 1))
    RuntimeError: Tensor size(-2) 3 needs to be divisible by blocksize[0] 2
    >>> x.to_sparse(layout=torch.sparse_csr, blocksize=(3, 1))
    RuntimeError: to_sparse for Strided to SparseCsr conversion does not use specified blocksize

    >>> x = torch.tensor([[[1], [0]], [[0], [0]], [[2], [3]]])
    >>> x.to_sparse(layout=torch.sparse_csr, dense_dim=1)
    tensor(crow_indices=tensor([0, 1, 1, 3]),
           col_indices=tensor([0, 0, 1]),
           values=tensor([[1],
                          [2],
                          [3]]), size=(3, 2, 1), nnz=3, layout=torch.sparse_csr)

""",
)

add_docstr_all(
    "to_sparse_csr",
    r"""
to_sparse_csr(dense_dim=None) -> Tensor

Convert a tensor to compressed row storage format (CSR).  Except for
strided tensors, only works with 2D tensors.  If the :attr:`self` is
strided, then the number of dense dimensions could be specified, and a
hybrid CSR tensor will be created, with `dense_dim` dense dimensions
and `self.dim() - 2 - dense_dim` batch dimension.

Args:

    dense_dim (int, optional): Number of dense dimensions of the
      resulting CSR tensor.  This argument should be used only if
      :attr:`self` is a strided tensor, and must be a value between 0
      and dimension of :attr:`self` tensor minus two.

Example::

    >>> dense = torch.randn(5, 5)
    >>> sparse = dense.to_sparse_csr()
    >>> sparse._nnz()
    25

    >>> dense = torch.zeros(3, 3, 1, 1)
    >>> dense[0, 0] = dense[1, 2] = dense[2, 1] = 1
    >>> dense.to_sparse_csr(dense_dim=2)
    tensor(crow_indices=tensor([0, 1, 2, 3]),
           col_indices=tensor([0, 2, 1]),
           values=tensor([[[1.]],

                          [[1.]],

                          [[1.]]]), size=(3, 3, 1, 1), nnz=3,
           layout=torch.sparse_csr)

""",
)

add_docstr_all(
    "to_sparse_csc",
    r"""
to_sparse_csc() -> Tensor

Convert a tensor to compressed column storage (CSC) format.  Except
for strided tensors, only works with 2D tensors.  If the :attr:`self`
is strided, then the number of dense dimensions could be specified,
and a hybrid CSC tensor will be created, with `dense_dim` dense
dimensions and `self.dim() - 2 - dense_dim` batch dimension.

Args:

    dense_dim (int, optional): Number of dense dimensions of the
      resulting CSC tensor.  This argument should be used only if
      :attr:`self` is a strided tensor, and must be a value between 0
      and dimension of :attr:`self` tensor minus two.

Example::

    >>> dense = torch.randn(5, 5)
    >>> sparse = dense.to_sparse_csc()
    >>> sparse._nnz()
    25

    >>> dense = torch.zeros(3, 3, 1, 1)
    >>> dense[0, 0] = dense[1, 2] = dense[2, 1] = 1
    >>> dense.to_sparse_csc(dense_dim=2)
    tensor(ccol_indices=tensor([0, 1, 2, 3]),
           row_indices=tensor([0, 2, 1]),
           values=tensor([[[1.]],

                          [[1.]],

                          [[1.]]]), size=(3, 3, 1, 1), nnz=3,
           layout=torch.sparse_csc)

""",
)

add_docstr_all(
    "to_sparse_bsr",
    r"""
to_sparse_bsr(blocksize, dense_dim) -> Tensor

Convert a tensor to a block sparse row (BSR) storage format of given
blocksize.  If the :attr:`self` is strided, then the number of dense
dimensions could be specified, and a hybrid BSR tensor will be
created, with `dense_dim` dense dimensions and `self.dim() - 2 -
dense_dim` batch dimension.

Args:

    blocksize (list, tuple, :class:`torch.Size`, optional): Block size
      of the resulting BSR tensor. A block size must be a tuple of
      length two such that its items evenly divide the two sparse
      dimensions.

    dense_dim (int, optional): Number of dense dimensions of the
      resulting BSR tensor.  This argument should be used only if
      :attr:`self` is a strided tensor, and must be a value between 0
      and dimension of :attr:`self` tensor minus two.

Example::

    >>> dense = torch.randn(10, 10)
    >>> sparse = dense.to_sparse_csr()
    >>> sparse_bsr = sparse.to_sparse_bsr((5, 5))
    >>> sparse_bsr.col_indices()
    tensor([0, 1, 0, 1])

    >>> dense = torch.zeros(4, 3, 1)
    >>> dense[0:2, 0] = dense[0:2, 2] = dense[2:4, 1] = 1
    >>> dense.to_sparse_bsr((2, 1), 1)
    tensor(crow_indices=tensor([0, 2, 3]),
           col_indices=tensor([0, 2, 1]),
           values=tensor([[[[1.]],

                           [[1.]]],


                          [[[1.]],

                           [[1.]]],


                          [[[1.]],

                           [[1.]]]]), size=(4, 3, 1), nnz=3,
           layout=torch.sparse_bsr)

""",
)

add_docstr_all(
    "to_sparse_bsc",
    r"""
to_sparse_bsc(blocksize, dense_dim) -> Tensor

Convert a tensor to a block sparse column (BSC) storage format of
given blocksize.  If the :attr:`self` is strided, then the number of
dense dimensions could be specified, and a hybrid BSC tensor will be
created, with `dense_dim` dense dimensions and `self.dim() - 2 -
dense_dim` batch dimension.

Args:

    blocksize (list, tuple, :class:`torch.Size`, optional): Block size
      of the resulting BSC tensor. A block size must be a tuple of
      length two such that its items evenly divide the two sparse
      dimensions.

    dense_dim (int, optional): Number of dense dimensions of the
      resulting BSC tensor.  This argument should be used only if
      :attr:`self` is a strided tensor, and must be a value between 0
      and dimension of :attr:`self` tensor minus two.

Example::

    >>> dense = torch.randn(10, 10)
    >>> sparse = dense.to_sparse_csr()
    >>> sparse_bsc = sparse.to_sparse_bsc((5, 5))
    >>> sparse_bsc.row_indices()
    tensor([0, 1, 0, 1])

    >>> dense = torch.zeros(4, 3, 1)
    >>> dense[0:2, 0] = dense[0:2, 2] = dense[2:4, 1] = 1
    >>> dense.to_sparse_bsc((2, 1), 1)
    tensor(ccol_indices=tensor([0, 1, 2, 3]),
           row_indices=tensor([0, 1, 0]),
           values=tensor([[[[1.]],

                           [[1.]]],


                          [[[1.]],

                           [[1.]]],


                          [[[1.]],

                           [[1.]]]]), size=(4, 3, 1), nnz=3,
           layout=torch.sparse_bsc)

""",
)

add_docstr_all(
    "to_mkldnn",
    r"""
to_mkldnn() -> Tensor
Returns a copy of the tensor in ``torch.mkldnn`` layout.

""",
)

add_docstr_all(
    "trace",
    r"""
trace() -> Tensor

See :func:`torch.trace`
""",
)

add_docstr_all(
    "transpose",
    r"""
transpose(dim0, dim1) -> Tensor

See :func:`torch.transpose`
""",
)

add_docstr_all(
    "transpose_",
    r"""
transpose_(dim0, dim1) -> Tensor

In-place version of :meth:`~Tensor.transpose`
""",
)

add_docstr_all(
    "triangular_solve",
    r"""
triangular_solve(A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)

See :func:`torch.triangular_solve`
""",
)

add_docstr_all(
    "tril",
    r"""
tril(diagonal=0) -> Tensor

See :func:`torch.tril`
""",
)

add_docstr_all(
    "tril_",
    r"""
tril_(diagonal=0) -> Tensor

In-place version of :meth:`~Tensor.tril`
""",
)

add_docstr_all(
    "triu",
    r"""
triu(diagonal=0) -> Tensor

See :func:`torch.triu`
""",
)

add_docstr_all(
    "triu_",
    r"""
triu_(diagonal=0) -> Tensor

In-place version of :meth:`~Tensor.triu`
""",
)

add_docstr_all(
    "true_divide",
    r"""
true_divide(value) -> Tensor

See :func:`torch.true_divide`
""",
)

add_docstr_all(
    "true_divide_",
    r"""
true_divide_(value) -> Tensor

In-place version of :meth:`~Tensor.true_divide_`
""",
)

add_docstr_all(
    "trunc",
    r"""
trunc() -> Tensor

See :func:`torch.trunc`
""",
)

add_docstr_all(
    "fix",
    r"""
fix() -> Tensor

See :func:`torch.fix`.
""",
)

add_docstr_all(
    "trunc_",
    r"""
trunc_() -> Tensor

In-place version of :meth:`~Tensor.trunc`
""",
)

add_docstr_all(
    "fix_",
    r"""
fix_() -> Tensor

In-place version of :meth:`~Tensor.fix`
""",
)

add_docstr_all(
    "type",
    r"""
type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
Returns the type if `dtype` is not provided, else casts this object to
the specified type.

If this is already of the correct type, no copy is performed and the
original object is returned.

Args:
    dtype (dtype or string): The desired type
    non_blocking (bool): If ``True``, and the source is in pinned memory
        and destination is on the GPU or vice versa, the copy is performed
        asynchronously with respect to the host. Otherwise, the argument
        has no effect.
    **kwargs: For compatibility, may contain the key ``async`` in place of
        the ``non_blocking`` argument. The ``async`` arg is deprecated.
""",
)

add_docstr_all(
    "type_as",
    r"""
type_as(tensor) -> Tensor

Returns this tensor cast to the type of the given tensor.

This is a no-op if the tensor is already of the correct type. This is
equivalent to ``self.type(tensor.type())``

Args:
    tensor (Tensor): the tensor which has the desired type
""",
)

add_docstr_all(
    "unfold",
    r"""
unfold(dimension, size, step) -> Tensor

Returns a view of the original tensor which contains all slices of size :attr:`size` from
:attr:`self` tensor in the dimension :attr:`dimension`.

Step between two slices is given by :attr:`step`.

If `sizedim` is the size of dimension :attr:`dimension` for :attr:`self`, the size of
dimension :attr:`dimension` in the returned tensor will be
`(sizedim - size) / step + 1`.

An additional dimension of size :attr:`size` is appended in the returned tensor.

Args:
    dimension (int): dimension in which unfolding happens
    size (int): the size of each slice that is unfolded
    step (int): the step between each slice

Example::

    >>> x = torch.arange(1., 8)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> x.unfold(0, 2, 1)
    tensor([[ 1.,  2.],
            [ 2.,  3.],
            [ 3.,  4.],
            [ 4.,  5.],
            [ 5.,  6.],
            [ 6.,  7.]])
    >>> x.unfold(0, 2, 2)
    tensor([[ 1.,  2.],
            [ 3.,  4.],
            [ 5.,  6.]])
""",
)

add_docstr_all(
    "uniform_",
    r"""
uniform_(from=0, to=1, *, generator=None) -> Tensor

Fills :attr:`self` tensor with numbers sampled from the continuous uniform
distribution:

.. math::
    f(x) = \dfrac{1}{\text{to} - \text{from}}
""",
)

add_docstr_all(
    "unsqueeze",
    r"""
unsqueeze(dim) -> Tensor

See :func:`torch.unsqueeze`
""",
)

add_docstr_all(
    "unsqueeze_",
    r"""
unsqueeze_(dim) -> Tensor

In-place version of :meth:`~Tensor.unsqueeze`
""",
)

add_docstr_all(
    "var",
    r"""
var(dim=None, *, correction=1, keepdim=False) -> Tensor

See :func:`torch.var`
""",
)

add_docstr_all(
    "vdot",
    r"""
vdot(other) -> Tensor

See :func:`torch.vdot`
""",
)

add_docstr_all(
    "view",
    r"""
view(*shape) -> Tensor

Returns a new tensor with the same data as the :attr:`self` tensor but of a
different :attr:`shape`.

The returned tensor shares the same data and must have the same number
of elements, but may have a different size. For a tensor to be viewed, the new
view size must be compatible with its original size and stride, i.e., each new
view dimension must either be a subspace of an original dimension, or only span
across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

.. math::

  \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
:meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
returns a view if the shapes are compatible, and copies (equivalent to calling
:meth:`contiguous`) otherwise.

Args:
    shape (torch.Size or int...): the desired size

Example::

    >>> x = torch.randn(4, 4)
    >>> x.size()
    torch.Size([4, 4])
    >>> y = x.view(16)
    >>> y.size()
    torch.Size([16])
    >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    >>> z.size()
    torch.Size([2, 8])

    >>> a = torch.randn(1, 2, 3, 4)
    >>> a.size()
    torch.Size([1, 2, 3, 4])
    >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
    >>> b.size()
    torch.Size([1, 3, 2, 4])
    >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
    >>> c.size()
    torch.Size([1, 3, 2, 4])
    >>> torch.equal(b, c)
    False


.. method:: view(dtype) -> Tensor
   :noindex:

Returns a new tensor with the same data as the :attr:`self` tensor but of a
different :attr:`dtype`.

If the element size of :attr:`dtype` is different than that of ``self.dtype``,
then the size of the last dimension of the output will be scaled
proportionally.  For instance, if :attr:`dtype` element size is twice that of
``self.dtype``, then each pair of elements in the last dimension of
:attr:`self` will be combined, and the size of the last dimension of the output
will be half that of :attr:`self`. If :attr:`dtype` element size is half that
of ``self.dtype``, then each element in the last dimension of :attr:`self` will
be split in two, and the size of the last dimension of the output will be
double that of :attr:`self`. For this to be possible, the following conditions
must be true:

    * ``self.dim()`` must be greater than 0.
    * ``self.stride(-1)`` must be 1.

Additionally, if the element size of :attr:`dtype` is greater than that of
``self.dtype``, the following conditions must be true as well:

    * ``self.size(-1)`` must be divisible by the ratio between the element
      sizes of the dtypes.
    * ``self.storage_offset()`` must be divisible by the ratio between the
      element sizes of the dtypes.
    * The strides of all dimensions, except the last dimension, must be
      divisible by the ratio between the element sizes of the dtypes.

If any of the above conditions are not met, an error is thrown.

.. warning::

    This overload is not supported by TorchScript, and using it in a Torchscript
    program will cause undefined behavior.


Args:
    dtype (:class:`torch.dtype`): the desired dtype

Example::

    >>> x = torch.randn(4, 4)
    >>> x
    tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
            [-0.1520,  0.7472,  0.5617, -0.8649],
            [-2.4724, -0.0334, -0.2976, -0.8499],
            [-0.2109,  1.9913, -0.9607, -0.6123]])
    >>> x.dtype
    torch.float32

    >>> y = x.view(torch.int32)
    >>> y
    tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
            [-1105482831,  1061112040,  1057999968, -1084397505],
            [-1071760287, -1123489973, -1097310419, -1084649136],
            [-1101533110,  1073668768, -1082790149, -1088634448]],
        dtype=torch.int32)
    >>> y[0, 0] = 1000000000
    >>> x
    tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
            [-0.1520,  0.7472,  0.5617, -0.8649],
            [-2.4724, -0.0334, -0.2976, -0.8499],
            [-0.2109,  1.9913, -0.9607, -0.6123]])

    >>> x.view(torch.cfloat)
    tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
            [-0.1520+0.7472j,  0.5617-0.8649j],
            [-2.4724-0.0334j, -0.2976-0.8499j],
            [-0.2109+1.9913j, -0.9607-0.6123j]])
    >>> x.view(torch.cfloat).size()
    torch.Size([4, 2])

    >>> x.view(torch.uint8)
    tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
               8, 191],
            [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
              93, 191],
            [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
              89, 191],
            [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
              28, 191]], dtype=torch.uint8)
    >>> x.view(torch.uint8).size()
    torch.Size([4, 16])
""",
)

add_docstr_all(
    "view_as",
    r"""
view_as(other) -> Tensor

View this tensor as the same size as :attr:`other`.
``self.view_as(other)`` is equivalent to ``self.view(other.size())``.

Please see :meth:`~Tensor.view` for more information about ``view``.

Args:
    other (:class:`torch.Tensor`): The result tensor has the same size
        as :attr:`other`.
""",
)

add_docstr_all(
    "expand",
    r"""
expand(*sizes) -> Tensor

Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
to a larger size.

Passing -1 as the size for a dimension means not changing the size of
that dimension.

Tensor can be also expanded to a larger number of dimensions, and the
new ones will be appended at the front. For the new dimensions, the
size cannot be set to -1.

Expanding a tensor does not allocate new memory, but only creates a
new view on the existing tensor where a dimension of size one is
expanded to a larger size by setting the ``stride`` to 0. Any dimension
of size 1 can be expanded to an arbitrary value without allocating new
memory.

Args:
    *sizes (torch.Size or int...): the desired expanded size

.. warning::

    More than one element of an expanded tensor may refer to a single
    memory location. As a result, in-place operations (especially ones that
    are vectorized) may result in incorrect behavior. If you need to write
    to the tensors, please clone them first.

Example::

    >>> x = torch.tensor([[1], [2], [3]])
    >>> x.size()
    torch.Size([3, 1])
    >>> x.expand(3, 4)
    tensor([[ 1,  1,  1,  1],
            [ 2,  2,  2,  2],
            [ 3,  3,  3,  3]])
    >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
    tensor([[ 1,  1,  1,  1],
            [ 2,  2,  2,  2],
            [ 3,  3,  3,  3]])
""",
)

add_docstr_all(
    "expand_as",
    r"""
expand_as(other) -> Tensor

Expand this tensor to the same size as :attr:`other`.
``self.expand_as(other)`` is equivalent to ``self.expand(other.size())``.

Please see :meth:`~Tensor.expand` for more information about ``expand``.

Args:
    other (:class:`torch.Tensor`): The result tensor has the same size
        as :attr:`other`.
""",
)

add_docstr_all(
    "sum_to_size",
    r"""
sum_to_size(*size) -> Tensor

Sum ``this`` tensor to :attr:`size`.
:attr:`size` must be broadcastable to ``this`` tensor size.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
""",
)


add_docstr_all(
    "zero_",
    r"""
zero_() -> Tensor

Fills :attr:`self` tensor with zeros.
""",
)

add_docstr_all(
    "matmul",
    r"""
matmul(tensor2) -> Tensor

See :func:`torch.matmul`
""",
)

add_docstr_all(
    "chunk",
    r"""
chunk(chunks, dim=0) -> List of Tensors

See :func:`torch.chunk`
""",
)

add_docstr_all(
    "unsafe_chunk",
    r"""
unsafe_chunk(chunks, dim=0) -> List of Tensors

See :func:`torch.unsafe_chunk`
""",
)

add_docstr_all(
    "unsafe_split",
    r"""
unsafe_split(split_size, dim=0) -> List of Tensors

See :func:`torch.unsafe_split`
""",
)

add_docstr_all(
    "tensor_split",
    r"""
tensor_split(indices_or_sections, dim=0) -> List of Tensors

See :func:`torch.tensor_split`
""",
)

add_docstr_all(
    "hsplit",
    r"""
hsplit(split_size_or_sections) -> List of Tensors

See :func:`torch.hsplit`
""",
)

add_docstr_all(
    "vsplit",
    r"""
vsplit(split_size_or_sections) -> List of Tensors

See :func:`torch.vsplit`
""",
)

add_docstr_all(
    "dsplit",
    r"""
dsplit(split_size_or_sections) -> List of Tensors

See :func:`torch.dsplit`
""",
)

add_docstr_all(
    "stft",
    r"""
stft(frame_length, hop, fft_size=None, return_onesided=True, window=None,
 pad_end=0, align_to_window=None) -> Tensor

See :func:`torch.stft`
""",
)

add_docstr_all(
    "istft",
    r"""
istft(n_fft, hop_length=None, win_length=None, window=None,
 center=True, normalized=False, onesided=True, length=None) -> Tensor

See :func:`torch.istft`
""",
)

add_docstr_all(
    "det",
    r"""
det() -> Tensor

See :func:`torch.det`
""",
)

add_docstr_all(
    "where",
    r"""
where(condition, y) -> Tensor

``self.where(condition, y)`` is equivalent to ``torch.where(condition, self, y)``.
See :func:`torch.where`
""",
)

add_docstr_all(
    "logdet",
    r"""
logdet() -> Tensor

See :func:`torch.logdet`
""",
)

add_docstr_all(
    "slogdet",
    r"""
slogdet() -> (Tensor, Tensor)

See :func:`torch.slogdet`
""",
)

add_docstr_all(
    "unbind",
    r"""
unbind(dim=0) -> seq

See :func:`torch.unbind`
""",
)

add_docstr_all(
    "pin_memory",
    r"""
pin_memory() -> Tensor

Copies the tensor to pinned memory, if it's not already pinned.
By default, the device pinned memory on will be the current :ref:`accelerator<accelerators>`.
""",
)

add_docstr_all(
    "pinverse",
    r"""
pinverse() -> Tensor

See :func:`torch.pinverse`
""",
)

add_docstr_all(
    "index_add",
    r"""
index_add(dim, index, source, *, alpha=1) -> Tensor

Out-of-place version of :meth:`torch.Tensor.index_add_`.
""",
)

add_docstr_all(
    "index_copy",
    r"""
index_copy(dim, index, tensor2) -> Tensor

Out-of-place version of :meth:`torch.Tensor.index_copy_`.
""",
)

add_docstr_all(
    "index_fill",
    r"""
index_fill(dim, index, value) -> Tensor

Out-of-place version of :meth:`torch.Tensor.index_fill_`.
""",
)

add_docstr_all(
    "scatter",
    r"""
scatter(dim, index, src) -> Tensor

Out-of-place version of :meth:`torch.Tensor.scatter_`
""",
)

add_docstr_all(
    "scatter_add",
    r"""
scatter_add(dim, index, src) -> Tensor

Out-of-place version of :meth:`torch.Tensor.scatter_add_`
""",
)

add_docstr_all(
    "scatter_reduce",
    r"""
scatter_reduce(dim, index, src, reduce, *, include_self=True) -> Tensor

Out-of-place version of :meth:`torch.Tensor.scatter_reduce_`
""",
)

add_docstr_all(
    "masked_scatter",
    r"""
masked_scatter(mask, tensor) -> Tensor

Out-of-place version of :meth:`torch.Tensor.masked_scatter_`

.. note::

    The inputs :attr:`self` and :attr:`mask`
    :ref:`broadcast <broadcasting-semantics>`.

Example:

    >>> self = torch.tensor([0, 0, 0, 0, 0])
    >>> mask = torch.tensor(
    ...     [[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]],
    ...     dtype=torch.bool,
    ... )
    >>> source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    >>> self.masked_scatter(mask, source)
    tensor([[0, 0, 0, 0, 1],
            [2, 3, 0, 4, 5]])

""",
)

add_docstr_all(
    "xlogy",
    r"""
xlogy(other) -> Tensor

See :func:`torch.xlogy`
""",
)

add_docstr_all(
    "xlogy_",
    r"""
xlogy_(other) -> Tensor

In-place version of :meth:`~Tensor.xlogy`
""",
)

add_docstr_all(
    "masked_fill",
    r"""
masked_fill(mask, value) -> Tensor

Out-of-place version of :meth:`torch.Tensor.masked_fill_`
""",
)

add_docstr_all(
    "grad",
    r"""
This attribute is ``None`` by default and becomes a Tensor the first time a call to
:func:`backward` computes gradients for ``self``.
The attribute will then contain the gradients computed and future calls to
:func:`backward` will accumulate (add) gradients into it.
""",
)

add_docstr_all(
    "retain_grad",
    r"""
retain_grad() -> None

Enables this Tensor to have their :attr:`grad` populated during
:func:`backward`. This is a no-op for leaf tensors.
""",
)

add_docstr_all(
    "retains_grad",
    r"""
Is ``True`` if this Tensor is non-leaf and its :attr:`grad` is enabled to be
populated during :func:`backward`, ``False`` otherwise.
""",
)

add_docstr_all(
    "requires_grad",
    r"""
Is ``True`` if gradients need to be computed for this Tensor, ``False`` otherwise.

.. note::

    The fact that gradients need to be computed for a Tensor do not mean that the :attr:`grad`
    attribute will be populated, see :attr:`is_leaf` for more details.

""",
)

add_docstr_all(
    "is_leaf",
    r"""
All Tensors that have :attr:`requires_grad` which is ``False`` will be leaf Tensors by convention.

For Tensors that have :attr:`requires_grad` which is ``True``, they will be leaf Tensors if they were
created by the user. This means that they are not the result of an operation and so
:attr:`grad_fn` is None.

Only leaf Tensors will have their :attr:`grad` populated during a call to :func:`backward`.
To get :attr:`grad` populated for non-leaf Tensors, you can use :func:`retain_grad`.

Example::

    >>> a = torch.rand(10, requires_grad=True)
    >>> a.is_leaf
    True
    >>> b = torch.rand(10, requires_grad=True).cuda()
    >>> b.is_leaf
    False
    # b was created by the operation that cast a cpu Tensor into a cuda Tensor
    >>> c = torch.rand(10, requires_grad=True) + 2
    >>> c.is_leaf
    False
    # c was created by the addition operation
    >>> d = torch.rand(10).cuda()
    >>> d.is_leaf
    True
    # d does not require gradients and so has no operation creating it (that is tracked by the autograd engine)
    >>> e = torch.rand(10).cuda().requires_grad_()
    >>> e.is_leaf
    True
    # e requires gradients and has no operations creating it
    >>> f = torch.rand(10, requires_grad=True, device="cuda")
    >>> f.is_leaf
    True
    # f requires grad, has no operation creating it


""",
)

add_docstr_all(
    "names",
    r"""
Stores names for each of this tensor's dimensions.

``names[idx]`` corresponds to the name of tensor dimension ``idx``.
Names are either a string if the dimension is named or ``None`` if the
dimension is unnamed.

Dimension names may contain characters or underscore. Furthermore, a dimension
name must be a valid Python variable name (i.e., does not start with underscore).

Tensors may not have two named dimensions with the same name.

.. warning::
    The named tensor API is experimental and subject to change.

""",
)

add_docstr_all(
    "is_cuda",
    r"""
Is ``True`` if the Tensor is stored on the GPU, ``False`` otherwise.
""",
)

add_docstr_all(
    "is_cpu",
    r"""
Is ``True`` if the Tensor is stored on the CPU, ``False`` otherwise.
""",
)

add_docstr_all(
    "is_xla",
    r"""
Is ``True`` if the Tensor is stored on an XLA device, ``False`` otherwise.
""",
)

add_docstr_all(
    "is_ipu",
    r"""
Is ``True`` if the Tensor is stored on the IPU, ``False`` otherwise.
""",
)

add_docstr_all(
    "is_xpu",
    r"""
Is ``True`` if the Tensor is stored on the XPU, ``False`` otherwise.
""",
)

add_docstr_all(
    "is_quantized",
    r"""
Is ``True`` if the Tensor is quantized, ``False`` otherwise.
""",
)

add_docstr_all(
    "is_meta",
    r"""
Is ``True`` if the Tensor is a meta tensor, ``False`` otherwise.  Meta tensors
are like normal tensors, but they carry no data.
""",
)

add_docstr_all(
    "is_mps",
    r"""
Is ``True`` if the Tensor is stored on the MPS device, ``False`` otherwise.
""",
)

add_docstr_all(
    "is_sparse",
    r"""
Is ``True`` if the Tensor uses sparse COO storage layout, ``False`` otherwise.
""",
)

add_docstr_all(
    "is_sparse_csr",
    r"""
Is ``True`` if the Tensor uses sparse CSR storage layout, ``False`` otherwise.
""",
)

add_docstr_all(
    "device",
    r"""
Is the :class:`torch.device` where this Tensor is.
""",
)

add_docstr_all(
    "ndim",
    r"""
Alias for :meth:`~Tensor.dim()`
""",
)

add_docstr_all(
    "itemsize",
    r"""
Alias for :meth:`~Tensor.element_size()`
""",
)

add_docstr_all(
    "nbytes",
    r"""
Returns the number of bytes consumed by the "view" of elements of the Tensor
if the Tensor does not use sparse storage layout.
Defined to be :meth:`~Tensor.numel()` * :meth:`~Tensor.element_size()`
""",
)

add_docstr_all(
    "T",
    r"""
Returns a view of this tensor with its dimensions reversed.

If ``n`` is the number of dimensions in ``x``,
``x.T`` is equivalent to ``x.permute(n-1, n-2, ..., 0)``.

.. warning::
    The use of :func:`Tensor.T` on tensors of dimension other than 2 to reverse their shape
    is deprecated and it will throw an error in a future release. Consider :attr:`~.Tensor.mT`
    to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse
    the dimensions of a tensor.
""",
)

add_docstr_all(
    "H",
    r"""
Returns a view of a matrix (2-D tensor) conjugated and transposed.

``x.H`` is equivalent to ``x.transpose(0, 1).conj()`` for complex matrices and
``x.transpose(0, 1)`` for real matrices.

.. seealso::

        :attr:`~.Tensor.mH`: An attribute that also works on batches of matrices.
""",
)

add_docstr_all(
    "mT",
    r"""
Returns a view of this tensor with the last two dimensions transposed.

``x.mT`` is equivalent to ``x.transpose(-2, -1)``.
""",
)

add_docstr_all(
    "mH",
    r"""
Accessing this property is equivalent to calling :func:`adjoint`.
""",
)

add_docstr_all(
    "adjoint",
    r"""
adjoint() -> Tensor

Alias for :func:`adjoint`
""",
)

add_docstr_all(
    "real",
    r"""
Returns a new tensor containing real values of the :attr:`self` tensor for a complex-valued input tensor.
The returned tensor and :attr:`self` share the same underlying storage.

Returns :attr:`self` if :attr:`self` is a real-valued tensor tensor.

Example::

    >>> x=torch.randn(4, dtype=torch.cfloat)
    >>> x
    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
    >>> x.real
    tensor([ 0.3100, -0.5445, -1.6492, -0.0638])

""",
)

add_docstr_all(
    "imag",
    r"""
Returns a new tensor containing imaginary values of the :attr:`self` tensor.
The returned tensor and :attr:`self` share the same underlying storage.

.. warning::
    :func:`imag` is only supported for tensors with complex dtypes.

Example::

    >>> x=torch.randn(4, dtype=torch.cfloat)
    >>> x
    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])
    >>> x.imag
    tensor([ 0.3553, -0.7896, -0.0633, -0.8119])

""",
)

add_docstr_all(
    "as_subclass",
    r"""
as_subclass(cls) -> Tensor

Makes a ``cls`` instance with the same data pointer as ``self``. Changes
in the output mirror changes in ``self``, and the output stays attached
to the autograd graph. ``cls`` must be a subclass of ``Tensor``.
""",
)

add_docstr_all(
    "crow_indices",
    r"""
crow_indices() -> IntTensor

Returns the tensor containing the compressed row indices of the :attr:`self`
tensor when :attr:`self` is a sparse CSR tensor of layout ``sparse_csr``.
The ``crow_indices`` tensor is strictly of shape (:attr:`self`.size(0) + 1)
and of type ``int32`` or ``int64``. When using MKL routines such as sparse
matrix multiplication, it is necessary to use ``int32`` indexing in order
to avoid downcasting and potentially losing information.

Example::

    >>> csr = torch.eye(5,5).to_sparse_csr()
    >>> csr.crow_indices()
    tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)

""",
)

add_docstr_all(
    "col_indices",
    r"""
col_indices() -> IntTensor

Returns the tensor containing the column indices of the :attr:`self`
tensor when :attr:`self` is a sparse CSR tensor of layout ``sparse_csr``.
The ``col_indices`` tensor is strictly of shape (:attr:`self`.nnz())
and of type ``int32`` or ``int64``.  When using MKL routines such as sparse
matrix multiplication, it is necessary to use ``int32`` indexing in order
to avoid downcasting and potentially losing information.

Example::

    >>> csr = torch.eye(5,5).to_sparse_csr()
    >>> csr.col_indices()
    tensor([0, 1, 2, 3, 4], dtype=torch.int32)

""",
)

add_docstr_all(
    "to_padded_tensor",
    r"""
to_padded_tensor(padding, output_size=None) -> Tensor
See :func:`to_padded_tensor`
""",
)
