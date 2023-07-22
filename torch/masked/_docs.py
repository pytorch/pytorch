# This file is generated, do not modify it!
#
# To update this file, run the update masked docs script as follows:
#
#   python tools/update_masked_docs.py
#
# The script must be called from an environment where the development
# version of torch package can be imported and is functional.
#

amax_docstring = """amax(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns maximum of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of maximum operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``-inf``, ``0``, and ``-2147483648``, respectively.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in maximum computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of maximum operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.amax(input, 1, mask=mask)
    tensor([                  -1, -9223372036854775808])
"""

amin_docstring = """amin(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns minimum of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of minimum operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``inf``, ``255``, and ``2147483647``, respectively.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in minimum computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of minimum operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.amin(input, 1, mask=mask)
    tensor([                 -3, 9223372036854775807])
"""

argmax_docstring = """argmax(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns argmax of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.
The identity value of argmax operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``-inf``, ``0``, and ``-2147483648``, respectively.
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in argmax computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of argmax operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which argmax is computed.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.
Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.argmax(input, 1, mask=mask)
    tensor([2, 0])
"""

argmin_docstring = """argmin(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns argmin of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.
The identity value of argmin operation, which is used to start the
reduction, depends on input dtype. For instance, for float32, uint8,
and int32 dtypes, the identity values are ``inf``, ``255``, and ``2147483647``, respectively.
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in argmin computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of argmin operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which argmin is computed.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.
Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.argmin(input, 1, mask=mask)
    tensor([0, 0])
"""

cumprod_docstring = """cumprod(input, dim, *, dtype=None, mask=None) -> Tensor

Returns cumulative_prod of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Cumsum of i-th element in ``x`` is
defined as ``prod(x[:i])``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
cumulative_prod computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the cumulative_prod output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which cumulative_prod is computed.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.cumprod(input, 1, mask=mask)
    tensor([[-3., -3.,  3.],
            [ 1.,  1.,  1.]])
"""

cumsum_docstring = """cumsum(input, dim, *, dtype=None, mask=None) -> Tensor

Returns cumulative_sum of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Cumsum of i-th element in ``x`` is
defined as ``sum(x[:i])``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
cumulative_sum computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the cumulative_sum output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which cumulative_sum is computed.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.cumsum(input, 1, mask=mask)
    tensor([[-3., -3., -4.],
            [ 0.,  0.,  0.]])
"""

log_softmax_docstring = """log_softmax(input, dim, *, dtype=None, mask=None) -> Tensor

Returns log_softmax of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. LogSoftmax of i-th element in ``x`` is
defined as ``log(exp(x[i])/sum(exp(x)))``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
log_softmax computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the log_softmax output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which log_softmax is computed.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.log_softmax(input, 1, mask=mask)
    tensor([[-2.1269,    -inf, -0.1269],
            [    nan,     nan,     nan]])
"""

logsumexp_docstring = """logsumexp(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns logsumexp of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of logsumexp operation, which is used to start the reduction, is ``-2147483648``.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in logsumexp computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of logsumexp operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.logsumexp(input, 1, mask=mask)
    tensor([                   0, -9223372036854775808])
"""

mean_docstring = """mean(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns mean of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

By definition, the identity value of a mean operation is the mean
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
mean is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in mean computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of mean operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.mean(input, 1, mask=mask)
    tensor([-2., nan])
"""

median_docstring = """median(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns median of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.
By definition, the identity value of a median operation is the median
value of the tensor. If all elements of the input tensor along given
dimension(s) :attr:`dim` are masked-out, the identity value of the
median is undefined.  Due to this ambiguity, the elements of output
tensor with strided layout, that correspond to fully masked-out
elements, have ``nan`` values.
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in median computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of median operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which median is computed.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.
Example::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.median(input, 1, mask=mask)
    tensor([-3., nan])
"""

norm_docstring = """norm(input, ord, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns norm of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of norm operation, which is used to start the
reduction, is ``0.0``, except for ``ord=-inf`` it is
``inf``.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in norm computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of norm operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    ord (int, float, optional): the order of vector norm. Default: 2.
      See :func:`torch.linalg.vector_norm` for a list of supported norms.
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.norm(input, 2.0, 1, mask=mask)
    tensor([3.1623, 0.0000])
"""

normalize_docstring = """normalize(input, ord, dim, *, eps=1e-12, dtype=None, mask=None) -> Tensor

Returns normalize of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Normalize of i-th element in ``x`` is
defined as ``x[i]/max(norm(x, p), eps)``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
normalize computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the normalize output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    ord (int, float): the order of vector norm. Default: 2.
      See :func:`torch.linalg.vector_norm` for a list of supported norms.
    dim (int): the dimension along which normalize is computed.

Keyword args:
    eps (float, optional): small value to avoid division by zero. Default: 1e-12.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.normalize(input, 2.0, 1, mask=mask)
    tensor([[-0.9487,  0.0000, -0.3162],
            [ 0.0000,  0.0000,  0.0000]])
"""

prod_docstring = """prod(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns product of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of product operation, which is used to start the reduction, is ``1``.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in product computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of product operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.prod(input, 1, mask=mask)
    tensor([3, 1])
"""

softmax_docstring = """softmax(input, dim, *, dtype=None, mask=None) -> Tensor

Returns softmax of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Softmax of i-th element in ``x`` is
defined as ``exp(x[i])/sum(exp(x))``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
softmax computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the softmax output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which softmax is computed.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.softmax(input, 1, mask=mask)
    tensor([[0.1192, 0.0000, 0.8808],
            [   nan,    nan,    nan]])
"""

softmin_docstring = """softmin(input, dim, *, dtype=None, mask=None) -> Tensor

Returns softmin of all the slices in the :attr:`input` tensor
along :attr:`dim` while the :attr:`input` elements are masked out
according to the boolean tensor :attr:`mask`.

Let ``x`` be a sequence of unmasked elements of one-dimensional slice
of the :attr:`input` tensor. Softmin of i-th element in ``x`` is
defined as ``exp(-x[i])/sum(exp(-x))``.

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True then
the corresponding element in :attr:`input` tensor will be included in
softmin computation, otherwise the element is ignored.

The values of masked-out elements of the output tensor have undefined
value: it may or may not be set to zero or nan; the choice may correspond to
the value that leads to the most efficient storage of :attr:`output`
tensor.

The mask of the softmin output tensor can be computed as
``torch.broadcast_to(mask, input.shape)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension along which softmin is computed.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3., -2., -1.], [ 0., 1., 2.]])
    >>> input
    tensor([[-3., -2., -1.],
            [ 0.,  1.,  2.]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.softmin(input, 1, mask=mask)
    tensor([[0.8808, 0.0000, 0.1192],
            [   nan,    nan,    nan]])
"""

std_docstring = """std(input, dim, unbiased, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns standard_deviation of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.
The identity value of sample standard deviation operation is undefined. The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in standard_deviation computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of standard_deviation operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.
    unbiased (bool): when True, use Bessel’s correction, otherwise, compute
      the uncorrected sample variance.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.
Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.std(input, 1, False, mask=mask)
    tensor([1., nan])
"""

sum_docstring = """sum(input, dim, *, keepdim=False, dtype=None, mask=None) -> Tensor

Returns sum of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.

The identity value of sum operation, which is used to start the reduction, is ``0``.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in sum computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of sum operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.

Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.sum(input, 1, mask=mask)
    tensor([-4,  0])
"""

var_docstring = """var(input, dim, unbiased, *, keepdim=False, dtype=None, mask=None) -> Tensor
Returns variance of all the elements in the :attr:`input`
tensor along the given dimension(s) :attr:`dim` while the :attr:`input`
elements are masked out according to the boolean tensor
:attr:`mask`.
The identity value of sample variance operation is undefined. The
elements of output tensor with strided layout, that correspond to
fully masked-out elements, have ``nan`` values.
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of
size 1. Otherwise, :attr:`dim` is squeezed (see
:func:`torch.squeeze`), resulting in the output tensor having 1 (or
``len(dim)``) fewer dimension(s).

The boolean tensor :attr:`mask` defines the "validity" of
:attr:`input` tensor elements: if :attr:`mask` element is True
then the corresponding element in :attr:`input` tensor will be
included in variance computation, otherwise the element is
ignored.

When all elements of :attr:`input` along the given dimension
:attr:`dim` are ignored (fully masked-out), the corresponding element
of the output tensor will have undefined value: it may or may not
correspond to the identity value of variance operation; the
choice may correspond to the value that leads to the most efficient
storage of :attr:`output` tensor.

The mask of the output tensor can be computed as
``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,
dtype=torch.bool)``.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor
don't need to match, but they must be :ref:`broadcastable
<broadcasting-semantics>` and the dimensionality of the :attr:`mask`
tensor must not be greater than of the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
      Default: None that is equivalent to ``tuple(range(input.ndim))``.
    unbiased (bool): when True, use Bessel’s correction, otherwise, compute
      the uncorrected sample variance.

Keyword args:
    keepdim (bool, optional): whether the output tensor has
      :attr:`dim` retained or not. Default: False.
    dtype (:class:`torch.dtype`, optional): the desired data type
      of returned tensor.  If specified, the input tensor is
      casted to :attr:`dtype` before the operation is
      performed. Default: None.
    mask (:class:`torch.Tensor`, optional): the boolean tensor
      containing the binary mask of validity of input tensor
      elements.
      Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.
Example::

    >>> input = tensor([[-3, -2, -1], [ 0, 1, 2]])
    >>> input
    tensor([[-3, -2, -1],
            [ 0,  1,  2]])
    >>> mask = tensor([[ True, False, True], [False, False, False]])
    >>> mask
    tensor([[ True, False,  True],
            [False, False, False]])
    >>> torch.masked._ops.var(input, 1, False, mask=mask)
    tensor([1., nan])
"""
