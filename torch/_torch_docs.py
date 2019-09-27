"""Adds docstrings to functions defined in the torch._C"""

import re

import torch._C
from torch._C import _add_docstr as add_docstr


def parse_kwargs(desc):
    """Maps a description of args to a dictionary of {argname: description}.
    Input:
        ('    weight (Tensor): a weight tensor\n' +
         '        Some optional description')
    Output: {
        'weight': \
        'weight (Tensor): a weight tensor\n        Some optional description'
    }
    """
    # Split on exactly 4 spaces after a newline
    regx = re.compile(r"\n\s{4}(?!\s)")
    kwargs = [section.strip() for section in regx.split(desc)]
    kwargs = [section for section in kwargs if len(section) > 0]
    return {desc.split(' ')[0]: desc for desc in kwargs}


def merge_dicts(*dicts):
    return {x: d[x] for d in dicts for x in d}


common_args = parse_kwargs("""
    input (Tensor): the input tensor.
    out (Tensor, optional): the output tensor.
""")

reduceops_common_args = merge_dicts(common_args, parse_kwargs("""
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
"""))

multi_dim_common = merge_dicts(reduceops_common_args, parse_kwargs("""
    dim (int or tuple of ints): the dimension or dimensions to reduce.
"""), {'keepdim_details': """
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).
"""})

single_dim_common = merge_dicts(reduceops_common_args, parse_kwargs("""
    dim (int): the dimension to reduce.
"""), {'keepdim_details': """If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensor having 1 fewer dimension than :attr:`input`."""})


factory_common_args = merge_dicts(common_args, parse_kwargs("""
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
"""))

factory_like_common_args = parse_kwargs("""
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
""")

factory_data_common_args = parse_kwargs("""
    data (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, infers data type from :attr:`data`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
""")

add_docstr(torch.abs,
           r"""
abs(input, out=None) -> Tensor

Computes the element-wise absolute value of the given :attr:`input` tensor.

.. math::
    \text{out}_{i} = |\text{input}_{i}|
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> torch.abs(torch.tensor([-1, -2, 3]))
    tensor([ 1,  2,  3])
""".format(**common_args))

add_docstr(torch.acos,
           r"""
acos(input, out=None) -> Tensor

Returns a new tensor with the arccosine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \cos^{-1}(\text{input}_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.3348, -0.5889,  0.2005, -0.1584])
    >>> torch.acos(a)
    tensor([ 1.2294,  2.2004,  1.3690,  1.7298])
""".format(**common_args))

add_docstr(torch.add,
           r"""
.. function:: add(input, other, out=None)

Adds the scalar :attr:`other` to each element of the input :attr:`input`
and returns a new resulting tensor.

.. math::
    \text{{out}} = \text{{input}} + \text{{other}}

If :attr:`input` is of type FloatTensor or DoubleTensor, :attr:`other` must be
a real number, otherwise it should be an integer.

Args:
    {input}
    value (Number): the number to be added to each element of :attr:`input`

Keyword arguments:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
    >>> torch.add(a, 20)
    tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

.. function:: add(input, alpha=1, other, out=None)

Each element of the tensor :attr:`other` is multiplied by the scalar
:attr:`alpha` and added to each element of the tensor :attr:`input`.
The resulting tensor is returned.

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

.. math::
    \text{{out}} = \text{{input}} + \text{{alpha}} \times \text{{other}}

If :attr:`other` is of type FloatTensor or DoubleTensor, :attr:`alpha` must be
a real number, otherwise it should be an integer.

Args:
    input (Tensor): the first input tensor
    alpha (Number): the scalar multiplier for :attr:`other`
    other (Tensor): the second input tensor

Keyword arguments:
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.9732, -0.3497,  0.6245,  0.4022])
    >>> b = torch.randn(4, 1)
    >>> b
    tensor([[ 0.3743],
            [-1.7724],
            [-0.5811],
            [-0.8017]])
    >>> torch.add(a, 10, b)
    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
            [-18.6971, -18.0736, -17.0994, -17.3216],
            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
""".format(**common_args))

add_docstr(torch.addbmm,
           r"""
addbmm(beta=1, input, alpha=1, batch1, batch2, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored
in :attr:`batch1` and :attr:`batch2`,
with a reduced add step (all matrix multiplications get accumulated
along the first dimension).
:attr:`input` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the
same number of matrices.

If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
:math:`(b \times m \times p)` tensor, :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

.. math::
    out = \beta\ \text{input} + \alpha\ (\sum_{i=0}^{b-1} \text{batch1}_i \mathbin{@} \text{batch2}_i)
""" + r"""
For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and :attr:`alpha`
must be real numbers, otherwise they should be integers.

Args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    input (Tensor): matrix to be added
    alpha (Number, optional): multiplier for `batch1 @ batch2` (:math:`\alpha`)
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied
    {out}

Example::

    >>> M = torch.randn(3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.addbmm(M, batch1, batch2)
    tensor([[  6.6311,   0.0503,   6.9768, -12.0362,  -2.1653],
            [ -4.8185,  -1.4255,  -6.6760,   8.9453,   2.5743],
            [ -3.8202,   4.3691,   1.0943,  -1.1109,   5.4730]])
""".format(**common_args))

add_docstr(torch.addcdiv,
           r"""
addcdiv(input, value=1, tensor1, tensor2, out=None) -> Tensor

Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
multiply the result by the scalar :attr:`value` and add it to :attr:`input`.

.. math::
    \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}
""" + r"""
The shapes of :attr:`input`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    input (Tensor): the tensor to be added
    value (Number, optional): multiplier for :math:`\text{{tensor1}} / \text{{tensor2}}`
    tensor1 (Tensor): the numerator tensor
    tensor2 (Tensor): the denominator tensor
    {out}

Example::

    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcdiv(t, 0.1, t1, t2)
    tensor([[-0.2312, -3.6496,  0.1312],
            [-1.0428,  3.4292, -0.1030],
            [-0.5369, -0.9829,  0.0430]])
""".format(**common_args))

add_docstr(torch.addcmul,
           r"""
addcmul(input, value=1, tensor1, tensor2, out=None) -> Tensor

Performs the element-wise multiplication of :attr:`tensor1`
by :attr:`tensor2`, multiply the result by the scalar :attr:`value`
and add it to :attr:`input`.

.. math::
    \text{out}_i = \text{input}_i + \text{value} \times \text{tensor1}_i \times \text{tensor2}_i
""" + r"""
The shapes of :attr:`tensor`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    input (Tensor): the tensor to be added
    value (Number, optional): multiplier for :math:`tensor1 .* tensor2`
    tensor1 (Tensor): the tensor to be multiplied
    tensor2 (Tensor): the tensor to be multiplied
    {out}

Example::

    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcmul(t, 0.1, t1, t2)
    tensor([[-0.8635, -0.6391,  1.6174],
            [-0.7617, -0.5879,  1.7388],
            [-0.8353, -0.6249,  1.6511]])
""".format(**common_args))

add_docstr(torch.addmm,
           r"""
addmm(beta=1, input, alpha=1, mat1, mat2, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
The matrix :attr:`input` is added to the final result.

If :attr:`mat1` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
:math:`(m \times p)` tensor, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat1` and :attr:`mat2` and the added matrix :attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)
""" + r"""
For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

Args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    input (Tensor): matrix to be added
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    mat1 (Tensor): the first matrix to be multiplied
    mat2 (Tensor): the second matrix to be multiplied
    {out}

Example::

    >>> M = torch.randn(2, 3)
    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.addmm(M, mat1, mat2)
    tensor([[-4.8716,  1.4671, -1.3746],
            [ 0.7573, -3.9555, -2.8681]])
""".format(**common_args))

add_docstr(torch.addmv,
           r"""
addmv(beta=1, input, alpha=1, mat, vec, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`mat` and
the vector :attr:`vec`.
The vector :attr:`input` is added to the final result.

If :attr:`mat` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
size `m`, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a 1-D tensor of size `n` and
:attr:`out` will be 1-D tensor of size `n`.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat` and :attr:`vec` and the added tensor :attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{mat} \mathbin{@} \text{vec})
""" + r"""
For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers

Args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    input (Tensor): vector to be added
    alpha (Number, optional): multiplier for :math:`mat @ vec` (:math:`\alpha`)
    mat (Tensor): matrix to be multiplied
    vec (Tensor): vector to be multiplied
    {out}

Example::

    >>> M = torch.randn(2)
    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.addmv(M, mat, vec)
    tensor([-0.3768, -5.5565])
""".format(**common_args))

add_docstr(torch.addr,
           r"""
addr(beta=1, input, alpha=1, vec1, vec2, out=None) -> Tensor

Performs the outer-product of vectors :attr:`vec1` and :attr:`vec2`
and adds it to the matrix :attr:`input`.

Optional values :attr:`beta` and :attr:`alpha` are scaling factors on the
outer product between :attr:`vec1` and :attr:`vec2` and the added matrix
:attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{vec1} \otimes \text{vec2})
""" + r"""
If :attr:`vec1` is a vector of size `n` and :attr:`vec2` is a vector
of size `m`, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a matrix of size
:math:`(n \times m)` and :attr:`out` will be a matrix of size
:math:`(n \times m)`.

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers

Args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    input (Tensor): matrix to be added
    alpha (Number, optional): multiplier for :math:`\text{{vec1}} \otimes \text{{vec2}}` (:math:`\alpha`)
    vec1 (Tensor): the first vector of the outer product
    vec2 (Tensor): the second vector of the outer product
    {out}

Example::

    >>> vec1 = torch.arange(1., 4.)
    >>> vec2 = torch.arange(1., 3.)
    >>> M = torch.zeros(3, 2)
    >>> torch.addr(M, vec1, vec2)
    tensor([[ 1.,  2.],
            [ 2.,  4.],
            [ 3.,  6.]])
""".format(**common_args))

add_docstr(torch.allclose,
           r"""
allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool

This function checks if all :attr:`input` and :attr:`other` satisfy the condition:

.. math::
    \lvert \text{input} - \text{other} \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert \text{other} \rvert
""" + r"""
elementwise, for all elements of :attr:`input` and :attr:`other`. The behaviour of this function is analogous to
`numpy.allclose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html>`_

Args:
    input (Tensor): first tensor to compare
    other (Tensor): second tensor to compare
    atol (float, optional): absolute tolerance. Default: 1e-08
    rtol (float, optional): relative tolerance. Default: 1e-05
    equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be compared as equal. Default: ``False``

Example::

    >>> torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))
    False
    >>> torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))
    True
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))
    False
    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)
    True
""")

add_docstr(torch.as_strided,
           r"""
as_strided(input, size, stride, storage_offset=0) -> Tensor

Create a view of an existing `torch.Tensor` :attr:`input` with specified
:attr:`size`, :attr:`stride` and :attr:`storage_offset`.

.. warning::
    More than one element of a created tensor may refer to a single memory
    location. As a result, in-place operations (especially ones that are
    vectorized) may result in incorrect behavior. If you need to write to
    the tensors, please clone them first.

    Many PyTorch functions, which return a view of a tensor, are internally
    implemented with this function. Those functions, like
    :meth:`torch.Tensor.expand`, are easier to read and are therefore more
    advisable to use.


Args:
    {input}
    size (tuple or ints): the shape of the output tensor
    stride (tuple or ints): the stride of the output tensor
    storage_offset (int, optional): the offset in the underlying storage of the output tensor

Example::

    >>> x = torch.randn(3, 3)
    >>> x
    tensor([[ 0.9039,  0.6291,  1.0795],
            [ 0.1586,  2.1939, -0.4900],
            [-0.1909, -0.7503,  1.9355]])
    >>> t = torch.as_strided(x, (2, 2), (1, 2))
    >>> t
    tensor([[0.9039, 1.0795],
            [0.6291, 0.1586]])
    >>> t = torch.as_strided(x, (2, 2), (1, 2), 1)
    tensor([[0.6291, 0.1586],
            [1.0795, 2.1939]])
""".format(**common_args))

add_docstr(torch.as_tensor,
           r"""
as_tensor(data, dtype=None, device=None) -> Tensor

Convert the data into a `torch.Tensor`. If the data is already a `Tensor` with the same `dtype` and `device`,
no copy will be performed, otherwise a new `Tensor` will be returned with computational graph retained if data
`Tensor` has ``requires_grad=True``. Similarly, if the data is an ``ndarray`` of the corresponding `dtype` and
the `device` is the cpu, no copy will be performed.

Args:
    {data}
    {dtype}
    {device}

Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a, device=torch.device('cuda'))
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([1,  2,  3])
""".format(**factory_data_common_args))

add_docstr(torch.asin,
           r"""
asin(input, out=None) -> Tensor

Returns a new tensor with the arcsine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin^{-1}(\text{input}_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5962,  1.4985, -0.4396,  1.4525])
    >>> torch.asin(a)
    tensor([-0.6387,     nan, -0.4552,     nan])
""".format(**common_args))

add_docstr(torch.atan,
           r"""
atan(input, out=None) -> Tensor

Returns a new tensor with the arctangent  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \tan^{-1}(\text{input}_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.2341,  0.2539, -0.6256, -0.6448])
    >>> torch.atan(a)
    tensor([ 0.2299,  0.2487, -0.5591, -0.5727])
""".format(**common_args))

add_docstr(torch.atan2,
           r"""
atan2(input, other, out=None) -> Tensor

Element-wise arctangent of :math:`\text{{input}}_{{i}} / \text{{other}}_{{i}}`
with consideration of the quadrant. Returns a new tensor with the signed angles
in radians between vector :math:`(\text{{other}}_{{i}}, \text{{input}}_{{i}})`
and vector :math:`(1, 0)`. (Note that :math:`\text{{other}}_{{i}}`, the second
parameter, is the x-coordinate, while :math:`\text{{input}}_{{i}}`, the first
parameter, is the y-coordinate.)

The shapes of ``input`` and ``other`` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the first input tensor
    other (Tensor): the second input tensor
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.9041,  0.0196, -0.3108, -2.4423])
    >>> torch.atan2(a, torch.randn(4))
    tensor([ 0.9833,  0.0811, -1.9743, -1.4151])
""".format(**common_args))

add_docstr(torch.baddbmm,
           r"""
baddbmm(beta=1, input, alpha=1, batch1, batch2, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices in :attr:`batch1`
and :attr:`batch2`.
:attr:`input` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the same
number of matrices.

If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
:math:`(b \times m \times p)` tensor, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a
:math:`(b \times n \times p)` tensor and :attr:`out` will be a
:math:`(b \times n \times p)` tensor. Both :attr:`alpha` and :attr:`beta` mean the
same as the scaling factors used in :meth:`torch.addbmm`.

.. math::
    \text{out}_i = \beta\ \text{input}_i + \alpha\ (\text{batch1}_i \mathbin{@} \text{batch2}_i)
""" + r"""
For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

Args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    input (Tensor): the tensor to be added
    alpha (Number, optional): multiplier for :math:`\text{{batch1}} \mathbin{{@}} \text{{batch2}}` (:math:`\alpha`)
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied
    {out}

Example::

    >>> M = torch.randn(10, 3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.baddbmm(M, batch1, batch2).size()
    torch.Size([10, 3, 5])
""".format(**common_args))

add_docstr(torch.bernoulli,
           r"""
bernoulli(input, *, generator=None, out=None) -> Tensor

Draws binary random numbers (0 or 1) from a Bernoulli distribution.

The :attr:`input` tensor should be a tensor containing probabilities
to be used for drawing the binary random number.
Hence, all values in :attr:`input` have to be in the range:
:math:`0 \leq \text{input}_i \leq 1`.

The :math:`\text{i}^{th}` element of the output tensor will draw a
value :math:`1` according to the :math:`\text{i}^{th}` probability value given
in :attr:`input`.

.. math::
    \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})
""" + r"""
The returned :attr:`out` tensor only has values 0 or 1 and is of the same
shape as :attr:`input`.

:attr:`out` can have integral ``dtype``, but :attr:`input` must have floating
point ``dtype``.

Args:
    input (Tensor): the input tensor of probability values for the Bernoulli distribution
    {out}

Example::

    >>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
    >>> a
    tensor([[ 0.1737,  0.0950,  0.3609],
            [ 0.7148,  0.0289,  0.2676],
            [ 0.9456,  0.8937,  0.7202]])
    >>> torch.bernoulli(a)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 1.,  1.,  1.]])

    >>> a = torch.ones(3, 3) # probability of drawing "1" is 1
    >>> torch.bernoulli(a)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
    >>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
    >>> torch.bernoulli(a)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
""".format(**common_args))

add_docstr(torch.bincount,
           r"""
bincount(input, weights=None, minlength=0) -> Tensor

Count the frequency of each value in an array of non-negative ints.

The number of bins (size 1) is one larger than the largest value in
:attr:`input` unless :attr:`input` is empty, in which case the result is a
tensor of size 0. If :attr:`minlength` is specified, the number of bins is at least
:attr:`minlength` and if :attr:`input` is empty, then the result is tensor of size
:attr:`minlength` filled with zeros. If ``n`` is the value at position ``i``,
``out[n] += weights[i]`` if :attr:`weights` is specified else
``out[n] += 1``.

.. include:: cuda_deterministic.rst

Arguments:
    input (Tensor): 1-d int tensor
    weights (Tensor): optional, weight for each value in the input tensor.
        Should be of same size as input tensor.
    minlength (int): optional, minimum number of bins. Should be non-negative.

Returns:
    output (Tensor): a tensor of shape ``Size([max(input) + 1])`` if
    :attr:`input` is non-empty, else ``Size(0)``

Example::

    >>> input = torch.randint(0, 8, (5,), dtype=torch.int64)
    >>> weights = torch.linspace(0, 1, steps=5)
    >>> input, weights
    (tensor([4, 3, 6, 3, 4]),
     tensor([ 0.0000,  0.2500,  0.5000,  0.7500,  1.0000])

    >>> torch.bincount(input)
    tensor([0, 0, 0, 2, 2, 0, 1])

    >>> input.bincount(weights)
    tensor([0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.5000])
""")

add_docstr(torch.bitwise_not,
           r"""
bitwise_not(input, out=None) -> Tensor

Computes the bitwise NOT of the given input tensor. The input tensor must be of
integral or Boolean types. For bool tensors, it computes the logical NOT.

Args:
    {input}
    {out}

Example:

    >>> torch.bitwise_not(torch.tensor([-1, -2, 3], dtype=torch.int8))
    tensor([ 0,  1, -4], dtype=torch.int8)
""".format(**common_args))

add_docstr(torch.bmm,
           r"""
bmm(input, mat2, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored in :attr:`input`
and :attr:`mat2`.

:attr:`input` and :attr:`mat2` must be 3-D tensors each containing
the same number of matrices.

If :attr:`input` is a :math:`(b \times n \times m)` tensor, :attr:`mat2` is a
:math:`(b \times m \times p)` tensor, :attr:`out` will be a
:math:`(b \times n \times p)` tensor.

.. math::
    \text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i
""" + r"""
.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Args:
    input (Tensor): the first batch of matrices to be multiplied
    mat2 (Tensor): the second batch of matrices to be multiplied
    {out}

Example::

    >>> input = torch.randn(10, 3, 4)
    >>> mat2 = torch.randn(10, 4, 5)
    >>> res = torch.bmm(input, mat2)
    >>> res.size()
    torch.Size([10, 3, 5])
""".format(**common_args))

add_docstr(torch.stack,
           r"""
stack(tensors, dim=0, out=None) -> Tensor

Concatenates sequence of tensors along a new dimension.

All tensors need to be of the same size.

Arguments:
    tensors (sequence of Tensors): sequence of tensors to concatenate
    dim (int): dimension to insert. Has to be between 0 and the number
        of dimensions of concatenated tensors (inclusive)
    {out}
""".format(**common_args))

add_docstr(torch.chunk,
           r"""
chunk(input, chunks, dim=0) -> List of Tensors

Splits a tensor into a specific number of chunks.

Last chunk will be smaller if the tensor size along the given dimension
:attr:`dim` is not divisible by :attr:`chunks`.

Arguments:
    input (Tensor): the tensor to split
    chunks (int): number of chunks to return
    dim (int): dimension along which to split the tensor
""")

add_docstr(torch.cat,
           r"""
cat(tensors, dim=0, out=None) -> Tensor

Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
All tensors must either have the same shape (except in the concatenating
dimension) or be empty.

:func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
and :func:`torch.chunk`.

:func:`torch.cat` can be best understood via examples.

Args:
    tensors (sequence of Tensors): any python sequence of tensors of the same type.
        Non-empty tensors provided must have the same shape, except in the
        cat dimension.
    dim (int, optional): the dimension over which the tensors are concatenated
    {out}

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
             -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
             -0.5790,  0.1497]])
""".format(**common_args))

add_docstr(torch.cdist,
           r"""
cdist(x1, x2, p=2) -> Tensor

Computes the p-norm distance between each pair of the two collections of row vectors.

If x1 has shape :math:`P \times M` and x2 has shape :math:`R \times M` then the 
output will have shape :math:`P \times R`.

This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)` 
if :math:`p \in (0, \infty)`. When :math:`p = 0` it is equivalent to 
`scipy.spatial.distance.cdist(input, 'hamming') * M`. When :math:`p = \infty`, the closest 
scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.

Args:
    x1 (Tensor): input tensor of shape :math:`P \times M`.
    x2 (Tensor): input tensor of shape :math:`R \times M`.
    p: p value for the p-norm distance to calculate between each vector pair
        :math:`\in [0, \infty]`.

Example::

    >>> a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
    >>> a
    tensor([[ 0.9041,  0.0196],
            [-0.3108, -2.4423],
            [-0.4821,  1.0590]])
    >>> b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
    >>> b
    tensor([[-2.1763, -0.4713],
            [-0.6986,  1.3702]])
    >>> torch.cdist(a, b, p=2)
    tensor([[3.1193, 2.0959],
            [2.7138, 3.8322],
            [2.2830, 0.3791]])
""".format(**common_args))

add_docstr(torch.ceil,
           r"""
ceil(input, out=None) -> Tensor

Returns a new tensor with the ceil of the elements of :attr:`input`,
the smallest integer greater than or equal to each element.

.. math::
    \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil = \left\lfloor \text{input}_{i} \right\rfloor + 1
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.6341, -1.4208, -1.0900,  0.5826])
    >>> torch.ceil(a)
    tensor([-0., -1., -1.,  1.])
""".format(**common_args))

add_docstr(torch.reciprocal,
           r"""
reciprocal(input, out=None) -> Tensor

Returns a new tensor with the reciprocal of the elements of :attr:`input`

.. math::
    \text{out}_{i} = \frac{1}{\text{input}_{i}}
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.4595, -2.1219, -1.4314,  0.7298])
    >>> torch.reciprocal(a)
    tensor([-2.1763, -0.4713, -0.6986,  1.3702])
""".format(**common_args))

add_docstr(torch.cholesky, r"""
cholesky(input, upper=False, out=None) -> Tensor

Computes the Cholesky decomposition of a symmetric positive-definite
matrix :math:`A` or for batches of symmetric positive-definite matrices.

If :attr:`upper` is ``True``, the returned matrix ``U`` is upper-triangular, and
the decomposition has the form:

.. math::

  A = U^TU

If :attr:`upper` is ``False``, the returned matrix ``L`` is lower-triangular, and
the decomposition has the form:

.. math::

    A = LL^T

If :attr:`upper` is ``True``, and :math:`A` is a batch of symmetric positive-definite
matrices, then the returned tensor will be composed of upper-triangular Cholesky factors
of each of the individual matrices. Similarly, when :attr:`upper` is ``False``, the returned
tensor will be composed of lower-triangular Cholesky factors of each of the individual
matrices.

Args:
    input (Tensor): the input tensor :math:`A` of size :math:`(*, n, n)` where `*` is zero or more
                batch dimensions consisting of symmetric positive-definite matrices.
    upper (bool, optional): flag that indicates whether to return a
                            upper or lower triangular matrix. Default: ``False``
    out (Tensor, optional): the output matrix

Example::

    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) # make symmetric positive-definite
    >>> l = torch.cholesky(a)
    >>> a
    tensor([[ 2.4112, -0.7486,  1.4551],
            [-0.7486,  1.3544,  0.1294],
            [ 1.4551,  0.1294,  1.6724]])
    >>> l
    tensor([[ 1.5528,  0.0000,  0.0000],
            [-0.4821,  1.0592,  0.0000],
            [ 0.9371,  0.5487,  0.7023]])
    >>> torch.mm(l, l.t())
    tensor([[ 2.4112, -0.7486,  1.4551],
            [-0.7486,  1.3544,  0.1294],
            [ 1.4551,  0.1294,  1.6724]])
    >>> a = torch.randn(3, 2, 2)
    >>> a = torch.matmul(a, a.transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
    >>> l = torch.cholesky(a)
    >>> z = torch.matmul(l, l.transpose(-1, -2))
    >>> torch.max(torch.abs(z - a)) # Max non-zero
    tensor(2.3842e-07)
""")

add_docstr(torch.cholesky_solve, r"""
cholesky_solve(input, input2, upper=False, out=None) -> Tensor

Solves a linear system of equations with a positive semidefinite
matrix to be inverted given its Cholesky factor matrix :math:`u`.

If :attr:`upper` is ``False``, :math:`u` is and lower triangular and `c` is
returned such that:

.. math::
    c = (u u^T)^{{-1}} b

If :attr:`upper` is ``True`` or not provided, :math:`u` is upper triangular
and `c` is returned such that:

.. math::
    c = (u^T u)^{{-1}} b

`torch.cholesky_solve(b, u)` can take in 2D inputs `b, u` or inputs that are
batches of 2D matrices. If the inputs are batches, then returns
batched outputs `c`

.. note::

    The :attr:`out` keyword only supports 2D matrix inputs, that is,
    `b, u` must be 2D matrices.

Args:
    input (Tensor): input matrix :math:`b` of size :math:`(*, m, k)`,
                where :math:`*` is zero or more batch dimensions
    input2 (Tensor): input matrix :math:`u` of size :math:`(*, m, m)`,
                where :math:`*` is zero of more batch dimensions composed of
                upper or lower triangular Cholesky factor
    upper (bool, optional): whether to consider the Cholesky factor as a
                            lower or upper triangular matrix. Default: ``False``.
    out (Tensor, optional): the output tensor for `c`

Example::

    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) # make symmetric positive definite
    >>> u = torch.cholesky(a)
    >>> a
    tensor([[ 0.7747, -1.9549,  1.3086],
            [-1.9549,  6.7546, -5.4114],
            [ 1.3086, -5.4114,  4.8733]])
    >>> b = torch.randn(3, 2)
    >>> b
    tensor([[-0.6355,  0.9891],
            [ 0.1974,  1.4706],
            [-0.4115, -0.6225]])
    >>> torch.cholesky_solve(b, u)
    tensor([[ -8.1625,  19.6097],
            [ -5.8398,  14.2387],
            [ -4.3771,  10.4173]])
    >>> torch.mm(a.inverse(), b)
    tensor([[ -8.1626,  19.6097],
            [ -5.8398,  14.2387],
            [ -4.3771,  10.4173]])
""")

add_docstr(torch.cholesky_inverse, r"""
cholesky_inverse(input, upper=False, out=None) -> Tensor

Computes the inverse of a symmetric positive-definite matrix :math:`A` using its
Cholesky factor :math:`u`: returns matrix ``inv``. The inverse is computed using
LAPACK routines ``dpotri`` and ``spotri`` (and the corresponding MAGMA routines).

If :attr:`upper` is ``False``, :math:`u` is lower triangular
such that the returned tensor is

.. math::
    inv = (uu^{{T}})^{{-1}}

If :attr:`upper` is ``True`` or not provided, :math:`u` is upper
triangular such that the returned tensor is

.. math::
    inv = (u^T u)^{{-1}}

Args:
    input (Tensor): the input 2-D tensor :math:`u`, a upper or lower triangular
           Cholesky factor
    upper (bool, optional): whether to return a lower (default) or upper triangular matrix
    out (Tensor, optional): the output tensor for `inv`

Example::

    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3) # make symmetric positive definite
    >>> u = torch.cholesky(a)
    >>> a
    tensor([[  0.9935,  -0.6353,   1.5806],
            [ -0.6353,   0.8769,  -1.7183],
            [  1.5806,  -1.7183,  10.6618]])
    >>> torch.cholesky_inverse(u)
    tensor([[ 1.9314,  1.2251, -0.0889],
            [ 1.2251,  2.4439,  0.2122],
            [-0.0889,  0.2122,  0.1412]])
    >>> a.inverse()
    tensor([[ 1.9314,  1.2251, -0.0889],
            [ 1.2251,  2.4439,  0.2122],
            [-0.0889,  0.2122,  0.1412]])
""")

add_docstr(torch.clamp,
           r"""
clamp(input, min, max, out=None) -> Tensor

Clamp all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]` and return
a resulting tensor:

.. math::
    y_i = \begin{cases}
        \text{min} & \text{if } x_i < \text{min} \\
        x_i & \text{if } \text{min} \leq x_i \leq \text{max} \\
        \text{max} & \text{if } x_i > \text{max}
    \end{cases}
""" + r"""
If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, args :attr:`min`
and :attr:`max` must be real numbers, otherwise they should be integers.

Args:
    {input}
    min (Number): lower-bound of the range to be clamped to
    max (Number): upper-bound of the range to be clamped to
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-1.7120,  0.1734, -0.0478, -0.0922])
    >>> torch.clamp(a, min=-0.5, max=0.5)
    tensor([-0.5000,  0.1734, -0.0478, -0.0922])

.. function:: clamp(input, *, min, out=None) -> Tensor

Clamps all elements in :attr:`input` to be larger or equal :attr:`min`.

If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`value`
should be a real number, otherwise it should be an integer.

Args:
    {input}
    value (Number): minimal value of each element in the output
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.0299, -2.3184,  2.1593, -0.8883])
    >>> torch.clamp(a, min=0.5)
    tensor([ 0.5000,  0.5000,  2.1593,  0.5000])

.. function:: clamp(input, *, max, out=None) -> Tensor

Clamps all elements in :attr:`input` to be smaller or equal :attr:`max`.

If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`value`
should be a real number, otherwise it should be an integer.

Args:
    {input}
    value (Number): maximal value of each element in the output
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.7753, -0.4702, -0.4599,  1.1899])
    >>> torch.clamp(a, max=0.5)
    tensor([ 0.5000, -0.4702, -0.4599,  0.5000])
""".format(**common_args))

add_docstr(torch.cos,
           r"""
cos(input, out=None) -> Tensor

Returns a new tensor with the cosine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \cos(\text{input}_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
    >>> torch.cos(a)
    tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
""".format(**common_args))

add_docstr(torch.cosh,
           r"""
cosh(input, out=None) -> Tensor

Returns a new tensor with the hyperbolic cosine  of the elements of
:attr:`input`.

.. math::
    \text{out}_{i} = \cosh(\text{input}_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.1632,  1.1835, -0.6979, -0.7325])
    >>> torch.cosh(a)
    tensor([ 1.0133,  1.7860,  1.2536,  1.2805])
""".format(**common_args))

add_docstr(torch.cross,
           r"""
cross(input, other, dim=-1, out=None) -> Tensor


Returns the cross product of vectors in dimension :attr:`dim` of :attr:`input`
and :attr:`other`.

:attr:`input` and :attr:`other` must have the same size, and the size of their
:attr:`dim` dimension should be 3.

If :attr:`dim` is not given, it defaults to the first dimension found with the
size 3.

Args:
    {input}
    other (Tensor): the second input tensor
    dim  (int, optional): the dimension to take the cross-product in.
    {out}

Example::

    >>> a = torch.randn(4, 3)
    >>> a
    tensor([[-0.3956,  1.1455,  1.6895],
            [-0.5849,  1.3672,  0.3599],
            [-1.1626,  0.7180, -0.0521],
            [-0.1339,  0.9902, -2.0225]])
    >>> b = torch.randn(4, 3)
    >>> b
    tensor([[-0.0257, -1.4725, -1.2251],
            [-1.1479, -0.7005, -1.9757],
            [-1.3904,  0.3726, -1.1836],
            [-0.9688, -0.7153,  0.2159]])
    >>> torch.cross(a, b, dim=1)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
    >>> torch.cross(a, b)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
""".format(**common_args))

add_docstr(torch.cumprod,
           r"""
cumprod(input, dim, out=None, dtype=None) -> Tensor

Returns the cumulative product of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be
a vector of size N, with elements.

.. math::
    y_i = x_1 \times x_2\times x_3\times \dots \times x_i

Args:
    {input}
    dim  (int): the dimension to do the operation over
    {dtype}
    {out}

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([ 0.6001,  0.2069, -0.1919,  0.9792,  0.6727,  1.0062,  0.4126,
            -0.2129, -0.4206,  0.1968])
    >>> torch.cumprod(a, dim=0)
    tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0158, -0.0065,
             0.0014, -0.0006, -0.0001])

    >>> a[5] = 0.0
    >>> torch.cumprod(a, dim=0)
    tensor([ 0.6001,  0.1241, -0.0238, -0.0233, -0.0157, -0.0000, -0.0000,
             0.0000, -0.0000, -0.0000])
""".format(**reduceops_common_args))

add_docstr(torch.cumsum,
           r"""
cumsum(input, dim, out=None, dtype=None) -> Tensor

Returns the cumulative sum of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be
a vector of size N, with elements.

.. math::
    y_i = x_1 + x_2 + x_3 + \dots + x_i

Args:
    {input}
    dim  (int): the dimension to do the operation over
    {dtype}
    {out}

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([-0.8286, -0.4890,  0.5155,  0.8443,  0.1865, -0.1752, -2.0595,
             0.1850, -1.1571, -0.4243])
    >>> torch.cumsum(a, dim=0)
    tensor([-0.8286, -1.3175, -0.8020,  0.0423,  0.2289,  0.0537, -2.0058,
            -1.8209, -2.9780, -3.4022])
""".format(**reduceops_common_args))

add_docstr(torch.diag,
           r"""
diag(input, diagonal=0, out=None) -> Tensor

- If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
  with the elements of :attr:`input` as the diagonal.
- If :attr:`input` is a matrix (2-D tensor), then returns a 1-D tensor with
  the diagonal elements of :attr:`input`.

The argument :attr:`diagonal` controls which diagonal to consider:

- If :attr:`diagonal` = 0, it is the main diagonal.
- If :attr:`diagonal` > 0, it is above the main diagonal.
- If :attr:`diagonal` < 0, it is below the main diagonal.

Args:
    {input}
    diagonal (int, optional): the diagonal to consider
    {out}

.. seealso::

        :func:`torch.diagonal` always returns the diagonal of its input.

        :func:`torch.diagflat` always constructs a tensor with diagonal elements
        specified by the input.

Examples:

Get the square matrix where the input vector is the diagonal::

    >>> a = torch.randn(3)
    >>> a
    tensor([ 0.5950,-0.0872, 2.3298])
    >>> torch.diag(a)
    tensor([[ 0.5950, 0.0000, 0.0000],
            [ 0.0000,-0.0872, 0.0000],
            [ 0.0000, 0.0000, 2.3298]])
    >>> torch.diag(a, 1)
    tensor([[ 0.0000, 0.5950, 0.0000, 0.0000],
            [ 0.0000, 0.0000,-0.0872, 0.0000],
            [ 0.0000, 0.0000, 0.0000, 2.3298],
            [ 0.0000, 0.0000, 0.0000, 0.0000]])

Get the k-th diagonal of a given matrix::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-0.4264, 0.0255,-0.1064],
            [ 0.8795,-0.2429, 0.1374],
            [ 0.1029,-0.6482,-1.6300]])
    >>> torch.diag(a, 0)
    tensor([-0.4264,-0.2429,-1.6300])
    >>> torch.diag(a, 1)
    tensor([ 0.0255, 0.1374])
""".format(**common_args))

add_docstr(torch.diag_embed,
           r"""
diag_embed(input, offset=0, dim1=-2, dim2=-1) -> Tensor

Creates a tensor whose diagonals of certain 2D planes (specified by
:attr:`dim1` and :attr:`dim2`) are filled by :attr:`input`.
To facilitate creating batched diagonal matrices, the 2D planes formed by
the last two dimensions of the returned tensor are chosen by default.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.
- If :attr:`offset` > 0, it is above the main diagonal.
- If :attr:`offset` < 0, it is below the main diagonal.

The size of the new matrix will be calculated to make the specified diagonal
of the size of the last input dimension.
Note that for :attr:`offset` other than :math:`0`, the order of :attr:`dim1`
and :attr:`dim2` matters. Exchanging them is equivalent to changing the
sign of :attr:`offset`.

Applying :meth:`torch.diagonal` to the output of this function with
the same arguments yields a matrix identical to input. However,
:meth:`torch.diagonal` has different default dimensions, so those
need to be explicitly specified.

Args:
    {input} Must be at least 1-dimensional.
    offset (int, optional): which diagonal to consider. Default: 0
        (main diagonal).
    dim1 (int, optional): first dimension with respect to which to
        take diagonal. Default: -2.
    dim2 (int, optional): second dimension with respect to which to
        take diagonal. Default: -1.

Example::

    >>> a = torch.randn(2, 3)
    >>> torch.diag_embed(a)
    tensor([[[ 1.5410,  0.0000,  0.0000],
             [ 0.0000, -0.2934,  0.0000],
             [ 0.0000,  0.0000, -2.1788]],

            [[ 0.5684,  0.0000,  0.0000],
             [ 0.0000, -1.0845,  0.0000],
             [ 0.0000,  0.0000, -1.3986]]])

    >>> torch.diag_embed(a, offset=1, dim1=0, dim2=2)
    tensor([[[ 0.0000,  1.5410,  0.0000,  0.0000],
             [ 0.0000,  0.5684,  0.0000,  0.0000]],

            [[ 0.0000,  0.0000, -0.2934,  0.0000],
             [ 0.0000,  0.0000, -1.0845,  0.0000]],

            [[ 0.0000,  0.0000,  0.0000, -2.1788],
             [ 0.0000,  0.0000,  0.0000, -1.3986]],

            [[ 0.0000,  0.0000,  0.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000,  0.0000]]])
""".format(**common_args))


add_docstr(torch.diagflat,
           r"""
diagflat(input, offset=0) -> Tensor

- If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
  with the elements of :attr:`input` as the diagonal.
- If :attr:`input` is a tensor with more than one dimension, then returns a
  2-D tensor with diagonal elements equal to a flattened :attr:`input`.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.
- If :attr:`offset` > 0, it is above the main diagonal.
- If :attr:`offset` < 0, it is below the main diagonal.

Args:
    {input}
    offset (int, optional): the diagonal to consider. Default: 0 (main
        diagonal).

Examples::

    >>> a = torch.randn(3)
    >>> a
    tensor([-0.2956, -0.9068,  0.1695])
    >>> torch.diagflat(a)
    tensor([[-0.2956,  0.0000,  0.0000],
            [ 0.0000, -0.9068,  0.0000],
            [ 0.0000,  0.0000,  0.1695]])
    >>> torch.diagflat(a, 1)
    tensor([[ 0.0000, -0.2956,  0.0000,  0.0000],
            [ 0.0000,  0.0000, -0.9068,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.1695],
            [ 0.0000,  0.0000,  0.0000,  0.0000]])

    >>> a = torch.randn(2, 2)
    >>> a
    tensor([[ 0.2094, -0.3018],
            [-0.1516,  1.9342]])
    >>> torch.diagflat(a)
    tensor([[ 0.2094,  0.0000,  0.0000,  0.0000],
            [ 0.0000, -0.3018,  0.0000,  0.0000],
            [ 0.0000,  0.0000, -0.1516,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  1.9342]])
""".format(**common_args))

add_docstr(torch.diagonal,
           r"""
diagonal(input, offset=0, dim1=0, dim2=1) -> Tensor

Returns a partial view of :attr:`input` with the its diagonal elements
with respect to :attr:`dim1` and :attr:`dim2` appended as a dimension
at the end of the shape.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.
- If :attr:`offset` > 0, it is above the main diagonal.
- If :attr:`offset` < 0, it is below the main diagonal.

Applying :meth:`torch.diag_embed` to the output of this function with
the same arguments yields a diagonal matrix with the diagonal entries
of the input. However, :meth:`torch.diag_embed` has different default
dimensions, so those need to be explicitly specified.

Args:
    {input} Must be at least 2-dimensional.
    offset (int, optional): which diagonal to consider. Default: 0
        (main diagonal).
    dim1 (int, optional): first dimension with respect to which to
        take diagonal. Default: 0.
    dim2 (int, optional): second dimension with respect to which to
        take diagonal. Default: 1.

.. note::  To take a batch diagonal, pass in dim1=-2, dim2=-1.

Examples::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-1.0854,  1.1431, -0.1752],
            [ 0.8536, -0.0905,  0.0360],
            [ 0.6927, -0.3735, -0.4945]])


    >>> torch.diagonal(a, 0)
    tensor([-1.0854, -0.0905, -0.4945])


    >>> torch.diagonal(a, 1)
    tensor([ 1.1431,  0.0360])


    >>> x = torch.randn(2, 5, 4, 2)
    >>> torch.diagonal(x, offset=-1, dim1=1, dim2=2)
    tensor([[[-1.2631,  0.3755, -1.5977, -1.8172],
             [-1.1065,  1.0401, -0.2235, -0.7938]],

            [[-1.7325, -0.3081,  0.6166,  0.2335],
             [ 1.0500,  0.7336, -0.3836, -1.1015]]])
""".format(**common_args))

add_docstr(torch.digamma,
           r"""
digamma(input, out=None) -> Tensor

Computes the logarithmic derivative of the gamma function on `input`.

.. math::
    \psi(x) = \frac{d}{dx} \ln\left(\Gamma\left(x\right)\right) = \frac{\Gamma'(x)}{\Gamma(x)}

Args:
    input (Tensor): the tensor to compute the digamma function on

Example::

    >>> a = torch.tensor([1, 0.5])
    >>> torch.digamma(a)
    tensor([-0.5772, -1.9635])
""")


add_docstr(torch.dist,
           r"""
dist(input, other, p=2) -> Tensor

Returns the p-norm of (:attr:`input` - :attr:`other`)

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    {input}
    other (Tensor): the Right-hand-side input tensor
    p (float, optional): the norm to be computed

Example::

    >>> x = torch.randn(4)
    >>> x
    tensor([-1.5393, -0.8675,  0.5916,  1.6321])
    >>> y = torch.randn(4)
    >>> y
    tensor([ 0.0967, -1.0511,  0.6295,  0.8360])
    >>> torch.dist(x, y, 3.5)
    tensor(1.6727)
    >>> torch.dist(x, y, 3)
    tensor(1.6973)
    >>> torch.dist(x, y, 0)
    tensor(inf)
    >>> torch.dist(x, y, 1)
    tensor(2.6537)
""".format(**common_args))

add_docstr(torch.div,
           r"""
.. function:: div(input, other, out=None) -> Tensor

Divides each element of the input :attr:`input` with the scalar :attr:`other`
and returns a new resulting tensor.

.. math::
    \text{out}_i = \frac{\text{input}_i}{\text{other}}

If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`other`
should be a real number, otherwise it should be an integer

Args:
    {input}
    other (Number): the number to be divided to each element of :attr:`input`
    {out}

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
    >>> torch.div(a, 0.5)
    tensor([ 0.7620,  2.5548, -0.5944, -0.7439,  0.9275])

.. function:: div(input, other, out=None) -> Tensor

Each element of the tensor :attr:`input` is divided by each element
of the tensor :attr:`other`. The resulting tensor is returned. The shapes of
:attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

.. math::
    \text{out}_i = \frac{\text{input}_i}{\text{other}_i}
""" + r"""
Args:
    input (Tensor): the numerator tensor
    other (Tensor): the denominator tensor
    {out}

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
            [ 0.1815, -1.0111,  0.9805, -1.5923],
            [ 0.1062,  1.4581,  0.7759, -1.2344],
            [-0.1830, -0.0313,  1.1908, -1.4757]])
    >>> b = torch.randn(4)
    >>> b
    tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
    >>> torch.div(a, b)
    tensor([[-0.4620, -6.6051,  0.5676,  1.2637],
            [ 0.2260, -3.4507, -1.2086,  6.8988],
            [ 0.1322,  4.9764, -0.9564,  5.3480],
            [-0.2278, -0.1068, -1.4678,  6.3936]])
""".format(**common_args))

add_docstr(torch.dot,
           r"""
dot(input, tensor) -> Tensor

Computes the dot product (inner product) of two tensors.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

Example::

    >>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
    tensor(7)
""")

add_docstr(torch.eig,
           r"""
eig(input, eigenvectors=False, out=None) -> (Tensor, Tensor)

Computes the eigenvalues and eigenvectors of a real square matrix.

.. note::
    Since eigenvalues and eigenvectors might be complex, backward pass is supported only
    for :func:`torch.symeig`

Args:
    input (Tensor): the square matrix of shape :math:`(n \times n)` for which the eigenvalues and eigenvectors
        will be computed
    eigenvectors (bool): ``True`` to compute both eigenvalues and eigenvectors;
        otherwise, only eigenvalues will be computed
    out (tuple, optional): the output tensors

Returns:
    (Tensor, Tensor): A namedtuple (eigenvalues, eigenvectors) containing

        - **eigenvalues** (*Tensor*): Shape :math:`(n \times 2)`. Each row is an eigenvalue of ``input``,
          where the first element is the real part and the second element is the imaginary part.
          The eigenvalues are not necessarily ordered.
        - **eigenvectors** (*Tensor*): If ``eigenvectors=False``, it's an empty tensor.
          Otherwise, this tensor of shape :math:`(n \times n)` can be used to compute normalized (unit length)
          eigenvectors of corresponding eigenvalues as follows.
          If the corresponding `eigenvalues[j]` is a real number, column `eigenvectors[:, j]` is the eigenvector
          corresponding to `eigenvalues[j]`.
          If the corresponding `eigenvalues[j]` and `eigenvalues[j + 1]` form a complex conjugate pair, then the
          true eigenvectors can be computed as
          :math:`\text{true eigenvector}[j] = eigenvectors[:, j] + i \times eigenvectors[:, j + 1]`,
          :math:`\text{true eigenvector}[j + 1] = eigenvectors[:, j] - i \times eigenvectors[:, j + 1]`.
""")

add_docstr(torch.eq,
           r"""
eq(input, other, out=None) -> Tensor

Computes element-wise equality

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare
    {out} Must be a `ByteTensor`

Returns:
    Tensor: A ``torch.BoolTensor`` containing a True at each location where comparison is true

Example::

    >>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[ 1,  0],
            [ 0,  1]], dtype=torch.uint8)
""".format(**common_args))

add_docstr(torch.equal,
           r"""
equal(input, other) -> bool

``True`` if two tensors have the same size and elements, ``False`` otherwise.

Example::

    >>> torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
    True
""")

add_docstr(torch.erf,
           r"""
erf(input, out=None) -> Tensor

Computes the error function of each element. The error function is defined as follows:

.. math::
    \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> torch.erf(torch.tensor([0, -1., 10.]))
    tensor([ 0.0000, -0.8427,  1.0000])
""".format(**common_args))

add_docstr(torch.erfc,
           r"""
erfc(input, out=None) -> Tensor

Computes the complementary error function of each element of :attr:`input`.
The complementary error function is defined as follows:

.. math::
    \mathrm{erfc}(x) = 1 - \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> torch.erfc(torch.tensor([0, -1., 10.]))
    tensor([ 1.0000, 1.8427,  0.0000])
""".format(**common_args))

add_docstr(torch.erfinv,
           r"""
erfinv(input, out=None) -> Tensor

Computes the inverse error function of each element of :attr:`input`.
The inverse error function is defined in the range :math:`(-1, 1)` as:

.. math::
    \mathrm{erfinv}(\mathrm{erf}(x)) = x
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> torch.erfinv(torch.tensor([0, 0.5, -1.]))
    tensor([ 0.0000,  0.4769,    -inf])
""".format(**common_args))

add_docstr(torch.exp,
           r"""
exp(input, out=None) -> Tensor

Returns a new tensor with the exponential of the elements
of the input tensor :attr:`input`.

.. math::
    y_{i} = e^{x_{i}}
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> torch.exp(torch.tensor([0, math.log(2.)]))
    tensor([ 1.,  2.])
""".format(**common_args))

add_docstr(torch.expm1,
           r"""
expm1(input, out=None) -> Tensor

Returns a new tensor with the exponential of the elements minus 1
of :attr:`input`.

.. math::
    y_{i} = e^{x_{i}} - 1
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> torch.expm1(torch.tensor([0, math.log(2.)]))
    tensor([ 0.,  1.])
""".format(**common_args))

add_docstr(torch.eye,
           r"""
eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

Args:
    n (int): the number of rows
    m (int, optional): the number of columns with default being :attr:`n`
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Returns:
    Tensor: A 2-D tensor with ones on the diagonal and zeros elsewhere

Example::

    >>> torch.eye(3)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])
""".format(**factory_common_args))

add_docstr(torch.floor,
           r"""
floor(input, out=None) -> Tensor

Returns a new tensor with the floor of the elements of :attr:`input`,
the largest integer less than or equal to each element.

.. math::
    \text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.8166,  1.5308, -0.2530, -0.2091])
    >>> torch.floor(a)
    tensor([-1.,  1., -1., -1.])
""".format(**common_args))

add_docstr(torch.fmod,
           r"""
fmod(input, other, out=None) -> Tensor

Computes the element-wise remainder of division.

The dividend and divisor may contain both for integer and floating point
numbers. The remainder has the same sign as the dividend :attr:`input`.

When :attr:`other` is a tensor, the shapes of :attr:`input` and
:attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the dividend
    other (Tensor or float): the divisor, which may be either a number or a tensor of the same shape as the dividend
    {out}

Example::

    >>> torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
    tensor([-1., -0., -1.,  1.,  0.,  1.])
    >>> torch.fmod(torch.tensor([1., 2, 3, 4, 5]), 1.5)
    tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])


""".format(**common_args))

add_docstr(torch.frac,
           r"""
frac(input, out=None) -> Tensor

Computes the fractional portion of each element in :attr:`input`.

.. math::
    \text{out}_{i} = \text{input}_{i} - \left\lfloor |\text{input}_{i}| \right\rfloor * \operatorname{sgn}(\text{input}_{i})

Example::

    >>> torch.frac(torch.tensor([1, 2.5, -3.2]))
    tensor([ 0.0000,  0.5000, -0.2000])
""")

add_docstr(torch.from_numpy,
           r"""
from_numpy(ndarray) -> Tensor

Creates a :class:`Tensor` from a :class:`numpy.ndarray`.

The returned tensor and :attr:`ndarray` share the same memory. Modifications to
the tensor will be reflected in the :attr:`ndarray` and vice versa. The returned
tensor is not resizable.

It currently accepts :attr:`ndarray` with dtypes of ``numpy.float64``,
``numpy.float32``, ``numpy.float16``, ``numpy.int64``, ``numpy.int32``,
``numpy.int16``, ``numpy.int8``, ``numpy.uint8``, and ``numpy.bool``.

Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.from_numpy(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])
""")

add_docstr(torch.flatten,
           r"""
flatten(input, start_dim=0, end_dim=-1) -> Tensor

Flattens a contiguous range of dims in a tensor.

Args:
    {input}
    start_dim (int): the first dim to flatten
    end_dim (int): the last dim to flatten

Example::

    >>> t = torch.tensor([[[1, 2],
                           [3, 4]],
                          [[5, 6],
                           [7, 8]]])
    >>> torch.flatten(t)
    tensor([1, 2, 3, 4, 5, 6, 7, 8])
    >>> torch.flatten(t, start_dim=1)
    tensor([[1, 2, 3, 4],
            [5, 6, 7, 8]])
""".format(**common_args))

add_docstr(torch.gather,
           r"""
gather(input, dim, index, out=None, sparse_grad=False) -> Tensor

Gathers values along an axis specified by `dim`.

For a 3-D tensor the output is specified by::

    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

If :attr:`input` is an n-dimensional tensor with size
:math:`(x_0, x_1..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
and ``dim = i``, then :attr:`index` must be an :math:`n`-dimensional tensor with
size :math:`(x_0, x_1, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})` where :math:`y \geq 1`
and :attr:`out` will have the same size as :attr:`index`.
""" + r"""
Args:
    input (Tensor): the source tensor
    dim (int): the axis along which to index
    index (LongTensor): the indices of elements to gather
    out (Tensor, optional): the destination tensor
    sparse_grad(bool,optional): If ``True``, gradient w.r.t. :attr:`input` will be a sparse tensor.

Example::

    >>> t = torch.tensor([[1,2],[3,4]])
    >>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
    tensor([[ 1,  1],
            [ 4,  3]])
""")

add_docstr(torch.ge,
           r"""
ge(input, other, out=None) -> Tensor

Computes :math:`\text{input} \geq \text{other}` element-wise.

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare
    out (Tensor, optional): the output tensor that must be a `BoolTensor`

Returns:
    Tensor: A ``torch.BoolTensor`` containing a True at each location where comparison is true

Example::

    >>> torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[True, True], [False, True]])
""")

add_docstr(torch.geqrf,
           r"""
geqrf(input, out=None) -> (Tensor, Tensor)

This is a low-level function for calling LAPACK directly. This function
returns a namedtuple (a, tau) as defined in `LAPACK documentation for geqrf`_ .

You'll generally want to use :func:`torch.qr` instead.

Computes a QR decomposition of :attr:`input`, but without constructing
:math:`Q` and :math:`R` as explicit separate matrices.

Rather, this directly calls the underlying LAPACK function `?geqrf`
which produces a sequence of 'elementary reflectors'.

See `LAPACK documentation for geqrf`_ for further details.

Args:
    input (Tensor): the input matrix
    out (tuple, optional): the output tuple of (Tensor, Tensor)

.. _LAPACK documentation for geqrf:
    https://software.intel.com/en-us/node/521004

""")

add_docstr(torch.ger,
           r"""
ger(input, vec2, out=None) -> Tensor

Outer product of :attr:`input` and :attr:`vec2`.
If :attr:`input` is a vector of size :math:`n` and :attr:`vec2` is a vector of
size :math:`m`, then :attr:`out` must be a matrix of size :math:`(n \times m)`.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

Args:
    input (Tensor): 1-D input vector
    vec2 (Tensor): 1-D input vector
    out (Tensor, optional): optional output matrix

Example::

    >>> v1 = torch.arange(1., 5.)
    >>> v2 = torch.arange(1., 4.)
    >>> torch.ger(v1, v2)
    tensor([[  1.,   2.,   3.],
            [  2.,   4.,   6.],
            [  3.,   6.,   9.],
            [  4.,   8.,  12.]])
""")

add_docstr(torch.solve,
           r"""
torch.solve(input, A, out=None) -> (Tensor, Tensor)

This function returns the solution to the system of linear
equations represented by :math:`AX = B` and the LU factorization of
A, in order as a namedtuple `solution, LU`.

`LU` contains `L` and `U` factors for LU factorization of `A`.

`torch.solve(B, A)` can take in 2D inputs `B, A` or inputs that are
batches of 2D matrices. If the inputs are batches, then returns
batched outputs `solution, LU`.

.. note::

    Irrespective of the original strides, the returned matrices
    `solution` and `LU` will be transposed, i.e. with strides like
    `B.contiguous().transpose(-1, -2).stride()` and
    `A.contiguous().transpose(-1, -2).stride()` respectively.

Args:
    input (Tensor): input matrix :math:`B` of size :math:`(*, m, k)` , where :math:`*`
                is zero or more batch dimensions.
    A (Tensor): input square matrix of size :math:`(*, m, m)`, where
                :math:`*` is zero or more batch dimensions.
    out ((Tensor, Tensor), optional): optional output tuple.

Example::

    >>> A = torch.tensor([[6.80, -2.11,  5.66,  5.97,  8.23],
                          [-6.05, -3.30,  5.36, -4.44,  1.08],
                          [-0.45,  2.58, -2.70,  0.27,  9.04],
                          [8.32,  2.71,  4.35,  -7.17,  2.14],
                          [-9.67, -5.14, -7.26,  6.08, -6.87]]).t()
    >>> B = torch.tensor([[4.02,  6.19, -8.22, -7.57, -3.03],
                          [-1.56,  4.00, -8.67,  1.75,  2.86],
                          [9.81, -4.09, -4.57, -8.61,  8.99]]).t()
    >>> X, LU = torch.solve(B, A)
    >>> torch.dist(B, torch.mm(A, X))
    tensor(1.00000e-06 *
           7.0977)

    >>> # Batched solver example
    >>> A = torch.randn(2, 3, 1, 4, 4)
    >>> B = torch.randn(2, 3, 1, 4, 6)
    >>> X, LU = torch.solve(B, A)
    >>> torch.dist(B, A.matmul(X))
    tensor(1.00000e-06 *
       3.6386)

""")

add_docstr(torch.get_default_dtype,
           r"""
get_default_dtype() -> torch.dtype

Get the current default floating point :class:`torch.dtype`.

Example::

    >>> torch.get_default_dtype()  # initial default for floating point is torch.float32
    torch.float32
    >>> torch.set_default_dtype(torch.float64)
    >>> torch.get_default_dtype()  # default is now changed to torch.float64
    torch.float64
    >>> torch.set_default_tensor_type(torch.FloatTensor)  # setting tensor type also affects this
    >>> torch.get_default_dtype()  # changed to torch.float32, the dtype for torch.FloatTensor
    torch.float32

""")

add_docstr(torch.get_num_threads,
           r"""
get_num_threads() -> int

Returns the number of threads used for parallelizing CPU operations
""")

add_docstr(torch.get_num_interop_threads,
           r"""
get_num_interop_threads() -> int

Returns the number of threads used for inter-op parallelism on CPU
(e.g. in JIT interpreter)
""")

add_docstr(torch.gt,
           r"""
gt(input, other, out=None) -> Tensor

Computes :math:`\text{input} > \text{other}` element-wise.

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare
    out (Tensor, optional): the output tensor that must be a `BoolTensor`

Returns:
    Tensor: A ``torch.BoolTensor`` containing a True at each location where comparison is true

Example::

    >>> torch.gt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [False, False]])
""")

add_docstr(torch.histc,
           r"""
histc(input, bins=100, min=0, max=0, out=None) -> Tensor

Computes the histogram of a tensor.

The elements are sorted into equal width bins between :attr:`min` and
:attr:`max`. If :attr:`min` and :attr:`max` are both zero, the minimum and
maximum values of the data are used.

Args:
    {input}
    bins (int): number of histogram bins
    min (int): lower end of the range (inclusive)
    max (int): upper end of the range (inclusive)
    {out}

Returns:
    Tensor: Histogram represented as a tensor

Example::

    >>> torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3)
    tensor([ 0.,  2.,  1.,  0.])
""".format(**common_args))

add_docstr(torch.index_select,
           r"""
index_select(input, dim, index, out=None) -> Tensor

Returns a new tensor which indexes the :attr:`input` tensor along dimension
:attr:`dim` using the entries in :attr:`index` which is a `LongTensor`.

The returned tensor has the same number of dimensions as the original tensor
(:attr:`input`).  The :attr:`dim`\ th dimension has the same size as the length
of :attr:`index`; other dimensions have the same size as in the original tensor.

.. note:: The returned tensor does **not** use the same storage as the original
          tensor.  If :attr:`out` has a different shape than expected, we
          silently change it to the correct shape, reallocating the underlying
          storage if necessary.

Args:
    {input}
    dim (int): the dimension in which we index
    index (LongTensor): the 1-D tensor containing the indices to index
    {out}

Example::

    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-0.4664,  0.2647, -0.1228, -1.1068],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> indices = torch.tensor([0, 2])
    >>> torch.index_select(x, 0, indices)
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> torch.index_select(x, 1, indices)
    tensor([[ 0.1427, -0.5414],
            [-0.4664, -0.1228],
            [-1.1734,  0.7230]])
""".format(**common_args))

add_docstr(torch.inverse,
           r"""
inverse(input, out=None) -> Tensor

Takes the inverse of the square matrix :attr:`input`. :attr:`input` can be batches
of 2D square tensors, in which case this function would return a tensor composed of
individual inverses.

.. note::

    Irrespective of the original strides, the returned tensors will be
    transposed, i.e. with strides like `input.contiguous().transpose(-2, -1).stride()`

Args:
    input (Tensor): the input tensor of size :math:`(*, n, n)` where `*` is zero or more
                    batch dimensions
    {out}

Example::

    >>> x = torch.rand(4, 4)
    >>> y = torch.inverse(x)
    >>> z = torch.mm(x, y)
    >>> z
    tensor([[ 1.0000, -0.0000, -0.0000,  0.0000],
            [ 0.0000,  1.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  1.0000,  0.0000],
            [ 0.0000, -0.0000, -0.0000,  1.0000]])
    >>> torch.max(torch.abs(z - torch.eye(4))) # Max non-zero
    tensor(1.1921e-07)
    >>> # Batched inverse example
    >>> x = torch.randn(2, 3, 4, 4)
    >>> y = torch.inverse(x)
    >>> z = torch.matmul(x, y)
    >>> torch.max(torch.abs(z - torch.eye(4).expand_as(x))) # Max non-zero
    tensor(1.9073e-06)
""".format(**common_args))

add_docstr(torch.isnan,
           r"""
Returns a new tensor with boolean elements representing if each element is `NaN` or not.

Arguments:
    input (Tensor): A tensor to check

Returns:
    Tensor: A ``torch.BoolTensor`` containing a True at each location of `NaN` elements.

Example::

    >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
    tensor([False, True, False])
""")

add_docstr(torch.is_floating_point,
           r"""
is_floating_point(input) -> (bool)

Returns True if the data type of :attr:`input` is a floating point data type i.e.,
one of ``torch.float64``, ``torch.float32`` and ``torch.float16``.

Args:
    input (Tensor): the PyTorch tensor to test
""")

add_docstr(torch.kthvalue,
           r"""
kthvalue(input, k, dim=None, keepdim=False, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the :attr:`k` th
smallest element of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each element found.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`keepdim` is ``True``, both the :attr:`values` and :attr:`indices` tensors
are the same size as :attr:`input`, except in the dimension :attr:`dim` where
they are of size 1. Otherwise, :attr:`dim` is squeezed
(see :func:`torch.squeeze`), resulting in both the :attr:`values` and
:attr:`indices` tensors having 1 fewer dimension than the :attr:`input` tensor.

Args:
    {input}
    k (int): k for the k-th smallest element
    dim (int, optional): the dimension to find the kth value along
    {keepdim}
    out (tuple, optional): the output tuple of (Tensor, LongTensor)
                           can be optionally given to be used as output buffers

Example::

    >>> x = torch.arange(1., 6.)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.])
    >>> torch.kthvalue(x, 4)
    torch.return_types.kthvalue(values=tensor(4.), indices=tensor(3))

    >>> x=torch.arange(1.,7.).resize_(2,3)
    >>> x
    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.]])
    >>> torch.kthvalue(x, 2, 0, True)
    torch.return_types.kthvalue(values=tensor([[4., 5., 6.]]), indices=tensor([[1, 1, 1]]))
""".format(**single_dim_common))

add_docstr(torch.le,
           r"""
le(input, other, out=None) -> Tensor

Computes :math:`\text{input} \leq \text{other}` element-wise.

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare
    out (Tensor, optional): the output tensor that must be a `BoolTensor`

Returns:
    Tensor: A ``torch.BoolTensor`` containing a True at each location where comparison is true

Example::

    >>> torch.le(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[True, False], [True, True]])
""")

add_docstr(torch.lerp,
           r"""
lerp(input, end, weight, out=None)

Does a linear interpolation of two tensors :attr:`start` (given by :attr:`input`) and :attr:`end` based
on a scalar or tensor :attr:`weight` and returns the resulting :attr:`out` tensor.

.. math::
    \text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)
""" + r"""
The shapes of :attr:`start` and :attr:`end` must be
:ref:`broadcastable <broadcasting-semantics>`. If :attr:`weight` is a tensor, then
the shapes of :attr:`weight`, :attr:`start`, and :attr:`end` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the tensor with the starting points
    end (Tensor): the tensor with the ending points
    weight (float or tensor): the weight for the interpolation formula
    {out}

Example::

    >>> start = torch.arange(1., 5.)
    >>> end = torch.empty(4).fill_(10)
    >>> start
    tensor([ 1.,  2.,  3.,  4.])
    >>> end
    tensor([ 10.,  10.,  10.,  10.])
    >>> torch.lerp(start, end, 0.5)
    tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
    >>> torch.lerp(start, end, torch.full_like(start, 0.5))
    tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
""".format(**common_args))

add_docstr(torch.linspace,
           r"""
linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a one-dimensional tensor of :attr:`steps`
equally spaced points between :attr:`start` and :attr:`end`.

The output tensor is 1-D of size :attr:`steps`.

Args:
    start (float): the starting value for the set of points
    end (float): the ending value for the set of points
    steps (int): number of points to sample between :attr:`start`
        and :attr:`end`. Default: ``100``.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}


Example::

    >>> torch.linspace(3, 10, steps=5)
    tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])
    >>> torch.linspace(-10, 10, steps=5)
    tensor([-10.,  -5.,   0.,   5.,  10.])
    >>> torch.linspace(start=-10, end=10, steps=5)
    tensor([-10.,  -5.,   0.,   5.,  10.])
    >>> torch.linspace(start=-10, end=10, steps=1)
    tensor([-10.])
""".format(**factory_common_args))

add_docstr(torch.log,
           r"""
log(input, out=None) -> Tensor

Returns a new tensor with the natural logarithm of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{e} (x_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([-0.7168, -0.5471, -0.8933, -1.4428, -0.1190])
    >>> torch.log(a)
    tensor([ nan,  nan,  nan,  nan,  nan])
""".format(**common_args))

add_docstr(torch.log10,
           r"""
log10(input, out=None) -> Tensor

Returns a new tensor with the logarithm to the base 10 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{10} (x_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.rand(5)
    >>> a
    tensor([ 0.5224,  0.9354,  0.7257,  0.1301,  0.2251])


    >>> torch.log10(a)
    tensor([-0.2820, -0.0290, -0.1392, -0.8857, -0.6476])

""".format(**common_args))

add_docstr(torch.log1p,
           r"""
log1p(input, out=None) -> Tensor

Returns a new tensor with the natural logarithm of (1 + :attr:`input`).

.. math::
    y_i = \log_{e} (x_i + 1)
""" + r"""
.. note:: This function is more accurate than :func:`torch.log` for small
          values of :attr:`input`

Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([-1.0090, -0.9923,  1.0249, -0.5372,  0.2492])
    >>> torch.log1p(a)
    tensor([    nan, -4.8653,  0.7055, -0.7705,  0.2225])
""".format(**common_args))

add_docstr(torch.log2,
           r"""
log2(input, out=None) -> Tensor

Returns a new tensor with the logarithm to the base 2 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{2} (x_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.rand(5)
    >>> a
    tensor([ 0.8419,  0.8003,  0.9971,  0.5287,  0.0490])


    >>> torch.log2(a)
    tensor([-0.2483, -0.3213, -0.0042, -0.9196, -4.3504])

""".format(**common_args))

add_docstr(torch.logical_not,
           r"""
logical_not(input, out=None) -> Tensor

Computes the element-wise logical NOT of the given input tensor. If not specified, the output tensor will have the bool
dtype. If the input tensor is not a bool tensor, zeros are treated as ``False`` and non-zeros are treated as ``True``.

Args:
    {input}
    {out}

Example::

    >>> torch.logical_not(torch.tensor([True, False]))
    tensor([ False,  True])
    >>> torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8))
    tensor([ True, False, False])
    >>> torch.logical_not(torch.tensor([0., 1.5, -10.], dtype=torch.double))
    tensor([ True, False, False])
    >>> torch.logical_not(torch.tensor([0., 1., -10.], dtype=torch.double), out=torch.empty(3, dtype=torch.int16))
    tensor([1, 0, 0], dtype=torch.int16)
""".format(**common_args))

add_docstr(torch.logical_xor,
           r"""
logical_xor(input, other, out=None) -> Tensor

Computes the element-wise logical XOR of the given input tensors. Both input tensors must have the bool dtype.

Args:
    {input}
    other (Tensor): the tensor to compute XOR with
    {out}

Example::

    >>> torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
    tensor([ False, False,  True])
""".format(**common_args))

add_docstr(torch.logspace,
           r"""
logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a one-dimensional tensor of :attr:`steps` points
logarithmically spaced with base :attr:`base` between
:math:`{{\text{{base}}}}^{{\text{{start}}}}` and :math:`{{\text{{base}}}}^{{\text{{end}}}}`.

The output tensor is 1-D of size :attr:`steps`.

Args:
    start (float): the starting value for the set of points
    end (float): the ending value for the set of points
    steps (int): number of points to sample between :attr:`start`
        and :attr:`end`. Default: ``100``.
    base (float): base of the logarithm function. Default: ``10.0``.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.logspace(start=-10, end=10, steps=5)
    tensor([ 1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10])
    >>> torch.logspace(start=0.1, end=1.0, steps=5)
    tensor([  1.2589,   2.1135,   3.5481,   5.9566,  10.0000])
    >>> torch.logspace(start=0.1, end=1.0, steps=1)
    tensor([1.2589])
    >>> torch.logspace(start=2, end=2, steps=1, base=2)
    tensor([4.0])
""".format(**factory_common_args))

add_docstr(torch.logsumexp,
           r"""
logsumexp(input, dim, keepdim=False, out=None)

Returns the log of summed exponentials of each row of the :attr:`input`
tensor in the given dimension :attr:`dim`. The computation is numerically
stabilized.

For summation index :math:`j` given by `dim` and other indices :math:`i`, the result is

    .. math::
        \text{{logsumexp}}(x)_{{i}} = \log \sum_j \exp(x_{{ij}})

{keepdim_details}

Args:
    {input}
    {dim}
    {keepdim}
    {out}


Example::
    >>> a = torch.randn(3, 3)
    >>> torch.logsumexp(a, 1)
    tensor([ 0.8442,  1.4322,  0.8711])
""".format(**multi_dim_common))

add_docstr(torch.lstsq,
           r"""
lstsq(input, A, out=None) -> Tensor

Computes the solution to the least squares and least norm problems for a full
rank matrix :math:`A` of size :math:`(m \times n)` and a matrix :math:`B` of
size :math:`(m \times k)`.

If :math:`m \geq n`, :func:`lstsq` solves the least-squares problem:

.. math::

   \begin{array}{ll}
   \min_X & \|AX-B\|_2.
   \end{array}

If :math:`m < n`, :func:`lstsq` solves the least-norm problem:

.. math::

   \begin{array}{ll}
   \min_X & \|X\|_2 & \text{subject to} & AX = B.
   \end{array}

Returned tensor :math:`X` has shape :math:`(\max(m, n) \times k)`. The first :math:`n`
rows of :math:`X` contains the solution. If :math:`m \geq n`, the residual sum of squares
for the solution in each column is given by the sum of squares of elements in the
remaining :math:`m - n` rows of that column.

.. note::
    The case when :math:`m < n` is not supported on the GPU.

Args:
    input (Tensor): the matrix :math:`B`
    A (Tensor): the :math:`m` by :math:`n` matrix :math:`A`
    out (tuple, optional): the optional destination tensor

Returns:
    (Tensor, Tensor): A namedtuple (solution, QR) containing:

        - **solution** (*Tensor*): the least squares solution
        - **QR** (*Tensor*): the details of the QR factorization

.. note::

    The returned matrices will always be transposed, irrespective of the strides
    of the input matrices. That is, they will have stride `(1, m)` instead of
    `(m, 1)`.

Example::

    >>> A = torch.tensor([[1., 1, 1],
                          [2, 3, 4],
                          [3, 5, 2],
                          [4, 2, 5],
                          [5, 4, 3]])
    >>> B = torch.tensor([[-10., -3],
                          [ 12, 14],
                          [ 14, 12],
                          [ 16, 16],
                          [ 18, 16]])
    >>> X, _ = torch.lstsq(B, A)
    >>> X
    tensor([[  2.0000,   1.0000],
            [  1.0000,   1.0000],
            [  1.0000,   2.0000],
            [ 10.9635,   4.8501],
            [  8.9332,   5.2418]])
""")

add_docstr(torch.lt,
           r"""
lt(input, other, out=None) -> Tensor

Computes :math:`\text{input} < \text{other}` element-wise.

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare
    out (Tensor, optional): the output tensor that must be a `BoolTensor`

Returns:
    Tensor: A `torch.BoolTensor` containing a True at each location where comparison is true

Example::

    >>> torch.lt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, False], [True, False]])
""")

add_docstr(torch.lu_solve,
           r"""
lu_solve(input, LU_data, LU_pivots, out=None) -> Tensor

Returns the LU solve of the linear system :math:`Ax = b` using the partially pivoted
LU factorization of A from :meth:`torch.lu`.

Arguments:
    b (Tensor): the RHS tensor of size :math:`(*, m, k)`, where :math:`*`
                is zero or more batch dimensions.
    LU_data (Tensor): the pivoted LU factorization of A from :meth:`torch.lu` of size :math:`(*, m, m)`,
                       where :math:`*` is zero or more batch dimensions.
    LU_pivots (IntTensor): the pivots of the LU factorization from :meth:`torch.lu` of size :math:`(*, m)`,
                           where :math:`*` is zero or more batch dimensions.
                           The batch dimensions of :attr:`LU_pivots` must be equal to the batch dimensions of
                           :attr:`LU_data`.
    {out}

Example::

    >>> A = torch.randn(2, 3, 3)
    >>> b = torch.randn(2, 3, 1)
    >>> A_LU = torch.lu(A)
    >>> x = torch.lu_solve(b, *A_LU)
    >>> torch.norm(torch.bmm(A, x) - b)
    tensor(1.00000e-07 *
           2.8312)
""".format(**common_args))

add_docstr(torch.masked_select,
           r"""
masked_select(input, mask, out=None) -> Tensor

Returns a new 1-D tensor which indexes the :attr:`input` tensor according to
the boolean mask :attr:`mask` which is a `BoolTensor`.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor don't need
to match, but they must be :ref:`broadcastable <broadcasting-semantics>`.

.. note:: The returned tensor does **not** use the same storage
          as the original tensor

Args:
    {input}
    mask  (ByteTensor): the tensor containing the binary mask to index with
    {out}

Example::

    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
            [-1.2035,  1.2252,  0.5002,  0.6248],
            [ 0.1307, -2.0608,  0.1244,  2.0139]])
    >>> mask = x.ge(0.5)
    >>> mask
    tensor([[False, False, False, False],
            [False, True, True, True],
            [False, False, False, True]])
    >>> torch.masked_select(x, mask)
    tensor([ 1.2252,  0.5002,  0.6248,  2.0139])
""".format(**common_args))

add_docstr(torch.matrix_rank,
           r"""
matrix_rank(input, tol=None, bool symmetric=False) -> Tensor

Returns the numerical rank of a 2-D tensor. The method to compute the
matrix rank is done using SVD by default. If :attr:`symmetric` is ``True``,
then :attr:`input` is assumed to be symmetric, and the computation of the
rank is done by obtaining the eigenvalues.

:attr:`tol` is the threshold below which the singular values (or the eigenvalues
when :attr:`symmetric` is ``True``) are considered to be 0. If :attr:`tol` is not
specified, :attr:`tol` is set to ``S.max() * max(S.size()) * eps`` where `S` is the
singular values (or the eigenvalues when :attr:`symmetric` is ``True``), and ``eps``
is the epsilon value for the datatype of :attr:`input`.

Args:
    input (Tensor): the input 2-D tensor
    tol (float, optional): the tolerance value. Default: ``None``
    symmetric(bool, optional): indicates whether :attr:`input` is symmetric.
                               Default: ``False``

Example::

    >>> a = torch.eye(10)
    >>> torch.matrix_rank(a)
    tensor(10)
    >>> b = torch.eye(10)
    >>> b[0, 0] = 0
    >>> torch.matrix_rank(b)
    tensor(9)
""")

add_docstr(torch.matrix_power,
           r"""
matrix_power(input, n) -> Tensor

Returns the matrix raised to the power :attr:`n` for square matrices.
For batch of matrices, each individual matrix is raised to the power :attr:`n`.

If :attr:`n` is negative, then the inverse of the matrix (if invertible) is
raised to the power :attr:`n`.  For a batch of matrices, the batched inverse
(if invertible) is raised to the power :attr:`n`. If :attr:`n` is 0, then an identity matrix
is returned.

Args:
    {input}
    n (int): the power to raise the matrix to

Example::

    >>> a = torch.randn(2, 2, 2)
    >>> a
    tensor([[[-1.9975, -1.9610],
             [ 0.9592, -2.3364]],

            [[-1.2534, -1.3429],
             [ 0.4153, -1.4664]]])
    >>> torch.matrix_power(a, 3)
    tensor([[[  3.9392, -23.9916],
             [ 11.7357,  -0.2070]],

            [[  0.2468,  -6.7168],
             [  2.0774,  -0.8187]]])
""".format(**common_args))

add_docstr(torch.max,
           r"""
.. function:: max(input) -> Tensor

Returns the maximum value of all elements in the :attr:`input` tensor.

Args:
    {input}

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6763,  0.7445, -2.2369]])
    >>> torch.max(a)
    tensor(0.7445)

.. function:: max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the maximum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each maximum value found
(argmax).

If :attr:`keepdim` is ``True``, the output tensors are of the same size
as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having 1 fewer dimension than :attr:`input`.

Args:
    {input}
    {dim}
    {keepdim} Default: ``False``.
    out (tuple, optional): the result tuple of two output tensors (max, max_indices)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
            [ 1.1949, -1.1127, -2.2379, -0.6702],
            [ 1.5717, -0.9207,  0.1297, -1.8768],
            [-0.6172,  1.0036, -0.6060, -0.2432]])
    >>> torch.max(a, 1)
    torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))

.. function:: max(input, other, out=None) -> Tensor

Each element of the tensor :attr:`input` is compared with the corresponding
element of the tensor :attr:`other` and an element-wise maximum is taken.

The shapes of :attr:`input` and :attr:`other` don't need to match,
but they must be :ref:`broadcastable <broadcasting-semantics>`.

.. math::
    \text{out}_i = \max(\text{tensor}_i, \text{other}_i)
""" + r"""
.. note:: When the shapes do not match, the shape of the returned output tensor
          follows the :ref:`broadcasting rules <broadcasting-semantics>`.

Args:
    {input}
    other (Tensor): the second input tensor
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.2942, -0.7416,  0.2653, -0.1584])
    >>> b = torch.randn(4)
    >>> b
    tensor([ 0.8722, -1.7421, -0.4141, -0.5055])
    >>> torch.max(a, b)
    tensor([ 0.8722, -0.7416,  0.2653, -0.1584])
""".format(**single_dim_common))

add_docstr(torch.argmax,
           r"""
.. function:: argmax(input) -> LongTensor

Returns the indices of the maximum value of all elements in the :attr:`input` tensor.

This is the second value returned by :meth:`torch.max`. See its
documentation for the exact semantics of this method.

Args:
    {input}

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a)
    tensor(0)

.. function:: argmax(input, dim, keepdim=False) -> LongTensor

Returns the indices of the maximum values of a tensor across a dimension.

This is the second value returned by :meth:`torch.max`. See its
documentation for the exact semantics of this method.

Args:
    {input}
    {dim} If ``None``, the argmax of the flattened input is returned.
    {keepdim} Ignored if ``dim=None``.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [ 0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092,  0.5419, -0.2993,  0.3195]])
    >>> torch.argmax(a, dim=1)
    tensor([ 0,  2,  0,  1])
""".format(**single_dim_common))

add_docstr(torch.mean,
           r"""
.. function:: mean(input) -> Tensor

Returns the mean value of all elements in the :attr:`input` tensor.

Args:
    {input}

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.2294, -0.5481,  1.3288]])
    >>> torch.mean(a)
    tensor(0.3367)

.. function:: mean(input, dim, keepdim=False, out=None) -> Tensor

Returns the mean value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.

{keepdim_details}

Args:
    {input}
    {dim}
    {keepdim}
    {out}

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
            [-0.9644,  1.0131, -0.6549, -1.4279],
            [-0.2951, -1.3350, -0.7694,  0.5600],
            [ 1.0842, -0.9580,  0.3623,  0.2343]])
    >>> torch.mean(a, 1)
    tensor([-0.0163, -0.5085, -0.4599,  0.1807])
    >>> torch.mean(a, 1, True)
    tensor([[-0.0163],
            [-0.5085],
            [-0.4599],
            [ 0.1807]])
""".format(**multi_dim_common))

add_docstr(torch.median,
           r"""
.. function:: median(input) -> Tensor

Returns the median value of all elements in the :attr:`input` tensor.

Args:
    {input}

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 1.5219, -1.5212,  0.2202]])
    >>> torch.median(a)
    tensor(0.2202)

.. function:: median(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the median
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each median value found.

By default, :attr:`dim` is the last dimension of the :attr:`input` tensor.

If :attr:`keepdim` is ``True``, the output tensors are of the same size
as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the outputs tensor having 1 fewer dimension than :attr:`input`.

Args:
    {input}
    {dim}
    {keepdim}
    values (Tensor, optional): the output tensor
    indices (Tensor, optional): the output index tensor

Example::

    >>> a = torch.randn(4, 5)
    >>> a
    tensor([[ 0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
            [ 0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
            [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
            [ 1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
    >>> torch.median(a, 1)
    torch.return_types.median(values=tensor([-0.3982,  0.2270,  0.2488,  0.4742]), indices=tensor([1, 4, 4, 3]))
""".format(**single_dim_common))

add_docstr(torch.min,
           r"""
.. function:: min(input) -> Tensor

Returns the minimum value of all elements in the :attr:`input` tensor.

Args:
    {input}

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.6750,  1.0857,  1.7197]])
    >>> torch.min(a)
    tensor(0.6750)

.. function:: min(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the minimum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each minimum value found
(argmin).

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensors having 1 fewer dimension than :attr:`input`.

Args:
    {input}
    {dim}
    {keepdim}
    out (tuple, optional): the tuple of two output tensors (min, min_indices)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.6248,  1.1334, -1.1899, -0.2803],
            [-1.4644, -0.2635, -0.3651,  0.6134],
            [ 0.2457,  0.0384,  1.0128,  0.7015],
            [-0.1153,  2.9849,  2.1458,  0.5788]])
    >>> torch.min(a, 1)
    torch.return_types.min(values=tensor([-1.1899, -1.4644,  0.0384, -0.1153]), indices=tensor([2, 0, 1, 0]))

.. function:: min(input, other, out=None) -> Tensor

Each element of the tensor :attr:`input` is compared with the corresponding
element of the tensor :attr:`other` and an element-wise minimum is taken.
The resulting tensor is returned.

The shapes of :attr:`input` and :attr:`other` don't need to match,
but they must be :ref:`broadcastable <broadcasting-semantics>`.

.. math::
    \text{out}_i = \min(\text{tensor}_i, \text{other}_i)
""" + r"""
.. note:: When the shapes do not match, the shape of the returned output tensor
          follows the :ref:`broadcasting rules <broadcasting-semantics>`.

Args:
    {input}
    other (Tensor): the second input tensor
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8137, -1.1740, -0.6460,  0.6308])
    >>> b = torch.randn(4)
    >>> b
    tensor([-0.1369,  0.1555,  0.4019, -0.1929])
    >>> torch.min(a, b)
    tensor([-0.1369, -1.1740, -0.6460, -0.1929])
""".format(**single_dim_common))

add_docstr(torch.argmin,
           r"""
.. function:: argmin(input) -> LongTensor

Returns the indices of the minimum value of all elements in the :attr:`input` tensor.

This is the second value returned by :meth:`torch.min`. See its
documentation for the exact semantics of this method.

Args:
    {input}

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
            [ 1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240,  0.1207, -0.7506, -1.0213],
            [ 1.7809, -1.2960,  0.9384,  0.1438]])
    >>> torch.argmin(a)
    tensor(13)

.. function:: argmin(input, dim, keepdim=False, out=None) -> LongTensor

Returns the indices of the minimum values of a tensor across a dimension.

This is the second value returned by :meth:`torch.min`. See its
documentation for the exact semantics of this method.

Args:
    {input}
    {dim} If ``None``, the argmin of the flattened input is returned.
    {keepdim} Ignored if ``dim=None``.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
            [ 1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240,  0.1207, -0.7506, -1.0213],
            [ 1.7809, -1.2960,  0.9384,  0.1438]])
    >>> torch.argmin(a, dim=1)
    tensor([ 2,  1,  3,  1])
""".format(**single_dim_common))

add_docstr(torch.mm,
           r"""
mm(input, mat2, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.

If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
:math:`(m \times p)` tensor, :attr:`out` will be a :math:`(n \times p)` tensor.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Args:
    input (Tensor): the first matrix to be multiplied
    mat2 (Tensor): the second matrix to be multiplied
    {out}

Example::

    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.mm(mat1, mat2)
    tensor([[ 0.4851,  0.5037, -0.3633],
            [-0.0760, -3.6705,  2.4784]])
""".format(**common_args))

add_docstr(torch.matmul,
           r"""
matmul(input, other, out=None) -> Tensor

Matrix product of two tensors.

The behavior depends on the dimensionality of the tensors as follows:

- If both tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
  The non-matrix (i.e. batch) dimensions are :ref:`broadcasted <broadcasting-semantics>` (and thus
  must be broadcastable).  For example, if :attr:`input` is a
  :math:`(j \times 1 \times n \times m)` tensor and :attr:`other` is a :math:`(k \times m \times p)`
  tensor, :attr:`out` will be an :math:`(j \times k \times n \times p)` tensor.

.. note::

    The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.

Arguments:
    input (Tensor): the first tensor to be multiplied
    other (Tensor): the second tensor to be multiplied
    {out}

Example::

    >>> # vector x vector
    >>> tensor1 = torch.randn(3)
    >>> tensor2 = torch.randn(3)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([])
    >>> # matrix x vector
    >>> tensor1 = torch.randn(3, 4)
    >>> tensor2 = torch.randn(4)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([3])
    >>> # batched matrix x broadcasted vector
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(4)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3])
    >>> # batched matrix x batched matrix
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(10, 4, 5)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3, 5])
    >>> # batched matrix x broadcasted matrix
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(4, 5)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3, 5])

""".format(**common_args))

add_docstr(torch.mode,
           r"""
mode(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)

Returns a namedtuple ``(values, indices)`` where ``values`` is the mode
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`, i.e. a value which appears most often
in that row, and ``indices`` is the index location of each mode value found.

By default, :attr:`dim` is the last dimension of the :attr:`input` tensor.

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having 1 fewer dimension than :attr:`input`.

.. note:: This function is not defined for ``torch.cuda.Tensor`` yet.

Args:
    {input}
    {dim}
    {keepdim}
    values (Tensor, optional): the output tensor
    indices (Tensor, optional): the output index tensor

Example::

    >>> a = torch.randint(10, (5,))
    >>> a
    tensor([6, 5, 1, 0, 2])
    >>> b = a + (torch.randn(50, 1) * 5).long()
    >>> torch.mode(b, 0)
    torch.return_types.mode(values=tensor([6, 5, 1, 0, 2]), indices=tensor([2, 2, 2, 2, 2]))
""".format(**single_dim_common))

add_docstr(torch.mul,
           r"""
.. function:: mul(input, other, out=None)

Multiplies each element of the input :attr:`input` with the scalar
:attr:`other` and returns a new resulting tensor.

.. math::
    \text{out}_i = \text{other} \times \text{input}_i
""" + r"""
If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`other`
should be a real number, otherwise it should be an integer

Args:
    {input}
    value (Number): the number to be multiplied to each element of :attr:`input`
    {out}

Example::

    >>> a = torch.randn(3)
    >>> a
    tensor([ 0.2015, -0.4255,  2.6087])
    >>> torch.mul(a, 100)
    tensor([  20.1494,  -42.5491,  260.8663])

.. function:: mul(input, other, out=None)

Each element of the tensor :attr:`input` is multiplied by the corresponding
element of the Tensor :attr:`other`. The resulting tensor is returned.

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

.. math::
    \text{out}_i = \text{input}_i \times \text{other}_i
""" + r"""
Args:
    input (Tensor): the first multiplicand tensor
    other (Tensor): the second multiplicand tensor
    {out}

Example::

    >>> a = torch.randn(4, 1)
    >>> a
    tensor([[ 1.1207],
            [-0.3137],
            [ 0.0700],
            [ 0.8378]])
    >>> b = torch.randn(1, 4)
    >>> b
    tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
    >>> torch.mul(a, b)
    tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
            [-0.1614, -0.0382,  0.1645, -0.7021],
            [ 0.0360,  0.0085, -0.0367,  0.1567],
            [ 0.4312,  0.1019, -0.4394,  1.8753]])
""".format(**common_args))

add_docstr(torch.multinomial,
           r"""
multinomial(input, num_samples, replacement=False, out=None) -> LongTensor

Returns a tensor where each row contains :attr:`num_samples` indices sampled
from the multinomial probability distribution located in the corresponding row
of tensor :attr:`input`.

.. note::
    The rows of :attr:`input` do not need to sum to one (in which case we use
    the values as weights), but must be non-negative, finite and have
    a non-zero sum.

Indices are ordered from left to right according to when each was sampled
(first samples are placed in first column).

If :attr:`input` is a vector, :attr:`out` is a vector of size :attr:`num_samples`.

If :attr:`input` is a matrix with `m` rows, :attr:`out` is an matrix of shape
:math:`(m \times \text{{num\_samples}})`.

If replacement is ``True``, samples are drawn with replacement.

If not, they are drawn without replacement, which means that when a
sample index is drawn for a row, it cannot be drawn again for that row.

.. note::
    When drawn without replacement, :attr:`num_samples` must be lower than
    number of non-zero elements in :attr:`input` (or the min number of non-zero
    elements in each row of :attr:`input` if it is a matrix).

Args:
    input (Tensor): the input tensor containing probabilities
    num_samples (int): number of samples to draw
    replacement (bool, optional): whether to draw with replacement or not
    {out}

Example::

    >>> weights = torch.tensor([0, 10, 3, 0], dtype=torch.float) # create a tensor of weights
    >>> torch.multinomial(weights, 2)
    tensor([1, 2])
    >>> torch.multinomial(weights, 4) # ERROR!
    RuntimeError: invalid argument 2: invalid multinomial distribution (with replacement=False,
    not enough non-negative category to sample) at ../aten/src/TH/generic/THTensorRandom.cpp:320
    >>> torch.multinomial(weights, 4, replacement=True)
    tensor([ 2,  1,  1,  1])
""".format(**common_args))

add_docstr(torch.mv,
           r"""
mv(input, vec, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`input` and the vector
:attr:`vec`.

If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
size :math:`m`, :attr:`out` will be 1-D of size :math:`n`.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

Args:
    input (Tensor): matrix to be multiplied
    vec (Tensor): vector to be multiplied
    {out}

Example::

    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.mv(mat, vec)
    tensor([ 1.0404, -0.6361])
""".format(**common_args))

add_docstr(torch.mvlgamma,
           r"""
mvlgamma(input, p) -> Tensor

Computes the multivariate log-gamma function (`[reference]`_) with dimension :math:`p` element-wise, given by

.. math::
    \log(\Gamma_{p}(a)) = C + \displaystyle \sum_{i=1}^{p} \log\left(\Gamma\left(a - \frac{i - 1}{2}\right)\right)

where :math:`C = \log(\pi) \times \frac{p (p - 1)}{4}` and :math:`\Gamma(\cdot)` is the Gamma function.

If any of the elements are less than or equal to :math:`\frac{p - 1}{2}`, then an error
is thrown.

Args:
    input (Tensor): the tensor to compute the multivariate log-gamma function
    p (int): the number of dimensions

Example::

    >>> a = torch.empty(2, 3).uniform_(1, 2)
    >>> a
    tensor([[1.6835, 1.8474, 1.1929],
            [1.0475, 1.7162, 1.4180]])
    >>> torch.mvlgamma(a, 2)
    tensor([[0.3928, 0.4007, 0.7586],
            [1.0311, 0.3901, 0.5049]])

.. _`[reference]`: https://en.wikipedia.org/wiki/Multivariate_gamma_function
""")

add_docstr(torch.narrow,
           r"""
narrow(input, dim, start, length) -> Tensor

Returns a new tensor that is a narrowed version of :attr:`input` tensor. The
dimension :attr:`dim` is input from :attr:`start` to :attr:`start + length`. The
returned tensor and :attr:`input` tensor share the same underlying storage.

Args:
    input (Tensor): the tensor to narrow
    dim (int): the dimension along which to narrow
    start (int): the starting dimension
    length (int): the distance to the ending dimension

Example::

    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> torch.narrow(x, 0, 0, 2)
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])
    >>> torch.narrow(x, 1, 1, 2)
    tensor([[ 2,  3],
            [ 5,  6],
            [ 8,  9]])
""")

add_docstr(torch.ne,
           r"""
ne(input, other, out=None) -> Tensor

Computes :math:`input \neq other` element-wise.

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare
    out (Tensor, optional): the output tensor that must be a `BoolTensor`

Returns:
    Tensor: A ``torch.BoolTensor`` containing a True at each location where comparison is true.

Example::

    >>> torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [True, False]])
""")

add_docstr(torch.neg,
           r"""
neg(input, out=None) -> Tensor

Returns a new tensor with the negative of the elements of :attr:`input`.

.. math::
    \text{out} = -1 \times \text{input}
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
    >>> torch.neg(a)
    tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])
""".format(**common_args))

add_docstr(torch.nonzero,
           r"""
nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors

**When** :attr:`as_tuple` **is false or unspecified:**

Returns a tensor containing the indices of all non-zero elements of
:attr:`input`.  Each row in the result contains the indices of a non-zero
element in :attr:`input`. The result is sorted lexicographically, with
the last index changing the fastest (C-style).

If :attr:`input` has `n` dimensions, then the resulting indices tensor
:attr:`out` is of size :math:`(z \times n)`, where :math:`z` is the total number of
non-zero elements in the :attr:`input` tensor.

**When** :attr:`as_tuple` **is true:**

Returns a tuple of 1-D tensors, one for each dimension in :attr:`input`,
each containing the indices (in that dimension) of all non-zero elements of
:attr:`input` .

If :attr:`input` has `n` dimensions, then the resulting tuple contains `n` tensors
of size `z`, where `z` is the total number of
non-zero elements in the :attr:`input` tensor.

As a special case, when :attr:`input` has zero dimensions and a nonzero scalar
value, it is treated as a one-dimensional tensor with one element.

Args:
    {input}
    out (LongTensor, optional): the output tensor containing indices

Returns:
    LongTensor or tuple of LongTensor: If :attr:`as_tuple` is false, the output
    tensor containing indices. If :attr:`as_tuple` is true, one 1-D tensor for
    each dimension, containing the indices of each nonzero element along that
    dimension.

Example::

    >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
    tensor([[ 0],
            [ 1],
            [ 2],
            [ 4]])
    >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]]))
    tensor([[ 0,  0],
            [ 1,  1],
            [ 2,  2],
            [ 3,  3]])
    >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
    (tensor([0, 1, 2, 4]),)
    >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
    (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
    >>> torch.nonzero(torch.tensor(5), as_tuple=True)
    (tensor([0]),)
""".format(**common_args))

add_docstr(torch.normal,
           r"""
.. function:: normal(mean, std, out=None) -> Tensor

Returns a tensor of random numbers drawn from separate normal distributions
whose mean and standard deviation are given.

The :attr:`mean` is a tensor with the mean of
each output element's normal distribution

The :attr:`std` is a tensor with the standard deviation of
each output element's normal distribution

The shapes of :attr:`mean` and :attr:`std` don't need to match, but the
total number of elements in each tensor need to be the same.

.. note:: When the shapes do not match, the shape of :attr:`mean`
          is used as the shape for the returned output tensor

Args:
    mean (Tensor): the tensor of per-element means
    std (Tensor): the tensor of per-element standard deviations
    {out}

Example::

    >>> torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
    tensor([  1.0425,   3.5672,   2.7969,   4.2925,   4.7229,   6.2134,
              8.0505,   8.1408,   9.0563,  10.0566])

.. function:: normal(mean=0.0, std, out=None) -> Tensor

Similar to the function above, but the means are shared among all drawn
elements.

Args:
    mean (float, optional): the mean for all distributions
    std (Tensor): the tensor of per-element standard deviations
    {out}

Example::

    >>> torch.normal(mean=0.5, std=torch.arange(1., 6.))
    tensor([-1.2793, -1.0732, -2.0687,  5.1177, -1.2303])

.. function:: normal(mean, std=1.0, out=None) -> Tensor

Similar to the function above, but the standard-deviations are shared among
all drawn elements.

Args:
    mean (Tensor): the tensor of per-element means
    std (float, optional): the standard deviation for all distributions
    out (Tensor, optional): the output tensor

Example::

    >>> torch.normal(mean=torch.arange(1., 6.))
    tensor([ 1.1552,  2.6148,  2.6535,  5.8318,  4.2361])

.. function:: normal(mean, std, size, *, out=None) -> Tensor

Similar to the function above, but the means and standard deviations are shared
among all drawn elements. The resulting tensor has size given by :attr:`size`.

Args:
    mean (float): the mean for all distributions
    std (float): the standard deviation for all distributions
    size (int...): a sequence of integers defining the shape of the output tensor.
    {out}

Example::

    >>> torch.normal(2, 3, size=(1, 4))
    tensor([[-1.3987, -1.9544,  3.6048,  0.7909]])
""".format(**common_args))

add_docstr(torch.numel,
           r"""
numel(input) -> int

Returns the total number of elements in the :attr:`input` tensor.

Args:
    {input}

Example::

    >>> a = torch.randn(1, 2, 3, 4, 5)
    >>> torch.numel(a)
    120
    >>> a = torch.zeros(4,4)
    >>> torch.numel(a)
    16

""".format(**common_args))

add_docstr(torch.ones,
           r"""
ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with the scalar value `1`, with the shape defined
by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.ones(2, 3)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])

    >>> torch.ones(5)
    tensor([ 1.,  1.,  1.,  1.,  1.])

""".format(**factory_common_args))

add_docstr(torch.ones_like,
           r"""
ones_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with the scalar value `1`, with the same size as
:attr:`input`. ``torch.ones_like(input)`` is equivalent to
``torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

.. warning::
    As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
    the old ``torch.ones_like(input, out=output)`` is equivalent to
    ``torch.ones(input.size(), out=output)``.

Args:
    {input}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> input = torch.empty(2, 3)
    >>> torch.ones_like(input)
    tensor([[ 1.,  1.,  1.],
            [ 1.,  1.,  1.]])
""".format(**factory_like_common_args))

add_docstr(torch.orgqr,
           r"""
orgqr(input, input2) -> Tensor

Computes the orthogonal matrix `Q` of a QR factorization, from the `(input, input2)`
tuple returned by :func:`torch.geqrf`.

This directly calls the underlying LAPACK function `?orgqr`.
See `LAPACK documentation for orgqr`_ for further details.

Args:
    input (Tensor): the `a` from :func:`torch.geqrf`.
    input2 (Tensor): the `tau` from :func:`torch.geqrf`.

.. _LAPACK documentation for orgqr:
    https://software.intel.com/en-us/mkl-developer-reference-c-orgqr

""")

add_docstr(torch.ormqr,
           r"""
ormqr(input, input2, input3, left=True, transpose=False) -> Tensor

Multiplies `mat` (given by :attr:`input3`) by the orthogonal `Q` matrix of the QR factorization
formed by :func:`torch.geqrf` that is represented by `(a, tau)` (given by (:attr:`input`, :attr:`input2`)).

This directly calls the underlying LAPACK function `?ormqr`.
See `LAPACK documentation for ormqr`_ for further details.

Args:
    input (Tensor): the `a` from :func:`torch.geqrf`.
    input2 (Tensor): the `tau` from :func:`torch.geqrf`.
    input3 (Tensor): the matrix to be multiplied.

.. _LAPACK documentation for ormqr:
    https://software.intel.com/en-us/mkl-developer-reference-c-ormqr

""")

add_docstr(torch.pow,
           r"""
.. function:: pow(input, exponent, out=None) -> Tensor

Takes the power of each element in :attr:`input` with :attr:`exponent` and
returns a tensor with the result.

:attr:`exponent` can be either a single ``float`` number or a `Tensor`
with the same number of elements as :attr:`input`.

When :attr:`exponent` is a scalar value, the operation applied is:

.. math::
    \text{out}_i = x_i ^ \text{exponent}

When :attr:`exponent` is a tensor, the operation applied is:

.. math::
    \text{out}_i = x_i ^ {\text{exponent}_i}
""" + r"""
When :attr:`exponent` is a tensor, the shapes of :attr:`input`
and :attr:`exponent` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    {input}
    exponent (float or tensor): the exponent value
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
    >>> torch.pow(a, 2)
    tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
    >>> exp = torch.arange(1., 5.)

    >>> a = torch.arange(1., 5.)
    >>> a
    tensor([ 1.,  2.,  3.,  4.])
    >>> exp
    tensor([ 1.,  2.,  3.,  4.])
    >>> torch.pow(a, exp)
    tensor([   1.,    4.,   27.,  256.])

.. function:: pow(self, exponent, out=None) -> Tensor

:attr:`self` is a scalar ``float`` value, and :attr:`exponent` is a tensor.
The returned tensor :attr:`out` is of the same shape as :attr:`exponent`

The operation applied is:

.. math::
    \text{{out}}_i = \text{{self}} ^ {{\text{{exponent}}_i}}

Args:
    self (float): the scalar base value for the power operation
    exponent (Tensor): the exponent tensor
    {out}

Example::

    >>> exp = torch.arange(1., 5.)
    >>> base = 2
    >>> torch.pow(base, exp)
    tensor([  2.,   4.,   8.,  16.])
""".format(**common_args))

add_docstr(torch.prod,
           r"""
.. function:: prod(input, dtype=None) -> Tensor

Returns the product of all elements in the :attr:`input` tensor.

Args:
    {input}
    {dtype}

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.8020,  0.5428, -1.5854]])
    >>> torch.prod(a)
    tensor(0.6902)

.. function:: prod(input, dim, keepdim=False, dtype=None) -> Tensor

Returns the product of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`.

{keepdim_details}

Args:
    {input}
    {dim}
    {keepdim}
    {dtype}

Example::

    >>> a = torch.randn(4, 2)
    >>> a
    tensor([[ 0.5261, -0.3837],
            [ 1.1857, -0.2498],
            [-1.1646,  0.0705],
            [ 1.1131, -1.0629]])
    >>> torch.prod(a, 1)
    tensor([-0.2018, -0.2962, -0.0821, -1.1831])
""".format(**single_dim_common))

add_docstr(torch.qr,
           r"""
qr(input, some=True, out=None) -> (Tensor, Tensor)

Computes the QR decomposition of a matrix or a batch of matrices :attr:`input`,
and returns a namedtuple (Q, R) of tensors such that :math:`\text{input} = Q R`
with :math:`Q` being an orthogonal matrix or batch of orthogonal matrices and
:math:`R` being an upper triangular matrix or batch of upper triangular matrices.

If :attr:`some` is ``True``, then this function returns the thin (reduced) QR factorization.
Otherwise, if :attr:`some` is ``False``, this function returns the complete QR factorization.

.. note:: precision may be lost if the magnitudes of the elements of :attr:`input`
          are large

.. note:: While it should always give you a valid decomposition, it may not
          give you the same one across platforms - it will depend on your
          LAPACK implementation.

Args:
    input (Tensor): the input tensor of size :math:`(*, m, n)` where `*` is zero or more
                batch dimensions consisting of matrices of dimension :math:`m \times n`.
    some (bool, optional): Set to ``True`` for reduced QR decomposition and ``False`` for
                complete QR decomposition.
    out (tuple, optional): tuple of `Q` and `R` tensors
                satisfying :code:`input = torch.matmul(Q, R)`.
                The dimensions of `Q` and `R` are :math:`(*, m, k)` and :math:`(*, k, n)`
                respectively, where :math:`k = \min(m, n)` if :attr:`some:` is ``True`` and
                :math:`k = m` otherwise.

Example::

    >>> a = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    >>> q, r = torch.qr(a)
    >>> q
    tensor([[-0.8571,  0.3943,  0.3314],
            [-0.4286, -0.9029, -0.0343],
            [ 0.2857, -0.1714,  0.9429]])
    >>> r
    tensor([[ -14.0000,  -21.0000,   14.0000],
            [   0.0000, -175.0000,   70.0000],
            [   0.0000,    0.0000,  -35.0000]])
    >>> torch.mm(q, r).round()
    tensor([[  12.,  -51.,    4.],
            [   6.,  167.,  -68.],
            [  -4.,   24.,  -41.]])
    >>> torch.mm(q.t(), q).round()
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1., -0.],
            [ 0., -0.,  1.]])
    >>> a = torch.randn(3, 4, 5)
    >>> q, r = torch.qr(a, some=False)
    >>> torch.allclose(torch.matmul(q, r), a)
    True
    >>> torch.allclose(torch.matmul(q.transpose(-2, -1), q), torch.eye(5))
    True
""")

add_docstr(torch.rand,
           r"""
rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random numbers from a uniform distribution
on the interval :math:`[0, 1)`

The shape of the tensor is defined by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.rand(4)
    tensor([ 0.5204,  0.2503,  0.3525,  0.5673])
    >>> torch.rand(2, 3)
    tensor([[ 0.8237,  0.5781,  0.6879],
            [ 0.3816,  0.7249,  0.0998]])
""".format(**factory_common_args))

add_docstr(torch.rand_like,
           r"""
rand_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor

Returns a tensor with the same size as :attr:`input` that is filled with
random numbers from a uniform distribution on the interval :math:`[0, 1)`.
``torch.rand_like(input)`` is equivalent to
``torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    {input}
    {dtype}
    {layout}
    {device}
    {requires_grad}

""".format(**factory_like_common_args))

add_docstr(torch.randint,
           r"""
randint(low=0, high, size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random integers generated uniformly
between :attr:`low` (inclusive) and :attr:`high` (exclusive).

The shape of the tensor is defined by the variable argument :attr:`size`.

.. note:
    With the global dtype default (``torch.float32``), this function returns
    a tensor with dtype ``torch.int64``.

Args:
    low (int, optional): Lowest integer to be drawn from the distribution. Default: 0.
    high (int): One above the highest integer to be drawn from the distribution.
    size (tuple): a tuple defining the shape of the output tensor.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.randint(3, 5, (3,))
    tensor([4, 3, 4])


    >>> torch.randint(10, (2, 2))
    tensor([[0, 2],
            [5, 5]])


    >>> torch.randint(3, 10, (2, 2))
    tensor([[4, 5],
            [6, 7]])


""".format(**factory_common_args))

add_docstr(torch.randint_like,
           r"""
randint_like(input, low=0, high, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor with the same shape as Tensor :attr:`input` filled with
random integers generated uniformly between :attr:`low` (inclusive) and
:attr:`high` (exclusive).

.. note:
    With the global dtype default (``torch.float32``), this function returns
    a tensor with dtype ``torch.int64``.

Args:
    {input}
    low (int, optional): Lowest integer to be drawn from the distribution. Default: 0.
    high (int): One above the highest integer to be drawn from the distribution.
    {dtype}
    {layout}
    {device}
    {requires_grad}

""".format(**factory_like_common_args))

add_docstr(torch.randn,
           r"""
randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random numbers from a normal distribution
with mean `0` and variance `1` (also called the standard normal
distribution).

.. math::
    \text{{out}}_{{i}} \sim \mathcal{{N}}(0, 1)

The shape of the tensor is defined by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.randn(4)
    tensor([-2.1436,  0.9966,  2.3426, -0.6366])
    >>> torch.randn(2, 3)
    tensor([[ 1.5954,  2.8929, -1.0923],
            [ 1.1719, -0.4709, -0.1996]])
""".format(**factory_common_args))

add_docstr(torch.randn_like,
           r"""
randn_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor

Returns a tensor with the same size as :attr:`input` that is filled with
random numbers from a normal distribution with mean 0 and variance 1.
``torch.randn_like(input)`` is equivalent to
``torch.randn(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    {input}
    {dtype}
    {layout}
    {device}
    {requires_grad}

""".format(**factory_like_common_args))

add_docstr(torch.randperm,
           r"""
randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) -> LongTensor

Returns a random permutation of integers from ``0`` to ``n - 1``.

Args:
    n (int): the upper bound (exclusive)
    {out}
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: ``torch.int64``.
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.randperm(4)
    tensor([2, 1, 0, 3])
""".format(**factory_common_args))

add_docstr(torch.tensor,
           r"""
tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Constructs a tensor with :attr:`data`.

.. warning::

    :func:`torch.tensor` always copies :attr:`data`. If you have a Tensor
    ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
    or :func:`torch.Tensor.detach`.
    If you have a NumPy ``ndarray`` and want to avoid a copy, use
    :func:`torch.as_tensor`.

.. warning::

    When data is a tensor `x`, :func:`torch.tensor` reads out 'the data' from whatever it is passed,
    and constructs a leaf variable. Therefore ``torch.tensor(x)`` is equivalent to ``x.clone().detach()``
    and ``torch.tensor(x, requires_grad=True)`` is equivalent to ``x.clone().detach().requires_grad_(True)``.
    The equivalents using ``clone()`` and ``detach()`` are recommended.

Args:
    {data}
    {dtype}
    {device}
    {requires_grad}
    {pin_memory}


Example::

    >>> torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    tensor([[ 0.1000,  1.2000],
            [ 2.2000,  3.1000],
            [ 4.9000,  5.2000]])

    >>> torch.tensor([0, 1])  # Type inference on data
    tensor([ 0,  1])

    >>> torch.tensor([[0.11111, 0.222222, 0.3333333]],
                     dtype=torch.float64,
                     device=torch.device('cuda:0'))  # creates a torch.cuda.DoubleTensor
    tensor([[ 0.1111,  0.2222,  0.3333]], dtype=torch.float64, device='cuda:0')

    >>> torch.tensor(3.14159)  # Create a scalar (zero-dimensional tensor)
    tensor(3.1416)

    >>> torch.tensor([])  # Create an empty tensor (of size (0,))
    tensor([])
""".format(**factory_data_common_args))

add_docstr(torch.range,
           r"""
range(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 1-D tensor of size :math:`\left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor + 1`
with values from :attr:`start` to :attr:`end` with step :attr:`step`. Step is
the gap between two values in the tensor.

.. math::
    \text{out}_{i+1} = \text{out}_i + \text{step}.
""" + r"""
.. warning::
    This function is deprecated in favor of :func:`torch.arange`.

Args:
    start (float): the starting value for the set of points. Default: ``0``.
    end (float): the ending value for the set of points
    step (float): the gap between each pair of adjacent points. Default: ``1``.
    {out}
    {dtype} If `dtype` is not given, infer the data type from the other input
        arguments. If any of `start`, `end`, or `stop` are floating-point, the
        `dtype` is inferred to be the default dtype, see
        :meth:`~torch.get_default_dtype`. Otherwise, the `dtype` is inferred to
        be `torch.int64`.
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.range(1, 4)
    tensor([ 1.,  2.,  3.,  4.])
    >>> torch.range(1, 4, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000,  2.5000,  3.0000,  3.5000,  4.0000])
""".format(**factory_common_args))

add_docstr(torch.arange,
           r"""
arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 1-D tensor of size :math:`\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil`
with values from the interval ``[start, end)`` taken with common difference
:attr:`step` beginning from `start`.

Note that non-integer :attr:`step` is subject to floating point rounding errors when
comparing against :attr:`end`; to avoid inconsistency, we advise adding a small epsilon to :attr:`end`
in such cases.

.. math::
    \text{out}_{{i+1}} = \text{out}_{i} + \text{step}
""" + r"""
Args:
    start (Number): the starting value for the set of points. Default: ``0``.
    end (Number): the ending value for the set of points
    step (Number): the gap between each pair of adjacent points. Default: ``1``.
    {out}
    {dtype} If `dtype` is not given, infer the data type from the other input
        arguments. If any of `start`, `end`, or `stop` are floating-point, the
        `dtype` is inferred to be the default dtype, see
        :meth:`~torch.get_default_dtype`. Otherwise, the `dtype` is inferred to
        be `torch.int64`.
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.arange(5)
    tensor([ 0,  1,  2,  3,  4])
    >>> torch.arange(1, 4)
    tensor([ 1,  2,  3])
    >>> torch.arange(1, 2.5, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000])
""".format(**factory_common_args))

add_docstr(torch.remainder,
           r"""
remainder(input, other, out=None) -> Tensor

Computes the element-wise remainder of division.

The divisor and dividend may contain both for integer and floating point
numbers. The remainder has the same sign as the divisor.

When :attr:`other` is a tensor, the shapes of :attr:`input` and
:attr:`other` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the dividend
    other (Tensor or float): the divisor that may be either a number or a
                               Tensor of the same shape as the dividend
    {out}

Example::

    >>> torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
    tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
    >>> torch.remainder(torch.tensor([1., 2, 3, 4, 5]), 1.5)
    tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])

.. seealso::

        :func:`torch.fmod`, which computes the element-wise remainder of
        division equivalently to the C library function ``fmod()``.
""".format(**common_args))

add_docstr(torch.renorm,
           r"""
renorm(input, p, dim, maxnorm, out=None) -> Tensor

Returns a tensor where each sub-tensor of :attr:`input` along dimension
:attr:`dim` is normalized such that the `p`-norm of the sub-tensor is lower
than the value :attr:`maxnorm`

.. note:: If the norm of a row is lower than `maxnorm`, the row is unchanged

Args:
    {input}
    p (float): the power for the norm computation
    dim (int): the dimension to slice over to get the sub-tensors
    maxnorm (float): the maximum norm to keep each sub-tensor under
    {out}

Example::

    >>> x = torch.ones(3, 3)
    >>> x[1].fill_(2)
    tensor([ 2.,  2.,  2.])
    >>> x[2].fill_(3)
    tensor([ 3.,  3.,  3.])
    >>> x
    tensor([[ 1.,  1.,  1.],
            [ 2.,  2.,  2.],
            [ 3.,  3.,  3.]])
    >>> torch.renorm(x, 1, 0, 5)
    tensor([[ 1.0000,  1.0000,  1.0000],
            [ 1.6667,  1.6667,  1.6667],
            [ 1.6667,  1.6667,  1.6667]])
""".format(**common_args))

add_docstr(torch.reshape,
           r"""
reshape(input, shape) -> Tensor

Returns a tensor with the same data and number of elements as :attr:`input`,
but with the specified shape. When possible, the returned tensor will be a view
of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and inputs
with compatible strides can be reshaped without copying, but you should not
depend on the copying vs. viewing behavior.

See :meth:`torch.Tensor.view` on when it is possible to return a view.

A single dimension may be -1, in which case it's inferred from the remaining
dimensions and the number of elements in :attr:`input`.

Args:
    input (Tensor): the tensor to be reshaped
    shape (tuple of ints): the new shape

Example::

    >>> a = torch.arange(4.)
    >>> torch.reshape(a, (2, 2))
    tensor([[ 0.,  1.],
            [ 2.,  3.]])
    >>> b = torch.tensor([[0, 1], [2, 3]])
    >>> torch.reshape(b, (-1,))
    tensor([ 0,  1,  2,  3])
""")


add_docstr(torch.result_type,
           r"""
result_type(tensor1, tensor2) -> dtype

Returns the :class:`torch.dtype` that would result from performing an arithmetic
operation on the provided input tensors. See type promotion :ref:`documentation <type-promotion-doc>`
for more information on the type promotion logic.

Args:
    tensor1 (Tensor or Number): an input tensor or number
    tensor2 (Tensor or Number): an input tensor or number

Example::

    >>> torch.result_type(torch.tensor([1, 2], dtype=torch.int), 1.0)
    torch.float32
    >>> torch.result_type(torch.tensor([1, 2], dtype=torch.uint8), torch.tensor(1))
    torch.uint8
""")


add_docstr(torch.round,
           r"""
round(input, out=None) -> Tensor

Returns a new tensor with each of the elements of :attr:`input` rounded
to the closest integer.

Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.9920,  0.6077,  0.9734, -1.0362])
    >>> torch.round(a)
    tensor([ 1.,  1.,  1., -1.])
""".format(**common_args))

add_docstr(torch.rsqrt,
           r"""
rsqrt(input, out=None) -> Tensor

Returns a new tensor with the reciprocal of the square-root of each of
the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.0370,  0.2970,  1.5420, -0.9105])
    >>> torch.rsqrt(a)
    tensor([    nan,  1.8351,  0.8053,     nan])
""".format(**common_args))

add_docstr(torch.set_flush_denormal,
           r"""
set_flush_denormal(mode) -> bool

Disables denormal floating numbers on CPU.

Returns ``True`` if your system supports flushing denormal numbers and it
successfully configures flush denormal mode.  :meth:`~torch.set_flush_denormal`
is only supported on x86 architectures supporting SSE3.

Args:
    mode (bool): Controls whether to enable flush denormal mode or not

Example::

    >>> torch.set_flush_denormal(True)
    True
    >>> torch.tensor([1e-323], dtype=torch.float64)
    tensor([ 0.], dtype=torch.float64)
    >>> torch.set_flush_denormal(False)
    True
    >>> torch.tensor([1e-323], dtype=torch.float64)
    tensor(9.88131e-324 *
           [ 1.0000], dtype=torch.float64)
""")

add_docstr(torch.set_num_threads,
           r"""
set_num_threads(int)

Sets the number of threads used for parallelizing CPU operations.
WARNING:
To ensure that the correct number of threads is used, set_num_threads
must be called before running eager, JIT or autograd code.
""")

add_docstr(torch.set_num_interop_threads,
           r"""
set_num_interop_threads(int)

Sets the number of threads used for interop parallelism
(e.g. in JIT interpreter) on CPU.
WARNING: Can only be called once and before any inter-op parallel work
is started (e.g. JIT execution).
""")

add_docstr(torch.sigmoid,
           r"""
sigmoid(input, out=None) -> Tensor

Returns a new tensor with the sigmoid of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{1 + e^{-\text{input}_{i}}}
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.9213,  1.0887, -0.8858, -1.7683])
    >>> torch.sigmoid(a)
    tensor([ 0.7153,  0.7481,  0.2920,  0.1458])
""".format(**common_args))

add_docstr(torch.sign,
           r"""
sign(input, out=None) -> Tensor

Returns a new tensor with the signs of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \operatorname{sgn}(\text{input}_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.tensor([0.7, -1.2, 0., 2.3])
    >>> a
    tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
    >>> torch.sign(a)
    tensor([ 1., -1.,  0.,  1.])
""".format(**common_args))

add_docstr(torch.sin,
           r"""
sin(input, out=None) -> Tensor

Returns a new tensor with the sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin(\text{input}_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5461,  0.1347, -2.7266, -0.2746])
    >>> torch.sin(a)
    tensor([-0.5194,  0.1343, -0.4032, -0.2711])
""".format(**common_args))

add_docstr(torch.sinh,
           r"""
sinh(input, out=None) -> Tensor

Returns a new tensor with the hyperbolic sine of the elements of
:attr:`input`.

.. math::
    \text{out}_{i} = \sinh(\text{input}_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.5380, -0.8632, -0.1265,  0.9399])
    >>> torch.sinh(a)
    tensor([ 0.5644, -0.9744, -0.1268,  1.0845])
""".format(**common_args))

add_docstr(torch.sort,
           r"""
sort(input, dim=-1, descending=False, out=None) -> (Tensor, LongTensor)

Sorts the elements of the :attr:`input` tensor along a given dimension
in ascending order by value.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`descending` is ``True`` then the elements are sorted in descending
order by value.

A namedtuple of (values, indices) is returned, where the `values` are the
sorted values and `indices` are the indices of the elements in the original
`input` tensor.

Args:
    {input}
    dim (int, optional): the dimension to sort along
    descending (bool, optional): controls the sorting order (ascending or descending)
    out (tuple, optional): the output tuple of (`Tensor`, `LongTensor`) that can
        be optionally given to be used as output buffers

Example::

    >>> x = torch.randn(3, 4)
    >>> sorted, indices = torch.sort(x)
    >>> sorted
    tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
            [-0.5793,  0.0061,  0.6058,  0.9497],
            [-0.5071,  0.3343,  0.9553,  1.0960]])
    >>> indices
    tensor([[ 1,  0,  2,  3],
            [ 3,  1,  0,  2],
            [ 0,  3,  1,  2]])

    >>> sorted, indices = torch.sort(x, 0)
    >>> sorted
    tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
            [ 0.0608,  0.0061,  0.9497,  0.3343],
            [ 0.6058,  0.9553,  1.0960,  2.3332]])
    >>> indices
    tensor([[ 2,  0,  0,  1],
            [ 0,  1,  1,  2],
            [ 1,  2,  2,  0]])
""".format(**common_args))

add_docstr(torch.argsort,
           r"""
argsort(input, dim=-1, descending=False, out=None) -> LongTensor

Returns the indices that sort a tensor along a given dimension in ascending
order by value.

This is the second value returned by :meth:`torch.sort`.  See its documentation
for the exact semantics of this method.

Args:
    {input}
    dim (int, optional): the dimension to sort along
    descending (bool, optional): controls the sorting order (ascending or descending)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
            [ 0.1598,  0.0788, -0.0745, -1.2700],
            [ 1.2208,  1.0722, -0.7064,  1.2564],
            [ 0.0669, -0.2318, -0.8229, -0.9280]])


    >>> torch.argsort(a, dim=1)
    tensor([[2, 0, 3, 1],
            [3, 2, 1, 0],
            [2, 1, 0, 3],
            [3, 2, 1, 0]])
""".format(**common_args))

add_docstr(torch.sparse_coo_tensor,
           r"""
sparse_coo_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False) -> Tensor

Constructs a sparse tensors in COO(rdinate) format with non-zero elements at the given :attr:`indices`
with the given :attr:`values`. A sparse tensor can be `uncoalesced`, in that case, there are duplicate
coordinates in the indices, and the value at that index is the sum of all duplicate value entries:
`torch.sparse`_.

Args:
    indices (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types. Will be cast to a :class:`torch.LongTensor`
        internally. The indices are the coordinates of the non-zero values in the matrix, and thus
        should be two-dimensional where the first dimension is the number of tensor dimensions and
        the second dimension is the number of non-zero values.
    values (array_like): Initial values for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.
    size (list, tuple, or :class:`torch.Size`, optional): Size of the sparse tensor. If not
        provided the size will be inferred as the minimum size big enough to hold all non-zero
        elements.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if None, infers data type from :attr:`values`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    {requires_grad}


Example::

    >>> i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]])
    >>> v = torch.tensor([3, 4, 5], dtype=torch.float32)
    >>> torch.sparse_coo_tensor(i, v, [2, 4])
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3., 4., 5.]),
           size=(2, 4), nnz=3, layout=torch.sparse_coo)

    >>> torch.sparse_coo_tensor(i, v)  # Shape inference
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3., 4., 5.]),
           size=(2, 3), nnz=3, layout=torch.sparse_coo)

    >>> torch.sparse_coo_tensor(i, v, [2, 4],
                                dtype=torch.float64,
                                device=torch.device('cuda:0'))
    tensor(indices=tensor([[0, 1, 1],
                           [2, 0, 2]]),
           values=tensor([3., 4., 5.]),
           device='cuda:0', size=(2, 4), nnz=3, dtype=torch.float64,
           layout=torch.sparse_coo)

    # Create an empty sparse tensor with the following invariants:
    #   1. sparse_dim + dense_dim = len(SparseTensor.shape)
    #   2. SparseTensor._indices().shape = (sparse_dim, nnz)
    #   3. SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])
    #
    # For instance, to create an empty sparse tensor with nnz = 0, dense_dim = 0 and
    # sparse_dim = 1 (hence indices is a 2D tensor of shape = (1, 0))
    >>> S = torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1])
    tensor(indices=tensor([], size=(1, 0)),
           values=tensor([], size=(0,)),
           size=(1,), nnz=0, layout=torch.sparse_coo)

    # and to create an empty sparse tensor with nnz = 0, dense_dim = 1 and
    # sparse_dim = 1
    >>> S = torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, 2]), [1, 2])
    tensor(indices=tensor([], size=(1, 0)),
           values=tensor([], size=(0, 2)),
           size=(1, 2), nnz=0, layout=torch.sparse_coo)

.. _torch.sparse: https://pytorch.org/docs/stable/sparse.html
""".format(**factory_common_args))

add_docstr(torch.sqrt,
           r"""
sqrt(input, out=None) -> Tensor

Returns a new tensor with the square-root of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sqrt{\text{input}_{i}}
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-2.0755,  1.0226,  0.0831,  0.4806])
    >>> torch.sqrt(a)
    tensor([    nan,  1.0112,  0.2883,  0.6933])
""".format(**common_args))

add_docstr(torch.squeeze,
           r"""
squeeze(input, dim=None, out=None) -> Tensor

Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.

For example, if `input` is of shape:
:math:`(A \times 1 \times B \times C \times 1 \times D)` then the `out` tensor
will be of shape: :math:`(A \times B \times C \times D)`.

When :attr:`dim` is given, a squeeze operation is done only in the given
dimension. If `input` is of shape: :math:`(A \times 1 \times B)`,
``squeeze(input, 0)`` leaves the tensor unchanged, but ``squeeze(input, 1)``
will squeeze the tensor to the shape :math:`(A \times B)`.

.. note:: The returned tensor shares the storage with the input tensor,
          so changing the contents of one will change the contents of the other.

Args:
    {input}
    dim (int, optional): if given, the input will be squeezed only in
           this dimension
    {out}

Example::

    >>> x = torch.zeros(2, 1, 2, 1, 2)
    >>> x.size()
    torch.Size([2, 1, 2, 1, 2])
    >>> y = torch.squeeze(x)
    >>> y.size()
    torch.Size([2, 2, 2])
    >>> y = torch.squeeze(x, 0)
    >>> y.size()
    torch.Size([2, 1, 2, 1, 2])
    >>> y = torch.squeeze(x, 1)
    >>> y.size()
    torch.Size([2, 2, 1, 2])
""".format(**common_args))

add_docstr(torch.std,
           r"""
.. function:: std(input, unbiased=True) -> Tensor

Returns the standard-deviation of all elements in the :attr:`input` tensor.

If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
via the biased estimator. Otherwise, Bessel's correction will be used.

Args:
    {input}
    unbiased (bool): whether to use the unbiased estimation or not

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.8166, -1.3802, -0.3560]])
    >>> torch.std(a)
    tensor(0.5130)

.. function:: std(input, dim, keepdim=False, unbiased=True, out=None) -> Tensor

Returns the standard-deviation of each row of the :attr:`input` tensor in the
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.

{keepdim_details}

If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
via the biased estimator. Otherwise, Bessel's correction will be used.

Args:
    {input}
    {dim}
    {keepdim}
    unbiased (bool): whether to use the unbiased estimation or not
    {out}

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.2035,  1.2959,  1.8101, -0.4644],
            [ 1.5027, -0.3270,  0.5905,  0.6538],
            [-1.5745,  1.3330, -0.5596, -0.6548],
            [ 0.1264, -0.5080,  1.6420,  0.1992]])
    >>> torch.std(a, dim=1)
    tensor([ 1.0311,  0.7477,  1.2204,  0.9087])
""".format(**multi_dim_common))

add_docstr(torch.std_mean,
           r"""
.. function:: std_mean(input, unbiased=True) -> (Tensor, Tensor)

Returns the standard-deviation and mean of all elements in the :attr:`input` tensor.

If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
via the biased estimator. Otherwise, Bessel's correction will be used.

Args:
    {input}
    unbiased (bool): whether to use the unbiased estimation or not

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[0.3364, 0.3591, 0.9462]])
    >>> torch.std_mean(a)
    (tensor(0.3457), tensor(0.5472))

.. function:: std(input, dim, keepdim=False, unbiased=True) -> (Tensor, Tensor)

Returns the standard-deviation and mean of each row of the :attr:`input` tensor in the
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.

{keepdim_details}

If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
via the biased estimator. Otherwise, Bessel's correction will be used.

Args:
    {input}
    {dim}
    {keepdim}
    unbiased (bool): whether to use the unbiased estimation or not

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.5648, -0.5984, -1.2676, -1.4471],
            [ 0.9267,  1.0612,  1.1050, -0.6014],
            [ 0.0154,  1.9301,  0.0125, -1.0904],
            [-1.9711, -0.7748, -1.3840,  0.5067]])
    >>> torch.std_mean(a, 1)
    (tensor([0.9110, 0.8197, 1.2552, 1.0608]), tensor([-0.6871,  0.6229,  0.2169, -0.9058]))
""".format(**multi_dim_common))

add_docstr(torch.sum,
           r"""
.. function:: sum(input, dtype=None) -> Tensor

Returns the sum of all elements in the :attr:`input` tensor.

Args:
    {input}
    {dtype}

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.1133, -0.9567,  0.2958]])
    >>> torch.sum(a)
    tensor(-0.5475)

.. function:: sum(input, dim, keepdim=False, dtype=None) -> Tensor

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.

{keepdim_details}

Args:
    {input}
    {dim}
    {keepdim}
    {dtype}

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
    >>> torch.sum(a, 1)
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
    >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
    >>> torch.sum(b, (2, 1))
    tensor([  435.,  1335.,  2235.,  3135.])
""".format(**multi_dim_common))

add_docstr(torch.svd,
           r"""
svd(input, some=True, compute_uv=True, out=None) -> (Tensor, Tensor, Tensor)

This function returns a namedtuple ``(U, S, V)`` which is the singular value
decomposition of a input real matrix or batches of real matrices :attr:`input` such that
:math:`input = U \times diag(S) \times V^T`.

If :attr:`some` is ``True`` (default), the method returns the reduced singular value decomposition
i.e., if the last two dimensions of :attr:`input` are ``m`` and ``n``, then the returned
`U` and `V` matrices will contain only :math:`min(n, m)` orthonormal columns.

If :attr:`compute_uv` is ``False``, the returned `U` and `V` matrices will be zero matrices
of shape :math:`(m \times m)` and :math:`(n \times n)` respectively. :attr:`some` will be ignored here.

.. note:: The implementation of SVD on CPU uses the LAPACK routine `?gesdd` (a divide-and-conquer
          algorithm) instead of `?gesvd` for speed. Analogously, the SVD on GPU uses the MAGMA routine
          `gesdd` as well.

.. note:: Irrespective of the original strides, the returned matrix `U`
          will be transposed, i.e. with strides :code:`U.contiguous().transpose(-2, -1).stride()`

.. note:: Extra care needs to be taken when backward through `U` and `V`
          outputs. Such operation is really only stable when :attr:`input` is
          full rank with all distinct singular values. Otherwise, ``NaN`` can
          appear as the gradients are not properly defined. Also, notice that
          double backward will usually do an additional backward through `U` and
          `V` even if the original backward is only on `S`.

.. note:: When :attr:`some` = ``False``, the gradients on :code:`U[..., :, min(m, n):]`
          and :code:`V[..., :, min(m, n):]` will be ignored in backward as those vectors
          can be arbitrary bases of the subspaces.

.. note:: When :attr:`compute_uv` = ``False``, backward cannot be performed since `U` and `V`
          from the forward pass is required for the backward operation.

Args:
    input (Tensor): the input tensor of size :math:`(*, m, n)` where `*` is zero or more
                    batch dimensions consisting of :math:`m \times n` matrices.
    some (bool, optional): controls the shape of returned `U` and `V`
    compute_uv (bool, optional): option whether to compute `U` and `V` or not
    out (tuple, optional): the output tuple of tensors

Example::

    >>> a = torch.randn(5, 3)
    >>> a
    tensor([[ 0.2364, -0.7752,  0.6372],
            [ 1.7201,  0.7394, -0.0504],
            [-0.3371, -1.0584,  0.5296],
            [ 0.3550, -0.4022,  1.5569],
            [ 0.2445, -0.0158,  1.1414]])
    >>> u, s, v = torch.svd(a)
    >>> u
    tensor([[ 0.4027,  0.0287,  0.5434],
            [-0.1946,  0.8833,  0.3679],
            [ 0.4296, -0.2890,  0.5261],
            [ 0.6604,  0.2717, -0.2618],
            [ 0.4234,  0.2481, -0.4733]])
    >>> s
    tensor([2.3289, 2.0315, 0.7806])
    >>> v
    tensor([[-0.0199,  0.8766,  0.4809],
            [-0.5080,  0.4054, -0.7600],
            [ 0.8611,  0.2594, -0.4373]])
    >>> torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t()))
    tensor(8.6531e-07)
    >>> a_big = torch.randn(7, 5, 3)
    >>> u, s, v = torch.svd(a_big)
    >>> torch.dist(a_big, torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1)))
    tensor(2.6503e-06)
""")

add_docstr(torch.symeig,
           r"""
symeig(input, eigenvectors=False, upper=True, out=None) -> (Tensor, Tensor)

This function returns eigenvalues and eigenvectors
of a real symmetric matrix :attr:`input` or a batch of real symmetric matrices,
represented by a namedtuple (eigenvalues, eigenvectors).

This function calculates all eigenvalues (and vectors) of :attr:`input`
such that :math:`\text{input} = V \text{diag}(e) V^T`.

The boolean argument :attr:`eigenvectors` defines computation of
both eigenvectors and eigenvalues or eigenvalues only.

If it is ``False``, only eigenvalues are computed. If it is ``True``,
both eigenvalues and eigenvectors are computed.

Since the input matrix :attr:`input` is supposed to be symmetric,
only the upper triangular portion is used by default.

If :attr:`upper` is ``False``, then lower triangular portion is used.

.. note:: Irrespective of the original strides, the returned matrix `V` will
          be transposed, i.e. with strides `V.contiguous().transpose(-1, -2).stride()`.

.. note:: Extra care needs to be taken when backward through outputs. Such
          operation is really only stable when all eigenvalues are distinct.
          Otherwise, ``NaN`` can appear as the gradients are not properly defined.

Args:
    input (Tensor): the input tensor of size :math:`(*, n, n)` where `*` is zero or more
                    batch dimensions consisting of symmetric matrices.
    eigenvectors(boolean, optional): controls whether eigenvectors have to be computed
    upper(boolean, optional): controls whether to consider upper-triangular or lower-triangular region
    out (tuple, optional): the output tuple of (Tensor, Tensor)

Returns:
    (Tensor, Tensor): A namedtuple (eigenvalues, eigenvectors) containing

        - **eigenvalues** (*Tensor*): Shape :math:`(*, m)`. The eigenvalues in ascending order.
        - **eigenvectors** (*Tensor*): Shape :math:`(*, m, m)`.
          If ``eigenvectors=False``, it's a tensor filled with zeros.
          Otherwise, this tensor contains the orthonormal eigenvectors of the ``input``.

Examples::


    >>> a = torch.randn(5, 5)
    >>> a = a + a.t()  # To make a symmetric
    >>> a
    tensor([[-5.7827,  4.4559, -0.2344, -1.7123, -1.8330],
            [ 4.4559,  1.4250, -2.8636, -3.2100, -0.1798],
            [-0.2344, -2.8636,  1.7112, -5.5785,  7.1988],
            [-1.7123, -3.2100, -5.5785, -2.6227,  3.1036],
            [-1.8330, -0.1798,  7.1988,  3.1036, -5.1453]])
    >>> e, v = torch.symeig(a, eigenvectors=True)
    >>> e
    tensor([-13.7012,  -7.7497,  -2.3163,   5.2477,   8.1050])
    >>> v
    tensor([[ 0.1643,  0.9034, -0.0291,  0.3508,  0.1817],
            [-0.2417, -0.3071, -0.5081,  0.6534,  0.4026],
            [-0.5176,  0.1223, -0.0220,  0.3295, -0.7798],
            [-0.4850,  0.2695, -0.5773, -0.5840,  0.1337],
            [ 0.6415, -0.0447, -0.6381, -0.0193, -0.4230]])
    >>> a_big = torch.randn(5, 2, 2)
    >>> a_big = a_big + a_big.transpose(-2, -1)  # To make a_big symmetric
    >>> e, v = a_big.symeig(eigenvectors=True)
    >>> torch.allclose(torch.matmul(v, torch.matmul(e.diag_embed(), v.transpose(-2, -1))), a_big)
    True
""")

add_docstr(torch.t,
           r"""
t(input) -> Tensor

Expects :attr:`input` to be <= 2-D tensor and transposes dimensions 0
and 1.

0-D and 1-D tensors are returned as it is and
2-D tensor can be seen as a short-hand function for ``transpose(input, 0, 1)``.

Args:
    {input}

Example::

    >>> x = torch.randn(())
    >>> x
    tensor(0.1995)
    >>> torch.t(x)
    tensor(0.1995)
    >>> x = torch.randn(3)
    >>> x
    tensor([ 2.4320, -0.4608,  0.7702])
    >>> torch.t(x)
    tensor([.2.4320,.-0.4608,..0.7702])
    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.4875,  0.9158, -0.5872],
            [ 0.3938, -0.6929,  0.6932]])
    >>> torch.t(x)
    tensor([[ 0.4875,  0.3938],
            [ 0.9158, -0.6929],
            [-0.5872,  0.6932]])
""".format(**common_args))

add_docstr(torch.flip,
           r"""
flip(input, dims) -> Tensor

Reverse the order of a n-D tensor along given axis in dims.

Args:
    {input}
    dims (a list or tuple): axis to flip on

Example::

    >>> x = torch.arange(8).view(2, 2, 2)
    >>> x
    tensor([[[ 0,  1],
             [ 2,  3]],

            [[ 4,  5],
             [ 6,  7]]])
    >>> torch.flip(x, [0, 1])
    tensor([[[ 6,  7],
             [ 4,  5]],

            [[ 2,  3],
             [ 0,  1]]])
""".format(**common_args))

add_docstr(torch.roll,
           r"""
roll(input, shifts, dims=None) -> Tensor

Roll the tensor along the given dimension(s). Elements that are shifted beyond the
last position are re-introduced at the first position. If a dimension is not
specified, the tensor will be flattened before rolling and then restored
to the original shape.

Args:
    {input}
    shifts (int or tuple of ints): The number of places by which the elements
        of the tensor are shifted. If shifts is a tuple, dims must be a tuple of
        the same size, and each dimension will be rolled by the corresponding
        value
    dims (int or tuple of ints): Axis along which to roll

Example::

    >>> x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
    >>> x
    tensor([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8]])
    >>> torch.roll(x, 1, 0)
    tensor([[7, 8],
            [1, 2],
            [3, 4],
            [5, 6]])
    >>> torch.roll(x, -1, 0)
    tensor([[3, 4],
            [5, 6],
            [7, 8],
            [1, 2]])
    >>> torch.roll(x, shifts=(2, 1), dims=(0, 1))
    tensor([[6, 5],
            [8, 7],
            [2, 1],
            [4, 3]])
""".format(**common_args))

add_docstr(torch.rot90,
           r"""
rot90(input, k, dims) -> Tensor

Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.
Rotation direction is from the first towards the second axis if k > 0, and from the second towards the first for k < 0.

Args:
    {input}
    k (int): number of times to rotate
    dims (a list or tuple): axis to rotate

Example::

    >>> x = torch.arange(4).view(2, 2)
    >>> x
    tensor([[0, 1],
            [2, 3]])
    >>> torch.rot90(x, 1, [0, 1])
    tensor([[1, 3],
            [0, 2]])

    >>> x = torch.arange(8).view(2, 2, 2)
    >>> x
    tensor([[[0, 1],
             [2, 3]],

            [[4, 5],
             [6, 7]]])
    >>> torch.rot90(x, 1, [1, 2])
    tensor([[[1, 3],
             [0, 2]],

            [[5, 7],
             [4, 6]]])
""".format(**common_args))

add_docstr(torch.take,
           r"""
take(input, index) -> Tensor

Returns a new tensor with the elements of :attr:`input` at the given indices.
The input tensor is treated as if it were viewed as a 1-D tensor. The result
takes the same shape as the indices.

Args:
    {input}
    indices (LongTensor): the indices into tensor

Example::

    >>> src = torch.tensor([[4, 3, 5],
                            [6, 7, 8]])
    >>> torch.take(src, torch.tensor([0, 2, 5]))
    tensor([ 4,  5,  8])
""".format(**common_args))

add_docstr(torch.tan,
           r"""
tan(input, out=None) -> Tensor

Returns a new tensor with the tangent of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \tan(\text{input}_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-1.2027, -1.7687,  0.4412, -1.3856])
    >>> torch.tan(a)
    tensor([-2.5930,  4.9859,  0.4722, -5.3366])
""".format(**common_args))

add_docstr(torch.tanh,
           r"""
tanh(input, out=None) -> Tensor

Returns a new tensor with the hyperbolic tangent of the elements
of :attr:`input`.

.. math::
    \text{out}_{i} = \tanh(\text{input}_{i})
""" + r"""
Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
    >>> torch.tanh(a)
    tensor([ 0.7156, -0.6218,  0.8257,  0.2553])
""".format(**common_args))

add_docstr(torch.topk,
           r"""
topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)

Returns the :attr:`k` largest elements of the given :attr:`input` tensor along
a given dimension.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`largest` is ``False`` then the `k` smallest elements are returned.

A namedtuple of `(values, indices)` is returned, where the `indices` are the indices
of the elements in the original `input` tensor.

The boolean option :attr:`sorted` if ``True``, will make sure that the returned
`k` elements are themselves sorted

Args:
    {input}
    k (int): the k in "top-k"
    dim (int, optional): the dimension to sort along
    largest (bool, optional): controls whether to return largest or
           smallest elements
    sorted (bool, optional): controls whether to return the elements
           in sorted order
    out (tuple, optional): the output tuple of (Tensor, LongTensor) that can be
        optionally given to be used as output buffers

Example::

    >>> x = torch.arange(1., 6.)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.])
    >>> torch.topk(x, 3)
    torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
""".format(**common_args))

add_docstr(torch.trace,
           r"""
trace(input) -> Tensor

Returns the sum of the elements of the diagonal of the input 2-D matrix.

Example::

    >>> x = torch.arange(1., 10.).view(3, 3)
    >>> x
    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.],
            [ 7.,  8.,  9.]])
    >>> torch.trace(x)
    tensor(15.)
""")

add_docstr(torch.transpose,
           r"""
transpose(input, dim0, dim1) -> Tensor

Returns a tensor that is a transposed version of :attr:`input`.
The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

The resulting :attr:`out` tensor shares it's underlying storage with the
:attr:`input` tensor, so changing the content of one would change the content
of the other.

Args:
    {input}
    dim0 (int): the first dimension to be transposed
    dim1 (int): the second dimension to be transposed

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]])
    >>> torch.transpose(x, 0, 1)
    tensor([[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [ 0.5809,  0.4942]])
""".format(**common_args))

add_docstr(torch.triangular_solve,
           r"""
triangular_solve(input, A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)

Solves a system of equations with a triangular coefficient matrix :math:`A`
and multiple right-hand sides :math:`b`.

In particular, solves :math:`AX = b` and assumes :math:`A` is upper-triangular
with the default keyword arguments.

`torch.triangular_solve(b, A)` can take in 2D inputs `b, A` or inputs that are
batches of 2D matrices. If the inputs are batches, then returns
batched outputs `X`

.. note::

    The :attr:`out` keyword only supports 2D matrix inputs, that is,
    `b, A` must be 2D matrices.

Args:
    input (Tensor): multiple right-hand sides of size :math:`(*, m, k)` where
                :math:`*` is zero of more batch dimensions (:math:`b`)
    A (Tensor): the input triangular coefficient matrix of size :math:`(*, m, m)`
                where :math:`*` is zero or more batch dimensions
    upper (bool, optional): whether to solve the upper-triangular system
        of equations (default) or the lower-triangular system of equations. Default: ``True``.
    transpose (bool, optional): whether :math:`A` should be transposed before
        being sent into the solver. Default: ``False``.
    unitriangular (bool, optional): whether :math:`A` is unit triangular.
        If True, the diagonal elements of :math:`A` are assumed to be
        1 and not referenced from :math:`A`. Default: ``False``.

Returns:
    A namedtuple `(solution, cloned_coefficient)` where `cloned_coefficient`
    is a clone of :math:`A` and `solution` is the solution :math:`X` to :math:`AX = b`
    (or whatever variant of the system of equations, depending on the keyword arguments.)

Examples::

    >>> A = torch.randn(2, 2).triu()
    >>> A
    tensor([[ 1.1527, -1.0753],
            [ 0.0000,  0.7986]])
    >>> b = torch.randn(2, 3)
    >>> b
    tensor([[-0.0210,  2.3513, -1.5492],
            [ 1.5429,  0.7403, -1.0243]])
    >>> torch.triangular_solve(b, A)
    torch.return_types.triangular_solve(
    solution=tensor([[ 1.7841,  2.9046, -2.5405],
            [ 1.9320,  0.9270, -1.2826]]),
    cloned_coefficient=tensor([[ 1.1527, -1.0753],
            [ 0.0000,  0.7986]]))
""")

add_docstr(torch.tril,
           r"""
tril(input, diagonal=0, out=None) -> Tensor

Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
:attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.

The lower triangular part of the matrix is defined as the elements on and
below the diagonal.

The argument :attr:`diagonal` controls which diagonal to consider. If
:attr:`diagonal` = 0, all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the main
diagonal, and similarly a negative value excludes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
:math:`d_{1}, d_{2}` are the dimensions of the matrix.
""" + r"""
Args:
    {input}
    diagonal (int, optional): the diagonal to consider
    {out}

Example::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-1.0813, -0.8619,  0.7105],
            [ 0.0935,  0.1380,  2.2112],
            [-0.3409, -0.9828,  0.0289]])
    >>> torch.tril(a)
    tensor([[-1.0813,  0.0000,  0.0000],
            [ 0.0935,  0.1380,  0.0000],
            [-0.3409, -0.9828,  0.0289]])

    >>> b = torch.randn(4, 6)
    >>> b
    tensor([[ 1.2219,  0.5653, -0.2521, -0.2345,  1.2544,  0.3461],
            [ 0.4785, -0.4477,  0.6049,  0.6368,  0.8775,  0.7145],
            [ 1.1502,  3.2716, -1.1243, -0.5413,  0.3615,  0.6864],
            [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0978]])
    >>> torch.tril(b, diagonal=1)
    tensor([[ 1.2219,  0.5653,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.4785, -0.4477,  0.6049,  0.0000,  0.0000,  0.0000],
            [ 1.1502,  3.2716, -1.1243, -0.5413,  0.0000,  0.0000],
            [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0000]])
    >>> torch.tril(b, diagonal=-1)
    tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.4785,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 1.1502,  3.2716,  0.0000,  0.0000,  0.0000,  0.0000],
            [-0.0614, -0.7344, -1.3164,  0.0000,  0.0000,  0.0000]])
""".format(**common_args))

# docstr is split in two parts to avoid format mis-captureing :math: braces '{}'
# as common args.
add_docstr(torch.tril_indices,
           r"""
tril_indices(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor

Returns the indices of the lower triangular part of a :attr:`row`-by-
:attr:`col` matrix in a 2-by-N Tensor, where the first row contains row
coordinates of all indices and the second row contains column coordinates.
Indices are ordered based on rows and then columns.

The lower triangular part of the matrix is defined as the elements on and
below the diagonal.

The argument :attr:`offset` controls which diagonal to consider. If
:attr:`offset` = 0, all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the main
diagonal, and similarly a negative value excludes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]`
where :math:`d_{1}, d_{2}` are the dimensions of the matrix.

NOTE: when running on 'cuda', row * col must be less than :math:`2^{59}` to
prevent overflow during calculation.
""" + r"""
Args:
    row (``int``): number of rows in the 2-D matrix.
    col (``int``): number of columns in the 2-D matrix.
    offset (``int``): diagonal offset from the main diagonal.
        Default: if not provided, 0.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, ``torch.long``.
    {device}
    layout (:class:`torch.layout`, optional): currently only support ``torch.strided``.

Example::
    >>> a = torch.tril_indices(3, 3)
    >>> a
    tensor([[0, 1, 1, 2, 2, 2],
            [0, 0, 1, 0, 1, 2]])

    >>> a = torch.tril_indices(4, 3, -1)
    >>> a
    tensor([[1, 2, 2, 3, 3, 3],
            [0, 0, 1, 0, 1, 2]])

    >>> a = torch.tril_indices(4, 3, 1)
    >>> a
    tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2]])
""".format(**factory_common_args))

add_docstr(torch.triu,
           r"""
triu(input, diagonal=0, out=None) -> Tensor

Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
:attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.

The upper triangular part of the matrix is defined as the elements on and
above the diagonal.

The argument :attr:`diagonal` controls which diagonal to consider. If
:attr:`diagonal` = 0, all elements on and below the main diagonal are
retained. A positive value excludes just as many diagonals above the main
diagonal, and similarly a negative value includes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
:math:`d_{1}, d_{2}` are the dimensions of the matrix.
""" + r"""
Args:
    {input}
    diagonal (int, optional): the diagonal to consider
    {out}

Example::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.3480, -0.5211, -0.4573]])
    >>> torch.triu(a)
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.0000, -1.0680,  0.6602],
            [ 0.0000,  0.0000, -0.4573]])
    >>> torch.triu(a, diagonal=1)
    tensor([[ 0.0000,  0.5207,  2.0049],
            [ 0.0000,  0.0000,  0.6602],
            [ 0.0000,  0.0000,  0.0000]])
    >>> torch.triu(a, diagonal=-1)
    tensor([[ 0.2309,  0.5207,  2.0049],
            [ 0.2072, -1.0680,  0.6602],
            [ 0.0000, -0.5211, -0.4573]])

    >>> b = torch.randn(4, 6)
    >>> b
    tensor([[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.4333,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
            [-0.9888,  1.0679, -1.3337, -1.6556,  0.4798,  0.2830]])
    >>> torch.triu(b, diagonal=1)
    tensor([[ 0.0000, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [ 0.0000,  0.0000, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.0000,  0.0000,  0.0000, -1.0432,  0.9348, -0.4410],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.4798,  0.2830]])
    >>> torch.triu(b, diagonal=-1)
    tensor([[ 0.5876, -0.0794, -1.8373,  0.6654,  0.2604,  1.5235],
            [-0.2447,  0.9556, -1.2919,  1.3378, -0.1768, -1.0857],
            [ 0.0000,  0.3146,  0.6576, -1.0432,  0.9348, -0.4410],
            [ 0.0000,  0.0000, -1.3337, -1.6556,  0.4798,  0.2830]])
""".format(**common_args))

# docstr is split in two parts to avoid format mis-capturing :math: braces '{}'
# as common args.
add_docstr(torch.triu_indices,
           r"""
triu_indices(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) -> Tensor

Returns the indices of the upper triangular part of a :attr:`row` by
:attr:`col` matrix in a 2-by-N Tensor, where the first row contains row
coordinates of all indices and the second row contains column coordinates.
Indices are ordered based on rows and then columns.

The upper triangular part of the matrix is defined as the elements on and
above the diagonal.

The argument :attr:`offset` controls which diagonal to consider. If
:attr:`offset` = 0, all elements on and above the main diagonal are
retained. A positive value excludes just as many diagonals above the main
diagonal, and similarly a negative value includes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]`
where :math:`d_{1}, d_{2}` are the dimensions of the matrix.

NOTE: when running on 'cuda', row * col must be less than :math:`2^{59}` to
prevent overflow during calculation.
""" + r"""
Args:
    row (``int``): number of rows in the 2-D matrix.
    col (``int``): number of columns in the 2-D matrix.
    offset (``int``): diagonal offset from the main diagonal.
        Default: if not provided, 0.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, ``torch.long``.
    {device}
    layout (:class:`torch.layout`, optional): currently only support ``torch.strided``.

Example::
    >>> a = torch.triu_indices(3, 3)
    >>> a
    tensor([[0, 0, 0, 1, 1, 2],
            [0, 1, 2, 1, 2, 2]])

    >>> a = torch.triu_indices(4, 3, -1)
    >>> a
    tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3],
            [0, 1, 2, 0, 1, 2, 1, 2, 2]])

    >>> a = torch.triu_indices(4, 3, 1)
    >>> a
    tensor([[0, 0, 1],
            [1, 2, 2]])
""".format(**factory_common_args))

add_docstr(torch.trunc,
           r"""
trunc(input, out=None) -> Tensor

Returns a new tensor with the truncated integer values of
the elements of :attr:`input`.

Args:
    {input}
    {out}

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 3.4742,  0.5466, -0.8008, -0.9079])
    >>> torch.trunc(a)
    tensor([ 3.,  0., -0., -0.])
""".format(**common_args))

add_docstr(torch.unsqueeze,
           r"""
unsqueeze(input, dim, out=None) -> Tensor

Returns a new tensor with a dimension of size one inserted at the
specified position.

The returned tensor shares the same underlying data with this tensor.

A :attr:`dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``
can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
applied at :attr:`dim` = ``dim + input.dim() + 1``.

Args:
    {input}
    dim (int): the index at which to insert the singleton dimension
    {out}

Example::

    >>> x = torch.tensor([1, 2, 3, 4])
    >>> torch.unsqueeze(x, 0)
    tensor([[ 1,  2,  3,  4]])
    >>> torch.unsqueeze(x, 1)
    tensor([[ 1],
            [ 2],
            [ 3],
            [ 4]])
""".format(**common_args))

add_docstr(torch.var,
           r"""
.. function:: var(input, unbiased=True) -> Tensor

Returns the variance of all elements in the :attr:`input` tensor.

If :attr:`unbiased` is ``False``, then the variance will be calculated via the
biased estimator. Otherwise, Bessel's correction will be used.

Args:
    {input}
    unbiased (bool): whether to use the unbiased estimation or not

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.3425, -1.2636, -0.4864]])
    >>> torch.var(a)
    tensor(0.2455)


.. function:: var(input, dim, keepdim=False, unbiased=True, out=None) -> Tensor

Returns the variance of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`.

{keepdim_details}

If :attr:`unbiased` is ``False``, then the variance will be calculated via the
biased estimator. Otherwise, Bessel's correction will be used.

Args:
    {input}
    {dim}
    {keepdim}
    unbiased (bool): whether to use the unbiased estimation or not
    {out}

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.3567,  1.7385, -1.3042,  0.7423],
            [ 1.3436, -0.1015, -0.9834, -0.8438],
            [ 0.6056,  0.1089, -0.3112, -1.4085],
            [-0.7700,  0.6074, -0.1469,  0.7777]])
    >>> torch.var(a, 1)
    tensor([ 1.7444,  1.1363,  0.7356,  0.5112])
""".format(**multi_dim_common))

add_docstr(torch.var_mean,
           r"""
.. function:: var_mean(input, unbiased=True) -> (Tensor, Tensor)

Returns the variance and mean of all elements in the :attr:`input` tensor.

If :attr:`unbiased` is ``False``, then the variance will be calculated via the
biased estimator. Otherwise, Bessel's correction will be used.

Args:
    {input}
    unbiased (bool): whether to use the unbiased estimation or not

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[0.0146, 0.4258, 0.2211]])
    >>> torch.var_mean(a)
    (tensor(0.0423), tensor(0.2205))

.. function:: var_mean(input, dim, keepdim=False, unbiased=True) -> (Tensor, Tensor)

Returns the variance and mean of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`.

{keepdim_details}

If :attr:`unbiased` is ``False``, then the variance will be calculated via the
biased estimator. Otherwise, Bessel's correction will be used.

Args:
    {input}
    {dim}
    {keepdim}
    unbiased (bool): whether to use the unbiased estimation or not

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-1.5650,  2.0415, -0.1024, -0.5790],
            [ 0.2325, -2.6145, -1.6428, -0.3537],
            [-0.2159, -1.1069,  1.2882, -1.3265],
            [-0.6706, -1.5893,  0.6827,  1.6727]])
    >>> torch.var_mean(a, 1)
    (tensor([2.3174, 1.6403, 1.4092, 2.0791]), tensor([-0.0512, -1.0946, -0.3403,  0.0239]))
""".format(**multi_dim_common))

add_docstr(torch.zeros,
           r"""
zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with the scalar value `0`, with the shape defined
by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.zeros(2, 3)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

    >>> torch.zeros(5)
    tensor([ 0.,  0.,  0.,  0.,  0.])
""".format(**factory_common_args))

add_docstr(torch.zeros_like,
           r"""
zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with the scalar value `0`, with the same size as
:attr:`input`. ``torch.zeros_like(input)`` is equivalent to
``torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

.. warning::
    As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,
    the old ``torch.zeros_like(input, out=output)`` is equivalent to
    ``torch.zeros(input.size(), out=output)``.

Args:
    {input}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> input = torch.empty(2, 3)
    >>> torch.zeros_like(input)
    tensor([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])
""".format(**factory_like_common_args))

add_docstr(torch.empty,
           r"""
empty(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor

Returns a tensor filled with uninitialized data. The shape of the tensor is
defined by the variable argument :attr:`size`.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}
    {pin_memory}

Example::

    >>> torch.empty(2, 3)
    tensor(1.00000e-08 *
           [[ 6.3984,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000]])

""".format(**factory_common_args))

add_docstr(torch.empty_like,
           r"""
empty_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor

Returns an uninitialized tensor with the same size as :attr:`input`.
``torch.empty_like(input)`` is equivalent to
``torch.empty(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    {input}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.empty((2,3), dtype=torch.int64)
    tensor([[ 9.4064e+13,  2.8000e+01,  9.3493e+13],
            [ 7.5751e+18,  7.1428e+18,  7.5955e+18]])
""".format(**factory_like_common_args))

add_docstr(torch.empty_strided,
           r"""
empty_strided(size, stride, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) -> Tensor

Returns a tensor filled with uninitialized data. The shape and strides of the tensor is
defined by the variable argument :attr:`size` and :attr:`stride` respectively.
``torch.empty_strided(size, stride)`` is equivalent to
``torch.empty(size).as_strided(size, stride)``.

.. warning::
    More than one element of the created tensor may refer to a single memory
    location. As a result, in-place operations (especially ones that are
    vectorized) may result in incorrect behavior. If you need to write to
    the tensors, please clone them first.

Args:
    size (tuple of ints): the shape of the output tensor
    stride (tuple of ints): the strides of the output tensor
    {dtype}
    {layout}
    {device}
    {requires_grad}
    {pin_memory}

Example::

    >>> a = torch.empty_strided((2, 3), (1, 2))
    >>> a
    tensor([[8.9683e-44, 4.4842e-44, 5.1239e+07],
            [0.0000e+00, 0.0000e+00, 3.0705e-41]])
    >>> a.stride()
    (1, 2)
    >>> a.size()
    torch.Size([2, 3])
""".format(**factory_common_args))

add_docstr(torch.full,
           r"""
full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor of size :attr:`size` filled with :attr:`fill_value`.

Args:
    size (int...): a list, tuple, or :class:`torch.Size` of integers defining the
        shape of the output tensor.
    fill_value: the number to fill the output tensor with.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.full((2, 3), 3.141592)
    tensor([[ 3.1416,  3.1416,  3.1416],
            [ 3.1416,  3.1416,  3.1416]])

""".format(**factory_common_args))

add_docstr(torch.full_like,
           r"""
full_like(input, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor with the same size as :attr:`input` filled with :attr:`fill_value`.
``torch.full_like(input, fill_value)`` is equivalent to
``torch.full(input.size(), fill_value, dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    {input}
    fill_value: the number to fill the output tensor with.
    {dtype}
    {layout}
    {device}
    {requires_grad}
""".format(**factory_like_common_args))

add_docstr(torch.det,
           r"""
det(input) -> Tensor

Calculates determinant of a square matrix or batches of square matrices.

.. note::
    Backward through :meth:`det` internally uses SVD results when :attr:`input` is
    not invertible. In this case, double backward through :meth:`det` will be
    unstable in when :attr:`input` doesn't have distinct singular values. See
    :meth:`~torch.svd` for details.

Arguments:
    input (Tensor): the input tensor of size (*, n, n) where `*` is zero or more
                batch dimensions.

Example::

    >>> A = torch.randn(3, 3)
    >>> torch.det(A)
    tensor(3.7641)

    >>> A = torch.randn(3, 2, 2)
    >>> A
    tensor([[[ 0.9254, -0.6213],
             [-0.5787,  1.6843]],

            [[ 0.3242, -0.9665],
             [ 0.4539, -0.0887]],

            [[ 1.1336, -0.4025],
             [-0.7089,  0.9032]]])
    >>> A.det()
    tensor([1.1990, 0.4099, 0.7386])
""")

add_docstr(torch.where,
           r"""
.. function:: where(condition, x, y) -> Tensor

Return a tensor of elements selected from either :attr:`x` or :attr:`y`, depending on :attr:`condition`.

The operation is defined as:

.. math::
    \text{out}_i = \begin{cases}
        \text{x}_i & \text{if } \text{condition}_i \\
        \text{y}_i & \text{otherwise} \\
    \end{cases}

.. note::
    The tensors :attr:`condition`, :attr:`x`, :attr:`y` must be :ref:`broadcastable <broadcasting-semantics>`.

Arguments:
    condition (BoolTensor): When True (nonzero), yield x, otherwise yield y
    x (Tensor): values selected at indices where :attr:`condition` is ``True``
    y (Tensor): values selected at indices where :attr:`condition` is ``False``

Returns:
    Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`x`, :attr:`y`

Example::

    >>> x = torch.randn(3, 2)
    >>> y = torch.ones(3, 2)
    >>> x
    tensor([[-0.4620,  0.3139],
            [ 0.3898, -0.7197],
            [ 0.0478, -0.1657]])
    >>> torch.where(x > 0, x, y)
    tensor([[ 1.0000,  0.3139],
            [ 0.3898,  1.0000],
            [ 0.0478,  1.0000]])

.. function:: where(condition) -> tuple of LongTensor

``torch.where(condition)`` is identical to
``torch.nonzero(condition, as_tuple=True)``.

.. note::
    See also :func:`torch.nonzero`.
""")

add_docstr(torch.logdet,
           r"""
logdet(input) -> Tensor

Calculates log determinant of a square matrix or batches of square matrices.

.. note::
    Result is ``-inf`` if :attr:`input` has zero log determinant, and is ``nan`` if
    :attr:`input` has negative determinant.

.. note::
    Backward through :meth:`logdet` internally uses SVD results when :attr:`input`
    is not invertible. In this case, double backward through :meth:`logdet` will
    be unstable in when :attr:`input` doesn't have distinct singular values. See
    :meth:`~torch.svd` for details.

Arguments:
    input (Tensor): the input tensor of size (*, n, n) where `*` is zero or more
                batch dimensions.

Example::

    >>> A = torch.randn(3, 3)
    >>> torch.det(A)
    tensor(0.2611)
    >>> torch.logdet(A)
    tensor(-1.3430)
    >>> A
    tensor([[[ 0.9254, -0.6213],
             [-0.5787,  1.6843]],

            [[ 0.3242, -0.9665],
             [ 0.4539, -0.0887]],

            [[ 1.1336, -0.4025],
             [-0.7089,  0.9032]]])
    >>> A.det()
    tensor([1.1990, 0.4099, 0.7386])
    >>> A.det().log()
    tensor([ 0.1815, -0.8917, -0.3031])
""")

add_docstr(torch.slogdet,
           r"""
slogdet(input) -> (Tensor, Tensor)

Calculates the sign and log absolute value of the determinant(s) of a square matrix or batches of square matrices.

.. note::
    If ``input`` has zero determinant, this returns ``(0, -inf)``.

.. note::
    Backward through :meth:`slogdet` internally uses SVD results when :attr:`input`
    is not invertible. In this case, double backward through :meth:`slogdet`
    will be unstable in when :attr:`input` doesn't have distinct singular values.
    See :meth:`~torch.svd` for details.

Arguments:
    input (Tensor): the input tensor of size (*, n, n) where `*` is zero or more
                batch dimensions.

Returns:
    A namedtuple (sign, logabsdet) containing the sign of the determinant, and the log
    value of the absolute determinant.

Example::

    >>> A = torch.randn(3, 3)
    >>> A
    tensor([[ 0.0032, -0.2239, -1.1219],
            [-0.6690,  0.1161,  0.4053],
            [-1.6218, -0.9273, -0.0082]])
    >>> torch.det(A)
    tensor(-0.7576)
    >>> torch.logdet(A)
    tensor(nan)
    >>> torch.slogdet(A)
    torch.return_types.slogdet(sign=tensor(-1.), logabsdet=tensor(-0.2776))
""")

add_docstr(torch.pinverse,
           r"""
pinverse(input, rcond=1e-15) -> Tensor

Calculates the pseudo-inverse (also known as the Moore-Penrose inverse) of a 2D tensor.
Please look at `Moore-Penrose inverse`_ for more details

.. note::
    This method is implemented using the Singular Value Decomposition.

.. note::
    The pseudo-inverse is not necessarily a continuous function in the elements of the matrix `[1]`_.
    Therefore, derivatives are not always existent, and exist for a constant rank only `[2]`_.
    However, this method is backprop-able due to the implementation by using SVD results, and
    could be unstable. Double-backward will also be unstable due to the usage of SVD internally.
    See :meth:`~torch.svd` for more details.

Arguments:
    input (Tensor): The input 2D tensor of dimensions :math:`m \times n`
    rcond (float): A floating point value to determine the cutoff for small singular values.
                   Default: 1e-15

Returns:
    The pseudo-inverse of :attr:`input` of dimensions :math:`n \times m`

Example::

    >>> input = torch.randn(3, 5)
    >>> input
    tensor([[ 0.5495,  0.0979, -1.4092, -0.1128,  0.4132],
            [-1.1143, -0.3662,  0.3042,  1.6374, -0.9294],
            [-0.3269, -0.5745, -0.0382, -0.5922, -0.6759]])
    >>> torch.pinverse(input)
    tensor([[ 0.0600, -0.1933, -0.2090],
            [-0.0903, -0.0817, -0.4752],
            [-0.7124, -0.1631, -0.2272],
            [ 0.1356,  0.3933, -0.5023],
            [-0.0308, -0.1725, -0.5216]])

.. _Moore-Penrose inverse: https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse

.. _[1]: https://epubs.siam.org/doi/10.1137/0117004

.. _[2]: https://www.jstor.org/stable/2156365
""")

add_docstr(torch.fft,
           r"""
fft(input, signal_ndim, normalized=False) -> Tensor

Complex-to-complex Discrete Fourier Transform

This method computes the complex-to-complex discrete Fourier transform.
Ignoring the batch dimensions, it computes the following expression:

.. math::
    X[\omega_1, \dots, \omega_d] =
        \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
         e^{-j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},

where :math:`d` = :attr:`signal_ndim` is number of dimensions for the
signal, and :math:`N_i` is the size of signal dimension :math:`i`.

This method supports 1D, 2D and 3D complex-to-complex transforms, indicated
by :attr:`signal_ndim`. :attr:`input` must be a tensor with last dimension
of size 2, representing the real and imaginary components of complex
numbers, and should have at least ``signal_ndim + 1`` dimensions with optionally
arbitrary number of leading batch dimensions. If :attr:`normalized` is set to
``True``, this normalizes the result by dividing it with
:math:`\sqrt{\prod_{i=1}^K N_i}` so that the operator is unitary.

Returns the real and the imaginary parts together as one tensor of the same
shape of :attr:`input`.

The inverse of this function is :func:`~torch.ifft`.

.. note::
    For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
    repeatedly running FFT methods on tensors of same geometry with same
    configuration. See :ref:`cufft-plan-cache` for more details on how to
    monitor and control the cache.

.. warning::
    For CPU tensors, this method is currently only available with MKL. Use
    :func:`torch.backends.mkl.is_available` to check if MKL is installed.

Arguments:
    input (Tensor): the input tensor of at least :attr:`signal_ndim` ``+ 1``
        dimensions
    signal_ndim (int): the number of dimensions in each signal.
        :attr:`signal_ndim` can only be 1, 2 or 3
    normalized (bool, optional): controls whether to return normalized results.
        Default: ``False``

Returns:
    Tensor: A tensor containing the complex-to-complex Fourier transform result

Example::

    >>> # unbatched 2D FFT
    >>> x = torch.randn(4, 3, 2)
    >>> torch.fft(x, 2)
    tensor([[[-0.0876,  1.7835],
             [-2.0399, -2.9754],
             [ 4.4773, -5.0119]],

            [[-1.5716,  2.7631],
             [-3.8846,  5.2652],
             [ 0.2046, -0.7088]],

            [[ 1.9938, -0.5901],
             [ 6.5637,  6.4556],
             [ 2.9865,  4.9318]],

            [[ 7.0193,  1.1742],
             [-1.3717, -2.1084],
             [ 2.0289,  2.9357]]])
    >>> # batched 1D FFT
    >>> torch.fft(x, 1)
    tensor([[[ 1.8385,  1.2827],
             [-0.1831,  1.6593],
             [ 2.4243,  0.5367]],

            [[-0.9176, -1.5543],
             [-3.9943, -2.9860],
             [ 1.2838, -2.9420]],

            [[-0.8854, -0.6860],
             [ 2.4450,  0.0808],
             [ 1.3076, -0.5768]],

            [[-0.1231,  2.7411],
             [-0.3075, -1.7295],
             [-0.5384, -2.0299]]])
    >>> # arbitrary number of batch dimensions, 2D FFT
    >>> x = torch.randn(3, 3, 5, 5, 2)
    >>> y = torch.fft(x, 2)
    >>> y.shape
    torch.Size([3, 3, 5, 5, 2])

""")

add_docstr(torch.ifft,
           r"""
ifft(input, signal_ndim, normalized=False) -> Tensor

Complex-to-complex Inverse Discrete Fourier Transform

This method computes the complex-to-complex inverse discrete Fourier
transform. Ignoring the batch dimensions, it computes the following
expression:

.. math::
    X[\omega_1, \dots, \omega_d] =
        \frac{1}{\prod_{i=1}^d N_i} \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
         e^{\ j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},

where :math:`d` = :attr:`signal_ndim` is number of dimensions for the
signal, and :math:`N_i` is the size of signal dimension :math:`i`.

The argument specifications are almost identical with :func:`~torch.fft`.
However, if :attr:`normalized` is set to ``True``, this instead returns the
results multiplied by :math:`\sqrt{\prod_{i=1}^d N_i}`, to become a unitary
operator. Therefore, to invert a :func:`~torch.fft`, the :attr:`normalized`
argument should be set identically for :func:`~torch.fft`.

Returns the real and the imaginary parts together as one tensor of the same
shape of :attr:`input`.

The inverse of this function is :func:`~torch.fft`.

.. note::
    For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
    repeatedly running FFT methods on tensors of same geometry with same
    configuration. See :ref:`cufft-plan-cache` for more details on how to
    monitor and control the cache.

.. warning::
    For CPU tensors, this method is currently only available with MKL. Use
    :func:`torch.backends.mkl.is_available` to check if MKL is installed.

Arguments:
    input (Tensor): the input tensor of at least :attr:`signal_ndim` ``+ 1``
        dimensions
    signal_ndim (int): the number of dimensions in each signal.
        :attr:`signal_ndim` can only be 1, 2 or 3
    normalized (bool, optional): controls whether to return normalized results.
        Default: ``False``

Returns:
    Tensor: A tensor containing the complex-to-complex inverse Fourier transform result

Example::

    >>> x = torch.randn(3, 3, 2)
    >>> x
    tensor([[[ 1.2766,  1.3680],
             [-0.8337,  2.0251],
             [ 0.9465, -1.4390]],

            [[-0.1890,  1.6010],
             [ 1.1034, -1.9230],
             [-0.9482,  1.0775]],

            [[-0.7708, -0.8176],
             [-0.1843, -0.2287],
             [-1.9034, -0.2196]]])
    >>> y = torch.fft(x, 2)
    >>> torch.ifft(y, 2)  # recover x
    tensor([[[ 1.2766,  1.3680],
             [-0.8337,  2.0251],
             [ 0.9465, -1.4390]],

            [[-0.1890,  1.6010],
             [ 1.1034, -1.9230],
             [-0.9482,  1.0775]],

            [[-0.7708, -0.8176],
             [-0.1843, -0.2287],
             [-1.9034, -0.2196]]])

""")

add_docstr(torch.rfft,
           r"""
rfft(input, signal_ndim, normalized=False, onesided=True) -> Tensor

Real-to-complex Discrete Fourier Transform

This method computes the real-to-complex discrete Fourier transform. It is
mathematically equivalent with :func:`~torch.fft` with differences only in
formats of the input and output.

This method supports 1D, 2D and 3D real-to-complex transforms, indicated
by :attr:`signal_ndim`. :attr:`input` must be a tensor with at least
``signal_ndim`` dimensions with optionally arbitrary number of leading batch
dimensions. If :attr:`normalized` is set to ``True``, this normalizes the result
by dividing it with :math:`\sqrt{\prod_{i=1}^K N_i}` so that the operator is
unitary, where :math:`N_i` is the size of signal dimension :math:`i`.

The real-to-complex Fourier transform results follow conjugate symmetry:

.. math::
    X[\omega_1, \dots, \omega_d] = X^*[N_1 - \omega_1, \dots, N_d - \omega_d],

where the index arithmetic is computed modulus the size of the corresponding
dimension, :math:`\ ^*` is the conjugate operator, and
:math:`d` = :attr:`signal_ndim`. :attr:`onesided` flag controls whether to avoid
redundancy in the output results. If set to ``True`` (default), the output will
not be full complex result of shape :math:`(*, 2)`, where :math:`*` is the shape
of :attr:`input`, but instead the last dimension will be halfed as of size
:math:`\lfloor \frac{N_d}{2} \rfloor + 1`.

The inverse of this function is :func:`~torch.irfft`.

.. note::
    For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
    repeatedly running FFT methods on tensors of same geometry with same
    configuration. See :ref:`cufft-plan-cache` for more details on how to
    monitor and control the cache.

.. warning::
    For CPU tensors, this method is currently only available with MKL. Use
    :func:`torch.backends.mkl.is_available` to check if MKL is installed.

Arguments:
    input (Tensor): the input tensor of at least :attr:`signal_ndim` dimensions
    signal_ndim (int): the number of dimensions in each signal.
        :attr:`signal_ndim` can only be 1, 2 or 3
    normalized (bool, optional): controls whether to return normalized results.
        Default: ``False``
    onesided (bool, optional): controls whether to return half of results to
        avoid redundancy. Default: ``True``

Returns:
    Tensor: A tensor containing the real-to-complex Fourier transform result

Example::

    >>> x = torch.randn(5, 5)
    >>> torch.rfft(x, 2).shape
    torch.Size([5, 3, 2])
    >>> torch.rfft(x, 2, onesided=False).shape
    torch.Size([5, 5, 2])

""")


add_docstr(torch.irfft,
           r"""
irfft(input, signal_ndim, normalized=False, onesided=True, signal_sizes=None) -> Tensor

Complex-to-real Inverse Discrete Fourier Transform

This method computes the complex-to-real inverse discrete Fourier transform.
It is mathematically equivalent with :func:`ifft` with differences only in
formats of the input and output.

The argument specifications are almost identical with :func:`~torch.ifft`.
Similar to :func:`~torch.ifft`, if :attr:`normalized` is set to ``True``,
this normalizes the result by multiplying it with
:math:`\sqrt{\prod_{i=1}^K N_i}` so that the operator is unitary, where
:math:`N_i` is the size of signal dimension :math:`i`.

.. note::
    Due to the conjugate symmetry, :attr:`input` do not need to contain the full
    complex frequency values. Roughly half of the values will be sufficient, as
    is the case when :attr:`input` is given by :func:`~torch.rfft` with
    ``rfft(signal, onesided=True)``. In such case, set the :attr:`onesided`
    argument of this method to ``True``. Moreover, the original signal shape
    information can sometimes be lost, optionally set :attr:`signal_sizes` to be
    the size of the original signal (without the batch dimensions if in batched
    mode) to recover it with correct shape.

    Therefore, to invert an :func:`~torch.rfft`, the :attr:`normalized` and
    :attr:`onesided` arguments should be set identically for :func:`~torch.irfft`,
    and preferrably a :attr:`signal_sizes` is given to avoid size mismatch. See the
    example below for a case of size mismatch.

    See :func:`~torch.rfft` for details on conjugate symmetry.

The inverse of this function is :func:`~torch.rfft`.

.. warning::
    Generally speaking, input to this function should contain values
    following conjugate symmetry. Note that even if :attr:`onesided` is
    ``True``, often symmetry on some part is still needed. When this
    requirement is not satisfied, the behavior of :func:`~torch.irfft` is
    undefined. Since :func:`torch.autograd.gradcheck` estimates numerical
    Jacobian with point perturbations, :func:`~torch.irfft` will almost
    certainly fail the check.

.. note::
    For CUDA tensors, an LRU cache is used for cuFFT plans to speed up
    repeatedly running FFT methods on tensors of same geometry with same
    configuration. See :ref:`cufft-plan-cache` for more details on how to
    monitor and control the cache.

.. warning::
    For CPU tensors, this method is currently only available with MKL. Use
    :func:`torch.backends.mkl.is_available` to check if MKL is installed.

Arguments:
    input (Tensor): the input tensor of at least :attr:`signal_ndim` ``+ 1``
        dimensions
    signal_ndim (int): the number of dimensions in each signal.
        :attr:`signal_ndim` can only be 1, 2 or 3
    normalized (bool, optional): controls whether to return normalized results.
        Default: ``False``
    onesided (bool, optional): controls whether :attr:`input` was halfed to avoid
        redundancy, e.g., by :func:`rfft`. Default: ``True``
    signal_sizes (list or :class:`torch.Size`, optional): the size of the original
        signal (without batch dimension). Default: ``None``

Returns:
    Tensor: A tensor containing the complex-to-real inverse Fourier transform result

Example::

    >>> x = torch.randn(4, 4)
    >>> torch.rfft(x, 2, onesided=True).shape
    torch.Size([4, 3, 2])
    >>>
    >>> # notice that with onesided=True, output size does not determine the original signal size
    >>> x = torch.randn(4, 5)

    >>> torch.rfft(x, 2, onesided=True).shape
    torch.Size([4, 3, 2])
    >>>
    >>> # now we use the original shape to recover x
    >>> x
    tensor([[-0.8992,  0.6117, -1.6091, -0.4155, -0.8346],
            [-2.1596, -0.0853,  0.7232,  0.1941, -0.0789],
            [-2.0329,  1.1031,  0.6869, -0.5042,  0.9895],
            [-0.1884,  0.2858, -1.5831,  0.9917, -0.8356]])
    >>> y = torch.rfft(x, 2, onesided=True)
    >>> torch.irfft(y, 2, onesided=True, signal_sizes=x.shape)  # recover x
    tensor([[-0.8992,  0.6117, -1.6091, -0.4155, -0.8346],
            [-2.1596, -0.0853,  0.7232,  0.1941, -0.0789],
            [-2.0329,  1.1031,  0.6869, -0.5042,  0.9895],
            [-0.1884,  0.2858, -1.5831,  0.9917, -0.8356]])

""")


add_docstr(torch.hann_window,
           """
hann_window(window_length, periodic=True, dtype=None, \
layout=torch.strided, device=None, requires_grad=False) -> Tensor
""" + r"""
Hann window function.

.. math::
    w[n] = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{N - 1} \right)\right] =
            \sin^2 \left( \frac{\pi n}{N - 1} \right),

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like
:meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
``torch.hann_window(L, periodic=True)`` equal to
``torch.hann_window(L + 1, periodic=False)[:-1])``.

.. note::
    If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.
""" + r"""
Arguments:
    window_length (int): the size of returned window
    periodic (bool, optional): If True, returns a window to be used as periodic
        function. If False, return a symmetric window.
    {dtype} Only floating point types are supported.
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    {device}
    {requires_grad}

Returns:
    Tensor: A 1-D tensor of size :math:`(\text{{window\_length}},)` containing the window

""".format(**factory_common_args))


add_docstr(torch.hamming_window,
           """
hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, dtype=None, \
layout=torch.strided, device=None, requires_grad=False) -> Tensor
""" + r"""
Hamming window function.

.. math::
    w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1} \right),

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like
:meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
``torch.hamming_window(L, periodic=True)`` equal to
``torch.hamming_window(L + 1, periodic=False)[:-1])``.

.. note::
    If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.

.. note::
    This is a generalized version of :meth:`torch.hann_window`.
""" + r"""
Arguments:
    window_length (int): the size of returned window
    periodic (bool, optional): If True, returns a window to be used as periodic
        function. If False, return a symmetric window.
    alpha (float, optional): The coefficient :math:`\alpha` in the equation above
    beta (float, optional): The coefficient :math:`\beta` in the equation above
    {dtype} Only floating point types are supported.
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    {device}
    {requires_grad}

Returns:
    Tensor: A 1-D tensor of size :math:`(\text{{window\_length}},)` containing the window

""".format(**factory_common_args))


add_docstr(torch.bartlett_window,
           """
bartlett_window(window_length, periodic=True, dtype=None, \
layout=torch.strided, device=None, requires_grad=False) -> Tensor
""" + r"""
Bartlett window function.

.. math::
    w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
        \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
        2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
    \end{cases},

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like
:meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
``torch.bartlett_window(L, periodic=True)`` equal to
``torch.bartlett_window(L + 1, periodic=False)[:-1])``.

.. note::
    If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.
""" + r"""
Arguments:
    window_length (int): the size of returned window
    periodic (bool, optional): If True, returns a window to be used as periodic
        function. If False, return a symmetric window.
    {dtype} Only floating point types are supported.
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    {device}
    {requires_grad}

Returns:
    Tensor: A 1-D tensor of size :math:`(\text{{window\_length}},)` containing the window

""".format(**factory_common_args))


add_docstr(torch.blackman_window,
           """
blackman_window(window_length, periodic=True, dtype=None, \
layout=torch.strided, device=None, requires_grad=False) -> Tensor
""" + r"""
Blackman window function.

.. math::
    w[n] = 0.42 - 0.5 \cos \left( \frac{2 \pi n}{N - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{N - 1} \right)

where :math:`N` is the full window size.

The input :attr:`window_length` is a positive integer controlling the
returned window size. :attr:`periodic` flag determines whether the returned
window trims off the last duplicate value from the symmetric window and is
ready to be used as a periodic window with functions like
:meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
above formula is in fact :math:`\text{window\_length} + 1`. Also, we always have
``torch.blackman_window(L, periodic=True)`` equal to
``torch.blackman_window(L + 1, periodic=False)[:-1])``.

.. note::
    If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.
""" + r"""
Arguments:
    window_length (int): the size of returned window
    periodic (bool, optional): If True, returns a window to be used as periodic
        function. If False, return a symmetric window.
    {dtype} Only floating point types are supported.
    layout (:class:`torch.layout`, optional): the desired layout of returned window tensor. Only
          ``torch.strided`` (dense layout) is supported.
    {device}
    {requires_grad}

Returns:
    Tensor: A 1-D tensor of size :math:`(\text{{window\_length}},)` containing the window

""".format(**factory_common_args))


add_docstr(torch.unbind,
           r"""
unbind(input, dim=0) -> seq

Removes a tensor dimension.

Returns a tuple of all slices along a given dimension, already without it.

Arguments:
    input (Tensor): the tensor to unbind
    dim (int): dimension to remove

Example::

    >>> torch.unbind(torch.tensor([[1, 2, 3],
    >>>                            [4, 5, 6],
    >>>                            [7, 8, 9]]))
    (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))
""")


add_docstr(torch.combinations,
           r"""
combinations(input, r=2, with_replacement=False) -> seq

Compute combinations of length :math:`r` of the given tensor. The behavior is similar to
python's `itertools.combinations` when `with_replacement` is set to `False`, and
`itertools.combinations_with_replacement` when `with_replacement` is set to `True`.

Arguments:
    input (Tensor): 1D vector.
    r (int, optional): number of elements to combine
    with_replacement (boolean, optional): whether to allow duplication in combination

Returns:
    Tensor: A tensor equivalent to converting all the input tensors into lists, do
    `itertools.combinations` or `itertools.combinations_with_replacement` on these
    lists, and finally convert the resulting list into tensor.

Example::

    >>> a = [1, 2, 3]
    >>> list(itertools.combinations(a, r=2))
    [(1, 2), (1, 3), (2, 3)]
    >>> list(itertools.combinations(a, r=3))
    [(1, 2, 3)]
    >>> list(itertools.combinations_with_replacement(a, r=2))
    [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    >>> tensor_a = torch.tensor(a)
    >>> torch.combinations(tensor_a)
    tensor([[1, 2],
            [1, 3],
            [2, 3]])
    >>> torch.combinations(tensor_a, r=3)
    tensor([[1, 2, 3]])
    >>> torch.combinations(tensor_a, with_replacement=True)
    tensor([[1, 1],
            [1, 2],
            [1, 3],
            [2, 2],
            [2, 3],
            [3, 3]])
""")

add_docstr(torch.trapz,
           r"""
.. function:: trapz(y, x, *, dim=-1) -> Tensor

Estimate :math:`\int y\,dx` along `dim`, using the trapezoid rule.

Arguments:
    y (Tensor): The values of the function to integrate
    x (Tensor): The points at which the function `y` is sampled.
        If `x` is not in ascending order, intervals on which it is decreasing
        contribute negatively to the estimated integral (i.e., the convention
        :math:`\int_a^b f = -\int_b^a f` is followed).
    dim (int): The dimension along which to integrate.
        By default, use the last dimension.

Returns:
    A Tensor with the same shape as the input, except with `dim` removed.
    Each element of the returned tensor represents the estimated integral
    :math:`\int y\,dx` along `dim`.

Example::

    >>> y = torch.randn((2, 3))
    >>> y
    tensor([[-2.1156,  0.6857, -0.2700],
            [-1.2145,  0.5540,  2.0431]])
    >>> x = torch.tensor([[1, 3, 4], [1, 2, 3]])
    >>> torch.trapz(y, x)
    tensor([-1.2220,  0.9683])

.. function:: trapz(y, *, dx=1, dim=-1) -> Tensor

As above, but the sample points are spaced uniformly at a distance of `dx`.

Arguments:
    y (Tensor): The values of the function to integrate
    dx (float): The distance between points at which `y` is sampled.
    dim (int): The dimension along which to integrate.
        By default, use the last dimension.

Returns:
    A Tensor with the same shape as the input, except with `dim` removed.
    Each element of the returned tensor represents the estimated integral
    :math:`\int y\,dx` along `dim`.
""")

add_docstr(torch.repeat_interleave,
           r"""
.. function:: repeat_interleave(input, repeats, dim=None) -> Tensor

Repeat elements of a tensor.

.. warning::

    This is different from :func:`torch.repeat` but similar to `numpy.repeat`.

Args:
    {input}
    repeats (Tensor or int): The number of repetitions for each element.
        repeats is broadcasted to fit the shape of the given axis.
    dim (int, optional): The dimension along which to repeat values.
        By default, use the flattened input array, and return a flat output
        array.

Returns:
    Tensor: Repeated tensor which has the same shape as input, except along the
     given axis.

Example::

    >>> x = torch.tensor([1, 2, 3])
    >>> x.repeat_interleave(2)
    tensor([1, 1, 2, 2, 3, 3])
    >>> y = torch.tensor([[1, 2], [3, 4]])
    >>> torch.repeat_interleave(y, 2)
    tensor([1, 1, 2, 2, 3, 3, 4, 4])
    >>> torch.repeat_interleave(y, 3, dim=1)
    tensor([[1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4]])
    >>> torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
    tensor([[1, 2],
            [3, 4],
            [3, 4]])

.. function:: repeat_interleave(repeats) -> Tensor

If the `repeats` is `tensor([n1, n2, n3, ...])`, then the output will be
`tensor([0, 0, ..., 1, 1, ..., 2, 2, ..., ...])` where `0` appears `n1` times,
`1` appears `n2` times, `2` appears `n3` times, etc.
""".format(**common_args))


add_docstr(torch._C.Generator,
           r"""
Generator(device='cpu') -> Generator

Creates and returns a generator object which manages the state of the algorithm that
produces pseudo random numbers. Used as a keyword argument in many :ref:`inplace-random-sampling`
functions.

Arguments:
    device (:class:`torch.device`, optional): the desired device for the generator.

Returns:
    Generator: An torch.Generator object.

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cuda = torch.Generator(device='cuda')
""")


add_docstr(torch._C.Generator.set_state,
           r"""
Generator.set_state(new_state) -> void

Sets the Generator state.

Arguments:
    new_state (torch.ByteTensor): The desired state.

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cpu_other = torch.Generator()
    >>> g_cpu.set_state(g_cpu_other.get_state())
""")


add_docstr(torch._C.Generator.get_state,
           r"""
Generator.get_state() -> Tensor

Returns the Generator state as a ``torch.ByteTensor``.

Returns:
    Tensor: A ``torch.ByteTensor`` which contains all the necessary bits
    to restore a Generator to a specific point in time.

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cpu.get_state()
""")


add_docstr(torch._C.Generator.manual_seed,
           r"""
Generator.manual_seed(seed) -> Generator

Sets the seed for generating random numbers. Returns a `torch.Generator` object.
It is recommended to set a large seed, i.e. a number that has a good balance of 0
and 1 bits. Avoid having many 0 bits in the seed.

Arguments:
    seed (int): The desired seed.

Returns:
    Generator: An torch.Generator object.

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cpu.manual_seed(2147483647)
""")


add_docstr(torch._C.Generator.initial_seed,
           r"""
Generator.initial_seed() -> int

Returns the initial seed for generating random numbers.

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cpu.initial_seed()
    2147483647
""")


add_docstr(torch._C.Generator.seed,
           r"""
Generator.seed() -> int

Gets a non-deterministic random number from std::random_device or the current
time and uses it to seed a Generator.

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cpu.seed()
    1516516984916
""")


add_docstr(torch._C.Generator.device,
           r"""
Generator.device -> device

Gets the current device of the generator.

Example::

    >>> g_cpu = torch.Generator()
    >>> g_cpu.device
    device(type='cpu')
""")
