"""Adds docstrings to functions defined in the torch._C"""

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
    # Split by indents. Assumes each arg starts on a new line with 4 spaces.
    kwargs = [section.strip() for section in desc.split('\n   ')]
    kwargs = [section for section in kwargs if len(section) > 0]
    return {desc.split(' ')[0]: desc for desc in kwargs}


factory_common_args = parse_kwargs("""
    out (Tensor, optional): the output tensor
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if None, uses a global default (see :func:`torch.set_default_tensor_type`)
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
""")

factory_like_common_args = parse_kwargs("""
    input (Tensor): the size of :attr:`input` will determine size of the output tensor
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if None, defaults to the layout of :attr:`input`.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if None, defaults to the dtype of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
""")

add_docstr(torch.abs,
           r"""
abs(input, out=None) -> Tensor

Computes the element-wise absolute value of the given :attr:`input` tensor.

.. math::
    \text{out}_{i} = |\text{input}_{i}|

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> torch.abs(torch.FloatTensor([-1, -2, 3]))
    tensor([ 1.0000,  2.0000,  3.0000])
""")

add_docstr(torch.acos,
           r"""
acos(input, out=None) -> Tensor

Returns a new tensor with the arccosine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \cos^{-1}(\text{input}_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.1368, -1.1582,  1.2223, -1.4254])
    >>> torch.acos(a)
    tensor([ 1.7080,     nan,     nan,     nan])
""")

add_docstr(torch.add,
           r"""
.. function:: add(input, value, out=None)

Adds the scalar :attr:`value` to each element of the input :attr:`input`
and returns a new resulting tensor.

.. math::
    out = input + value

If :attr:`input` is of type FloatTensor or DoubleTensor, :attr:`value` must be
a real number, otherwise it should be an integer.

Args:
    input (Tensor): the input tensor
    value (Number): the number to be added to each element of :attr:`input`

Keyword arguments:
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8486,  0.4360,  0.1347,  1.9301])
    >>> torch.add(a, 20)
    tensor([ 20.8486,  20.4360,  20.1347,  21.9301])

.. function:: add(input, value=1, other, out=None)

Each element of the tensor :attr:`other` is multiplied by the scalar
:attr:`value` and added to each element of the tensor :attr:`input`.
The resulting tensor is returned.

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

.. math::
    out = input + value \times other

If :attr:`other` is of type FloatTensor or DoubleTensor, :attr:`value` must be
a real number, otherwise it should be an integer.

Args:
    input (Tensor): the first input tensor
    value (Number): the scalar multiplier for :attr:`other`
    other (Tensor): the second input tensor

Keyword arguments:
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.8758, -0.1567,  0.1016, -1.3956])
    >>> b = torch.randn(4, 1)
    >>> b
    tensor([[ 0.3001],
            [-0.6742],
            [ 0.0372],
            [-0.0529]])
    >>> torch.add(a, 10, b)
    tensor([[ 2.1253,  2.8444,  3.1027,  1.6055],
            [-7.6181, -6.8991, -6.6408, -8.1380],
            [-0.5034,  0.2156,  0.4739, -1.0233],
            [-1.4050, -0.6860, -0.4277, -1.9249]])
""")

add_docstr(torch.addbmm,
           r"""
addbmm(beta=1, mat, alpha=1, batch1, batch2, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored
in :attr:`batch1` and :attr:`batch2`,
with a reduced add step (all matrix multiplications get accumulated
along the first dimension).
:attr:`mat` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the
same number of matrices.

If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
:math:`(b \times m \times p)` tensor, :attr:`mat` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

.. math::
    out = \beta\ mat + \alpha\ (\sum_{i=0}^{b} batch1_i \mathbin{@} batch2_i)

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and :attr:`alpha`
must be real numbers, otherwise they should be integers.

Args:
    beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
    mat (Tensor): matrix to be added
    alpha (Number, optional): multiplier for `batch1 @ batch2` (:math:`\alpha`)
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied
    out (Tensor, optional): the output tensor

Example::

    >>> M = torch.randn(3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.addbmm(M, batch1, batch2)
    tensor([[  2.8224,   4.5371,   6.8394,  -4.6322,  -1.2176],
            [  1.3098,   6.1356, -10.6471,  -7.4279,  -4.2940],
            [ -6.2601,  -6.5135,  -8.8350,   2.3402,  -4.2389]])
""")

add_docstr(torch.addcdiv,
           r"""
addcdiv(tensor, value=1, tensor1, tensor2, out=None) -> Tensor

Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
multiply the result by the scalar :attr:`value` and add it to :attr:`tensor`.

.. math::
    out_i = tensor_i + value \times \frac{tensor1_i}{tensor2_i}

The shapes of :attr:`tensor`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    tensor (Tensor): the tensor to be added
    value (Number, optional): multiplier for :math:`tensor1 ./ tensor2`
    tensor1 (Tensor): the numerator tensor
    tensor2 (Tensor): the denominator tensor
    out (Tensor, optional): the output tensor

Example::

    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcdiv(t, 0.1, t1, t2)
    tensor([[ 1.4924, -0.2334, -0.5559],
            [ 1.3655, -0.3157, -0.4686],
            [ 1.4688, -0.2487, -0.5397]])
""")

add_docstr(torch.addcmul,
           r"""
addcmul(tensor, value=1, tensor1, tensor2, out=None) -> Tensor

Performs the element-wise multiplication of :attr:`tensor1`
by :attr:`tensor2`, multiply the result by the scalar :attr:`value`
and add it to :attr:`tensor`.

.. math::
    out_i = tensor_i + value \times tensor1_i \times tensor2_i

The shapes of :attr:`tensor`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    tensor (Tensor): the tensor to be added
    value (Number, optional): multiplier for :math:`tensor1 .* tensor2`
    tensor1 (Tensor): the tensor to be multiplied
    tensor2 (Tensor): the tensor to be multiplied
    out (Tensor, optional): the output tensor

Example::

    >>> t = torch.randn(1, 3)
    >>> t1 = torch.randn(3, 1)
    >>> t2 = torch.randn(1, 3)
    >>> torch.addcmul(t, 0.1, t1, t2)
    tensor([[-3.2876,  0.1203, -0.1891],
            [-3.2980,  0.1214, -0.1919],
            [-3.3554,  0.1272, -0.2073]])
""")

add_docstr(torch.addmm,
           r"""
addmm(beta=1, mat, alpha=1, mat1, mat2, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
The matrix :attr:`mat` is added to the final result.

If :attr:`mat1` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
:math:`(m \times p)` tensor, then :attr:`mat` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat1` and :attr`mat2` and the added matrix :attr:`mat` respectively.

.. math::
    out = \beta\ mat + \alpha\ (mat1_i \mathbin{@} mat2_i)

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

Args:
    beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
    mat (Tensor): matrix to be added
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)
    mat1 (Tensor): the first matrix to be multiplied
    mat2 (Tensor): the second matrix to be multiplied
    out (Tensor, optional): the output tensor

Example::

    >>> M = torch.randn(2, 3)
    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.addmm(M, mat1, mat2)
    tensor([[ 1.5677,  2.3343,  1.3511],
            [ 2.0484,  3.0190,  1.3055]])
""")

add_docstr(torch.addmv,
           r"""
addmv(beta=1, tensor, alpha=1, mat, vec, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`mat` and
the vector :attr:`vec`.
The vector :attr:`tensor` is added to the final result.

If :attr:`mat` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
size `m`, then :attr:`tensor` must be
:ref:`broadcastable <broadcasting-semantics>` with a 1-D tensor of size `n` and
:attr:`out` will be 1-D tensor of size `n`.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat` and :attr:`vec` and the added tensor :attr:`tensor` respectively.

.. math::
    out = \beta\ tensor + \alpha\ (mat \mathbin{@} vec)

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers

Args:
    beta (Number, optional): multiplier for :attr:`tensor` (:math:`\beta`)
    tensor (Tensor): vector to be added
    alpha (Number, optional): multiplier for :math:`mat @ vec` (:math:`\alpha`)
    mat (Tensor): matrix to be multiplied
    vec (Tensor): vector to be multiplied
    out (Tensor, optional): the output tensor

Example::

    >>> M = torch.randn(2)
    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.addmv(M, mat, vec)
    tensor([-0.8592, -1.2799])
""")

add_docstr(torch.addr,
           r"""
addr(beta=1, mat, alpha=1, vec1, vec2, out=None) -> Tensor

Performs the outer-product of vectors :attr:`vec1` and :attr:`vec2`
and adds it to the matrix :attr:`mat`.

Optional values :attr:`beta` and :attr:`alpha` are scaling factors on the
outer product between :attr:`vec1` and :attr:`vec2` and the added matrix
:attr:`mat` respectively.

.. math::
    out = \beta\ mat + \alpha\ (vec1 \otimes vec2)

If :attr:`vec1` is a vector of size `n` and :attr:`vec2` is a vector
of size `m`, then :attr:`mat` must be
:ref:`broadcastable <broadcasting-semantics>` with a matrix of size
:math:`(n \times m)` and :attr:`out` will be a matrix of size
:math:`(n \times m)`.

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers

Args:
    beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
    mat (Tensor): matrix to be added
    alpha (Number, optional): multiplier for :math:`vec1 \otimes vec2` (:math:`\alpha`)
    vec1 (Tensor): the first vector of the outer product
    vec2 (Tensor): the second vector of the outer product
    out (Tensor, optional): the output tensor

Example::

    >>> vec1 = torch.arange(1, 4)
    >>> vec2 = torch.arange(1, 3)
    >>> M = torch.zeros(3, 2)
    >>> torch.addr(M, vec1, vec2)
    tensor([[ 1.0000,  2.0000],
            [ 2.0000,  4.0000],
            [ 3.0000,  6.0000]])
""")

add_docstr(torch.asin,
           r"""
asin(input, out=None) -> Tensor

Returns a new tensor with the arcsine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin^{-1}(\text{input}_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.7086,  0.5379,  0.1389, -0.4416])
    >>> torch.asin(a)
    tensor([ 0.7874,  0.5680,  0.1393, -0.4574])
""")

add_docstr(torch.atan,
           r"""
atan(input, out=None) -> Tensor

Returns a new tensor with the arctangent  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \tan^{-1}(\text{input}_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.5334, -0.0945, -0.8493,  0.5141])
    >>> torch.atan(a)
    tensor([ 0.4900, -0.0942, -0.7041,  0.4748])
""")

add_docstr(torch.atan2,
           r"""
atan2(input1, input2, out=None) -> Tensor

Returns a new tensor with the arctangent of the elements of :attr:`input1`
and :attr:`input2`.

The shapes of :attr:`input1` and :attr:`input2` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    input1 (Tensor): the first input tensor
    input2 (Tensor): the second input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-1.9168, -0.1342, -0.3433,  1.1039])
    >>> torch.atan2(a, torch.randn(4))
    tensor([-1.5080, -1.6206, -2.8957,  0.9546])
""")

add_docstr(torch.baddbmm,
           r"""
baddbmm(beta=1, mat, alpha=1, batch1, batch2, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices in :attr:`batch1`
and :attr:`batch2`.
:attr:`mat` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the same
number of matrices.

If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
:math:`(b \times m \times p)` tensor, then :attr:`mat` must be
:ref:`broadcastable <broadcasting-semantics>` with a
:math:`(b \times n \times p)` tensor and :attr:`out` will be a
:math:`(b \times n \times p)` tensor. Both :attr:`alpha` and :attr:`beta` mean the
same as the scaling factors used in :meth:`torch.addbmm`.

.. math::
    out_i = \beta\ mat_i + \alpha\ (batch1_i \mathbin{@} batch2_i)

For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.

Args:
    beta (Number, optional): multiplier for :attr:`mat` (:math:`\beta`)
    mat (Tensor): the tensor to be added
    alpha (Number, optional): multiplier for `batch1 @ batch2` (:math:`\alpha`)
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied
    out (Tensor, optional): the output tensor

Example::

    >>> M = torch.randn(10, 3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.baddbmm(M, batch1, batch2).size()
    torch.Size([10, 3, 5])
""")

add_docstr(torch.bernoulli,
           r"""
bernoulli(input, out=None) -> Tensor

Draws binary random numbers (0 or 1) from a Bernoulli distribution.

The :attr:`input` tensor should be a tensor containing probabilities
to be used for drawing the binary random number.
Hence, all values in :attr:`input` have to be in the range:
:math:`0 \leq \text{input}_i \leq 1`.

The :math:`\text{i}^{th}` element of the output tensor will draw a
value `1` according to the :math:`\text{i}^{th}` probability value given
in :attr:`input`.

.. math::
    \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})

The returned :attr:`out` tensor only has values 0 or 1 and is of the same
shape as :attr:`input`

Args:
    input (Tensor): the input tensor of probability values for the Bernoulli distribution
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.Tensor(3, 3).uniform_(0, 1) # generate a uniform random matrix with range [0, 1]
    >>> a
    tensor([[ 0.7298,  0.7832,  0.8442],
            [ 0.1343,  0.8033,  0.7676],
            [ 0.6618,  0.5497,  0.6643]])
    >>> torch.bernoulli(a)
    tensor([[ 1.0000,  1.0000,  1.0000],
            [ 0.0000,  0.0000,  1.0000],
            [ 1.0000,  1.0000,  1.0000]])

    >>> a = torch.ones(3, 3) # probability of drawing "1" is 1
    >>> torch.bernoulli(a)
    tensor([[ 1.0000,  1.0000,  1.0000],
            [ 1.0000,  1.0000,  1.0000],
            [ 1.0000,  1.0000,  1.0000]])
    >>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
    >>> torch.bernoulli(a)
    tensor([[ 0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000]])
""")

add_docstr(torch.bmm,
           r"""
bmm(batch1, batch2, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored in :attr:`batch1`
and :attr:`batch2`.

:attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing
the same number of matrices.

If :attr:`batch1` is a :math:`(b \times n \times m)` tensor, :attr:`batch2` is a
:math:`(b \times m \times p)` tensor, :attr:`out` will be a
:math:`(b \times n \times p)` tensor.

.. math::
    out_i = batch1_i \mathbin{@} batch2_i

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Args:
    batch1 (Tensor): the first batch of matrices to be multiplied
    batch2 (Tensor): the second batch of matrices to be multiplied
    out (Tensor, optional): the output tensor

Example::

    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> res = torch.bmm(batch1, batch2)
    >>> res.size()
    torch.Size([10, 3, 5])
""")

add_docstr(torch.stack,
           r"""
stack(seq, dim=0, out=None) -> Tensor

Concatenates sequence of tensors along a new dimension.

All tensors need to be of the same size.

Arguments:
    seq (sequence of Tensors): sequence of tensors to concatenate
    dim (int): dimension to insert. Has to be between 0 and the number
        of dimensions of concatenated tensors (inclusive)
    out (Tensor, optional): the output tensor
""")

add_docstr(torch.chunk,
           r"""
chunk(tensor, chunks, dim=0) -> List of Tensors

Splits a tensor into a specific number of chunks.

Last chunk will be smaller if the tensor size along the given dimension
:attr:`dim` is not divisible by :attr:`chunks`.

Arguments:
    tensor (Tensor): the tensor to split
    chunks (int): number of chunks to return
    dim (int): dimension along which to split the tensor
""")

add_docstr(torch.cat,
           r"""
cat(seq, dim=0, out=None) -> Tensor

Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
All tensors must either have the same shape (except in the concatenating
dimension) or be empty.

:func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
and :func:`torch.chunk`.

:func:`torch.cat` can be best understood via examples.

Args:
    seq (sequence of Tensors): any python sequence of tensors of the same type.
        Non-empty tensors provided must have the same shape, except in the
        cat dimension.
    dim (int, optional): the dimension over which the tensors are concatenated
    out (Tensor, optional): the output tensor

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.5857,  0.6747,  0.8185],
            [ 0.0201, -0.2561,  0.3456]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.5857,  0.6747,  0.8185],
            [ 0.0201, -0.2561,  0.3456],
            [ 0.5857,  0.6747,  0.8185],
            [ 0.0201, -0.2561,  0.3456],
            [ 0.5857,  0.6747,  0.8185],
            [ 0.0201, -0.2561,  0.3456]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.5857,  0.6747,  0.8185,  0.5857,  0.6747,  0.8185,  0.5857,
              0.6747,  0.8185],
            [ 0.0201, -0.2561,  0.3456,  0.0201, -0.2561,  0.3456,  0.0201,
             -0.2561,  0.3456]])
""")

add_docstr(torch.ceil,
           r"""
ceil(input, out=None) -> Tensor

Returns a new tensor with the ceil of the elements of :attr:`input`,
the smallest integer greater than or equal to each element.

.. math::
    \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil = \left\lfloor \text{input}_{i} \right\rfloor + 1

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.7487, -1.2559, -0.5207,  1.9312])
    >>> torch.ceil(a)
    tensor([-0.0000, -1.0000, -0.0000,  2.0000])
""")

add_docstr(torch.reciprocal,
           r"""
reciprocal(input, out=None) -> Tensor

Returns a new tensor with the reciprocal of the elements of :attr:`input`

.. math::
    \text{out}_{i} = \frac{1}{\text{input}_{i}}

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.6609, -2.1292, -1.7494,  1.1981])
    >>> torch.reciprocal(a)
    tensor([-1.5131, -0.4696, -0.5716,  0.8346])
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

If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, args :attr:`min`
and :attr:`max` must be real numbers, otherwise they should be integers.

Args:
    input (Tensor): the input tensor
    min (Number): lower-bound of the range to be clamped to
    max (Number): upper-bound of the range to be clamped to
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8282,  1.6225,  1.9388,  2.9110])
    >>> torch.clamp(a, min=-0.5, max=0.5)
    tensor([ 0.5000,  0.5000,  0.5000,  0.5000])

.. function:: clamp(input, *, min, out=None) -> Tensor

Clamps all elements in :attr:`input` to be larger or equal :attr:`min`.

If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`value`
should be a real number, otherwise it should be an integer.

Args:
    input (Tensor): the input tensor
    value (Number): minimal value of each element in the output
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.0593, -0.4008,  1.1595, -0.2758])
    >>> torch.clamp(a, min=0.5)
    tensor([ 0.5000,  0.5000,  1.1595,  0.5000])

.. function:: clamp(input, *, max, out=None) -> Tensor

Clamps all elements in :attr:`input` to be smaller or equal :attr:`max`.

If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`value`
should be a real number, otherwise it should be an integer.

Args:
    input (Tensor): the input tensor
    value (Number): maximal value of each element in the output
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 1.9670,  0.1386,  0.5641, -1.5649])
    >>> torch.clamp(a, max=0.5)
    tensor([ 0.5000,  0.1386,  0.5000, -1.5649])
""")

add_docstr(torch.cos,
           r"""
cos(input, out=None) -> Tensor

Returns a new tensor with the cosine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \cos(\text{input}_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.9628, -0.0437,  0.7992,  2.2426])
    >>> torch.cos(a)
    tensor([ 0.5712,  0.9990,  0.6973, -0.6224])
""")

add_docstr(torch.cosh,
           r"""
cosh(input, out=None) -> Tensor

Returns a new tensor with the hyperbolic cosine  of the elements of
:attr:`input`.

.. math::
    \text{out}_{i} = \cosh(\text{input}_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.1596,  0.0238,  0.5990,  1.7250])
    >>> torch.cosh(a)
    tensor([ 1.0128,  1.0003,  1.1848,  2.8954])
""")

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
    input (Tensor): the input tensor
    other (Tensor): the second input tensor
    dim  (int, optional): the dimension to take the cross-product in.
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4, 3)
    >>> a
    tensor([[-0.2174, -0.2861,  1.2017],
            [-0.0878, -0.5996,  1.1663],
            [-0.1758,  0.6903,  0.2780],
            [ 1.5252,  1.0001, -0.5286]])
    >>> b = torch.randn(4, 3)
    >>> b
    tensor([[-3.2886, -0.7895, -0.1608],
            [ 0.5153, -0.6462,  0.0486],
            [ 0.4277,  1.2727,  0.9031],
            [-1.6784,  0.5861,  0.9299]])
    >>> torch.cross(a, b, dim=1)
    tensor([[ 0.9947, -3.9870, -0.7694],
            [ 0.7245,  0.6052,  0.3657],
            [ 0.2696,  0.2776, -0.5189],
            [ 1.2399, -0.5310,  2.5726]])
    >>> torch.cross(a, b)
    tensor([[ 0.9947, -3.9870, -0.7694],
            [ 0.7245,  0.6052,  0.3657],
            [ 0.2696,  0.2776, -0.5189],
            [ 1.2399, -0.5310,  2.5726]])
""")

add_docstr(torch.cumprod,
           r"""
cumprod(input, dim, out=None) -> Tensor

Returns the cumulative product of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be
a vector of size N, with elements.

.. math::
    y_i = x_1 \times x_2\times x_3\times \dots \times x_i

Args:
    input (Tensor): the input tensor
    dim  (int): the dimension to do the operation over
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([-1.2165,  1.0188, -0.1334,  0.3236,  0.0236,  1.5730,  0.0802,
            -0.5040,  0.0378, -1.5511])
    >>> torch.cumprod(a, dim=0)
    tensor([-1.2165e+00, -1.2394e+00,  1.6530e-01,  5.3495e-02,  1.2635e-03,
             1.9876e-03,  1.5935e-04, -8.0316e-05, -3.0370e-06,  4.7106e-06])

    >>> a[5] = 0.0
    >>> torch.cumprod(a, dim=0)
    tensor([-1.2165, -1.2394,  0.1653,  0.0535,  0.0013,  0.0000,  0.0000,
            -0.0000, -0.0000,  0.0000])
""")

add_docstr(torch.cumsum,
           r"""
cumsum(input, dim, out=None) -> Tensor

Returns the cumulative sum of elements of :attr:`input` in the dimension
:attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be
a vector of size N, with elements.

.. math::
    y_i = x_1 + x_2 + x_3 + \dots + x_i

Args:
    input (Tensor): the input tensor
    dim  (int): the dimension to do the operation over
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(10)
    >>> a
    tensor([-1.0478,  0.1700, -0.6439, -0.0426,  0.0999, -0.0719,  0.6987,
             0.5395,  0.5180, -1.6904])
    >>> torch.cumsum(a, dim=0)
    tensor([-1.0478, -0.8778, -1.5217, -1.5643, -1.4643, -1.5363, -0.8375,
            -0.2981,  0.2199, -1.4705])
""")

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
    input (Tensor): the input tensor
    diagonal (int, optional): the diagonal to consider
    out (Tensor, optional): the output tensor

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
""")

add_docstr(torch.diagflat,
           r"""
diagflat(input, diagonal=0) -> Tensor

- If :attr:`input` is a vector (1-D tensor), then returns a 2-D square tensor
  with the elements of :attr:`input` as the diagonal.
- If :attr:`input` is a tensor with more than one dimension, then returns a
  2-D tensor with diagonal elements equal to a flattened :attr:`input`.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.
- If :attr:`offset` > 0, it is above the main diagonal.
- If :attr:`offset` < 0, it is below the main diagonal.

Args:
    input (Tensor): the input tensor
    offset (int, optional): the diagonal to consider. Default: 0 (main
        diagonal).

Examples::

    >>> a = torch.randn(3)
    >>> a
    tensor([-0.1576, -0.4283, -0.7582])
    >>> torch.diagflat(a)
    tensor([[-0.1576,  0.0000,  0.0000],
            [ 0.0000, -0.4283,  0.0000],
            [ 0.0000,  0.0000, -0.7582]])
    >>> torch.diagflat(a, 1)
    tensor([[ 0.0000, -0.1576,  0.0000,  0.0000],
            [ 0.0000,  0.0000, -0.4283,  0.0000],
            [ 0.0000,  0.0000,  0.0000, -0.7582],
            [ 0.0000,  0.0000,  0.0000,  0.0000]])

    >>> a = torch.randn(2, 2)
    >>> a
    tensor([[ 2.0131, -1.5898],
            [ 2.1878, -0.4076]])
    >>> torch.diagflat(a)
    tensor([[ 2.0131,  0.0000,  0.0000,  0.0000],
            [ 0.0000, -1.5898,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  2.1878,  0.0000],
            [ 0.0000,  0.0000,  0.0000, -0.4076]])
""")

add_docstr(torch.diagonal,
           r"""
diagonal(input, offset=0) -> Tensor

Returns a 1-D tensor with the diagonal elements of :attr:`input`.

The argument :attr:`offset` controls which diagonal to consider:

- If :attr:`offset` = 0, it is the main diagonal.
- If :attr:`offset` > 0, it is above the main diagonal.
- If :attr:`offset` < 0, it is below the main diagonal.

Args:
    input (Tensor): the input tensor. Must be 2-dimensional.
    offset (int, optional): which diagonal to consider. Default: 0
        (main diagonal).

Examples::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[ 0.4191,  1.6296,  0.3231],
            [-0.5476,  0.5836, -0.5146],
            [-0.5777, -0.7556, -0.9061]])


    >>> torch.diagonal(a, 0)
    tensor([ 0.4191,  0.5836, -0.9061])


    >>> torch.diagonal(a, 1)
    tensor([ 1.6296, -0.5146])


""")

add_docstr(torch.dist,
           r"""
dist(input, other, p=2) -> Tensor

Returns the p-norm of (:attr:`input` - :attr:`other`)

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the input tensor
    other (Tensor): the Right-hand-side input tensor
    p (float, optional): the norm to be computed

Example::

    >>> x = torch.randn(4)
    >>> x
    tensor([ 1.9799, -0.0406, -0.3442, -0.6584])
    >>> y = torch.randn(4)
    >>> y
    tensor([-0.3176,  0.0261,  0.5617,  0.4493])
    >>> torch.dist(x, y, 3.5)
    tensor(2.3708)
    >>> torch.dist(x, y, 3)
    tensor(2.4232)
    >>> torch.dist(x, y, 0)
    tensor(inf)
    >>> torch.dist(x, y, 1)
    tensor(4.3778)
""")

add_docstr(torch.div,
           r"""
.. function:: div(input, value, out=None) -> Tensor

Divides each element of the input :attr:`input` with the scalar :attr:`value`
and returns a new resulting tensor.

.. math::
    out_i = \frac{input_i}{value}

If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`value`
should be a real number, otherwise it should be an integer

Args:
    input (Tensor): the input tensor
    value (Number): the number to be divided to each element of :attr:`input`
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([-0.3993,  1.1122, -0.3620,  0.6232, -2.2470])
    >>> torch.div(a, 0.5)
    tensor([-0.7985,  2.2244, -0.7240,  1.2463, -4.4940])

.. function:: div(input, other, out=None) -> Tensor

Each element of the tensor :attr:`input` is divided by each element
of the tensor :attr:`other`. The resulting tensor is returned. The shapes of
:attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

.. math::
    out_i = \frac{input_i}{other_i}

Args:
    input (Tensor): the numerator tensor
    other (Tensor): the denominator tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-1.4741,  0.5542, -0.4958,  0.4914],
            [-1.8274,  0.8200,  0.1076, -0.0882],
            [ 0.1202, -1.4301,  1.7035,  1.2488],
            [-0.1365,  0.3649,  0.3313, -0.0526]])
    >>> b = torch.randn(4)
    >>> b
    tensor([ 0.6238,  1.2417, -0.3095,  0.0717])
    >>> torch.div(a, b)
    tensor([[ -2.3631,   0.4463,   1.6017,   6.8490],
            [ -2.9295,   0.6604,  -0.3476,  -1.2288],
            [  0.1927,  -1.1517,  -5.5033,  17.4071],
            [ -0.2187,   0.2938,  -1.0703,  -0.7328]])
""")

add_docstr(torch.dot,
           r"""
dot(tensor1, tensor2) -> Tensor

Computes the dot product (inner product) of two tensors.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

Example::

    >>> torch.dot(torch.Tensor([2, 3]), torch.Tensor([2, 1]))
    tensor(7.0000)
""")

add_docstr(torch.eig,
           r"""
eig(a, eigenvectors=False, out=None) -> (Tensor, Tensor)

Computes the eigenvalues and eigenvectors of a real square matrix.

Args:
    a (Tensor): the square matrix for which the eigenvalues and eigenvectors will be computed
    eigenvectors (bool): ``True`` to compute both eigenvalues and eigenvectors;
        otherwise, only eigenvalues will be computed
    out (tuple, optional): the output tensors

Returns:
    (Tensor, Tensor): A tuple containing

        - **e** (*Tensor*): the right eigenvalues of ``a``
        - **v** (*Tensor*): the eigenvectors of ``a`` if ``eigenvectors`` is ``True``; otherwise an empty tensor
""")

add_docstr(torch.einsum,
           r"""
einsum(equation, operands) -> Tensor

This function provides a way of computing multilinear expressions (i.e. sums of products) using the
Einstein summation convention.

Args:
    equation (string): The equation is given in terms of lower case letters (indices) to be associated
           with each dimension of the operands and result. The left hand side lists the operands
           dimensions, separated by commas. There should be one index letter per tensor dimension.
           The right hand side follows after `->` and gives the indices for the output.
           If the `->` and right hand side are omitted, it implicitly defined as the alphabetically
           sorted list of all indices appearing exactly once in the left hand side.
           The indices not apprearing in the output are summed over after multiplying the operands
           entries.
           `einsum` does not implement diagonals (multiple occurences of a single index for one tensor,
           e.g. `ii->i`) and ellipses (`...`).
    operands (list of Tensors): The operands to compute the Einstein sum of.
           Note that the operands are passed as a list, not as individual arguments.

Examples::

    >>> x = torch.randn(5)
    >>> y = torch.randn(4)
    >>> torch.einsum('i,j->ij', (x,y))  # outer product
    tensor([[-1.3023, -2.3609,  0.3160, -2.4465],
            [-1.3585, -2.4627,  0.3296, -2.5520],
            [ 0.8542,  1.5486, -0.2073,  1.6048],
            [-0.6734, -1.2208,  0.1634, -1.2651],
            [ 0.5103,  0.9251, -0.1238,  0.9586]])


    >>> A = torch.randn(3,5,4)
    >>> l = torch.randn(2,5)
    >>> r = torch.randn(2,4)
    >>> torch.einsum('bn,anm,bm->ba', (l,A,r)) # compare torch.nn.functional.bilinear
    tensor([[-1.1338, -0.5169, -0.9493],
            [ 0.7466,  0.8637,  1.1233]])


    >>> As = torch.randn(3,2,5)
    >>> Bs = torch.randn(3,5,4)
    >>> torch.einsum('bij,bjk->bik', (As, Bs)) # batch matrix multiplication
    tensor([[[-6.6958, -1.5956,  0.1554, -0.2422],
             [ 1.0180, -0.3066,  1.9425, -0.6323]],

            [[-1.0202, -0.4110, -0.2561, -1.0726],
             [-0.5159,  0.1312, -0.3356,  1.0438]],

            [[ 0.7815, -3.9945, -4.8426,  1.4048],
             [ 1.3494, -0.6137, -1.1748, -0.0393]]])



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
    out (Tensor, optional): the output tensor. Must be a `ByteTensor` or the same type as `input`.

Returns:
    Tensor: A ``torch.ByteTensor`` containing a 1 at each location where comparison is true

Example::

    >>> torch.eq(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
    tensor([[ 1,  0],
            [ 0,  1]], dtype=torch.uint8)
""")

add_docstr(torch.equal,
           r"""
equal(tensor1, tensor2) -> bool

``True`` if two tensors have the same size and elements, ``False`` otherwise.

Example::

    >>> torch.equal(torch.Tensor([1, 2]), torch.Tensor([1, 2]))
    True
""")

add_docstr(torch.erf,
           r"""
erf(tensor, out=None) -> Tensor

Computes the error function of each element. The error function is defined as follows:

.. math::
    \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt

Args:
    tensor (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> torch.erf(torch.Tensor([0, -1., 10.]))
    tensor([ 0.0000, -0.8427,  1.0000])
""")

add_docstr(torch.erfinv,
           r"""
erfinv(tensor, out=None) -> Tensor

Computes the inverse error function of each element. The inverse error function is defined
in the range :math:`(-1, 1)` as:

.. math::
    \mathrm{erfinv}(\mathrm{erf}(x)) = x

Args:
    tensor (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> torch.erfinv(torch.Tensor([0, 0.5, -1.]))
    tensor([ 0.0000,  0.4769,    -inf])
""")

add_docstr(torch.exp,
           r"""
exp(tensor, out=None) -> Tensor

Returns a new tensor with the exponential of the elements
of :attr:`input`.

.. math::
    y_{i} = e^{x_{i}}

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Args:
    tensor (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> torch.exp(torch.Tensor([0, math.log(2)]))
    tensor([ 1.0000,  2.0000])
""")

add_docstr(torch.expm1,
           r"""
expm1(tensor, out=None) -> Tensor

Returns a new tensor with the exponential of the elements minus 1
of :attr:`input`.

.. math::
    y_{i} = e^{x_{i}} - 1

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Args:
    tensor (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> torch.expm1(torch.Tensor([0, math.log(2)]))
    tensor([ 0.0000,  1.0000])
""")

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
    tensor([[ 1.0000,  0.0000,  0.0000],
            [ 0.0000,  1.0000,  0.0000],
            [ 0.0000,  0.0000,  1.0000]])
""".format(**factory_common_args))

add_docstr(torch.floor,
           r"""
floor(input, out=None) -> Tensor

Returns a new tensor with the floor of the elements of :attr:`input`,
the largest integer less than or equal to each element.

.. math::
    \text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 1.4317,  0.7110,  1.1331,  0.3740])
    >>> torch.floor(a)
    tensor([ 1.0000,  0.0000,  1.0000,  0.0000])
""")

add_docstr(torch.fmod,
           r"""
fmod(input, divisor, out=None) -> Tensor

Computes the element-wise remainder of division.

The dividend and divisor may contain both for integer and floating point
numbers. The remainder has the same sign as the dividend :attr:`input`.

When :attr:`divisor` is a tensor, the shapes of :attr:`input` and
:attr:`divisor` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the dividend
    divisor (Tensor or float): the divisor, which may be either a number or a tensor of the same shape as the dividend
    out (Tensor, optional): the output tensor

Example::

    >>> torch.fmod(torch.Tensor([-3, -2, -1, 1, 2, 3]), 2)
    tensor([-1.0000, -0.0000, -1.0000,  1.0000,  0.0000,  1.0000])
    >>> torch.fmod(torch.Tensor([1, 2, 3, 4, 5]), 1.5)
    tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])


""")

add_docstr(torch.frac,
           r"""
frac(tensor, out=None) -> Tensor

Computes the fractional portion of each element in :attr:`tensor`.

.. math::
    \text{out}_{i} = \text{input}_{i} - \left\lfloor \text{input}_{i} \right\rfloor

Example::

    >>> torch.frac(torch.Tensor([1, 2.5, -3.2]))
    tensor([ 0.0000,  0.5000, -0.2000])
""")

add_docstr(torch.from_numpy,
           r"""
from_numpy(ndarray) -> Tensor

Creates a :class:`Tensor` from a :class:`numpy.ndarray`.

The returned tensor and `ndarray` share the same memory. Modifications to the
tensor will be reflected in the `ndarray` and vice versa. The returned tensor
is not resizable.

Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.from_numpy(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])
""")

add_docstr(torch.gather,
           r"""
gather(input, dim, index, out=None) -> Tensor

Gathers values along an axis specified by `dim`.

For a 3-D tensor the output is specified by::

    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

If :attr:`input` is an n-dimensional tensor with size
:math:`(x_0, x_1..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
and :attr:`dim` :math:`= i`, then :attr:`index` must be an :math:`n`-dimensional tensor with
size :math:`(x_0, x_1, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})` where :math:`y \geq 1`
and :attr:`out` will have the same size as :attr:`index`.

Args:
    input (Tensor): the source tensor
    dim (int): the axis along which to index
    index (LongTensor): the indices of elements to gather
    out (Tensor, optional): the destination tensor

Example::

    >>> t = torch.Tensor([[1,2],[3,4]])
    >>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
    tensor([[ 1.0000,  1.0000],
            [ 4.0000,  3.0000]])
""")

add_docstr(torch.ge,
           r"""
ge(input, other, out=None) -> Tensor

Computes :math:`input \geq other` element-wise.

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare
    out (Tensor, optional): the output tensor that must be a `ByteTensor` or the same type as :attr:`input`

Returns:
    Tensor: A ``torch.ByteTensor`` containing a 1 at each location where comparison is true

Example::

    >>> torch.ge(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
    tensor([[ 1,  1],
            [ 0,  1]], dtype=torch.uint8)
""")

add_docstr(torch.gels,
           r"""
gels(B, A, out=None) -> Tensor

Computes the solution to the least squares and least norm problems for a full
rank matrix :math:`A` of size :math:`(m \times n)` and a matrix :math:`B` of
size :math:`(n \times k)`.

If :math:`m \geq n`, :func:`gels` solves the least-squares problem:

.. math::

   \begin{array}{ll}
   \min_X & \|AX-B\|_2.
   \end{array}

If :math:`m < n`, :func:`gels` solves the least-norm problem:

.. math::

   \begin{array}{ll}
   \min_X & \|X\|_2 & \mbox{subject to} & AX = B.
   \end{array}

Returned tensor :math:`X` has shape :math:`(\max(m, n) \times k)`. The first :math:`n`
rows of :math:`X` contains the solution. If :math`m \geq n`, the residual sum of squares
for the solution in each column is given by the sum of squares of elements in the
remaining :math:`m - n` rows of that column.

Args:
    B (Tensor): the matrix :math:`B`
    A (Tensor): the :math:`m` by :math:`n` matrix :math:`A`
    out (tuple, optional): the optional destination tensor

Returns:
    (Tensor, Tensor): A tuple containing:

        - **X** (*Tensor*): the least squares solution
        - **qr** (*Tensor*): the details of the QR factorization

.. note::

    The returned matrices will always be transposed, irrespective of the strides
    of the input matrices. That is, they will have stride `(1, m)` instead of
    `(m, 1)`.

Example::

    >>> A = torch.Tensor([[1, 1, 1],
                          [2, 3, 4],
                          [3, 5, 2],
                          [4, 2, 5],
                          [5, 4, 3]])
    >>> B = torch.Tensor([[-10, -3],
                          [ 12, 14],
                          [ 14, 12],
                          [ 16, 16],
                          [ 18, 16]])
    >>> X, _ = torch.gels(B, A)
    >>> X
    tensor([[  2.0000,   1.0000],
            [  1.0000,   1.0000],
            [  1.0000,   2.0000],
            [ 10.9635,   4.8501],
            [  8.9332,   5.2418]])
""")

add_docstr(torch.geqrf,
           r"""
geqrf(input, out=None) -> (Tensor, Tensor)

This is a low-level function for calling LAPACK directly.

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
ger(vec1, vec2, out=None) -> Tensor

Outer product of :attr:`vec1` and :attr:`vec2`.
If :attr:`vec1` is a vector of size :math:`n` and :attr:`vec2` is a vector of
size :math:`m`, then :attr:`out` must be a matrix of size :math:`(n \times m)`.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

Args:
    vec1 (Tensor): 1-D input vector
    vec2 (Tensor): 1-D input vector
    out (Tensor, optional): optional output matrix

Example::

    >>> v1 = torch.arange(1, 5)
    >>> v2 = torch.arange(1, 4)
    >>> torch.ger(v1, v2)
    tensor([[  1.0000,   2.0000,   3.0000],
            [  2.0000,   4.0000,   6.0000],
            [  3.0000,   6.0000,   9.0000],
            [  4.0000,   8.0000,  12.0000]])
""")

add_docstr(torch.gesv,
           r"""
gesv(B, A, out=None) -> (Tensor, Tensor)

This function returns the solution to the system of linear
equations represented by :math:`AX = B` and the LU factorization of
A, in order as a tuple `X, LU`.

`LU` contains `L` and `U` factors for LU factorization of `A`.

:attr:`A` has to be a square and non-singular matrix (2-D tensor).

If `A` is an :math:`(m \times m)` matrix and `B` is :math:`(m \times k)`,
the result `LU` is :math:`(m \times m)` and `X` is :math:`(m \times k)`.

.. note::

    Irrespective of the original strides, the returned matrices
    `X` and `LU` will be transposed, i.e. with strides `(1, m)`
    instead of `(m, 1)`.

Args:
    B (Tensor): input matrix of :math:`(m \times k)` dimensions
    A (Tensor): input square matrix of :math:`(m \times m)` dimensions
    out (Tensor, optional): optional output matrix

Example::

    >>> A = torch.Tensor([[6.80, -2.11,  5.66,  5.97,  8.23],
                          [-6.05, -3.30,  5.36, -4.44,  1.08],
                          [-0.45,  2.58, -2.70,  0.27,  9.04],
                          [8.32,  2.71,  4.35,  -7.17,  2.14],
                          [-9.67, -5.14, -7.26,  6.08, -6.87]]).t()
    >>> B = torch.Tensor([[4.02,  6.19, -8.22, -7.57, -3.03],
                          [-1.56,  4.00, -8.67,  1.75,  2.86],
                          [9.81, -4.09, -4.57, -8.61,  8.99]]).t()
    >>> X, LU = torch.gesv(B, A)
    >>> torch.dist(B, torch.mm(A, X))
    tensor(1.00000e-06 *
           7.0977)
""")

add_docstr(torch.get_num_threads,
           r"""
get_num_threads() -> int

Gets the number of OpenMP threads used for parallelizing CPU operations
""")

add_docstr(torch.gt,
           r"""
gt(input, other, out=None) -> Tensor

Computes :math:`input > other` element-wise.

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare
    out (Tensor, optional): the output tensor that must be a `ByteTensor` or the same type as :attr:`input`

Returns:
    Tensor: A ``torch.ByteTensor`` containing a 1 at each location where comparison is true

Example::

    >>> torch.gt(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
    tensor([[ 0,  1],
            [ 0,  0]], dtype=torch.uint8)
""")

add_docstr(torch.histc,
           r"""
histc(input, bins=100, min=0, max=0, out=None) -> Tensor

Computes the histogram of a tensor.

The elements are sorted into equal width bins between :attr:`min` and
:attr:`max`. If :attr:`min` and :attr:`max` are both zero, the minimum and
maximum values of the data are used.

Args:
    input (Tensor): the input tensor
    bins (int): number of histogram bins
    min (int): lower end of the range (inclusive)
    max (int): upper end of the range (inclusive)
    out (Tensor, optional): the output tensor

Returns:
    Tensor: Histogram represented as a tensor

Example::

    >>> torch.histc(torch.FloatTensor([1, 2, 1]), bins=4, min=0, max=3)
    tensor([ 0.0000,  2.0000,  1.0000,  0.0000])
""")

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
    input (Tensor): the input tensor
    dim (int): the dimension in which we index
    index (LongTensor): the 1-D tensor containing the indices to index
    out (Tensor, optional): the output tensor

Example::

    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[-1.6727, -1.2638, -0.3851,  0.9598],
            [-0.6904, -0.6232,  1.4447,  0.1557],
            [-0.6359, -0.0336, -1.5539,  1.3314]])
    >>> indices = torch.tensor([0, 2])
    >>> torch.index_select(x, 0, indices)
    tensor([[-1.6727, -1.2638, -0.3851,  0.9598],
            [-0.6359, -0.0336, -1.5539,  1.3314]])
    >>> torch.index_select(x, 1, indices)
    tensor([[-1.6727, -0.3851],
            [-0.6904,  1.4447],
            [-0.6359, -1.5539]])
""")

add_docstr(torch.inverse,
           r"""
inverse(input, out=None) -> Tensor

Takes the inverse of the square matrix :attr:`input`.

.. note::

    Irrespective of the original strides, the returned matrix will be
    transposed, i.e. with strides `(1, m)` instead of `(m, 1)`

Args:
    input (Tensor): the input 2-D square tensor
    out (Tensor, optional): the optional output tensor

Example::

    >>> x = torch.rand(4, 4)
    >>> y = torch.inverse(x)
    >>> z = torch.mm(x, y)
    >>> z
    tensor([[ 1.0000, -0.0000,  0.0000, -0.0000],
            [ 0.0000,  1.0000,  0.0000,  0.0000],
            [ 0.0000, -0.0000,  1.0000, -0.0000],
            [ 0.0000, -0.0000,  0.0000,  1.0000]])
    >>> torch.max(torch.abs(z - torch.eye(4))) # Max nonzero
    tensor(1.00000e-07 *
           7.1526)
""")

add_docstr(torch.kthvalue,
           r"""
kthvalue(input, k, dim=None, keepdim=False, out=None) -> (Tensor, LongTensor)

Returns the :attr:`k` th smallest element of the given :attr:`input` tensor
along a given dimension.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

A tuple of `(values, indices)` is returned, where the `indices` is the indices
of the kth-smallest element in the original `input` tensor in dimension `dim`.

If :attr:`keepdim` is ``True``, both the :attr:`values` and :attr:`indices` tensors
are the same size as :attr:`input`, except in the dimension :attr:`dim` where
they are of size 1. Otherwise, :attr:`dim` is squeezed
(see :func:`torch.squeeze`), resulting in both the :attr:`values` and
:attr:`indices` tensors having 1 fewer dimension than the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    k (int): k for the k-th smallest element
    dim (int, optional): the dimension to find the kth value along
    keepdim (bool): whether the output tensors have :attr:`dim` retained or not
    out (tuple, optional): the output tuple of (Tensor, LongTensor)
                           can be optionally given to be used as output buffers

Example::

    >>> x = torch.arange(1, 6)
    >>> x
    tensor([ 1.0000,  2.0000,  3.0000,  4.0000,  5.0000])
    >>> torch.kthvalue(x, 4)
    (tensor(4.0000), tensor(3))

    >>> x=torch.arange(1,7).resize_(2,3)
    >>> x
    tensor([[ 1.0000,  2.0000,  3.0000],
            [ 4.0000,  5.0000,  6.0000]])
    >>> torch.kthvalue(x,2,0,True)
    (tensor([[ 4.0000,  5.0000,  6.0000]]), tensor([[ 1,  1,  1]]))
""")

add_docstr(torch.le,
           r"""
le(input, other, out=None) -> Tensor

Computes :math:`input \leq other` element-wise.

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare
    out (Tensor, optional): the output tensor that must be a `ByteTensor` or the same type as :attr:`input`

Returns:
    Tensor: A ``torch.ByteTensor`` containing a 1 at each location where comparison is true

Example::

    >>> torch.le(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
    tensor([[ 1,  0],
            [ 1,  1]], dtype=torch.uint8)
""")

add_docstr(torch.lerp,
           r"""
lerp(start, end, weight, out=None)

Does a linear interpolation of two tensors :attr:`start` and :attr:`end` based
on a scalar :attr:`weight` and returns the resulting :attr:`out` tensor.

.. math::
    out_i = start_i + weight \times (end_i - start_i)

The shapes of :attr:`start` and :attr:`end` must be
:ref:`broadcastable <broadcasting-semantics>`.

Args:
    start (Tensor): the tensor with the starting points
    end (Tensor): the tensor with the ending points
    weight (float): the weight for the interpolation formula
    out (Tensor, optional): the output tensor

Example::

    >>> start = torch.arange(1, 5)
    >>> end = torch.Tensor(4).fill_(10)
    >>> start
    tensor([ 1.0000,  2.0000,  3.0000,  4.0000])
    >>> end
    tensor([ 10.0000,  10.0000,  10.0000,  10.0000])
    >>> torch.lerp(start, end, 0.5)
    tensor([ 5.5000,  6.0000,  6.5000,  7.0000])
""")

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
    tensor([-10.0000,  -5.0000,   0.0000,   5.0000,  10.0000])
    >>> torch.linspace(start=-10, end=10, steps=5)
    tensor([-10.0000,  -5.0000,   0.0000,   5.0000,  10.0000])
""".format(**factory_common_args))

add_docstr(torch.log,
           r"""
log(input, out=None) -> Tensor

Returns a new tensor with the natural logarithm of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{e} (x_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.1601,  0.3173, -0.4866,  1.6391, -0.3976])
    >>> torch.log(a)
    tensor([-1.8319, -1.1480,     nan,  0.4941,     nan])
""")

add_docstr(torch.log10,
           r"""
log10(input, out=None) -> Tensor

Returns a new tensor with the logarithm to the base 10 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{10} (x_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.rand(5)
    >>> a
    tensor([ 0.6062,  0.9727,  0.0785,  0.2565,  0.1639])


    >>> torch.log10(a)
    tensor([-0.2174, -0.0120, -1.1053, -0.5909, -0.7854])

""")

add_docstr(torch.log1p,
           r"""
log1p(input, out=None) -> Tensor

Returns a new tensor with the natural logarithm of (1 + :attr:`input`).

.. math::
    y_i = \log_{e} (x_i + 1)

.. note:: This function is more accurate than :func:`torch.log` for small
          values of :attr:`input`

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([-0.7779,  0.6704, -0.3631, -0.4028, -1.2597])
    >>> torch.log1p(a)
    tensor([-1.5045,  0.5131, -0.4512, -0.5156,     nan])
""")

add_docstr(torch.log2,
           r"""
log2(input, out=None) -> Tensor

Returns a new tensor with the logarithm to the base 2 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{2} (x_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.rand(5)
    >>> a
    tensor([ 0.4062,  0.3548,  0.4376,  0.2219,  0.8875])


    >>> torch.log2(a)
    tensor([-1.2997, -1.4949, -1.1923, -2.1720, -0.1722])

""")

add_docstr(torch.logspace,
           r"""
logspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a one-dimensional tensor of :attr:`steps` points
logarithmically spaced between :math:`10^{{\text{{start}}}}` and :math:`10^{{\text{{end}}}}`.

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

    >>> torch.logspace(start=-10, end=10, steps=5)
    tensor([ 1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10])
    >>> torch.logspace(start=0.1, end=1.0, steps=5)
    tensor([  1.2589,   2.1135,   3.5481,   5.9566,  10.0000])
""".format(**factory_common_args))

add_docstr(torch.lt,
           r"""
lt(input, other, out=None) -> Tensor

Computes :math:`input < other` element-wise.

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare
    out (Tensor, optional): the output tensor that must be a `ByteTensor` or the same type as :attr:`input`

Returns:
    Tensor: A `torch.ByteTensor` containing a 1 at each location where comparison is true

Example::

    >>> torch.lt(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
    tensor([[ 0,  0],
            [ 1,  0]], dtype=torch.uint8)
""")

add_docstr(torch.masked_select,
           r"""
masked_select(input, mask, out=None) -> Tensor

Returns a new 1-D tensor which indexes the :attr:`input` tensor according to
the binary mask :attr:`mask` which is a `ByteTensor`.

The shapes of the :attr:`mask` tensor and the :attr:`input` tensor don't need
to match, but they must be :ref:`broadcastable <broadcasting-semantics>`.

.. note:: The returned tensor does **not** use the same storage
          as the original tensor

Args:
    input (Tensor): the input data
    mask  (ByteTensor): the tensor containing the binary mask to index with
    out (Tensor, optional): the output tensor

Example::

    >>> x = torch.randn(3, 4)
    >>> x
    tensor([[ 0.4362,  2.3587,  0.3741,  1.2278],
            [ 0.1954,  1.7649,  0.7508, -0.6300],
            [ 2.9308,  0.2213,  0.5486, -0.1254]])
    >>> mask = x.ge(0.5)
    >>> mask
    tensor([[ 0,  1,  0,  1],
            [ 0,  1,  1,  0],
            [ 1,  0,  1,  0]], dtype=torch.uint8)
    >>> torch.masked_select(x, mask)
    tensor([ 2.3587,  1.2278,  1.7649,  0.7508,  2.9308,  0.5486])
""")

add_docstr(torch.max,
           r"""
.. function:: max(input) -> Tensor

Returns the maximum value of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.1294,  0.5709, -0.9577]])
    >>> torch.max(a)
    tensor(0.5709)

.. function:: max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)

Returns the maximum value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. The second return value is the index location of each
maximum value found (argmax).

If :attr:`keepdim` is ``True``, the output tensors are of the same size
as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having 1 fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension to reduce
    keepdim (bool): whether the output tensors have :attr:`dim` retained or not
    out (tuple, optional): the result tuple of two output tensors (max, max_indices)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.3041,  0.2033,  0.5305,  0.3849],
            [ 0.1237,  0.4398,  0.5018,  0.7867],
            [ 0.1560,  0.9370,  0.1027,  0.1854],
            [ 0.8522,  0.8932,  0.8115,  0.8216]])
    >>> torch.max(a, 1)
    (tensor([ 0.5305,  0.7867,  0.9370,  0.8932]), tensor([ 2,  3,  1,  1]))

.. function:: max(input, other, out=None) -> Tensor

Each element of the tensor :attr:`input` is compared with the corresponding
element of the tensor :attr:`other` and an element-wise maximum is taken.

The shapes of :attr:`input` and :attr:`other` don't need to match,
but they must be :ref:`broadcastable <broadcasting-semantics>`.

.. math::
    out_i = \max(tensor_i, other_i)

.. note:: When the shapes do not match, the shape of the returned output tensor
          follows the :ref:`broadcasting rules <broadcasting-semantics>`.

Args:
    input (Tensor): the input tensor
    other (Tensor): the second input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.0435, -0.3766,  1.0115,  0.1909])
    >>> b = torch.randn(4)
    >>> b
    tensor([-0.2109,  1.2140, -0.2816, -1.6602])
    >>> torch.max(a, b)
    tensor([-0.0435,  1.2140,  1.0115,  0.1909])
""")

add_docstr(torch.mean,
           r"""
.. function:: mean(input) -> Tensor

Returns the mean value of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.1111, -0.8677,  0.0987]])
    >>> torch.mean(a)
    tensor(-0.2934)

.. function:: mean(input, dim, keepdim=False, out=None) -> Tensor

Returns the mean value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 fewer dimension.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension to reduce
    keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not
    out (Tensor): the output tensor

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.6167,  0.8762, -0.3248, -0.5165],
            [ 1.1352,  1.1577, -0.1370,  0.4928],
            [ 0.2611,  0.5410, -0.8800, -0.9362],
            [ 0.2100, -0.0954,  1.1671,  0.1345]])
    >>> torch.mean(a, 1)
    tensor([ 0.1629,  0.6622, -0.2535,  0.3540])
    >>> torch.mean(a, 1, True)
    tensor([[ 0.1629],
            [ 0.6622],
            [-0.2535],
            [ 0.3540]])
""")

add_docstr(torch.median,
           r"""
.. function:: median(input) -> Tensor

Returns the median value of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.9927,  0.3166, -2.3463]])
    >>> torch.median(a)
    tensor(-0.9927)

.. function:: median(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)

Returns the median value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. Also returns the index location of the median value
as a `LongTensor`.

By default, :attr:`dim` is the last dimension of the :attr:`input` tensor.

If :attr:`keepdim` is ``True``, the output tensors are of the same size
as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the outputs tensor having 1 fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension to reduce
    keepdim (bool): whether the output tensors have :attr:`dim` retained or not
    values (Tensor, optional): the output tensor
    indices (Tensor, optional): the output index tensor

Example::

    >>> a = torch.randn(4, 5)
    >>> a
    tensor([[-0.5363,  0.9879, -0.0106, -0.3536, -0.1169],
            [-1.7425, -1.1041,  1.7837, -0.2429, -3.1225],
            [ 0.0725, -0.5303,  0.5481, -0.0843,  1.4926],
            [ 0.3137,  1.2695,  0.7058, -1.2642,  0.6918]])
    >>> torch.median(a, 1)
    (tensor([-0.1169, -1.1041,  0.0725,  0.6918]), tensor([ 4,  1,  0,  4]))
""")

add_docstr(torch.min,
           r"""
.. function:: min(input) -> Tensor

Returns the minimum value of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-2.2179,  0.9030, -1.6512]])
    >>> torch.min(a)
    tensor(-2.2179)

.. function:: min(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)

Returns the minimum value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. The second return value is the index location of each
minimum value found (argmin).

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensors having 1 fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension to reduce
    keepdim (bool): whether the output tensors have :attr:`dim` retained or not
    out (tuple, optional): the tuple of two output tensors (min, min_indices)

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.6018,  0.0051,  0.7931,  0.0348],
            [ 0.7706,  0.1635,  0.2319,  0.2989],
            [ 0.2058,  0.2304,  0.0451,  0.8141],
            [ 0.1085,  0.1561,  0.1410,  0.4926]])
    >>> torch.min(a, 1)
    (tensor([ 0.0051,  0.1635,  0.0451,  0.1085]), tensor([ 1,  1,  2,  0]))

.. function:: min(input, other, out=None) -> Tensor

Each element of the tensor :attr:`input` is compared with the corresponding
element of the tensor :attr:`other` and an element-wise minimum is taken.
The resulting tensor is returned.

The shapes of :attr:`input` and :attr:`other` don't need to match,
but they must be :ref:`broadcastable <broadcasting-semantics>`.

.. math::
    out_i = \min(tensor_i, other_i)

.. note:: When the shapes do not match, the shape of the returned output tensor
          follows the :ref:`broadcasting rules <broadcasting-semantics>`.

Args:
    input (Tensor): the input tensor
    other (Tensor): the second input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.1347,  1.9329,  0.8875, -0.8992])
    >>> b = torch.randn(4)
    >>> b
    tensor([-1.0371,  1.4444,  0.9654,  0.1224])
    >>> torch.min(a, b)
    tensor([-1.0371,  1.4444,  0.8875, -0.8992])
""")

add_docstr(torch.mm,
           r"""
mm(mat1, mat2, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.

If :attr:`mat1` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
:math:`(m \times p)` tensor, :attr:`out` will be a :math:`(n \times p)` tensor.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Args:
    mat1 (Tensor): the first matrix to be multiplied
    mat2 (Tensor): the second matrix to be multiplied
    out (Tensor, optional): the output tensor

Example::

    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.mm(mat1, mat2)
    tensor([[ 4.2812, -5.6143,  1.4047],
            [ 0.3845, -0.7837, -3.7774]])
""")

add_docstr(torch.matmul,
           r"""
matmul(tensor1, tensor2, out=None) -> Tensor

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
  must be broadcastable).  For example, if :attr:`tensor1` is a
  :math:`(j \times 1 \times n \times m)` tensor and :attr:`tensor2` is a :math:`(k \times m \times p)`
  tensor, :attr:`out` will be an :math:`(j \times k \times n \times p)` tensor.

.. note::

    The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.

Arguments:
    tensor1 (Tensor): the first tensor to be multiplied
    tensor2 (Tensor): the second tensor to be multiplied
    out (Tensor, optional): the output tensor

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

""")

add_docstr(torch.mode,
           r"""
mode(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor)

Returns the mode value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. Also returns the index location of the mode value
as a `LongTensor`.

By default, :attr:`dim` is the last dimension of the :attr:`input` tensor.

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensors having 1 fewer dimension than :attr:`input`.

.. note:: This function is not defined for ``torch.cuda.Tensor`` yet.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension to reduce
    keepdim (bool): whether the output tensors have :attr:`dim` retained or not
    values (Tensor, optional): the output tensor
    indices (Tensor, optional): the output index tensor

Example::

    >>> a = torch.randn(4, 5)
    >>> a
    tensor([[-0.1817,  1.7509, -1.2523,  0.7943,  2.7027],
            [-2.5194,  0.2956, -0.1925, -1.1190,  0.3535],
            [-1.9061,  0.2553, -0.5484,  1.1925,  0.6611],
            [-1.3532,  0.4144,  0.3994, -0.4238, -1.2152]])
    >>> torch.mode(a, 1)
    (tensor([-1.2523, -2.5194, -1.9061, -1.3532]), tensor([ 2,  0,  0,  0]))
""")

add_docstr(torch.mul,
           r"""
.. function:: mul(input, value, out=None)

Multiplies each element of the input :attr:`input` with the scalar
:attr:`value` and returns a new resulting tensor.

.. math::
    out_i = value \times input_i

If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, :attr:`value`
should be a real number, otherwise it should be an integer

Args:
    input (Tensor): the input tensor
    value (Number): the number to be multiplied to each element of :attr:`input`
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(3)
    >>> a
    tensor([-0.4807,  0.2841, -1.7772])
    >>> torch.mul(a, 100)
    tensor([ -48.0659,   28.4073, -177.7179])

.. function:: mul(input, other, out=None)

Each element of the tensor :attr:`input` is multiplied by each element of the
Tensor :attr:`other`. The resulting tensor is returned.

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

.. math::
    out_i = input_i \times other_i

Args:
    input (Tensor): the first multiplicand tensor
    other (Tensor): the second multiplicand tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4, 1)
    >>> a
    tensor([[-0.3070],
            [-0.2209],
            [-0.6848],
            [ 1.7139]])
    >>> b = torch.randn(1, 4)
    >>> b
    tensor([[ 0.2300,  0.2405,  1.3416, -1.2338]])
    >>> torch.mul(a, b)
    tensor([[-0.0706, -0.0738, -0.4119,  0.3788],
            [-0.0508, -0.0531, -0.2964,  0.2726],
            [-0.1575, -0.1647, -0.9187,  0.8449],
            [ 0.3942,  0.4122,  2.2994, -2.1147]])
""")

add_docstr(torch.multinomial,
           r"""
multinomial(input, num_samples, replacement=False, out=None) -> LongTensor

Returns a tensor where each row contains :attr:`num_samples` indices sampled
from the multinomial probability distribution located in the corresponding row
of tensor :attr:`input`.

.. note::
    The rows of :attr:`input` do not need to sum to one (in which case we use
    the values as weights), but must be non-negative and have a non-zero sum.

Indices are ordered from left to right according to when each was sampled
(first samples are placed in first column).

If :attr:`input` is a vector, :attr:`out` is a vector of size :attr:`num_samples`.

If :attr:`input` is a matrix with `m` rows, :attr:`out` is an matrix of shape
:math:`(m \times num\_samples)`.

If replacement is ``True``, samples are drawn with replacement.

If not, they are drawn without replacement, which means that when a
sample index is drawn for a row, it cannot be drawn again for that row.

This implies the constraint that :attr:`num_samples` must be lower than
:attr:`input` length (or number of columns of :attr:`input` if it is a matrix).

Args:
    input (Tensor): the input tensor containing probabilities
    num_samples (int): number of samples to draw
    replacement (bool, optional): whether to draw with replacement or not
    out (Tensor, optional): the output tensor

Example::

    >>> weights = torch.Tensor([0, 10, 3, 0]) # create a tensor of weights
    >>> torch.multinomial(weights, 4)
    tensor([ 2,  1,  0,  0])
    >>> torch.multinomial(weights, 4, replacement=True)
    tensor([ 1,  1,  1,  2])
""")

add_docstr(torch.mv,
           r"""
mv(mat, vec, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`mat` and the vector
:attr:`vec`.

If :attr:`mat` is a :math:`(n \times m)` tensor, :attr:`vec` is a 1-D tensor of
size :math:`m`, :attr:`out` will be 1-D of size :math:`n`.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

Args:
    mat (Tensor): matrix to be multiplied
    vec (Tensor): vector to be multiplied
    out (Tensor, optional): the output tensor

Example::

    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.mv(mat, vec)
    tensor([ 0.0270, -0.3607])
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
    out (Tensor, optional): the output tensor that must be a `ByteTensor` or the same type as `input`

Returns:
    Tensor: A ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example::

    >>> torch.ne(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
    tensor([[ 0,  1],
            [ 1,  0]], dtype=torch.uint8)
""")

add_docstr(torch.neg,
           r"""
neg(input, out=None) -> Tensor

Returns a new tensor with the negative of the elements of :attr:`input`.

.. math::
    out = -1 \times input

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.2783,  0.3317,  0.1550,  0.7447,  0.6273])
    >>> torch.neg(a)
    tensor([-0.2783, -0.3317, -0.1550, -0.7447, -0.6273])
""")

add_docstr(torch.nonzero,
           r"""
nonzero(input, out=None) -> LongTensor

Returns a tensor containing the indices of all non-zero elements of
:attr:`input`.  Each row in the result contains the indices of a non-zero
element in :attr:`input`.

If :attr:`input` has `n` dimensions, then the resulting indices tensor
:attr:`out` is of size :math:`(z \times n)`, where :math:`z` is the total number of
non-zero elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor
    out (LongTensor, optional): the output tensor containing indices

Example::

    >>> torch.nonzero(torch.Tensor([1, 1, 1, 0, 1]))
    tensor([[ 0],
            [ 1],
            [ 2],
            [ 4]])
    >>> torch.nonzero(torch.Tensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]]))
    tensor([[ 0,  0],
            [ 1,  1],
            [ 2,  2],
            [ 3,  3]])
""")

add_docstr(torch.norm,
           r"""
.. function:: norm(input, p=2) -> Tensor

Returns the p-norm of the :attr:`input` tensor.

.. math::
    ||x||_{p} = \sqrt[p]{x_{1}^{p} + x_{2}^{p} + \ldots + x_{N}^{p}}

Args:
    input (Tensor): the input tensor
    p (float, optional): the exponent value in the norm formulation
Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.3548,  1.9243,  0.6060]])
    >>> torch.norm(a, 3)
    tensor(1.9480)

.. function:: norm(input, p, dim, keepdim=False, out=None) -> Tensor

Returns the p-norm of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`.

If :attr:`keepdim` is ``True``, the output tensor is of the same size as
:attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensor having 1 fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor
    p (float):  the exponent value in the norm formulation
    dim (int): the dimension to reduce
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4, 2)
    >>> a
    tensor([[ 2.3629,  1.5555],
            [-0.2794,  1.8902],
            [-0.8452,  0.6533],
            [ 1.1464,  0.7593]])
    >>> torch.norm(a, 2, 1)
    tensor([ 2.8289,  1.9108,  1.0683,  1.3750])
    >>> torch.norm(a, 0, 1, True)
    tensor([[ 2.0000],
            [ 2.0000],
            [ 2.0000],
            [ 2.0000]])
""")

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
    out (Tensor, optional): the output tensor

Example::

    >>> torch.normal(mean=torch.arange(1, 11), std=torch.arange(1, 0, -0.1))
    tensor([ 0.8147,  3.3710,  2.3070,  4.4390,  3.9799,  6.5671,  7.3342,
             7.8960,  9.2627,  9.8802])

.. function:: normal(mean=0.0, std, out=None) -> Tensor

Similar to the function above, but the means are shared among all drawn
elements.

Args:
    mean (float, optional): the mean for all distributions
    std (Tensor): the tensor of per-element standard deviations
    out (Tensor, optional): the output tensor

Example::

    >>> torch.normal(mean=0.5, std=torch.arange(1, 6))
    tensor([ 1.2682,  1.4109, -1.9055,  4.0908, -0.0062])

.. function:: normal(mean, std=1.0, out=None) -> Tensor

Similar to the function above, but the standard-deviations are shared among
all drawn elements.

Args:
    mean (Tensor): the tensor of per-element means
    std (float, optional): the standard deviation for all distributions
    out (Tensor, optional): the output tensor

Example::

    >>> torch.normal(mean=torch.arange(1, 6))
    tensor([-0.1213,  2.7886,  2.6975,  3.0108,  5.2194])
""")

add_docstr(torch.numel,
           r"""
numel(input) -> int

Returns the total number of elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor

Example::

    >>> a = torch.randn(1, 2, 3, 4, 5)
    >>> torch.numel(a)
    120
    >>> a = torch.zeros(4,4)
    >>> torch.numel(a)
    16

""")

add_docstr(torch.ones,
           r"""
ones(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with the scalar value `1`, with the shape defined
by the variable argument :attr:`sizes`.

Args:
    sizes (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.ones(2, 3)
    tensor([[ 1.0000,  1.0000,  1.0000],
            [ 1.0000,  1.0000,  1.0000]])

    >>> torch.ones(5)
    tensor([ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000])

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

    >>> input = torch.FloatTensor(2, 3)
    >>> torch.ones_like(input)
    tensor([[ 1.0000,  1.0000,  1.0000],
            [ 1.0000,  1.0000,  1.0000]])
""".format(**factory_like_common_args))

add_docstr(torch.orgqr,
           r"""
orgqr(a, tau) -> Tensor

Computes the orthogonal matrix `Q` of a QR factorization, from the `(a, tau)`
tuple returned by :func:`torch.geqrf`.

This directly calls the underlying LAPACK function `?orgqr`.
See `LAPACK documentation for orgqr`_ for further details.

Args:
    a (Tensor): the `a` from :func:`torch.geqrf`.
    tau (Tensor): the `tau` from :func:`torch.geqrf`.

.. _LAPACK documentation for orgqr:
    https://software.intel.com/en-us/mkl-developer-reference-c-orgqr

""")

add_docstr(torch.ormqr,
           r"""
ormqr(a, tau, mat, left=True, transpose=False) -> (Tensor, Tensor)

Multiplies `mat` by the orthogonal `Q` matrix of the QR factorization
formed by :func:`torch.geqrf` that is represented by `(a, tau)`.

This directly calls the underlying LAPACK function `?ormqr`.
See `LAPACK documentation for ormqr`_ for further details.

Args:
    a (Tensor): the `a` from :func:`torch.geqrf`.
    tau (Tensor): the `tau` from :func:`torch.geqrf`.
    mat (Tensor): the matrix to be multiplied.

.. _LAPACK documentation for ormqr:
    https://software.intel.com/en-us/mkl-developer-reference-c-ormqr

""")

add_docstr(torch.potrf, r"""
potrf(a, upper=True, out=None) -> Tensor

Computes the Cholesky decomposition of a symmetric positive-definite
matrix :math:`A`.

If :attr:`upper` is ``True``, the returned matrix `U` is upper-triangular, and
the decomposition has the form:

.. math::

  A = U^TU

If :attr:`upper` is ``False``, the returned matrix `L` is lower-triangular, and
the decomposition has the form:

.. math::

    A = LL^T

Args:
    a (Tensor): the input 2-D tensor, a symmetric positive-definite matrix
    upper (bool, optional): flag that indicates whether to return the
                            upper or lower triangular matrix
    out (Tensor, optional): the output matrix

Example::

    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) # make symmetric positive definite
    >>> u = torch.potrf(a)
    >>> a
    tensor([[  5.4379,   6.0726,  -0.4573],
            [  6.0726,  12.3294,  -0.6223],
            [ -0.4573,  -0.6223,   0.0691]])
    >>> u
    tensor([[ 2.3319,  2.6041, -0.1961],
            [ 0.0000,  2.3554, -0.0474],
            [ 0.0000,  0.0000,  0.1686]])
    >>> torch.mm(u.t(), u)
    tensor([[  5.4379,   6.0726,  -0.4573],
            [  6.0726,  12.3294,  -0.6223],
            [ -0.4573,  -0.6223,   0.0691]])
""")

add_docstr(torch.potri, r"""
potri(u, upper=True, out=None) -> Tensor

Computes the inverse of a positive semidefinite matrix given its
Cholesky factor :attr:`u`: returns matrix `inv`

If :attr:`upper` is ``True`` or not provided, :attr:`u` is upper
triangular such that:

.. math::
    inv = (u^T u)^{-1}

If :attr:`upper` is ``False``, :attr:`u` is lower triangular
such that:

.. math::
    inv = (uu^{T})^{-1}

Args:
    u (Tensor): the input 2-D tensor, a upper or lower triangular
           Cholesky factor
    upper (bool, optional): whether to return a upper (default) or lower triangular matrix
    out (Tensor, optional): the output tensor for `inv`

Example::

    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) # make symmetric positive definite
    >>> u = torch.potrf(a)
    >>> a
    tensor([[ 14.2154,  -2.6578,   0.8001],
            [ -2.6578,   2.3240,  -0.2354],
            [  0.8001,  -0.2354,   1.7715]])
    >>> torch.potri(u)
    tensor([[ 0.0908,  0.1010, -0.0276],
            [ 0.1010,  0.5486,  0.0273],
            [-0.0276,  0.0273,  0.5806]])
    >>> a.inverse()
    tensor([[ 0.0908,  0.1010, -0.0276],
            [ 0.1010,  0.5486,  0.0273],
            [-0.0276,  0.0273,  0.5806]])
""")

add_docstr(torch.potrs, r"""
potrs(b, u, upper=True, out=None) -> Tensor

Solves a linear system of equations with a positive semidefinite
matrix to be inverted given its Cholesky factor matrix :attr:`u`.

If :attr:`upper` is ``True`` or not provided, :attr:`u` is upper triangular
and `c` is returned such that:

.. math::
    c = (u^T u)^{-1} b

If :attr:`upper` is ``False``, :attr:`u` is and lower triangular and `c` is
returned such that:

.. math::
    c = (u u^T)^{-1} b

.. note:: :attr:`b` is always a 2-D tensor, use `b.unsqueeze(1)` to convert a vector.

Args:
    b (Tensor): the right hand side 2-D tensor
    u (Tensor): the input 2-D tensor, a upper or lower triangular Cholesky factor
    upper (bool, optional): whether to return a upper (default) or lower triangular matrix
    out (Tensor, optional): the output tensor for `c`

Example::

    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) # make symmetric positive definite
    >>> u = torch.potrf(a)
    >>> a
    tensor([[ 9.8496, -3.7952, -1.4156],
            [-3.7952,  2.9128,  2.1382],
            [-1.4156,  2.1382,  2.3875]])
    >>> b = torch.randn(3, 2)
    >>> b
    tensor([[-0.3974, -1.6325],
            [-0.8379,  0.6529],
            [-0.6202,  1.0989]])
    >>> torch.potrs(b,u)
    tensor([[-0.5677, -0.6978],
            [-1.7212, -2.0995],
            [ 0.9451,  1.9268]])
    >>> torch.mm(a.inverse(),b)
    tensor([[-0.5677, -0.6978],
            [-1.7212, -2.0995],
            [ 0.9451,  1.9268]])
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
    out_i = x_i ^ {exponent}

When :attr:`exponent` is a tensor, the operation applied is:

.. math::
    out_i = x_i ^ {exponent_i}

When :attr:`exponent` is a tensor, the shapes of :attr:`input`
and :attr:`exponent` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the input tensor
    exponent (float or tensor): the exponent value
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.3895, -0.6493,  1.4697, -0.4497])
    >>> torch.pow(a, 2)
    tensor([ 0.1517,  0.4216,  2.1601,  0.2023])
    >>> exp = torch.arange(1, 5)

    >>> a = torch.arange(1, 5)
    >>> a
    tensor([ 1.0000,  2.0000,  3.0000,  4.0000])
    >>> exp
    tensor([ 1.0000,  2.0000,  3.0000,  4.0000])
    >>> torch.pow(a, exp)
    tensor([   1.0000,    4.0000,   27.0000,  256.0000])

.. function:: pow(base, input, out=None) -> Tensor

:attr:`base` is a scalar ``float`` value, and :attr:`input` is a tensor.
The returned tensor :attr:`out` is of the same shape as :attr:`input`

The operation applied is:

.. math::
    out_i = base ^ {input_i}

Args:
    base (float): the scalar base value for the power operation
    input (Tensor): the exponent tensor
    out (Tensor, optional): the output tensor

Example::

    >>> exp = torch.arange(1, 5)
    >>> base = 2
    >>> torch.pow(base, exp)
    tensor([  2.0000,   4.0000,   8.0000,  16.0000])
""")

add_docstr(torch.prod,
           r"""
.. function:: prod(input) -> Tensor

Returns the product of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.2674,  0.8551, -0.9454]])
    >>> torch.prod(a)
    tensor(-0.2162)

.. function:: prod(input, dim, keepdim=False, out=None) -> Tensor

Returns the product of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`.

If :attr:`keepdim` is ``True``, the output tensor is of the same size as
:attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensor having 1 fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension to reduce
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4, 2)
    >>> a
    tensor([[-1.5657,  0.1930],
            [-0.2821,  2.2779],
            [-1.6599, -0.7662],
            [-1.2148, -0.0238]])
    >>> torch.prod(a, 1)
    tensor([-0.3021, -0.6427,  1.2719,  0.0289])
""")

add_docstr(torch.pstrf, r"""
pstrf(a, upper=True, out=None) -> (Tensor, Tensor)

Computes the pivoted Cholesky decomposition of a positive semidefinite
matrix :attr:`a`. returns matrices `u` and `piv`.

If :attr:`upper` is ``True`` or not provided, `u` is upper triangular
such that :math:`a = p^T u^T u p`, with `p` the permutation given by `piv`.

If :attr:`upper` is ``False``, `u` is lower triangular such that
:math:`a = p^T u u^T p`.

Args:
    a (Tensor): the input 2-D tensor
    upper (bool, optional): whether to return a upper (default) or lower triangular matrix
    out (tuple, optional): tuple of `u` and `piv` tensors

Example::

    >>> a = torch.randn(3, 3)
    >>> a = torch.mm(a, a.t()) # make symmetric positive definite
    >>> a
    tensor([[ 4.3810,  0.8462, -0.2961],
            [ 0.8462,  3.3601,  0.0339],
            [-0.2961,  0.0339,  2.5215]])
    >>> u,piv = torch.pstrf(a)
    >>> u
    tensor([[ 2.0931,  0.4043, -0.1414],
            [ 0.0000,  1.7879,  0.0510],
            [ 0.0000,  0.0000,  1.5808]])
    >>> piv
    tensor([ 0,  1,  2], dtype=torch.int32)
    >>> p = torch.eye(3).index_select(0,piv.long()).index_select(0,piv.long()).t() # make pivot permutation
    >>> torch.mm(torch.mm(p.t(),torch.mm(u.t(),u)),p) # reconstruct
    tensor([[ 4.3810,  0.8462, -0.2961],
            [ 0.8462,  3.3601,  0.0339],
            [-0.2961,  0.0339,  2.5215]])
""")

add_docstr(torch.qr,
           r"""
qr(input, out=None) -> (Tensor, Tensor)

Computes the QR decomposition of a matrix :attr:`input`, and returns matrices
`Q` and `R` such that :math:`\text{input} = Q R`, with :math:`Q` being an
orthogonal matrix and :math:`R` being an upper triangular matrix.

This returns the thin (reduced) QR factorization.

.. note:: precision may be lost if the magnitudes of the elements of :attr:`input`
          are large

.. note:: While it should always give you a valid decomposition, it may not
          give you the same one across platforms - it will depend on your
          LAPACK implementation.

.. note:: Irrespective of the original strides, the returned matrix :math:`Q` will be
          transposed, i.e. with strides `(1, m)` instead of `(m, 1)`.

Args:
    input (Tensor): the input 2-D tensor
    out (tuple, optional): tuple of `Q` and `R` tensors

Example::

    >>> a = torch.Tensor([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
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
    tensor([[  12.0000,  -51.0000,    4.0000],
            [   6.0000,  167.0000,  -68.0000],
            [  -4.0000,   24.0000,  -41.0000]])
    >>> torch.mm(q.t(), q).round()
    tensor([[ 1.0000,  0.0000,  0.0000],
            [ 0.0000,  1.0000, -0.0000],
            [ 0.0000, -0.0000,  1.0000]])
""")

add_docstr(torch.rand,
           r"""
rand(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random numbers from a uniform distribution
on the interval :math:`[0, 1)`

The shape of the tensor is defined by the variable argument :attr:`sizes`.

Args:
    sizes (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.rand(4)
    tensor([ 0.2033,  0.0322,  0.8374,  0.7259])
    >>> torch.rand(2, 3)
    tensor([[ 0.8948,  0.8730,  0.0477],
            [ 0.2862,  0.1847,  0.2875]])
""")

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
    With the global dtype default (`torch.float32`), this function returns
    a tensor with dtype `torch.float32`, NOT an integer dtype.

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
    tensor([ 3.0000,  3.0000,  4.0000])


    >>> torch.randint(3, 10, (2,2), dtype=torch.long)
    tensor([[ 6,  6],
            [ 9,  7]])


    >>> torch.randint(3, 10, (2,2))
    tensor([[ 4.0000,  8.0000],
            [ 3.0000,  8.0000]])


""".format(**factory_common_args))

add_docstr(torch.randint_like,
           r"""
randint_like(input, low=0, high, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor with the same shape as Tensor :attr:`input` filled with
random integers generated uniformly between :attr:`low` (inclusive) and
:attr:`high` (exclusive).

.. note:
    With the global dtype default (`torch.float32`), this function returns
    a tensor with dtype `torch.float32`, NOT an integer dtype.

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
randn(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with random numbers from a normal distribution
with mean `0` and variance `1` (also called the standard normal
distribution).

.. math::
    \text{{out}}_{{i}} \sim \mathcal{{N}}(0, 1)

The shape of the tensor is defined by the variable argument :attr:`sizes`.

Args:
    sizes (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.randn(4)
    tensor([ 1.9327,  1.1202, -0.8140,  0.0011])
    >>> torch.randn(2, 3)
    tensor([[-0.3226, -0.0438, -0.5925],
            [ 0.1605,  1.8522,  0.8656]])
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
    tensor([ 2,  0,  1,  3])
""".format(**factory_common_args))

add_docstr(torch.tensor,
           r"""
tensor(data, dtype=None, device=None, requires_grad=False) -> Tensor

Constructs a tensor with :attr:`data`.

.. warning::

    :func:`torch.tensor` always copies :attr:`data`. If you have a Tensor
    ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
    or :func:`torch.Tensor.detach`.
    If you have a numpy array and want to avoid a copy, use
    :func:`torch.from_numpy`.

Args:
    data (array_like): Initial data for the tensor. Can be a list, tuple,
        numpy array, scalar, and other types.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if None, infers data type from :attr:`data`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if None, uses the current device for the default tensor type
        (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.


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
""")

add_docstr(torch.range,
           r"""
range(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 1-D tensor of size :math:`\left\lfloor \frac{{end - start}}{{step}} \right\rfloor + 1`
with values from :attr:`start` to :attr:`end` with step :attr:`step`. Step is
the gap between two values in the tensor.

.. math::
    \text{{out}}_{{i+1}} = \text{{out}}_i + step.

.. warning::
    This function is deprecated in favor of :func:`torch.arange`.

Args:
    start (float): the starting value for the set of points. Default: ``0``.
    end (float): the ending value for the set of points
    step (float): the gap between each pair of adjacent points. Default: ``1``.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.range(1, 4)
    tensor([ 1.0000,  2.0000,  3.0000,  4.0000])
    >>> torch.range(1, 4, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000,  2.5000,  3.0000,  3.5000,  4.0000])
""".format(**factory_common_args))

add_docstr(torch.arange,
           r"""
arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 1-D tensor of size :math:`\left\lfloor \frac{{end - start}}{{step}} \right\rfloor`
with values from the interval ``[start, end)`` taken with common difference
:attr:`step` beginning from `start`.

Note that non-integer :attr:`step` is subject to floating point rounding errors when
comparing against :attr:`end`; to avoid inconsistency, we advise adding a small epsilon to :attr:`end`
in such cases.

.. math::
    \text{{out}}_{{i+1}} = \text{{out}}_{{i}} + \text{{step}}

Args:
    start (float): the starting value for the set of points. Default: ``0``.
    end (float): the ending value for the set of points
    step (float): the gap between each pair of adjacent points. Default: ``1``.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.arange(5)
    tensor([ 0.0000,  1.0000,  2.0000,  3.0000,  4.0000])
    >>> torch.arange(1, 4)
    tensor([ 1.0000,  2.0000,  3.0000])
    >>> torch.arange(1, 2.5, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000])
""".format(**factory_common_args))

add_docstr(torch.remainder,
           r"""
remainder(input, divisor, out=None) -> Tensor

Computes the element-wise remainder of division.

The divisor and dividend may contain both for integer and floating point
numbers. The remainder has the same sign as the divisor.

When :attr:`divisor` is a tensor, the shapes of :attr:`input` and
:attr:`divisor` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (Tensor): the dividend
    divisor (Tensor or float): the divisor that may be either a number or a
                               Tensor of the same shape as the dividend
    out (Tensor, optional): the output tensor

Example::

    >>> torch.remainder(torch.Tensor([-3, -2, -1, 1, 2, 3]), 2)
    tensor([ 1.0000,  0.0000,  1.0000,  1.0000,  0.0000,  1.0000])
    >>> torch.remainder(torch.Tensor([1, 2, 3, 4, 5]), 1.5)
    tensor([ 1.0000,  0.5000,  0.0000,  1.0000,  0.5000])

.. seealso::

        :func:`torch.fmod`, which computes the element-wise remainder of
        division equivalently to the C library function ``fmod()``.
""")

add_docstr(torch.renorm,
           r"""
renorm(input, p, dim, maxnorm, out=None) -> Tensor

Returns a tensor where each sub-tensor of :attr:`input` along dimension
:attr:`dim` is normalized such that the `p`-norm of the sub-tensor is lower
than the value :attr:`maxnorm`

.. note:: If the norm of a row is lower than `maxnorm`, the row is unchanged

Args:
    input (Tensor): the input tensor
    p (float): the power for the norm computation
    dim (int): the dimension to slice over to get the sub-tensors
    maxnorm (float): the maximum norm to keep each sub-tensor under
    out (Tensor, optional): the output tensor

Example::

    >>> x = torch.ones(3, 3)
    >>> x[1].fill_(2)
    tensor([ 2.0000,  2.0000,  2.0000])
    >>> x[2].fill_(3)
    tensor([ 3.0000,  3.0000,  3.0000])
    >>> x
    tensor([[ 1.0000,  1.0000,  1.0000],
            [ 2.0000,  2.0000,  2.0000],
            [ 3.0000,  3.0000,  3.0000]])
    >>> torch.renorm(x, 1, 0, 5)
    tensor([[ 1.0000,  1.0000,  1.0000],
            [ 1.6667,  1.6667,  1.6667],
            [ 1.6667,  1.6667,  1.6667]])
""")

add_docstr(torch.reshape,
           r"""
reshape(input, shape) -> Tensor

Returns a tensor with the same data and number of elements as :attr:`input`,
but with the specified shape. When possible, the returned tensor will be a view
of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and inputs
with compatible strides can be reshaped without copying, but you should not
depend on the copying vs. viewing behavior.

A single dimension may be -1, in which case it's inferred from the remaining
dimensions and the number of elements in :attr:`input`.

Args:
    input (Tensor): the tensor to be reshaped
    shape (tuple of ints): the new shape

Example::

    >>> a = torch.arange(4)
    >>> torch.reshape(a, (2, 2))
    tensor([[ 0.0000,  1.0000],
            [ 2.0000,  3.0000]])
    >>> b = torch.tensor([[0, 1], [2, 3]])
    >>> torch.reshape(b, (-1,))
    tensor([ 0,  1,  2,  3])
""")


add_docstr(torch.round,
           r"""
round(input, out=None) -> Tensor

Returns a new tensor with each of the elements of :attr:`input` rounded
to the closest integer.

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.2513, -0.0762, -3.0545,  0.1637])
    >>> torch.round(a)
    tensor([-0.0000, -0.0000, -3.0000,  0.0000])
""")

add_docstr(torch.rsqrt,
           r"""
rsqrt(input, out=None) -> Tensor

Returns a new tensor with the reciprocal of the square-root of each of
the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.0093,  3.2577,  0.0648, -0.4293])
    >>> torch.rsqrt(a)
    tensor([ 10.3777,   0.5540,   3.9298,      nan])
""")

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
    >>> torch.DoubleTensor([1e-323])
    tensor([ 0.0000], dtype=torch.float64)
    >>> torch.set_flush_denormal(False)
    True
    >>> torch.DoubleTensor([1e-323])
    tensor(9.88131e-324 *
           [ 1.0000], dtype=torch.float64)
""")

add_docstr(torch.set_num_threads,
           r"""
set_num_threads(int)

Sets the number of OpenMP threads used for parallelizing CPU operations
""")

add_docstr(torch.sigmoid,
           r"""
sigmoid(input, out=None) -> Tensor

Returns a new tensor with the sigmoid of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{1 + e^{-\text{input}_{i}}}

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.4196,  0.6749, -0.2549, -0.7658])
    >>> torch.sigmoid(a)
    tensor([ 0.6034,  0.6626,  0.4366,  0.3174])
""")

add_docstr(torch.sign,
           r"""
sign(input, out=None) -> Tensor

Returns a new tensor with the sign of the elements of :attr:`input`.

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.5370, -0.2723, -0.6947,  1.0846])
    >>> torch.sign(a)
    tensor([ 1.0000, -1.0000, -1.0000,  1.0000])
""")

add_docstr(torch.sin,
           r"""
sin(input, out=None) -> Tensor

Returns a new tensor with the sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin(\text{input}_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-1.1110,  0.7380,  0.5011, -0.7487])
    >>> torch.sin(a)
    tensor([-0.8962,  0.6728,  0.4804, -0.6807])
""")

add_docstr(torch.sinh,
           r"""
sinh(input, out=None) -> Tensor

Returns a new tensor with the hyperbolic sine of the elements of
:attr:`input`.

.. math::
    \text{out}_{i} = \sinh(\text{input}_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.2136, -1.4427,  1.1918, -0.6938])
    >>> torch.sinh(a)
    tensor([ 0.2153, -1.9980,  1.4946, -0.7508])
""")

add_docstr(torch.sort,
           r"""
sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor)

Sorts the elements of the :attr:`input` tensor along a given dimension
in ascending order by value.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`descending` is ``True`` then the elements are sorted in descending
order by value.

A tuple of (sorted_tensor, sorted_indices) is returned, where the
sorted_indices are the indices of the elements in the original `input` tensor.

Args:
    input (Tensor): the input tensor
    dim (int, optional): the dimension to sort along
    descending (bool, optional): controls the sorting order (ascending or descending)
    out (tuple, optional): the output tuple of (`Tensor`, `LongTensor`) that can
        be optionally given to be used as output buffers

Example::

    >>> x = torch.randn(3, 4)
    >>> sorted, indices = torch.sort(x)
    >>> sorted
    tensor([[-0.4771,  0.1587,  0.7142,  1.5067],
            [-1.0707, -0.9171, -0.8405,  0.1450],
            [-0.3216, -0.1589,  1.1513,  1.8629]])
    >>> indices
    tensor([[ 2,  3,  0,  1],
            [ 3,  1,  2,  0],
            [ 3,  0,  2,  1]])

    >>> sorted, indices = torch.sort(x, 0)
    >>> sorted
    tensor([[-0.1589, -0.9171, -0.8405, -1.0707],
            [ 0.1450,  1.5067, -0.4771, -0.3216],
            [ 0.7142,  1.8629,  1.1513,  0.1587]])
    >>> indices
    tensor([[ 2,  1,  1,  1],
            [ 1,  0,  0,  2],
            [ 0,  2,  2,  0]])
""")

add_docstr(torch.sqrt,
           r"""
sqrt(input, out=None) -> Tensor

Returns a new tensor with the square-root of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sqrt{\text{input}_{i}}

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.7750,  1.2628, -1.2400,  0.5827])
    >>> torch.sqrt(a)
    tensor([    nan,  1.1237,     nan,  0.7634])
""")

add_docstr(torch.squeeze,
           r"""
squeeze(input, dim=None, out=None) -> Tensor

Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.

For example, if `input` is of shape:
:math:`(A \times 1 \times B \times C \times 1 \times D)` then the `out` tensor
will be of shape: :math:`(A \times B \times C \times D)`.

When :attr:`dim` is given, a squeeze operation is done only in the given
dimension. If `input` is of shape: :math:`(A \times 1 \times B)`,
`squeeze(input, 0)` leaves the tensor unchanged, but :func:`squeeze(input, 1)` will
squeeze the tensor to the shape :math:`(A \times B)`.

.. note:: As an exception to the above, a 1-dimensional tensor of size 1 will
          not have its dimensions changed.

.. note:: The returned tensor shares the storage with the input tensor,
          so changing the contents of one will change the contents of the other.

Args:
    input (Tensor): the input tensor
    dim (int, optional): if given, the input will be squeezed only in
           this dimension
    out (Tensor, optional): the output tensor

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
""")

add_docstr(torch.std,
           r"""
.. function:: std(input, unbiased=True) -> Tensor

Returns the standard-deviation of all elements in the :attr:`input` tensor.

If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
via the biased estimator. Otherwise, Bessel's correction will be used.

Args:
    input (Tensor): the input tensor
    unbiased (bool): whether to use the unbiased estimation or not

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[-0.4194,  0.0566,  0.9036]])
    >>> torch.std(a)
    tensor(0.6701)

.. function:: std(input, dim, keepdim=False, unbiased=True, out=None) -> Tensor

Returns the standard-deviation of each row of the :attr:`input` tensor in the
given dimension :attr:`dim`.

If :attr:`keepdim` is ``True``, the output tensor is of the same size as
:attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting
in the output tensor having 1 fewer dimension than :attr:`input`.

If :attr:`unbiased` is ``False``, then the standard-deviation will be calculated
via the biased estimator. Otherwise, Bessel's correction will be used.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension to reduce
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not
    unbiased (bool): whether to use the unbiased estimation or not
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.6817,  0.5348, -0.2383,  1.5072],
            [ 0.6634,  0.0507,  0.4516,  0.8058],
            [-1.8223, -0.1172,  1.5032,  0.6274],
            [ 0.9677,  0.8441,  0.1069,  1.1794]])
    >>> torch.std(a, dim=1)
    tensor([ 0.7153,  0.3287,  1.4117,  0.4661])
""")

add_docstr(torch.sum,
           r"""
.. function:: sum(input) -> Tensor

Returns the sum of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 1.0823,  1.6739,  0.0819]])
    >>> torch.sum(a)
    tensor(2.8381)

.. function:: sum(input, dim, keepdim=False, out=None) -> Tensor

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the output tensor having 1 fewer dimension than :attr:`input`.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension to reduce
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-0.7612,  1.0837,  0.1990, -1.2412],
            [-1.2026,  0.6901, -0.4490,  0.5841],
            [ 0.4772,  1.6272,  0.5197,  0.8401],
            [ 1.4151, -0.8769, -0.8977,  0.7094]])
    >>> torch.sum(a, 1)
    tensor([-0.7196, -0.3774,  3.4642,  0.3499])
""")

add_docstr(torch.svd,
           r"""
svd(input, some=True, out=None) -> (Tensor, Tensor, Tensor)

`U, S, V = torch.svd(A)` returns the singular value decomposition of a
real matrix `A` of size `(n x m)` such that :math:`A = USV^T`.

`U` is of shape :math:`(n \times n)`.

`S` is a diagonal matrix of shape :math:`(n \times m)`, represented as a vector
of size :math:`\min(n, m)` containing the non-negative diagonal entries.

`V` is of shape :math:`(m \times m)`.

If :attr:`some` is ``True`` (default), the returned `U` and `V` matrices will
contain only :math:`min(n, m)` orthonormal columns.

.. note:: Irrespective of the original strides, the returned matrix `U`
          will be transposed, i.e. with strides `(1, n)` instead of `(n, 1)`.

.. note:: Extra care needs to be taken when backward through `U` and `V`
          outputs. Such operation is really only stable when :attr:`input` is
          full rank with all distinct singular values. Otherwise, ``NaN`` can
          appear as the gradients are not properly defined. Also, notice that
          double backward will usually do an additional backward through `U` and
          `V` even if the original backward is only on `S`.

.. note:: When :attr:`some` = ``False``, the gradients on ``U[:, min(n, m):]``
          and ``V[:, min(n, m):]`` will be ignored in backward as those vectors
          can be arbitrary bases of the subspaces.

Args:
    input (Tensor): the input 2-D tensor
    some (bool, optional): controls the shape of returned `U` and `V`
    out (tuple, optional): the output tuple of tensors

Example::

    >>> a = torch.Tensor([[8.79,  6.11, -9.15,  9.57, -3.49,  9.84],
                          [9.93,  6.91, -7.93,  1.64,  4.02,  0.15],
                          [9.83,  5.04,  4.86,  8.83,  9.80, -8.99],
                          [5.45, -0.27,  4.85,  0.74, 10.00, -6.02],
                          [3.16,  7.98,  3.01,  5.80,  4.27, -5.31]]).t()

    >>> u, s, v = torch.svd(a)
    >>> u
    tensor([[-0.5911,  0.2632,  0.3554,  0.3143,  0.2299],
            [-0.3976,  0.2438, -0.2224, -0.7535, -0.3636],
            [-0.0335, -0.6003, -0.4508,  0.2334, -0.3055],
            [-0.4297,  0.2362, -0.6859,  0.3319,  0.1649],
            [-0.4697, -0.3509,  0.3874,  0.1587, -0.5183],
            [ 0.2934,  0.5763, -0.0209,  0.3791, -0.6526]])
    >>> s
    tensor([ 27.4687,  22.6432,   8.5584,   5.9857,   2.0149])
    >>> v
    tensor([[-0.2514,  0.8148, -0.2606,  0.3967, -0.2180],
            [-0.3968,  0.3587,  0.7008, -0.4507,  0.1402],
            [-0.6922, -0.2489, -0.2208,  0.2513,  0.5891],
            [-0.3662, -0.3686,  0.3859,  0.4342, -0.6265],
            [-0.4076, -0.0980, -0.4933, -0.6227, -0.4396]])
    >>> torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t()))
    tensor(1.00000e-06 *
           9.3738)
""")

add_docstr(torch.symeig,
           r"""
symeig(input, eigenvectors=False, upper=True, out=None) -> (Tensor, Tensor)

This function returns eigenvalues and eigenvectors
of a real symmetric matrix :attr:`input`, represented by a tuple :math:`(e, V)`.

:attr:`input` and :math:`V` are :math:`(m \times m)` matrices and :math:`e` is a
:math:`m` dimensional vector.

This function calculates all eigenvalues (and vectors) of :attr:`input`
such that :math:`input = V diag(e) V^T`.

The boolean argument :attr:`eigenvectors` defines computation of
eigenvectors or eigenvalues only.

If it is ``False``, only eigenvalues are computed. If it is ``True``,
both eigenvalues and eigenvectors are computed.

Since the input matrix :attr:`input` is supposed to be symmetric,
only the upper triangular portion is used by default.

If :attr:`upper` is ``False``, then lower triangular portion is used.

Note: Irrespective of the original strides, the returned matrix `V` will
be transposed, i.e. with strides `(1, m)` instead of `(m, 1)`.

Args:
    input (Tensor): the input symmetric matrix
    eigenvectors(boolean, optional): controls whether eigenvectors have to be computed
    upper(boolean, optional): controls whether to consider upper-triangular or lower-triangular region
    out (tuple, optional): the output tuple of (Tensor, Tensor)

Examples::


    >>> a = torch.Tensor([[ 1.96,  0.00,  0.00,  0.00,  0.00],
                          [-6.49,  3.80,  0.00,  0.00,  0.00],
                          [-0.47, -6.39,  4.17,  0.00,  0.00],
                          [-7.20,  1.50, -1.51,  5.70,  0.00],
                          [-0.65, -6.34,  2.67,  1.80, -7.10]]).t()
    >>> e, v = torch.symeig(a, eigenvectors=True)
    >>> e
    tensor([-11.0656,  -6.2287,   0.8640,   8.8655,  16.0948])
    >>> v
    tensor([[-0.2981, -0.6075,  0.4026, -0.3745,  0.4896],
            [-0.5078, -0.2880, -0.4066, -0.3572, -0.6053],
            [-0.0816, -0.3843, -0.6600,  0.5008,  0.3991],
            [-0.0036, -0.4467,  0.4553,  0.6204, -0.4564],
            [-0.8041,  0.4480,  0.1725,  0.3108,  0.1622]])
""")

add_docstr(torch.t,
           r"""
t(input, out=None) -> Tensor

Expects :attr:`input` to be a matrix (2-D tensor) and transposes dimensions 0
and 1.

Can be seen as a short-hand function for :meth:`transpose(input, 0, 1)`

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[-0.0002, -0.0641, -0.0965],
            [-0.4993, -0.7675, -0.2518]])
    >>> torch.t(x)
    tensor([[-0.0002, -0.4993],
            [-0.0641, -0.7675],
            [-0.0965, -0.2518]])
""")

add_docstr(torch.take,
           r"""
take(input, indices) -> Tensor

Returns a new tensor with the elements of :attr:`input` at the given indices.
The input tensor is treated as if it were viewed as a 1-D tensor. The result
takes the same shape as the indices.

Args:
    input (Tensor): the input tensor
    indices (LongTensor): the indices into tensor

Example::

    >>> src = torch.Tensor([[4, 3, 5],
                            [6, 7, 8]])
    >>> torch.take(src, torch.tensor([0, 2, 5]))
    tensor([ 4.0000,  5.0000,  8.0000])
""")

add_docstr(torch.tan,
           r"""
tan(input, out=None) -> Tensor

Returns a new tensor with the tangent of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \tan(\text{input}_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.6240,  1.7646,  1.2947,  0.5490])
    >>> torch.tan(a)
    tensor([-0.7200, -5.0949,  3.5288,  0.6118])
""")

add_docstr(torch.tanh,
           r"""
tanh(input, out=None) -> Tensor

Returns a new tensor with the hyperbolic tangent of the elements
of :attr:`input`.

.. math::
    \text{out}_{i} = \tanh(\text{input}_{i})

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8002,  1.2058,  0.1128, -0.1276])
    >>> torch.tanh(a)
    tensor([ 0.6641,  0.8354,  0.1124, -0.1269])
""")

add_docstr(torch.topk,
           r"""
topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)

Returns the :attr:`k` largest elements of the given :attr:`input` tensor along
a given dimension.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`largest` is ``False`` then the `k` smallest elements are returned.

A tuple of `(values, indices)` is returned, where the `indices` are the indices
of the elements in the original `input` tensor.

The boolean option :attr:`sorted` if ``True``, will make sure that the returned
`k` elements are themselves sorted

Args:
    input (Tensor): the input tensor
    k (int): the k in "top-k"
    dim (int, optional): the dimension to sort along
    largest (bool, optional): controls whether to return largest or
           smallest elements
    sorted (bool, optional): controls whether to return the elements
           in sorted order
    out (tuple, optional): the output tuple of (Tensor, LongTensor) that can be
        optionally given to be used as output buffers

Example::

    >>> x = torch.arange(1, 6)
    >>> x
    tensor([ 1.0000,  2.0000,  3.0000,  4.0000,  5.0000])
    >>> torch.topk(x, 3)
    (tensor([ 5.0000,  4.0000,  3.0000]), tensor([ 4,  3,  2]))
""")

add_docstr(torch.trace,
           r"""
trace(input) -> Tensor

Returns the sum of the elements of the diagonal of the input 2-D matrix.

Example::

    >>> x = torch.arange(1, 10).view(3, 3)
    >>> x
    tensor([[ 1.0000,  2.0000,  3.0000],
            [ 4.0000,  5.0000,  6.0000],
            [ 7.0000,  8.0000,  9.0000]])
    >>> torch.trace(x)
    tensor(15.0000)
""")

add_docstr(torch.transpose,
           r"""
transpose(input, dim0, dim1, out=None) -> Tensor

Returns a tensor that is a transposed version of :attr:`input`.
The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

The resulting :attr:`out` tensor shares it's underlying storage with the
:attr:`input` tensor, so changing the content of one would change the content
of the other.

Args:
    input (Tensor): the input tensor
    dim0 (int): the first dimension to be transposed
    dim1 (int): the second dimension to be transposed
    out (Tensor, optional): the output tensor

Example::

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.4867,  1.2055, -0.3030],
            [ 0.2103, -1.1797, -0.0506]])
    >>> torch.transpose(x, 0, 1)
    tensor([[ 0.4867,  0.2103],
            [ 1.2055, -1.1797],
            [-0.3030, -0.0506]])
""")

add_docstr(torch.tril,
           r"""
tril(input, diagonal=0, out=None) -> Tensor

Returns the lower triangular part of the matrix (2-D tensor) :attr:`input`,
the other elements of the result tensor :attr:`out` are set to 0.

The lower triangular part of the matrix is defined as the elements on and
below the diagonal.

The argument :attr:`diagonal` controls which diagonal to consider. If
:attr:`diagonal` = 0, all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the main
diagonal, and similarly a negative value excludes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
:math:`d_{1}, d_{2}` are the dimensions of the matrix.

Args:
    input (Tensor): the input tensor
    diagonal (int, optional): the diagonal to consider
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[ 0.2364, -0.4589, -0.3503],
            [ 1.6808,  0.5177, -0.9070],
            [-0.3948,  0.8844, -0.9413]])
    >>> torch.tril(a)
    tensor([[ 0.2364,  0.0000,  0.0000],
            [ 1.6808,  0.5177,  0.0000],
            [-0.3948,  0.8844, -0.9413]])

    >>> b = torch.randn(4, 6)
    >>> b
    tensor([[ 0.0150,  1.1514,  0.3797,  0.9921, -0.6125, -0.5048],
            [ 0.8251, -0.3581,  3.0886,  0.4950, -0.5821,  2.2400],
            [ 2.0356, -1.4413, -0.9576, -0.4324,  0.1293, -1.1867],
            [-0.3544, -0.4803, -1.0744,  1.3239, -0.1060, -1.8037]])
    >>> torch.tril(b, diagonal=1)
    tensor([[ 0.0150,  1.1514,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.8251, -0.3581,  3.0886,  0.0000,  0.0000,  0.0000],
            [ 2.0356, -1.4413, -0.9576, -0.4324,  0.0000,  0.0000],
            [-0.3544, -0.4803, -1.0744,  1.3239, -0.1060,  0.0000]])
    >>> torch.tril(b, diagonal=-1)
    tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.8251,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 2.0356, -1.4413,  0.0000,  0.0000,  0.0000,  0.0000],
            [-0.3544, -0.4803, -1.0744,  0.0000,  0.0000,  0.0000]])
""")

add_docstr(torch.triu,
           r"""
triu(input, diagonal=0, out=None) -> Tensor

Returns the upper triangular part of the matrix (2-D tensor) :attr:`input`,
the other elements of the result tensor :attr:`out` are set to 0.

The upper triangular part of the matrix is defined as the elements on and
above the diagonal.

The argument :attr:`diagonal` controls which diagonal to consider. If
:attr:`diagonal` = 0, all elements on and below the main diagonal are
retained. A positive value excludes just as many diagonals above the main
diagonal, and similarly a negative value includes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
:math:`d_{1}, d_{2}` are the dimensions of the matrix.

Args:
    input (Tensor): the input tensor
    diagonal (int, optional): the diagonal to consider
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[ 0.7720, -1.0626, -0.2119],
            [-1.8061, -1.2357,  0.0124],
            [ 0.5707, -0.1016,  0.9165]])
    >>> torch.triu(a)
    tensor([[ 0.7720, -1.0626, -0.2119],
            [ 0.0000, -1.2357,  0.0124],
            [ 0.0000,  0.0000,  0.9165]])
    >>> torch.triu(a, diagonal=1)
    tensor([[ 0.0000, -1.0626, -0.2119],
            [ 0.0000,  0.0000,  0.0124],
            [ 0.0000,  0.0000,  0.0000]])
    >>> torch.triu(a, diagonal=-1)
    tensor([[ 0.7720, -1.0626, -0.2119],
            [-1.8061, -1.2357,  0.0124],
            [ 0.0000, -0.1016,  0.9165]])

    >>> b = torch.randn(4, 6)
    >>> b
    tensor([[ 1.3253,  0.3691,  1.0717,  1.3463,  0.5515, -0.1886],
            [ 0.0203, -0.6024, -0.0106, -0.5577,  1.4009, -0.7636],
            [-0.7425,  0.4365, -0.4011, -0.7721, -0.8810,  1.3765],
            [-0.0754, -1.2553, -1.3518,  1.0655,  0.5552, -2.2156]])
    >>> torch.tril(b, diagonal=1)
    tensor([[ 1.3253,  0.3691,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0203, -0.6024, -0.0106,  0.0000,  0.0000,  0.0000],
            [-0.7425,  0.4365, -0.4011, -0.7721,  0.0000,  0.0000],
            [-0.0754, -1.2553, -1.3518,  1.0655,  0.5552,  0.0000]])
    >>> torch.tril(b, diagonal=-1)
    tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0203,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [-0.7425,  0.4365,  0.0000,  0.0000,  0.0000,  0.0000],
            [-0.0754, -1.2553, -1.3518,  0.0000,  0.0000,  0.0000]])
""")

add_docstr(torch.trtrs,
           r"""
trtrs(b, A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)

Solves a system of equations with a triangular coefficient matrix `A`
and multiple right-hand sides `b`.

In particular, solves :math:`AX = b` and assumes `A` is upper-triangular
with the default keyword arguments.

This method is NOT implemented for CUDA tensors.

Args:
    A (Tensor): the input triangular coefficient matrix
    b (Tensor): multiple right-hand sides. Each column of `b` is a
        right-hand side for the system of equations.
    upper (bool, optional): whether to solve the upper-triangular system
        of equations (default) or the lower-triangular system of equations. Default: True.
    transpose (bool, optional): whether `A` should be transposed before
        being sent into the solver. Default: False.
    unitriangular (bool, optional): whether `A` is unit triangular.
        If True, the diagonal elements of `A` are assumed to be
        1 and not referenced from `A`. Default: False.

Returns:
    A tuple (X, M) where `M` is a clone of `A` and `X` is the solution to
    `AX = b` (or whatever variant of the system of equations, depending on
    the keyword arguments.)

Shape:
    - A: :math:`(N, N)`
    - b: :math:`(N, C)`
    - output[0]: :math:`(N, C)`
    - output[1]: :math:`(N, N)`

Examples::

    >>> A = torch.randn(2, 2).triu()
    >>> A
    tensor([[-0.3846, -0.4981],
            [ 0.0000, -0.3679]])
    >>> b = torch.randn(2, 3)
    >>> b
    tensor([[ 0.5021, -1.3358,  0.6177],
            [ 0.3076, -1.3518, -0.7889]])
    >>> torch.trtrs(b, A)
    (tensor([[-0.2228, -1.2853, -4.3828],
            [-0.8361,  3.6742,  2.1442]]), tensor([[-0.3846, -0.4981],
            [ 0.0000, -0.3679]]))
""")

add_docstr(torch.trunc,
           r"""
trunc(input, out=None) -> Tensor

Returns a new tensor with the truncated integer values of
the elements of :attr:`input`.

Args:
    input (Tensor): the input tensor
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.3490, -0.1184, -1.6461, -0.4134])
    >>> torch.trunc(a)
    tensor([ 0.0000, -0.0000, -1.0000, -0.0000])
""")

add_docstr(torch.unsqueeze,
           r"""
unsqueeze(input, dim, out=None) -> Tensor

Returns a new tensor with a dimension of size one inserted at the
specified position.

The returned tensor shares the same underlying data with this tensor.

A negative `dim` value within the range
[-:attr:`input.dim()`, :attr:`input.dim()`) can be used and
will correspond to :meth:`unsqueeze` applied at :attr:`dim` = :attr:`dim + input.dim() + 1`

Args:
    input (Tensor): the input tensor
    dim (int): the index at which to insert the singleton dimension
    out (Tensor, optional): the output tensor

Example::

    >>> x = torch.Tensor([1, 2, 3, 4])
    >>> torch.unsqueeze(x, 0)
    tensor([[ 1.0000,  2.0000,  3.0000,  4.0000]])
    >>> torch.unsqueeze(x, 1)
    tensor([[ 1.0000],
            [ 2.0000],
            [ 3.0000],
            [ 4.0000]])
""")

add_docstr(torch.var,
           r"""
.. function:: var(input, unbiased=True) -> Tensor

Returns the variance of all elements in the :attr:`input` tensor.

If :attr:`unbiased` is ``False``, then the variance will be calculated via the
biased estimator. Otherwise, Bessel's correction will be used.

Args:
    input (Tensor): the input tensor
    unbiased (bool): whether to use the unbiased estimation or not

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.8279, -0.0047, -0.5285]])
    >>> torch.var(a)
    tensor(0.4679)


.. function:: var(input, dim, keepdim=False, unbiased=True, out=None) -> Tensor

Returns the variance of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`.

If :attr:`keepdim` is ``True``, the output tensors are of the same size
as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in
the outputs tensor having 1 fewer dimension than :attr:`input`.

If :attr:`unbiased` is ``False``, then the variance will be calculated via the
biased estimator. Otherwise, Bessel's correction will be used.

Args:
    input (Tensor): the input tensor
    dim (int): the dimension to reduce
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not
    unbiased (bool): whether to use the unbiased estimation or not
    out (Tensor, optional): the output tensor

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[-1.0646, -0.8755, -0.2777, -1.6317],
            [-0.3234,  0.1469,  1.4604, -0.4194],
            [ 1.0068,  0.0540, -0.2668,  0.6470],
            [ 0.1469,  0.8101, -0.2545, -1.2596]])
    >>> torch.var(a, 1)
    tensor([ 0.3116,  0.7493,  0.3291,  0.7505])
""")

add_docstr(torch.zeros,
           r"""
zeros(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with the scalar value `0`, with the shape defined
by the variable argument :attr:`sizes`.

Args:
    sizes (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.zeros(2, 3)
    tensor([[ 0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000]])

    >>> torch.zeros(5)
    tensor([ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000])
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

    >>> input = torch.FloatTensor(2, 3)
    >>> torch.zeros_like(input)
    tensor([[ 0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000]])
""".format(**factory_like_common_args))

add_docstr(torch.btrifact_with_info,
           r"""
btrifact_with_info(A, pivot=True) -> (Tensor, IntTensor, IntTensor)

Batch LU factorization with additional error information.

This is a version of :meth:`torch.btrifact` that always creates an info
`IntTensor`, and returns it as the third return value.

Arguments:
    A (Tensor): the tensor to factor
    pivot (bool, optional): controls whether pivoting is done

Returns:
    A tuple containing factorization, pivots, and an `IntTensor` where non-zero
    values indicate whether factorization for each minibatch sample succeeds.

Example::

    >>> A = torch.randn(2, 3, 3)
    >>> A_LU, pivots, info = A.btrifact_with_info()
    >>> if info.nonzero().size(0) == 0:
    >>>   print('LU factorization succeeded for all samples!')
    LU factorization succeeded for all samples!
""")

add_docstr(torch.btrisolve,
           r"""
btrisolve(b, LU_data, LU_pivots) -> Tensor

Batch LU solve.

Returns the LU solve of the linear system :math:`Ax = b`.

Arguments:
    b (Tensor): the RHS tensor
    LU_data (Tensor): the pivoted LU factorization of A from :meth:`btrifact`.
    LU_pivots (IntTensor): the pivots of the LU factorization

Example::

    >>> A = torch.randn(2, 3, 3)
    >>> b = torch.randn(2, 3)
    >>> A_LU = torch.btrifact(A)
    >>> x = torch.btrisolve(b, *A_LU)
    >>> torch.norm(torch.bmm(A, x.unsqueeze(2)) - b.unsqueeze(2))
    tensor(1.00000e-07 *
           1.4818)
""")

add_docstr(torch.empty,
           r"""
empty(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a tensor filled with uninitialized data. The shape of the tensor is
defined by the variable argument :attr:`sizes`.

Args:
    sizes (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.
    {out}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Example::

    >>> torch.empty(2, 3)
    tensor(1.00000e-19 *
           [[ 2.9164,  0.0000,  0.1163],
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

    >>> input = torch.empty((2,3), dtype=torch.int64)
    >>> input.new(input.size())
    tensor([[ 9.3962e+13,  2.8000e+01,  9.3965e+13],
            [ 7.5751e+18,  7.1428e+18,  7.5955e+18]])
""".format(**factory_like_common_args))

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
``torch.full_like(input.size(), fill_value, dtype=input.dtype, layout=input.layout, device=input.device)``.

Args:
    {input}
    fill_value: the number to fill the output tensor with.
    {dtype}
    {layout}
    {device}
    {requires_grad}
""".format(**factory_like_common_args))

add_docstr(torch.stft,
           r"""
stft(signal, frame_length, hop, fft_size=None, normalized=False, onesided=True, window=None, pad_end=0) -> Tensor

Short-time Fourier transform (STFT).

Ignoring the batch dimension, this method computes the following expression:

.. math::
    X[m, \omega] = \sum_{k = 0}^{\text{frame_length}}%
                        window[k]\ signal[m \times hop + k]\ e^{- j \frac{2 \pi \cdot \omega k}{\text{frame_length}}},

where :math:`m` is the index of the sliding window, and :math:`\omega` is
the frequency that :math:`0 \leq \omega <` :attr:`fft_size`. When
:attr:`return_onsesided` is the default value ``True``, only values for
:math:`\omega` in range :math:`\left[0, 1, 2, \dots, \left\lfloor \frac{\text{fft_size}}{2} \right\rfloor + 1\right]`
are returned because the real-to-complex transform satisfies the Hermitian
symmetry, i.e., :math:`X[m, \omega] = X[m, \text{fft_size} - \omega]^*`.

The input :attr:`signal` must be 1-D sequence :math:`(T)` or 2-D a batch of
sequences :math:`(N \times T)`. If :attr:`fft_size` is ``None``, it is
default to same value as  :attr:`frame_length`. :attr:`window` can be a
1-D tensor of size :attr:`frame_length`, e.g., see
:meth:`torch.hann_window`. If :attr:`window` is the default value ``None``,
it is treated as if having :math:`1` everywhere in the frame.
:attr:`pad_end` indicates the amount of zero padding at the end of
:attr:`signal` before STFT. If :attr:`normalized` is set to ``True``, the
function returns the normalized STFT results, i.e., multiplied by
:math:`(frame\_length)^{-0.5}`.

Returns the real and the imaginary parts together as one tensor of size
:math:`(* \times N \times 2)`, where :math:`*` is the shape of input :attr:`signal`,
:math:`N` is the number of :math:`\omega` s considered depending on
:attr:`fft_size` and :attr:`return_onesided`, and each pair in the last
dimension represents a complex number as real part and imaginary part.

Arguments:
    signal (Tensor): the input tensor
    frame_length (int): the size of window frame and STFT filter
    hop (int): the distance between neighboring sliding window frames
    fft_size (int, optional): size of Fourier transform. Default: ``None``
    normalized (bool, optional): controls whether to return the normalized STFT results
         Default: ``False``
    onesided (bool, optional): controls whether to return half of results to
        avoid redundancy Default: ``True``
    window (Tensor, optional): the optional window function. Default: ``None``
    pad_end (int, optional): implicit zero padding at the end of :attr:`signal`. Default: 0

Returns:
    Tensor: A tensor containing the STFT result
""")

add_docstr(torch.det,
           r"""
det(A) -> Tensor

Calculates determinant of a 2D square tensor.

.. note::
    Backward through :meth:`det` internally uses SVD results when :attr:`A` is
    not invertible. In this case, double backward through :meth:`det` will be
    unstable in when :attr:`A` doesn't have distinct singular values. See
    :meth:`~torch.svd` for details.

Arguments:
    A (Tensor): The input 2D square tensor

Example::

    >>> A = torch.randn(3, 3)
    >>> torch.det(A)
    tensor(-0.9116)
""")

add_docstr(torch.where,
           r"""
where(condition, x, y) -> Tensor

Return a tensor of elements selected from either :attr:`x` or :attr:`y`, depending on :attr:`condition`.

The operation is defined as:

.. math::
    out_i = \begin{cases}
        x_i & \text{if } condition_i \\
        y_i & \text{otherwise} \\
    \end{cases}

.. note::
    The tensors :attr:`condition`, :attr:`x`, :attr:`y` must be :ref:`broadcastable <broadcasting-semantics>`.

Arguments:
    condition (ByteTensor): When True (nonzero), yield x, otherwise yield y
    x (Tensor): values selected at indices where :attr:`condition` is ``True``
    y (Tensor): values selected at indices where :attr:`condition` is ``False``

Returns:
    Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`x`, :attr:`y`

Example::

    >>> x = torch.randn(3, 2)
    >>> y = torch.ones(3, 2)
    >>> x
    tensor([[-1.0578,  0.6016],
            [-0.4729,  0.8593],
            [-0.5911, -0.8716]])
    >>> torch.where(x > 0, x, y)
    tensor([[ 1.0000,  0.6016],
            [ 1.0000,  0.8593],
            [ 1.0000,  1.0000]])
""")

add_docstr(torch.logdet,
           r"""
logdet(A) -> Tensor

Calculates log determinant of a 2D square tensor.

.. note::
    Result is ``-inf`` if :attr:`A` has zero log determinant, and is ``nan`` if
    :attr:`A` has negative determinant.

.. note::
    Backward through :meth:`logdet` internally uses SVD results when :attr:`A`
    is not invertible. In this case, double backward through :meth:`logdet` will
    be unstable in when :attr:`A` doesn't have distinct singular values. See
    :meth:`~torch.svd` for details.

Arguments:
    A (Tensor): The input 2D square tensor

Example::

    >>> A = torch.randn(3, 3)
    >>> torch.det(A)
    tensor(6.9680)
    >>> torch.logdet(A)
    tensor(1.9413)
""")

add_docstr(torch.slogdet,
           r"""
slogdet(A) -> (Tensor, Tensor)

Calculates the sign and log value of a 2D square tensor's determinant.

.. note::
    If ``A`` has zero determinant, this returns ``(0, -inf)``.

.. note::
    Backward through :meth:`slogdet` internally uses SVD results when :attr:`A`
    is not invertible. In this case, double backward through :meth:`slogdet`
    will be unstable in when :attr:`A` doesn't have distinct singular values.
    See :meth:`~torch.svd` for details.

Arguments:
    A (Tensor): The input 2D square tensor

Returns:
    A tuple containing the sign of the determinant, and the log value of the
    absolute determinant.

Example::

    >>> A = torch.randn(3, 3)
    >>> torch.det(A)
    tensor(-3.9515)
    >>> torch.logdet(A)
    tensor(nan)
    >>> torch.slogdet(A)
    (tensor(-1.0000), tensor(1.3741))
""")

add_docstr(torch.fft,
           r"""
fft(input, signal_ndim, normalized=False) -> Tensor

Complex-to-complex Discrete Fourier Transform

This method computes the complex-to-complex discrete Fourier transform.
Ignoring the batch dimensions, it computes the following expression:

.. math::
    X[\omega_1, \dots, \omega_d] =
        \frac{1}{\prod_{i=1}^d N_i} \sum_{n_1=0}^{N_1} \dots \sum_{n_d=0}^{N_d} x[n_1, \dots, n_d]
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

.. warning::
    For CPU tensors, this method is currently only available with MKL. Check
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
        \frac{1}{\prod_{i=1}^d N_i} \sum_{n_1=0}^{N_1} \dots \sum_{n_d=0}^{N_d} x[n_1, \dots, n_d]
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

.. warning::
    For CPU tensors, this method is currently only available with MKL. Check
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
by multiplying it with :math:`\sqrt{\prod_{i=1}^K N_i}` so that the operator is
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

.. warning::
    For CPU tensors, this method is currently only available with MKL. Check
    :func:`torch.backends.mkl.is_available` to check if MKL is installed.

Arguments:
    input (Tensor): the input tensor of at least :attr:`signal_ndim` dimensions
    signal_ndim (int): the number of dimensions in each signal.
        :attr:`signal_ndim` can only be 1, 2 or 3
    normalized (bool, optional): controls whether to return normalized results.
        Default: ``False``
    onesided (bool, optional): controls whether to return half of results to
        avoid redundancy Default: ``True``

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
    Generally speaking, the input of this function should contain values
    following conjugate symmetry. Note that even if :attr:`onesided` is
    ``True``, often symmetry on some part is still needed. When this
    requirement is not satisfied, the behavior of :func:`~torch.irfft` is
    undefined. Since :func:`torch.autograd.gradcheck` estimates numerical
    Jacobian with point perturbations, :func:`~torch.irfft` will almost
    certainly fail the check.

.. warning::
    For CPU tensors, this method is currently only available with MKL. Check
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
    tensor([[-0.0411,  1.1699,  0.7434, -0.2244,  0.6236],
            [-0.5433, -0.5229, -0.6124, -0.3391, -0.8716],
            [ 0.6708, -2.0167, -1.1008, -0.2531,  0.1826],
            [-0.1724, -1.0674, -0.3188, -1.0868, -0.3041]])
    >>> y = torch.rfft(x, 2, onesided=True)
    >>> torch.irfft(y, 2, onesided=True, signal_sizes=x.shape)  # recover x
    tensor([[-0.0411,  1.1699,  0.7434, -0.2244,  0.6236],
            [-0.5433, -0.5229, -0.6124, -0.3391, -0.8716],
            [ 0.6708, -2.0167, -1.1008, -0.2531,  0.1826],
            [-0.1724, -1.0674, -0.3188, -1.0868, -0.3041]])

""")
