"""Adds docstrings to functions defined in the torch._C"""

import torch._C
from torch._C import _add_docstr as add_docstr

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

     1
     2
     3
    [torch.FloatTensor of size (3,)]

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

    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.acos(a)

     2.2608
     1.2956
     1.1075
        nan
    [torch.FloatTensor of size (4,)]

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

     0.4050
    -1.2227
     1.8688
    -0.4185
    [torch.FloatTensor of size (4,)]

    >>> torch.add(a, 20)

     20.4050
     18.7773
     21.8688
     19.5815
    [torch.FloatTensor of size (4,)]



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

    >>> import torch
    >>> a = torch.randn(4)
    >>> a

    -0.9310
     2.0330
     0.0852
    -0.2941
    [torch.FloatTensor of size (4,)]

    >>> b = torch.randn(2, 2)
    >>> b

     1.0663  0.2544
    -0.1513  0.0749
    [torch.FloatTensor of size (2,2)]

    >>> torch.add(a, 10, b)
     9.7322
     4.5770
    -1.4279
     0.4552
    [torch.FloatTensor of size (4,)]


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

     -3.1162  11.0071   7.3102   0.1824  -7.6892
      1.8265   6.0739   0.4589  -0.5641  -5.4283
     -9.3387  -0.1794  -1.2318  -6.8841  -4.7239
    [torch.FloatTensor of size (3,5)]
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

    >>> t = torch.randn(2, 3)
    >>> t1 = torch.randn(1, 6)
    >>> t2 = torch.randn(6, 1)
    >>> torch.addcdiv(t, 0.1, t1, t2)

     0.0122 -0.0188 -0.2354
     0.7396 -1.5721  1.2878
    [torch.FloatTensor of size (2,3)]
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

    >>> t = torch.randn(2, 3)
    >>> t1 = torch.randn(1, 6)
    >>> t2 = torch.randn(6, 1)
    >>> torch.addcmul(t, 0.1, t1, t2)

     0.0122 -0.0188 -0.2354
     0.7396 -1.5721  1.2878
    [torch.FloatTensor of size (2,3)]
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

    -0.4095 -1.9703  1.3561
     5.7674 -4.9760  2.7378
    [torch.FloatTensor of size (2,3)]
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

    -2.0939
    -2.2950
    [torch.FloatTensor of size (2,)]
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

     1  2
     2  4
     3  6
    [torch.FloatTensor of size (3,2)]
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

    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.asin(a)

    -0.6900
     0.2752
     0.4633
        nan
    [torch.FloatTensor of size (4,)]
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

    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.atan(a)

    -0.5669
     0.2653
     0.4203
     0.9196
    [torch.FloatTensor of size (4,)]
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

    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.atan2(a, torch.randn(4))

    -2.4167
     2.9755
     0.9363
     1.6613
    [torch.FloatTensor of size (4,)]
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

     0.7544  0.8140  0.9842
     0.5282  0.0595  0.6445
     0.1925  0.9553  0.9732
    [torch.FloatTensor of size (3,3)]

    >>> torch.bernoulli(a)

     1  1  1
     0  0  1
     0  1  1
    [torch.FloatTensor of size (3,3)]

    >>> a = torch.ones(3, 3) # probability of drawing "1" is 1
    >>> torch.bernoulli(a)

     1  1  1
     1  1  1
     1  1  1
    [torch.FloatTensor of size (3,3)]

    >>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
    >>> torch.bernoulli(a)

     0  0  0
     0  0  0
     0  0  0
    [torch.FloatTensor of size (3,3)]

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

     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size (2,3)]

    >>> torch.cat((x, x, x), 0)

     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size (6,3)]

    >>> torch.cat((x, x, x), 1)

     0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size (2,9)]

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

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size (4,)]

    >>> torch.ceil(a)

     2
     1
    -0
    -0
    [torch.FloatTensor of size (4,)]

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

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size (4,)]

    >>> torch.reciprocal(a)

     0.7210
     2.5565
    -1.1583
    -1.8289
    [torch.FloatTensor of size (4,)]

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

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size (4,)]

    >>> torch.clamp(a, min=-0.5, max=0.5)

     0.5000
     0.3912
    -0.5000
    -0.5000
    [torch.FloatTensor of size (4,)]

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

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size (4,)]

    >>> torch.clamp(a, min=0.5)

     1.3869
     0.5000
     0.5000
     0.5000
    [torch.FloatTensor of size (4,)]

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

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size (4,)]

    >>> torch.clamp(a, max=0.5)

     0.5000
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size (4,)]

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
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.cos(a)
     0.8041
     0.9633
     0.9018
     0.2557
    [torch.FloatTensor of size (4,)]
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
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.cosh(a)
     1.2095
     1.0372
     1.1015
     1.9917
    [torch.FloatTensor of size (4,)]
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

    -0.6652 -1.0116 -0.6857
     0.2286  0.4446 -0.5272
     0.0476  0.2321  1.9991
     0.6199  1.1924 -0.9397
    [torch.FloatTensor of size (4,3)]

    >>> b = torch.randn(4, 3)
    >>> b

    -0.1042 -1.1156  0.1947
     0.9947  0.1149  0.4701
    -1.0108  0.8319 -0.0750
     0.9045 -1.3754  1.0976
    [torch.FloatTensor of size (4,3)]

    >>> torch.cross(a, b, dim=1)

    -0.9619  0.2009  0.6367
     0.2696 -0.6318 -0.4160
    -1.6805 -2.0171  0.2741
     0.0163 -1.5304 -1.9311
    [torch.FloatTensor of size (4,3)]

    >>> torch.cross(a, b)

    -0.9619  0.2009  0.6367
     0.2696 -0.6318 -0.4160
    -1.6805 -2.0171  0.2741
     0.0163 -1.5304 -1.9311
    [torch.FloatTensor of size (4,3)]
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

     1.1148
     1.8423
     1.4143
    -0.4403
     1.2859
    -1.2514
    -0.4748
     1.1735
    -1.6332
    -0.4272
    [torch.FloatTensor of size (10,)]

    >>> torch.cumprod(a, dim=0)

     1.1148
     2.0537
     2.9045
    -1.2788
    -1.6444
     2.0578
    -0.9770
    -1.1466
     1.8726
    -0.8000
    [torch.FloatTensor of size (10,)]

    >>> a[5] = 0.0
    >>> torch.cumprod(a, dim=0)

     1.1148
     2.0537
     2.9045
    -1.2788
    -1.6444
    -0.0000
     0.0000
     0.0000
    -0.0000
     0.0000
    [torch.FloatTensor of size (10,)]

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

    -0.6039
    -0.2214
    -0.3705
    -0.0169
     1.3415
    -0.1230
     0.9719
     0.6081
    -0.1286
     1.0947
    [torch.FloatTensor of size (10,)]

    >>> torch.cumsum(a, dim=0)

    -0.6039
    -0.8253
    -1.1958
    -1.2127
     0.1288
     0.0058
     0.9777
     1.5858
     1.4572
     2.5519
    [torch.FloatTensor of size (10,)]


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

     1.0480
    -2.3405
    -1.1138
    [torch.FloatTensor of size (3,)]

    >>> torch.diag(a)

     1.0480  0.0000  0.0000
     0.0000 -2.3405  0.0000
     0.0000  0.0000 -1.1138
    [torch.FloatTensor of size (3,3)]

    >>> torch.diag(a, 1)

     0.0000  1.0480  0.0000  0.0000
     0.0000  0.0000 -2.3405  0.0000
     0.0000  0.0000  0.0000 -1.1138
     0.0000  0.0000  0.0000  0.0000
    [torch.FloatTensor of size (4,4)]


Get the k-th diagonal of a given matrix::

    >>> a = torch.randn(3, 3)
    >>> a

    -1.5328 -1.3210 -1.5204
     0.8596  0.0471 -0.2239
    -0.6617  0.0146 -1.0817
    [torch.FloatTensor of size (3,3)]

    >>> torch.diag(a, 0)

    -1.5328
     0.0471
    -1.0817
    [torch.FloatTensor of size (3,)]

    >>> torch.diag(a, 1)

    -1.3210
    -0.2239
    [torch.FloatTensor of size (2,)]

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

     1.0480
    -2.3405
    -1.1138
    [torch.FloatTensor of size 3]

    >>> torch.diagflat(a)

     1.0480  0.0000  0.0000
     0.0000 -2.3405  0.0000
     0.0000  0.0000 -1.1138
    [torch.FloatTensor of size 3x3]

    >>> torch.diagflat(a, 1)

     0.0000  1.0480  0.0000  0.0000
     0.0000  0.0000 -2.3405  0.0000
     0.0000  0.0000  0.0000 -1.1138
     0.0000  0.0000  0.0000  0.0000
    [torch.FloatTensor of size 4x4]

    >>> a = torch.randn(2, 2)
    >>> a

     0.1761 -0.9121
    -0.5722  1.5219
    [torch.FloatTensor of size (2,2)]

    >>> torch.diagflat(a)

     0.1761  0.0000  0.0000  0.0000
     0.0000 -0.9121  0.0000  0.0000
     0.0000  0.0000 -0.5722  0.0000
     0.0000  0.0000  0.0000  1.5219
    [torch.FloatTensor of size (4,4)]

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

    -1.5328 -1.3210 -1.5204
     0.8596  0.0471 -0.2239
    -0.6617  0.0146 -1.0817
    [torch.FloatTensor of size 3x3]

    >>> torch.diagonal(a, 0)

    -1.5328
     0.0471
    -1.0817
    [torch.FloatTensor of size 3]

    >>> torch.diagonal(a, 1)

    -1.3210
    -0.2239
    [torch.FloatTensor of size 2]

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

    -1.5474
    -0.4649
     0.5954
    -0.8610
    [torch.FloatTensor of size (4,)]

    >>> y = torch.randn(4)
    >>> y

     1.7141
     0.3274
    -1.2772
    -0.4725
    [torch.FloatTensor of size (4,)]

    >>> torch.dist(x, y, 3.5)

     3.3953
    [torch.FloatTensor of size ()]

    >>> torch.dist(x, y, 3)

     3.4710
    [torch.FloatTensor of size ()]

    >>> torch.dist(x, y, 0)

    inf
    [torch.FloatTensor of size ()]

    >>> torch.dist(x, y, 1)

     6.3150
    [torch.FloatTensor of size ()]


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

    -0.6147
    -1.1237
    -0.1604
    -0.6853
     0.1063
    [torch.FloatTensor of size (5,)]

    >>> torch.div(a, 0.5)

    -1.2294
    -2.2474
    -0.3208
    -1.3706
     0.2126
    [torch.FloatTensor of size (5,)]


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

    -0.1810  0.4017  0.2863 -0.1013
     0.6183  2.0696  0.9012 -1.5933
     0.5679  0.4743 -0.0117 -0.1266
    -0.1213  0.9629  0.2682  1.5968
    [torch.FloatTensor of size (4,4)]

    >>> b = torch.randn(8, 2)
    >>> b

     0.8774  0.7650
     0.8866  1.4805
    -0.6490  1.1172
     1.4259 -0.8146
     1.4633 -0.1228
     0.4643 -0.6029
     0.3492  1.5270
     1.6103 -0.6291
    [torch.FloatTensor of size (8,2)]

    >>> torch.div(a, b)

    -0.2062  0.5251  0.3229 -0.0684
    -0.9528  1.8525  0.6320  1.9559
     0.3881 -3.8625 -0.0253  0.2099
    -0.3473  0.6306  0.1666 -2.5381
    [torch.FloatTensor of size (4,4)]


""")

add_docstr(torch.dot,
           r"""
dot(tensor1, tensor2) -> Tensor

Computes the dot product (inner product) of two tensors.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

Example::

    >>> torch.dot(torch.Tensor([2, 3]), torch.Tensor([2, 1]))

     7
    [torch.FloatTensor of size ()]
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

    1  0
    0  1
    [torch.ByteTensor of size (2,2)]
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

     0.0000
    -0.8427
     1.0000
    [torch.FloatTensor of size (3,)]
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

     0.0000
     0.4769
       -inf
    [torch.FloatTensor of size (3,)]
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

     1
     2
    [torch.FloatTensor of size (2,)]
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

     0
     1
    [torch.FloatTensor of size (2,)]
""")

add_docstr(torch.eye,
           r"""
eye(n, m=None, out=None)

Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

Args:
    n (int): the number of rows
    m (int, optional): the number of columns with default being :attr:`n`
    out (Tensor, optional): the output tensor

Returns:
    Tensor: A 2-D tensor with ones on the diagonal and zeros elsewhere

Example::

    >>> torch.eye(3)

     1  0  0
     0  1  0
     0  0  1
    [torch.FloatTensor of size (3,3)]
""")

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

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size (4,)]

    >>> torch.floor(a)

     1
     0
    -1
    -1
    [torch.FloatTensor of size (4,)]


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

    -1
    -0
    -1
     1
     0
     1
    [torch.FloatTensor of size (6,)]

    >>> torch.fmod(torch.Tensor([1, 2, 3, 4, 5]), 1.5)

     1.0000
     0.5000
     0.0000
     1.0000
     0.5000
    [torch.FloatTensor of size (5,)]

.. seealso::

        :func:`torch.remainder`, which computes the element-wise remainder of
        division equivalently to Python's `%` operator
""")

add_docstr(torch.frac,
           r"""
frac(tensor, out=None) -> Tensor

Computes the fractional portion of each element in :attr:`tensor`.

.. math::
    \text{out}_{i} = \text{input}_{i} - \left\lfloor \text{input}_{i} \right\rfloor

Example::

    >>> torch.frac(torch.Tensor([1, 2.5, -3.2]))

     0.0000
     0.5000
    -0.2000
    [torch.FloatTensor of size (3,)]
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

     1
     2
     3
    [torch.LongTensor of size (3,)]

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
    >>> torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))

     1  1
     4  3
    [torch.FloatTensor of size (2,2)]
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

     1  1
     0  1
    [torch.ByteTensor of size (2,2)]
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

      2.0000   1.0000
      1.0000   1.0000
      1.0000   2.0000
     10.9635   4.8501
      8.9332   5.2418
    [torch.FloatTensor of size (5,2)]
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

      1   2   3
      2   4   6
      3   6   9
      4   8  12
    [torch.FloatTensor of size (4,3)]

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

    1.00000e-06 *
      7.0977
    [torch.FloatTensor of size ()]

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

     0  1
     0  0
    [torch.ByteTensor of size (2,2)]
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

     0
     2
     1
     0
    [torch.FloatTensor of size (4,)]


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

     1.2045  2.4084  0.4001  1.1372
     0.5596  1.5677  0.6219 -0.7954
     1.3635 -1.2313 -0.5414 -1.8478
    [torch.FloatTensor of size (3,4)]

    >>> indices = torch.LongTensor([0, 2])
    >>> torch.index_select(x, 0, indices)

     1.2045  2.4084  0.4001  1.1372
     1.3635 -1.2313 -0.5414 -1.8478
    [torch.FloatTensor of size (2,4)]

    >>> torch.index_select(x, 1, indices)

     1.2045  0.4001
     0.5596  0.6219
     1.3635 -0.5414
    [torch.FloatTensor of size (3,2)]

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

    >>> x = torch.rand(10, 10)
    >>> y = torch.inverse(x)
    >>> z = torch.mm(x, y)
    >>> z

     1.0000  0.0000  0.0000 -0.0000  0.0000  0.0000  0.0000  0.0000 -0.0000 -0.0000
     0.0000  1.0000 -0.0000  0.0000  0.0000  0.0000 -0.0000 -0.0000 -0.0000 -0.0000
     0.0000  0.0000  1.0000 -0.0000 -0.0000  0.0000  0.0000  0.0000 -0.0000 -0.0000
     0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000 -0.0000 -0.0000  0.0000
     0.0000  0.0000 -0.0000 -0.0000  1.0000  0.0000  0.0000 -0.0000 -0.0000 -0.0000
     0.0000  0.0000  0.0000 -0.0000  0.0000  1.0000 -0.0000 -0.0000 -0.0000 -0.0000
     0.0000  0.0000  0.0000 -0.0000  0.0000  0.0000  1.0000  0.0000 -0.0000  0.0000
     0.0000  0.0000 -0.0000 -0.0000  0.0000  0.0000 -0.0000  1.0000 -0.0000  0.0000
    -0.0000  0.0000 -0.0000 -0.0000  0.0000  0.0000 -0.0000 -0.0000  1.0000 -0.0000
    -0.0000  0.0000 -0.0000 -0.0000 -0.0000  0.0000 -0.0000 -0.0000  0.0000  1.0000
    [torch.FloatTensor of size (10,10)]

    >>> torch.max(torch.abs(z - torch.eye(10))) # Max nonzero

    1.00000e-07 *
      5.0967
    [torch.FloatTensor of size ()]

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

     1
     2
     3
     4
     5
    [torch.FloatTensor of size (5,)]

    >>> torch.kthvalue(x, 4)
    (
     4
    [torch.FloatTensor of size (1,)]
    ,
     3
    [torch.LongTensor of size (1,)]
    )

    >>> x=torch.arange(1,7).resize_(2,3)
    >>> x

    1  2  3
    4  5  6
    [torch.FloatTensor of size (2,3)]

    >>> torch.kthvalue(x,2,0,True)
    (
    4  5  6
    [torch.FloatTensor of size (1,3)]
           ,
    1  1  1
    [torch.LongTensor of size (1,3)]
    )
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

     1  0
     1  1
    [torch.ByteTensor of size (2,2)]
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

     1
     2
     3
     4
    [torch.FloatTensor of size (4,)]

    >>> end

     10
     10
     10
     10
    [torch.FloatTensor of size (4,)]

    >>> torch.lerp(start, end, 0.5)

     5.5000
     6.0000
     6.5000
     7.0000
    [torch.FloatTensor of size (4,)]

""")

add_docstr(torch.linspace,
           r"""
linspace(start, end, steps=100, out=None) -> Tensor

Returns a one-dimensional tensor of :attr:`steps`
equally spaced points between :attr:`start` and :attr:`end`.

The output tensor is 1-D of size :attr:`steps`.

Args:
    start (float): the starting value for the set of points
    end (float): the ending value for the set of points
    steps (int): number of points to sample between :attr:`start`
        and :attr:`end`
    out (Tensor, optional): the output tensor

Example::

    >>> torch.linspace(3, 10, steps=5)

      3.0000
      4.7500
      6.5000
      8.2500
     10.0000
    [torch.FloatTensor of size (5,)]

    >>> torch.linspace(-10, 10, steps=5)

    -10
     -5
      0
      5
     10
    [torch.FloatTensor of size (5,)]

    >>> torch.linspace(start=-10, end=10, steps=5)

    -10
     -5
      0
      5
     10
    [torch.FloatTensor of size (5,)]

""")

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

    -0.4183
     0.3722
    -0.3091
     0.4149
     0.5857
    [torch.FloatTensor of size (5,)]

    >>> torch.log(a)

        nan
    -0.9883
        nan
    -0.8797
    -0.5349
    [torch.FloatTensor of size (5,)]

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

     0.4496
     0.1608
     0.6884
     0.8989
     0.1774
    [torch.FloatTensor of size (5,)]

    >>> torch.log10(a)

    -0.3472
    -0.7937
    -0.1622
    -0.0463
    -0.7511
    [torch.FloatTensor of size (5,)]
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

    -0.4183
     0.3722
    -0.3091
     0.4149
     0.5857
    [torch.FloatTensor of size (5,)]

    >>> torch.log1p(a)

    -0.5418
     0.3164
    -0.3697
     0.3471
     0.4611
    [torch.FloatTensor of size (5,)]

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

     0.2260
     0.0541
     0.3393
     0.7210
     0.0058
    [torch.FloatTensor of size (5,)]

    >>> torch.log2(a)

    -2.1458
    -4.2070
    -1.5593
    -0.4719
    -7.4246
    [torch.FloatTensor of size (5,)]
""")

add_docstr(torch.logspace,
           r"""
logspace(start, end, steps=100, out=None) -> Tensor

Returns a one-dimensional tensor of :attr:`steps` points
logarithmically spaced between :math:`10^{\text{start}}` and :math:`10^{\text{end}}`.

The output is a 1-D tensor of size :attr:`steps`.

Args:
    start (float): the starting value for the set of points
    end (float): the ending value for the set of points
    steps (int): number of points to sample between
        :attr:`start` and :attr:`end`
    out (Tensor, optional): the output tensor

Example::

    >>> torch.logspace(start=-10, end=10, steps=5)

     1.0000e-10
     1.0000e-05
     1.0000e+00
     1.0000e+05
     1.0000e+10
    [torch.FloatTensor of size (5,)]

    >>> torch.logspace(start=0.1, end=1.0, steps=5)

      1.2589
      2.1135
      3.5481
      5.9566
     10.0000
    [torch.FloatTensor of size (5,)]

""")

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

     0  0
     1  0
    [torch.ByteTensor of size (2,2)]
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

     1.2045  2.4084  0.4001  1.1372
     0.5596  1.5677  0.6219 -0.7954
     1.3635 -1.2313 -0.5414 -1.8478
    [torch.FloatTensor of size (3,4)]

    >>> mask = x.ge(0.5)
    >>> mask

     1  1  0  1
     1  1  1  0
     1  0  0  0
    [torch.ByteTensor of size (3,4)]

    >>> torch.masked_select(x, mask)

     1.2045
     2.4084
     1.1372
     0.5596
     1.5677
     0.6219
     1.3635
    [torch.FloatTensor of size (7,)]

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

     0.4729 -0.2266 -0.2085
    [torch.FloatTensor of size (1,3)]

    >>> torch.max(a)

     0.4729
    [torch.FloatTensor of size ()]


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

    >> a = torch.randn(4, 4)
    >> a

    0.0692  0.3142  1.2513 -0.5428
    0.9288  0.8552 -0.2073  0.6409
    1.0695 -0.0101 -2.4507 -1.2230
    0.7426 -0.7666  0.4862 -0.6628
    torch.FloatTensor of size (4,4)]

    >>> torch.max(a, 1)
    (
     1.2513
     0.9288
     1.0695
     0.7426
    [torch.FloatTensor of size (4,)]
    ,
     2
     0
     0
     0
    [torch.LongTensor of size (4,)]
    )

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

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size (4,)]

    >>> b = torch.randn(4)
    >>> b

     1.0067
    -0.8010
     0.6258
     0.3627
    [torch.FloatTensor of size (4,)]

    >>> torch.max(a, b)

     1.3869
     0.3912
     0.6258
     0.3627
    [torch.FloatTensor of size (4,)]

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

    -1.4550  0.8839 -1.3408
    [torch.FloatTensor of size (1,3)]

    >>> torch.mean(a)

    -0.6373
    [torch.FloatTensor of size ()]


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

    -1.2738 -0.3058  0.1230 -1.9615
     0.8771 -0.5430 -0.9233  0.9879
     1.4107  0.0317 -0.6823  0.2255
    -1.3854  0.4953 -0.2160  0.2435
    [torch.FloatTensor of size (4,4)]

    >>> torch.mean(a, 1)

    -0.8545
     0.0997
     0.2464
    -0.2157
    [torch.FloatTensor of size (4,)]

    >>> torch.mean(a, 1, True)

    -0.8545
     0.0997
     0.2464
    -0.2157
    [torch.FloatTensor of size (4,1)]

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

     0.5749 -0.2804 -0.7931
    [torch.FloatTensor of size (1,3)]

    >>> torch.median(a)

    -0.2804
    [torch.FloatTensor of size ()]


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

    >>> a

     -0.6891 -0.6662
     0.2697  0.7412
     0.5254 -0.7402
     0.5528 -0.2399
    [torch.FloatTensor of size (4,2)]

    >>> a = torch.randn(4, 5)
    >>> a

     0.4056 -0.3372  1.0973 -2.4884  0.4334
     2.1336  0.3841  0.1404 -0.1821 -0.7646
    -0.2403  1.3975 -2.0068  0.1298  0.0212
    -1.5371 -0.7257 -0.4871 -0.2359 -1.1724
    [torch.FloatTensor of size (4,5)]

    >>> torch.median(a, 1)
    (
     0.4056
     0.1404
     0.0212
    -0.7257
    [torch.FloatTensor of size (4,)]
    ,
     0
     2
     4
     1
    [torch.LongTensor of size (4,)]
    )

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

     0.4729 -0.2266 -0.2085
    [torch.FloatTensor of size (1,3)]

    >>> torch.min(a)
    -0.22663167119026184


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

    >> a = torch.randn(4, 4)
    >> a

    0.0692  0.3142  1.2513 -0.5428
    0.9288  0.8552 -0.2073  0.6409
    1.0695 -0.0101 -2.4507 -1.2230
    0.7426 -0.7666  0.4862 -0.6628
    torch.FloatTensor of size (4,4)]

    >> torch.min(a, 1)

    0.5428
    0.2073
    2.4507
    0.7666
    torch.FloatTensor of size (4,)]

    3
    2
    2
    1
    torch.LongTensor of size (4,)]

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

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size (4,)]

    >>> b = torch.randn(4)
    >>> b

     1.0067
    -0.8010
     0.6258
     0.3627
    [torch.FloatTensor of size (4,)]

    >>> torch.min(a, b)

     1.0067
    -0.8010
    -0.8634
    -0.5468
    [torch.FloatTensor of size (4,)]

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

     0.0519 -0.3304  1.2232
     4.3910 -5.1498  2.7571
    [torch.FloatTensor of size (2,3)]
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

    -0.4334
    [torch.FloatTensor of size ()]

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

    >>> a

     -0.6891 -0.6662
     0.2697  0.7412
     0.5254 -0.7402
     0.5528 -0.2399
    [torch.FloatTensor of size (4,2)]

    >>> a = torch.randn(4, 5)
    >>> a

     0.4056 -0.3372  1.0973 -2.4884  0.4334
     2.1336  0.3841  0.1404 -0.1821 -0.7646
    -0.2403  1.3975 -2.0068  0.1298  0.0212
    -1.5371 -0.7257 -0.4871 -0.2359 -1.1724
    [torch.FloatTensor of size (4,5)]

    >>> torch.mode(a, 1)
    (
    -2.4884
    -0.7646
    -2.0068
    -1.5371
    [torch.FloatTensor of size (4,)]
    ,
     3
     4
     2
     0
    [torch.LongTensor of size (4,)]
    )

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

    -0.9374
    -0.5254
    -0.6069
    [torch.FloatTensor of size (3,)]

    >>> torch.mul(a, 100)

    -93.7411
    -52.5374
    -60.6908
    [torch.FloatTensor of size (3,)]


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

    >>> a = torch.randn(4, 4)
    >>> a

    -0.7280  0.0598 -1.4327 -0.5825
    -0.1427 -0.0690  0.0821 -0.3270
    -0.9241  0.5110  0.4070 -1.1188
    -0.8308  0.7426 -0.6240 -1.1582
    [torch.FloatTensor of size (4,4)]

    >>> b = torch.randn(2, 8)
    >>> b

     0.0430 -1.0775  0.6015  1.1647 -0.6549  0.0308 -0.1670  1.0742
    -1.2593  0.0292 -0.0849  0.4530  1.2404 -0.4659 -0.1840  0.5974
    [torch.FloatTensor of size (2,8)]

    >>> torch.mul(a, b)

    -0.0313 -0.0645 -0.8618 -0.6784
     0.0934 -0.0021 -0.0137 -0.3513
     1.1638  0.0149 -0.0346 -0.5068
    -1.0304 -0.3460  0.1148 -0.6919
    [torch.FloatTensor of size (4,4)]

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

     1
     2
     0
     0
    [torch.LongTensor of size (4,)]

    >>> torch.multinomial(weights, 4, replacement=True)

     1
     2
     1
     2
    [torch.LongTensor of size (4,)]

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

    -2.0939
    -2.2950
    [torch.FloatTensor of size (2,)]

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

     0  1
     1  0
    [torch.ByteTensor of size (2,2)]

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

    -0.4430
     1.1690
    -0.8836
    -0.4565
     0.2968
    [torch.FloatTensor of size (5,)]

    >>> torch.neg(a)

     0.4430
    -1.1690
     0.8836
     0.4565
    -0.2968
    [torch.FloatTensor of size (5,)]

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

     0
     1
     2
     4
    [torch.LongTensor of size (4,1)]

    >>> torch.nonzero(torch.Tensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]]))

     0  0
     1  1
     2  2
     3  3
    [torch.LongTensor of size (4,2)]

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

     0.1628  0.1210 -0.9801
    [torch.FloatTensor of size (1,3)]

    >>> torch.norm(a, 3)

     0.9822
    [torch.FloatTensor of size ()]



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

    -0.6891 -0.6662
     0.2697  0.7412
     0.5254 -0.7402
     0.5528 -0.2399
    [torch.FloatTensor of size (4,2)]

    >>> torch.norm(a, 2, 1)

     0.9585
     0.7888
     0.9077
     0.6026
    [torch.FloatTensor of size (4,)]

    >>> torch.norm(a, 0, 1, True)

     2
     2
     2
     2
    [torch.FloatTensor of size (4,1)]

""")

add_docstr(torch.normal,
           r"""
.. function:: normal(means, std, out=None) -> Tensor

Returns a tensor of random numbers drawn from separate normal distributions
whose mean and standard deviation are given.

The :attr:`means` is a tensor with the mean of
each output element's normal distribution

The :attr:`std` is a tensor with the standard deviation of
each output element's normal distribution

The shapes of :attr:`means` and :attr:`std` don't need to match, but the
total number of elements in each tensor need to be the same.

.. note:: When the shapes do not match, the shape of :attr:`means`
          is used as the shape for the returned output tensor

Args:
    means (Tensor): the tensor of per-element means
    std (Tensor): the tensor of per-element standard deviations
    out (Tensor, optional): the output tensor

Example::

    >>> torch.normal(means=torch.arange(1, 11), std=torch.arange(1, 0, -0.1))

     1.5104
     1.6955
     2.4895
     4.9185
     4.9895
     6.9155
     7.3683
     8.1836
     8.7164
     9.8916
    [torch.FloatTensor of size (10,)]

.. function:: normal(mean=0.0, std, out=None) -> Tensor

Similar to the function above, but the means are shared among all drawn
elements.

Args:
    means (float, optional): the mean for all distributions
    std (Tensor): the tensor of per-element standard deviations
    out (Tensor, optional): the output tensor

Example::

    >>> torch.normal(mean=0.5, std=torch.arange(1, 6))

      0.5723
      0.0871
     -0.3783
     -2.5689
     10.7893
    [torch.FloatTensor of size (5,)]

.. function:: normal(means, std=1.0, out=None) -> Tensor

Similar to the function above, but the standard-deviations are shared among
all drawn elements.

Args:
    means (Tensor): the tensor of per-element means
    std (float, optional): the standard deviation for all distributions
    out (Tensor, optional): the output tensor

Example::

    >>> torch.normal(means=torch.arange(1, 6))

     1.1681
     2.8884
     3.7718
     2.5616
     4.2500
    [torch.FloatTensor of size (5,)]

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
ones(*sizes, out=None) -> Tensor

Returns a tensor filled with the scalar value `1`, with the shape defined
by the variable argument :attr:`sizes`.

Args:
    sizes (int...): a set of integers defining the shape of the output tensor
    out (Tensor, optional): the output tensor

Example::

    >>> torch.ones(2, 3)

     1  1  1
     1  1  1
    [torch.FloatTensor of size (2,3)]

    >>> torch.ones(5)

     1
     1
     1
     1
     1
    [torch.FloatTensor of size (5,)]

""")

add_docstr(torch.ones_like,
           r"""
ones_like(input, out=None) -> Tensor

Returns a tensor filled with the scalar value `1`, with the same size as
:attr:`input`.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor
    out (Tensor, optional): the output tensor

Example::

    >>> input = torch.FloatTensor(2, 3)
    >>> torch.ones_like(input)

     1  1  1
     1  1  1
    [torch.FloatTensor of size (2,3)]
""")

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

     2.3563  3.2318 -0.9406
     3.2318  4.9557 -2.1618
    -0.9406 -2.1618  2.2443
    [torch.FloatTensor of size (3,3)]

    >>> u

     1.5350  2.1054 -0.6127
     0.0000  0.7233 -1.2053
     0.0000  0.0000  0.6451
    [torch.FloatTensor of size (3,3)]

    >>> torch.mm(u.t(), u)

     2.3563  3.2318 -0.9406
     3.2318  4.9557 -2.1618
    -0.9406 -2.1618  2.2443
    [torch.FloatTensor of size (3,3)]

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

     2.3563  3.2318 -0.9406
     3.2318  4.9557 -2.1618
    -0.9406 -2.1618  2.2443
    [torch.FloatTensor of size (3,3)]

    >>> torch.potri(u)

     12.5724 -10.1765  -4.5333
    -10.1765   8.5852   4.0047
     -4.5333   4.0047   2.4031
    [torch.FloatTensor of size (3,3)]

    >>> a.inverse()

     12.5723 -10.1765  -4.5333
    -10.1765   8.5852   4.0047
     -4.5333   4.0047   2.4031
    [torch.FloatTensor of size (3,3)]

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

     2.3563  3.2318 -0.9406
     3.2318  4.9557 -2.1618
    -0.9406 -2.1618  2.2443
    [torch.FloatTensor of size (3,3)]

    >>> b = torch.randn(3, 2)
    >>> b

    -0.3119 -1.8224
    -0.2798  0.1789
    -0.3735  1.7451
    [torch.FloatTensor of size (3,2)]

    >>> torch.potrs(b,u)

     0.6187 -32.6438
    -0.7234  27.0703
    -0.6039  13.1717
    [torch.FloatTensor of size (3,2)]

    >>> torch.mm(a.inverse(),b)

     0.6187 -32.6436
    -0.7234  27.0702
    -0.6039  13.1717
    [torch.FloatTensor of size (3,2)]

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

    -0.5274
    -0.8232
    -2.1128
     1.7558
    [torch.FloatTensor of size (4,)]

    >>> torch.pow(a, 2)

     0.2781
     0.6776
     4.4640
     3.0829
    [torch.FloatTensor of size (4,)]

    >>> exp = torch.arange(1, 5)
    >>> a = torch.arange(1, 5)
    >>> a

     1
     2
     3
     4
    [torch.FloatTensor of size (4,)]

    >>> exp

     1
     2
     3
     4
    [torch.FloatTensor of size (4,)]

    >>> torch.pow(a, exp)

       1
       4
      27
     256
    [torch.FloatTensor of size (4,)]


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

      2
      4
      8
     16
    [torch.FloatTensor of size (4,)]

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

     0.7624 -0.4892 -0.1841
    [torch.FloatTensor of size (1,3)]

    >>> torch.prod(a)

    1.00000e-02 *
      6.8676
    [torch.FloatTensor of size ()]


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

     0.1598 -0.6884
    -0.1831 -0.4412
    -0.9925 -0.6244
    -0.2416 -0.8080
    [torch.FloatTensor of size (4,2)]

    >>> torch.prod(a, 1)

    -0.1100
     0.0808
     0.6197
     0.1952
    [torch.FloatTensor of size (4,)]

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

     5.4417 -2.5280  1.3643
    -2.5280  2.9689 -2.1368
     1.3643 -2.1368  4.6116
    [torch.FloatTensor of size (3,3)]

    >>> u,piv = torch.pstrf(a)
    >>> u

     2.3328  0.5848 -1.0837
     0.0000  2.0663 -0.7274
     0.0000  0.0000  1.1249
    [torch.FloatTensor of size (3,3)]

    >>> piv

     0
     2
     1
    [torch.IntTensor of size (3,)]

    >>> p = torch.eye(3).index_select(0,piv.long()).index_select(0,piv.long()).t() # make pivot permutation
    >>> torch.mm(torch.mm(p.t(),torch.mm(u.t(),u)),p) # reconstruct

     5.4417  1.3643 -2.5280
     1.3643  4.6116 -2.1368
    -2.5280 -2.1368  2.9689
    [torch.FloatTensor of size (3,3)]

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

    -0.8571  0.3943  0.3314
    -0.4286 -0.9029 -0.0343
     0.2857 -0.1714  0.9429
    [torch.FloatTensor of size (3,3)]

    >>> r

     -14.0000  -21.0000   14.0000
       0.0000 -175.0000   70.0000
       0.0000    0.0000  -35.0000
    [torch.FloatTensor of size (3,3)]

    >>> torch.mm(q, r).round()

      12  -51    4
       6  167  -68
      -4   24  -41
    [torch.FloatTensor of size (3,3)]

    >>> torch.mm(q.t(), q).round()

     1 -0  0
    -0  1  0
     0  0  1
    [torch.FloatTensor of size (3,3)]

""")

add_docstr(torch.rand,
           r"""
rand(*sizes, out=None) -> Tensor

Returns a tensor filled with random numbers from a uniform distribution
on the interval :math:`[0, 1)`

The shape of the tensor is defined by the variable argument :attr:`sizes`.

Args:
    sizes (int...): a set of ints defining the shape of the output tensor.
    out (Tensor, optional): the output tensor

Example::

    >>> torch.rand(4)

     0.9193
     0.3347
     0.3232
     0.7715
    [torch.FloatTensor of size (4,)]

    >>> torch.rand(2, 3)

     0.5010  0.5140  0.0719
     0.1435  0.5636  0.0538
    [torch.FloatTensor of size (2,3)]

""")

add_docstr(torch.randint,
           r"""
randint(low=0, high, sizes, out=None, dtype=torch.float32) -> Tensor

Returns a tensor filled with random integers generated uniformly
between :attr:`low` (inclusive) and :attr:`high` (exclusive).

The shape of the tensor is defined by the variable argument :attr:`sizes`.

Args:
    low (int, optional): Lowest integer to be drawn from the distribution. Default: 0.
    high (int): One above the highest integer to be drawn from the distribution.
    sizes (tuple): a tuple defining the shape of the output tensor.
    out (Tensor, optional): the output tensor
    dtype (:class:`torch.dtype`, optional): the desired type of returned Tensor. Default: torch.float32

Example::

    >>> torch.randint(3, 5, (3,))

     4
     4
     3
    [torch.FloatTensor of size (3,)]

    >>> torch.randint(3, 10, (2,2), dtype=torch.long)

     7  5
     9  4
    [torch.LongTensor of size (2,2)]

    >>> torch.randint(3, 10, (2,2))

     6  8
     9  4
    [torch.FloatTensor of size (2,2)]

""")

add_docstr(torch.randn,
           r"""
randn(*sizes, out=None) -> Tensor

Returns a tensor filled with random numbers from a normal distribution
with zero mean and variance of one (also called the standard normal
distribution).

.. math::
    \text{out}_{i} \sim \mathcal{N}(0, 1)

The shape of the tensor is defined by the variable argument :attr:`sizes`.

Args:
    sizes (int...): a set of ints defining the shape of the output tensor.
    out (Tensor, optional): the output tensor

Example::

    >>> torch.randn(4)

    -0.1145
     0.0094
    -1.1717
     0.9846
    [torch.FloatTensor of size (4,)]

    >>> torch.randn(2, 3)

     1.4339  0.3351 -1.0999
     1.5458 -0.9643 -0.3558
    [torch.FloatTensor of size (2,3)]

""")

add_docstr(torch.randperm,
           r"""
randperm(n, out=None) -> LongTensor

Returns a random permutation of integers from ``0`` to ``n - 1``.

Args:
    n (int): the upper bound (exclusive)
    out (Tensor, optional): the output tensor

Example::

    >>> torch.randperm(4)

     2
     1
     3
     0
    [torch.LongTensor of size (4,)]
""")

add_docstr(torch.range,
           r"""
range(start, end, step=1, out=None) -> Tensor

Returns a 1-D tensor of size :math:`\left\lfloor \frac{end - start}{step} \right\rfloor + 1`
with values from :attr:`start` to :attr:`end` with step :attr:`step`. Step is
the gap between two values in the tensor.

.. math::
    \text{out}_{i+1} = \text{out}_i + step.

.. warning::
    This function is deprecated in favor of :func:`torch.arange`.

Args:
    start (float): the starting value for the set of points
    end (float): the ending value for the set of points
    step (float): the gap between each pair of adjacent points
    out (Tensor, optional): the output tensor

Example::

    >>> torch.range(1, 4)

     1
     2
     3
     4
    [torch.FloatTensor of size (4,)]

    >>> torch.range(1, 4, 0.5)

     1.0000
     1.5000
     2.0000
     2.5000
     3.0000
     3.5000
     4.0000
    [torch.FloatTensor of size (7,)]

""")

add_docstr(torch.arange,
           r"""
arange(start=0, end, step=1, out=None) -> Tensor

Returns a 1-D tensor of size :math:`\left\lfloor \frac{end - start}{step} \right\rfloor`
with values from the interval ``[start, end)`` taken with common difference
:attr:`step` beginning from `start`.

Note that non-integer `step` is subject to floating point rounding errors when
comparing against `end`; to avoid inconsistency, we advise adding a small epsilon to `end`
in such cases.

.. math::
    \text{out}_{i+1} = \text{out}_{i} + \text{step}

Args:
    start (float): the starting value for the set of points
    end (float): the ending value for the set of points
    step (float): the gap between each pair of adjacent points
    out (Tensor, optional): the output tensor

Example::

    >>> torch.arange(5)

     0
     1
     2
     3
     4
    [torch.FloatTensor of size (5,)]

    >>> torch.arange(1, 4)

     1
     2
     3
    [torch.FloatTensor of size (3,)]

    >>> torch.arange(1, 2.5, 0.5)

     1.0000
     1.5000
     2.0000
    [torch.FloatTensor of size (3,)]

""")


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

     1
     0
     1
     1
     0
     1
    [torch.FloatTensor of size (6,)]

    >>> torch.remainder(torch.Tensor([1, 2, 3, 4, 5]), 1.5)

     1.0000
     0.5000
     0.0000
     1.0000
     0.5000
    [torch.FloatTensor of size (5,)]

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
    >>> x[2].fill_(3)
    >>> x

     1  1  1
     2  2  2
     3  3  3
    [torch.FloatTensor of size (3,3)]

    >>> torch.renorm(x, 1, 0, 5)

     1.0000  1.0000  1.0000
     1.6667  1.6667  1.6667
     1.6667  1.6667  1.6667
    [torch.FloatTensor of size (3,3)]

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
     0  1
     2  3
    [torch.FloatTensor of size (2,2)]

    >>> b = torch.tensor([[0, 1], [2, 3]])
    >>> torch.reshape(b, (-1,))
     0
     1
     2
     3
    [torch.FloatTensor of size (4,)]
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

     1.2290
     1.3409
    -0.5662
    -0.0899
    [torch.FloatTensor of size (4,)]

    >>> torch.round(a)

     1
     1
    -1
    -0
    [torch.FloatTensor of size (4,)]

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

     1.2290
     1.3409
    -0.5662
    -0.0899
    [torch.FloatTensor of size (4,)]

    >>> torch.rsqrt(a)

     0.9020
     0.8636
        nan
        nan
    [torch.FloatTensor of size (4,)]

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

     0
    [torch.DoubleTensor of size (1,)]

    >>> torch.set_flush_denormal(False)
    True
    >>> torch.DoubleTensor([1e-323])

    9.88131e-324 *
      1.0000
    [torch.DoubleTensor of size (1,)]

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

    -0.4972
     1.3512
     0.1056
    -0.2650
    [torch.FloatTensor of size (4,)]

    >>> torch.sigmoid(a)

     0.3782
     0.7943
     0.5264
     0.4341
    [torch.FloatTensor of size (4,)]

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

    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.sign(a)

    -1
     1
     1
     1
    [torch.FloatTensor of size (4,)]

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

    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.sin(a)

    -0.5944
     0.2684
     0.4322
     0.9667
    [torch.FloatTensor of size (4,)]
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

    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.sinh(a)

    -0.6804
     0.2751
     0.4619
     1.7225
    [torch.FloatTensor of size (4,)]
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

    -1.6747  0.0610  0.1190  1.4137
    -1.4782  0.7159  1.0341  1.3678
    -0.3324 -0.0782  0.3518  0.4763
    [torch.FloatTensor of size (3,4)]

    >>> indices

     0  1  3  2
     2  1  0  3
     3  1  0  2
    [torch.LongTensor of size (3,4)]

    >>> sorted, indices = torch.sort(x, 0)
    >>> sorted

    -1.6747 -0.0782 -1.4782 -0.3324
     0.3518  0.0610  0.4763  0.1190
     1.0341  0.7159  1.4137  1.3678
    [torch.FloatTensor of size (3,4)]

    >>> indices

     0  2  1  2
     2  0  2  0
     1  1  0  1
    [torch.LongTensor of size (3,4)]

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

     1.2290
     1.3409
    -0.5662
    -0.0899
    [torch.FloatTensor of size (4,)]

    >>> torch.sqrt(a)

     1.1086
     1.1580
        nan
        nan
    [torch.FloatTensor of size (4,)]

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

     0.1665  0.4876 -0.2155
    [torch.FloatTensor of size (1,3)]

    >>> torch.std(a)

     0.3520
    [torch.FloatTensor of size ()]



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

     0.1889 -2.4856  0.0043  1.8169
    -0.7701 -0.4682 -2.2410  0.4098
     0.1919 -1.1856 -1.0361  0.9085
     0.0173  1.0662  0.2143 -0.5576
    [torch.FloatTensor of size (4,4)]

    >>> torch.std(a, dim=1)

     1.7756
     1.1025
     1.0045
     0.6725
    [torch.FloatTensor of size (4,)]

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

    -0.0281  1.0131 -0.0384
    [torch.FloatTensor of size (1,3)]

    >>> torch.sum(a)

     0.9466
    [torch.FloatTensor of size ()]


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

    -0.4640  0.0609  0.1122  0.4784
    -1.3063  1.6443  0.4714 -0.7396
    -1.3561 -0.1959  1.0609 -1.9855
     2.6833  0.5746 -0.5709 -0.4430
    [torch.FloatTensor of size (4,4)]

    >>> torch.sum(a, 1)

     0.1874
     0.0698
    -2.4767
     2.2440
    [torch.FloatTensor of size (4,)]

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

    -0.5911  0.2632  0.3554  0.3143  0.2299
    -0.3976  0.2438 -0.2224 -0.7535 -0.3636
    -0.0335 -0.6003 -0.4508  0.2334 -0.3055
    -0.4297  0.2362 -0.6859  0.3319  0.1649
    -0.4697 -0.3509  0.3874  0.1587 -0.5183
     0.2934  0.5763 -0.0209  0.3791 -0.6526
    [torch.FloatTensor of size (6,5)]

    >>> s

     27.4687
     22.6432
      8.5584
      5.9857
      2.0149
    [torch.FloatTensor of size (5,)]

    >>> v

    -0.2514  0.8148 -0.2606  0.3967 -0.2180
    -0.3968  0.3587  0.7008 -0.4507  0.1402
    -0.6922 -0.2489 -0.2208  0.2513  0.5891
    -0.3662 -0.3686  0.3859  0.4342 -0.6265
    -0.4076 -0.0980 -0.4932 -0.6227 -0.4396
    [torch.FloatTensor of size (5,5)]

    >>> torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t()))

    1.00000e-05 *
      1.0918
    [torch.FloatTensor of size ()]

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

    -11.0656
     -6.2287
      0.8640
      8.8655
     16.0948
    [torch.FloatTensor of size (5,)]

    >>> v

    -0.2981 -0.6075  0.4026 -0.3745  0.4896
    -0.5078 -0.2880 -0.4066 -0.3572 -0.6053
    -0.0816 -0.3843 -0.6600  0.5008  0.3991
    -0.0036 -0.4467  0.4553  0.6204 -0.4564
    -0.8041  0.4480  0.1725  0.3108  0.1622
    [torch.FloatTensor of size (5,5)]

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

     0.4834  0.6907  1.3417
    -0.1300  0.5295  0.2321
    [torch.FloatTensor of size (2,3)]

    >>> torch.t(x)

     0.4834 -0.1300
     0.6907  0.5295
     1.3417  0.2321
    [torch.FloatTensor of size (3,2)]

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
    >>> torch.take(src, torch.LongTensor([0, 2, 5]))

     4
     5
     8
    [torch.FloatTensor of size (3,)]

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

    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.tan(a)

    -0.7392
     0.2786
     0.4792
     3.7801
    [torch.FloatTensor of size (4,)]

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

    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size (4,)]

    >>> torch.tanh(a)

    -0.5625
     0.2653
     0.4193
     0.8648
    [torch.FloatTensor of size (4,)]

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

     1
     2
     3
     4
     5
    [torch.FloatTensor of size (5,)]

    >>> torch.topk(x, 3)
    (
     5
     4
     3
    [torch.FloatTensor of size (3,)]
    ,
     4
     3
     2
    [torch.LongTensor of size (3,)]
    )
    >>> torch.topk(x, 3, 0, largest=False)
    (
     1
     2
     3
    [torch.FloatTensor of size (3,)]
    ,
     0
     1
     2
    [torch.LongTensor of size (3,)]
    )

""")

add_docstr(torch.trace,
           r"""
trace(input) -> Tensor

Returns the sum of the elements of the diagonal of the input 2-D matrix.

Example::

    >>> x = torch.arange(1, 10).view(3, 3)
    >>> x

     1  2  3
     4  5  6
     7  8  9
    [torch.FloatTensor of size (3,3)]

    >>> torch.trace(x)

     15
    [torch.FloatTensor of size ()]


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

     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size (2,3)]

    >>> torch.transpose(x, 0, 1)

     0.5983  1.5981
    -0.0341 -0.5265
     2.4918 -0.8735
    [torch.FloatTensor of size (3,2)]

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

     1.3225  1.7304  1.4573
    -0.3052 -0.3111 -0.1809
     1.2469  0.0064 -1.6250
    [torch.FloatTensor of size (3,3)]

    >>> torch.tril(a)

     1.3225  0.0000  0.0000
    -0.3052 -0.3111  0.0000
     1.2469  0.0064 -1.6250
    [torch.FloatTensor of size (3,3)]

    >>> b = torch.randn(4, 6)
    >>> b

     0.2762  0.1640  0.3947 -0.8633 -0.4150  2.4491
    -2.8177 -1.0580  0.3659 -0.0797  0.2294  1.3660
    -1.8665 -0.4127 -0.7031 -0.4697 -0.2383 -0.1321
     1.0998  0.2726  0.2512  0.4557  0.7012 -0.9356
    [torch.FloatTensor of size (4,6)]

    >>> torch.tril(b, diagonal=1)

     0.2762  0.1640  0.0000  0.0000  0.0000  0.0000
    -2.8177 -1.0580  0.3659  0.0000  0.0000  0.0000
    -1.8665 -0.4127 -0.7031 -0.4697  0.0000  0.0000
     1.0998  0.2726  0.2512  0.4557  0.7012  0.0000
    [torch.FloatTensor of size (4,6)]

    >>> torch.tril(b, diagonal=-1)

     0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
    -2.8177  0.0000  0.0000  0.0000  0.0000  0.0000
    -1.8665 -0.4127  0.0000  0.0000  0.0000  0.0000
     1.0998  0.2726  0.2512  0.0000  0.0000  0.0000
    [torch.FloatTensor of size (4,6)]

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

     1.3225  1.7304  1.4573
    -0.3052 -0.3111 -0.1809
     1.2469  0.0064 -1.6250
    [torch.FloatTensor of size (3,3)]

    >>> torch.triu(a)

     1.3225  1.7304  1.4573
     0.0000 -0.3111 -0.1809
     0.0000  0.0000 -1.6250
    [torch.FloatTensor of size (3,3)]

    >>> torch.triu(a, diagonal=1)

     0.0000  1.7304  1.4573
     0.0000  0.0000 -0.1809
     0.0000  0.0000  0.0000
    [torch.FloatTensor of size (3,3)]

    >>> torch.triu(a, diagonal=-1)

     1.3225  1.7304  1.4573
    -0.3052 -0.3111 -0.1809
     0.0000  0.0064 -1.6250
    [torch.FloatTensor of size (3,3)]

    >>> b = torch.randn(4, 6)
    >>> b

     0.2762  0.1640  0.3947 -0.8633 -0.4150  2.4491
    -2.8177 -1.0580  0.3659 -0.0797  0.2294  1.3660
    -1.8665 -0.4127 -0.7031 -0.4697 -0.2383 -0.1321
     1.0998  0.2726  0.2512  0.4557  0.7012 -0.9356
    [torch.FloatTensor of size (4,6)]

    >>> torch.tril(b, diagonal=1)

     0.0000  0.1640  0.3947 -0.8633 -0.4150  2.4491
     0.0000  0.0000  0.3659 -0.0797  0.2294  1.3660
     0.0000  0.0000  0.0000 -0.4697 -0.2383 -0.1321
     0.0000  0.0000  0.0000  0.0000  0.7012 -0.9356
    [torch.FloatTensor of size (4,6)]

    >>> torch.tril(a, diagonal=-1)

     0.2762  0.1640  0.3947 -0.8633 -0.4150  2.4491
    -2.8177 -1.0580  0.3659 -0.0797  0.2294  1.3660
     0.0000 -0.4127 -0.7031 -0.4697 -0.2383 -0.1321
     0.0000  0.0000  0.2512  0.4557  0.7012 -0.9356
    [torch.FloatTensor of size (4,6)]

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

    -1.8793  0.1567
     0.0000 -2.1972
    [torch.FloatTensor of size (2,2)]

    >>> b = torch.randn(2, 3)
    >>> b

     1.8776 -0.0759  1.6590
    -0.5676  0.4771  0.7477
    [torch.FloatTensor of size (2,3)]

    >>> torch.trtrs(b, A)
    (
    -0.9775  0.0223 -0.9112
     0.2583 -0.2172 -0.3403
    [torch.FloatTensor of size (2,3)]
    ,
    -1.8793  0.1567
     0.0000 -2.1972
    [torch.FloatTensor of size (2,2)]
    )

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

    -0.4972
     1.3512
     0.1056
    -0.2650
    [torch.FloatTensor of size (4,)]

    >>> torch.trunc(a)

    -0
     1
     0
    -0
    [torch.FloatTensor of size (4,)]

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

     1  2  3  4
    [torch.FloatTensor of size (1,4)]

    >>> torch.unsqueeze(x, 1)

     1
     2
     3
     4
    [torch.FloatTensor of size (4,1)]

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

     1.4529 -0.0128  0.6240
    [torch.FloatTensor of size (1,3)]

    >>> torch.var(a)

     0.5401
    [torch.FloatTensor of size ()]


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

    -1.2738 -0.3058  0.1230 -1.9615
     0.8771 -0.5430 -0.9233  0.9879
     1.4107  0.0317 -0.6823  0.2255
    -1.3854  0.4953 -0.2160  0.2435
    [torch.FloatTensor of size (4,4)]

    >>> torch.var(a, 1)

     0.8859
     0.9509
     0.7548
     0.6949
    [torch.FloatTensor of size (4,)]

""")

add_docstr(torch.zeros,
           r"""
zeros(*sizes, out=None) -> Tensor

Returns a tensor filled with the scalar value `0`, with the shape defined
by the variable argument :attr:`sizes`.

Args:
    sizes (int...): a set of integers defining the shape of the output tensor
    out (Tensor, optional): the output tensor

Example::

    >>> torch.zeros(2, 3)

     0  0  0
     0  0  0
    [torch.FloatTensor of size (2,3)]

    >>> torch.zeros(5)

     0
     0
     0
     0
     0
    [torch.FloatTensor of size (5,)]

""")

add_docstr(torch.zeros_like,
           r"""
zeros_like(input, out=None) -> Tensor

Returns a tensor filled with the scalar value `0`, with the same size as
:attr:`input`.

Args:
    input (Tensor): the size of the input will determine the size of the output.
    out (Tensor, optional): the output tensor

Example::

    >>> input = torch.FloatTensor(2, 3)
    >>> torch.zeros_like(input)

     0  0  0
     0  0  0
    [torch.FloatTensor of size (2,3)]

""")

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

    1.00000e-08 *
      7.1293
    [torch.FloatTensor of size ()]

""")

add_docstr(torch.empty_like,
           r"""
empty_like(input) -> Tensor

Returns an uninitialized tensor with the same size as :attr:`input`.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor

Example::

    >>> input = torch.LongTensor(2,3)
    >>> input.new(input.size())

    1.3996e+14  1.3996e+14  1.3996e+14
    4.0000e+00  0.0000e+00  0.0000e+00
    [torch.LongTensor of size (2,3)]

""")

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

    0.3690
    [torch.FloatTensor of size ()]

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

    -2.2068  1.2589
    -0.9796 -0.7586
    -0.5561  0.5734
    [torch.FloatTensor of size (3,2)]

    >>> torch.where(x > 0, x, y)

     1.0000  1.2589
     1.0000  1.0000
     1.0000  0.5734
    [torch.FloatTensor of size (3,2)]

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

    1.9386
    [torch.FloatTensor of size ()]

    >>> torch.logdet(A)

    0.6620
    [torch.FloatTensor of size ()]

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

    -0.3534
    [torch.FloatTensor of size ()]

    >>> torch.logdet(A)

    nan
    [torch.FloatTensor of size ()]

    >>> torch.slogdet(A)
    (
    -1
    [torch.FloatTensor of size ()]
    ,
    -1.0402
    [torch.FloatTensor of size ()]
    )

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

    (0 ,.,.) =
      6.8901 -1.7571
      0.4166 -1.1500
      3.9736  0.5400

    (1 ,.,.) =
     -0.3050  0.4976
      0.2072  1.1015
      1.3850 -0.0566

    (2 ,.,.) =
      3.1463  3.3727
      1.4051 -3.3523
      1.5072 -4.1700

    (3 ,.,.) =
      5.1215 -2.5402
      5.1859  1.4077
      2.0077  1.3137
    [torch.FloatTensor of size (4,3,2)]

    >>> # batched 1D FFT
    >>> torch.fft(x, 1)

    (0 ,.,.) =
      3.7132 -0.1067
      1.8037 -0.4983
      2.2184 -0.5932

    (1 ,.,.) =
      0.1765 -2.6391
     -0.1705 -0.6941
      0.9592  1.0218

    (2 ,.,.) =
      1.3050  0.9145
     -0.8929 -1.7529
      0.5220 -1.2218

    (3 ,.,.) =
      1.6954  0.0742
     -0.3237  1.7953
      0.2740  1.3332
    [torch.FloatTensor of size (4,3,2)]

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

    (0 ,.,.) =
      1.2735 -0.9441
     -1.0940  0.2728
      0.8997  0.4231

    (1 ,.,.) =
     -0.5239 -1.4942
      0.5248  3.3432
      1.0976 -2.0426

    (2 ,.,.) =
      1.1039  1.9541
     -0.2774  0.2631
      0.3102  0.8129
    [torch.FloatTensor of size (3,3,2)]

    >>> y = torch.fft(x, 2)
    >>> torch.ifft(y, 2)  # recover x

    (0 ,.,.) =
      1.2735 -0.9441
     -1.0940  0.2728
      0.8997  0.4231

    (1 ,.,.) =
     -0.5239 -1.4942
      0.5248  3.3432
      1.0976 -2.0426

    (2 ,.,.) =
      1.1039  1.9541
     -0.2774  0.2631
      0.3102  0.8129
    [torch.FloatTensor of size (3,3,2)]

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
    >>> x

    -0.5052 -0.0420  0.5773 -0.2224  1.8413
    -0.0873 -0.7571 -0.1982  0.5722 -1.8076
    -1.5447  0.4344  0.8692  1.7930  2.6886
     0.4674  0.0517 -0.7564 -0.5118 -0.7023
    [torch.FloatTensor of size (4,5)]

    >>> y = torch.rfft(x, 2, onesided=True)
    >>> torch.irfft(y, 2, onesided=True, signal_sizes=x.shape)  # recover x

    -0.5052 -0.0420  0.5773 -0.2224  1.8413
    -0.0873 -0.7571 -0.1982  0.5722 -1.8076
    -1.5447  0.4344  0.8692  1.7930  2.6886
     0.4674  0.0517 -0.7564 -0.5118 -0.7023
    [torch.FloatTensor of size (4,5)]

""")
