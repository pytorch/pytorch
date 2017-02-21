"""Adds docstrings to functions defined in the torch._C"""

import torch._C
from torch._C import _add_docstr as add_docstr

add_docstr(torch._C.abs,
           """abs(input, out=None) -> Tensor

Computes the element-wise absolute value of the given :attr:`input` a tensor.

Example::

    >>> torch.abs(torch.FloatTensor([-1, -2, 3]))
    FloatTensor([1, 2, 3])
""")

add_docstr(torch._C.acos,
           """
acos(input, out=None) -> Tensor

Returns a new `Tensor` with the arccosine  of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.acos(a)
     2.2608
     1.2956
     1.1075
        nan
    [torch.FloatTensor of size 4]
""")

add_docstr(torch._C.add,
           """
.. function:: add(input, value, out=None)

Adds the scalar :attr:`value` to each element of the input :attr:`input`
and returns a new resulting tensor.

:math:`out = tensor + value`

Args:
    input (Tensor): the input `Tensor`
    value (float): the number to be added to each element of :attr:`input`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     0.4050
    -1.2227
     1.8688
    -0.4185
    [torch.FloatTensor of size 4]

    >>> torch.add(a, 20)

     20.4050
     18.7773
     21.8688
     19.5815
    [torch.FloatTensor of size 4]


.. function:: add(input, value=1, other, out=None)

Each element of the Tensor :attr:`other` is multiplied by the scalar
:attr:`value` and added to each element of the Tensor :attr:`input`.
The resulting Tensor is returned.

The shapes of :attr:`input` and :attr:`other` don't need to match.
The total number of elements in each Tensor need to be the same.

.. note:: When the shapes do not match, the shape of :attr:`input`
          is used as the shape for the returned output Tensor

:math:`out = input + (other * value)`

Args:
    input (Tensor): the first input `Tensor`
    value (float): the scalar multiplier for :attr:`other`
    other (Tensor): the second input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> import torch
    >>> a = torch.randn(4)
    >>> a

    -0.9310
     2.0330
     0.0852
    -0.2941
    [torch.FloatTensor of size 4]

    >>> b = torch.randn(2, 2)
    >>> b

     1.0663  0.2544
    -0.1513  0.0749
    [torch.FloatTensor of size 2x2]

    >>> torch.add(a, 10, b)
     9.7322
     4.5770
    -1.4279
     0.4552
    [torch.FloatTensor of size 4]


""")

add_docstr(torch._C.addbmm,
           """
addbmm(beta=1, mat, alpha=1, batch1, batch2, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored
in :attr:`batch1` and :attr:`batch2`,
with a reduced add step (all matrix multiplications get accumulated
along the first dimension).
:attr:`mat` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3D Tensors each containing the
same number of matrices.

If :attr:`batch1` is a `b x n x m` Tensor, :attr:`batch2` is a `b x m x p`
Tensor, :attr:`out` and :attr:`mat` will be `n x p` Tensors.

In other words,
:math:`res = (beta * M) + (alpha * sum(batch1_i @ batch2_i, i = 0, b))`

Args:
    beta (float, optional): multiplier for :attr:`mat`
    mat (Tensor): matrix to be added
    alpha (float, optional): multiplier for `batch1 @ batch2`
    batch1 (Tensor): First batch of matrices to be multiplied
    batch2 (Tensor): Second batch of matrices to be multiplied
    out (Tensor, optional): Output tensor

Example::

    >>> M = torch.randn(3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.addbmm(M, batch1, batch2)

     -3.1162  11.0071   7.3102   0.1824  -7.6892
      1.8265   6.0739   0.4589  -0.5641  -5.4283
     -9.3387  -0.1794  -1.2318  -6.8841  -4.7239
    [torch.FloatTensor of size 3x5]
""")

add_docstr(torch._C.addcdiv,
           """
addcdiv(tensor, value=1, tensor1, tensor2, out=None) -> Tensor

Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
multiply the result by the scalar :attr:`value` and add it to :attr:`tensor`.

The number of elements must match, but sizes do not matter.

Args:
    tensor (Tensor): the tensor to be added
    value (float, optional): multiplier for `tensor1 ./ tensor2`
    tensor1 (Tensor): Numerator tensor
    tensor2 (Tensor): Denominator tensor
    out (Tensor, optional): Output tensor

Example::

    >>> t = torch.randn(2, 3)
    >>> t1 = torch.randn(1, 6)
    >>> t2 = torch.randn(6, 1)
    >>> torch.addcdiv(t, 0.1, t1, t2)

     0.0122 -0.0188 -0.2354
     0.7396 -1.5721  1.2878
    [torch.FloatTensor of size 2x3]
""")

add_docstr(torch._C.addcmul,
           """
addcmul(tensor, value=1, tensor1, tensor2, out=None) -> Tensor

Performs the element-wise multiplication of :attr:`tensor1`
by :attr:`tensor2`, multiply the result by the scalar :attr:`value`
and add it to :attr:`tensor`.

The number of elements must match, but sizes do not matter.

Args:
    tensor (Tensor): the tensor to be added
    value (float, optional): multiplier for `tensor1 .* tensor2`
    tensor1 (Tensor): tensor to be multiplied
    tensor2 (Tensor): tensor to be multiplied
    out (Tensor, optional): Output tensor

Example::

    >>> t = torch.randn(2, 3)
    >>> t1 = torch.randn(1, 6)
    >>> t2 = torch.randn(6, 1)
    >>> torch.addcmul(t, 0.1, t1, t2)

     0.0122 -0.0188 -0.2354
     0.7396 -1.5721  1.2878
    [torch.FloatTensor of size 2x3]
""")

add_docstr(torch._C.addmm,
           """
addmm(beta=1, mat, alpha=1, mat1, mat2, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
The matrix :attr:`mat` is added to the final result.

If :attr:`mat1` is a `n x m` Tensor, :attr:`mat2` is a `m x p` Tensor,
:attr:`out` and :attr:`mat` will be `n x p` Tensors.

`alpha` and `beta` are scaling factors on `mat1 @ mat2` and `mat` respectively.

In other words,
:math:`out = (beta * M) + (alpha * mat1 @ mat2)`

Args:
    beta (float, optional): multiplier for :attr:`mat`
    mat (Tensor): matrix to be added
    alpha (float, optional): multiplier for `mat1 @ mat2`
    mat1 (Tensor): First matrix to be multiplied
    mat2 (Tensor): Second matrix to be multiplied
    out (Tensor, optional): Output tensor

Example::

    >>> M = torch.randn(2, 3)
    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.addmm(M, mat1, mat2)

    -0.4095 -1.9703  1.3561
     5.7674 -4.9760  2.7378
    [torch.FloatTensor of size 2x3]
""")

add_docstr(torch._C.addmv,
           """
addmv(beta=1, tensor, alpha=1, mat, vec, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`mat` and
the vector :attr:`vec`.
The vector :attr:`tensor` is added to the final result.

If :attr:`mat` is a `n x m` Tensor, :attr:`vec` is a 1D Tensor of size `m`,
:attr:`out` and :attr:`tensor` will be 1D of size `n`.

`alpha` and `beta` are scaling factors on `mat * vec` and `tensor` respectively.

In other words:

:math:`out = (beta * tensor) + (alpha * (mat @ vec2))`

Args:
    beta (float, optional): multiplier for :attr:`tensor`
    tensor (Tensor): vector to be added
    alpha (float, optional): multiplier for `mat @ vec`
    mat (Tensor): matrix to be multiplied
    vec (Tensor): vector to be multiplied
    out (Tensor, optional): Output tensor

Example::

    >>> M = torch.randn(2)
    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.addmv(M, mat, vec)

    -2.0939
    -2.2950
    [torch.FloatTensor of size 2]
""")

add_docstr(torch._C.addr,
           r"""
addr(beta=1, mat, alpha=1, vec1, vec2, out=None) -> Tensor

Performs the outer-product of vectors :attr:`vec1` and :attr:`vec2`
and adds it to the matrix :attr:`mat`.

Optional values :attr:`beta` and :attr:`alpha` are scalars that multiply
:attr:`mat` and :math:`(vec1 \otimes vec2)` respectively

In other words,
:math:`out = (beta * mat) + (alpha * vec1 \otimes vec2)`

If :attr:`vec1` is a vector of size `n` and :attr:`vec2` is a vector of size `m`,
then :attr:`mat` must be a matrix of size `n x m`

Args:
    beta (float, optional): Multiplier for :attr:`mat`
    mat (Tensor): Matrix to be added
    alpha (float, optional): Multiplier for outer product of for :attr:`vec1` and :attr:`vec2`
    vec1 (Tensor): First vector of the outer product
    vec2 (Tensor): Second vector of the outer product
    out (Tensor, optional): Output tensor

Example::

    >>> vec1 = torch.range(1, 3)
    >>> vec2 = torch.range(1, 2)
    >>> M = torch.zeros(3, 2)
    >>> torch.addr(M, vec1, vec2)
     1  2
     2  4
     3  6
    [torch.FloatTensor of size 3x2]
""")

add_docstr(torch._C.asin,
           """
asin(input, out=None) -> Tensor

Returns a new `Tensor` with the arcsine  of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.asin(a)
    -0.6900
     0.2752
     0.4633
        nan
    [torch.FloatTensor of size 4]
""")

add_docstr(torch._C.atan,
           """
atan(input, out=None) -> Tensor

Returns a new `Tensor` with the arctangent  of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.atan(a)
    -0.5669
     0.2653
     0.4203
     0.9196
    [torch.FloatTensor of size 4]
""")

add_docstr(torch._C.atan2,
           """
atan2(input1, input2, out=None) -> Tensor

Returns a new `Tensor` with the arctangent of the elements of :attr:`input1`
and :attr:`input2`.

Args:
    input1 (Tensor): the first input `Tensor`
    input2 (Tensor): the second input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.atan2(a, torch.randn(4))
    -2.4167
     2.9755
     0.9363
     1.6613
    [torch.FloatTensor of size 4]
""")

add_docstr(torch._C.baddbmm,
           r"""
baddbmm(beta=1, mat, alpha=1, batch1, batch2, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices in :attr:`batch1`
and :attr:`batch2`.
:attr:`mat` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3D Tensors each containing the same
number of matrices.

If :attr:`batch1` is a `b x n x m` Tensor, :attr:`batch2` is a `b x m x p`
Tensor, :attr:`out` and :attr:`mat` will be `b x n x p` Tensors.

In other words,
:math:`res_i = (beta * M_i) + (alpha * batch1_i \times batch2_i)`

Args:
    beta (float, optional): multiplier for :attr:`mat`
    mat (Tensor): tensor to be added
    alpha (float, optional): multiplier for `batch1 @ batch2`
    batch1 (Tensor): First batch of matrices to be multiplied
    batch2 (Tensor): Second batch of matrices to be multiplied
    out (Tensor, optional): Output tensor

Example::

    >>> M = torch.randn(10, 3, 5)
    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> torch.baddbmm(M, batch1, batch2).size()
    torch.Size([10, 3, 5])
""")

add_docstr(torch._C.bernoulli,
           """
bernoulli(input, out=None) -> Tensor

Draws binary random numbers (0 or 1) from a bernoulli distribution.

The :attr:`input` Tensor should be a tensor containing probabilities
to be used for drawing the binary random number.
Hence, all values in :attr:`input` have to be in the range:
:math:`0 <= input_i <= 1`

The `i-th` element of the output tensor will draw a value `1` according
to the `i-th` probability value given in :attr:`input`.

The returned :attr:`out` Tensor only has values 0 or 1 and is of the same
shape as :attr:`input`

Args:
    input (Tensor): Probability values for the bernoulli distribution
    out (Tensor, optional): Output tensor

Example::

    >>> a = torch.Tensor(3, 3).uniform_(0, 1) # generate a uniform random matrix with range [0, 1]
    >>> a

     0.7544  0.8140  0.9842
     0.5282  0.0595  0.6445
     0.1925  0.9553  0.9732
    [torch.FloatTensor of size 3x3]

    >>> torch.bernoulli(a)

     1  1  1
     0  0  1
     0  1  1
    [torch.FloatTensor of size 3x3]

    >>> a = torch.ones(3, 3) # probability of drawing "1" is 1
    >>> torch.bernoulli(a)

     1  1  1
     1  1  1
     1  1  1
    [torch.FloatTensor of size 3x3]

    >>> a = torch.zeros(3, 3) # probability of drawing "1" is 0
    >>> torch.bernoulli(a)

     0  0  0
     0  0  0
     0  0  0
    [torch.FloatTensor of size 3x3]

""")

add_docstr(torch._C.bmm,
           """
bmm(batch1, batch2, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices stored in :attr:`batch1` and :attr:`batch2`.

:attr:`batch1` and :attr:`batch2` must be 3D Tensors each containing the same number of matrices.

If :attr:`batch1` is a `b x n x m` Tensor, :attr:`batch2` is a `b x m x p` Tensor,
:attr:`out` will be a `b x n x p` Tensor.

Args:
    batch1 (Tensor): First batch of matrices to be multiplied
    batch2 (Tensor): Second batch of matrices to be multiplied
    out (Tensor, optional): Output tensor

Example::

    >>> batch1 = torch.randn(10, 3, 4)
    >>> batch2 = torch.randn(10, 4, 5)
    >>> res = torch.bmm(batch1, batch2)
    >>> res.size()
    torch.Size([10, 3, 5])
""")

add_docstr(torch._C.cat,
           """
cat(inputs, dimension=0) -> Tensor

Concatenates the given sequence of :attr:`inputs` Tensors in the given dimension.

:func:`torch.cat` can be seen as an inverse operation for :func:`torch.split` and :func:`torch.chunk`

:func:`cat` can be best understood via examples.

Args:
    inputs (sequence of Tensors): Can be any python sequence of `Tensor` of the same type.
    dimension (int, optional): The dimension over which the tensors are concatenated

Example::

    >>> x = torch.randn(2, 3)
    >>> x

     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size 2x3]

    >>> torch.cat((x, x, x), 0)

     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size 6x3]

    >>> torch.cat((x, x, x), 1)

     0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size 2x9]

""")

add_docstr(torch._C.ceil,
           """
ceil(input, out=None) -> Tensor

Returns a new `Tensor` with the ceil of the elements of :attr:`input`,
the smallest integer greater than or equal to each element.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]

    >>> torch.ceil(a)

     2
     1
    -0
    -0
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.reciprocal,
           """
reciprocal(input, out=None) -> Tensor

Returns a new `Tensor` with the reciprocal of the elements of :attr:`input`, i.e. :math:`1.0 / x`

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]

    >>> torch.reciprocal(a)

     0.7210
     2.5565
    -1.1583
    -1.8289
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.clamp,
           """
clamp(input, min, max, out=None) -> Tensor

Clamp all elements in :attr:`input` into the range `[min, max]` and return a resulting Tensor.

::

          | min, if x_i < min
    y_i = | x_i, if min <= x_i <= max
          | max, if x_i > max

Args:
    input (Tensor): the input `Tensor`
    min (float): lower-bound of the range to be clamped to
    max (float): upper-bound of the range to be clamped to
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]

    >>> torch.clamp(a, min=-0.5, max=0.5)

     0.5000
     0.3912
    -0.5000
    -0.5000
    [torch.FloatTensor of size 4]

.. function:: clamp(input, *, min, out=None) -> Tensor

Clamps all elements in :attr:`input` to be larger or equal :attr:`min`.

Args:
    input (Tensor): the input `Tensor`
    value (float): minimal value of each element in the output
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]

    >>> torch.clamp(a, min=0.5)

     1.3869
     0.5000
     0.5000
     0.5000
    [torch.FloatTensor of size 4]

.. function:: clamp(input, *, max, out=None) -> Tensor

Clamps all elements in :attr:`input` to be smaller or equal :attr:`max`.

Args:
    input (Tensor): the input `Tensor`
    value (float): maximal value of each element in the output
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]

    >>> torch.clamp(a, max=0.5)

     0.5000
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.cos,
           """
cos(input, out=None) -> Tensor

Returns a new `Tensor` with the cosine  of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.cos(a)
     0.8041
     0.9633
     0.9018
     0.2557
    [torch.FloatTensor of size 4]
""")

add_docstr(torch._C.cosh,
           """
cosh(input, out=None) -> Tensor

Returns a new `Tensor` with the hyperbolic cosine  of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.cosh(a)
     1.2095
     1.0372
     1.1015
     1.9917
    [torch.FloatTensor of size 4]
""")

add_docstr(torch._C.cross,
           """
cross(input, other, dim=-1, out=None) -> Tensor


Returns the cross product of vectors in dimension :attr:`dim` of :attr:`input` and :attr:`other`.

:attr:`input` and :attr:`other` must have the same size, and the size of their :attr:`dim` dimension should be 3.

If :attr:`dim` is not given, it defaults to the first dimension found with the size 3.

Args:
    input (Tensor): the input `Tensor`
    other (Tensor): the second input `Tensor`
    dim  (int, optional): the dimension to take the cross-product in.
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4, 3)
    >>> a

    -0.6652 -1.0116 -0.6857
     0.2286  0.4446 -0.5272
     0.0476  0.2321  1.9991
     0.6199  1.1924 -0.9397
    [torch.FloatTensor of size 4x3]

    >>> b = torch.randn(4, 3)
    >>> b

    -0.1042 -1.1156  0.1947
     0.9947  0.1149  0.4701
    -1.0108  0.8319 -0.0750
     0.9045 -1.3754  1.0976
    [torch.FloatTensor of size 4x3]

    >>> torch.cross(a, b, dim=1)

    -0.9619  0.2009  0.6367
     0.2696 -0.6318 -0.4160
    -1.6805 -2.0171  0.2741
     0.0163 -1.5304 -1.9311
    [torch.FloatTensor of size 4x3]

    >>> torch.cross(a, b)

    -0.9619  0.2009  0.6367
     0.2696 -0.6318 -0.4160
    -1.6805 -2.0171  0.2741
     0.0163 -1.5304 -1.9311
    [torch.FloatTensor of size 4x3]
""")

add_docstr(torch._C.cumprod,
           """
cumprod(input, dim, out=None) -> Tensor

Returns the cumulative product of elements of :attr:`input` in the dimension :attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be a vector of size N, with elements:
:math:`y_i = x_1 * x_2 * x_3 * ... * x_i`

Args:
    input (Tensor): the input `Tensor`
    dim  (int): the dimension to do the operation over
    out (Tensor, optional): The result `Tensor`

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
    [torch.FloatTensor of size 10]

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
    [torch.FloatTensor of size 10]

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
    [torch.FloatTensor of size 10]

""")

add_docstr(torch._C.cumsum,
           """
cumsum(input, dim, out=None) -> Tensor

Returns the cumulative sum of elements of :attr:`input` in the dimension :attr:`dim`.

For example, if :attr:`input` is a vector of size N, the result will also be a vector of size N, with elements:
:math:`y_i = x_1 + x_2 + x_3 + ... + x_i`

Args:
    input (Tensor): the input `Tensor`
    dim  (int): the dimension to do the operation over
    out (Tensor, optional): The result `Tensor`

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
    [torch.FloatTensor of size 10]

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
    [torch.FloatTensor of size 10]


""")

add_docstr(torch._C.diag,
           """
diag(input, diagonal=0, out=None) -> Tensor

- If :attr:`input` is a vector (1D Tensor), then returns a 2D square Tensor with the elements of :attr:`input`
  as the diagonal.
- If :attr:`input` is a matrix (2D Tensor), then returns a 1D Tensor with the diagonal elements of :attr:`input`.

The argument :attr:`diagonal` controls which diagonal to consider.

- :attr:`diagonal` = 0, is the main diagonal.
- :attr:`diagonal` > 0, is above the main diagonal.
- :attr:`diagonal` < 0, is below the main diagonal.

Args:
    input (Tensor): the input `Tensor`
    diagonal (int, optional): the diagonal to consider
    out (Tensor, optional): The result `Tensor`

Example:

Get the square matrix where the input vector is the diagonal::

    >>> a = torch.randn(3)
    >>> a

     1.0480
    -2.3405
    -1.1138
    [torch.FloatTensor of size 3]

    >>> torch.diag(a)

     1.0480  0.0000  0.0000
     0.0000 -2.3405  0.0000
     0.0000  0.0000 -1.1138
    [torch.FloatTensor of size 3x3]

    >>> torch.diag(a, 1)

     0.0000  1.0480  0.0000  0.0000
     0.0000  0.0000 -2.3405  0.0000
     0.0000  0.0000  0.0000 -1.1138
     0.0000  0.0000  0.0000  0.0000
    [torch.FloatTensor of size 4x4]


Get the k-th diagonal of a given matrix::

    >>> a = torch.randn(3, 3)
    >>> a

    -1.5328 -1.3210 -1.5204
     0.8596  0.0471 -0.2239
    -0.6617  0.0146 -1.0817
    [torch.FloatTensor of size 3x3]

    >>> torch.diag(a, 0)

    -1.5328
     0.0471
    -1.0817
    [torch.FloatTensor of size 3]

    >>> torch.diag(a, 1)

    -1.3210
    -0.2239
    [torch.FloatTensor of size 2]

""")

add_docstr(torch._C.dist,
           """
dist(input, other, p=2, out=None) -> Tensor

Returns the p-norm of (:attr:`input` - :attr:`other`)

Args:
    input (Tensor): the input `Tensor`
    other (Tensor): the Right-hand-side input `Tensor`
    p (float, optional): The norm to be computed.
    out (Tensor, optional): The result `Tensor`

Example::

    >>> x = torch.randn(4)
    >>> x

     0.2505
    -0.4571
    -0.3733
     0.7807
    [torch.FloatTensor of size 4]

    >>> y = torch.randn(4)
    >>> y

     0.7782
    -0.5185
     1.4106
    -2.4063
    [torch.FloatTensor of size 4]

    >>> torch.dist(x, y, 3.5)
    3.302832063224223
    >>> torch.dist(x, y, 3)
    3.3677282206393286
    >>> torch.dist(x, y, 0)
    inf
    >>> torch.dist(x, y, 1)
    5.560028076171875


""")

add_docstr(torch._C.div,
           """
.. function:: div(input, value, out=None)

Divides each element of the input :attr:`input` with the scalar :attr:`value` and returns a new resulting tensor.

:math:`out = tensor / value`

Args:
    input (Tensor): the input `Tensor`
    value (float): the number to be divided to each element of :attr:`input`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(5)
    >>> a

    -0.6147
    -1.1237
    -0.1604
    -0.6853
     0.1063
    [torch.FloatTensor of size 5]

    >>> torch.div(a, 0.5)

    -1.2294
    -2.2474
    -0.3208
    -1.3706
     0.2126
    [torch.FloatTensor of size 5]


.. function:: div(input, other, out=None)

Each element of the Tensor :attr:`input` is divided by each element of the Tensor :attr:`other`.
The resulting Tensor is returned. The shapes of :attr:`input` and :attr:`other` don't need to match.
The total number of elements in each Tensor need to be the same.

.. note:: When the shapes do not match, the shape of :attr:`input` is used as the shape for the returned output Tensor

:math:`out_i = input_i / other_i`

Args:
    input (Tensor): the numerator `Tensor`
    other (Tensor): the denominator `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4,4)
    >>> a

    -0.1810  0.4017  0.2863 -0.1013
     0.6183  2.0696  0.9012 -1.5933
     0.5679  0.4743 -0.0117 -0.1266
    -0.1213  0.9629  0.2682  1.5968
    [torch.FloatTensor of size 4x4]

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
    [torch.FloatTensor of size 8x2]

    >>> torch.div(a, b)

    -0.2062  0.5251  0.3229 -0.0684
    -0.9528  1.8525  0.6320  1.9559
     0.3881 -3.8625 -0.0253  0.2099
    -0.3473  0.6306  0.1666 -2.5381
    [torch.FloatTensor of size 4x4]


""")

add_docstr(torch._C.dot,
           """
dot(tensor1, tensor2) -> float

Computes the dot product (inner product) of two tensors. Both tensors are
treated as 1-D vectors.

Example::

    >>> torch.dot(torch.Tensor([2, 3]), torch.Tensor([2, 1]))
    7.0
""")

add_docstr(torch._C.eig,
           """
eig(a, eigenvectors=False, out=None) -> (Tensor, Tensor)

Computes the eigenvalues and eigenvectors of a real square matrix.

Args:
    a (Tensor): A square matrix for which the eigenvalues and eigenvectors will
                be computed
    eigenvectors (bool): `True` to compute both eigenvalues and eigenvectors.
                         Otherwise, only eigenvalues will be computed.
    out (tuple, optional): Output tensors

Returns:
    (Tensor, Tensor): tuple containing

        - **e** (*Tensor*): the right eigenvalues of ``a``
        - **v** (*Tensor*): the eigenvectors of ``a`` if ``eigenvectors` is ``True``; otherwise an empty tensor
""")

add_docstr(torch._C.eq,
           """
eq(input, other, out=None) -> Tensor

Computes element-wise equality

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    input (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where the tensors are equal and
            a 0 at every other location

Example::

    >>> torch.eq(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
    1  0
    0  1
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.equal,
           """
equal(tensor1, tensor2) -> bool

True if two tensors have the same size and elements, False otherwise.

Example::

    >>> torch.equal(torch.Tensor([1, 2]), torch.Tensor([1, 2]))
    True
""")

add_docstr(torch._C.exp,
           """
exp(tensor, out=None) -> Tensor

Computes the exponential of each element.

Example::

    >>> torch.exp(torch.Tensor([0, math.log(2)]))
    torch.FloatTensor([1, 2])
""")

add_docstr(torch._C.eye,
           """
eye(n, m=None, out=None)

Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

Args:
    n (int): Number of rows
    m (int, optional): Number of columns. If None, defaults to `n`
    out (Tensor, optional): Output tensor

Returns:
    Tensor: a 2-D tensor with ones on the diagonal and zeros elsewhere

Example::

    >>> torch.eye(3)
     1  0  0
     0  1  0
     0  0  1
    [torch.FloatTensor of size 3x3]
""")

add_docstr(torch._C.floor,
           """
floor(input, out=None) -> Tensor

Returns a new `Tensor` with the floor of the elements of :attr:`input`,
the largest integer less than or equal to each element.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]

    >>> torch.floor(a)

     1
     0
    -1
    -1
    [torch.FloatTensor of size 4]


""")

add_docstr(torch._C.fmod,
           """
fmod(input, divisor, out=None) -> Tensor

Computes the element-wise remainder of division.

The dividend and divisor may contain both for integer and floating point
numbers. The remainder has the same sign as the dividend `tensor`.

Args:
    input (Tensor): The dividend
    divisor (Tensor or float): The divisor. This may be either a number or a
                               tensor of the same shape as the dividend.
    out (Tensor, optional): Output tensor

Example::

    >>> torch.fmod(torch.Tensor([-3, -2, -1, 1, 2, 3]), 2)
    torch.FloatTensor([-1, -0, -1, 1, 0, 1])
    >>> torch.fmod(torch.Tensor([1, 2, 3, 4, 5]), 1.5)
    torch.FloatTensor([1.0, 0.5, 0.0, 1.0, 0.5])

.. seealso::

        :func:`torch.remainder`, which computes the element-wise remainder of
        division equivalently to Python's `%` operator
""")

add_docstr(torch._C.frac,
           """
frac(tensor, out=None) -> Tensor

Computes the fractional portion of each element in `tensor`.

Example::

    >>> torch.frac(torch.Tensor([1, 2.5, -3.2])
    torch.FloatTensor([0, 0.5, -0.2])
""")

add_docstr(torch._C.from_numpy,
           """
from_numpy(ndarray) -> Tensor

Creates a :class:`Tensor` from a :class:`numpy.ndarray`.

The returned tensor and `ndarray` share the same memory. Modifications to the
tensor will be reflected in the `ndarray` and vice versa. The returned tensor
is not resizable.

Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.from_numpy(a)
    >>> t
    torch.LongTensor([1, 2, 3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])
""")

add_docstr(torch._C.gather,
           """
gather(input, dim, index, out=None) -> Tensor

Gathers values along an axis specified by `dim`.

For a 3-D tensor the output is specified by::

    out[i][j][k] = tensor[index[i][j][k]][j][k]  # dim=0
    out[i][j][k] = tensor[i][index[i][j][k]][k]  # dim=1
    out[i][j][k] = tensor[i][j][index[i][j][k]]  # dim=3

Args:
    input (Tensor): The source tensor
    dim (int): The axis along which to index
    index (LongTensor): The indices of elements to gather
    out (Tensor, optional): Destination tensor

Example::

    >>> t = torch.Tensor([[1,2],[3,4]])
    >>> torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
     1  1
     4  3
    [torch.FloatTensor of size 2x2]
""")

add_docstr(torch._C.ge,
           """
ge(input, other, out=None) -> Tensor

Computes `tensor >= other` element-wise.

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    input (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example::

    >>> torch.ge(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
     1  1
     0  1
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.gels,
           r"""
gels(B, A, out=None) -> Tensor

Computes the solution to the least squares and least norm problems for a full
rank :math:`m` by :math:`n` matrix :math:`A`.

If :math:`m >= n`, :func:`gels` solves the least-squares problem:

.. math::

   \begin{array}{ll}
   \mbox{minimize} & \|AX-B\|_F.
   \end{array}

If :math:`m < n`, :func:`gels` solves the least-norm problem:

.. math::

   \begin{array}{ll}
   \mbox{minimize} & \|X\|_F & \mbox{subject to} & AX = B.
   \end{array}

The first :math:`n` rows of the returned matrix :math:`X` contains the
solution. The remaining rows contain residual information: the euclidean norm
of each column starting at row :math:`n` is the residual for the corresponding
column.

Args:
    B (Tensor): The matrix :math:`B`
    A (Tensor): The :math:`m` by :math:`n` matrix :math:`A`
    out (tuple, optional): Optional destination tensor

Returns:
    (Tensor, Tensor): tuple containing:

        - **X** (*Tensor*): the least squares solution
        - **qr** (*Tensor*): the details of the QR factorization

.. note::

    The returned matrices will always be tranposed, irrespective of the strides
    of the input matrices. That is, they will have stride `(1, m)` instead of
    `(m, 1)`.

Example::


    >>> A = torch.Tensor([[1, 1, 1],
    ...                   [2, 3, 4],
    ...                   [3, 5, 2],
    ...                   [4, 2, 5],
    ...                   [5, 4, 3]])
    >>> B = torch.Tensor([[-10, -3],
                          [ 12, 14],
                          [ 14, 12],
                          [ 16, 16],
                          [ 18, 16]])
    >>> X, _ = torch.gels(B, A)
    >>> X
    2.0000  1.0000
    1.0000  1.0000
    1.0000  2.0000
    [torch.FloatTensor of size 3x2]
""")

add_docstr(torch._C.geqrf,
           r"""
geqrf(input, out=None) -> (Tensor, Tensor)

This is a low-level function for calling LAPACK directly.

You'll generally want to use :func:`torch.qr` instead.

Computes a QR decomposition of :attr:`input`, but without constructing `Q` and `R` as explicit separate matrices.

Rather, this directly calls the underlying LAPACK function `?geqrf`
which produces a sequence of 'elementary reflectors'.

See `LAPACK documentation`_ for further details.

Args:
    input (Tensor): the input matrix
    out (tuple, optional): The result tuple of (Tensor, Tensor)

.. _LAPACK documentation:
    https://software.intel.com/en-us/node/521004

""")

add_docstr(torch._C.ger,
           """
ger(vec1, vec2, out=None) -> Tensor

Outer product of :attr:`vec1` and :attr:`vec2`.
If :attr:`vec1` is a vector of size `n` and :attr:`vec2` is a vector of size `m`,
then :attr:`out` must be a matrix of size `n x m`.

Args:
    vec1 (Tensor): 1D input vector
    vec2 (Tensor): 1D input vector
    out (Tensor, optional): optional output matrix

Example::

    >>> v1 = torch.range(1, 4)
    >>> v2 = torch.range(1, 3)
    >>> torch.ger(v1, v2)

      1   2   3
      2   4   6
      3   6   9
      4   8  12
    [torch.FloatTensor of size 4x3]

""")

add_docstr(torch._C.gesv,
           """
gesv(B, A, out=None) -> (Tensor, Tensor)

`X, LU = torch.gesv(B, A)` returns the solution to the system of linear
equations represented by :math:`AX = B`

`LU` contains `L` and `U` factors for LU factorization of `A`.

:attr:`A` has to be a square and non-singular matrix (2D Tensor).

If `A` is an `m x m` matrix and `B` is `m x k`,
the result `LU` is `m x m` and `X` is `m x k` .

.. note::

    Irrespective of the original strides, the returned matrices
    `X` and `LU` will be transposed, i.e. with strides `(1, m)`
    instead of `(m, 1)`.

Args:
    B (Tensor): input matrix of `m x k` dimensions
    A (Tensor): input square matrix of `m x m` dimensions
    out (Tensor, optional): optional output matrix

Example::

    >>> A = torch.Tensor([[6.80, -2.11,  5.66,  5.97,  8.23],
    ...                   [-6.05, -3.30,  5.36, -4.44,  1.08],
    ...                   [-0.45,  2.58, -2.70,  0.27,  9.04],
    ...                   [8.32,  2.71,  4.35,  -7.17,  2.14],
    ...                   [-9.67, -5.14, -7.26,  6.08, -6.87]]).t()
    >>> B = torch.Tensor([[4.02,  6.19, -8.22, -7.57, -3.03],
    ...                   [-1.56,  4.00, -8.67,  1.75,  2.86],
    ...                   [9.81, -4.09, -4.57, -8.61,  8.99]]).t()
    >>> X, LU = torch.gesv(B, A)
    >>> torch.dist(B, torch.mm(A, X))
    9.250057093890353e-06

""")

add_docstr(torch._C.get_num_threads,
           """
get_num_threads() -> int

Gets the number of OpenMP threads used for parallelizing CPU operations
""")

add_docstr(torch._C.gt,
           """
gt(input, other, out=None) -> Tensor

Computes `tensor > other` element-wise.

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    input (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example::

    >>> torch.gt(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
     0  1
     0  0
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.histc,
           """
histc(input, bins=100, min=0, max=0, out=None) -> Tensor

Computes the histogram of a tensor.

The elements are sorted into equal width bins between `min` and `max`. If `min`
and `max` are both zero, the minimum and maximum values of the data are used.

Args:
    input (Tensor): Input data
    bins (int): Number of histogram bins
    min (int): Lower end of the range (inclusive)
    max (int): Upper end of the range (inclusive)
    out (Tensor, optional): Output argument

Returns:
    Tensor: the histogram

Example::

    >>> torch.histc(torch.FloatTensor([1, 2, 1]), bins=4, min=0, max=3)
    FloatTensor([0, 2, 1, 0])

""")

add_docstr(torch._C.index_select,
           """
index_select(input, dim, index, out=None) -> Tensor

Returns a new `Tensor` which indexes the :attr:`input` `Tensor` along dimension :attr:`dim`
using the entries in :attr:`index` which is a `LongTensor`.

The returned `Tensor` has the same number of dimensions as the original `Tensor`.

.. note:: The returned `Tensor` does **not** use the same storage as the original `Tensor`

Args:
    input (Tensor): Input data
    dim (int): the dimension in which we index
    index (LongTensor): the 1D tensor containing the indices to index
    out (Tensor, optional): Output argument

Example::

    >>> x = torch.randn(3, 4)
    >>> x

     1.2045  2.4084  0.4001  1.1372
     0.5596  1.5677  0.6219 -0.7954
     1.3635 -1.2313 -0.5414 -1.8478
    [torch.FloatTensor of size 3x4]

    >>> indices = torch.LongTensor([0, 2])
    >>> torch.index_select(x, 0, indices)

     1.2045  2.4084  0.4001  1.1372
     1.3635 -1.2313 -0.5414 -1.8478
    [torch.FloatTensor of size 2x4]

    >>> torch.index_select(x, 1, indices)

     1.2045  0.4001
     0.5596  0.6219
     1.3635 -0.5414
    [torch.FloatTensor of size 3x2]

""")

add_docstr(torch._C.inverse,
           """
inverse(input, out=None) -> Tensor

Takes the inverse of the square matrix :attr:`input`.

.. note::

    Irrespective of the original strides, the returned matrix will be transposed,
    i.e. with strides `(1, m)` instead of `(m, 1)`

Args:
    input (Tensor): the input 2D square `Tensor`
    out (Tensor, optional): the optional output `Tensor`

Example::

    >>> x = torch.rand(10, 10)
    >>> x

     0.7800  0.2267  0.7855  0.9479  0.5914  0.7119  0.4437  0.9131  0.1289  0.1982
     0.0045  0.0425  0.2229  0.4626  0.6210  0.0207  0.6338  0.7067  0.6381  0.8196
     0.8350  0.7810  0.8526  0.9364  0.7504  0.2737  0.0694  0.5899  0.8516  0.3883
     0.6280  0.6016  0.5357  0.2936  0.7827  0.2772  0.0744  0.2627  0.6326  0.9153
     0.7897  0.0226  0.3102  0.0198  0.9415  0.9896  0.3528  0.9397  0.2074  0.6980
     0.5235  0.6119  0.6522  0.3399  0.3205  0.5555  0.8454  0.3792  0.4927  0.6086
     0.1048  0.0328  0.5734  0.6318  0.9802  0.4458  0.0979  0.3320  0.3701  0.0909
     0.2616  0.3485  0.4370  0.5620  0.5291  0.8295  0.7693  0.1807  0.0650  0.8497
     0.1655  0.2192  0.6913  0.0093  0.0178  0.3064  0.6715  0.5101  0.2561  0.3396
     0.4370  0.4695  0.8333  0.1180  0.4266  0.4161  0.0699  0.4263  0.8865  0.2578
    [torch.FloatTensor of size 10x10]

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
    [torch.FloatTensor of size 10x10]

    >>> torch.max(torch.abs(z - torch.eye(10))) # Max nonzero
    5.096662789583206e-07

""")

add_docstr(torch._C.kthvalue,
           """
kthvalue(input, k, dim=None, out=None) -> (Tensor, LongTensor)

Returns the :attr:`k`th smallest element of the given :attr:`input` Tensor along a given dimension.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

A tuple of `(values, indices)` is returned, where the `indices` is the indices of
the kth-smallest element in the original `input` Tensor in dimention `dim`.

Args:
    input (Tensor): the input `Tensor`
    k (int): k for the k-th smallest element
    dim (int, optional): The dimension to sort along
    out (tuple, optional): The output tuple of (Tensor, LongTensor)
                           can be optionally given to be used as output buffers

Example::

    >>> x = torch.range(1, 5)
    >>> x

     1
     2
     3
     4
     5
    [torch.FloatTensor of size 5]

    >>> torch.kthvalue(x, 4)
    (
     4
    [torch.FloatTensor of size 1]
    ,
     3
    [torch.LongTensor of size 1]
    )

""")

add_docstr(torch._C.le,
           """
le(input, other, out=None) -> Tensor

Computes `tensor <= other` element-wise.

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    input (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example::

    >>> torch.le(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
     1  0
     1  1
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.lerp,
           """
lerp(start, end, weight, out=None)

Does a linear interpolation of two tensors :attr:`start` and :attr:`end` based
on a scalar :attr:`weight`: and returns the resulting :attr:`out` Tensor.

:math:`out_i = start_i + weight * (end_i - start_i)`

Args:
    start (Tensor): the `Tensor` with the starting points
    end (Tensor): the `Tensor` with the ending points
    weight (float): the weight for the interpolation formula
    out (Tensor, optional): The result `Tensor`

Example::

    >>> start = torch.range(1, 4)
    >>> end = torch.Tensor(4).fill_(10)
    >>> start

     1
     2
     3
     4
    [torch.FloatTensor of size 4]

    >>> end

     10
     10
     10
     10
    [torch.FloatTensor of size 4]

    >>> torch.lerp(start, end, 0.5)

     5.5000
     6.0000
     6.5000
     7.0000
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.linspace,
           """
linspace(start, end, steps=100, out=None) -> Tensor

Returns a one-dimensional Tensor of :attr:`steps`
equally spaced points between :attr:`start` and :attr:`end`

The output tensor is 1D of size :attr:`steps`

Args:
    start (float): The starting value for the set of points
    end (float): The ending value for the set of points
    steps (int): Number of points to sample between :attr:`start` and :attr:`end`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> torch.linspace(3, 10, steps=5)

      3.0000
      4.7500
      6.5000
      8.2500
     10.0000
    [torch.FloatTensor of size 5]

    >>> torch.linspace(-10, 10, steps=5)

    -10
     -5
      0
      5
     10
    [torch.FloatTensor of size 5]

    >>> torch.linspace(start=-10, end=10, steps=5)

    -10
     -5
      0
      5
     10
    [torch.FloatTensor of size 5]

""")

add_docstr(torch._C.log,
           """
log(input, out=None) -> Tensor

Returns a new `Tensor` with the natural logarithm of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(5)
    >>> a

    -0.4183
     0.3722
    -0.3091
     0.4149
     0.5857
    [torch.FloatTensor of size 5]

    >>> torch.log(a)

        nan
    -0.9883
        nan
    -0.8797
    -0.5349
    [torch.FloatTensor of size 5]

""")

add_docstr(torch._C.log1p,
           """
log1p(input, out=None) -> Tensor

Returns a new `Tensor` with the natural logarithm of (1 + :attr:`input`).

:math:`y_i = log(x_i + 1)`

.. note:: This function is more accurate than :func:`torch.log` for small values of :attr:`input`

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(5)
    >>> a

    -0.4183
     0.3722
    -0.3091
     0.4149
     0.5857
    [torch.FloatTensor of size 5]

    >>> torch.log1p(a)

    -0.5418
     0.3164
    -0.3697
     0.3471
     0.4611
    [torch.FloatTensor of size 5]

""")

add_docstr(torch._C.logspace,
           """
logspace(start, end, steps=100, out=None) -> Tensor

Returns a one-dimensional Tensor of :attr:`steps` points
logarithmically spaced between :math:`10^{start}` and :math:`10^{end}`

The output is a 1D tensor of size :attr:`steps`

Args:
    start (float): The starting value for the set of points
    end (float): The ending value for the set of points
    steps (int): Number of points to sample between :attr:`start` and :attr:`end`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> torch.logspace(start=-10, end=10, steps=5)

     1.0000e-10
     1.0000e-05
     1.0000e+00
     1.0000e+05
     1.0000e+10
    [torch.FloatTensor of size 5]

    >>> torch.logspace(start=0.1, end=1.0, steps=5)

      1.2589
      2.1135
      3.5481
      5.9566
     10.0000
    [torch.FloatTensor of size 5]

""")

add_docstr(torch._C.lt,
           """
lt(input, other, out=None) -> Tensor

Computes `tensor < other` element-wise.

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    input (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example::

    >>> torch.lt(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
     0  0
     1  0
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.masked_select,
           """
masked_select(input, mask, out=None) -> Tensor

Returns a new 1D `Tensor` which indexes the :attr:`input` `Tensor` according to
the binary mask :attr:`mask` which is a `ByteTensor`.

The :attr:`mask` tensor needs to have the same number of elements as
:attr:`input`, but it's shape or dimensionality are irrelevant.

.. note:: The returned `Tensor` does **not** use the same storage as the original `Tensor`

Args:
    input (Tensor): Input data
    mask  (ByteTensor): the tensor containing the binary mask to index with
    out (Tensor, optional): Output argument

Example::

    >>> x = torch.randn(3, 4)
    >>> x

     1.2045  2.4084  0.4001  1.1372
     0.5596  1.5677  0.6219 -0.7954
     1.3635 -1.2313 -0.5414 -1.8478
    [torch.FloatTensor of size 3x4]

    >>> mask = x.ge(0.5)
    >>> mask

     1  1  0  1
     1  1  1  0
     1  0  0  0
    [torch.ByteTensor of size 3x4]

    >>> torch.masked_select(x, mask)

     1.2045
     2.4084
     1.1372
     0.5596
     1.5677
     0.6219
     1.3635
    [torch.FloatTensor of size 7]

""")

add_docstr(torch._C.max,
           """
.. function:: max(input) -> float

Returns the maximum value of all elements in the :attr:`input` Tensor.

Args:
    input (Tensor): the input `Tensor`

Example::

    >>> a = torch.randn(1, 3)
    >>> a

     0.4729 -0.2266 -0.2085
    [torch.FloatTensor of size 1x3]

    >>> torch.max(a)
    0.4729


.. function:: max(input, dim, max=None, max_indices=None) -> (Tensor, LongTensor)

Returns the maximum value of each row of the :attr:`input` Tensor in the given dimension :attr:`dim`.
Also returns the index location of each maximum value found.

The output Tensors are of the same size as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.

Args:
    input (Tensor): the input `Tensor`
    dim (int): the dimension to reduce
    max (Tensor, optional): the result Tensor with maximum values in dimension :attr:`dim`
    max_indices (LongTensor, optional): the result Tensor with the index locations of the
                                        maximum values in dimension :attr:`dim`

Example::

    >> a = torch.randn(4, 4)
    >> a

    0.0692  0.3142  1.2513 -0.5428
    0.9288  0.8552 -0.2073  0.6409
    1.0695 -0.0101 -2.4507 -1.2230
    0.7426 -0.7666  0.4862 -0.6628
    torch.FloatTensor of size 4x4]

    >>> torch.max(a, 1)
    (
     1.2513
     0.9288
     1.0695
     0.7426
    [torch.FloatTensor of size 4x1]
    ,
     2
     0
     0
     0
    [torch.LongTensor of size 4x1]
    )

.. function:: max(input, other, out=None) -> Tensor

Each element of the Tensor :attr:`input` is compared with the corresponding element of the Tensor :attr:`other`
and an element-wise `max` is taken.

The shapes of :attr:`input` and :attr:`other` don't need to match.
The total number of elements in each Tensor need to be the same.

.. note:: When the shapes do not match, the shape of :attr:`input` is used as the shape for the returned output Tensor

:math:`out_i = max(tensor_i, other_i)`

Args:
    input (Tensor): the input `Tensor`
    other (Tensor): the second input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]

    >>> b = torch.randn(4)
    >>> b

     1.0067
    -0.8010
     0.6258
     0.3627
    [torch.FloatTensor of size 4]

    >>> torch.max(a, b)

     1.3869
     0.3912
     0.6258
     0.3627
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.mean,
           """
.. function:: mean(input) -> float

Returns the mean value of all elements in the :attr:`input` Tensor.

Args:
    input (Tensor): the input `Tensor`

Example::

    >>> a = torch.randn(1, 3)
    >>> a

    -0.2946 -0.9143  2.1809
    [torch.FloatTensor of size 1x3]

    >>> torch.mean(a)
    0.32398951053619385


.. function:: mean(input, dim, out=None) -> Tensor

Returns the mean value of each row of the :attr:`input` Tensor in the given dimension :attr:`dim`.

The output Tensor is of the same size as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.

Args:
    input (Tensor): the input `Tensor`
    dim (int): the dimension to reduce
    out (Tensor, optional): the result Tensor

Example::

    >>> a = torch.randn(4, 4)
    >>> a

    -1.2738 -0.3058  0.1230 -1.9615
     0.8771 -0.5430 -0.9233  0.9879
     1.4107  0.0317 -0.6823  0.2255
    -1.3854  0.4953 -0.2160  0.2435
    [torch.FloatTensor of size 4x4]

    >>> torch.mean(a, 1)

    -0.8545
     0.0997
     0.2464
    -0.2157
    [torch.FloatTensor of size 4x1]

""")

add_docstr(torch._C.median,
           """
median(input, dim=-1, values=None, indices=None) -> (Tensor, LongTensor)

Returns the median value of each row of the :attr:`input` Tensor in the given dimension :attr:`dim`.
Also returns the index location of the median value as a `LongTensor`.

By default, :attr:`dim` is the last dimension of the :attr:`input` Tensor.

The output Tensors are of the same size as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.

.. note:: This function is not defined for ``torch.cuda.Tensor`` yet.

Args:
    input (Tensor): the input `Tensor`
    dim (int): the dimension to reduce
    values (Tensor, optional): the result Tensor
    indices (Tensor, optional): the result index Tensor

Example::

    >>> a

     -0.6891 -0.6662
     0.2697  0.7412
     0.5254 -0.7402
     0.5528 -0.2399
    [torch.FloatTensor of size 4x2]

    >>> a = torch.randn(4, 5)
    >>> a

     0.4056 -0.3372  1.0973 -2.4884  0.4334
     2.1336  0.3841  0.1404 -0.1821 -0.7646
    -0.2403  1.3975 -2.0068  0.1298  0.0212
    -1.5371 -0.7257 -0.4871 -0.2359 -1.1724
    [torch.FloatTensor of size 4x5]

    >>> torch.median(a, 1)
    (
     0.4056
     0.1404
     0.0212
    -0.7257
    [torch.FloatTensor of size 4x1]
    ,
     0
     2
     4
     1
    [torch.LongTensor of size 4x1]
    )

""")

add_docstr(torch._C.min,
           """
.. function:: min(input) -> float

Returns the minimum value of all elements in the :attr:`input` Tensor.

Args:
    input (Tensor): the input `Tensor`

Example::

    >>> a = torch.randn(1, 3)
    >>> a

     0.4729 -0.2266 -0.2085
    [torch.FloatTensor of size 1x3]

    >>> torch.min(a)
    -0.22663167119026184


.. function:: min(input, dim, min=None, min_indices=None) -> (Tensor, LongTensor)

Returns the minimum value of each row of the :attr:`input` Tensor in the given dimension :attr:`dim`.
Also returns the index location of each minimum value found.

The output Tensors are of the same size as :attr:`input` except in the dimension :attr:`dim` where they are of size 1.

Args:
    input (Tensor): the input `Tensor`
    dim (int): the dimension to reduce
    min (Tensor, optional): the result Tensor with minimum values in dimension :attr:`dim`
    min_indices (LongTensor, optional): the result Tensor with the index locations of the
                                        minimum values in dimension :attr:`dim`

Example::

    >> a = torch.randn(4, 4)
    >> a

    0.0692  0.3142  1.2513 -0.5428
    0.9288  0.8552 -0.2073  0.6409
    1.0695 -0.0101 -2.4507 -1.2230
    0.7426 -0.7666  0.4862 -0.6628
    torch.FloatTensor of size 4x4]

    >> torch.min(a, 1)

    0.5428
    0.2073
    2.4507
    0.7666
    torch.FloatTensor of size 4x1]

    3
    2
    2
    1
    torch.LongTensor of size 4x1]

.. function:: min(input, other, out=None) -> Tensor

Each element of the Tensor :attr:`input` is compared with the corresponding element of the Tensor :attr:`other`
and an element-wise `min` is taken. The resulting Tensor is returned.

The shapes of :attr:`input` and :attr:`other` don't need to match.
The total number of elements in each Tensor need to be the same.

.. note:: When the shapes do not match, the shape of :attr:`input` is used as the shape for the returned output Tensor

:math:`out_i = min(tensor_i, other_i)`

Args:
    input (Tensor): the input `Tensor`
    other (Tensor): the second input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]

    >>> b = torch.randn(4)
    >>> b

     1.0067
    -0.8010
     0.6258
     0.3627
    [torch.FloatTensor of size 4]

    >>> torch.min(a, b)

     1.0067
    -0.8010
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.mm,
           """
mm(mat1, mat2, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.

If :attr:`mat1` is a `n x m` Tensor, :attr:`mat2` is a `m x p` Tensor, :attr:`out` will be a `n x p` Tensor.

Args:
    mat1 (Tensor): First matrix to be multiplied
    mat2 (Tensor): Second matrix to be multiplied
    out (Tensor, optional): Output tensor

Example::

    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.mm(mat1, mat2)
     0.0519 -0.3304  1.2232
     4.3910 -5.1498  2.7571
    [torch.FloatTensor of size 2x3]
""")

add_docstr(torch._C.mode,
           """
mode(input, dim=-1, values=None, indices=None) -> (Tensor, LongTensor)

Returns the mode value of each row of the :attr:`input` Tensor in the given dimension :attr:`dim`.
Also returns the index location of the mode value as a `LongTensor`.

By default, :attr:`dim` is the last dimension of the :attr:`input` Tensor.

The output Tensors are of the same size as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.

.. note:: This function is not defined for ``torch.cuda.Tensor`` yet.

Args:
    input (Tensor): the input `Tensor`
    dim (int): the dimension to reduce
    values (Tensor, optional): the result Tensor
    indices (Tensor, optional): the result index Tensor

Example::

    >>> a

     -0.6891 -0.6662
     0.2697  0.7412
     0.5254 -0.7402
     0.5528 -0.2399
    [torch.FloatTensor of size 4x2]

    >>> a = torch.randn(4, 5)
    >>> a

     0.4056 -0.3372  1.0973 -2.4884  0.4334
     2.1336  0.3841  0.1404 -0.1821 -0.7646
    -0.2403  1.3975 -2.0068  0.1298  0.0212
    -1.5371 -0.7257 -0.4871 -0.2359 -1.1724
    [torch.FloatTensor of size 4x5]

    >>> torch.mode(a, 1)
    (
    -2.4884
    -0.7646
    -2.0068
    -1.5371
    [torch.FloatTensor of size 4x1]
    ,
     3
     4
     2
     0
    [torch.LongTensor of size 4x1]
    )

""")

add_docstr(torch._C.mul,
           """
.. function:: mul(input, value, out=None)

Multiplies each element of the input :attr:`input` with the scalar :attr:`value` and returns a new resulting tensor.

:math:`out = tensor * value`

Args:
    input (Tensor): the input `Tensor`
    value (float): the number to be multiplied to each element of :attr:`input`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(3)
    >>> a

    -0.9374
    -0.5254
    -0.6069
    [torch.FloatTensor of size 3]

    >>> torch.mul(a, 100)

    -93.7411
    -52.5374
    -60.6908
    [torch.FloatTensor of size 3]


.. function:: mul(input, other, out=None)

Each element of the Tensor :attr:`input` is multiplied by each element of the Tensor :attr:`other`.
The resulting Tensor is returned. The shapes of :attr:`input` and :attr:`other` don't need to match.
The total number of elements in each Tensor need to be the same.

.. note:: When the shapes do not match, the shape of :attr:`input` is used as the shape for the returned output Tensor

:math:`out_i = input_i * other_i`

Args:
    input (Tensor): the first multiplicand `Tensor`
    other (Tensor): the second multiplicand `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4,4)
    >>> a

    -0.7280  0.0598 -1.4327 -0.5825
    -0.1427 -0.0690  0.0821 -0.3270
    -0.9241  0.5110  0.4070 -1.1188
    -0.8308  0.7426 -0.6240 -1.1582
    [torch.FloatTensor of size 4x4]

    >>> b = torch.randn(2, 8)
    >>> b

     0.0430 -1.0775  0.6015  1.1647 -0.6549  0.0308 -0.1670  1.0742
    -1.2593  0.0292 -0.0849  0.4530  1.2404 -0.4659 -0.1840  0.5974
    [torch.FloatTensor of size 2x8]

    >>> torch.mul(a, b)

    -0.0313 -0.0645 -0.8618 -0.6784
     0.0934 -0.0021 -0.0137 -0.3513
     1.1638  0.0149 -0.0346 -0.5068
    -1.0304 -0.3460  0.1148 -0.6919
    [torch.FloatTensor of size 4x4]

""")

add_docstr(torch._C.multinomial,
           u"""
multinomial(input, num_samples, replacement=False, out=None) -> LongTensor

Returns a Tensor where each row
contains :attr:`num_samples` indices sampled from the multinomial probability distribution
located in the corresponding row of Tensor :attr:`input`.

.. note::
    The rows of :attr:`input` do not need to sum to one (in which case we use the values
    as weights), but must be non-negative and have a non-zero sum.

Indices are ordered from left to right according to when each was sampled
(first samples are placed in first column).

If :attr:`input` is a vector, :attr:`out` is a matrix of size `num_samples`.

If :attr:`input` is a matrix with `m` rows, :attr:`out` is an matrix of shape `m \u00D7 n`.

If replacement is `True`, samples are drawn with replacement.

If not, they are drawn without replacement, which means that when a
sample index is drawn for a row, it cannot be drawn again for that row.

This implies the constraint that :attr:`num_samples` must be lower than :attr:`input` length
(or number of columns of :attr:`input` if it is a matrix).

Args:
    input (Tensor): Tensor containing probabilities
    num_samples (int): number of samples to draw
    replacement (bool, optional): Whether to draw with replacement or not
    out (Tensor, optional): The result `Tensor`

Example::

    >>> weights = torch.Tensor([0, 10, 3, 0]) # create a Tensor of weights
    >>> torch.multinomial(weights, 4)

     1
     2
     0
     0
    [torch.LongTensor of size 4]

    >>> torch.multinomial(weights, 4, replacement=True)

     1
     2
     1
     2
    [torch.LongTensor of size 4]

""")

add_docstr(torch._C.mv,
           """
mv(mat, vec, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`mat` and the vector :attr:`vec`.

If :attr:`mat` is a `n x m` Tensor, :attr:`vec` is a 1D Tensor of size `m`, :attr:`out` will be 1D of size `n`.

Args:
    mat (Tensor): matrix to be multiplied
    vec (Tensor): vector to be multiplied
    out (Tensor, optional): Output tensor

Example::

    >>> mat = torch.randn(2, 3)
    >>> vec = torch.randn(3)
    >>> torch.mv(mat, vec)
    -2.0939
    -2.2950
    [torch.FloatTensor of size 2]
""")

add_docstr(torch._C.ne,
           """
ne(input, other, out=None) -> Tensor

Computes `tensor != other` element-wise.

The second argument can be a number or a tensor of the same shape and
type as the first argument.

Args:
    input (Tensor): Tensor to compare
    other (Tensor or float): Tensor or value to compare
    out (Tensor, optional): Output tensor. Must be a `ByteTensor` or the same type as `tensor`.

Returns:
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where comparison is true.

Example::

    >>> torch.ne(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 1], [4, 4]]))
     0  1
     1  0
    [torch.ByteTensor of size 2x2]
""")

add_docstr(torch._C.neg,
           """
neg(input, out=None) -> Tensor

Returns a new `Tensor` with the negative of the elements of :attr:`input`.

:math:`out = -1 * input`

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(5)
    >>> a

    -0.4430
     1.1690
    -0.8836
    -0.4565
     0.2968
    [torch.FloatTensor of size 5]

    >>> torch.neg(a)

     0.4430
    -1.1690
     0.8836
     0.4565
    -0.2968
    [torch.FloatTensor of size 5]

""")

add_docstr(torch._C.nonzero,
           """
nonzero(input, out=None) -> LongTensor

Returns a tensor containing the indices of all non-zero elements of :attr:`input`.
Each row in the result contains the indices of a non-zero element in :attr:`input`.

If :attr:`input` has `n` dimensions, then the resulting indices Tensor
:attr:`out` is of size `z x n`, where `z` is the total number of non-zero
elements in the :attr:`input` Tensor.

Args:
    input (Tensor): the input `Tensor`
    out (LongTensor, optional): The result `Tensor` containing indices

Example::

    >>> torch.nonzero(torch.Tensor([1, 1, 1, 0, 1]))

     0
     1
     2
     4
    [torch.LongTensor of size 4x1]

    >>> torch.nonzero(torch.Tensor([[0.6, 0.0, 0.0, 0.0],
    ...                             [0.0, 0.4, 0.0, 0.0],
    ...                             [0.0, 0.0, 1.2, 0.0],
    ...                             [0.0, 0.0, 0.0,-0.4]]))

     0  0
     1  1
     2  2
     3  3
    [torch.LongTensor of size 4x2]

""")

add_docstr(torch._C.norm,
           """
.. function:: norm(input, p=2) -> float

Returns the p-norm of the :attr:`input` Tensor.

Args:
    input (Tensor): the input `Tensor`
    p (float, optional): the exponent value in the norm formulation
Example::

    >>> a = torch.randn(1, 3)
    >>> a

    -0.4376 -0.5328  0.9547
    [torch.FloatTensor of size 1x3]

    >>> torch.norm(a, 3)
    1.0338925067372466


.. function:: norm(input, p, dim, out=None) -> Tensor

Returns the p-norm of each row of the :attr:`input` Tensor in the given dimension :attr:`dim`.

The output Tensor is of the same size as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.

Args:
    input (Tensor): the input `Tensor`
    p (float):  the exponent value in the norm formulation
    dim (int): the dimension to reduce
    out (Tensor, optional): the result Tensor

Example::

    >>> a = torch.randn(4, 2)
    >>> a

    -0.6891 -0.6662
     0.2697  0.7412
     0.5254 -0.7402
     0.5528 -0.2399
    [torch.FloatTensor of size 4x2]

    >>> torch.norm(a, 2, 1)

     0.9585
     0.7888
     0.9077
     0.6026
    [torch.FloatTensor of size 4x1]

    >>> torch.norm(a, 0, 1)

     2
     2
     2
     2
    [torch.FloatTensor of size 4x1]

""")

add_docstr(torch._C.normal,
           """
.. function:: normal(means, std, out=None)

Returns a Tensor of random numbers drawn from separate normal distributions
who's mean and standard deviation are given.

The :attr:`means` is a Tensor with the mean of
each output element's normal distribution

The :attr:`std` is a Tensor with the standard deviation of
each output element's normal distribution

The shapes of :attr:`means` and :attr:`std` don't need to match.
The total number of elements in each Tensor need to be the same.

.. note:: When the shapes do not match, the shape of :attr:`means`
          is used as the shape for the returned output Tensor

Args:
    means (Tensor): the Tensor of per-element means
    std (Tensor): the Tensor of per-element standard deviations
    out (Tensor): the optional result Tensor

Example::

    torch.normal(means=torch.range(1, 10), std=torch.range(1, 0.1, -0.1))

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
    [torch.FloatTensor of size 10]

.. function:: normal(mean=0.0, std, out=None)

Similar to the function above, but the means are shared among all drawn elements.

Args:
    means (float, optional): the mean for all distributions
    std (Tensor): the Tensor of per-element standard deviations
    out (Tensor): the optional result Tensor

Example::

    >>> torch.normal(mean=0.5, std=torch.range(1, 5))

      0.5723
      0.0871
     -0.3783
     -2.5689
     10.7893
    [torch.FloatTensor of size 5]

.. function:: normal(means, std=1.0, out=None)

Similar to the function above, but the standard-deviations are shared among all drawn elements.

Args:
    means (Tensor): the Tensor of per-element means
    std (float, optional): the standard deviation for all distributions
    out (Tensor): the optional result Tensor

Example::

    >>> torch.normal(means=torch.range(1, 5))

     1.1681
     2.8884
     3.7718
     2.5616
     4.2500
    [torch.FloatTensor of size 5]

""")

add_docstr(torch._C.numel,
           """
numel(input) -> int

Returns the total number of elements in the :attr:`input` Tensor.

Args:
    input (Tensor): the input `Tensor`

Example::

    >>> a = torch.randn(1,2,3,4,5)
    >>> torch.numel(a)
    120
    >>> a = torch.zeros(4,4)
    >>> torch.numel(a)
    16

""")

add_docstr(torch._C.ones,
           """
ones(*sizes, out=None) -> Tensor

Returns a Tensor filled with the scalar value `1`, with the shape defined
by the varargs :attr:`sizes`.

Args:
    sizes (int...): a set of ints defining the shape of the output Tensor.
    out (Tensor, optional): the result Tensor

Example::

    >>> torch.ones(2, 3)

     1  1  1
     1  1  1
    [torch.FloatTensor of size 2x3]

    >>> torch.ones(5)

     1
     1
     1
     1
     1
    [torch.FloatTensor of size 5]

""")

# TODO
# add_docstr(torch._C.orgqr,
# """
# """)

# add_docstr(torch._C.ormqr,
# """
# """)

# add_docstr(torch._C.potrf,
# """
# """)

# add_docstr(torch._C.potri,
# """
# """)

# add_docstr(torch._C.potrs,
# """
# """)

add_docstr(torch._C.pow,
           """
.. function:: pow(input, exponent, out=None)

Takes the power of each element in :attr:`input` with :attr:`exponent` and returns a Tensor with the result.

:attr:`exponent` can be either a single ``float`` number or a ``Tensor``
with the same number of elements as :attr:`input`.

When :attr:`exponent` is a scalar value, the operation applied is:

:math:`out_i = x_i ^ {exponent}`

When :attr:`exponent` is a Tensor, the operation applied is:

:math:`out_i = x_i ^ {exponent_i}`

Args:
    input (Tensor): the input `Tensor`
    exponent (float or Tensor): the exponent value
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

    -0.5274
    -0.8232
    -2.1128
     1.7558
    [torch.FloatTensor of size 4]

    >>> torch.pow(a, 2)

     0.2781
     0.6776
     4.4640
     3.0829
    [torch.FloatTensor of size 4]

    >>> exp = torch.range(1, 4)
    >>> a = torch.range(1, 4)
    >>> a

     1
     2
     3
     4
    [torch.FloatTensor of size 4]

    >>> exp

     1
     2
     3
     4
    [torch.FloatTensor of size 4]

    >>> torch.pow(a, exp)

       1
       4
      27
     256
    [torch.FloatTensor of size 4]


.. function:: pow(base, input, out=None)

:attr:`base` is a scalar ``float`` value, and :attr:`input` is a Tensor.
The returned Tensor :attr:`out` is of the same shape as :attr:`input`

The operation applied is:

:math:`out_i = base ^ {input_i}`

Args:
    base (float): the scalar base value for the power operation
    input (Tensor): the exponent `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> exp = torch.range(1, 4)
    >>> base = 2
    >>> torch.pow(base, exp)

      2
      4
      8
     16
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.prod,
           """
.. function:: prod(input) -> float

Returns the product of all elements in the :attr:`input` Tensor.

Args:
    input (Tensor): the input `Tensor`

Example::

    >>> a = torch.randn(1, 3)
    >>> a

     0.6170  0.3546  0.0253
    [torch.FloatTensor of size 1x3]

    >>> torch.prod(a)
    0.005537458061418483


.. function:: prod(input, dim, out=None) -> Tensor

Returns the product of each row of the :attr:`input` Tensor in the given dimension :attr:`dim`.

The output Tensor is of the same size as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.

Args:
    input (Tensor): the input `Tensor`
    dim (int): the dimension to reduce
    out (Tensor, optional): the result Tensor

Example::

    >>> a = torch.randn(4, 2)
    >>> a

     0.1598 -0.6884
    -0.1831 -0.4412
    -0.9925 -0.6244
    -0.2416 -0.8080
    [torch.FloatTensor of size 4x2]

    >>> torch.prod(a, 1)

    -0.1100
     0.0808
     0.6197
     0.1952
    [torch.FloatTensor of size 4x1]

""")

# TODO
# add_docstr(torch._C.pstrf,
# """
# """)

add_docstr(torch._C.qr,
           """
qr(input, out=None) -> (Tensor, Tensor)

Computes the QR decomposition of a matrix :attr:`input`: returns matrices
`q` and `r` such that :math:`x = q * r`, with `q` being an orthogonal matrix
and `r` being an upper triangular matrix.

This returns the thin (reduced) QR factorization.

.. note:: precision may be lost if the magnitudes of the elements of `input` are large

.. note:: while it should always give you a valid decomposition, it may not
          give you the same one across platforms - it will depend on your
          LAPACK implementation.

.. note:: Irrespective of the original strides, the returned matrix `q` will be
          transposed, i.e. with strides `(1, m)` instead of `(m, 1)`.

Args:
    input (Tensor): the input 2D `Tensor`
    out (tuple, optional): A tuple of Q and R Tensors

Example::

    >>> a = torch.Tensor([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
    >>> q, r = torch.qr(a)
    >>> q

    -0.8571  0.3943  0.3314
    -0.4286 -0.9029 -0.0343
     0.2857 -0.1714  0.9429
    [torch.FloatTensor of size 3x3]

    >>> r

     -14.0000  -21.0000   14.0000
       0.0000 -175.0000   70.0000
       0.0000    0.0000  -35.0000
    [torch.FloatTensor of size 3x3]

    >>> torch.mm(q, r).round()

      12  -51    4
       6  167  -68
      -4   24  -41
    [torch.FloatTensor of size 3x3]

    >>> torch.mm(q.t(), q).round()

     1 -0  0
    -0  1  0
     0  0  1
    [torch.FloatTensor of size 3x3]

""")

add_docstr(torch._C.rand,
           """
rand(*sizes, out=None) -> Tensor

Returns a Tensor filled with random numbers from a uniform distribution
on the interval :math:`[0, 1)`

The shape of the Tensor is defined by the varargs :attr:`sizes`.

Args:
    sizes (int...): a set of ints defining the shape of the output Tensor.
    out (Tensor, optional): the result Tensor

Example::

    >>> torch.rand(4)

     0.9193
     0.3347
     0.3232
     0.7715
    [torch.FloatTensor of size 4]

    >>> torch.rand(2, 3)

     0.5010  0.5140  0.0719
     0.1435  0.5636  0.0538
    [torch.FloatTensor of size 2x3]

""")

add_docstr(torch._C.randn,
           """
randn(*sizes, out=None) -> Tensor

Returns a Tensor filled with random numbers from a normal distribution
with zero mean and variance of one.

The shape of the Tensor is defined by the varargs :attr:`sizes`.

Args:
    sizes (int...): a set of ints defining the shape of the output Tensor.
    out (Tensor, optional): the result Tensor

Example::

    >>> torch.randn(4)

    -0.1145
     0.0094
    -1.1717
     0.9846
    [torch.FloatTensor of size 4]

    >>> torch.randn(2, 3)

     1.4339  0.3351 -1.0999
     1.5458 -0.9643 -0.3558
    [torch.FloatTensor of size 2x3]

""")

add_docstr(torch._C.randperm,
           """
randperm(n, out=None) -> LongTensor

Returns a random permutation of integers from ``0`` to ``n - 1``.

Args:
    n (int): the upper bound (exclusive)

Example::

    >>> torch.randperm(4)

     2
     1
     3
     0
    [torch.LongTensor of size 4]
""")

add_docstr(torch._C.range,
           """
range(start, end, step=1, out=None) -> Tensor

returns a 1D Tensor of size :math:`floor((end - start) / step) + 1` with values
from :attr:`start` to :attr:`end` with step :attr:`step`. Step is the gap between two values in the tensor.
:math:`x_{i+1} = x_i + step`

Args:
    start (float): The starting value for the set of points
    end (float): The ending value for the set of points
    step (float): The gap between each pair of adjacent points
    out (Tensor, optional): The result `Tensor`

Example::

    >>> torch.range(1, 4)

     1
     2
     3
     4
    [torch.FloatTensor of size 4]

    >>> torch.range(1, 4, 0.5)

     1.0000
     1.5000
     2.0000
     2.5000
     3.0000
     3.5000
     4.0000
    [torch.FloatTensor of size 7]

""")

add_docstr(torch._C.remainder,
           """
remainder(input, divisor, out=None) -> Tensor

Computes the element-wise remainder of division.

The divisor and dividend may contain both for integer and floating point
numbers. The remainder has the same sign as the divisor.

Args:
    input (Tensor): The dividend
    divisor (Tensor or float): The divisor. This may be either a number or a
                               tensor of the same shape as the dividend.
    out (Tensor, optional): Output tensor

Example::

    >>> torch.remainder(torch.Tensor([-3, -2, -1, 1, 2, 3]), 2)
    torch.FloatTensor([1, 0, 1, 1, 0, 1])
    >>> torch.remainder(torch.Tensor([1, 2, 3, 4, 5]), 1.5)
    torch.FloatTensor([1.0, 0.5, 0.0, 1.0, 0.5])

.. seealso::

        :func:`torch.fmod`, which computes the element-wise remainder of
        division equivalently to the C library function ``fmod()``
""")

add_docstr(torch._C.renorm,
           """
renorm(input, p, dim, maxnorm, out=None) -> Tensor

Returns a Tensor where each sub-tensor of :attr:`input` along dimension :attr:`dim`
is normalized such that the `p`-norm of the sub-tensor is lower than the value :attr:`maxnorm`

.. note:: If the norm of a row is lower than `maxnorm`, the row is unchanged

Args:
    input (Tensor): The input Tensor
    p (float): The power for the norm computation
    dim (int): The dimension to slice over to get the sub-tensors
    maxnorm (float): The maximum norm to keep each sub-tensor under
    out (Tensor, optional): Output tensor

Example::

    >>> x = torch.ones(3, 3)
    >>> x[1].fill_(2)
    >>> x[2].fill_(3)
    >>> x

     1  1  1
     2  2  2
     3  3  3
    [torch.FloatTensor of size 3x3]

    >>> torch.renorm(x, 1, 0, 5)

     1.0000  1.0000  1.0000
     1.6667  1.6667  1.6667
     1.6667  1.6667  1.6667
    [torch.FloatTensor of size 3x3]

""")

add_docstr(torch._C.round,
           """
round(input, out=None) -> Tensor

Returns a new `Tensor` with each of the elements of :attr:`input` rounded to the closest integer.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.2290
     1.3409
    -0.5662
    -0.0899
    [torch.FloatTensor of size 4]

    >>> torch.round(a)

     1
     1
    -1
    -0
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.rsqrt,
           """
rsqrt(input, out=None) -> Tensor

Returns a new `Tensor` with the reciprocal of the square-root of each of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.2290
     1.3409
    -0.5662
    -0.0899
    [torch.FloatTensor of size 4]

    >>> torch.rsqrt(a)

     0.9020
     0.8636
        nan
        nan
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.set_num_threads,
           """
set_num_threads(int)

Sets the number of OpenMP threads used for parallelizing CPU operations
""")

add_docstr(torch._C.sigmoid,
           """
sigmoid(input, out=None) -> Tensor

Returns a new `Tensor` with the sigmoid of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

    -0.4972
     1.3512
     0.1056
    -0.2650
    [torch.FloatTensor of size 4]

    >>> torch.sigmoid(a)

     0.3782
     0.7943
     0.5264
     0.4341
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.sign,
           """
sign(input, out=None) -> Tensor

Returns a new `Tensor` with the sign of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.sign(a)

    -1
     1
     1
     1
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.sin,
           """
sin(input, out=None) -> Tensor

Returns a new `Tensor` with the sine of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.sin(a)
    -0.5944
     0.2684
     0.4322
     0.9667
    [torch.FloatTensor of size 4]
""")

add_docstr(torch._C.sinh,
           """
sinh(input, out=None) -> Tensor

Returns a new `Tensor` with the hyperbolic sine of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.sinh(a)
    -0.6804
     0.2751
     0.4619
     1.7225
    [torch.FloatTensor of size 4]
""")

add_docstr(torch._C.sort,
           """
sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor)

Sorts the elements of the :attr:`input` Tensor along a given dimension in ascending order by value.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`descending` is `True` then the elements are sorted in descending order by value.

A tuple of (sorted_tensor, sorted_indices) is returned, where the sorted_indices are the
indices of the elements in the original `input` Tensor.

Args:
    input (Tensor): the input `Tensor`
    dim (int, optional): The dimension to sort along
    descending (bool, optional): Controls the sorting order (ascending or descending)
    out (tuple, optional): The output tuple of (Tensor, LongTensor)
                           can be optionally given to be used as output buffers

Example::

    >>> x = torch.randn(3, 4)
    >>> sorted, indices = torch.sort(x)
    >>> sorted

    -1.6747  0.0610  0.1190  1.4137
    -1.4782  0.7159  1.0341  1.3678
    -0.3324 -0.0782  0.3518  0.4763
    [torch.FloatTensor of size 3x4]

    >>> indices

     0  1  3  2
     2  1  0  3
     3  1  0  2
    [torch.LongTensor of size 3x4]

    >>> sorted, indices = torch.sort(x, 0)
    >>> sorted

    -1.6747 -0.0782 -1.4782 -0.3324
     0.3518  0.0610  0.4763  0.1190
     1.0341  0.7159  1.4137  1.3678
    [torch.FloatTensor of size 3x4]

    >>> indices

     0  2  1  2
     2  0  2  0
     1  1  0  1
    [torch.LongTensor of size 3x4]

""")

add_docstr(torch._C.sqrt,
           """
sqrt(input, out=None) -> Tensor

Returns a new `Tensor` with the square-root of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

     1.2290
     1.3409
    -0.5662
    -0.0899
    [torch.FloatTensor of size 4]

    >>> torch.sqrt(a)

     1.1086
     1.1580
        nan
        nan
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.squeeze,
           """
squeeze(input, dim=None, out=None)

Returns a `Tensor` with all the dimensions of :attr:`input` of size `1` removed.

If `input` is of shape: :math:`(A x 1 x B x C x 1 x D)` then the `out` Tensor
will be of shape: :math:`(A x B x C x D)`

When :attr:`dim` is given, a squeeze operation is done only in the given dimension.
If `input` is of shape: :math:`(A x 1 x B)`, `squeeze(input, 0)` leaves the Tensor unchanged,
but `squeeze(input, 1)` will squeeze the tensor to the shape :math:`(A x B)`.

.. note:: The returned Tensor shares the storage with the input Tensor,
          so changing the contents of one will change the contents of the other.

Args:
    input (Tensor): the input `Tensor`
    dim (int, optional): if given, the input will be squeezed only in this dimension
    out (Tensor, optional): The result `Tensor`

Example::

    >>> x = torch.zeros(2,1,2,1,2)
    >>> x.size()
    (2L, 1L, 2L, 1L, 2L)
    >>> y = torch.squeeze(x)
    >>> y.size()
    (2L, 2L, 2L)
    >>> y = torch.squeeze(x, 0)
    >>> y.size()
    (2L, 1L, 2L, 1L, 2L)
    >>> y = torch.squeeze(x, 1)
    >>> y.size()
    (2L, 2L, 1L, 2L)
""")

add_docstr(torch._C.std,
           """
.. function:: std(input) -> float

Returns the standard-deviation of all elements in the :attr:`input` Tensor.

Args:
    input (Tensor): the input `Tensor`

Example::

    >>> a = torch.randn(1, 3)
    >>> a

    -1.3063  1.4182 -0.3061
    [torch.FloatTensor of size 1x3]

    >>> torch.std(a)
    1.3782334731508061


.. function:: std(input, dim, out=None) -> Tensor

Returns the standard-deviation of each row of the :attr:`input` Tensor in the given dimension :attr:`dim`.

The output Tensor is of the same size as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.

Args:
    input (Tensor): the input `Tensor`
    dim (int): the dimension to reduce
    out (Tensor, optional): the result Tensor

Example::

    >>> a = torch.randn(4, 4)
    >>> a

     0.1889 -2.4856  0.0043  1.8169
    -0.7701 -0.4682 -2.2410  0.4098
     0.1919 -1.1856 -1.0361  0.9085
     0.0173  1.0662  0.2143 -0.5576
    [torch.FloatTensor of size 4x4]

    >>> torch.std(a, dim=1)

     1.7756
     1.1025
     1.0045
     0.6725
    [torch.FloatTensor of size 4x1]

""")

add_docstr(torch._C.sum,
           """
.. function:: sum(input) -> float

Returns the sum of all elements in the :attr:`input` Tensor.

Args:
    input (Tensor): the input `Tensor`

Example::

    >>> a = torch.randn(1, 3)
    >>> a

     0.6170  0.3546  0.0253
    [torch.FloatTensor of size 1x3]

    >>> torch.sum(a)
    0.9969287421554327


.. function:: sum(input, dim, out=None) -> Tensor

Returns the sum of each row of the :attr:`input` Tensor in the given dimension :attr:`dim`.

The output Tensor is of the same size as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.

Args:
    input (Tensor): the input `Tensor`
    dim (int): the dimension to reduce
    out (Tensor, optional): the result Tensor

Example::

    >>> a = torch.randn(4, 4)
    >>> a

    -0.4640  0.0609  0.1122  0.4784
    -1.3063  1.6443  0.4714 -0.7396
    -1.3561 -0.1959  1.0609 -1.9855
     2.6833  0.5746 -0.5709 -0.4430
    [torch.FloatTensor of size 4x4]

    >>> torch.sum(a, 1)

     0.1874
     0.0698
    -2.4767
     2.2440
    [torch.FloatTensor of size 4x1]

""")

add_docstr(torch._C.svd,
           """
svd(input, some=True, out=None) -> (Tensor, Tensor, Tensor)

`U, S, V = torch.svd(A)` returns the singular value decomposition of a
real matrix `A` of size `(n x m)` such that :math:`A = USV'*`.

`U` is of shape `n x n`

`S` is of shape `n x m`

`V` is of shape `m x m`.

:attr:`some` represents the number of singular values to be computed.
If `some=True`, it computes some and `some=False` computes all.

.. note:: Irrespective of the original strides, the returned matrix `U`
          will be transposed, i.e. with strides `(1, n)` instead of `(n, 1)`.

Args:
    input (Tensor): the input 2D Tensor
    some (bool, optional): controls the number of singular values to be computed
    out (tuple, optional): the result tuple

Example::

    >>> a = torch.Tensor([[8.79,  6.11, -9.15,  9.57, -3.49,  9.84],
    ...                   [9.93,  6.91, -7.93,  1.64,  4.02,  0.15],
    ...                   [9.83,  5.04,  4.86,  8.83,  9.80, -8.99],
    ...                   [5.45, -0.27,  4.85,  0.74, 10.00, -6.02],
    ...                   [3.16,  7.98,  3.01,  5.80,  4.27, -5.31]]).t()
    >>> a

      8.7900   9.9300   9.8300   5.4500   3.1600
      6.1100   6.9100   5.0400  -0.2700   7.9800
     -9.1500  -7.9300   4.8600   4.8500   3.0100
      9.5700   1.6400   8.8300   0.7400   5.8000
     -3.4900   4.0200   9.8000  10.0000   4.2700
      9.8400   0.1500  -8.9900  -6.0200  -5.3100
    [torch.FloatTensor of size 6x5]

    >>> u, s, v = torch.svd(a)
    >>> u

    -0.5911  0.2632  0.3554  0.3143  0.2299
    -0.3976  0.2438 -0.2224 -0.7535 -0.3636
    -0.0335 -0.6003 -0.4508  0.2334 -0.3055
    -0.4297  0.2362 -0.6859  0.3319  0.1649
    -0.4697 -0.3509  0.3874  0.1587 -0.5183
     0.2934  0.5763 -0.0209  0.3791 -0.6526
    [torch.FloatTensor of size 6x5]

    >>> s

     27.4687
     22.6432
      8.5584
      5.9857
      2.0149
    [torch.FloatTensor of size 5]

    >>> v

    -0.2514  0.8148 -0.2606  0.3967 -0.2180
    -0.3968  0.3587  0.7008 -0.4507  0.1402
    -0.6922 -0.2489 -0.2208  0.2513  0.5891
    -0.3662 -0.3686  0.3859  0.4342 -0.6265
    -0.4076 -0.0980 -0.4932 -0.6227 -0.4396
    [torch.FloatTensor of size 5x5]

    >>> torch.dist(a, torch.mm(torch.mm(u, torch.diag(s)), v.t()))
    8.934150226306685e-06

""")

add_docstr(torch._C.symeig,
           """
symeig(input, eigenvectors=False, upper=True, out=None) -> (Tensor, Tensor)

`e, V = torch.symeig(input)` returns eigenvalues and eigenvectors
of a symmetric real matrix :attr:`input`.

`input` and `V` are `m x m` matrices and `e` is a `m` dimensional vector.

This function calculates all eigenvalues (and vectors) of `input`
such that `input = V diag(e) V'`

The boolean argument :attr:`eigenvectors` defines computation of
eigenvectors or eigenvalues only.

If it is `False`, only eigenvalues are computed. If it is `True`,
both eigenvalues and eigenvectors are computed.

Since the input matrix `input` is supposed to be symmetric,
only the upper triangular portion is used by default.

If :attr:`upper` is `False`, then lower triangular portion is used.

Note: Irrespective of the original strides, the returned matrix `V` will
be transposed, i.e. with strides `(1, m)` instead of `(m, 1)`.

Args:
    input (Tensor): the input symmetric matrix
    eigenvectors(boolean, optional): controls whether eigenvectors have to be computed
    upper(boolean, optional): controls whether to consider upper-triangular or lower-triangular region
    out (tuple, optional): The result tuple of (Tensor, Tensor)

Examples::


    >>> a = torch.Tensor([[ 1.96,  0.00,  0.00,  0.00,  0.00],
    ...                   [-6.49,  3.80,  0.00,  0.00,  0.00],
    ...                   [-0.47, -6.39,  4.17,  0.00,  0.00],
    ...                   [-7.20,  1.50, -1.51,  5.70,  0.00],
    ...                   [-0.65, -6.34,  2.67,  1.80, -7.10]]).t()

    >>> e, v = torch.symeig(a, eigenvectors=True)
    >>> e

    -11.0656
     -6.2287
      0.8640
      8.8655
     16.0948
    [torch.FloatTensor of size 5]

    >>> v

    -0.2981 -0.6075  0.4026 -0.3745  0.4896
    -0.5078 -0.2880 -0.4066 -0.3572 -0.6053
    -0.0816 -0.3843 -0.6600  0.5008  0.3991
    -0.0036 -0.4467  0.4553  0.6204 -0.4564
    -0.8041  0.4480  0.1725  0.3108  0.1622
    [torch.FloatTensor of size 5x5]

""")

add_docstr(torch._C.t,
           """
t(input, out=None) -> Tensor

Expects :attr:`input` to be a matrix (2D Tensor) and transposes dimensions 0 and 1.

Can be seen as a short-hand function for `transpose(input, 0, 1)`

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> x = torch.randn(2, 3)
    >>> x

     0.4834  0.6907  1.3417
    -0.1300  0.5295  0.2321
    [torch.FloatTensor of size 2x3]

    >>> torch.t(x)

     0.4834 -0.1300
     0.6907  0.5295
     1.3417  0.2321
    [torch.FloatTensor of size 3x2]

""")

add_docstr(torch._C.tan,
           """
tan(input, out=None) -> Tensor

Returns a new `Tensor` with the tangent of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.tan(a)
    -0.7392
     0.2786
     0.4792
     3.7801
    [torch.FloatTensor of size 4]
""")

add_docstr(torch._C.tanh,
           """
tanh(input, out=None) -> Tensor

Returns a new `Tensor` with the hyperbolic tangent of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    -0.6366
     0.2718
     0.4469
     1.3122
    [torch.FloatTensor of size 4]

    >>> torch.tanh(a)
    -0.5625
     0.2653
     0.4193
     0.8648
    [torch.FloatTensor of size 4]
""")

add_docstr(torch._C.topk,
           """
topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)

Returns the :attr:`k` largest elements of the given :attr:`input` Tensor along a given dimension.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`largest` is `False` then the `k` smallest elements are returned.

A tuple of `(values, indices)` is returned, where the `indices` are the indices of
the elements in the original `input` Tensor.

The boolean option :attr:`sorted` if `True`, will make sure that the returned `k`
elements are themselves sorted

Args:
    input (Tensor): the input `Tensor`
    k (int): the k in "top-k"
    dim (int, optional): The dimension to sort along
    largest (bool, optional): Controls whether to return largest or smallest elements
    sorted (bool, optional): Controls whether to return the elements in sorted order
    out (tuple, optional): The output tuple of (Tensor, LongTensor)
                           can be optionally given to be used as output buffers

Example::

    >>> x = torch.range(1, 5)
    >>> x

     1
     2
     3
     4
     5
    [torch.FloatTensor of size 5]

    >>> torch.topk(x, 3)
    (
     5
     4
     3
    [torch.FloatTensor of size 3]
    ,
     4
     3
     2
    [torch.LongTensor of size 3]
    )
    >>> torch.topk(x, 3, 0, largest=False)
    (
     1
     2
     3
    [torch.FloatTensor of size 3]
    ,
     0
     1
     2
    [torch.LongTensor of size 3]
    )

""")

add_docstr(torch._C.trace,
           """
trace(input) -> float

Returns the sum of the elements of the diagonal of the input 2D matrix.

Example::

    >>> x = torch.range(1, 9).view(3, 3)
    >>> x

     1  2  3
     4  5  6
     7  8  9
    [torch.FloatTensor of size 3x3]

    >>> torch.trace(x)
    15.0

""")

add_docstr(torch._C.transpose,
           """
transpose(input, dim0, dim1, out=None) -> Tensor

Returns a `Tensor` that is a transposed version of :attr:`input`.
The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

The resulting :attr:`out` Tensor shares it's underlying storage with the
:attr:`input` Tensor, so changing the content of one would change the content
of the other.

Args:
    input (Tensor): the input `Tensor`
    dim0 (int): The first dimension to be transposed
    dim1 (int): The second dimension to be transposed

Example::

    >>> x = torch.randn(2, 3)
    >>> x

     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size 2x3]

    >>> torch.transpose(x, 0, 1)

     0.5983  1.5981
    -0.0341 -0.5265
     2.4918 -0.8735
    [torch.FloatTensor of size 3x2]

""")

add_docstr(torch._C.tril,
           """
tril(input, k=0, out=None) -> Tensor

Returns the lower triangular part of the matrix (2D Tensor) :attr:`input`,
the other elements of the result Tensor :attr:`out` are set to 0.

The lower triangular part of the matrix is defined as the elements on and below the diagonal.

The argument :attr:`k` controls which diagonal to consider.

- :attr:`k` = 0, is the main diagonal.
- :attr:`k` > 0, is above the main diagonal.
- :attr:`k` < 0, is below the main diagonal.

Args:
    input (Tensor): the input `Tensor`
    k (int, optional): the diagonal to consider
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(3,3)
    >>> a

     1.3225  1.7304  1.4573
    -0.3052 -0.3111 -0.1809
     1.2469  0.0064 -1.6250
    [torch.FloatTensor of size 3x3]

    >>> torch.tril(a)

     1.3225  0.0000  0.0000
    -0.3052 -0.3111  0.0000
     1.2469  0.0064 -1.6250
    [torch.FloatTensor of size 3x3]

    >>> torch.tril(a, k=1)

     1.3225  1.7304  0.0000
    -0.3052 -0.3111 -0.1809
     1.2469  0.0064 -1.6250
    [torch.FloatTensor of size 3x3]

    >>> torch.tril(a, k=-1)

     0.0000  0.0000  0.0000
    -0.3052  0.0000  0.0000
     1.2469  0.0064  0.0000
    [torch.FloatTensor of size 3x3]

""")

add_docstr(torch._C.triu,
           """
triu(input, k=0, out=None) -> Tensor

Returns the upper triangular part of the matrix (2D Tensor) :attr:`input`,
the other elements of the result Tensor :attr:`out` are set to 0.

The upper triangular part of the matrix is defined as the elements on and above the diagonal.

The argument :attr:`k` controls which diagonal to consider.

- :attr:`k` = 0, is the main diagonal.
- :attr:`k` > 0, is above the main diagonal.
- :attr:`k` < 0, is below the main diagonal.

Args:
    input (Tensor): the input `Tensor`
    k (int, optional): the diagonal to consider
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(3,3)
    >>> a

     1.3225  1.7304  1.4573
    -0.3052 -0.3111 -0.1809
     1.2469  0.0064 -1.6250
    [torch.FloatTensor of size 3x3]

    >>> torch.triu(a)

     1.3225  1.7304  1.4573
     0.0000 -0.3111 -0.1809
     0.0000  0.0000 -1.6250
    [torch.FloatTensor of size 3x3]

    >>> torch.triu(a, k=1)

     0.0000  1.7304  1.4573
     0.0000  0.0000 -0.1809
     0.0000  0.0000  0.0000
    [torch.FloatTensor of size 3x3]

    >>> torch.triu(a, k=-1)

     1.3225  1.7304  1.4573
    -0.3052 -0.3111 -0.1809
     0.0000  0.0064 -1.6250
    [torch.FloatTensor of size 3x3]

""")

# TODO
# add_docstr(torch._C.trtrs,
# """
# """)

add_docstr(torch._C.trunc,
           """
trunc(input, out=None) -> Tensor

Returns a new `Tensor` with the truncated integer values of the elements of :attr:`input`.

Args:
    input (Tensor): the input `Tensor`
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a

    -0.4972
     1.3512
     0.1056
    -0.2650
    [torch.FloatTensor of size 4]

    >>> torch.trunc(a)

    -0
     1
     0
    -0
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.unsqueeze,
           """
unsqueeze(input, dim, out=None)

Returns a new tensor with a dimension of size one inserted at the
specified position.

The returned tensor shares the same underlying data with this tensor.

Args:
    input (Tensor): the input `Tensor`
    dim (int): The index at which to insert the singleton dimension
    out (Tensor, optional): The result `Tensor`

Example:
    >>> x = torch.Tensor([1, 2, 3, 4])
    >>> torch.unsqueeze(x, 0)
     1  2  3  4
    [torch.FloatTensor of size 1x4]
    >>> torch.unsqueeze(x, 1)
     1
     2
     3
     4
    [torch.FloatTensor of size 4x1]
""")

add_docstr(torch._C.var,
           """
.. function:: var(input) -> float

Returns the variance of all elements in the :attr:`input` Tensor.

Args:
    input (Tensor): the input `Tensor`

Example::

    >>> a = torch.randn(1, 3)
    >>> a

    -1.3063  1.4182 -0.3061
    [torch.FloatTensor of size 1x3]

    >>> torch.var(a)
    1.899527506513334


.. function:: var(input, dim, out=None) -> Tensor

Returns the variance of each row of the :attr:`input` Tensor in the given dimension :attr:`dim`.

The output Tensor is of the same size as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.

Args:
    input (Tensor): the input `Tensor`
    dim (int): the dimension to reduce
    out (Tensor, optional): the result Tensor

Example::

    >>> a = torch.randn(4, 4)
    >>> a

    -1.2738 -0.3058  0.1230 -1.9615
     0.8771 -0.5430 -0.9233  0.9879
     1.4107  0.0317 -0.6823  0.2255
    -1.3854  0.4953 -0.2160  0.2435
    [torch.FloatTensor of size 4x4]

    >>> torch.var(a, 1)

     0.8859
     0.9509
     0.7548
     0.6949
    [torch.FloatTensor of size 4x1]

""")

add_docstr(torch._C.zeros,
           """
zeros(*sizes, out=None) -> Tensor

Returns a Tensor filled with the scalar value `0`, with the shape defined
by the varargs :attr:`sizes`.

Args:
    sizes (int...): a set of ints defining the shape of the output Tensor.
    out (Tensor, optional): the result Tensor

Example::

    >>> torch.zeros(2, 3)

     0  0  0
     0  0  0
    [torch.FloatTensor of size 2x3]

    >>> torch.zeros(5)

     0
     0
     0
     0
     0
    [torch.FloatTensor of size 5]

""")
