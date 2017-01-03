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

Adds the scalar :attr:`value` to each element of the input :attr:`input` and returns a new resulting tensor.

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

Each element of the Tensor :attr:`other` is multiplied by the scalar :attr:`value` and added to each element of the Tensor :attr:`input`. The resulting Tensor is returned.
The shapes of :attr:`input` and :attr:`other` dont need to match. The total number of elements in each Tensor need to be the same. 

.. note:: When the shapes do not match, the shape of :attr:`input` is used as the shape for the returned output Tensor

:math:`out = tensor + (other * value)`

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

Performs a batch matrix-matrix product of matrices stored in :attr:`batch1` and :attr:`batch2`, 
with a reduced add step (all matrix multiplications get accumulated along the first dimension). 
:attr:`mat` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3D Tensors each containing the same number of matrices.

If :attr:`batch1` is a `b x n x m` Tensor, :attr:`batch2` is a `b x m x p` Tensor, :attr:`out` and :attr:`mat` will be `n x p` Tensors.

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

Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`, multiply
the result by the scalar :attr:`value` and add it to :attr:`tensor`.

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

Performs the element-wise multiplication of :attr:`tensor1` by :attr:`tensor2`, multiply
the result by the scalar :attr:`value` and add it to :attr:`tensor`.

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

If :attr:`mat1` is a `n x m` Tensor, :attr:`mat2` is a `m x p` Tensor, :attr:`out` and :attr:`mat` will be `n x p` Tensors.

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

Performs a matrix-vector product of the matrix :attr:`mat` and the vector :attr:`vec`.
The vector :attr:`tensor` is added to the final result.

If :attr:`mat` is a `n x m` Tensor, :attr:`vec` is a 1D Tensor of size `m`, :attr:`out` and :attr:`tensor` will be 1D of size `n`.

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
"""
addr(beta=1, mat, alpha=1, vec1, vec2, out=None) -> Tensor

Performs the outer-product between vectors :attr:`vec1` and :attr:`vec2` and adds it to the matrix :attr:`mat`.

Optional values :attr:`beta` and :attr:`alpha` are scalars that multiply :attr:`mat` and `(vec1 [out] vec2)` respectively

In other words,

:math:`res_{ij} = (beta * mat_i_j) + (alpha * vec1_i @ vec2_j)`

If :attr:`vec1` is a vector of size `n` and :attr:`vec2` is a vector of size `m`, then :attr:`mat` must be a matrix of size `n x m`

Args:
    beta (float, optional): multiplier for :attr:`mat`
    mat (Tensor): matrix to be added
    alpha (float, optional): multiplier for `vec1 (out) vec2`
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

Returns a new `Tensor` with the arctangent of the elements of :attr:`input1` and :attr:`input2`.

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
"""
baddbmm(beta=1, mat, alpha=1, batch1, batch2, out=None) -> Tensor

Performs a batch matrix-matrix product of matrices in :attr:`batch1` and :attr:`batch2`. 
:attr:`mat` is added to the final result.

:attr:`batch1` and :attr:`batch2` must be 3D Tensors each containing the same number of matrices.

If :attr:`batch1` is a `b x n x m` Tensor, :attr:`batch2` is a `b x m x p` Tensor, :attr:`out` and :attr:`mat` will be `b x n x p` Tensors.

In other words,
:math:`res_i = (beta * M_i) + (alpha * batch1_i @ batch2_i)`

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
    >>> res = torch.bmm(M, batch1, batch2)
    >>> res.size()
    torch.Size([10, 3, 5])
""")

add_docstr(torch._C.cat,
"""
""")

add_docstr(torch._C.cauchy,
"""
""")

add_docstr(torch._C.ceil,
"""
ceil(input, out=None) -> Tensor

Returns a new `Tensor` with the ceil of the elements of :attr:`input`, the smallest integer greater than or equal to each element.

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

add_docstr(torch._C.cinv,
"""
cinv(input, out=None) -> Tensor

Returns a new `Tensor` with the scalar inverse of the elements of :attr:`input`, i.e. :math:`1.0 / x`

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
    
    >>> torch.cinv(a)
    
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
    
""")

add_docstr(torch._C.cmax,
"""
.. function:: cmax(input, value, out=None) -> Tensor

Takes the element-wise `max` of the scalar :attr:`value` and each element of the input :attr:`input` and returns a new tensor with the result.

Args:
    input (Tensor): the input `Tensor`
    value (float): the scalar to be compared with
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    
     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]
    
    >>> torch.cmax(a, 0.5)

     1.3869
     0.5000
     0.5000
     0.5000
    [torch.FloatTensor of size 4]
    
    
.. function:: cmax(input, other, out=None) -> Tensor

Each element of the Tensor :attr:`other` is compared with the corresponding element of the Tensor :attr:`input` 
and an element-wise `max` is taken. The resulting Tensor is returned.

The shapes of :attr:`input` and :attr:`other` dont need to match. The total number of elements in each Tensor need to be the same. 

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
    
    >>> torch.cmax(a, b)
    
     1.3869
     0.3912
     0.6258
     0.3627
    [torch.FloatTensor of size 4]

""")

add_docstr(torch._C.cmin,
"""
.. function:: cmin(input, value, out=None) -> Tensor

Takes the element-wise `min` of the scalar :attr:`value` and each element of the input :attr:`input` and returns a new tensor with the result.

Args:
    input (Tensor): the input `Tensor`
    value (float): the scalar to be compared with
    out (Tensor, optional): The result `Tensor`

Example::

    >>> a = torch.randn(4)
    >>> a
    
     1.3869
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]
    
    >>> torch.cmin(a, 0.5)
    
     0.5000
     0.3912
    -0.8634
    -0.5468
    [torch.FloatTensor of size 4]
        
    
.. function:: cmin(input, other, out=None) -> Tensor

Each element of the Tensor :attr:`other` is compared with the corresponding element of the Tensor :attr:`input` 
and an element-wise `min` is taken. The resulting Tensor is returned.

The shapes of :attr:`input` and :attr:`other` dont need to match. The total number of elements in each Tensor need to be the same. 

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
    
    >>> torch.cmin(a, b)    
    
     1.0067
    -0.8010
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
    other  (Tensor): the second input `Tensor`
    dim  (long, optional): the dimension to take the cross-product in.
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
    dim  (long): the dimension to do the operation over
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
    dim  (long): the dimension to do the operation over
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
""")

add_docstr(torch._C.dist,
"""
""")

add_docstr(torch._C.div,
"""
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
    Tensor: a ``torch.ByteTensor`` containing a 1 at each location where the tensors are equal and a 0 at every other location

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

Returns a new `Tensor` with the floor of the elements of :attr:`input`, the largest integer less than or equal to each element.

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
    [torch.FloatTensor of size 2x2]""")

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
"""
""")

add_docstr(torch._C.ger,
"""
""")

add_docstr(torch._C.gesv,
"""
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
""")

add_docstr(torch._C.inverse,
"""
""")

add_docstr(torch._C.kthvalue,
"""
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
""")

add_docstr(torch._C.linspace,
"""
""")

add_docstr(torch._C.log,
"""
""")

add_docstr(torch._C.log1p,
"""
""")

add_docstr(torch._C.log_normal,
"""
""")

add_docstr(torch._C.logspace,
"""
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
""")

add_docstr(torch._C.max,
"""
""")

add_docstr(torch._C.mean,
"""
""")

add_docstr(torch._C.median,
"""
""")

add_docstr(torch._C.min,
"""
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
""")

add_docstr(torch._C.mul,
"""
""")

add_docstr(torch._C.multinomial,
"""
""")

add_docstr(torch._C.mv,
"""
addmv(mat, vec, out=None) -> Tensor

Performs a matrix-vector product of the matrix :attr:`mat` and the vector :attr:`vec`.

If :attr:`mat` is a `n x m` Tensor, :attr:`vec` is a 1D Tensor of size `m`, :attr:`out` will be 1D of size `n`.

Args:
    mat (Tensor): matrix to be multiplied
    vec (Tensor): vector to be multiplied
    out (Tensor, optional): Output tensor

Example::

    >>> M = torch.randn(2)
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
""")

add_docstr(torch._C.nonzero,
"""
""")

add_docstr(torch._C.norm,
"""
""")

add_docstr(torch._C.normal,
"""
""")

add_docstr(torch._C.numel,
"""
""")

add_docstr(torch._C.ones,
"""
""")

add_docstr(torch._C.orgqr,
"""
""")

add_docstr(torch._C.ormqr,
"""
""")

add_docstr(torch._C.potrf,
"""
""")

add_docstr(torch._C.potri,
"""
""")

add_docstr(torch._C.potrs,
"""
""")

add_docstr(torch._C.pow,
"""
""")

add_docstr(torch._C.prod,
"""
""")

add_docstr(torch._C.pstrf,
"""
""")

add_docstr(torch._C.qr,
"""
""")

add_docstr(torch._C.rand,
"""
""")

add_docstr(torch._C.randn,
"""
""")

add_docstr(torch._C.random,
"""
""")

add_docstr(torch._C.randperm,
"""
""")

add_docstr(torch._C.range,
"""
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
""")

add_docstr(torch._C.reshape,
"""
""")

add_docstr(torch._C.round,
"""
""")

add_docstr(torch._C.rsqrt,
"""
""")

add_docstr(torch._C.scatter,
"""
""")

add_docstr(torch._C.set_num_threads,
"""
set_num_threads(int)

Sets the number of OpenMP threads used for parallelizing CPU operations
""")

add_docstr(torch._C.sigmoid,
"""
""")

add_docstr(torch._C.sign,
"""
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
""")

add_docstr(torch._C.sqrt,
"""
""")

add_docstr(torch._C.squeeze,
"""
""")

add_docstr(torch._C.std,
"""
""")

add_docstr(torch._C.sum,
"""
""")

add_docstr(torch._C.svd,
"""
""")

add_docstr(torch._C.symeig,
"""
""")

add_docstr(torch._C.t,
"""
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
""")

add_docstr(torch._C.trace,
"""
""")

add_docstr(torch._C.transpose,
"""
""")

add_docstr(torch._C.tril,
"""
""")

add_docstr(torch._C.triu,
"""
""")

add_docstr(torch._C.trtrs,
"""
""")

add_docstr(torch._C.trunc,
"""
""")

add_docstr(torch._C.unfold,
"""
""")

add_docstr(torch._C.uniform,
"""
""")

add_docstr(torch._C.var,
"""
""")

add_docstr(torch._C.zero,
"""
""")

add_docstr(torch._C.zeros,
"""
""")
