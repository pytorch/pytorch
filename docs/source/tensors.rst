.. currentmodule:: torch

.. _tensor-doc:

torch.Tensor
===================================

A :class:`torch.Tensor` is a multi-dimensional matrix containing elements of
a single data type.

Torch defines seven CPU tensor types and eight GPU tensor types:

======================== ===========================   ================================
Data type                CPU tensor                    GPU tensor
======================== ===========================   ================================
32-bit floating point    :class:`torch.FloatTensor`    :class:`torch.cuda.FloatTensor`
64-bit floating point    :class:`torch.DoubleTensor`   :class:`torch.cuda.DoubleTensor`
16-bit floating point    :class:`torch.HalfTensor`     :class:`torch.cuda.HalfTensor`
8-bit integer (unsigned) :class:`torch.ByteTensor`     :class:`torch.cuda.ByteTensor`
8-bit integer (signed)   :class:`torch.CharTensor`     :class:`torch.cuda.CharTensor`
16-bit integer (signed)  :class:`torch.ShortTensor`    :class:`torch.cuda.ShortTensor`
32-bit integer (signed)  :class:`torch.IntTensor`      :class:`torch.cuda.IntTensor`
64-bit integer (signed)  :class:`torch.LongTensor`     :class:`torch.cuda.LongTensor`
======================== ===========================   ================================

The :class:`torch.Tensor` constructor is an alias for the default tensor type
(:class:`torch.FloatTensor`).

A tensor can be constructed from a Python :class:`list` or sequence:

::

    >>> torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    1  2  3
    4  5  6
    [torch.FloatTensor of size 2x3]

An empty tensor can be constructed by specifying its size:

::

    >>> torch.IntTensor(2, 4).zero_()
    0  0  0  0
    0  0  0  0
    [torch.IntTensor of size 2x4]

The contents of a tensor can be accessed and modified using Python's indexing
and slicing notation:

::

    >>> x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    >>> print(x[1][2])
    6.0
    >>> x[0][1] = 8
    >>> print(x)
     1  8  3
     4  5  6
    [torch.FloatTensor of size 2x3]

Each tensor has an associated :class:`torch.Storage`, which holds its data.
The tensor class provides multi-dimensional, `strided <https://en.wikipedia.org/wiki/Stride_of_an_array>`_
view of a storage and defines numeric operations on it.

.. note::
   Methods which mutate a tensor are marked with an underscore suffix.
   For example, :func:`torch.FloatTensor.abs_` computes the absolute value
   in-place and returns the modified tensor, while :func:`torch.FloatTensor.abs`
   computes the result in a new tensor.

.. class:: Tensor()
           Tensor(*sizes)
           Tensor(size)
           Tensor(sequence)
           Tensor(ndarray)
           Tensor(tensor)
           Tensor(storage)

   Creates a new tensor from an optional size or data.

   If no arguments are given, an empty zero-dimensional tensor is returned.
   If a :class:`numpy.ndarray`, :class:`torch.Tensor`, or :class:`torch.Storage`
   is given, a new tensor that shares the same data is returned. If a Python
   sequence is given, a new tensor is created from a copy of the sequence.

   .. automethod:: abs
   .. automethod:: abs_
   .. automethod:: acos
   .. automethod:: acos_
   .. automethod:: add
   .. automethod:: add_
   .. automethod:: addbmm
   .. automethod:: addbmm_
   .. automethod:: addcdiv
   .. automethod:: addcdiv_
   .. automethod:: addcmul
   .. automethod:: addcmul_
   .. automethod:: addmm
   .. automethod:: addmm_
   .. automethod:: addmv
   .. automethod:: addmv_
   .. automethod:: addr
   .. automethod:: addr_
   .. automethod:: apply_
   .. automethod:: argmax
   .. automethod:: argmin
   .. automethod:: asin
   .. automethod:: asin_
   .. automethod:: atan
   .. automethod:: atan2
   .. automethod:: atan2_
   .. automethod:: atan_
   .. automethod:: baddbmm
   .. automethod:: baddbmm_
   .. automethod:: bernoulli
   .. automethod:: bernoulli_
   .. automethod:: bmm
   .. automethod:: byte
   .. automethod:: btrifact
   .. automethod:: btrifact_with_info
   .. automethod:: btrisolve
   .. automethod:: cauchy_
   .. automethod:: ceil
   .. automethod:: ceil_
   .. automethod:: char
   .. automethod:: chunk
   .. automethod:: clamp
   .. automethod:: clamp_
   .. automethod:: clone
   .. automethod:: contiguous
   .. automethod:: copy_
   .. automethod:: cos
   .. automethod:: cos_
   .. automethod:: cosh
   .. automethod:: cosh_
   .. automethod:: cpu
   .. automethod:: cross
   .. automethod:: cuda
   .. automethod:: cumprod
   .. automethod:: cumsum
   .. automethod:: data_ptr
   .. automethod:: det
   .. automethod:: diag
   .. automethod:: dim
   .. automethod:: dist
   .. automethod:: div
   .. automethod:: div_
   .. automethod:: dot
   .. automethod:: double
   .. automethod:: eig
   .. automethod:: element_size
   .. automethod:: eq
   .. automethod:: eq_
   .. automethod:: equal
   .. automethod:: erf
   .. automethod:: erf_
   .. automethod:: erfinv
   .. automethod:: erfinv_
   .. automethod:: exp
   .. automethod:: exp_
   .. automethod:: expm1
   .. automethod:: expm1_
   .. automethod:: expand
   .. automethod:: expand_as
   .. automethod:: exponential_
   .. automethod:: fill_
   .. automethod:: float
   .. automethod:: floor
   .. automethod:: floor_
   .. automethod:: fmod
   .. automethod:: fmod_
   .. automethod:: frac
   .. automethod:: frac_
   .. automethod:: gather
   .. automethod:: ge
   .. automethod:: ge_
   .. automethod:: gels
   .. automethod:: geometric_
   .. automethod:: geqrf
   .. automethod:: ger
   .. automethod:: gesv
   .. automethod:: gt
   .. automethod:: gt_
   .. automethod:: half
   .. automethod:: histc
   .. automethod:: index
   .. automethod:: index_add_
   .. automethod:: index_copy_
   .. automethod:: index_fill_
   .. automethod:: index_select
   .. automethod:: int
   .. automethod:: inverse
   .. automethod:: is_contiguous
   .. autoattribute:: is_cuda
      :annotation:
   .. automethod:: is_pinned
   .. automethod:: is_set_to
   .. automethod:: is_signed
   .. automethod:: item
   .. automethod:: kthvalue
   .. automethod:: le
   .. automethod:: le_
   .. automethod:: lerp
   .. automethod:: lerp_
   .. automethod:: log
   .. automethod:: log_
   .. automethod:: logdet
   .. automethod:: log10
   .. automethod:: log10_
   .. automethod:: log1p
   .. automethod:: log1p_
   .. automethod:: log2
   .. automethod:: log2_
   .. automethod:: log_normal_
   .. automethod:: long
   .. automethod:: lt
   .. automethod:: lt_
   .. automethod:: map_
   .. automethod:: masked_scatter_
   .. automethod:: masked_fill_
   .. automethod:: masked_select
   .. automethod:: matmul
   .. automethod:: max
   .. automethod:: mean
   .. automethod:: median
   .. automethod:: min
   .. automethod:: mm
   .. automethod:: mode
   .. automethod:: mul
   .. automethod:: mul_
   .. automethod:: multinomial
   .. automethod:: mv
   .. automethod:: narrow
   .. automethod:: ndimension
   .. automethod:: ne
   .. automethod:: ne_
   .. automethod:: neg
   .. automethod:: neg_
   .. automethod:: nelement
   .. automethod:: new
   .. automethod:: nonzero
   .. automethod:: norm
   .. automethod:: normal_
   .. automethod:: numel
   .. automethod:: numpy
   .. automethod:: orgqr
   .. automethod:: ormqr
   .. automethod:: permute
   .. automethod:: pin_memory
   .. automethod:: potrf
   .. automethod:: potri
   .. automethod:: potrs
   .. automethod:: pow
   .. automethod:: pow_
   .. automethod:: prod
   .. automethod:: pstrf
   .. automethod:: put_
   .. automethod:: qr
   .. automethod:: random_
   .. automethod:: reciprocal
   .. automethod:: reciprocal_
   .. automethod:: remainder
   .. automethod:: remainder_
   .. automethod:: renorm
   .. automethod:: renorm_
   .. automethod:: repeat
   .. automethod:: reshape
   .. automethod:: resize_
   .. automethod:: resize_as_
   .. automethod:: round
   .. automethod:: round_
   .. automethod:: rsqrt
   .. automethod:: rsqrt_
   .. automethod:: scatter_
   .. automethod:: select
   .. automethod:: set_
   .. automethod:: share_memory_
   .. automethod:: short
   .. automethod:: sigmoid
   .. automethod:: sigmoid_
   .. automethod:: sign
   .. automethod:: sign_
   .. automethod:: sin
   .. automethod:: sin_
   .. automethod:: sinh
   .. automethod:: sinh_
   .. automethod:: size
   .. automethod:: slogdet
   .. automethod:: sort
   .. automethod:: split
   .. automethod:: sqrt
   .. automethod:: sqrt_
   .. automethod:: squeeze
   .. automethod:: squeeze_
   .. automethod:: std
   .. automethod:: storage
   .. automethod:: storage_offset
   .. automethod:: storage_type
   .. automethod:: stride
   .. automethod:: sub
   .. automethod:: sub_
   .. automethod:: sum
   .. automethod:: svd
   .. automethod:: symeig
   .. automethod:: t
   .. automethod:: t_
   .. automethod:: take
   .. automethod:: tan
   .. automethod:: tan_
   .. automethod:: tanh
   .. automethod:: tanh_
   .. automethod:: tolist
   .. automethod:: topk
   .. automethod:: trace
   .. automethod:: transpose
   .. automethod:: transpose_
   .. automethod:: tril
   .. automethod:: tril_
   .. automethod:: triu
   .. automethod:: triu_
   .. automethod:: trtrs
   .. automethod:: trunc
   .. automethod:: trunc_
   .. automethod:: type
   .. automethod:: type_as
   .. automethod:: unfold
   .. automethod:: uniform_
   .. automethod:: unique
   .. automethod:: unsqueeze
   .. automethod:: unsqueeze_
   .. automethod:: var
   .. automethod:: view
   .. automethod:: view_as
   .. automethod:: zero_

.. class:: ByteTensor()

   The following methods are unique to :class:`torch.ByteTensor`.

   .. automethod:: all
   .. automethod:: any
