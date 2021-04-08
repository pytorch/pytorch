.. currentmodule:: torch

.. _tensor-doc:

torch.Tensor
===================================

A :class:`torch.Tensor` is a multi-dimensional matrix containing elements of
a single data type.

Torch defines 10 tensor types with CPU and GPU variants which are as follows:

======================================= =========================================== ============================= ================================
Data type                               dtype                                       CPU tensor                    GPU tensor
======================================= =========================================== ============================= ================================
32-bit floating point                   ``torch.float32`` or ``torch.float``        :class:`torch.FloatTensor`    :class:`torch.cuda.FloatTensor`
64-bit floating point                   ``torch.float64`` or ``torch.double``       :class:`torch.DoubleTensor`   :class:`torch.cuda.DoubleTensor`
16-bit floating point [1]_              ``torch.float16`` or ``torch.half``         :class:`torch.HalfTensor`     :class:`torch.cuda.HalfTensor`
16-bit floating point [2]_              ``torch.bfloat16``                          :class:`torch.BFloat16Tensor` :class:`torch.cuda.BFloat16Tensor`
32-bit complex                          ``torch.complex32``
64-bit complex                          ``torch.complex64``
128-bit complex                         ``torch.complex128`` or ``torch.cdouble``
8-bit integer (unsigned)                ``torch.uint8``                             :class:`torch.ByteTensor`     :class:`torch.cuda.ByteTensor`
8-bit integer (signed)                  ``torch.int8``                              :class:`torch.CharTensor`     :class:`torch.cuda.CharTensor`
16-bit integer (signed)                 ``torch.int16`` or ``torch.short``          :class:`torch.ShortTensor`    :class:`torch.cuda.ShortTensor`
32-bit integer (signed)                 ``torch.int32`` or ``torch.int``            :class:`torch.IntTensor`      :class:`torch.cuda.IntTensor`
64-bit integer (signed)                 ``torch.int64`` or ``torch.long``           :class:`torch.LongTensor`     :class:`torch.cuda.LongTensor`
Boolean                                 ``torch.bool``                              :class:`torch.BoolTensor`     :class:`torch.cuda.BoolTensor`
quantized 8-bit integer (unsigned)      ``torch.quint8``                            :class:`torch.ByteTensor`     /
quantized 8-bit integer (signed)        ``torch.qint8``                             :class:`torch.CharTensor`     /
quantized 32-bit integer (signed)       ``torch.qfint32``                           :class:`torch.IntTensor`      /
quantized 4-bit integer (unsigned) [3]_ ``torch.quint4x2``                          :class:`torch.ByteTensor`     /
======================================= =========================================== ============================= ================================

.. [1]
  Sometimes referred to as binary16: uses 1 sign, 5 exponent, and 10
  significand bits. Useful when precision is important at the expense of range.
.. [2]
  Sometimes referred to as Brain Floating Point: uses 1 sign, 8 exponent, and 7
  significand bits. Useful when range is important, since it has the same
  number of exponent bits as ``float32``
.. [3]
  quantized 4-bit integer is stored as a 8-bit signed integer. Currently it's only supported in EmbeddingBag operator.

:class:`torch.Tensor` is an alias for the default tensor type (:class:`torch.FloatTensor`).

A tensor can be constructed from a Python :class:`list` or sequence using the
:func:`torch.tensor` constructor:

::

    >>> torch.tensor([[1., -1.], [1., -1.]])
    tensor([[ 1.0000, -1.0000],
            [ 1.0000, -1.0000]])
    >>> torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])

.. warning::

    :func:`torch.tensor` always copies :attr:`data`. If you have a Tensor
    :attr:`data` and just want to change its ``requires_grad`` flag, use
    :meth:`~torch.Tensor.requires_grad_` or
    :meth:`~torch.Tensor.detach` to avoid a copy.
    If you have a numpy array and want to avoid a copy, use
    :func:`torch.as_tensor`.

A tensor of specific data type can be constructed by passing a
:class:`torch.dtype` and/or a :class:`torch.device` to a
constructor or tensor creation op:

::

    >>> torch.zeros([2, 4], dtype=torch.int32)
    tensor([[ 0,  0,  0,  0],
            [ 0,  0,  0,  0]], dtype=torch.int32)
    >>> cuda0 = torch.device('cuda:0')
    >>> torch.ones([2, 4], dtype=torch.float64, device=cuda0)
    tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
            [ 1.0000,  1.0000,  1.0000,  1.0000]], dtype=torch.float64, device='cuda:0')

The contents of a tensor can be accessed and modified using Python's indexing
and slicing notation:

::

    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> print(x[1][2])
    tensor(6)
    >>> x[0][1] = 8
    >>> print(x)
    tensor([[ 1,  8,  3],
            [ 4,  5,  6]])

Use :meth:`torch.Tensor.item` to get a Python number from a tensor containing a
single value:

::

    >>> x = torch.tensor([[1]])
    >>> x
    tensor([[ 1]])
    >>> x.item()
    1
    >>> x = torch.tensor(2.5)
    >>> x
    tensor(2.5000)
    >>> x.item()
    2.5

A tensor can be created with :attr:`requires_grad=True` so that
:mod:`torch.autograd` records operations on them for automatic differentiation.

::

    >>> x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
    >>> out = x.pow(2).sum()
    >>> out.backward()
    >>> x.grad
    tensor([[ 2.0000, -2.0000],
            [ 2.0000,  2.0000]])

Each tensor has an associated :class:`torch.Storage`, which holds its data.
The tensor class also provides multi-dimensional, `strided <https://en.wikipedia.org/wiki/Stride_of_an_array>`_
view of a storage and defines numeric operations on it.

.. note::
   For more information on tensor views, see :ref:`tensor-view-doc`.

.. note::
   For more information on the :class:`torch.dtype`, :class:`torch.device`, and
   :class:`torch.layout` attributes of a :class:`torch.Tensor`, see
   :ref:`tensor-attributes-doc`.

.. note::
   Methods which mutate a tensor are marked with an underscore suffix.
   For example, :func:`torch.FloatTensor.abs_` computes the absolute value
   in-place and returns the modified tensor, while :func:`torch.FloatTensor.abs`
   computes the result in a new tensor.

.. note::
    To change an existing tensor's :class:`torch.device` and/or :class:`torch.dtype`, consider using
    :meth:`~torch.Tensor.to` method on the tensor.

.. warning::
   Current implementation of :class:`torch.Tensor` introduces memory overhead,
   thus it might lead to unexpectedly high memory usage in the applications with many tiny tensors.
   If this is your case, consider using one large structure.

.. class:: Tensor()

   There are a few main ways to create a tensor, depending on your use case.

   - To create a tensor with pre-existing data, use :func:`torch.tensor`.
   - To create a tensor with specific size, use ``torch.*`` tensor creation
     ops (see :ref:`tensor-creation-ops`).
   - To create a tensor with the same size (and similar types) as another tensor,
     use ``torch.*_like`` tensor creation ops
     (see :ref:`tensor-creation-ops`).
   - To create a tensor with similar type but different size as another tensor,
     use ``tensor.new_*`` creation ops.

   .. warning::
      The :class:`torch.Tensor` constructor is deprecated. Instead, consider using:
      :func:`torch.tensor` for creating tensors from tensor-like objects (e.g. lists and tuples);
      or :func:`torch.empty` for creating uninitialized tensors with specific sizes (e.g. int).

   .. automethod:: new_tensor
   .. automethod:: new_full
   .. automethod:: new_empty
   .. automethod:: new_ones
   .. automethod:: new_zeros

   .. autoattribute:: is_cuda
   .. autoattribute:: is_quantized
   .. autoattribute:: is_meta
   .. autoattribute:: device
   .. autoattribute:: grad
      :noindex:
   .. autoattribute:: ndim
   .. autoattribute:: T
   .. autoattribute:: real
   .. autoattribute:: imag

   .. automethod:: abs
   .. automethod:: abs_
   .. automethod:: absolute
   .. automethod:: absolute_
   .. automethod:: acos
   .. automethod:: acos_
   .. automethod:: arccos
   .. automethod:: arccos_
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
   .. automethod:: sspaddmm
      :noindex:
   .. automethod:: addmv
   .. automethod:: addmv_
   .. automethod:: addr
   .. automethod:: addr_
   .. automethod:: allclose
   .. automethod:: amax
   .. automethod:: amin
   .. automethod:: angle
   .. automethod:: apply_
   .. automethod:: argmax
   .. automethod:: argmin
   .. automethod:: argsort
   .. automethod:: asin
   .. automethod:: asin_
   .. automethod:: arcsin
   .. automethod:: arcsin_
   .. automethod:: as_strided
   .. automethod:: atan
   .. automethod:: atan_
   .. automethod:: arctan
   .. automethod:: arctan_
   .. automethod:: atan2
   .. automethod:: atan2_
   .. automethod:: all
   .. automethod:: any
   .. automethod:: backward
      :noindex:
   .. automethod:: baddbmm
   .. automethod:: baddbmm_
   .. automethod:: bernoulli
   .. automethod:: bernoulli_
   .. automethod:: bfloat16
   .. automethod:: bincount
   .. automethod:: bitwise_not
   .. automethod:: bitwise_not_
   .. automethod:: bitwise_and
   .. automethod:: bitwise_and_
   .. automethod:: bitwise_or
   .. automethod:: bitwise_or_
   .. automethod:: bitwise_xor
   .. automethod:: bitwise_xor_
   .. automethod:: bmm
   .. automethod:: bool
   .. automethod:: byte
   .. automethod:: broadcast_to
   .. automethod:: cauchy_
   .. automethod:: ceil
   .. automethod:: ceil_
   .. automethod:: char
   .. automethod:: cholesky
   .. automethod:: cholesky_inverse
   .. automethod:: cholesky_solve
   .. automethod:: chunk
   .. automethod:: clamp
   .. automethod:: clamp_
   .. automethod:: clip
   .. automethod:: clip_
   .. automethod:: clone
   .. automethod:: contiguous
   .. automethod:: copy_
   .. automethod:: conj
   .. automethod:: copysign
   .. automethod:: copysign_
   .. automethod:: cos
   .. automethod:: cos_
   .. automethod:: cosh
   .. automethod:: cosh_
   .. automethod:: count_nonzero
   .. automethod:: acosh
   .. automethod:: acosh_
   .. automethod:: arccosh
   .. automethod:: arccosh_
   .. automethod:: cpu
   .. automethod:: cross
   .. automethod:: cuda
   .. automethod:: logcumsumexp
   .. automethod:: cummax
   .. automethod:: cummin
   .. automethod:: cumprod
   .. automethod:: cumprod_
   .. automethod:: cumsum
   .. automethod:: cumsum_
   .. automethod:: data_ptr
   .. automethod:: deg2rad
   .. automethod:: dequantize
   .. automethod:: det
   .. automethod:: dense_dim
      :noindex:
   .. automethod:: detach
      :noindex:
   .. automethod:: detach_
      :noindex:
   .. automethod:: diag
   .. automethod:: diag_embed
   .. automethod:: diagflat
   .. automethod:: diagonal
   .. automethod:: fill_diagonal_
   .. automethod:: fmax
   .. automethod:: fmin
   .. automethod:: diff
   .. automethod:: digamma
   .. automethod:: digamma_
   .. automethod:: dim
   .. automethod:: dist
   .. automethod:: div
   .. automethod:: div_
   .. automethod:: divide
   .. automethod:: divide_
   .. automethod:: dot
   .. automethod:: double
   .. automethod:: eig
   .. automethod:: element_size
   .. automethod:: eq
   .. automethod:: eq_
   .. automethod:: equal
   .. automethod:: erf
   .. automethod:: erf_
   .. automethod:: erfc
   .. automethod:: erfc_
   .. automethod:: erfinv
   .. automethod:: erfinv_
   .. automethod:: exp
   .. automethod:: exp_
   .. automethod:: expm1
   .. automethod:: expm1_
   .. automethod:: expand
   .. automethod:: expand_as
   .. automethod:: exponential_
   .. automethod:: fix
   .. automethod:: fix_
   .. automethod:: fill_
   .. automethod:: flatten
   .. automethod:: flip
   .. automethod:: fliplr
   .. automethod:: flipud
   .. automethod:: float
   .. automethod:: float_power
   .. automethod:: float_power_
   .. automethod:: floor
   .. automethod:: floor_
   .. automethod:: floor_divide
   .. automethod:: floor_divide_
   .. automethod:: fmod
   .. automethod:: fmod_
   .. automethod:: frac
   .. automethod:: frac_
   .. automethod:: frexp
   .. automethod:: gather
   .. automethod:: gcd
   .. automethod:: gcd_
   .. automethod:: ge
   .. automethod:: ge_
   .. automethod:: greater_equal
   .. automethod:: greater_equal_
   .. automethod:: geometric_
   .. automethod:: geqrf
   .. automethod:: ger
   .. automethod:: get_device
   .. automethod:: gt
   .. automethod:: gt_
   .. automethod:: greater
   .. automethod:: greater_
   .. automethod:: half
   .. automethod:: hardshrink
   .. automethod:: heaviside
   .. automethod:: histc
   .. automethod:: hypot
   .. automethod:: hypot_
   .. automethod:: i0
   .. automethod:: i0_
   .. automethod:: igamma
   .. automethod:: igamma_
   .. automethod:: igammac
   .. automethod:: igammac_
   .. automethod:: index_add_
   .. automethod:: index_add
   .. automethod:: index_copy_
   .. automethod:: index_copy
   .. automethod:: index_fill_
   .. automethod:: index_fill
   .. automethod:: index_put_
   .. automethod:: index_put
   .. automethod:: index_select
   .. automethod:: indices
      :noindex:
   .. automethod:: inner
   .. automethod:: int
   .. automethod:: int_repr
   .. automethod:: inverse
   .. automethod:: isclose
   .. automethod:: isfinite
   .. automethod:: isinf
   .. automethod:: isposinf
   .. automethod:: isneginf
   .. automethod:: isnan
   .. automethod:: is_contiguous
   .. automethod:: is_complex
   .. automethod:: is_floating_point
   .. autoattribute:: is_leaf
      :noindex:
   .. automethod:: is_pinned
   .. automethod:: is_set_to
   .. automethod:: is_shared
   .. automethod:: is_signed
   .. autoattribute:: is_sparse
      :noindex:
   .. automethod:: istft
   .. automethod:: isreal
   .. automethod:: item
   .. automethod:: kthvalue
   .. automethod:: lcm
   .. automethod:: lcm_
   .. automethod:: ldexp
   .. automethod:: ldexp_
   .. automethod:: le
   .. automethod:: le_
   .. automethod:: less_equal
   .. automethod:: less_equal_
   .. automethod:: lerp
   .. automethod:: lerp_
   .. automethod:: lgamma
   .. automethod:: lgamma_
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
   .. automethod:: logaddexp
   .. automethod:: logaddexp2
   .. automethod:: logsumexp
   .. automethod:: logical_and
   .. automethod:: logical_and_
   .. automethod:: logical_not
   .. automethod:: logical_not_
   .. automethod:: logical_or
   .. automethod:: logical_or_
   .. automethod:: logical_xor
   .. automethod:: logical_xor_
   .. automethod:: logit
   .. automethod:: logit_
   .. automethod:: long
   .. automethod:: lstsq
   .. automethod:: lt
   .. automethod:: lt_
   .. automethod:: less
   .. automethod:: less_
   .. automethod:: lu
   .. automethod:: lu_solve
   .. automethod:: as_subclass
   .. automethod:: map_
   .. automethod:: masked_scatter_
   .. automethod:: masked_scatter
   .. automethod:: masked_fill_
   .. automethod:: masked_fill
   .. automethod:: masked_select
   .. automethod:: matmul
   .. automethod:: matrix_power
   .. automethod:: matrix_exp
   .. automethod:: max
   .. automethod:: maximum
   .. automethod:: mean
   .. automethod:: median
   .. automethod:: nanmedian
   .. automethod:: min
   .. automethod:: minimum
   .. automethod:: mm
   .. automethod:: smm
      :noindex:
   .. automethod:: mode
   .. automethod:: movedim
   .. automethod:: moveaxis
   .. automethod:: msort
   .. automethod:: mul
   .. automethod:: mul_
   .. automethod:: multiply
   .. automethod:: multiply_
   .. automethod:: multinomial
   .. automethod:: mv
   .. automethod:: mvlgamma
   .. automethod:: mvlgamma_
   .. automethod:: nansum
   .. automethod:: narrow
   .. automethod:: narrow_copy
   .. automethod:: ndimension
   .. automethod:: nan_to_num
   .. automethod:: nan_to_num_
   .. automethod:: ne
   .. automethod:: ne_
   .. automethod:: not_equal
   .. automethod:: not_equal_
   .. automethod:: neg
   .. automethod:: neg_
   .. automethod:: negative
   .. automethod:: negative_
   .. automethod:: nelement
   .. automethod:: nextafter
   .. automethod:: nextafter_
   .. automethod:: nonzero
   .. automethod:: norm
   .. automethod:: normal_
   .. automethod:: numel
   .. automethod:: numpy
   .. automethod:: orgqr
   .. automethod:: ormqr
   .. automethod:: outer
   .. automethod:: permute
   .. automethod:: pin_memory
   .. automethod:: pinverse
   .. automethod:: polygamma
   .. automethod:: polygamma_
   .. automethod:: pow
   .. automethod:: pow_
   .. automethod:: prod
   .. automethod:: put_
   .. automethod:: qr
   .. automethod:: qscheme
   .. automethod:: quantile
   .. automethod:: nanquantile
   .. automethod:: q_scale
   .. automethod:: q_zero_point
   .. automethod:: q_per_channel_scales
   .. automethod:: q_per_channel_zero_points
   .. automethod:: q_per_channel_axis
   .. automethod:: rad2deg
   .. automethod:: random_
   .. automethod:: ravel
   .. automethod:: reciprocal
   .. automethod:: reciprocal_
   .. automethod:: record_stream
   .. automethod:: register_hook
      :noindex:
   .. automethod:: remainder
   .. automethod:: remainder_
   .. automethod:: renorm
   .. automethod:: renorm_
   .. automethod:: repeat
   .. automethod:: repeat_interleave
   .. autoattribute:: requires_grad
      :noindex:
   .. automethod:: requires_grad_
   .. automethod:: reshape
   .. automethod:: reshape_as
   .. automethod:: resize_
   .. automethod:: resize_as_
   .. automethod:: retain_grad
      :noindex:
   .. automethod:: roll
   .. automethod:: rot90
   .. automethod:: round
   .. automethod:: round_
   .. automethod:: rsqrt
   .. automethod:: rsqrt_
   .. automethod:: scatter
   .. automethod:: scatter_
   .. automethod:: scatter_add_
   .. automethod:: scatter_add
   .. automethod:: select
   .. automethod:: set_
   .. automethod:: share_memory_
   .. automethod:: short
   .. automethod:: sigmoid
   .. automethod:: sigmoid_
   .. automethod:: sign
   .. automethod:: sign_
   .. automethod:: signbit
   .. automethod:: sgn
   .. automethod:: sgn_
   .. automethod:: sin
   .. automethod:: sin_
   .. automethod:: sinc
   .. automethod:: sinc_
   .. automethod:: sinh
   .. automethod:: sinh_
   .. automethod:: asinh
   .. automethod:: asinh_
   .. automethod:: arcsinh
   .. automethod:: arcsinh_
   .. automethod:: size
   .. automethod:: slogdet
   .. automethod:: solve
   .. automethod:: sort
   .. automethod:: split
   .. automethod:: sparse_mask
      :noindex:
   .. automethod:: sparse_dim
      :noindex:
   .. automethod:: sqrt
   .. automethod:: sqrt_
   .. automethod:: square
   .. automethod:: square_
   .. automethod:: squeeze
   .. automethod:: squeeze_
   .. automethod:: std
   .. automethod:: stft
   .. automethod:: storage
   .. automethod:: storage_offset
   .. automethod:: storage_type
   .. automethod:: stride
   .. automethod:: sub
   .. automethod:: sub_
   .. automethod:: subtract
   .. automethod:: subtract_
   .. automethod:: sum
   .. automethod:: sum_to_size
   .. automethod:: svd
   .. automethod:: swapaxes
   .. automethod:: swapdims
   .. automethod:: symeig
   .. automethod:: t
   .. automethod:: t_
   .. automethod:: tensor_split
   .. automethod:: tile
   .. automethod:: to
   .. automethod:: to_mkldnn
   .. automethod:: take
   .. automethod:: take_along_dim
   .. automethod:: tan
   .. automethod:: tan_
   .. automethod:: tanh
   .. automethod:: tanh_
   .. automethod:: atanh
   .. automethod:: atanh_
   .. automethod:: arctanh
   .. automethod:: arctanh_
   .. automethod:: tolist
   .. automethod:: topk
   .. automethod:: to_sparse
      :noindex:
   .. automethod:: trace
   .. automethod:: transpose
   .. automethod:: transpose_
   .. automethod:: triangular_solve
   .. automethod:: tril
   .. automethod:: tril_
   .. automethod:: triu
   .. automethod:: triu_
   .. automethod:: true_divide
   .. automethod:: true_divide_
   .. automethod:: trunc
   .. automethod:: trunc_
   .. automethod:: type
   .. automethod:: type_as
   .. automethod:: unbind
   .. automethod:: unfold
   .. automethod:: uniform_
   .. automethod:: unique
   .. automethod:: unique_consecutive
   .. automethod:: unsqueeze
   .. automethod:: unsqueeze_
   .. automethod:: values
      :noindex:
   .. automethod:: var
   .. automethod:: vdot
   .. automethod:: view
   .. automethod:: view_as
   .. automethod:: where
   .. automethod:: xlogy
   .. automethod:: xlogy_
   .. automethod:: zero_
