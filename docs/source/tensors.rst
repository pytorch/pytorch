.. currentmodule:: torch

.. _tensor-doc:

torch.Tensor
===================================

A :class:`torch.Tensor` is a multi-dimensional matrix containing elements of
a single data type.


Data types
----------

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

Initializing and basic operations
---------------------------------

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

For more information about building Tensors, see :ref:`tensor-creation-ops`


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

For more information about indexing, see :ref:`indexing-slicing-joining`

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


Tensor class reference
----------------------

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

<<<<<<< HEAD
=======
<<<<<<< HEAD
   .. warning::
      The :class:`torch.Tensor` constructor is deprecated. Instead, consider using:
      :func:`torch.tensor` for creating tensors from tensor-like objects (e.g. lists and tuples);
      or :func:`torch.empty` for creating uninitialized tensors with specific sizes (e.g. int).

>>>>>>> Adding hsplit,vsplit and dsplit methods
.. autoattribute:: Tensor.T

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.new_tensor
    Tensor.new_full
    Tensor.new_empty
    Tensor.new_ones
    Tensor.new_zeros

    Tensor.is_cuda
    Tensor.is_quantized
    Tensor.is_meta
    Tensor.device
    Tensor.grad
    Tensor.ndim
    Tensor.real
    Tensor.imag

    Tensor.abs
    Tensor.abs_
    Tensor.absolute
    Tensor.absolute_
    Tensor.acos
    Tensor.acos_
    Tensor.arccos
    Tensor.arccos_
    Tensor.add
    Tensor.add_
    Tensor.addbmm
    Tensor.addbmm_
    Tensor.addcdiv
    Tensor.addcdiv_
    Tensor.addcmul
    Tensor.addcmul_
    Tensor.addmm
    Tensor.addmm_
    Tensor.sspaddmm
    Tensor.addmv
    Tensor.addmv_
    Tensor.addr
    Tensor.addr_
    Tensor.allclose
    Tensor.amax
    Tensor.amin
    Tensor.angle
    Tensor.apply_
    Tensor.argmax
    Tensor.argmin
    Tensor.argsort
    Tensor.asin
    Tensor.asin_
    Tensor.arcsin
    Tensor.arcsin_
    Tensor.as_strided
    Tensor.atan
    Tensor.atan_
    Tensor.arctan
    Tensor.arctan_
    Tensor.atan2
    Tensor.atan2_
    Tensor.all
    Tensor.any
    Tensor.backward
    Tensor.baddbmm
    Tensor.baddbmm_
    Tensor.bernoulli
    Tensor.bernoulli_
    Tensor.bfloat16
    Tensor.bincount
    Tensor.bitwise_not
    Tensor.bitwise_not_
    Tensor.bitwise_and
    Tensor.bitwise_and_
    Tensor.bitwise_or
    Tensor.bitwise_or_
    Tensor.bitwise_xor
    Tensor.bitwise_xor_
    Tensor.bmm
    Tensor.bool
    Tensor.byte
    Tensor.broadcast_to
    Tensor.cauchy_
    Tensor.ceil
    Tensor.ceil_
    Tensor.char
    Tensor.cholesky
    Tensor.cholesky_inverse
    Tensor.cholesky_solve
    Tensor.chunk
    Tensor.clamp
    Tensor.clamp_
    Tensor.clip
    Tensor.clip_
    Tensor.clone
    Tensor.contiguous
    Tensor.copy_
    Tensor.conj
    Tensor.copysign
    Tensor.copysign_
    Tensor.cos
    Tensor.cos_
    Tensor.cosh
    Tensor.cosh_
    Tensor.count_nonzero
    Tensor.acosh
    Tensor.acosh_
    Tensor.arccosh
    Tensor.arccosh_
    Tensor.cpu
    Tensor.cross
    Tensor.cuda
    Tensor.logcumsumexp
    Tensor.cummax
    Tensor.cummin
    Tensor.cumprod
    Tensor.cumprod_
    Tensor.cumsum
    Tensor.cumsum_
    Tensor.data_ptr
    Tensor.deg2rad
    Tensor.dequantize
    Tensor.det
    Tensor.dense_dim
    Tensor.detach
    Tensor.detach_
    Tensor.diag
    Tensor.diag_embed
    Tensor.diagflat
    Tensor.diagonal
    Tensor.fill_diagonal_
    Tensor.fmax
    Tensor.fmin
    Tensor.diff
    Tensor.digamma
    Tensor.digamma_
    Tensor.dim
    Tensor.dist
    Tensor.div
    Tensor.div_
    Tensor.divide
    Tensor.divide_
    Tensor.dot
    Tensor.double
    Tensor.eig
    Tensor.element_size
    Tensor.eq
    Tensor.eq_
    Tensor.equal
    Tensor.erf
    Tensor.erf_
    Tensor.erfc
    Tensor.erfc_
    Tensor.erfinv
    Tensor.erfinv_
    Tensor.exp
    Tensor.exp_
    Tensor.expm1
    Tensor.expm1_
    Tensor.expand
    Tensor.expand_as
    Tensor.exponential_
    Tensor.fix
    Tensor.fix_
    Tensor.fill_
    Tensor.flatten
    Tensor.flip
    Tensor.fliplr
    Tensor.flipud
    Tensor.float
    Tensor.float_power
    Tensor.float_power_
    Tensor.floor
    Tensor.floor_
    Tensor.floor_divide
    Tensor.floor_divide_
    Tensor.fmod
    Tensor.fmod_
    Tensor.frac
    Tensor.frac_
    Tensor.frexp
    Tensor.gather
    Tensor.gcd
    Tensor.gcd_
    Tensor.ge
    Tensor.ge_
    Tensor.greater_equal
    Tensor.greater_equal_
    Tensor.geometric_
    Tensor.geqrf
    Tensor.ger
    Tensor.get_device
    Tensor.gt
    Tensor.gt_
    Tensor.greater
    Tensor.greater_
    Tensor.half
    Tensor.hardshrink
    Tensor.heaviside
    Tensor.histc
    Tensor.hypot
    Tensor.hypot_
    Tensor.i0
    Tensor.i0_
    Tensor.igamma
    Tensor.igamma_
    Tensor.igammac
    Tensor.igammac_
    Tensor.index_add_
    Tensor.index_add
    Tensor.index_copy_
    Tensor.index_copy
    Tensor.index_fill_
    Tensor.index_fill
    Tensor.index_put_
    Tensor.index_put
    Tensor.index_select
    Tensor.indices
    Tensor.inner
    Tensor.int
    Tensor.int_repr
    Tensor.inverse
    Tensor.isclose
    Tensor.isfinite
    Tensor.isinf
    Tensor.isposinf
    Tensor.isneginf
    Tensor.isnan
    Tensor.is_contiguous
    Tensor.is_complex
    Tensor.is_floating_point
    Tensor.is_leaf
    Tensor.is_pinned
    Tensor.is_set_to
    Tensor.is_shared
    Tensor.is_signed
    Tensor.is_sparse
    Tensor.istft
    Tensor.isreal
    Tensor.item
    Tensor.kthvalue
    Tensor.lcm
    Tensor.lcm_
    Tensor.ldexp
    Tensor.ldexp_
    Tensor.le
    Tensor.le_
    Tensor.less_equal
    Tensor.less_equal_
    Tensor.lerp
    Tensor.lerp_
    Tensor.lgamma
    Tensor.lgamma_
    Tensor.log
    Tensor.log_
    Tensor.logdet
    Tensor.log10
    Tensor.log10_
    Tensor.log1p
    Tensor.log1p_
    Tensor.log2
    Tensor.log2_
    Tensor.log_normal_
    Tensor.logaddexp
    Tensor.logaddexp2
    Tensor.logsumexp
    Tensor.logical_and
    Tensor.logical_and_
    Tensor.logical_not
    Tensor.logical_not_
    Tensor.logical_or
    Tensor.logical_or_
    Tensor.logical_xor
    Tensor.logical_xor_
    Tensor.logit
    Tensor.logit_
    Tensor.long
    Tensor.lstsq
    Tensor.lt
    Tensor.lt_
    Tensor.less
    Tensor.less_
    Tensor.lu
    Tensor.lu_solve
    Tensor.as_subclass
    Tensor.map_
    Tensor.masked_scatter_
    Tensor.masked_scatter
    Tensor.masked_fill_
    Tensor.masked_fill
    Tensor.masked_select
    Tensor.matmul
    Tensor.matrix_power
    Tensor.matrix_exp
    Tensor.max
    Tensor.maximum
    Tensor.mean
    Tensor.median
    Tensor.nanmedian
    Tensor.min
    Tensor.minimum
    Tensor.mm
    Tensor.smm
    Tensor.mode
    Tensor.movedim
    Tensor.moveaxis
    Tensor.msort
    Tensor.mul
    Tensor.mul_
    Tensor.multiply
    Tensor.multiply_
    Tensor.multinomial
    Tensor.mv
    Tensor.mvlgamma
    Tensor.mvlgamma_
    Tensor.nansum
    Tensor.narrow
    Tensor.narrow_copy
    Tensor.ndimension
    Tensor.nan_to_num
    Tensor.nan_to_num_
    Tensor.ne
    Tensor.ne_
    Tensor.not_equal
    Tensor.not_equal_
    Tensor.neg
    Tensor.neg_
    Tensor.negative
    Tensor.negative_
    Tensor.nelement
    Tensor.nextafter
    Tensor.nextafter_
    Tensor.nonzero
    Tensor.norm
    Tensor.normal_
    Tensor.numel
    Tensor.numpy
    Tensor.orgqr
    Tensor.ormqr
    Tensor.outer
    Tensor.permute
    Tensor.pin_memory
    Tensor.pinverse
    Tensor.polygamma
    Tensor.polygamma_
    Tensor.pow
    Tensor.pow_
    Tensor.prod
    Tensor.put_
    Tensor.qr
    Tensor.qscheme
    Tensor.quantile
    Tensor.nanquantile
    Tensor.q_scale
    Tensor.q_zero_point
    Tensor.q_per_channel_scales
    Tensor.q_per_channel_zero_points
    Tensor.q_per_channel_axis
    Tensor.rad2deg
    Tensor.random_
    Tensor.ravel
    Tensor.reciprocal
    Tensor.reciprocal_
    Tensor.record_stream
    Tensor.register_hook
    Tensor.remainder
    Tensor.remainder_
    Tensor.renorm
    Tensor.renorm_
    Tensor.repeat
    Tensor.repeat_interleave
    Tensor.requires_grad
    Tensor.requires_grad_
    Tensor.reshape
    Tensor.reshape_as
    Tensor.resize_
    Tensor.resize_as_
    Tensor.retain_grad
    Tensor.roll
    Tensor.rot90
    Tensor.round
    Tensor.round_
    Tensor.rsqrt
    Tensor.rsqrt_
    Tensor.scatter
    Tensor.scatter_
    Tensor.scatter_add_
    Tensor.scatter_add
    Tensor.select
    Tensor.set_
    Tensor.share_memory_
    Tensor.short
    Tensor.sigmoid
    Tensor.sigmoid_
    Tensor.sign
    Tensor.sign_
    Tensor.signbit
    Tensor.sgn
    Tensor.sgn_
    Tensor.sin
    Tensor.sin_
    Tensor.sinc
    Tensor.sinc_
    Tensor.sinh
    Tensor.sinh_
    Tensor.asinh
    Tensor.asinh_
    Tensor.arcsinh
    Tensor.arcsinh_
    Tensor.size
    Tensor.slogdet
    Tensor.solve
    Tensor.sort
    Tensor.split
    Tensor.sparse_mask
    Tensor.sparse_dim
    Tensor.sqrt
    Tensor.sqrt_
    Tensor.square
    Tensor.square_
    Tensor.squeeze
    Tensor.squeeze_
    Tensor.std
    Tensor.stft
    Tensor.storage
    Tensor.storage_offset
    Tensor.storage_type
    Tensor.stride
    Tensor.sub
    Tensor.sub_
    Tensor.subtract
    Tensor.subtract_
    Tensor.sum
    Tensor.sum_to_size
    Tensor.svd
    Tensor.swapaxes
    Tensor.swapdims
    Tensor.symeig
    Tensor.t
    Tensor.t_
    Tensor.tensor_split
    Tensor.tile
    Tensor.to
    Tensor.to_mkldnn
    Tensor.take
    Tensor.take_along_dim
    Tensor.tan
    Tensor.tan_
    Tensor.tanh
    Tensor.tanh_
    Tensor.atanh
    Tensor.atanh_
    Tensor.arctanh
    Tensor.arctanh_
    Tensor.tolist
    Tensor.topk
    Tensor.to_sparse
    Tensor.trace
    Tensor.transpose
    Tensor.transpose_
    Tensor.triangular_solve
    Tensor.tril
    Tensor.tril_
    Tensor.triu
    Tensor.triu_
    Tensor.true_divide
    Tensor.true_divide_
    Tensor.trunc
    Tensor.trunc_
    Tensor.type
    Tensor.type_as
    Tensor.unbind
    Tensor.unfold
    Tensor.uniform_
    Tensor.unique
    Tensor.unique_consecutive
    Tensor.unsqueeze
    Tensor.unsqueeze_
    Tensor.values
    Tensor.var
    Tensor.vdot
    Tensor.view
    Tensor.view_as
    Tensor.where
    Tensor.xlogy
    Tensor.xlogy_
    Tensor.zero_
=======
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
   .. automethod:: dsplit
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
   .. automethod:: hsplit
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
   .. automethod:: vsplit
   .. automethod:: where
   .. automethod:: xlogy
   .. automethod:: xlogy_
   .. automethod:: zero_
>>>>>>> Adding hsplit,vsplit and dsplit methods
