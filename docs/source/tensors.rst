.. currentmodule:: torch

.. _tensor-doc:

torch.Tensor
===================================

A :class:`torch.Tensor` is a multi-dimensional matrix containing elements of
a single data type.

Torch defines nine CPU tensor types and nine GPU tensor types:

========================   ===========================================   ===========================   ================================
Data type                  dtype                                         CPU tensor                    GPU tensor
========================   ===========================================   ===========================   ================================
32-bit floating point      ``torch.float32`` or ``torch.float``          :class:`torch.FloatTensor`    :class:`torch.cuda.FloatTensor`
64-bit floating point      ``torch.float64`` or ``torch.double``         :class:`torch.DoubleTensor`   :class:`torch.cuda.DoubleTensor`
16-bit floating point      ``torch.float16`` or ``torch.half``           :class:`torch.HalfTensor`     :class:`torch.cuda.HalfTensor`
8-bit integer (unsigned)   ``torch.uint8``                               :class:`torch.ByteTensor`     :class:`torch.cuda.ByteTensor`
8-bit integer (signed)     ``torch.int8``                                :class:`torch.CharTensor`     :class:`torch.cuda.CharTensor`
16-bit integer (signed)    ``torch.int16`` or ``torch.short``            :class:`torch.ShortTensor`    :class:`torch.cuda.ShortTensor`
32-bit integer (signed)    ``torch.int32`` or ``torch.int``              :class:`torch.IntTensor`      :class:`torch.cuda.IntTensor`
64-bit integer (signed)    ``torch.int64`` or ``torch.long``             :class:`torch.LongTensor`     :class:`torch.cuda.LongTensor`
Boolean                    ``torch.bool``                                :class:`torch.BoolTensor`     :class:`torch.cuda.BoolTensor`
========================   ===========================================   ===========================   ================================

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
The tensor class provides multi-dimensional, `strided <https://en.wikipedia.org/wiki/Stride_of_an_array>`_
view of a storage and defines numeric operations on it.

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

   .. automethod:: new_tensor
   .. automethod:: new_full
   .. automethod:: new_empty
   .. automethod:: new_ones
   .. automethod:: new_zeros

   .. autoattribute:: is_cuda
   .. autoattribute:: device
   .. autoattribute:: grad
   .. autoattribute:: ndim
   .. autoattribute:: T

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
   .. automethod:: allclose
   .. automethod:: apply_
   .. automethod:: argmax
   .. automethod:: argmin
   .. automethod:: argsort
   .. automethod:: asin
   .. automethod:: asin_
   .. automethod:: as_strided
   .. automethod:: atan
   .. automethod:: atan2
   .. automethod:: atan2_
   .. automethod:: atan_
   .. automethod:: backward
   .. automethod:: baddbmm
   .. automethod:: baddbmm_
   .. automethod:: bernoulli
   .. automethod:: bernoulli_
   .. automethod:: bfloat16
   .. automethod:: bincount
   .. automethod:: bitwise_not
   .. automethod:: bitwise_not_
   .. automethod:: bmm
   .. automethod:: bool
   .. automethod:: byte
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
   .. automethod:: dequantize
   .. automethod:: det
   .. automethod:: dense_dim
   .. automethod:: detach
   .. automethod:: detach_
   .. automethod:: diag
   .. automethod:: diag_embed
   .. automethod:: diagflat
   .. automethod:: diagonal
   .. automethod:: fill_diagonal_
   .. automethod:: digamma
   .. automethod:: digamma_
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
   .. automethod:: fft
   .. automethod:: fill_
   .. automethod:: flatten
   .. automethod:: flip
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
   .. automethod:: get_device
   .. automethod:: gt
   .. automethod:: gt_
   .. automethod:: half
   .. automethod:: hardshrink
   .. automethod:: histc
   .. automethod:: ifft
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
   .. automethod:: int
   .. automethod:: int_repr
   .. automethod:: inverse
   .. automethod:: irfft
   .. automethod:: is_contiguous
   .. automethod:: is_floating_point
   .. automethod:: is_leaf
   .. automethod:: is_pinned
   .. automethod:: is_set_to
   .. automethod:: is_shared
   .. automethod:: is_signed
   .. automethod:: is_sparse
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
   .. automethod:: logsumexp
   .. automethod:: logical_not
   .. automethod:: logical_not_
   .. automethod:: long
   .. automethod:: lstsq
   .. automethod:: lt
   .. automethod:: lt_
   .. automethod:: lu
   .. automethod:: lu_solve
   .. automethod:: map_
   .. automethod:: masked_scatter_
   .. automethod:: masked_scatter
   .. automethod:: masked_fill_
   .. automethod:: masked_fill
   .. automethod:: masked_select
   .. automethod:: matmul
   .. automethod:: matrix_power
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
   .. automethod:: mvlgamma
   .. automethod:: mvlgamma_
   .. automethod:: narrow
   .. automethod:: narrow_copy
   .. automethod:: ndimension
   .. automethod:: ne
   .. automethod:: ne_
   .. automethod:: neg
   .. automethod:: neg_
   .. automethod:: nelement
   .. automethod:: nonzero
   .. automethod:: norm
   .. automethod:: normal_
   .. automethod:: numel
   .. automethod:: numpy
   .. automethod:: orgqr
   .. automethod:: ormqr
   .. automethod:: permute
   .. automethod:: pin_memory
   .. automethod:: pinverse
   .. automethod:: pow
   .. automethod:: pow_
   .. automethod:: prod
   .. automethod:: put_
   .. automethod:: qr
   .. automethod:: qscheme
   .. automethod:: q_scale
   .. automethod:: q_zero_point
   .. automethod:: random_
   .. automethod:: reciprocal
   .. automethod:: reciprocal_
   .. automethod:: register_hook
   .. automethod:: remainder
   .. automethod:: remainder_
   .. automethod:: renorm
   .. automethod:: renorm_
   .. automethod:: repeat
   .. automethod:: repeat_interleave
   .. automethod:: requires_grad
   .. automethod:: requires_grad_
   .. automethod:: reshape
   .. automethod:: reshape_as
   .. automethod:: resize_
   .. automethod:: resize_as_
   .. automethod:: retain_grad
   .. automethod:: rfft
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
   .. automethod:: sin
   .. automethod:: sin_
   .. automethod:: sinh
   .. automethod:: sinh_
   .. automethod:: size
   .. automethod:: slogdet
   .. automethod:: solve
   .. automethod:: sort
   .. automethod:: split
   .. automethod:: sparse_mask
   .. automethod:: sparse_dim
   .. automethod:: sqrt
   .. automethod:: sqrt_
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
   .. automethod:: sum
   .. automethod:: sum_to_size
   .. automethod:: svd
   .. automethod:: symeig
   .. automethod:: t
   .. automethod:: t_
   .. automethod:: to
   .. automethod:: to_mkldnn
   .. automethod:: take
   .. automethod:: tan
   .. automethod:: tan_
   .. automethod:: tanh
   .. automethod:: tanh_
   .. automethod:: tolist
   .. automethod:: topk
   .. automethod:: to_sparse
   .. automethod:: trace
   .. automethod:: transpose
   .. automethod:: transpose_
   .. automethod:: triangular_solve
   .. automethod:: tril
   .. automethod:: tril_
   .. automethod:: triu
   .. automethod:: triu_
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
   .. automethod:: var
   .. automethod:: view
   .. automethod:: view_as
   .. automethod:: where
   .. automethod:: zero_

.. class:: BoolTensor()

   The following methods are unique to :class:`torch.BoolTensor`.

   .. automethod:: all
   .. automethod:: any
