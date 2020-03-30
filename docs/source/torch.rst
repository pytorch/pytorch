torch
===================================
.. automodule:: torch

Tensors
----------------------------------
.. autofunction:: is_tensor
.. autofunction:: is_storage
.. autofunction:: is_complex
.. autofunction:: is_floating_point
.. autofunction:: set_default_dtype
.. autofunction:: get_default_dtype
.. autofunction:: set_default_tensor_type
.. autofunction:: numel
.. autofunction:: set_printoptions
.. autofunction:: set_flush_denormal

.. _tensor-creation-ops:

Creation Ops
~~~~~~~~~~~~~~~~~~~~~~

.. note::
    Random sampling creation ops are listed under :ref:`random-sampling` and
    include:
    :func:`torch.rand`
    :func:`torch.rand_like`
    :func:`torch.randn`
    :func:`torch.randn_like`
    :func:`torch.randint`
    :func:`torch.randint_like`
    :func:`torch.randperm`
    You may also use :func:`torch.empty` with the :ref:`inplace-random-sampling`
    methods to create :class:`torch.Tensor` s with values sampled from a broader
    range of distributions.

.. autofunction:: tensor
.. autofunction:: sparse_coo_tensor
.. autofunction:: as_tensor
.. autofunction:: as_strided
.. autofunction:: from_numpy
.. autofunction:: zeros
.. autofunction:: zeros_like
.. autofunction:: ones
.. autofunction:: ones_like
.. autofunction:: arange
.. autofunction:: range
.. autofunction:: linspace
.. autofunction:: logspace
.. autofunction:: eye
.. autofunction:: empty
.. autofunction:: empty_like
.. autofunction:: empty_strided
.. autofunction:: full
.. autofunction:: full_like
.. autofunction:: quantize_per_tensor
.. autofunction:: quantize_per_channel
.. autofunction:: dequantize

Indexing, Slicing, Joining, Mutating Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: cat
.. autofunction:: chunk
.. autofunction:: gather
.. autofunction:: index_select
.. autofunction:: masked_select
.. autofunction:: narrow
.. autofunction:: nonzero
.. autofunction:: reshape
.. autofunction:: split
.. autofunction:: squeeze
.. autofunction:: stack
.. autofunction:: t
.. autofunction:: take
.. autofunction:: transpose
.. autofunction:: unbind
.. autofunction:: unsqueeze
.. autofunction:: where

.. _generators:

Generators
----------------------------------
.. autoclass:: torch._C.Generator
   :members:

.. _random-sampling:

Random sampling
----------------------------------
.. autofunction:: seed
.. autofunction:: manual_seed
.. autofunction:: initial_seed
.. autofunction:: get_rng_state
.. autofunction:: set_rng_state
.. autoattribute:: torch.default_generator
   :annotation:  Returns the default CPU torch.Generator

.. The following doesn't actually seem to exist.
   https://github.com/pytorch/pytorch/issues/27780
   .. autoattribute:: torch.cuda.default_generators
      :annotation:  If cuda is available, returns a tuple of default CUDA torch.Generator-s.
                    The number of CUDA torch.Generator-s returned is equal to the number of
                    GPUs available in the system.
.. autofunction:: bernoulli
.. autofunction:: multinomial
.. autofunction:: normal
.. autofunction:: poisson
.. autofunction:: rand
.. autofunction:: rand_like
.. autofunction:: randint
.. autofunction:: randint_like
.. autofunction:: randn
.. autofunction:: randn_like
.. autofunction:: randperm

.. _inplace-random-sampling:

In-place random sampling
~~~~~~~~~~~~~~~~~~~~~~~~

There are a few more in-place random sampling functions defined on Tensors as well. Click through to refer to their documentation:

- :func:`torch.Tensor.bernoulli_` - in-place version of :func:`torch.bernoulli`
- :func:`torch.Tensor.cauchy_` - numbers drawn from the Cauchy distribution
- :func:`torch.Tensor.exponential_` - numbers drawn from the exponential distribution
- :func:`torch.Tensor.geometric_` - elements drawn from the geometric distribution
- :func:`torch.Tensor.log_normal_` - samples from the log-normal distribution
- :func:`torch.Tensor.normal_` - in-place version of :func:`torch.normal`
- :func:`torch.Tensor.random_` - numbers sampled from the discrete uniform distribution
- :func:`torch.Tensor.uniform_` - numbers sampled from the continuous uniform distribution

Quasi-random sampling
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torch.quasirandom.SobolEngine
    :members:
    :exclude-members: MAXBIT, MAXDIM
    :undoc-members:

Serialization
----------------------------------
.. autofunction:: save
.. autofunction:: load


Parallelism
----------------------------------
.. autofunction:: get_num_threads
.. autofunction:: set_num_threads
.. autofunction:: get_num_interop_threads
.. autofunction:: set_num_interop_threads

Locally disabling gradient computation
--------------------------------------
The context managers :func:`torch.no_grad`, :func:`torch.enable_grad`, and
:func:`torch.set_grad_enabled` are helpful for locally disabling and enabling
gradient computation. See :ref:`locally-disable-grad` for more details on
their usage.  These context managers are thread local, so they won't
work if you send work to another thread using the ``threading`` module, etc.

Examples::

  >>> x = torch.zeros(1, requires_grad=True)
  >>> with torch.no_grad():
  ...     y = x * 2
  >>> y.requires_grad
  False

  >>> is_train = False
  >>> with torch.set_grad_enabled(is_train):
  ...     y = x * 2
  >>> y.requires_grad
  False

  >>> torch.set_grad_enabled(True)  # this can also be used as a function
  >>> y = x * 2
  >>> y.requires_grad
  True

  >>> torch.set_grad_enabled(False)
  >>> y = x * 2
  >>> y.requires_grad
  False

.. autofunction:: no_grad
.. autofunction:: enable_grad
.. autofunction:: set_grad_enabled

Math operations
----------------------------------

Pointwise Ops
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: abs
.. autofunction:: acos
.. autofunction:: add
.. autofunction:: addcdiv
.. autofunction:: addcmul
.. autofunction:: angle
.. autofunction:: asin
.. autofunction:: atan
.. autofunction:: atan2
.. autofunction:: bitwise_not
.. autofunction:: bitwise_and
.. autofunction:: bitwise_or
.. autofunction:: bitwise_xor
.. autofunction:: ceil
.. autofunction:: clamp
.. autofunction:: conj
.. autofunction:: cos
.. autofunction:: cosh
.. autofunction:: div
.. autofunction:: digamma
.. autofunction:: erf
.. autofunction:: erfc
.. autofunction:: erfinv
.. autofunction:: exp
.. autofunction:: expm1
.. autofunction:: floor
.. autofunction:: floor_divide
.. autofunction:: fmod
.. autofunction:: frac
.. autofunction:: imag
.. autofunction:: lerp
.. autofunction:: lgamma
.. autofunction:: log
.. autofunction:: log10
.. autofunction:: log1p
.. autofunction:: log2
.. autofunction:: logical_and
.. autofunction:: logical_not
.. autofunction:: logical_or
.. autofunction:: logical_xor
.. autofunction:: mul
.. autofunction:: mvlgamma
.. autofunction:: neg
.. autofunction:: polygamma
.. autofunction:: pow
.. autofunction:: real
.. autofunction:: reciprocal
.. autofunction:: remainder
.. autofunction:: round
.. autofunction:: rsqrt
.. autofunction:: sigmoid
.. autofunction:: sign
.. autofunction:: sin
.. autofunction:: sinh
.. autofunction:: sqrt
.. autofunction:: square
.. autofunction:: tan
.. autofunction:: tanh
.. autofunction:: true_divide
.. autofunction:: trunc


Reduction Ops
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: argmax
.. autofunction:: argmin
.. autofunction:: dist
.. autofunction:: logsumexp
.. autofunction:: mean
.. autofunction:: median
.. autofunction:: mode
.. autofunction:: norm
.. autofunction:: prod
.. autofunction:: std
.. autofunction:: std_mean
.. autofunction:: sum
.. autofunction:: unique
.. autofunction:: unique_consecutive
.. autofunction:: var
.. autofunction:: var_mean


Comparison Ops
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: allclose
.. autofunction:: argsort
.. autofunction:: eq
.. autofunction:: equal
.. autofunction:: ge
.. autofunction:: gt
.. autofunction:: isfinite
.. autofunction:: isinf
.. autofunction:: isnan
.. autofunction:: kthvalue
.. autofunction:: le
.. autofunction:: lt
.. autofunction:: max
.. autofunction:: min
.. autofunction:: ne
.. autofunction:: sort
.. autofunction:: topk


Spectral Ops
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: fft
.. autofunction:: ifft
.. autofunction:: rfft
.. autofunction:: irfft
.. autofunction:: stft
.. autofunction:: bartlett_window
.. autofunction:: blackman_window
.. autofunction:: hamming_window
.. autofunction:: hann_window


Other Operations
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: bincount
.. autofunction:: broadcast_tensors
.. autofunction:: cartesian_prod
.. autofunction:: cdist
.. autofunction:: combinations
.. autofunction:: cross
.. autofunction:: cummax
.. autofunction:: cummin
.. autofunction:: cumprod
.. autofunction:: cumsum
.. autofunction:: diag
.. autofunction:: diag_embed
.. autofunction:: diagflat
.. autofunction:: diagonal
.. autofunction:: einsum
.. autofunction:: flatten
.. autofunction:: flip
.. autofunction:: rot90
.. autofunction:: histc
.. autofunction:: meshgrid
.. autofunction:: renorm
.. autofunction:: repeat_interleave
.. autofunction:: roll
.. autofunction:: tensordot
.. autofunction:: trace
.. autofunction:: tril
.. autofunction:: tril_indices
.. autofunction:: triu
.. autofunction:: triu_indices


BLAS and LAPACK Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: addbmm
.. autofunction:: addmm
.. autofunction:: addmv
.. autofunction:: addr
.. autofunction:: baddbmm
.. autofunction:: bmm
.. autofunction:: chain_matmul
.. autofunction:: cholesky
.. autofunction:: cholesky_inverse
.. autofunction:: cholesky_solve
.. autofunction:: dot
.. autofunction:: eig
.. autofunction:: geqrf
.. autofunction:: ger
.. autofunction:: inverse
.. autofunction:: det
.. autofunction:: logdet
.. autofunction:: slogdet
.. autofunction:: lstsq
.. autofunction:: lu
.. autofunction:: lu_solve
.. autofunction:: lu_unpack
.. autofunction:: matmul
.. autofunction:: matrix_power
.. autofunction:: matrix_rank
.. autofunction:: mm
.. autofunction:: mv
.. autofunction:: orgqr
.. autofunction:: ormqr
.. autofunction:: pinverse
.. autofunction:: qr
.. autofunction:: solve
.. autofunction:: svd
.. autofunction:: svd_lowrank
.. autofunction:: pca_lowrank
.. autofunction:: symeig
.. autofunction:: lobpcg
.. autofunction:: trapz
.. autofunction:: triangular_solve


Utilities
----------------------------------
.. autofunction:: compiled_with_cxx11_abi
.. autofunction:: result_type
.. autofunction:: can_cast
.. autofunction:: promote_types
