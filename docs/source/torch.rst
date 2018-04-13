torch
===================================
.. automodule:: torch

Tensors
----------------------------------
.. autofunction:: is_tensor
.. autofunction:: is_storage
.. autofunction:: set_default_tensor_type
.. autofunction:: numel
.. autofunction:: set_printoptions
.. autofunction:: set_flush_denormal


Creation Ops
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: eye
.. autofunction:: from_numpy
.. autofunction:: linspace
.. autofunction:: logspace
.. autofunction:: ones
.. autofunction:: ones_like
.. autofunction:: arange
.. autofunction:: range
.. autofunction:: zeros
.. autofunction:: zeros_like
.. autofunction:: empty_like

Indexing, Slicing, Joining, Mutating Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: cat
.. autofunction:: chunk
.. autofunction:: gather
.. autofunction:: index_select
.. autofunction:: masked_select
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


Random sampling
----------------------------------
.. autofunction:: manual_seed
.. autofunction:: initial_seed
.. autofunction:: get_rng_state
.. autofunction:: set_rng_state
.. autodata:: default_generator
.. autofunction:: bernoulli
.. autofunction:: multinomial
.. autofunction:: normal
.. autofunction:: rand
.. autofunction:: randint
.. autofunction:: randn
.. autofunction:: randperm

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


Serialization
----------------------------------
.. autofunction:: save
.. autofunction:: load


Parallelism
----------------------------------
.. autofunction:: get_num_threads
.. autofunction:: set_num_threads


Math operations
----------------------------------

Pointwise Ops
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: abs
.. autofunction:: acos
.. autofunction:: add
.. autofunction:: addcdiv
.. autofunction:: addcmul
.. autofunction:: asin
.. autofunction:: atan
.. autofunction:: atan2
.. autofunction:: ceil
.. autofunction:: clamp
.. autofunction:: cos
.. autofunction:: cosh
.. autofunction:: div
.. autofunction:: erf
.. autofunction:: erfinv
.. autofunction:: exp
.. autofunction:: expm1
.. autofunction:: floor
.. autofunction:: fmod
.. autofunction:: frac
.. autofunction:: lerp
.. autofunction:: log
.. autofunction:: log10
.. autofunction:: log1p
.. autofunction:: log2
.. autofunction:: mul
.. autofunction:: neg
.. autofunction:: pow
.. autofunction:: reciprocal
.. autofunction:: remainder
.. autofunction:: round
.. autofunction:: rsqrt
.. autofunction:: sigmoid
.. autofunction:: sign
.. autofunction:: sin
.. autofunction:: sinh
.. autofunction:: sqrt
.. autofunction:: tan
.. autofunction:: tanh
.. autofunction:: trunc


Reduction Ops
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: argmax
.. autofunction:: argmin
.. autofunction:: cumprod
.. autofunction:: cumsum
.. autofunction:: dist
.. autofunction:: mean
.. autofunction:: median
.. autofunction:: mode
.. autofunction:: norm
.. autofunction:: prod
.. autofunction:: std
.. autofunction:: sum
.. autofunction:: unique
.. autofunction:: var


Comparison Ops
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: eq
.. autofunction:: equal
.. autofunction:: ge
.. autofunction:: gt
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
.. autofunction:: hann_window
.. autofunction:: hamming_window
.. autofunction:: bartlett_window


Other Operations
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: cross
.. autofunction:: diag
.. autofunction:: diagflat
.. autofunction:: diagonal
.. autofunction:: histc
.. autofunction:: renorm
.. autofunction:: trace
.. autofunction:: tril
.. autofunction:: triu


BLAS and LAPACK Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: addbmm
.. autofunction:: addmm
.. autofunction:: addmv
.. autofunction:: addr
.. autofunction:: baddbmm
.. autofunction:: bmm
.. autofunction:: btrifact
.. autofunction:: btrifact_with_info
.. autofunction:: btrisolve
.. autofunction:: btriunpack
.. autofunction:: dot
.. autofunction:: eig
.. autofunction:: gels
.. autofunction:: geqrf
.. autofunction:: ger
.. autofunction:: gesv
.. autofunction:: inverse
.. autofunction:: det
.. autofunction:: logdet
.. autofunction:: slogdet
.. autofunction:: matmul
.. autofunction:: mm
.. autofunction:: mv
.. autofunction:: orgqr
.. autofunction:: ormqr
.. autofunction:: potrf
.. autofunction:: potri
.. autofunction:: potrs
.. autofunction:: pstrf
.. autofunction:: qr
.. autofunction:: svd
.. autofunction:: symeig
.. autofunction:: trtrs
