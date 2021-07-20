torch
=====
The torch package contains data structures for multi-dimensional
tensors and defines mathematical operations over these tensors.
Additionally, it provides many utilities for efficient serializing of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0

.. currentmodule:: torch

Tensors
-------
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_tensor
    is_storage
    is_complex
    is_conj
    is_floating_point
    is_nonzero
    set_default_dtype
    get_default_dtype
    set_default_tensor_type
    numel
    set_printoptions
    set_flush_denormal

.. _tensor-creation-ops:

Creation Ops
~~~~~~~~~~~~

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

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    sparse_coo_tensor
    as_tensor
    as_strided
    from_numpy
    zeros
    zeros_like
    ones
    ones_like
    arange
    range
    linspace
    logspace
    eye
    empty
    empty_like
    empty_strided
    full
    full_like
    quantize_per_tensor
    quantize_per_channel
    dequantize
    complex
    polar
    heaviside

.. _indexing-slicing-joining:

Indexing, Slicing, Joining, Mutating Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    cat
    conj
    chunk
    dsplit
    column_stack
    dstack
    gather
    hsplit
    hstack
    index_select
    masked_select
    movedim
    moveaxis
    narrow
    nonzero
    permute
    reshape
    row_stack
    scatter
    scatter_add
    split
    squeeze
    stack
    swapaxes
    swapdims
    t
    take
    take_along_dim
    tensor_split
    tile
    transpose
    unbind
    unsqueeze
    vsplit
    vstack
    where

.. _generators:

Generators
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    Generator

.. _random-sampling:

Random sampling
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    seed
    manual_seed
    initial_seed
    get_rng_state
    set_rng_state

.. autoattribute:: torch.default_generator
   :annotation:  Returns the default CPU torch.Generator

.. The following doesn't actually seem to exist.
   https://github.com/pytorch/pytorch/issues/27780
   .. autoattribute:: torch.cuda.default_generators
      :annotation:  If cuda is available, returns a tuple of default CUDA torch.Generator-s.
                    The number of CUDA torch.Generator-s returned is equal to the number of
                    GPUs available in the system.
.. autosummary::
    :toctree: generated
    :nosignatures:

    bernoulli
    multinomial
    normal
    poisson
    rand
    rand_like
    randint
    randint_like
    randn
    randn_like
    randperm

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
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: sobolengine.rst

    quasirandom.SobolEngine

Serialization
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    save
    load

Parallelism
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_num_threads
    set_num_threads
    get_num_interop_threads
    set_num_interop_threads

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

.. autosummary::
    :toctree: generated
    :nosignatures:

    no_grad
    enable_grad
    set_grad_enabled
    is_grad_enabled
    inference_mode
    is_inference_mode_enabled

Math operations
---------------

Pointwise Ops
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    absolute
    acos
    arccos
    acosh
    arccosh
    add
    addcdiv
    addcmul
    angle
    asin
    arcsin
    asinh
    arcsinh
    atan
    arctan
    atanh
    arctanh
    atan2
    bitwise_not
    bitwise_and
    bitwise_or
    bitwise_xor
    bitwise_left_shift
    bitwise_right_shift
    ceil
    clamp
    clip
    conj_physical
    copysign
    cos
    cosh
    deg2rad
    div
    divide
    digamma
    erf
    erfc
    erfinv
    exp
    exp2
    expm1
    fake_quantize_per_channel_affine
    fake_quantize_per_tensor_affine
    fix
    float_power
    floor
    floor_divide
    fmod
    frac
    frexp
    gradient
    imag
    ldexp
    lerp
    lgamma
    log
    log10
    log1p
    log2
    logaddexp
    logaddexp2
    logical_and
    logical_not
    logical_or
    logical_xor
    logit
    hypot
    i0
    igamma
    igammac
    mul
    multiply
    mvlgamma
    nan_to_num
    neg
    negative
    nextafter
    polygamma
    positive
    pow
    rad2deg
    real
    reciprocal
    remainder
    round
    rsqrt
    sigmoid
    sign
    sgn
    signbit
    sin
    sinc
    sinh
    sqrt
    square
    sub
    subtract
    tan
    tanh
    true_divide
    trunc
    xlogy

Reduction Ops
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    argmax
    argmin
    amax
    amin
    all
    any
    max
    min
    dist
    logsumexp
    mean
    median
    nanmedian
    mode
    norm
    nansum
    prod
    quantile
    nanquantile
    std
    std_mean
    sum
    unique
    unique_consecutive
    var
    var_mean
    count_nonzero

Comparison Ops
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    allclose
    argsort
    eq
    equal
    ge
    greater_equal
    gt
    greater
    isclose
    isfinite
    isin
    isinf
    isposinf
    isneginf
    isnan
    isreal
    kthvalue
    le
    less_equal
    lt
    less
    maximum
    minimum
    fmax
    fmin
    ne
    not_equal
    sort
    topk
    msort


Spectral Ops
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    stft
    istft
    bartlett_window
    blackman_window
    hamming_window
    hann_window
    kaiser_window


Other Operations
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    atleast_1d
    atleast_2d
    atleast_3d
    bincount
    block_diag
    broadcast_tensors
    broadcast_to
    broadcast_shapes
    bucketize
    cartesian_prod
    cdist
    clone
    combinations
    corrcoef
    cov
    cross
    cummax
    cummin
    cumprod
    cumsum
    diag
    diag_embed
    diagflat
    diagonal
    diff
    einsum
    flatten
    flip
    fliplr
    flipud
    kron
    rot90
    gcd
    histc
    histogram
    meshgrid
    lcm
    logcumsumexp
    ravel
    renorm
    repeat_interleave
    roll
    searchsorted
    tensordot
    trace
    tril
    tril_indices
    triu
    triu_indices
    vander
    view_as_real
    view_as_complex
    resolve_conj
    resolve_neg


BLAS and LAPACK Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    addbmm
    addmm
    addmv
    addr
    baddbmm
    bmm
    chain_matmul
    cholesky
    cholesky_inverse
    cholesky_solve
    dot
    eig
    geqrf
    ger
    inner
    inverse
    det
    logdet
    slogdet
    lstsq
    lu
    lu_solve
    lu_unpack
    matmul
    matrix_power
    matrix_rank
    matrix_exp
    mm
    mv
    orgqr
    ormqr
    outer
    pinverse
    qr
    solve
    svd
    svd_lowrank
    pca_lowrank
    symeig
    lobpcg
    trapz
    triangular_solve
    vdot

Utilities
----------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    compiled_with_cxx11_abi
    result_type
    can_cast
    promote_types
    use_deterministic_algorithms
    are_deterministic_algorithms_enabled
    set_warn_always
    is_warn_always_enabled
    vmap
    _assert
