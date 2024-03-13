torch
=====
.. automodule:: torch
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
    set_default_device
    get_default_device
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
    sparse_csr_tensor
    sparse_csc_tensor
    sparse_bsr_tensor
    sparse_bsc_tensor
    asarray
    as_tensor
    as_strided
    from_file
    from_numpy
    from_dlpack
    frombuffer
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

    adjoint
    argwhere
    cat
    concat
    concatenate
    conj
    chunk
    dsplit
    column_stack
    dstack
    gather
    hsplit
    hstack
    index_add
    index_copy
    index_reduce
    index_select
    masked_select
    movedim
    moveaxis
    narrow
    narrow_copy
    nonzero
    permute
    reshape
    row_stack
    select
    scatter
    diagonal_scatter
    select_scatter
    slice_scatter
    scatter_add
    scatter_reduce
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
    unravel_index
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

.. _torch-rst-local-disable-grad:

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
    autograd.grad_mode.set_grad_enabled
    is_grad_enabled
    autograd.grad_mode.inference_mode
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
    arctan2
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
    quantized_batch_norm
    quantized_max_pool1d
    quantized_max_pool2d
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
    softmax
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
    aminmax
    all
    any
    max
    min
    dist
    logsumexp
    mean
    nanmean
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
    histogramdd
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
    unflatten
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
    geqrf
    ger
    inner
    inverse
    det
    logdet
    slogdet
    lu
    lu_solve
    lu_unpack
    matmul
    matrix_power
    matrix_exp
    mm
    mv
    orgqr
    ormqr
    outer
    pinverse
    qr
    svd
    svd_lowrank
    pca_lowrank
    lobpcg
    trapz
    trapezoid
    cumulative_trapezoid
    triangular_solve
    vdot

Foreach Operations
~~~~~~~~~~~~~~~~~~

.. warning::
    This API is in beta and subject to future changes.
    Forward-mode AD is not supported.

.. autosummary::
    :toctree: generated
    :nosignatures:

    _foreach_abs
    _foreach_abs_
    _foreach_acos
    _foreach_acos_
    _foreach_asin
    _foreach_asin_
    _foreach_atan
    _foreach_atan_
    _foreach_ceil
    _foreach_ceil_
    _foreach_cos
    _foreach_cos_
    _foreach_cosh
    _foreach_cosh_
    _foreach_erf
    _foreach_erf_
    _foreach_erfc
    _foreach_erfc_
    _foreach_exp
    _foreach_exp_
    _foreach_expm1
    _foreach_expm1_
    _foreach_floor
    _foreach_floor_
    _foreach_log
    _foreach_log_
    _foreach_log10
    _foreach_log10_
    _foreach_log1p
    _foreach_log1p_
    _foreach_log2
    _foreach_log2_
    _foreach_neg
    _foreach_neg_
    _foreach_tan
    _foreach_tan_
    _foreach_sin
    _foreach_sin_
    _foreach_sinh
    _foreach_sinh_
    _foreach_round
    _foreach_round_
    _foreach_sqrt
    _foreach_sqrt_
    _foreach_lgamma
    _foreach_lgamma_
    _foreach_frac
    _foreach_frac_
    _foreach_reciprocal
    _foreach_reciprocal_
    _foreach_sigmoid
    _foreach_sigmoid_
    _foreach_trunc
    _foreach_trunc_
    _foreach_zero_

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
    is_deterministic_algorithms_warn_only_enabled
    set_deterministic_debug_mode
    get_deterministic_debug_mode
    set_float32_matmul_precision
    get_float32_matmul_precision
    set_warn_always
    is_warn_always_enabled
    vmap
    _assert

Symbolic Numbers
----------------
.. autoclass:: SymInt
    :members:

.. autoclass:: SymFloat
    :members:

.. autoclass:: SymBool
    :members:

.. autosummary::
    :toctree: generated
    :nosignatures:

    sym_float
    sym_int
    sym_max
    sym_min
    sym_not
    sym_ite

Export Path
-------------
.. autosummary::
    :toctree: generated
    :nosignatures:

.. warning::
    This feature is a prototype and may have compatibility breaking changes in the future.

    export
    generated/exportdb/index

Control Flow
------------

.. warning::
    This feature is a prototype and may have compatibility breaking changes in the future.

.. autosummary::
    :toctree: generated
    :nosignatures:

    cond

Optimizations
-------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    compile

`torch.compile documentation <https://pytorch.org/docs/main/compile/index.html>`__

Operator Tags
------------------------------------
.. autoclass:: Tag
    :members:

.. Empty submodules added only for tracking.
.. py:module:: torch.contrib
.. py:module:: torch.utils.backcompat

.. This module is only used internally for ROCm builds.
.. py:module:: torch.utils.hipify

.. This module needs to be documented. Adding here in the meantime
.. for tracking purposes
.. py:module:: torch.utils.model_dump
.. py:module:: torch.utils.viz
.. py:module:: torch.functional
.. py:module:: torch.quasirandom
.. py:module:: torch.return_types
.. py:module:: torch.serialization
.. py:module:: torch.signal.windows.windows
.. py:module:: torch.sparse.semi_structured
.. py:module:: torch.storage
.. py:module:: torch.torch_version
.. py:module:: torch.types
.. py:module:: torch.version
