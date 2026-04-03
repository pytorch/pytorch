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
    is_inference
    is_neg
    is_nonzero
    is_same_size
    is_signed
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
    empty_permuted
    empty_quantized
    empty_strided
    full
    full_like
    quantize_per_tensor
    quantize_per_tensor_dynamic
    quantize_per_channel
    dequantize
    complex
    polar
    scalar_tensor
    heaviside

.. _indexing-slicing-joining:

Indexing, Slicing, Joining, Mutating Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:

    adjoint
    alias_copy
    argwhere
    as_strided_copy
    as_strided_scatter
    cat
    ccol_indices_copy
    col_indices_copy
    concat
    concatenate
    conj
    chunk
    crow_indices_copy
    detach
    detach_copy
    diagonal_copy
    dsplit
    column_stack
    dstack
    expand_copy
    fill
    gather
    hsplit
    hstack
    index_add
    index_copy
    index_put_
    index_reduce
    index_select
    indices_copy
    masked_fill
    masked_select
    movedim
    moveaxis
    narrow
    narrow_copy
    nonzero
    nonzero_static
    permute
    permute_copy
    put
    reshape
    row_indices_copy
    row_stack
    select
    select_copy
    scatter
    diagonal_scatter
    select_scatter
    slice_copy
    slice_inverse
    slice_scatter
    scatter_add
    scatter_reduce
    segment_reduce
    split
    split_copy
    split_with_sizes_copy
    squeeze
    squeeze_copy
    stack
    swapaxes
    swapdims
    t
    t_copy
    take
    take_along_dim
    tensor_split
    tile
    transpose
    transpose_copy
    unbind
    unbind_copy
    unfold_copy
    unravel_index
    unsqueeze
    unsqueeze_copy
    values_copy
    view_as_complex_copy
    view_as_real_copy
    view_copy
    vsplit
    vstack
    where

.. _accelerators:

Accelerators
----------------------------------
Within the PyTorch repo, we define an "Accelerator" as a :class:`torch.device` that is being used
alongside a CPU to speed up computation. These devices use an asynchronous execution scheme,
using :class:`torch.Stream` and :class:`torch.Event` as their main way to perform synchronization.
We also assume that only one such accelerator can be available at once on a given host. This allows
us to use the current accelerator as the default device for relevant concepts such as pinned memory,
Stream device_type, FSDP, etc.

As of today, accelerator devices are (in no particular order) :doc:`"CUDA" <cuda>`, :doc:`"MTIA" <mtia>`,
:doc:`"XPU" <xpu>`, :doc:`"MPS" <mps>`, "HPU", and PrivateUse1 (many device not in the PyTorch repo itself).

Many tools in the PyTorch Ecosystem use fork to create subprocesses (for example dataloading
or intra-op parallelism), it is thus important to delay as much as possible any
operation that would prevent further forks. This is especially important here as most accelerator's initialization has such effect.
In practice, you should keep in mind that checking :func:`torch.accelerator.current_accelerator`
is a compile-time check by default, it is thus always fork-safe.
On the contrary, passing the ``check_available=True`` flag to this function or calling
:func:`torch.accelerator.is_available()` will usually prevent later fork.

Some backends provide an experimental opt-in option to make the runtime availability
check fork-safe. When using the CUDA device ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` can be
used for example.

.. autosummary::
    :toctree: generated
    :nosignatures:

    Stream
    Event

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

    fork
    get_num_threads
    init_num_threads
    set_num_threads
    get_num_interop_threads
    set_num_interop_threads
    wait

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

Constants
~~~~~~~~~~~~~~~~~~~~~~

======================================= ===========================================
``e``                                       Euler's number, the base of natural logarithms (~2.7183). Alias for :attr:`math.e`.
``inf``                                     A floating-point positive infinity. Alias for :attr:`math.inf`.
``nan``                                     A floating-point "not a number" value. This value is not a legal number. Alias for :attr:`math.nan`.
``pi``                                      The ratio of a circle's circumference to its diameter (~3.1416). Alias for :attr:`math.pi`.
======================================= ===========================================

Pointwise Ops
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    abs_
    absolute
    acos
    acos_
    arccos
    arccos_
    acosh
    acosh_
    arccosh
    arccosh_
    add
    addcdiv
    addcmul
    angle
    asin
    asin_
    arcsin
    arcsin_
    asinh
    asinh_
    arcsinh
    arcsinh_
    atan
    atan_
    arctan
    arctan_
    atanh
    atanh_
    arctanh
    arctanh_
    atan2
    arctan2
    bitwise_not
    bitwise_and
    bitwise_or
    bitwise_xor
    bitwise_left_shift
    bitwise_right_shift
    ceil
    ceil_
    clamp
    clamp_
    clamp_max_
    clamp_min_
    clip
    clip_
    conj_physical
    conj_physical_
    copysign
    cos
    cos_
    cosh
    cosh_
    deg2rad
    deg2rad_
    div
    divide
    digamma
    erf
    erf_
    erfc
    erfc_
    erfinv
    exp
    exp_
    exp2
    exp2_
    expm1
    expm1_
    fake_quantize_per_channel_affine
    fake_quantize_per_tensor_affine
    fill_
    fix
    fix_
    float_power
    floor
    floor_
    floor_divide
    fmod
    frac
    frac_
    frexp
    gradient
    imag
    ldexp
    ldexp_
    lerp
    lgamma
    log
    log_
    log10
    log10_
    log1p
    log1p_
    log2
    log2_
    logaddexp
    logaddexp2
    logical_and
    logical_not
    logical_or
    logical_xor
    logit
    logit_
    hypot
    i0
    i0_
    igamma
    igammac
    mul
    multiply
    mvlgamma
    nan_to_num
    nan_to_num_
    neg
    neg_
    negative
    negative_
    nextafter
    polygamma
    positive
    pow
    quantized_batch_norm
    quantized_max_pool1d
    quantized_max_pool2d
    rad2deg
    rad2deg_
    real
    reciprocal
    reciprocal_
    remainder
    round
    round_
    rsqrt
    rsqrt_
    sigmoid
    sigmoid_
    sign
    sgn
    signbit
    sin
    sin_
    sinc
    sinc_
    sinh
    sinh_
    softmax
    sqrt
    sqrt_
    square
    square_
    sub
    subtract
    tan
    tan_
    tanh
    tanh_
    true_divide
    trunc
    trunc_
    xlogy
    xlogy_
    zero_

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
    norm_except_dim
    nuclear_norm
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
    hash_tensor

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

    adaptive_avg_pool1d
    adaptive_max_pool1d
    affine_grid_generator
    alpha_dropout
    alpha_dropout_
    as_strided_
    atleast_1d
    atleast_2d
    atleast_3d
    avg_pool1d
    batch_norm_backward_elemt
    batch_norm_backward_reduce
    batch_norm_elemt
    batch_norm_gather_stats
    batch_norm_gather_stats_with_counts
    batch_norm_stats
    batch_norm_update_stats
    bilinear
    bincount
    binomial
    block_diag
    broadcast_tensors
    broadcast_to
    broadcast_shapes
    bucketize
    cartesian_prod
    cdist
    celu_
    channel_shuffle
    choose_qparams_optimized
    clone
    combinations
    conv1d
    conv3d
    conv_tbc
    conv_transpose1d
    conv_transpose2d
    conv_transpose3d
    convolution
    corrcoef
    cosine_embedding_loss
    cosine_similarity
    cov
    cross
    ctc_loss
    cudnn_affine_grid_generator
    cudnn_batch_norm
    cudnn_convolution
    cudnn_convolution_add_relu
    cudnn_convolution_relu
    cudnn_convolution_transpose
    cudnn_grid_sampler
    cudnn_is_acceptable
    cummax
    cummin
    cumprod
    cumsum
    detach_
    diag
    diag_embed
    diagflat
    diagonal
    diff
    dropout_
    einsum
    embedding
    embedding_renorm_
    fbgemm_linear_fp16_weight
    fbgemm_linear_fp16_weight_fp32_activation
    fbgemm_linear_int8_weight
    fbgemm_linear_int8_weight_fp32_activation
    fbgemm_linear_quantize_weight
    fbgemm_pack_gemm_matrix_fp16
    fbgemm_pack_quantized_matrix
    feature_alpha_dropout
    feature_alpha_dropout_
    feature_dropout
    feature_dropout_
    flatten
    flip
    fliplr
    flipud
    fused_moving_avg_obs_fake_quant
    gcd
    gcd_
    grid_sampler_2d
    grid_sampler_3d
    group_norm
    gru
    gru_cell
    hardshrink
    hinge_embedding_loss
    histc
    histogram
    histogramdd
    instance_norm
    int_repr
    kl_div
    kron
    lcm
    lcm_
    logcumsumexp
    lstm
    lstm_cell
    margin_ranking_loss
    max_pool1d
    max_pool3d
    meshgrid
    miopen_batch_norm
    miopen_convolution
    miopen_convolution_add_relu
    miopen_convolution_relu
    miopen_convolution_transpose
    miopen_ctc_loss
    miopen_depthwise_convolution
    miopen_rnn
    mkldnn_adaptive_avg_pool2d
    mkldnn_convolution
    mkldnn_linear_backward_weights
    mkldnn_max_pool2d
    mkldnn_max_pool3d
    mkldnn_rnn_layer
    native_batch_norm
    native_channel_shuffle
    native_group_norm
    native_layer_norm
    native_norm
    pairwise_distance
    pdist
    pixel_unshuffle
    poisson_nll_loss
    prelu
    q_per_channel_axis
    q_per_channel_scales
    q_per_channel_zero_points
    q_scale
    q_zero_point
    quantized_gru_cell
    quantized_lstm_cell
    quantized_max_pool3d
    quantized_rnn_relu_cell
    quantized_rnn_tanh_cell
    ravel
    relu_
    renorm
    repeat_interleave
    resize_as_
    resize_as_sparse_
    rms_norm
    rnn_relu
    rnn_relu_cell
    rnn_tanh
    rnn_tanh_cell
    roll
    rot90
    rrelu
    rrelu_
    rsub
    searchsorted
    selu
    selu_
    tensordot
    threshold
    threshold_
    trace
    tril
    tril_indices
    triu
    triu_indices
    triplet_margin_loss
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
    addmv_
    addr
    baddbmm
    bmm
    chain_matmul
    cholesky
    cholesky_inverse
    cholesky_solve
    dot
    dsmm
    geqrf
    ger
    hsmm
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
    saddmm
    spmm
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
    _foreach_clone
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

    autocast_decrement_nesting
    autocast_increment_nesting
    clear_autocast_cache
    compiled_with_cxx11_abi
    get_autocast_cpu_dtype
    get_autocast_dtype
    get_autocast_gpu_dtype
    get_autocast_ipu_dtype
    get_autocast_xla_dtype
    get_device
    get_device_module
    import_ir_module
    import_ir_module_from_buffer
    is_anomaly_check_nan_enabled
    is_anomaly_enabled
    is_autocast_cache_enabled
    is_autocast_cpu_enabled
    is_autocast_enabled
    is_autocast_ipu_enabled
    is_autocast_xla_enabled
    is_distributed
    is_vulkan_available
    merge_type_from_type_comment
    parse_ir
    parse_schema
    parse_type_comment
    result_type
    can_cast
    promote_types
    read_vitals
    set_anomaly_enabled
    set_autocast_cache_enabled
    set_autocast_cpu_dtype
    set_autocast_cpu_enabled
    set_autocast_dtype
    set_autocast_enabled
    set_autocast_gpu_dtype
    set_autocast_ipu_dtype
    set_autocast_ipu_enabled
    set_autocast_xla_dtype
    set_autocast_xla_enabled
    set_vital
    use_deterministic_algorithms
    are_deterministic_algorithms_enabled
    is_deterministic_algorithms_warn_only_enabled
    set_deterministic_debug_mode
    get_deterministic_debug_mode
    set_float32_matmul_precision
    get_float32_matmul_precision
    set_warn_always
    is_warn_always_enabled
    vitals_enabled
    vmap
    _assert
    typename

Type Information
----------------
.. autoclass:: TensorType
    :no-members:

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

    sym_constrain_range
    sym_constrain_range_for_size
    sym_float
    sym_fresh_size
    sym_int
    sym_max
    sym_min
    sym_not
    sym_ite
    sym_sqrt
    sym_sum

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

`torch.compile documentation <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html>`__

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
.. py:module:: torch.quasirandom
.. py:module:: torch.return_types
.. automodule:: torch.serialization

.. currentmodule:: torch.serialization

.. autosummary::
    :toctree: generated
    :nosignatures:

    StorageType

.. py:module:: torch.serialization
   :noindex:
.. py:module:: torch.signal.windows.windows
.. py:module:: torch.sparse.semi_structured
.. py:module:: torch.storage
.. py:module:: torch.torch_version
.. py:module:: torch.types
.. py:module:: torch.version

.. Compiler configuration module - documented in torch.compiler.config.md
.. py:module:: torch.compiler.config
   :noindex:

.. Hidden aliases (e.g. torch.functional.broadcast_tensors()). We want `torch.broadcast_tensors()` to
   be visible only.
.. toctree::
    :hidden:

    torch.aliases.md
