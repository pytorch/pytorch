.. role:: hidden
    :class: hidden-section

torch.backends
==============
.. automodule:: torch.backends

`torch.backends` controls the behavior of various backends that PyTorch supports.

These backends include:

- ``torch.backends.cpu``
- ``torch.backends.cuda``
- ``torch.backends.cudnn``
- ``torch.backends.cusparselt``
- ``torch.backends.mha``
- ``torch.backends.mps``
- ``torch.backends.mkl``
- ``torch.backends.mkldnn``
- ``torch.backends.nnpack``
- ``torch.backends.openmp``
- ``torch.backends.opt_einsum``
- ``torch.backends.xeon``

torch.backends.cpu
^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.cpu

.. autofunction::  torch.backends.cpu.get_cpu_capability

torch.backends.cuda
^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.cuda

.. autofunction::  torch.backends.cuda.is_built

.. currentmodule:: torch.backends.cuda.matmul
.. attribute::  allow_tf32

    A :class:`bool` that controls whether TensorFloat-32 tensor cores may be used in matrix
    multiplications on Ampere or newer GPUs. See :ref:`tf32_on_ampere`.

.. attribute::  allow_fp16_reduced_precision_reduction

    A :class:`bool` that controls whether reduced precision reductions (e.g., with fp16 accumulation type) are allowed with fp16 GEMMs.

.. attribute::  allow_bf16_reduced_precision_reduction

    A :class:`bool` that controls whether reduced precision reductions are allowed with bf16 GEMMs.

.. currentmodule:: torch.backends.cuda
.. attribute::  cufft_plan_cache

    ``cufft_plan_cache`` contains the cuFFT plan caches for each CUDA device.
    Query a specific device `i`'s cache via `torch.backends.cuda.cufft_plan_cache[i]`.

    .. currentmodule:: torch.backends.cuda.cufft_plan_cache
    .. attribute::  size

        A readonly :class:`int` that shows the number of plans currently in a cuFFT plan cache.

    .. attribute::  max_size

        A :class:`int` that controls the capacity of a cuFFT plan cache.

    .. method::  clear()

        Clears a cuFFT plan cache.

.. autofunction:: torch.backends.cuda.preferred_blas_library

.. autofunction:: torch.backends.cuda.preferred_linalg_library

.. autoclass:: torch.backends.cuda.SDPAParams

.. autofunction:: torch.backends.cuda.flash_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_mem_efficient_sdp

.. autofunction:: torch.backends.cuda.mem_efficient_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_flash_sdp

.. autofunction:: torch.backends.cuda.math_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_math_sdp

.. autofunction:: torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed

.. autofunction:: torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp

.. autofunction:: torch.backends.cuda.cudnn_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_cudnn_sdp

.. autofunction:: torch.backends.cuda.is_flash_attention_available

.. autofunction:: torch.backends.cuda.can_use_flash_attention

.. autofunction:: torch.backends.cuda.can_use_efficient_attention

.. autofunction:: torch.backends.cuda.can_use_cudnn_attention

.. autofunction:: torch.backends.cuda.sdp_kernel

torch.backends.cudnn
^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.cudnn

.. autofunction:: torch.backends.cudnn.version

.. autofunction:: torch.backends.cudnn.is_available

.. attribute::  enabled

    A :class:`bool` that controls whether cuDNN is enabled.

.. attribute::  allow_tf32

    A :class:`bool` that controls where TensorFloat-32 tensor cores may be used in cuDNN
    convolutions on Ampere or newer GPUs. See :ref:`tf32_on_ampere`.

.. attribute::  deterministic

    A :class:`bool` that, if True, causes cuDNN to only use deterministic convolution algorithms.
    See also :func:`torch.are_deterministic_algorithms_enabled` and
    :func:`torch.use_deterministic_algorithms`.

.. attribute::  benchmark

    A :class:`bool` that, if True, causes cuDNN to benchmark multiple convolution algorithms
    and select the fastest.

.. attribute::  benchmark_limit

    A :class:`int` that specifies the maximum number of cuDNN convolution algorithms to try when
    `torch.backends.cudnn.benchmark` is True. Set `benchmark_limit` to zero to try every
    available algorithm. Note that this setting only affects convolutions dispatched via the
    cuDNN v8 API.

.. py:module:: torch.backends.cudnn.rnn

torch.backends.cusparselt
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.cusparselt

.. autofunction:: torch.backends.cusparselt.version

.. autofunction:: torch.backends.cusparselt.is_available

torch.backends.mha
^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.mha

.. autofunction::  torch.backends.mha.get_fastpath_enabled
.. autofunction::  torch.backends.mha.set_fastpath_enabled


torch.backends.mps
^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.mps

.. autofunction::  torch.backends.mps.is_available

.. autofunction::  torch.backends.mps.is_built


torch.backends.mkl
^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.mkl

.. autofunction::  torch.backends.mkl.is_available

.. autoclass::  torch.backends.mkl.verbose


torch.backends.mkldnn
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.mkldnn

.. autofunction::  torch.backends.mkldnn.is_available

.. autoclass::  torch.backends.mkldnn.verbose

torch.backends.nnpack
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.nnpack

.. autofunction::  torch.backends.nnpack.is_available

.. autofunction::  torch.backends.nnpack.flags

.. autofunction::  torch.backends.nnpack.set_flags

torch.backends.openmp
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.openmp

.. autofunction::  torch.backends.openmp.is_available

.. Docs for other backends need to be added here.
.. Automodules are just here to ensure checks run but they don't actually
.. add anything to the rendered page for now.
.. py:module:: torch.backends.quantized
.. py:module:: torch.backends.xnnpack
.. py:module:: torch.backends.kleidiai


torch.backends.opt_einsum
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.opt_einsum

.. autofunction:: torch.backends.opt_einsum.is_available

.. autofunction:: torch.backends.opt_einsum.get_opt_einsum

.. attribute::  enabled

    A :class:`bool` that controls whether opt_einsum is enabled (``True`` by default). If so,
    torch.einsum will use opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html)
    if available to calculate an optimal path of contraction for faster performance.

    If opt_einsum is not available, torch.einsum will fall back to the default contraction path
    of left to right.

.. attribute::  strategy

    A :class:`str` that specifies which strategies to try when ``torch.backends.opt_einsum.enabled``
    is ``True``. By default, torch.einsum will try the "auto" strategy, but the "greedy" and "optimal"
    strategies are also supported. Note that the "optimal" strategy is factorial on the number of
    inputs as it tries all possible paths. See more details in opt_einsum's docs
    (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html).


torch.backends.xeon
^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.xeon
.. py:module:: torch.backends.xeon.run_cpu
