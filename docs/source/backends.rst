.. role:: hidden
    :class: hidden-section

torch.backends
==============
.. automodule:: torch.backends

`torch.backends` controls the behavior of various backends that PyTorch supports.

These backends include:

- ``torch.backends.cuda``
- ``torch.backends.cudnn``
- ``torch.backends.mps``
- ``torch.backends.mkl``
- ``torch.backends.mkldnn``
- ``torch.backends.openmp``
- ``torch.backends.opt_einsum``
- ``torch.backends.xeon``


torch.backends.cuda
^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.cuda

.. autofunction::  torch.backends.cuda.is_built

.. attribute::  torch.backends.cuda.matmul.allow_tf32

    A :class:`bool` that controls whether TensorFloat-32 tensor cores may be used in matrix
    multiplications on Ampere or newer GPUs. See :ref:`tf32_on_ampere`.

.. attribute::  torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction

    A :class:`bool` that controls whether reduced precision reductions (e.g., with fp16 accumulation type) are allowed with fp16 GEMMs.

.. attribute::  torch.backends.cuda.cufft_plan_cache

    ``cufft_plan_cache`` caches the cuFFT plans

    .. attribute::  size

        A readonly :class:`int` that shows the number of plans currently in the cuFFT plan cache.

    .. attribute::  max_size

        A :class:`int` that controls cache capacity of cuFFT plan.

    .. method::  clear()

        Clears the cuFFT plan cache.

.. autofunction:: torch.backends.cuda.preferred_linalg_library

.. autofunction:: torch.backends.cuda.flash_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_mem_efficient_sdp

.. autofunction:: torch.backends.cuda.mem_efficient_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_flash_sdp

.. autofunction:: torch.backends.cuda.math_sdp_enabled

.. autofunction:: torch.backends.cuda.enable_math_sdp

.. autofunction:: torch.backends.cuda.sdp_kernel

torch.backends.cudnn
^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.cudnn

.. autofunction:: torch.backends.cudnn.version

.. autofunction:: torch.backends.cudnn.is_available

.. attribute::  torch.backends.cudnn.enabled

    A :class:`bool` that controls whether cuDNN is enabled.

.. attribute::  torch.backends.cudnn.allow_tf32

    A :class:`bool` that controls where TensorFloat-32 tensor cores may be used in cuDNN
    convolutions on Ampere or newer GPUs. See :ref:`tf32_on_ampere`.

.. attribute::  torch.backends.cudnn.deterministic

    A :class:`bool` that, if True, causes cuDNN to only use deterministic convolution algorithms.
    See also :func:`torch.are_deterministic_algorithms_enabled` and
    :func:`torch.use_deterministic_algorithms`.

.. attribute::  torch.backends.cudnn.benchmark

    A :class:`bool` that, if True, causes cuDNN to benchmark multiple convolution algorithms
    and select the fastest.

.. attribute::  torch.backends.cudnn.benchmark_limit

    A :class:`int` that specifies the maximum number of cuDNN convolution algorithms to try when
    `torch.backends.cudnn.benchmark` is True. Set `benchmark_limit` to zero to try every
    available algorithm. Note that this setting only affects convolutions dispatched via the
    cuDNN v8 API.


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


torch.backends.openmp
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.openmp

.. autofunction::  torch.backends.openmp.is_available

.. Docs for other backends need to be added here.
.. Automodules are just here to ensure checks run but they don't actually
.. add anything to the rendered page for now.
.. py:module:: torch.backends.quantized
.. py:module:: torch.backends.xnnpack


torch.backends.opt_einsum
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.opt_einsum

.. autofunction:: torch.backends.opt_einsum.is_available

.. autofunction:: torch.backends.opt_einsum.get_opt_einsum

.. attribute::  torch.backends.opt_einsum.enabled

    A :class:``bool`` that controls whether opt_einsum is enabled (``True`` by default). If so,
    torch.einsum will use opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html)
    if available to calculate an optimal path of contraction for faster performance.

    If opt_einsum is not available, torch.einsum will fall back to the default contraction path
    of left to right.

.. attribute::  torch.backends.opt_einsum.strategy

    A :class:``str`` that specifies which strategies to try when ``torch.backends.opt_einsum.enabled``
    is ``True``. By default, torch.einsum will try the "auto" strategy, but the "greedy" and "optimal"
    strategies are also supported. Note that the "optimal" strategy is factorial on the number of
    inputs as it tries all possible paths. See more details in opt_einsum's docs
    (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html).


torch.backends.xeon
^^^^^^^^^^^^^^^^^^^
.. automodule:: torch.backends.xeon
