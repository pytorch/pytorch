.. role:: hidden
    :class: hidden-section

torch.backends
==============

`torch.backends` controls the behavior of various backends that PyTorch supports.

These backends include:

- ``torch.backends.cuda``
- ``torch.backends.cudnn``
- ``torch.backends.mkl``
- ``torch.backends.mkldnn``
- ``torch.backends.openmp``


torch.backends.cuda
^^^^^^^^^^^^^^^^^^^

.. autofunction::  torch.backends.cuda.is_built

.. attribute::  torch.backends.cuda.matmul.allow_tf32

    A :class:`bool` that controls whether TensorFloat-32 tensor cores may be used in matrix
    multiplications on Ampere or newer GPUs. See :ref:`tf32_on_ampere`.

.. attribute::  torch.backends.cuda.cufft_plan_cache

    ``cufft_plan_cache`` caches the cuFFT plans

    .. attribute::  size

        A readonly :class:`int` that shows the number of plans currently in the cuFFT plan cache.

    .. attribute::  max_size

        A :class:`int` that controls cache capacity of cuFFT plan.

    .. method::  clear()

        Clears the cuFFT plan cache.


torch.backends.cudnn
^^^^^^^^^^^^^^^^^^^^

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


torch.backends.mkl
^^^^^^^^^^^^^^^^^^

.. autofunction::  torch.backends.mkl.is_available


torch.backends.mkldnn
^^^^^^^^^^^^^^^^^^^^^

.. autofunction::  torch.backends.mkldnn.is_available


torch.backends.openmp
^^^^^^^^^^^^^^^^^^^^^

.. autofunction::  torch.backends.openmp.is_available
