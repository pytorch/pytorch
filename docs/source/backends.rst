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
- ``torch.backends.quantized``
- ``torch.backends.xnnpack``


torch.backends.cuda
^^^^^^^^^^^^^^^^^^^

.. autofunction::  torch.backends.cuda.is_built

.. attribute::  torch.backends.cuda.matmul.allow_tf32

    A :class:`bool` that controls whether TensorFloat-32 tensor cores are allowed for the
    computation of matrix multiplications. TensorFloat-32 tensor cores is a new technology
    available on Ampere or newer GPUs that allows speeding up certain computations at the
    cost of precision.

    See :ref:`tf32_on_ampere` for more information.

.. attribute::  torch.backends.cuda.cufft_plan_cache

    ``cufft_plan_cache`` caches the cuFFT plans

    .. attribute::  size

        A readonly :class:`int` that shows the number of plans currently in the cuFFT plan cache.

    .. attribute::  max_size

        A :class:`int` that controls cache capacity of cuFFT plan.

    .. method::  clear()

        Clear cuFFT plan cache
    

torch.backends.cudnn
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torch.backends.cudnn.version

.. autofunction:: torch.backends.cudnn.is_available

.. attribute::  torch.backends.cudnn.enabled

    A :class:`bool` that controls whether cuDNN is enabled.

.. attribute::  torch.backends.cudnn.allow_tf32

    A :class:`bool` that controls whether cuDNN is allowed to use TensorFloat-32 tensor cores
    for the computation of convolutions. TensorFloat-32 tensor cores is a new technology
    available on Ampere or newer GPUs that allows speeding up certain computations at the
    cost of precision.

    See :ref:`tf32_on_ampere` for more information.

.. attribute::  torch.backends.cudnn.deterministic

    A :class:`bool` that controls whether cuDNN is should return deterministic results.
    See also :func:`torch.is_deterministic` and :func:`torch.set_deterministic`.

.. attribute::  torch.backends.cudnn.benchmark

    A :class:`bool` that controls whether cuDNN should try various algorithms and find the
    fastest algorithm when doing convolutions.


torch.backends.mkl
^^^^^^^^^^^^^^^^^^

.. autofunction::  torch.backends.mkl.is_available


torch.backends.mkldnn
^^^^^^^^^^^^^^^^^^^^^

.. autofunction::  torch.backends.mkldnn.is_available


torch.backends.openmp
^^^^^^^^^^^^^^^^^^^^^

.. autofunction::  torch.backends.openmp.is_available


torch.backends.quantized
^^^^^^^^^^^^^^^^^^^^^^^^

This documentation is not completed yet. Contributions are welcome!


torch.backends.xnnpack
^^^^^^^^^^^^^^^^^^^^^^

This documentation is not completed yet. Contributions are welcome!
