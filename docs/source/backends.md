```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# torch.backends

```{eval-rst}
.. automodule:: torch.backends
```

`torch.backends` controls the behavior of various backends that PyTorch supports.

These backends include:

- `torch.backends.cpu`
- `torch.backends.cuda`
- `torch.backends.cudnn`
- `torch.backends.cusparselt`
- `torch.backends.mha`
- `torch.backends.mps`
- `torch.backends.mkl`
- `torch.backends.mkldnn`
- `torch.backends.nnpack`
- `torch.backends.openmp`
- `torch.backends.opt_einsum`
- `torch.backends.xeon`

## torch.backends.cpu

```{eval-rst}
.. automodule:: torch.backends.cpu
```

```{eval-rst}
.. autofunction::  torch.backends.cpu.get_cpu_capability
```

## torch.backends.cuda

```{eval-rst}
.. automodule:: torch.backends.cuda
```

```{eval-rst}
.. autofunction::  torch.backends.cuda.is_built
```

```{eval-rst}
.. currentmodule:: torch.backends.cuda.matmul
```

```{eval-rst}
.. attribute::  allow_tf32

    A :class:`bool` that controls whether TensorFloat-32 tensor cores may be used in matrix
    multiplications on Ampere or newer GPUs. allow_tf32 is going to be deprecated. See :ref:`tf32_on_ampere`.
```

```{eval-rst}
.. attribute::  allow_fp16_reduced_precision_reduction

    A :class:`bool` that controls whether reduced precision reductions (e.g., with fp16 accumulation type) are allowed with fp16 GEMMs.
```

```{eval-rst}
.. attribute::  allow_bf16_reduced_precision_reduction

    A :class:`bool` that controls whether reduced precision reductions are allowed with bf16 GEMMs.
```

```{eval-rst}
.. currentmodule:: torch.backends.cuda
```

```{eval-rst}
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
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.preferred_blas_library
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.preferred_rocm_fa_library
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.preferred_linalg_library
```

```{eval-rst}
.. autoclass:: torch.backends.cuda.SDPAParams
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.flash_sdp_enabled
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.enable_mem_efficient_sdp
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.mem_efficient_sdp_enabled
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.enable_flash_sdp
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.math_sdp_enabled
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.enable_math_sdp
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.cudnn_sdp_enabled
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.enable_cudnn_sdp
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.is_flash_attention_available
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.can_use_flash_attention
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.can_use_efficient_attention
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.can_use_cudnn_attention
```

```{eval-rst}
.. autofunction:: torch.backends.cuda.sdp_kernel
```

## torch.backends.cudnn

```{eval-rst}
.. automodule:: torch.backends.cudnn
```

```{eval-rst}
.. autofunction:: torch.backends.cudnn.version
```

```{eval-rst}
.. autofunction:: torch.backends.cudnn.is_available
```

```{eval-rst}
.. attribute::  enabled

    A :class:`bool` that controls whether cuDNN is enabled.
```

```{eval-rst}
.. attribute::  allow_tf32

    A :class:`bool` that controls where TensorFloat-32 tensor cores may be used in cuDNN
    convolutions on Ampere or newer GPUs. allow_tf32 is going to be deprecated. See :ref:`tf32_on_ampere`.
```

```{eval-rst}
.. attribute::  deterministic

    A :class:`bool` that, if True, causes cuDNN to only use deterministic convolution algorithms.
    See also :func:`torch.are_deterministic_algorithms_enabled` and
    :func:`torch.use_deterministic_algorithms`.
```

```{eval-rst}
.. attribute::  benchmark

    A :class:`bool` that, if True, causes cuDNN to benchmark multiple convolution algorithms
    and select the fastest.
```

```{eval-rst}
.. attribute::  benchmark_limit

    A :class:`int` that specifies the maximum number of cuDNN convolution algorithms to try when
    `torch.backends.cudnn.benchmark` is True. Set `benchmark_limit` to zero to try every
    available algorithm. Note that this setting only affects convolutions dispatched via the
    cuDNN v8 API.
```

```{eval-rst}
.. py:module:: torch.backends.cudnn.rnn
```

## torch.backends.cusparselt

```{eval-rst}
.. automodule:: torch.backends.cusparselt
```

```{eval-rst}
.. autofunction:: torch.backends.cusparselt.version
```

```{eval-rst}
.. autofunction:: torch.backends.cusparselt.is_available
```

## torch.backends.mha

```{eval-rst}
.. automodule:: torch.backends.mha
```

```{eval-rst}
.. autofunction::  torch.backends.mha.get_fastpath_enabled
```

```{eval-rst}
.. autofunction::  torch.backends.mha.set_fastpath_enabled

```

## torch.backends.miopen

```{eval-rst}
.. automodule:: torch.backends.miopen
```

```{eval-rst}
.. attribute::  immediate

    A :class:`bool` that, if True, causes MIOpen to use Immediate Mode
    (https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html).
```

## torch.backends.mps

```{eval-rst}
.. automodule:: torch.backends.mps
```

```{eval-rst}
.. autofunction::  torch.backends.mps.is_available
```

```{eval-rst}
.. autofunction::  torch.backends.mps.is_built

```

## torch.backends.mkl

```{eval-rst}
.. automodule:: torch.backends.mkl
```

```{eval-rst}
.. autofunction::  torch.backends.mkl.is_available
```

```{eval-rst}
.. autoclass::  torch.backends.mkl.verbose

```

## torch.backends.mkldnn

```{eval-rst}
.. automodule:: torch.backends.mkldnn
```

```{eval-rst}
.. autofunction::  torch.backends.mkldnn.is_available
```

```{eval-rst}
.. autoclass::  torch.backends.mkldnn.verbose
```

## torch.backends.nnpack

```{eval-rst}
.. automodule:: torch.backends.nnpack
```

```{eval-rst}
.. autofunction::  torch.backends.nnpack.is_available
```

```{eval-rst}
.. autofunction::  torch.backends.nnpack.flags
```

```{eval-rst}
.. autofunction::  torch.backends.nnpack.set_flags
```

## torch.backends.openmp

```{eval-rst}
.. automodule:: torch.backends.openmp
```

```{eval-rst}
.. autofunction::  torch.backends.openmp.is_available
```

% Docs for other backends need to be added here.
% Automodules are just here to ensure checks run but they don't actually
% add anything to the rendered page for now.

```{eval-rst}
.. py:module:: torch.backends.quantized
```

```{eval-rst}
.. py:module:: torch.backends.xnnpack
```

```{eval-rst}
.. py:module:: torch.backends.kleidiai

```

## torch.backends.opt_einsum

```{eval-rst}
.. automodule:: torch.backends.opt_einsum
```

```{eval-rst}
.. autofunction:: torch.backends.opt_einsum.is_available
```

```{eval-rst}
.. autofunction:: torch.backends.opt_einsum.get_opt_einsum
```

```{eval-rst}
.. attribute::  enabled

    A :class:`bool` that controls whether opt_einsum is enabled (``True`` by default). If so,
    torch.einsum will use opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html)
    if available to calculate an optimal path of contraction for faster performance.

    If opt_einsum is not available, torch.einsum will fall back to the default contraction path
    of left to right.
```

```{eval-rst}
.. attribute::  strategy

    A :class:`str` that specifies which strategies to try when ``torch.backends.opt_einsum.enabled``
    is ``True``. By default, torch.einsum will try the "auto" strategy, but the "greedy" and "optimal"
    strategies are also supported. Note that the "optimal" strategy is factorial on the number of
    inputs as it tries all possible paths. See more details in opt_einsum's docs
    (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html).

```

## torch.backends.xeon

```{eval-rst}
.. automodule:: torch.backends.xeon
```

```{eval-rst}
.. py:module:: torch.backends.xeon.run_cpu
```
