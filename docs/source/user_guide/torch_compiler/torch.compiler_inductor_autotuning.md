(torch.compiler_inductor_autotuning)=

# Autotuning

TorchInductor uses **autotuning** to select the best-performing kernel
configurations at runtime. Rather than relying on static heuristics alone,
it benchmarks multiple candidate configurations on the actual hardware and
input shapes, then caches the results for future runs.

**Source**: [torch/_inductor/runtime/triton_heuristics.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/runtime/triton_heuristics.py),
[torch/_inductor/runtime/coordinate_descent_tuner.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/runtime/coordinate_descent_tuner.py)

## Triton Autotuning

When TorchInductor generates a Triton kernel, it typically produces multiple
**configurations** — combinations of parameters like block sizes, number of
warps, and number of stages. At runtime, each configuration is benchmarked on
the target device and the fastest one is selected.

The autotuning flow works as follows:

1. **Config generation** — heuristics in `triton_heuristics.py` produce a set
   of candidate configurations based on the kernel type (pointwise, reduction,
   template, etc.) and problem shape.
2. **Benchmarking** — each candidate is executed with the actual input tensors.
   By default, the kernel is warmed up for 25 iterations and then timed over
   100 iterations to get a stable measurement.
3. **Selection** — the fastest configuration is chosen and cached for reuse.

For GEMM operations with `max_autotune` enabled, TorchInductor also considers
template kernels from multiple backends (Triton, CUTLASS, CK, ATen) and
benchmarks them alongside the generated candidates.

### Example: Viewing Autotuning Results

To see which configurations are being benchmarked and their timings, enable
autotuning logging:

```python
import torch

# Enable autotuning logging
torch._inductor.config.trace.log_autotuning_results = True

@torch.compile
def fn(x, y):
    return x @ y

x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
fn(x, y)
```

The logs will show each candidate configuration, its benchmark time, and which
one was selected as the winner.

## Autotuning Cache

Autotuning results are cached to avoid re-benchmarking on subsequent runs.
The cache key includes:

- **Source code hash** — the generated kernel source
- **Tensor metadata** — shapes, strides, and dtypes of input tensors
- **Device properties** — GPU architecture, number of SMs, etc.
- **Triton version** — to invalidate cache when the compiler changes

The cache lookup is performed before any benchmarking. If a cache hit occurs,
the winning configuration is reused immediately without re-running benchmarks.

:::{note}
This is the **Triton-level** autotuning cache for individual kernel configs,
which is separate from the [FXGraphCache](torch.compiler_inductor_caching.md)
that caches entire compiled graphs.
:::

## Coordinate Descent Tuning

Standard autotuning selects the best configuration from a pre-defined set of
candidates. **Coordinate descent tuning** goes further by searching the space
around the winning configuration to find better parameters.

After the initial autotuning selects a winner, coordinate descent:

1. Takes the winning configuration as a starting point.
2. Adjusts one parameter at a time (e.g., block size, number of warps) in each
   direction.
3. Benchmarks the modified configuration.
4. If the modification improves performance, adopts it as the new baseline.
5. Repeats until no further improvement is found.

Enable coordinate descent tuning via config or environment variable:

```python
torch._inductor.config.coordinate_descent_tuning = True
```

Or:

```bash
TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 python my_script.py
```

Additional coordinate descent settings:

| Config | Env Variable | Default | Description |
|--------|-------------|---------|-------------|
| `coordinate_descent_check_all_directions` | `TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS` | `False` | Check both increase and decrease for each parameter |
| `coordinate_descent_search_radius` | `TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS` | `1` | Number of steps to search in each direction |

## Configuration Reference

The following `torch._inductor.config` options control autotuning behavior:

### Enabling Autotuning

| Config | Env Variable | Default | Description |
|--------|-------------|---------|-------------|
| `max_autotune` | `TORCHINDUCTOR_MAX_AUTOTUNE` | `False` | Enable full autotuning for all operation types. Increases compile time but can improve runtime performance. |
| `max_autotune_gemm` | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM` | `False` | Enable autotuning for GEMM (matrix multiplication) operations only. |
| `max_autotune_pointwise` | `TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE` | `False` | Enable autotuning for pointwise operations. |

### Backend Selection

| Config | Env Variable | Default | Description |
|--------|-------------|---------|-------------|
| `max_autotune_gemm_backends` | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` | `"ATEN,TRITON,CPP"` | Comma-separated list of backends to consider for GEMM autotuning. Options: `ATEN`, `TRITON`, `CUTLASS`, `CUTEDSL`, `NVGEMM`, `CK`, `CKTILE`, `CPP`. |
| `max_autotune_conv_backends` | `TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS` | `"ATEN,TRITON"` | Comma-separated list of backends for convolution autotuning. |
| `max_autotune_gemm_search_space` | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE` | `"DEFAULT"` | Size of the GEMM search space. `"DEFAULT"` balances compile time and performance; `"EXHAUSTIVE"` maximizes performance. |

### Benchmarking Parameters

| Config | Env Variable | Default | Description |
|--------|-------------|---------|-------------|
| `inductor_default_autotune_warmup` | `TORCHINDUCTOR_DEFAULT_AUTOTUNE_WARMUP` | `25` | Number of warmup iterations before timing. |
| `inductor_default_autotune_rep` | `TORCHINDUCTOR_DEFAULT_AUTOTUNE_REP` | `100` | Number of timed iterations for benchmarking. |
| `autotune_in_subproc` | `TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC` | `False` | Run autotuning benchmarks in a subprocess to avoid polluting the main process state. |
| `autotune_multi_device` | `TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE` | `False` | When autotuning in subprocess, use multiple GPU devices in parallel. |

### Coordinate Descent

| Config | Env Variable | Default | Description |
|--------|-------------|---------|-------------|
| `coordinate_descent_tuning` | `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING` | `False` | Enable coordinate descent search around the autotuning winner. |
| `coordinate_descent_check_all_directions` | `TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS` | `False` | Check both directions for each parameter dimension. |
| `coordinate_descent_search_radius` | `TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS` | `1` | Number of steps to search in each direction. |

## AutoHeuristic Framework

TorchInductor includes the **AutoHeuristic** framework, which learns decision
trees from autotuning data to make better heuristic choices without runtime
benchmarking. It works by:

1. Collecting autotuning data across representative workloads.
2. Training decision trees on the collected data.
3. Generating code from the learned trees that ships with the compiler.

This allows TorchInductor to make informed choices (e.g., selecting between
GEMM implementations) without the overhead of runtime benchmarking.

For details on collecting data and contributing heuristics, see
[torchgen/_autoheuristic/README.md](https://github.com/pytorch/pytorch/blob/main/torchgen/_autoheuristic/README.md).

Related config options:

| Config | Env Variable | Default | Description |
|--------|-------------|---------|-------------|
| `autoheuristic_collect` | `TORCHINDUCTOR_AUTOHEURISTIC_COLLECT` | `""` | Comma-separated list of optimizations to collect autotuning data for. |
| `autoheuristic_use` | `TORCHINDUCTOR_AUTOHEURISTIC_USE` | `"mixed_mm"` | Comma-separated list of optimizations to use learned heuristics for. |
