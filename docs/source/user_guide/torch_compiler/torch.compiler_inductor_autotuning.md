(torch.compiler_inductor_autotuning)=

# Autotuning

Autotuning is the process by which TorchInductor automatically searches for the
best-performing implementation of a given operation on a specific problem size
and hardware configuration. The performance on modern accelerators depends
heavily on input shapes, hardware, memory layouts, and backend selection (ATen,
Triton, cuBLAS). There is no single configuration that works optimally for all
scenarios, so autotuning attempts to maximize kernel performance by benchmarking
candidate variants based on configs or heuristics, and picking the most
performant option.

There are two different types of autotuning in TorchInductor: A compile-time
autotuner (we call it **Inductor Autotuning** in this section),
[autotune_select_algorithm](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/select_algorithm.py#L3908),
that compares different backend and configs, typically used for matmul/conv/flex
atten; and a run-time autotuner (we call it **Triton Autotuning** in this
section),
[CachingAutotuner](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/runtime/triton_heuristics.py#L260),
that is a modified Triton autotuner, used for codegen'd Triton kernels.

**Source**: [torch/_inductor/select_algorithm.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/select_algorithm.py),
[torch/_inductor/runtime/triton_heuristics.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/runtime/triton_heuristics.py),
[torch/_inductor/runtime/coordinate_descent_tuner.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/runtime/coordinate_descent_tuner.py)

## Inductor Autotuning

The Inductor autotuner
([autotune_select_algorithm](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/select_algorithm.py#L3908))
is typically used for benchmarking kernels with different backends and configs.
The system allows users to provide multiple backends, such as ATen, Triton,
CUTLASS, and CPU. When the Triton backend is enabled,
[handwritten Triton templates](https://github.com/pytorch/pytorch/tree/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/kernel)
(for example, matmul/conv/flex_atten) are used with various
[configs'](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/template_heuristics/triton.py#L214)
parameters such as block sizes, number of warps, and number of stages.

### Algorithm

The flow of the compile-time autotuning is as follows: During the graph lowering
phase, TorchInductor detects template patterns for nodes defined in
[torch/_inductor/kernel](https://github.com/pytorch/pytorch/tree/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/kernel)
with `@register_lowering`, and registers the corresponding Template choices. For
example, matmul is registered in
[mm.py](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/kernel/mm.py#L1100);
and bmm is registered in
[bmm.py](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/kernel/bmm.py#L135).
It then calls `autotune_select_algorithm` in
[select_algorithm.py](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/select_algorithm.py#L3908)
with the template choices. If previously-cached results are available, we are
done autotuning. Otherwise, an
[AlgorithmSelectorCache](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/select_algorithm.py#L2619)
is created. Then each backend applies their pre-defined configs' parameters to
their kernel Template, expanding the number of variants in the choices candidate
lists. Triton choices are defined in
[template_heuristics/triton.py](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/template_heuristics/triton.py#L65)
and CUTLASS choices are defined in
[cutlass_utils.py](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/nn/modules/conv.py).
For example, mm has 19 configs under max-autotune and 1,875 configs under
exhaustive mode, and mm has 40 configs for ROCm. These configs are then
benchmarked together.

There are two different benchmark paths, an **immediate benchmark** and a
**lazy mode** (`MultiTemplateBuffer`) that considers fusion, determined by a
`return_multi_template` flag and `max-autotune` flag
[together](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/select_algorithm.py#L2919).
These flags can be passed in from env vars, set directly via
`torch._inductor.config.max_autotune` and
`torch._inductor.config.benchmark_epilogue_fusion`, or from
[kwargs](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/select_algorithm.py#L3911).

- For the **immediate mode**, we benchmark the choices immediately and return
  the fastest implementation for codegen. This mode calls
  [make_precompile_fn()](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/select_algorithm.py#L3001)
  to create a threadpool to parallel-compile all the choices and then calls
  [do_autotuning](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/select_algorithm.py#L2792)
  sequentially to benchmark all choices. There is also a fallback to the
  external kernel if all result timings are
  [empty](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/select_algorithm.py#L2972).
  The advantage of immediate mode is that it runs faster than lazy mode; the
  disadvantage is that it has a more limited search space.

- Unlike immediate mode, **lazy mode** is fusion-aware. It delays benchmarking
  until the scheduler phase, which has the benefit of considering potential
  epilogue fusions and other optimizations into the benchmarking. In this mode,
  `select_algorithm` only generates a
  [MultiTemplateBuffer](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/select_algorithm.py#L2956)
  instead of returning the best choice immediately. Then, in `scheduler.py`, it
  calls
  [finalize_multi_template_buffers](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/scheduler.py#L3446)()
  to do the benchmarking (with fusion) and selects the best choice.

Either path should then generate the fastest choice for codegen. The results
will be cached under `/tmp/torchinductor_$USER/` for the next run.

### Backend Support

The table below compares different backend implementations. Each backend
represents a fundamentally different approach to executing the operation. The
effectiveness and speed of autotuning can vary significantly between backends
due to differences in compilation time, kernel launch overhead, and available
hardware features.

Template-based backends, such as
[Triton](https://github.com/pytorch/pytorch/tree/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/kernel/templates),
[CUDA/CUTLASS](https://github.com/pytorch/pytorch/tree/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/codegen/cuda),
and
[CK/ROCm](https://github.com/pytorch/pytorch/tree/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/codegen/rocm),
generate kernels from parameterized templates. Parameters include:

- **Tile sizes**: How to partition the computation.
- **Thread organization**: Warps, blocks, clusters.
- **Memory access patterns**: Swizzling, pipeline stages.
- **Hardware features**: TMA (Tensor Memory Accelerator), fast accumulation, etc.

(This is not exhaustive.)

Determining the optimal configuration requires benchmarking multiple candidates
because performance depends heavily on input shapes, different configurations,
and memory layouts. During autotuning, TorchInductor will generate all variants
of the same kernel based on different backends, and benchmark all the variants
to select the best choice. Please refer to the examples below for detailed
flows.

Supported backends are as follows:

| Backend | Description | Implementation | Platform |
|---------|-------------|----------------|----------|
| ATEN | PyTorch's native backend using vendor-optimized libraries (cuBLAS, cuDNN, hipBLAS, MIOpen) | Pre-compiled library calls | CUDA |
| TRITON | Triton template-based kernels generated by PyTorch | Just-in-time compiled from templates | CUDA |
| CUTLASS | NVIDIA's CUTLASS library for high-performance GEMM operations | Pre-compiled template library | CUDA |
| CK | AMD's Composable Kernel library for ROCm | Pre-compiled template library | ROCm |
| CPP | C++ templates for CPU | Generated C++ code | CPU |

### Autotune Template Configs

The autotune choices depend on variants with different parameters. Note that
the larger the search space, the slower the overall compile time. The search
space is defined by the number of template configurations. Please refer to the
[Configs](#configs) section for more details.

- **Default**: By default, only a minimal set of configurations are considered.
  For most operations, TorchInductor typically uses heuristics or the backend's
  default implementation without benchmarking alternative configs. Heuristics
  quickly select kernel configurations based on input sizes and can generate
  reasonable performance without benchmarking. Note that some codegen kernels,
  such as Pointwise, may still be autotuned using simple hint heuristics that
  generate a small config set
  ([code](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/runtime/triton_heuristics.py#L2592)).

- **Max-autotune**: If the `max-autotune` flag is specified, TorchInductor will
  perform benchmarking for Triton template kernels (mm, conv, flexAttn). This
  mode balances performance and compile time, e.g., it considers 19 GEMM Triton
  Template configs
  ([code](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/template_heuristics/triton.py#L229)).
  Max-autotune can be enabled with `TORCHINDUCTOR_MAX_AUTOTUNE=1`.

- **Exhaustive**: This mode searches a much larger space than max-autotune
  (hundreds to thousands of choices). Essentially, exhaustive autotuning
  considers all possible parameter combinations for the kernel. As a result, it
  can take a very long time to execute. Exhaustive mode can be turned on with
  `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE`.

Some kernels have their specific tuning knobs, such as for flexAttn, it has
`SPARSE_Q_BLOCK_SIZE`, `SPARSE_KV_BLOCK_SIZE`, `USE_TMA`, `FLOAT32_PRECISION`
and `HAS_FULL_BLOCKS`.

### User Flow

To see information about the autotune process in the logs, use:
`TORCH_LOGS=inductor`. When debugging over multiple runs, it's often helpful to
disable caching in order to guarantee that TorchInductor doesn't reuse
autotuning decisions from previous runs. Disable all caching by setting the
environment variable `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`.

Users can turn on max autotune with:

```python
# Enable max autotune (off by default)
torch._inductor.config.max_autotune = True
# It is also configurable via environment variables, e.g., TORCHINDUCTOR_MAX_AUTOTUNE=1
```

Users can specify which backend to tune for GEMMs with
`torch._inductor.config.max_autotune_gemm_backends`. For example, default
backends include ATen, Triton and CPP:

```python
# Specify the default backends for GEMM operations
torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON,CPP"  # default
```

Or users can change it to CUTLASS on CUDA:

```python
# Add CUTLASS
torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON,CUTLASS"
```

Users can also specify these within a `config.patch` context manager:

```python
with config.patch({
    'max_autotune': True,
    'max_autotune_gemm_backends': 'TRITON,CUTLASS',
}):
```

### Example: Matmul Autotuning

We can use an example to better illustrate TorchInductor's autotune process. We
will use matmul examples for autotuning on Triton and ATen.

```python
def mm(a, b):
    return a @ b

with config.patch({
    'max_autotune': True,
    'max_autotune_gemm_backends': 'TRITON,ATEN',
}):
    result = torch.compile(mm)(a, b)
```

Then we run:

```bash
TORCH_LOGS='inductor' python example.py
```

The overall process can be described with the following flowchart. The phases
that are relevant to autotuning are highlighted and explained below.

```
A[User Code: a @ b]
  --> B[FX Graph: aten.mm.default]
  --> C[Graph Lowering: call_function]
  --> D[Lowering Dispatch: lowerings[target]]
  --> E[tuned_mm function]
  --> F[Backend Collection]
  --> G[TemplateChoice Generation]
  --> H[autotune_select_algorithm]
  --> I{Execution Mode}
        |-- Immediate --> J[do_autotuning]
        |-- Delayed  --> K[MultiTemplateBuffer]
  J --> L[Parallel Compile + Sequential Benchmark]
  K --> M[Scheduler: finalize_multi_template_buffers]
  M --> N[Choice Selection with Fusion Context]
  L --> O[Best Choice Selection]
  N --> O
  O --> P[Code Generation]
  P --> Q[Optimized Kernel Execution]
```

**B→D GraphLowering**: During the graph lowering phase, the lowering loop
detects the node as a `call_function()` and then refers to the registered
lowering map for the corresponding kernel, i.e.,
[mm.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py)
in our demo.

**D→F mm.py**: Since we have both Triton and ATEN backends, the default ATen
(cuBLAS) and Triton templates will be used. In the registered lowering of
[mm.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py),
it registers all enabled backends to `templates_to_use`. In our case,
`KernelTemplate` (for generating Triton kernel choices) and an
`ExternCallerChoice` (for the ATen kernel) are added. Then these templates can
generate various kernel choices based on autotune configs for benchmarking.
Finally, it calls `autotune_select_algorithm()`, the main invocation to
benchmark the candidate choices.

**F→O select_algorithm.py**: In
[select_algorithm](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/select_algorithm.py),
candidate choices are generated from kernel templates. The
[AlgorithmSelectorCache](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/select_algorithm.py#L3909)
class enables caching and returns cached results if there are previous compiled
results. In our example, since we only have a single matmul kernel, there is no
difference if we choose the immediate benchmark or a lazy mode with
`MultiTemplateBuffer`. If there are nodes following the matmul, TorchInductor
might perform epilogue fusion when using `MultiTemplateBuffer`.

During benchmarking, TorchInductor generates the following benchmark log:

```
dtypes: torch.float16, torch.float16

triton_mm_4 0.0076 ms 100.0%
  ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=32,
  EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4

mm 0.0078 ms 96.7%

triton_mm_8 0.0078 ms 96.3%
  ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=64,
  EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=4

triton_mm_12 0.0084 ms 90.1%
  ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=64, BLOCK_N=128,
  EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=4

triton_mm_3 0.0087 ms 87.1%
  ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=32,
  EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8

triton_mm_7 0.0089 ms 84.9%
  ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=64,
  EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=8

triton_mm_2 0.0090 ms 83.7%
  ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=32, BLOCK_N=64,
  EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8

triton_mm_11 0.0092 ms 82.5%
  ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128,
  EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=3, num_warps=4

triton_mm_18 0.0095 ms 79.2%
  ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=128, BLOCK_N=128,
  EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=5, num_warps=8

triton_mm_10 0.0097 ms 77.9%
  ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=64, BLOCK_N=128,
  EVEN_K=True, GROUP_M=8, USE_FAST_ACCUM=False, num_stages=4, num_warps=8

SingleProcess AUTOTUNE benchmarking takes 0.1747 seconds and 0.0019 seconds
precompiling for 20 choices
```

The autotuning output shows that mm has multiple backend variants:
`triton_mm_xx` (Triton) and `mm` (ATen/cuBLAS). In this example, among 20
choices (1 ATen + 19 Triton), the most performant option is `triton_mm_4`.

**O→P Codegen**: After autotuning, TorchInductor knows the parameters of the
fastest implementation on the specific hardware. The codegen phase will generate
the final kernel for execution given these parameters.

**P→Q Execution**: We can see a slight execution time speed up compared to
eager mode:

| Mode | Time (ms) | Speedup |
|------|-----------|---------|
| Eager | 0.0081 | 1.00x |
| Compiled | 0.0079 | 1.03x |

### Caching

Unless explicitly disabled, autotuning automatically caches the results of the
autotuning for subsequent executions. `AlgorithmSelectorCache.lookup()` will
return the cached result instead of performing the actual benchmarking work.

## Triton Autotuning

Triton Autotuning is a modified version of Triton's autotuning. Triton
autotuning is typically used for pointwise and reduction kernels, and it has
multiple configurations for these kernels. It optimizes how a Triton kernel
executes by tuning parameters like block sizes, number of warps, and pipeline
stages. It also enables caching into local disks to reduce cold start time.

The core of Triton autotuning is the
[CachingAutotuner](https://github.com/pytorch/pytorch/blob/4e1b772103786e914abe91a0048bc2e98df5a7e1/torch/_inductor/runtime/triton_heuristics.py#L260)
class.

```python
class CachingAutotuner:
    def run(self, *args, **kwargs):
        """Main entry - checks cache, runs autotuning if needed"""

    def benchmark_all_configs(self, *args, **kwargs):
        """Core benchmarking logic"""

    def precompile(self, ...):
        """Async compilation of all configs"""
```

Triton autotuning optimizes these key parameters for different kernels:

- **Block Sizes** (most important for performance)
  - `XBLOCK`, `YBLOCK`: For 1D/2D pointwise operations.
  - `BLOCK_M`, `BLOCK_N`, `BLOCK_K`: For GEMM operations.
  - `RBLOCK`: For reduction dimensions.
- **Thread Organization**
  - `num_warps`: Number of warps per thread block (typically 1, 2, 4, 8).
- **Pipeline Stages**
  - `num_stages`: Software pipelining depth (typically 2–5).
- **Accumulator Type**
  - `ACC_TYPE`: Accumulator precision (e.g., `tl.float32` for mixed precision).

The configurations can be determined based on the
[heuristics and type of operations](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/runtime/triton_heuristics.py),
pointwise, reduction, etc.

```python
def pointwise_heuristics(...) -> List[Config]:
    """Generate configs for pointwise ops based on input size"""

def reduction_heuristics(...) -> List[Config]:
    """Generate configs for reduction ops"""

def persistent_reduction_heuristics(...) -> List[Config]:
    """Generate configs for persistent reduction"""
```

The main difference between Triton autotuning and Inductor autotuning is that
the Triton autotuner only considers Triton kernels. During compilation, the
above configurations are compiled into a Triton kernel. During the first
execution at runtime, it calls `do_bench` to benchmark the compiled Triton
kernel with different configs.

```python
def do_bench(fn, warmup=25, rep=100):
    """
    Benchmark a function:
    1. Warmup iterations to stabilize GPU state
    2. Multiple repetitions to get median time
    3. Returns median time in milliseconds
    """
```

Then it caches the best configuration for future use (see
[torch/_inductor/codecache.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codecache.py)):

```python
class LocalCache:
    def lookup(self, key):
        """Check if result exists in cache"""

    def save(self, key, value):
        """Save autotuning result to cache"""
```

### Example: Reduction Autotuning

Here is a simple example of a reduction operation that illustrates
`CachingAutotuner` usage.

```python
def simple_sum(x):
    return x.sum()

compiled_fn = torch.compile(simple_sum, mode='max-autotune-no-cudagraphs')

eager_test = simple_op(x_test)

compiled_test = compiled_fn(x_test)
```

Running this command:

```bash
rm -rf /tmp/torchinductor_$USER ~/.triton/cache
TORCH_LOGS='inductor' python example.py
```

The output shows:

```
torch/_inductor/codegen/simd.py:1858] [0/0] Generating kernel code with
  kernel_name: triton_per_fused_sum_1
torch/_inductor/graph.py:2310] [0/0] Finished codegen for all nodes.
  The list of kernel names available: OrderedSet([])
torch/_inductor/runtime/triton_heuristics.py:315] [0/0]
  CachingAutotuner gets 5 configs for triton_red_fused_sum_0
torch/_inductor/runtime/triton_heuristics.py:321] [0/0]
  XBLOCK: 1, R0_BLOCK: 2048, num_warps: 16, num_ctas: 1, num_stages: 1, maxnreg: None
torch/_inductor/runtime/triton_heuristics.py:321] [0/0]
  XBLOCK: 8, R0_BLOCK: 64, num_warps: 2, num_ctas: 1, num_stages: 1, maxnreg: None
torch/_inductor/runtime/triton_heuristics.py:321] [0/0]
  XBLOCK: 64, R0_BLOCK: 64, num_warps: 2, num_ctas: 1, num_stages: 1, maxnreg: None
torch/_inductor/runtime/triton_heuristics.py:321] [0/0]
  XBLOCK: 8, R0_BLOCK: 512, num_warps: 4, num_ctas: 1, num_stages: 1, maxnreg: None
torch/_inductor/runtime/triton_heuristics.py:321] [0/0]
  XBLOCK: 64, R0_BLOCK: 4, num_warps: 8, num_ctas: 1, num_stages: 1, maxnreg: None
torch/_inductor/runtime/triton_heuristics.py:330] [0/0]
  Triton cache dir: /tmp/torchinductor_tianren/triton/0
```

The output above shows multiple heuristic configs are benchmarked for the
reduction op. Then a Coordinate Descent Tuning step further explores the config
space and selects the best option. (See the next section for a description of
Coordinate Descent Tuning).

```
V1110 19:50:54.043000 = Do coordinate descent tuning for triton_red_fused_sum_0 =
V1110 19:50:54.043000 Baseline Config XBLOCK: 1, R0_BLOCK: 2048,
  num_warps: 16,  baseline timing 0.215680
```

At the end, the log shows a speed-up of 1.042x compared to eager baseline.

```
V1110 19:50:55.004000 triton_red_fused_sum_0: Improve from
  XBLOCK: 1, R0_BLOCK: 2048, num_warps: 16 (0.215680)
  XBLOCK: 1, R0_BLOCK: 4096, num_warps: 8 (0.207072) Speedup: 1.042x
```

### Caching Mechanism

**Cache Key Components:**

- Kernel source code hash.
- Input tensor shapes and strides.
- Device properties (GPU model, compute capability).
- Triton version.

The autotuning cache is located in this location: `~/.triton/cache/`

```
<hash1>/
    metadata.json       # Config info, timings
    kernel.cubin        # Compiled CUDA binary

<hash2>/
    ...
```

On a subsequent run, it follows the pseudocode to determine whether or not to
serve a cached entry:

```python
def run(self, *args, **kwargs):
    # 1. Compute cache key from inputs
    key = self.key(*args, **kwargs)

    # 2. Check cache
    if key in self.cache:
        return self.cache[key](*args, **kwargs)

    # 3. Cache miss - run autotuning
    timings = self.benchmark_all_configs(*args, **kwargs)
    best_config = min(timings, key=lambda x: x[1])  # Lowest time

    # 4. Save to cache
    self.cache[key] = best_config

    # 5. Execute with best config
    return best_config(*args, **kwargs)
```

## Coordinate Descent Tuning

The autotuning we discussed so far searches for the best config among the list
of candidates using pre-defined heuristics. There can be several limitations:

1. The pre-defined heuristics may not transfer well to new hardware.
2. The pre-defined heuristics may not work well for shapes that are rare in our
   benchmarks.

Coordinate descent tuning expands the search space by looking at neighbor Triton
configs. If any of those neighbor configs has better benchmarking results, it
expands the search along that direction. See the entry point for coordinate
descent tuning in
[triton_heuristics.py](https://github.com/pytorch/pytorch/blob/4414e1bff06487f85b1e2ebd1919625298f1444f/torch/_inductor/runtime/triton_heuristics.py#L1202)
and the implementation in
[coordinate_descent_tuner.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/runtime/coordinate_descent_tuner.py).

## Configs

Some TorchInductor config options (in
[torch/_inductor/config.py](https://github.com/pytorch/pytorch/blob/3cfbf98ea9d937d23f3700168b22706c957308ce/torch/_inductor/config.py))
that control top-level autotuning behavior include:

- **`max_autotune`**: Master switch to enable all autotuning, including template
  kernels (GEMM, Conv, Attn) and codegen multi-configs. Increases compile time
  and runtime performance.
- **`max_autotune_gemm`**: Enables only GEMM/BMM template autotuning. More
  granular than `max_autotune`. Increases compile time and GEMM performance.
- **`max_autotune_pointwise`**: Adds extra pointwise configs beyond defaults.
  Increases compile time and pointwise performance.
- **`autotune_pointwise`**: Enables multi-config for pointwise operations
  (1D: 2 configs, 2D: 6 configs, 3D: 7 configs). Increases compile time and
  pointwise performance.
- **`autotune_local_cache`**: Enables local caching of autotune results. Reduces
  subsequent compile time.
- **`autotune_fallback_to_aten`**: Includes ATen (cuBLAS/cuDNN) in autotuning
  and always benchmarks native PyTorch ops. Increases compile time and provides
  a baseline.
- **`force_disable_caches`**: Disables all caches (for testing). Increases
  compile time for every compile.
