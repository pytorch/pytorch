(torch.compiler_inductor_debugging)=

# Debugging Inductor

This page covers debugging techniques specific to TorchInductor — the code
generation and optimization backend of `torch.compile`. For general
`torch.compile` debugging (graph breaks, recompilations, logging, minifier),
see [Troubleshooting](torch.compiler_troubleshooting.md).

## Compiler Bisector

The **compiler bisector** is a tool for narrowing down which part of the
compilation pipeline is responsible for a bug. It works by systematically
disabling subsystems within each `torch.compile` backend until the issue
disappears, then binary-searching within that subsystem to pinpoint the
exact transformation or lowering that triggers the failure.

**Source**: [torch/_inductor/compiler_bisector.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/compiler_bisector.py)

### What It Tests

The bisector works across multiple backends, progressively narrowing the
search:

1. **Backend selection** — tests `eager` → `aot_eager` →
   `aot_eager_decomp_partition` → `inductor` to find which backend introduces
   the issue.

2. **Subsystem isolation** — within the identified backend, it disables
   individual subsystems. For the Inductor backend, these include:

   | Subsystem | What It Controls |
   |-----------|-----------------|
   | `pre_grad_passes` | Pattern matching passes on pre-grad IR |
   | `joint_graph_passes` | Passes on the combined forward-backward graph |
   | `post_grad_passes` | Passes on individual forward/backward graphs |
   | `lowerings` | Lowering ATen operators to Inductor IR |
   | `cudagraphs` | CUDA graph wrapping of compiled graphs |
   | `fallback_random` | Config: use deterministic random number generation |
   | `emulate_precision_casts` | Config: disable floating-point fusion for precision |
   | `layout_optimization` | Config: NHWC layout optimization |
   | `comprehensive_padding` | Config: memory alignment padding |

3. **Binary search** — for bisectable subsystems (passes, lowerings), it
   binary-searches the number of applications to find the exact one that
   triggers the bug.

### Usage

#### Automatic Mode (Recommended)

The simplest way to use the bisector is with the `run` command, which
automatically determines good/bad based on process exit codes:

```bash
python -m torch._inductor.compiler_bisector run python my_failing_script.py
```

The command's exit code determines the result: `0` = good (no issue),
non-zero = bad (issue reproduced). The bisector sets `TORCH_COMPILE_BACKEND`
automatically for each test iteration.

#### Manual Mode

For cases where exit codes don't capture the failure (e.g., numerical
differences), use the interactive CLI:

```bash
# Start a bisection session
python -m torch._inductor.compiler_bisector start

# Run your test, then report result
python -m torch._inductor.compiler_bisector good   # if the run was correct
python -m torch._inductor.compiler_bisector bad    # if the run reproduced the issue

# Repeat until the bisector identifies the culprit, then clean up
python -m torch._inductor.compiler_bisector end
```

#### Programmatic Mode

You can also use `CompilerBisector.do_bisect()` directly from Python:

```python
from torch._inductor.compiler_bisector import CompilerBisector

def test_fn():
    """Returns True if the issue is reproduced, False otherwise."""
    # ... your test code ...
    return has_issue

result = CompilerBisector.do_bisect(test_fn)
print(f"Backend: {result.backend}")
print(f"Subsystem: {result.subsystem}")
print(f"Bisect number: {result.bisect_number}")
```

## Debugging Numerical Issues

TorchInductor may produce results that differ numerically from eager mode.
This is expected behavior, not a bug, because:

- **Operation reordering** — fusion changes the order in which floating-point
  operations are evaluated, and floating-point arithmetic is not associative.
- **Precision differences** — Triton may use different intermediate precision
  than PyTorch eager (e.g., TF32 for matmul on NVIDIA GPUs).
- **Fused multiply-add** — GPU kernels may use FMA instructions that have
  different rounding behavior than separate multiply and add.

### fp64 Baseline Comparison

The standard methodology for checking whether numerical differences are within
acceptable tolerance:

1. Run the model in eager mode with `float64` precision as the reference.
2. Run both eager mode and compiled mode at the original precision.
3. Compare each against the `float64` reference using relative tolerance.

If both eager and compiled results have similar error relative to the `float64`
baseline, the Inductor output is numerically acceptable.

### Emulating Precision Casts

The `emulate_precision_casts` config disables floating-point fusion in Triton
kernels, forcing explicit precision casts at each operation boundary. This
makes Inductor's numerical behavior match eager mode more closely, at the
cost of performance.

```python
torch._inductor.config.emulate_precision_casts = True
```

Or via environment variable:

```bash
TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 python my_script.py
```

This is useful for:
- Determining whether numerical differences are due to fusion-related
  precision changes.
- Producing bit-exact results when strict numerical equivalence is required.

## Debugging NaN Issues

TorchInductor can inject NaN-checking assertions into generated code to catch
NaN values at the point they are produced, rather than when they propagate to
the output.

### Input NaN Assertions

Enable NaN checks on graph inputs and kernel outputs:

```python
torch._inductor.config.nan_asserts = True
```

Or via environment variable:

```bash
TORCHINDUCTOR_NAN_ASSERTS=1 python my_script.py
```

When enabled, the generated wrapper code inserts assertions that check for NaN
values in:
- All graph input tensors before executing any kernels.
- The output of each generated kernel before proceeding to the next one.

If a NaN is detected, the assertion will fail with a message indicating which
tensor or kernel output contains the NaN.

### Runtime Triton NaN Assertions

For finer-grained NaN detection inside Triton kernels:

```python
torch._inductor.config.runtime_triton_nan_asserts = True
```

Or:

```bash
TORCHINDUCTOR_RUNTIME_TRITON_NAN_ASSERTS=1 python my_script.py
```

This codegen option inserts NaN checks directly into the generated Triton
kernel code, checking values at each store operation within the kernel. This
pinpoints exactly which operation within a fused kernel first produces a NaN.

## Inductor Fuzzer

The Inductor fuzzer (`torch._inductor.fuzzer`) is a tool for testing
TorchInductor's correctness by:

1. **Generating random configurations** — sampling from the space of Inductor
   config options (e.g., different autotuning settings, fusion strategies,
   codegen options).
2. **Running models** under each configuration and comparing outputs against
   a reference.
3. **Bisecting failures** — when a configuration produces incorrect results,
   using the compiler bisector to narrow down the root cause.

The fuzzer disables caches (`force_disable_caches = True`) to ensure each run
exercises the full compilation pipeline.

### MSE Analysis

When investigating numerical differences found by the fuzzer, the standard
approach is:

1. Compute the Mean Squared Error (MSE) between the fuzzer output and the
   reference output.
2. Compare against the MSE between eager mode and the reference at the same
   precision.
3. Flag results where the Inductor MSE is significantly larger than the eager
   MSE.

## Configuration Reference

| Config | Env Variable | Default | Description |
|--------|-------------|---------|-------------|
| `nan_asserts` | `TORCHINDUCTOR_NAN_ASSERTS` | `False` | Insert NaN checks on graph inputs and kernel outputs. |
| `runtime_triton_nan_asserts` | `TORCHINDUCTOR_RUNTIME_TRITON_NAN_ASSERTS` | `False` | Insert NaN checks inside generated Triton kernels. |
| `emulate_precision_casts` | `TORCHINDUCTOR_EMULATE_PRECISION_CASTS` | `False` | Disable FP fusion to match eager-mode precision. |
| `size_asserts` | `TORCHINDUCTOR_SIZE_ASSERTS` | `True` | Insert tensor size validation assertions. |
| `force_disable_caches` | — | `False` | Disable all compilation caches (useful for debugging). |

:::{seealso}
For general `torch.compile` debugging techniques including graph break
analysis, recompilation debugging, TORCH_LOGS options, and the minifier, see
[Troubleshooting torch.compile](torch.compiler_troubleshooting.md).
:::
