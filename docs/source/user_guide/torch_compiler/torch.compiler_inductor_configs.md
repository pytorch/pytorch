(torch.compiler_inductor_configs)=

# Configs

## How to Set Configs

### config.py

All of TorchInductor's configs are in
[torch/inductor/config.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py).
The simplest way to set a config while debugging is just to alter the value in
that file. Alternatively, most configs also have a corresponding environment
variable that can be used to initialize from the environment.

### config.patch

This works as a context manager. For example:

```python
from torch._inductor import config
import torch

def fn(a, b):
    return a * b

with config.patch({"max_autotune": True}):
    a = torch.tensor(1)
    b = torch.tensor(1)
    torch.compile(fn)(a, b)
```

When using patch, make sure that the compiled function call is inside of the
patch. Just wrapping the compile call won't work because compilation is delayed
until the first call:

```python
with config.patch({"max_autotune": True}):
    foo = torch.compile(fn)

# max_autotune will be False here:
foo(a, b)
```

### Options

Passing options to a `torch.compile` call will also work for setting configs:

```python
compiled = torch.compile(fn, backend="inductor", options={"max_autotune": True})
```

## Notable Configs

Below is a non-exhaustive list of configs that you may find useful when working
with Inductor.

### Debugging Configs

- **force_disable_caches**: Without this set, TorchInductor will attempt to
  cache compiled artifacts in order to save compile time on subsequent
  compilations. For TorchInductor developers, caching can ruin bisections or
  other debugging techniques, so make sure to turn on. Also note that this
  option disables *all* `torch.compile`-related caches — not just the
  TorchInductor cache.
- **size_asserts, nan_asserts, size_asserts, alignment_asserts**: Puts
  correctness assertions in the generated code.
- **max_fusion_size**: The maximum number of Scheduler Nodes allowed in a
  single fused kernel. Setting this to 1 can help debug issues with fusion.
- **compile_threads**: The number of threads used to compile. Setting this to 1
  can sometimes help with debugging. Note that the compiler leverages
  subprocesses for Triton compilation, so this option controls the maximum
  number of worker subprocesses.
- **debug**: Some debugging printouts. Generally skip this one in favor of
  `TORCH_LOGS="+inductor"` or similar.
- **emulate_precision_casts**: Emulate numerics from eager mode with fp16,
  bf16.

### Performance Configs

- **max_autotune**: Enables an autotuning process that speeds up generated
  kernels.
- **max_autotune_gemm_backends**: Specify additional backends for max-autotune.
- **coordinate_descent_tuning**: Another setting to increase the speed of
  generated kernels. This config composes with max_autotune.
- **shape_padding**: Pad input tensors of matmul/bmm/addmm.
- **triton.cudagraphs**: Enable cudagraphs.

### Feature Configs

- **cpp_wrapper**: This changes the wrapper code from python to cpp.

## Additional `torch.compile` Options

These are options on the `torch.compile` call. For example, they can be set
like:

```python
torch.compile(fn, fullgraph=True, dynamic=True, backend="inductor")
```

Documentation for these options can be found in the `torch.compile`
[Doc String](https://github.com/pytorch/pytorch/blob/2e4e5ab4be9e0aeffd9c49b5b2f9f820bd0895b1/torch/__init__.py#L2481).
