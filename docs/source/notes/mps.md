(mps-backend)=

# MPS backend

{mod}`mps` device enables high-performance
training on GPU for macOS devices with Metal programming framework. It
introduces a new device to map Machine Learning computational graphs and
primitives on highly efficient Metal Performance Shaders Graph framework and
tuned kernels provided by Metal Performance Shaders framework respectively.

The new MPS backend extends the PyTorch ecosystem and provides existing scripts
capabilities to setup and run operations on GPU.

To get started, simply move your Tensor and Module to the `mps` device:

```python
# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current macOS version is not 14.0+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happens on the GPU
    y = x * 2

    # Move your model to mps just like any other device
    model = YourFavoriteNet()
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(x)
```

## Kernel autotuning

Set `torch.backends.mps.benchmark = True` to benchmark supported MPS kernel
tile configurations and cache the fastest configuration for each input shape
and layout. The first executions of a new shape can therefore be slower. The
setting is disabled by default.

Currently, autotuning applies to eligible GEMV paths used by `torch.mm`,
`torch.addmm`, `torch.mv`, `torch.addmv`, and equivalent non-batched
`torch.matmul` calls. It supports `float32`, `float16`, and `bfloat16` when the
matrix result has `M = 1` or `N = 1`; other MPS operations are not yet
autotuned.

Use {func}`torch.backends.mps.autotune_trace` to inspect the configurations and
exact kernels used by a workload:

```python
with torch.backends.mps.autotune_trace() as trace:
    output = model(input)

for record in trace.records:
    print(record["phase"], record["config"], record["kernel"])
```

Trace records contain tensor metadata, but not tensor contents or memory
addresses.
