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

## Floating-point dtypes

The `mps` device supports `torch.float32`, `torch.float16`, `torch.bfloat16`
and `torch.complex64`. What it does not have is double precision: Metal
Shading Language has no `double` type, so `torch.float64` and
`torch.complex128` are rejected:

```python
>>> torch.ones(3, dtype=torch.float64, device="mps")
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework
doesn't support float64. Please use float32 instead.
```

This is a dtype restriction, not a missing operator, so
`PYTORCH_ENABLE_MPS_FALLBACK=1` does not apply to it: the error is raised when
the tensor is allocated or copied to the device, before any operator is
dispatched.

Rather than handling the error at each tensor construction site, resolve the
dtype once, next to the device:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# MPS has no float64, so use the wider dtype only where it is available.
float_dtype = torch.float32 if device.type == "mps" else torch.float64

x = torch.ones(3, dtype=float_dtype, device=device)
```

The same holds for tensors created on the CPU: cast to a supported dtype
before moving them, since `Tensor.to("mps")` on a float64 tensor raises rather
than downcasting implicitly.

```python
x = torch.ones(3, dtype=torch.float64)
x = x.to(device=device, dtype=float_dtype)
```

If double precision is a hard requirement for part of a model, that part has
to stay on the CPU.
