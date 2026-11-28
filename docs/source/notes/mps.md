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
        print("MPS not available because the current macOS version is not 15.0+ "
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

## Double-precision types

The MPS backend does not support `torch.float64` (`torch.double`) or
`torch.complex128` (`torch.cdouble`) because Metal Shading Language has no
`double` type:

```python
>>> torch.ones(3, dtype=torch.float64, device="mps")
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework
doesn't support float64. Please use float32 instead.
```

This is a fundamental limitation of Metal Shading Language rather than an
unimplemented PyTorch operator. Consequently, `PYTORCH_ENABLE_MPS_FALLBACK=1`
does not apply: tensors with double-precision types cannot be allocated or
copied to MPS in the first place. `Tensor.to("mps")` on a float64 tensor raises
for the same reason, rather than downcasting implicitly.

A computation that genuinely requires double precision has to run on the CPU.

## torch.compile

Two Dynamo settings help with patterns that are common in models run on MPS.
Neither is specific to MPS.

* `capture_scalar_outputs`: by default Dynamo graph-breaks at every
  `Tensor.item()` call, on any device. With this setting the scalar is traced
  into the graph instead. `torch.compile(..., fullgraph=True)` already turns it
  on, and the environment variable `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` sets
  it for the whole process.
* `allow_unspec_int_on_nn_module`: by default Dynamo specializes on integer
  attributes of an `nn.Module`, so a step counter incremented in `forward`
  recompiles the model on every call until the recompile limit is reached.
  With this setting it is treated like any other integer input, so it becomes
  dynamic after one recompile. See
  {ref}`dynamic_shapes_advanced_control_options` for related settings.

Capturing `.item()` removes the graph break, not the GPU->CPU sync: when the
value is needed on the CPU, for example when the compiled function returns it
as a Python number, reading it waits for the GPU to finish the queued work, as
`Tensor.item()` does in eager mode. With the default backend, a scalar that
only feeds tensor operations can stay on the GPU, as in the example below.

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.step = 0

    def forward(self, x):
        self.step += 1
        return x * x.sum().item() + self.step

model = Model()
x = torch.randn(4, device="mps")

with torch._dynamo.config.patch(
    capture_scalar_outputs=True,
    allow_unspec_int_on_nn_module=True,
):
    compiled_model = torch.compile(model)
    for _ in range(10):
        out = compiled_model(x)
```

Run it with `TORCH_LOGS=graph_breaks,recompiles` to see the effect. Without
the two settings, Dynamo breaks the graph at `.item()` and recompiles on every
call. With both, there is no graph break and a single recompile after the first
call, when the counter becomes dynamic.

Both settings can also be set for the whole process, and are available under
the same names in `torch.compiler.config`:

```python
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True
```
