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

## torch.compile on MPS

Two Dynamo config flags come up often enough with MPS models that they are
worth calling out here, even though neither is MPS-specific:

* `capture_scalar_outputs`: without it, any `.item()` call on a GPU tensor
  graph-breaks (`Reason: Unsupported: Tensor.item`), since not all backends
  support tracing a scalar readout into the FX graph. Note this only fixes
  that specific break -- if the scalar is then used in a data-dependent
  `if`/`while`, tracing still breaks separately (`Reason: Data-dependent
  jump`); see {ref}`the graph-break troubleshooting guide<torch.compiler_troubleshooting>`
  for that case.
* `allow_unspec_int_on_nn_module`: without it, an integer `nn.Module`
  attribute that changes every call (a step counter incremented in
  `forward`, for example) causes Dynamo to specialize on its exact value,
  recompiling on every change. See
  {ref}`dynamic_shapes_advanced_control_options` for the full set of related
  static/dynamic control flags.

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.step = 0

    def forward(self, x):
        self.step += 1
        return x.sum().item() + self.step

model = Model()
x = torch.randn(4, device="mps")

with torch._dynamo.config.patch(
    capture_scalar_outputs=True,
    allow_unspec_int_on_nn_module=True,
):
    compiled_model = torch.compile(model, backend="eager")
    out = compiled_model(x)
```

To apply these for the whole process instead of a single call, set the
attributes directly instead of using the context manager:

```python
import torch._dynamo as dynamo

dynamo.config.capture_scalar_outputs = True
dynamo.config.allow_unspec_int_on_nn_module = True
```

Both flags are also available, under the same names, through the newer
backend-agnostic `torch.compiler.config` module
(`torch.compiler.config.patch(...)`), which aliases the same underlying
Dynamo settings. Prefer it for new code; `torch._dynamo.config.patch` remains
supported and behaves identically for these two flags.
