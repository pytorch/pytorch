# AOTInductor Minifier

If you encounter an error while using AOT Inductor APIs such as `torch._inductor.aoti_compile_and_package`, `torch._inductor.aoti_load_package`, or when running the loaded model from `aoti_load_package` on some inputs, you can use the AOTInductor Minifier to create a minimal `nn.Module` that reproduces the error. Enable this by setting:

```python
from torch._inductor import config
config.aot_inductor.dump_aoti_minifier = True
```

## High-Level Usage

There are two main steps to use the minifier:

1. **Enable Minifier Dumping**
    Set `from torch._inductor import config; config.aot_inductor.dump_aoti_minifier = True` or set the environment variable `DUMP_AOTI_MINIFIER=1`. Running the script that triggers the error will produce a `minifier_launcher.py` script. The output directory can be configured by setting `torch._dynamo.config.debug_dir_root` to a valid directory.

2. **Run the Minifier Script**
    Execute the generated `minifier_launcher.py` script. If the minifier runs successfully, it generates runnable Python code in `repro.py` that reproduces the exact error.

## Example Code

Below is sample code that generates an error by injecting a bug into ReLU with `torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = "compile_error"`:

```python
import torch
from torch._inductor import config as inductor_config

class Model(torch.nn.Module):
     def __init__(self):
          super().__init__()
          self.fc1 = torch.nn.Linear(10, 16)
          self.relu = torch.nn.ReLU()
          self.sigmoid = torch.nn.Sigmoid()

     def forward(self, x):
          x = self.fc1(x)
          x = self.relu(x)
          x = self.sigmoid(x)
          return x

inductor_config.aot_inductor.dump_aoti_minifier = True
torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = "compile_error"

with torch.no_grad():
     model = Model().to("cuda")
     example_inputs = (torch.randn(8, 10).to("cuda"),)
     ep = torch.export.export(model, example_inputs)
     package_path = torch._inductor.aoti_compile_and_package(ep)
     compiled_model = torch._inductor.aoti_load_package(package_path)
     result = compiled_model(*example_inputs)
```

The code above produces an error like:

```
RuntimeError: Failed to import /tmp/torchinductor_shangdiy/fr/cfrlf4smkwe4lub4i4cahkrb3qiczhf7hliqqwpewbw3aplj5g3s.py
SyntaxError: invalid syntax (cfrlf4smkwe4lub4i4cahkrb3qiczhf7hliqqwpewbw3aplj5g3s.py, line 29)
```

This is because we injected an error into ReLU, so the generated Triton kernel contains `compile error!` instead of `relu`, resulting in a `SyntaxError`:

```python
@triton.jit
def triton_poi_fused_addmm_relu_sigmoid_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 128
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
     xmask = xindex < xnumel
     x2 = xindex
     x0 = xindex % 16
     tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
     tmp1 =
