# Demo commands

A demo of minifier for AOTI. Code mostly copied from `_dynamo/repro/after_aot.py`.


Run the code snippet below:

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc2(x)
        x = self.sigmoid(x)
        return x


torch._inductor.config.dump_aoti_minifier = True

package_path = os.path.join(os.getcwd(), "my_package.pt2")
so_path = os.path.join(os.getcwd(), "model.so")

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device=device)
    example_inputs = (torch.randn(8, 10, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    ep = torch.export.export(
        model, example_inputs, dynamic_shapes={"x": {0: batch_dim}}
    )
    torch._inductor.aot_compile(
        ep.module(), example_inputs, options={"aot_inductor.output_path": so_path}
    )
```

We purposefully error out in aot_compile in this draft PR, so you would see result like this:

```bash
/data/users/shangdiy/torch_compile_debug/run_2024_10_23_17_55_27_416503-pid_1767636/minifier/checkpoints
W1023 17:55:30.015000 1767636 pytorch/torch/_dynamo/debug_utils.py:279] Writing minified repro to:
W1023 17:55:30.015000 1767636 pytorch/torch/_dynamo/debug_utils.py:279] /data/users/shangdiy/torch_compile_debug/run_2024_10_23_17_55_27_416503-pid_1767636/minifier/minifier_launcher.py
```


The minifier_launcher.py looks like this:

```python

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims
import torch.fx._pytree as fx_pytree

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.dump_aoti_minifier = True




isolate_fails_code_str = None



# torch version: 2.6.0a0+gitcd9c6e9
# torch cuda version: 12.0
# torch git version: cd9c6e9408dd79175712223895eed36dbdc84f84


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Fri_Jan__6_16:45:21_PST_2023
# Cuda compilation tools, release 12.0, V12.0.140
# Build cuda_12.0.r12.0/compiler.32267302_0

# GPU Hardware Info:
# NVIDIA PG509-210 : 8


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, x):
        relu = torch.ops.aten.relu.default(x);  x = None
        sigmoid = torch.ops.aten.sigmoid.default(relu);  relu = None
        return (sigmoid,)

def load_args(reader):
    buf0 = reader.storage('db41273a3f311fd0658a07f88e73563410d83758', 320, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 10), is_leaf=True)  # x
load_args._version = 0
mod = Repro()
options={'aot_inductor.output_path': '/data/users/shangdiy/model.so', 'aot_inductor.serialized_in_spec': '[1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.dict", "context": "[]", "children_spec": []}]}]', 'aot_inductor.serialized_out_spec': '[1, {"type": null, "context": null, "children_spec": []}]'}
if __name__ == '__main__':
    from torch._dynamo.repro.aoti import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, config_patches=options, accuracy=False, command='minify', save_dir='/data/users/shangdiy/torch_compile_debug/run_2024_10_23_18_07_31_452542-pid_1963264/minifier/checkpoints', check_str=None)
        # To run it separately, do
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir='/data/users/shangdiy/torch_compile_debug/run_2024_10_23_18_07_31_452542-pid_1963264/minifier/checkpoints', check_str=None)
        # mod(*args)
```


Now you can run this minifier_launcher.py file to get the minified result.

[TODO]: Still need to set up this part of the demo with a better example. Currently there's no error in the graph, so it dumps the same graph.


The output looks like this:

```bash
...
Made 4 queries
W1023 18:01:43.485000 1873218 torch/_dynamo/repro/after_aot.py:331] Writing checkpoint with 2 nodes to /data/users/shangdiy/torch_compile_debug/run_2024_10_23_18_01_39_963670-pid_1873218/minifier/checkpoints/2.py
W1023 18:01:43.487000 1873218 torch/_dynamo/repro/after_aot.py:342] Copying repro file for convenience to /data/users/shangdiy/repro.py
Wrote minimal repro out to repro.py
```








[TODO]: If we have ` self.fc1 = torch.nn.Linear(10, 16)`, it doesn't work yet. The same error exists for Inductor Minifier.

Sample error msg:
```bash
  File "/home/shangdiy/.conda/envs/pytorch-3.10/lib/python3.10/inspect.py", line 1769, in getattr_static
    raise AttributeError(attr)
torch._dynamo.exc.InternalTorchDynamoError: AttributeError: weight

from user code:
   File "/data/users/shangdiy/torch_compile_debug/run_2024_10_23_17_20_10_091390-pid_1170971/minifier/minifier_launcher.py", line 47, in forward
    fc1_weight = self.fc1.weight
```
