# Demo commands

A demo of minifier for AOTI. Code mostly copied from `_dynamo/repro/after_aot.py`.


Run the code snippet below, which inject an error on relu.

```python
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

# inject error on relu
torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = "compile_error"
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


The minifier_launcher.py looks like this. It's a bit different with the Inductor minifier. Here we dump the module into an ExportedProgram and load it. In Inducotr minifier, it dumps the graph module directly into the minifier_launcher.py file as a string. See the note at the end for why we use ExportedProgram instead.

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

def load_args(reader):
    buf0 = reader.storage('a480bc76d85b84685abb6c9741633686b8227ee3', 320, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 10), is_leaf=True)  # x
load_args._version = 0
mod = torch.export.load('/data/users/shangdiy/torch_compile_debug/run_2024_10_24_11_04_40_110265-pid_2899125/minifier/checkpoints/exported_program.pt2').module()
options={'aot_inductor.output_path': '/data/users/shangdiy/torch_compile_debug/model.so', 'aot_inductor.serialized_in_spec': '[1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.dict", "context": "[]", "children_spec": []}]}]', 'aot_inductor.serialized_out_spec': '[1, {"type": null, "context": null, "children_spec": []}]'}
if __name__ == '__main__':
    from torch._dynamo.repro.aoti import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, config_patches=options, accuracy=False, command='minify', save_dir='/data/users/shangdiy/torch_compile_debug/run_2024_10_24_11_04_40_110265-pid_2899125/minifier/checkpoints', check_str=None)
        # To run it separately, do
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir='/data/users/shangdiy/torch_compile_debug/run_2024_10_24_11_04_40_110265-pid_2899125/minifier/checkpoints', check_str=None)
        # mod(*args)
```


Now you can run this minifier_launcher.py file to get the minified result.


The output looks like this:

```bash
...
Made 4 queries
W1023 18:01:43.485000 1873218 torch/_dynamo/repro/after_aot.py:331] Writing checkpoint with 2 nodes to /data/users/shangdiy/torch_compile_debug/run_2024_10_23_18_01_39_963670-pid_1873218/minifier/checkpoints/2.py
W1023 18:01:43.487000 1873218 torch/_dynamo/repro/after_aot.py:342] Copying repro file for convenience to /data/users/shangdiy/repro.py
Wrote minimal repro out to repro.py
```

Here's `repro.py`. If you print out `mod`, you can see that it successfully identified the `relu` node that we purposefully error on.

```python
from math import inf

import torch

import torch._dynamo.config
import torch._functorch.config
import torch._inductor.config
import torch._inductor.inductor_prims
import torch.fx as fx
import torch.fx._pytree as fx_pytree
import torch.fx.experimental._config
from torch import device, tensor
from torch._dynamo.testing import rand_strided

torch._inductor.config.dump_aoti_minifier = True
torch._inductor.config.generate_intermediate_hooks = True


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


def load_args(reader):
    buf0 = reader.storage(
        "5278226fd34c7b5768476768232f86c125e8d484",
        512,
        device=device(type="cuda", index=0),
    )
    reader.tensor(buf0, (8, 16), is_leaf=True)  # linear


load_args._version = 0
mod = torch.export.load(
    "/data/users/shangdiy/torch_compile_debug/run_2024_10_24_11_08_39_767620-pid_2957565/minifier/checkpoints/exported_program.pt2"
).module()
options = None
if __name__ == "__main__":
    from torch._dynamo.repro.aoti import run_repro

    with torch.no_grad():
        run_repro(
            mod,
            load_args,
            config_patches=options,
            accuracy=False,
            command="run",
            save_dir="/data/users/shangdiy/torch_compile_debug/run_2024_10_24_11_08_39_767620-pid_2957565/minifier/checkpoints",
            check_str=None,
        )
        # To run it separately, do
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir='/data/users/shangdiy/torch_compile_debug/run_2024_10_24_11_08_39_767620-pid_2957565/minifier/checkpoints', check_str=None)
        # mod(*args)

```

print(exported_program.graph)
```python
graph():
    %linear : [num_users=1] = placeholder[target=linear]
    %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%linear,), kwargs = {})
    return (relu,)
```


[TODO]: write some test cases


### Note: why we use ExportedProgram to save and load graphs?
If we have ` self.fc1 = torch.nn.Linear(10, 16)`, it doesn't work in Inductor Minifier. The reason is we need to recursively generate all submodules to dump the graph module.

If we run minifier_launcher.py, we'll see error msg:
```bash
  File "/home/shangdiy/.conda/envs/pytorch-3.10/lib/python3.10/inspect.py", line 1769, in getattr_static
    raise AttributeError(attr)
torch._dynamo.exc.InternalTorchDynamoError: AttributeError: weight

from user code:
   File "/data/users/shangdiy/torch_compile_debug/run_2024_10_23_17_20_10_091390-pid_1170971/minifier/minifier_launcher.py", line 47, in forward
    fc1_weight = self.fc1.weight
```

minifier_launcher:
```python

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Module().cuda() ## self.fc1 needs to be converted recursively



    def forward(self, x):
        fc1_weight = self.fc1.weight
        fc1_bias = self.fc1.bias
        linear = torch.ops.aten.linear.default(x, fc1_weight, fc1_bias);  x = fc1_weight = fc1_bias = None
        relu = torch.ops.aten.relu.default(linear);  linear = None
        sigmoid = torch.ops.aten.sigmoid.default(relu);  relu = None
        return (sigmoid,)
```
