# AOTInductor Minifier

If you encounter an error while using AOT Inductor APIs such as
`torch._inductor.aoti_compile_and_package`, `torch._indcutor.aoti_load_package`,
or running the loaded model of `aoti_load_package` on some inputs, you can use the AOTInductor Minifier
to create a minimal nn.Module that reproduce the error by setting `from torch._inductor import config; config.aot_inductor.dump_aoti_minifier = True`.

One a high-level, there are two steps in using the minifier:

- Set `from torch._inductor import config; config.aot_inductor.dump_aoti_minifier = True` or set the environment variable `DUMP_AOTI_MINIFIER=1`. Then running the script that errors would produce a `minifier_launcher.py` script. The output directory is configurable by setting `torch._dynamo.config.debug_dir_root` to a valid directory name.

- Run the `minifier_launcher.py` script. If the minifier runs successfully, it generates runnable python code in `repro.py` which reproduces the exact error.

## Example Code

Here is sample code which will generate an error because we injected an error on relu with
`torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = "compile_error"`.


```
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

The code above generates the following error:

```text
RuntimeError: Failed to import /tmp/torchinductor_shangdiy/fr/cfrlf4smkwe4lub4i4cahkrb3qiczhf7hliqqwpewbw3aplj5g3s.py
SyntaxError: invalid syntax (cfrlf4smkwe4lub4i4cahkrb3qiczhf7hliqqwpewbw3aplj5g3s.py, line 29)
```


This is because we injected an error on relu, and so the generated triton kernel looks like below. Note that we have `compile error!`
instead if `relu`, so we get a `SyntaxError`.

```
@triton.jit
def triton_poi_fused_addmm_relu_sigmoid_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = compile error!
    tmp4 = tl.sigmoid(tmp3)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
```


Since we have `torch._inductor.config.aot_inductor.dump_aoti_minifier=True`, we also see an additional line indicating where `minifier_launcher.py` has
been written to. The output directory is configurable by setting
`torch._dynamo.config.debug_dir_root` to a valid directory name.

```text
W1031 16:21:08.612000 2861654 pytorch/torch/_dynamo/debug_utils.py:279] Writing minified repro to:
W1031 16:21:08.612000 2861654 pytorch/torch/_dynamo/debug_utils.py:279] /data/users/shangdiy/pytorch/torch_compile_debug/run_2024_10_31_16_21_08_602433-pid_2861654/minifier/minifier_launcher.py
```


## Minifier Launcher

The `minifier_launcher.py` file has the following code. The `exported_program` contains the inputs to `torch._inductor.aoti_compile_and_package`.
The `command='minify'` parameter means the script will run the minifier to create a minimal graph module that reproduce the error. Alternatively, you set
use `command='run'` to just compile, load, and run the loaded model (without running the minifier).


```
import torch
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = 'compile_error'
torch._inductor.config.aot_inductor.dump_aoti_minifier = True




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

exported_program = torch.export.load('/data/users/shangdiy/pytorch/torch_compile_debug/run_2024_11_06_13_52_35_711642-pid_3567062/minifier/checkpoints/exported_program.pt2')
# print(exported_program.graph)
config_patches={}
if __name__ == '__main__':
    from torch._dynamo.repro.aoti import run_repro
    with torch.no_grad():
        run_repro(exported_program, config_patches=config_patches, accuracy=False, command='minify', save_dir='/data/users/shangdiy/pytorch/torch_compile_debug/run_2024_11_06_13_52_35_711642-pid_3567062/minifier/checkpoints', check_str=None)
```


Suppose we kept the `command='minify'` option, and run the script, we would get the following output:

```text
...
W1031 16:48:08.938000 3598491 torch/_dynamo/repro/aoti.py:89] Writing checkpoint with 3 nodes to /data/users/shangdiy/pytorch/torch_compile_debug/run_2024_10_31_16_48_02_720863-pid_3598491/minifier/checkpoints/3.py
W1031 16:48:08.975000 3598491 torch/_dynamo/repro/aoti.py:101] Copying repro file for convenience to /data/users/shangdiy/pytorch/repro.py
Wrote minimal repro out to repro.py
```


If you get an `AOTIMinifierError` when running `minifier_launcher.py`, please report a bug [here](https://github.com/pytorch/pytorch/issues/new?assignees=&labels=&projects=&template=bug-report.yml).

## Minified Result

The `repro.py` looks like this. Notice that the exported program is printed at the top of the file, and it contains only the relu node. The minifier successfully reduced the graph to the op that raises the error.


```
# from torch.nn import *
# class Repro(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()



#     def forward(self, linear):
#         relu = torch.ops.aten.relu.default(linear);  linear = None
#         return (relu,)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.generate_intermediate_hooks = True
torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = 'compile_error'
torch._inductor.config.aot_inductor.dump_aoti_minifier = True




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


exported_program = torch.export.load('/data/users/shangdiy/pytorch/torch_compile_debug/run_2024_11_25_13_59_33_102283-pid_3658904/minifier/checkpoints/exported_program.pt2')
# print(exported_program.graph)
config_patches={'aot_inductor.package': True}
if __name__ == '__main__':
    from torch._dynamo.repro.aoti import run_repro
    with torch.no_grad():
        run_repro(exported_program, config_patches=config_patches, accuracy=False, command='run', save_dir='/data/users/shangdiy/pytorch/torch_compile_debug/run_2024_11_25_13_59_33_102283-pid_3658904/minifier/checkpoints', check_str=None)
```