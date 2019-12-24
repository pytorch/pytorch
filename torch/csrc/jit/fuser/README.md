# Fusion in PyTorch (In Progress)

## Simple Python script for e2e validation

```
import torch

def add(x, y):
    return x + y

scripted_add = torch.jit.script(add)

x = torch.zeros(5, 2)
y = torch.ones(5, 2)

# Second iteration runs the optimizations
scripted_add(x, y)
z = scripted_add(x, y)

print(z)
```

## Testing

CPP tests are available at test/cpp/jit/test_fuser.cpp. Tests can be run
by building (you may need to rebuild with the --cmake option the first time)
and runniong /build/bin/test_jit. This will run all jit cpp tests. To run
the fusion tests, you can filter like this:

```
./test_jit --gtest_filter='*Fusion*'
```

Our two current CPP tests extensibility points are CPUFusion and GPUFusion.
See the readme in test/cpp/jit for more information on how these tests
are generated.

Python tests are not available yet. test_jit_fuser.py will be repurposed shortly.

## Adding and removing files

All cpp (.cpp) files must be added to or removed from Caffe2/CmakeLists.txt and
tools/build_variables.py.

Header files (.h) are processed automatically.

Do not use non-standard suffixes like .cc or .hpp.


## Fusion Pass

Currently contains considerable legacy code and only supports singleton
fusion. Vertical fusion coming soon.


## TODO

- Prototype TorchScript -> LoopIR
- Add vertical fusion
- Restore removed debug functionality (see torch/csrc/jit/init.cpp)
- Repair custom fusion pass (see test/cpp/jit/test_misc.cpp)
- Replace test_fuser.cpp
- Change "fuser" name(space) to "fusion"
- Update Python tests
- Create cpp tests
- Benchmark speedups on CPU and GPU with benchmark networks / snippets
- Remove remaining unused "fuser" files
- Update fusion README to reflect new requirements, capabilities, and design
- CPU: LoopIR -> CPUIR
- CPU: CPUIR -> callable
- Benchmark kernel caching
- move branch to pytorch/pytorch (once tests are ready)
