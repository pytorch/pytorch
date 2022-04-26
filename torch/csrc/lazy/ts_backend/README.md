# Lazy Tensor TorchScript Backend

The TorchScript backend runs programs traced via lazy tensor tracing using the torchscript compiler and runtime.  It also serves as a reference backend.

## Using the TorchScript Backend

There are multiple downstream backends and executors hooked up to torchscript, and not all of them have been tested with lazy tensor.  The backend most used during development is NVFuser + Cuda GPU.  In principle, any torchscript backend should work with lazy tensors, but in practice there may be some device-specific implementation that needs to be updated.

1. Initialize the lazy tensor TS backend:

```
import  torch._lazy.ts_backend
 torch._lazy.ts_backend.init()
```

2. Configure the torchscript executor:
e.g. for nvfuser (fuser2) use
```
from  torch.jit  import  fuser
with fuser('fuser2'):
   ...
```
3. Move your tensors/model to the *lazy* device
`x.to(device='lazy')`

4. Run your program with LTC_TS_CUDA=1 env to enable the GPU device

See <todo@whc link to example.py> for an end to end example script using the torchscript backend.

See [lazy_bench.py in pytorch/benchmark](https://github.com/pytorch/benchmark/blob/9d8e569c857eaf82c5c85f4c9ff9a9de679b529c/lazy_bench.py) for a script that runs multiple torchbench models through the lazy tensor TS backend.