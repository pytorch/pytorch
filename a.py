import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._dispatch.python import enable_python_dispatcher, patch_py_impls

with enable_python_dispatcher(), patch_py_impls({
    torch.ops.aten.matmul.default: {torch._C.DispatchKey.AutogradCPU: torch._C.DispatchKey.Autograd}
}):
    print(make_fx(torch.matmul)(torch.randn(2, 3), torch.randn(3, 4)))
